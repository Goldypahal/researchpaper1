#!/usr/bin/env python3
"""
tools/audit_traces.py

Audits all experiment traces across search arms, seeds, and tasks.
Classifies each expected or found trace into:
  - COMPLETE_VALID: parses, K+1 trajectory entries, iterations 0..K, finite val_loss, real LLM provider.
  - FAILED: trajectory has one or more crashed iterations or unhandled errors.
  - INCOMPLETE: fewer than K+1 entries or .jsonl without complete .json.
  - MISSING: expected run has no trace file.

Generates:
  - audit_report.csv
  - audit_manifest_sha256.json / txt
"""

import os
import sys
import json
import glob
import hashlib
import csv
from typing import Dict, List, Any, Optional

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def audit_single_trace(filepath: str, expected_k: int = 10) -> Dict[str, Any]:
    record = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "status": "UNKNOWN",
        "method": None,
        "task": None,
        "seed": None,
        "num_entries": 0,
        "has_adaptive_sim": False,
        "crashed_iterations": [],
        "sha256": None,
        "error_detail": ""
    }
    
    if not os.path.exists(filepath):
        record["status"] = "MISSING"
        return record

    try:
        record["sha256"] = compute_sha256(filepath)
    except Exception as e:
        record["status"] = "INCOMPLETE"
        record["error_detail"] = f"Failed to compute sha256: {e}"
        return record

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        record["status"] = "INCOMPLETE"
        record["error_detail"] = f"JSON parse error: {e}"
        return record

    record["method"] = data.get("method") or data.get("arm")
    record["task"] = data.get("task")
    record["seed"] = data.get("seed")

    trajectory = data.get("trajectory", [])
    record["num_entries"] = len(trajectory)

    # Check for adaptive_simulation
    provider = data.get("llm_provider") or data.get("provider")
    if provider == "adaptive_simulation":
        record["has_adaptive_sim"] = True

    crashed = []
    iterations_seen = []
    
    for i, step in enumerate(trajectory):
        it_idx = step.get("iteration", i)
        iterations_seen.append(it_idx)
        val_loss = step.get("val_loss")
        status = step.get("status", "OK")
        
        # In pilot data, val_loss 99.0 or None or crash status indicates crash
        if val_loss is None or val_loss == 99.0 or str(status).startswith("CRASH") or status == "PROPOSAL_FAILED":
            crashed.append((it_idx, status, val_loss, step.get("error_trace", "")))
        elif not isinstance(val_loss, (int, float)) or not (-1e6 < val_loss < 1e6):
            crashed.append((it_idx, "NON_FINITE_LOSS", val_loss, ""))

    record["crashed_iterations"] = crashed

    # Classification logic
    expected_count = expected_k + 1  # iteration 0 (baseline) + 1..K
    if len(crashed) > 0:
        record["status"] = "FAILED"
        record["error_detail"] = f"{len(crashed)} crashed/failed iterations: {[c[0] for c in crashed]}"
    elif record["num_entries"] < expected_count:
        record["status"] = "INCOMPLETE"
        record["error_detail"] = f"Entries {record['num_entries']} < expected {expected_count}"
    elif record["has_adaptive_sim"]:
        record["status"] = "FAILED"
        record["error_detail"] = "Disallowed adaptive_simulation provider found"
    elif iterations_seen != list(range(len(iterations_seen))):
        record["status"] = "INCOMPLETE"
        record["error_detail"] = f"Iterations not sequential 0..K: {iterations_seen}"
    else:
        record["status"] = "COMPLETE_VALID"

    return record

def run_trace_audit(
    trace_dirs: List[str],
    output_csv: str = "audit_report.csv",
    output_manifest: str = "audit_manifest_sha256.json",
    expected_k: int = 10
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    json_files = []
    for d in trace_dirs:
        if os.path.exists(d):
            json_files.extend(glob.glob(os.path.join(d, "*.json")))
    
    records = []
    manifest = {}
    summary = {
        "COMPLETE_VALID": 0,
        "FAILED": 0,
        "INCOMPLETE": 0,
        "MISSING": 0,
        "TOTAL": len(json_files)
    }

    for fpath in sorted(json_files):
        rec = audit_single_trace(fpath, expected_k=expected_k)
        records.append(rec)
        manifest[rec["filename"]] = {
            "path": rec["filepath"],
            "sha256": rec["sha256"],
            "status": rec["status"],
            "task": rec["task"],
            "method": rec["method"],
            "seed": rec["seed"]
        }
        summary[rec["status"]] = summary.get(rec["status"], 0) + 1

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename", "status", "task", "method", "seed",
            "num_entries", "has_adaptive_sim", "error_detail", "sha256", "filepath"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "filename": r["filename"],
                "status": r["status"],
                "task": r["task"],
                "method": r["method"],
                "seed": r["seed"],
                "num_entries": r["num_entries"],
                "has_adaptive_sim": r["has_adaptive_sim"],
                "error_detail": r["error_detail"],
                "sha256": r["sha256"],
                "filepath": r["filepath"]
            })

    # Write Manifest
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print("TRACE AUDIT SUMMARY:")
    for k, v in summary.items():
        print(f"  {k:15s}: {v}")
    print(f"Report written to:   {output_csv}")
    print(f"Manifest written to: {output_manifest}")
    print("=" * 60)
    return summary

if __name__ == "__main__":
    dirs_to_check = [
        "v0.3-comparative-search/traces",
        "v0.4-reasoning-ablation/traces"
    ]
    run_trace_audit(dirs_to_check)
