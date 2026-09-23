"""
Subprocess Execution Sandbox.
Runs candidate experimental scripts in isolated subprocesses with:
- Strict wall-clock timeouts
- Structured traceback parsing (syntax errors vs runtime exceptions)
- Exit code capture and metric extraction
"""

import subprocess
import sys
import os
import json
import time

def execute_in_sandbox(script_path, args=None, timeout_sec=180):
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env
        )
        elapsed = time.time() - start_time
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode

        # Attempt to parse JSON block from stdout
        metrics = None
        # Try full output as JSON first
        try:
            metrics = json.loads(stdout.strip())
        except:
            # Look for JSON block delimited by { and }
            start_idx = stdout.find("{")
            end_idx = stdout.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    metrics = json.loads(stdout[start_idx:end_idx+1])
                except:
                    pass

        status = "SUCCESS" if exit_code == 0 and metrics else "RUNTIME_CRASH"
        return {
            "status": status,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 2),
            "metrics": metrics,
            "stdout_tail": stdout[-500:],
            "stderr_tail": stderr[-500:] if stderr else ""
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "status": "TIMEOUT",
            "exit_code": -1,
            "elapsed_seconds": round(elapsed, 2),
            "metrics": None,
            "stdout_tail": "",
            "stderr_tail": f"Execution timed out after {timeout_sec} seconds."
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "SANDBOX_ERROR",
            "exit_code": -2,
            "elapsed_seconds": round(elapsed, 2),
            "metrics": None,
            "stdout_tail": "",
            "stderr_tail": str(e)
        }
