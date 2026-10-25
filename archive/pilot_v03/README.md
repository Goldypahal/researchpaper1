# Pilot Study Archive (v0.3 / v0.4 Traces)

**Status:** ARCHIVED PILOT DATA (Frozen)  
**Archive Date:** October 25, 2026  
**Artifact Archive:** `pilot_v03_traces.zip`  
**Integrity Manifest:** `audit_manifest_sha256.json`  
**Detailed Audit:** `audit_report.csv`  

---

## Methodological Notice
These historical trajectories represent initial pilot explorations conducted prior to protocol freezing:
1. **Search Space Asymmetry:** LLM search space and classical baseline spaces contained known discrepancies (e.g., $d_{\text{model}} \in \{128, 256, 384\}$ in the LLM prompt vs. $\{128, 256, 512\}$ in the classical sampler; `d_ff` omitted from LLM prompt).
2. **Optuna Startup Default:** Optuna TPE used default `n_startup_trials=10`, causing $K=10$ runs to evaluate uniform random candidates rather than a trained KDE acquisition function.
3. **Metric Alignment:** Trajectories logged best-of-$K$ validation loss rather than post-selection retrained holdout test loss under independent data/train seeds.

### Scientific Purpose in Manuscript
As planned in the research protocol:
- **No data from this pilot directory is merged into the final paper results.**
- These data are retained exclusively for:
  1. Estimating baseline training and inference timings.
  2. Initial noise variance estimation ($\sigma_{\text{retrain}}$).
  3. Appendix: *"Pilot Results Under Unequal Search Spaces and Retrospective Confound Analysis"*.
