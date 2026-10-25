# Pre-Registration: Empirical Decomposition of Semantic LLM Reasoning in Autonomous Architecture Search

**Registration Date:** October 25, 2026  
**Protocol Version:** `v1.0-final` (Air-Tight Matched-Compute Framework)  
**Primary Question:**  
> *"When an LLM agent improves hyperparameter and neural architecture search over classical baselines, does the gain originate from pre-trained domain priors (textbook associations) or from active sequential learning from experimental feedback?"*

---

## 1. Experimental Design & Hypotheses

### Primary Hypotheses (Formal Pre-Registration)
* **$H_1$ (Semantic Prior Effect):** Naming the task ($D_{\text{named}}$) vs anonymizing the task ($D_{\text{anonymized}}$) significantly influences the candidate proposal distribution and final test loss ($L^*_{\text{test}}$). Specifically, on Dyck-4, naming the task concentrates proposals on Rotary Position Encodings (RoPE) due to semantic prior bias.
* **$H_2$ (Feedback Sensitivity Effect):** LLM agents provided with real loss history ($D_{\text{real}}$) outperform agents provided with randomly shuffled past performance history ($D_{\text{shuffled}}$) where losses are permuted among evaluated configurations.
  * *Null Hypothesis ($H_{2,0}$):* $L^*_{\text{real}} = L^*_{\text{shuffled}}$ (the LLM's proposals are invariant to quantitative numerical loss feedback).
* **$H_3$ (Reasoning Loop vs. Static Prior):** An active sequential reasoning loop outperforms uniform random sampling from the LLM's zero-shot proposal distribution ($D_{\text{prior\_sampler}}$).
* **$H_4$ (Inductive Bias Friction on Anti-Prior Tasks):** On the `abs_position` task (where targets are deterministic functions of absolute sequence indices and RoPE underperforms absolute/learned embeddings), the LLM agent's semantic bias towards RoPE leads to negative transfer compared to Classical Random Search and TPE.

---

## 2. Experimental Arms & Matched Conditions

All search arms explore the identical single-source-of-truth search space ([search_space.py](file:///c:/Users/goldy/Desktop/Researchpapers/Research_Paper_1_AI_Improves_AI/search_space.py)) with $K=10$ candidate evaluations per trajectory:

| Arm Name | Type | Task Prompting | Feedback Context | What it Isolates |
| :--- | :--- | :--- | :--- | :--- |
| **`random`** | Baseline | N/A | None | Uniform random search floor |
| **`tpe`** | Baseline | N/A | Bayesian KDE | Real Bayesian Opt ($n_{\text{startup}}=3$, multivariate) |
| **`evolutionary`** | Baseline | N/A | Fitness selection | Real Genetic Algorithm (tournament + mutation) |
| **`prior_sampler`** | Hybrid | Named | None | Pure zero-shot LLM prior without reasoning loop |
| **`tpe_warm`** | Hybrid | N/A | Bayesian + Warm | TPE warm-started with first 3 LLM zero-shot proposals |
| **`llm_named_real`** | LLM | Named ("Dyck-4") | Real history | Total observed LLM effect |
| **`llm_anonymized_real`**| LLM | Anonymized ("Task A")| Real history | Isolates feedback learning without semantic prior |
| **`llm_named_shuffled`** | LLM | Named ("Dyck-4") | Shuffled losses | Isolates whether LLM actually uses numerical loss feedback |
| **`llm_named_none`** | LLM | Named ("Dyck-4") | None (Zero-shot) | Pure prior without historical context |
| **`llm_obfuscated_real`**| LLM | Obfuscated names | Real history | Eliminates all parameter keyword priors (`p1, opt_A`) |

---

## 3. Statistical Analysis Protocol

1. **Primary Metric (Unbiased Selection):**
   * The headline metric is **Holdout Test Loss ($L_{\text{test}}$)** evaluated by retraining the search winner across **$R=5$ fresh, independent random seeds** on the uncorrupted test split.
   * Best-of-$K$ validation loss is reported only as a secondary search dynamic metric.
2. **Paired Experimental Units:**
   * Comparisons are matched strictly by seed: $D_i = \text{Arm}_A[i] - \text{Arm}_B[i]$.
   * Primary test: **Two-sided Wilcoxon Signed-Rank Test**.
   * Effect size: **Paired Cohen's $d_z$** and non-parametric **95% Bootstrap Confidence Intervals** (2,000 resamples).
3. **Multiplicity Control:**
   * **Holm-Bonferroni Step-Down Correction** applied across the family of hypothesis tests at family-wise $\alpha = 0.05$.
4. **Three Matched Compute Budgets:**
   * Anytime curves plotted against:
     1. Equal Iteration Horizon ($K=10$)
     2. Equal Training GPU Seconds
     3. Equal Total Wall-Clock Seconds (Training GPU + LLM autoregressive inference)

---

## 4. Failure Policy & Quality Control

* **Proposal Validation:** All proposals validated strictly against [search_space.py](file:///c:/Users/goldy/Desktop/Researchpapers/Research_Paper_1_AI_Improves_AI/search_space.py). Violations trigger a single automated re-prompt. If uncorrected, the iteration is logged with `status="INVALID_PROPOSAL"`.
* **Zero Missing Runs:** Trajectories never raise uncaught fatal exceptions. Failed evaluations receive `val_loss=None` (not 99.0) and Optuna `TrialState.FAIL`.
* **Integrity Audit:** Traces must pass [tools/audit_traces.py](file:///c:/Users/goldy/Desktop/Researchpapers/Research_Paper_1_AI_Improves_AI/tools/audit_traces.py) with `COMPLETE_VALID` to be eligible for analysis and publication figures.
