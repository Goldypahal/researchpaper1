# Pre-Registered Research Questions & Hypotheses

**Program**: ResearchPaper1 — Empirical Investigation of Semantic Reasoning in Autonomous Machine Learning Optimization  
**Registration Date**: 2026-09-24  
**Primary Author/Lead**: Pair Research Program  
**Git Commit at Registration**: `9c475fa`  

---

## 1. Central Research Question

> **Under what conditions does semantic LLM reasoning actually provide marginal value over established search algorithms (Random Search, TPE Bayesian Optimization, Genetic Algorithms) for autonomous machine learning optimization under equal compute budgets, and why?**

---

## 2. Formal Research Questions (RQs)

- **RQ1 (Search Optimality)**: Does LLM-guided architecture search discover lower holdout validation loss than classical stochastic and sequential optimizers (Random, TPE, GA) under strictly equal compute budgets?
- **RQ2 (Search Efficiency & Early Discovery)**: Does semantic reasoning improve search efficiency by finding high-performing architectures earlier in the compute horizon, as measured by Area Under the Search Curve ($\text{AUC} = \int_0^B P(b)\,db$)?
- **RQ3 (Task Complexity Scaling)**: Does the marginal advantage of LLM reasoning expand as task complexity scales from regular state-space tracking (Hidden FSM) to context-free hierarchical grammar (Dyck-$k$ with deeper stack nesting)?
- **RQ4 (Out-of-Distribution Generalization)**: Do LLM-generated architectures exhibit smaller Generalization Gaps ($\text{Gap}_{\text{rel}} = (L_{\text{OOD}} - L_{\text{val}}) / L_{\text{val}}$) compared to purely metric-driven baselines?
- **RQ5 (Hypothesis Calibration & Metacognition)**: Can an LLM search agent accurately predict the quantitative performance delta ($\widehat{\Delta L}$) of its own proposed modifications, or does it suffer from severe over-confidence?
- **RQ6 (Ablation of Cognitive Primitives)**: Which specific reasoning component (Historical Trajectory Memory, Qualitative Verbal Reflection, or Adversarial Critic Verification) provides the largest causal impact on search success?

---

## 3. Pre-Registered Hypotheses (H1–H7)

The experimental evaluation will test and explicitly accept or reject the following pre-registered hypotheses:

### Hypothesis 1 (Search Superiority over Random Search)
$$\mathcal{H}_1: \quad L^*_{\text{val}}(\text{LLM}) < L^*_{\text{val}}(\text{Random}) \quad \text{at matched compute budget } B.$$
- *Null ($\mathcal{H}_{1,0}$)*: LLM semantic search yields no statistically significant reduction in validation loss over uniform random sampling ($p \ge 0.05$).
- *Decision Rule*: Reject null if Welch's $t$-test and Mann-Whitney $U$ show $p < 0.05$ with positive effect size (Cohen's $d > 0.5$).

### Hypothesis 2 (Value Beyond Sequential Optimizers)
$$\mathcal{H}_2: \quad L^*_{\text{val}}(\text{LLM}) < \min(L^*_{\text{val}}(\text{TPE}), L^*_{\text{val}}(\text{GA})).$$
- *Null ($\mathcal{H}_{2,0}$)*: Sequential heuristics with surrogate models (TPE) or genetic mutation equal or outperform LLM reasoning in finding optimal configurations within fixed horizons.

### Hypothesis 3 (Complexity-Dependent Reasoning Value)
$$\mathcal{H}_3: \quad \frac{\Delta_{\text{LLM-Random}}(\text{Dyck-}k)}{\text{Compute}} > \frac{\Delta_{\text{LLM-Random}}(\text{FSM})}{\text{Compute}}.$$
- *Theoretical Justification*: Context-free hierarchical bracket tracking possesses structured structural inductive requirements (e.g. Rotary embeddings, stack-depth matching) that semantic reasoning can diagnose, whereas regular Markovian state machines can be adequately fit by random width/depth tuning.

### Hypothesis 4 (Causal Role of Qualitative Verbal Reflection)
$$\mathcal{H}_4: \quad \text{AUC}(\text{Full Agent}) > \text{AUC}(\text{History Without Reflection}).$$
- *Null ($\mathcal{H}_{4,0}$)*: Qualitative natural language reflections do not improve search efficiency beyond raw numeric history logging.

### Hypothesis 5 (Critic Verification Improves Calibration)
$$\mathcal{H}_5: \quad \text{MAE}_{\text{calibration}}(\text{Proposer-Critic}) < \text{MAE}_{\text{calibration}}(\text{Single Proposer}).$$
- *Theoretical Justification*: Adversarial critic review reins in optimistic prediction bias, reducing $|\widehat{\Delta L} - \Delta L_{\text{actual}}|$.

### Hypothesis 6 (Inductive Bias and OOD Generalization)
$$\mathcal{H}_6: \quad \text{Gap}_{\text{rel}}(\text{LLM}) < \text{Gap}_{\text{rel}}(\text{Random}) \quad \text{on Dyck-}k \text{ depth extrapolation } (d \in [7, 12]).$$
- *Null ($\mathcal{H}_{6,0}$)*: LLM-selected architectures generalize no better out-of-distribution than randomly sampled valid architectures.

### Hypothesis 7 (Search Policy Adaptability)
$$\mathcal{H}_7: \quad \text{Performance}(\pi_{\text{Agent}, v1}) > \text{Performance}(\pi_{\text{Agent}, v0}).$$
- *Theoretical Justification*: An agent that inspects its historical search graph and dynamically adjusts its exploration/exploitation trade-off outperforms a static search prompt.

---

## 4. Rejection Policy
In accordance with open scientific rigor, **negative outcomes are explicitly welcomed**. If classical optimizers (TPE or Random Search) outperform the LLM under matched compute, this result directly answers our core research question:
*Semantic reasoning incurs substantial inference and inductive overhead that can hinder optimization efficiency when the design landscape is sufficiently dense for classical sequential search.*
