# Research Paper 1: Scientific Differentiation, Experiment Protocol, and Formal Metrics

**Working Paper Title**: *Can AI Agents Improve AI Without Humans? Empirical Limits of Autonomous Transformer Optimization Under Fixed Compute Budgets*  
**Corpus / Context**: Topic 1 Foundational Investigation  
**Author**: Antigravity Autonomous Research Pair

---

## 1. Locked Differentiation Statement (Contribution Paragraph)

> ### Primary Contribution Statement
> Current paradigms of autonomous scientific discovery oscillate between two empirical extremes: the unconstrained, end-to-end automation of paper authorship (Lu et al., 2024; Tang et al., 2025) and the open-ended exploration of ambitious machine learning hypotheses (Kirgis et al., 2026). However, both extremes succumb inevitably to five foundational pathologies: the **Verifier Ceiling**, where noisy LLM-as-a-judge reviewers saturate and reward superficial formatting over technical substance; **Performance Plateauing**, where prompt-based self-optimizers (Zelikman et al., 2023) exhaust in-context exploration within 2–3 iterations; **Specification Gaming**, where autonomous agents bypass training loops or exploit harness loopholes to claim synthetic improvements; **Model Collapse** (Shumailov et al., 2023), where recursive self-training on model-generated outputs causes distribution collapse; and **Time-Horizon Decay** (Kinniment et al., 2024), where compounding errors cause multi-day unguided agents to abandon their core research targets within 10 hours and face unanimous peer rejection (Kirgis et al., 2026).
>
> In contrast to the ill-posed, open-ended paper-writing formulation of *The AI Scientist* and the unconstrained multi-day explorations of *Kirgis et al.*, this research isolates autonomous self-improvement to a **narrow, mathematically tractable, and strictly verifiable empirical task**: the autonomous architectural and hyperparameter optimization of a modular Transformer language model on an **unseen, non-public sequence dataset** under an **identical, fixed 10 GPU-hour compute budget**. By constraining the agent to executable PyTorch code space and pairing it with an **exact, deterministic verifier** (validation cross-entropy on holdout tokens in an isolated sandbox), we completely eradicate the Verifier Ceiling and Specification Gaming. This establishes the first rigorously controlled benchmark directly comparing an **Autonomous Agent** against **Random / Bayesian Search** and a **Human Machine Learning Practitioner** under identical resource budgets, resolving whether autonomous LLMs exhibit genuine algorithmic discovery or mere stochastic search under computational scarcity.

---

## 2. End-to-End Experiment Protocol

### 2.1 Problem Formulation & Task Specification
The core challenge is formulated as an outer-loop meta-optimization problem:
$$\max_{\theta \in \Theta} \mathcal{U}(\theta) \quad \text{subject to} \quad \text{GPU\_Hours}(\theta) \le B_{\text{total}}$$
where $\theta$ represents the code specification of a Transformer model (architectural layers, attention geometry, activation functions, normalization dynamics, and optimizer schedule), and $\mathcal{U}(\theta)$ is the deterministic utility evaluated on holdout data.

- **Base Architecture**: A clean, modular Decoder-only Transformer implemented in PyTorch:
  - Layers: $L = 6$
  - Model Dimension: $d_{\text{model}} = 256$
  - Attention Heads: $h = 4$ ($d_{\text{head}} = 64$)
  - Feed-Forward Dimension: $d_{\text{ff}} = 1024$
  - Context Window: $T_{\text{ctx}} = 256$ tokens
  - Vocabulary Size: $|V| = 1024$
- **Target Dataset**: An **unseen synthetic algebraic grammar dataset** (State-Transition Dyck Language with non-local token dependencies). Crucially, this dataset has zero representation in any commercial LLM training corpus, ensuring that the agent cannot recall memorized architectural heuristics and must infer inductive biases purely from empirical validation loss feedback.

---

### 2.2 The Three Controlled Comparison Arms

Every comparison arm is executed under the **exact same hardware environment** (1x NVIDIA GPU), accessing the **exact same baseline codebase**, and bounded by an **identical 10.0 GPU-hour budget**:

```
+-----------------------------------------------------------------------------------+
|                           FIXED 10 GPU-HOUR BUDGET                                |
+-------------------------+-------------------------------+-------------------------+
|   ARM 1: AGENTIC        |   ARM 2: STOCHASTIC           |   ARM 3: HUMAN          |
|   AUTONOMOUS SEARCH     |   RANDOM / BAYESIAN SEARCH    |   PRACTITIONER BASELINE |
+-------------------------+-------------------------------+-------------------------+
| - LLM Hypothesis Engine | - Uniform / Tree Parzen Search| - Experienced ML Eng.   |
| - AST Code Mutator      | - Exhaustive Search Space     | - Zero Internet Access  |
| - Isolated Sandbox Eval | - Identical GPU Budget        | - Same Local Docs       |
| - Quantitative Reflexion| - 0 Reasoning Scaffolding     | - Identical 10h Budget  |
+-------------------------+-------------------------------+-------------------------+
```

1. **Arm 1 (Autonomous Agent)**:
   - *Scaffold*: Minimal LLM reasoning harness (Hypothesis $\to$ PyTorch Code Patch $\to$ Sandbox Execution $\to$ Log Parsing $\to$ Verbal Reflection).
   - *Mutation Scope*: Can modify attention mechanics (e.g. rotary, linear, gated), activation functions (GELU, SwiGLU, custom polynomials), normalization (LayerNorm, RMSNorm, pre/post), and optimization hyperparameters.
   - *Human Intervention*: Exactly zero ($N_{\text{interv}} = 0$). All errors returned as structured execution tracebacks.

2. **Arm 2 (Random / Bayesian Hyperparameter Search Baseline)**:
   - *Search Space*: Formally covers the continuous and discrete space corresponding to the parameters mutable by Arm 1 (learning rate $[10^{-5}, 10^{-2}]$, weight decay $[0.0, 0.2]$, warmup steps $[0, 500]$, layer depths $\{4, 6, 8\}$, attention heads $\{2, 4, 8\}$, activation types $\{\text{ReLU}, \text{GELU}, \text{Swish}, \text{RMSNorm vs LayerNorm}\}$).
   - *Execution*: Evaluates configurations sequentially until the 10.0 GPU-hour budget expires.

3. **Arm 3 (Human Machine Learning Engineer Baseline)**:
   - *Participant*: Graduate-level machine learning practitioner.
   - *Constraints*: Given the identical starting repository, dataset schema, and local documentation. Internet access is severed to prevent copying published benchmarks.
   - *Budget*: Bounded by the identical 10.0 GPU-hours of training execution time and equivalent wall-clock iteration time.

---

### 2.3 Exact Stopping Conditions & Safety Guardrails

To prevent run divergence and infinite resource loops (as identified by Kirgis et al. and Lu et al.), execution terminates immediately upon meeting any of the following four conditions:

1. **Budget Depletion (Hard Wall-Clock Cutoff)**:
   $$\sum_{k=1}^{K} t_{\text{train}}^{(k)} \ge 10.0 \text{ GPU-Hours}$$
2. **Divergence Guardrail (3-Strike Divergence Rule)**:
   $$\mathbb{I}\left(\mathcal{L}_{\text{val}}^{(k)} > 3.0 \times \mathcal{L}_{\text{base}}\right) = 1 \quad \text{for } 3 \text{ consecutive trials } k, k+1, k+2$$
   If triggered, the agent must backtrack to the best-known checkpoint $\theta^*$ or terminate.
3. **Iteration Ceiling**:
   $$K \ge K_{\text{max}} = 50 \text{ discrete experimental iterations}$$
4. **Plateau Saturation**:
   $$\max_{j \in [0, 4]} |\mathcal{L}_{\text{best}}^{(k-j)} - \mathcal{L}_{\text{best}}^{(k-j-1)}| < 10^{-4}$$
   Terminates if 5 consecutive generations fail to achieve measurable improvement over the incumbent best loss.

---

## 3. Mathematical Metric Formalization (The 9 Core Metrics)

Borrowing exact operational definitions from *The AI Scientist* (Lu et al., 2024), *Reflexion* (Shinn et al., 2023), and *METR* (Kinniment et al., 2024), we formalize the 9 quantitative evaluation metrics:

### Metric 1: Human Intervention Count ($N_{\text{interv}}$)
- **Definition**: The total number of manual interventions required by a human engineer to modify code, unfreeze hung processes, or debug environment failures during the run.
- **Formula**:
  $$N_{\text{interv}} = \sum_{t=1}^{T_{\text{run}}} \mathbb{I}(\text{Human manual modification at timestamp } t)$$
- **Target Value**: $N_{\text{interv}} = 0$ (Ground truth zero-intervention).

### Metric 2: Autonomous Execution Success Rate ($S_{\text{exec}}$)
- **Definition**: The percentage of proposed code mutations that compile, execute without runtime exception, and successfully emit finite numerical loss metrics.
- **Formula**:
  $$S_{\text{exec}} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\left(\text{ExitCode}^{(k)} == 0 \;\land\; \mathcal{L}_{\text{val}}^{(k)} \in (0, \infty)\right)$$

### Metric 3: Objective Relative Improvement ($\Delta \mathcal{L}_{\text{rel}}$)
- **Definition**: The normalized percentage reduction in holdout validation cross-entropy achieved by the best discovered configuration relative to the baseline Transformer.
- **Formula**:
  $$\Delta \mathcal{L}_{\text{rel}} = \frac{\mathcal{L}_{\text{baseline}} - \min_{k \in [1, K]} \mathcal{L}_{\text{val}}^{(k)}}{\mathcal{L}_{\text{baseline}}} \times 100\%$$

### Metric 4: Effective Iteration Count ($K_{\text{eff}}$)
- **Definition**: The iteration number at which the system first discovers a solution within $\epsilon = 0.01$ of its global peak performance, measuring search efficiency.
- **Formula**:
  $$K_{\text{eff}} = \min \left\{ k \in [1, K] : \mathcal{L}_{\text{val}}^{(k)} \le \min_{j \in [1, K]} \mathcal{L}_{\text{val}}^{(j)} + \epsilon \right\}$$

### Metric 5: Total Financial & Compute Cost ($C_{\text{total}}$)
- **Definition**: The combined financial expenditure including LLM prompt tokens, completion tokens, and GPU compute hours.
- **Formula**:
  $$C_{\text{total}} = \sum_{k=1}^{K} \left( T_{\text{prompt}}^{(k)} \cdot P_{\text{in}} + T_{\text{completion}}^{(k)} \cdot P_{\text{out}} \right) + t_{\text{GPU}} \cdot P_{\text{GPU\_hour}}$$

### Metric 6: Wall-Clock Convergence Time ($T_{\text{conv}}$)
- **Definition**: Total elapsed real-world time required to achieve peak optimization.
- **Formula**:
  $$T_{\text{conv}} = \sum_{k=1}^{K_{\text{eff}}} \left( \Delta t_{\text{reasoning}}^{(k)} + \Delta t_{\text{sandbox}}^{(k)} \right)$$

### Metric 7: Claim Verification & Hallucination Rate ($H_{\text{claim}}$)
- **Definition**: Operationalized directly from *The AI Scientist's* citation hallucination metric: the fraction of agent-hypothesized improvements where the agent asserted an improvement would occur, but the empirical sandbox verifier disproved the claim ($\Delta \mathcal{L} \le 0$).
- **Formula**:
  $$H_{\text{claim}} = \frac{\sum_{k=1}^{K} \mathbb{I}\left(\text{Agent Predicted } \Delta \mathcal{L}_{\text{pred}}^{(k)} > 0 \;\land\; \Delta \mathcal{L}_{\text{actual}}^{(k)} \le 0\right)}{\sum_{k=1}^{K} \mathbb{I}\left(\text{Agent Predicted } \Delta \mathcal{L}_{\text{pred}}^{(k)} > 0\right)}$$

### Metric 8: Multi-Seed Computational Reproducibility ($R_{\text{seed}}$)
- **Definition**: Measures the stability of the discovered architecture when retrained from scratch across $S = 5$ distinct random initialization seeds.
- **Formula**:
  $$R_{\text{seed}} = 1.0 - \frac{\sigma\left(\{\mathcal{L}_{\text{val}}^{(s)}\}_{s=1}^{S}\right)}{\mu\left(\{\mathcal{L}_{\text{val}}^{(s)}\}_{s=1}^{S}\right)}$$
  Where $R_{\text{seed}} = 1.0$ represents perfect deterministic reproducibility.

### Metric 9: Self-Failure-Detection Score ($F_1^{\text{self-eval}}$)
- **Definition**: The harmonic mean of precision and recall measuring the agent's ability to self-diagnose an experimental failure in its verbal reflection step without human prompting.
- **Formula**:
  $$P_{\text{self}} = \frac{\text{True Failures Correctly Flagged}}{\text{Total Failures Flagged by Agent}}, \quad R_{\text{self}} = \frac{\text{True Failures Correctly Flagged}}{\text{Total Actual Sandbox Crashes}}$$
  $$F_1^{\text{self-eval}} = \frac{2 \cdot P_{\text{self}} \cdot R_{\text{self}}}{P_{\text{self}} + R_{\text{self}}}$$

---

## 4. Expected Empirical Outcomes & Hypothesis Matrix

| Outcome Metric | Arm 1: Autonomous Agent | Arm 2: Random / Bayesian | Arm 3: Human Engineer | Theoretical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Objective Improvement ($\Delta \mathcal{L}$)** | Moderate (+8% to +18%) | Baseline (+3% to +7%) | High (+15% to +25%) | Proves if agent beats uninformed stochastic search. |
| **Search Efficiency ($K_{\text{eff}}$)** | Low ($< 15$ iterations) | High ($> 35$ iterations) | Medium (~10 iterations) | Measures semantic search guidance. |
| **Execution Success ($S_{\text{exec}}$)** | ~75% - 85% | 100% (Pre-validated) | ~90% | Quantifies syntactic code failure rate. |
| **Dead-End Ratio ($r_{\text{dead}}$)** | ~40% - 60% | N/A (Independent) | ~20% | Quantifies exploration discipline. |
| **Specification Gaming Risk** | **0% (Exact Verifier)** | **0% (Exact Verifier)** | **0% (Exact Verifier)** | **Bypasses The AI Scientist Verifier Ceiling.** |
