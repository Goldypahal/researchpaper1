# Arm 3: Standardized Human Engineering Baseline Protocol

## 1. Objective & Methodological Equivalence

To establish a scientifically rigorous, fair empirical benchmark against **Arm 1 (Autonomous LLM Agent)** and **Arm 2 (Random Search Baseline)**, this protocol defines the exact procedure for human machine learning engineers and researchers to optimize the small Transformer architecture on the unseen synthetic Dyck algebraic state-transition sequence modeling task.

### Strict Methodological Controls
To guarantee that differences in outcomes are attributable strictly to **search policy and inductive bias** rather than evaluation disparity, human experimenters are bound to the identical constraints as Arms 1 and 2:

1. **Identical Search Space**:
   - Learning rate $\eta \in [10^{-5}, 5 \times 10^{-2}]$
   - Weight decay $\lambda \in [0.0, 0.2]$
   - Number of layers $L \in \{2, 4, 6, 8\}$
   - Number of attention heads $H \in \{2, 4, 8\}$
   - Model dimension $d_{\text{model}} \in \{128, 256, 512\}$ (must satisfy $d_{\text{model}} \pmod H = 0$)
   - Activation function $\sigma \in \{\text{gelu}, \text{relu}, \text{silu}\}$
   - Normalization type $\text{Norm} \in \{\text{layernorm}, \text{rmsnorm}\}$
   - Attention scaling factor $\gamma \in \text{None} \cup [0.05, 0.5]$

2. **Identical Compute Budget**:
   - Maximum cumulative training budget: **10.0 GPU-hours** (or equivalent 50 training steps per trial).
   - Maximum iteration cap: **50 iterations**.

3. **Identical Stopping & Backtracking Conditions**:
   - **3-Strike Divergence Rule**: If 3 consecutive trials diverge ($\mathcal{L}_{\text{val}} > 3 \times \mathcal{L}_{\text{base}}$) or crash, the session terminates immediately.
   - **Backtracking Rule**: If a proposed trial fails to beat the incumbent best validation loss, the candidate is marked `REJECT_AND_BACKTRACK`, and the next hypothesis must branch from the incumbent best.

---

## 2. Standardized Trial Log Template

Human participants must record every trial in this exact schema before and after running the sandbox evaluation harness:

```bash
# Command to execute a human-selected trial:
python Research_Paper_1_AI_Improves_AI/agent_scaffold/eval_harness.py \
  --no-dry-run \
  --steps 50 \
  --lr <float> \
  --weight-decay <float> \
  --n-layers <int> \
  --n-heads <int> \
  --d-model <int> \
  --activation <gelu|relu|silu> \
  --norm-type <layernorm|rmsnorm> \
  [--scale-factor <float>]
```

### Trial Record Sheet

| Field | Description | Example Record |
| :--- | :--- | :--- |
| **Iteration Index** | Chronological integer $k \ge 0$ | `Iteration 1` |
| **Timestamp Start** | Wall-clock start of human reasoning | `2026-09-23 10:15:00 UTC` |
| **Timestamp End** | Wall-clock completion of human decision | `2026-09-23 10:27:00 UTC` |
| **Human Think Time ($T_{\text{human}}$)** | Active minutes spent reading logs and reasoning | `12 minutes` |
| **Target Component** | Targeted inductive bias | `depth_width` \| `optimizer` \| `normalization` \| `activation` \| `attention` |
| **Hypothesis Rationale** | Formal technical justification | *"Increasing depth to 8 layers with RMSNorm will stabilize multi-step Dyck state tracking while mitigating gradient norm variance."* |
| **Predicted Delta Loss ($\widehat{\Delta \mathcal{L}}$)** | Quantitative loss reduction expected by the human | `+0.0400` |
| **Proposed Hyperparameters** | Concrete parameter dictionary | `{"lr": 0.0005, "weight_decay": 0.05, "n_layers": 8, "n_heads": 4, "d_model": 256, "activation": "gelu", "norm_type": "layernorm"}` |
| **Sandbox Execution Status** | `SUCCESS` \| `DIVERGED` \| `RUNTIME_CRASH` | `SUCCESS` |
| **GPU Execution Time** | Runtime in seconds reported by harness | `67.02s` |
| **Final Train Loss** | Cross-entropy loss on train set | `6.7926` |
| **Final Val Loss ($\mathcal{L}_{\text{val}}$)** | Cross-entropy loss on unseen holdout | `7.0295` |
| **Relative Gain ($\Delta \mathcal{L}_{\text{rel}}$)** | $\frac{\mathcal{L}_{\text{base}} - \mathcal{L}_{\text{val}}}{\mathcal{L}_{\text{base}}} \times 100\%$ | `+0.51%` |
| **Branch Decision** | `ACCEPT_AND_EXTEND` or `REJECT_AND_BACKTRACK` | `ACCEPT_AND_EXTEND` (New incumbent best) |
| **Self-Detected Failure** | Did the human anticipate failure before running? | `False` |
| **Post-Run Reflection** | Human retrospective on why hypothesis held/failed | *"Reducing head count to 4 constrained parameter growth and prevented overfitting observed in Iteration 1."* |

---

## 3. Comparative Metric Computation

Upon completion of the 10 GPU-hour budget or stopping condition, human trial logs are compiled into the identical 9 metrics evaluated for Arm 1 and Arm 2:

1. **Number of Interventions ($N_{\text{interv}}$)**: Total human decisions made.
2. **Success Rate ($S_{\text{exec}}$)**: Fraction of runs with status `SUCCESS` and non-divergent loss.
3. **Relative Loss Improvement ($\Delta \mathcal{L}_{\text{rel}}$)**: Percentage improvement from baseline anchor to final incumbent.
4. **Effective Knowledge Yield ($K_{\text{eff}}$)**: Ratio of hypotheses that achieved statistical improvement over prior incumbent:
   $$K_{\text{eff}} = \frac{N_{\text{accepted}}}{N_{\text{total}}}$$
5. **Total Compute Cost ($C_{\text{total}}$)**: Cumulative GPU wall-clock seconds spent across all sandbox runs.
6. **Path to First Improvement ($T_{\text{conv}}$)**: Iteration index at which the first statistically valid improvement over baseline occurred.
7. **Hypothesis Claim Accuracy ($H_{\text{claim}}$)**: Mean absolute error between predicted delta loss $\widehat{\Delta \mathcal{L}}$ and observed delta loss $\Delta \mathcal{L}$:
   $$\text{MAE}_{\text{pred}} = \frac{1}{M} \sum_{i=1}^{M} |\widehat{\Delta \mathcal{L}}_i - \Delta \mathcal{L}_i|$$
8. **Dead-End Ratio ($\text{DER}$)**: Fraction of iterations that resulted in backtracking:
   $$\text{DER} = \frac{N_{\text{backtrack}}}{N_{\text{total}}}$$
9. **Total Human Labor Time ($\sum T_{\text{human}}$)**: Net human engineering labor in hours (measured strictly as 0.0 for Arm 1 and Arm 2).
