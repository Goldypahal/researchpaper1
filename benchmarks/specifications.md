# Formal Benchmark Specifications for Autonomous ML Optimization

This document defines the formal grammar, generative dynamics, data splits, and generalization gap formulations for the benchmark suite powering **ResearchPaper1**.

---

## 1. Benchmark Hierarchy & Grammar Classes

| Benchmark | Language Class | Chomsky Level | Latent Structure | Minimal Required Computational Inductive Bias |
| :--- | :--- | :--- | :--- | :--- |
| **True Dyck-$k$ ($k=4$)** | Context-Free Language (CFL) | Level 2 | Hierarchical stack memory $\mathcal{O}(d)$ | Stack/prefix memory, long-range attention tracking |
| **Hidden FSM** | Regular Language | Level 3 | Hidden algebraic state transition $T: S \times \Sigma \to S$ | Finite state tracking, recurrent/Markovian state updates |
| **Parity Legacy** | Regular / Parity Counter | Level 3 | Modulo counter over token attributes | Global parity accumulation (legacy compatibility) |

---

## 2. Benchmark 1: True Dyck-$k$ ($k=4$)

### 2.1 Grammar Definition
Dyck-$k$ is the language of balanced bracket structures over $k$ types of brackets.
Here $k=4$, with bracket types:
- Type 0: `(` (token 10), `)` (token 11)
- Type 1: `[` (token 12), `]` (token 13)
- Type 2: `{` (token 14), `}` (token 15)
- Type 3: `<` (token 16), `>` (token 17)

Distractor tokens are drawn uniformly at random from vocabulary space:
$$\tau_{\text{distractor}} \sim \text{Uniform}(\{20, 21, \dots, V-1\}), \quad V=1024$$

A sequence $x = (x_1, x_2, \dots, x_T)$ of length $T$ is syntactically valid if and only if:
1. Every closing bracket $x_t \in \{11, 13, 15, 17\}$ matches the most recent unmatched opening bracket $\text{Top}(\text{Stack}_{t-1})$.
2. Upon emission of $x_t$, the stack updates via:
   $$\text{Stack}_t = \begin{cases} \text{Stack}_{t-1} \mathbin{\Vert} x_t & \text{if } x_t \in \{10, 12, 14, 16\} \\ \text{Pop}(\text{Stack}_{t-1}) & \text{if } x_t \in \{11, 13, 15, 17\} \\ \text{Stack}_{t-1} & \text{otherwise} \end{cases}$$
3. All brackets opened within the horizon are closed by $T$, or the stack remaining at $t$ satisfies $| \text{Stack}_t | \le T - t$.

### 2.2 Formal Splits & Depth Boundaries
To test inductive generalization rather than superficial memorization, data splits are stratified strictly by stack nesting depth $d$:

$$d(x) = \max_{1 \le t \le T} |\text{Stack}_t|$$

- **In-Distribution Training Split ($D_{\text{train}}$)**:
  - $N_{\text{train}} = 1000$ sequences, sequence length $T = 128$.
  - Nesting depth: $d \in [1, 6]$. Bracket probability: $p = 0.65$.
- **In-Distribution Validation Split ($D_{\text{val}}$)**:
  - $N_{\text{val}} = 200$ sequences, $T = 128$, $d \in [1, 6]$, disjoint seed.
- **In-Distribution Test Split ($D_{\text{test}}$)**:
  - $N_{\text{test}} = 200$ sequences, $T = 128$, $d \in [1, 6]$, disjoint seed.
- **Out-of-Distribution Test Split ($D_{\text{OOD}}$)**:
  - $N_{\text{OOD}} = 300$ sequences, $T = 128$.
  - Nesting depth: $d \in [7, 12]$, bracket probability: $p = 0.75$.
  - Evaluates recursive generalization beyond the maximum training horizon ($d_{\text{OOD}} > d_{\text{train}}$).

### 2.3 Non-Overlap & Leakage Proof
For any generated dataset, we guarantee by construction:
$$D_{\text{train}} \cap D_{\text{val}} = \emptyset, \quad D_{\text{train}} \cap D_{\text{test}} = \emptyset, \quad D_{\text{val}} \cap D_{\text{test}} = \emptyset$$
$$D_{\text{train}} \cap D_{\text{OOD}} = \emptyset \quad (\text{since } \forall x \in D_{\text{OOD}}, \, d(x) \ge 7 > \max_{x' \in D_{\text{train}}} d(x'))$$

---

## 3. Benchmark 2: Hidden Finite State Machine (Hidden FSM)

### 3.1 Formal Definition
The Hidden FSM is defined as a deterministic Mealy/Moore automaton:
$$\mathcal{M} = (S, \Sigma, T, E, s_0)$$
- $S$: Latent state space $\{0, 1, \dots, |S|-1\}$.
- $\Sigma$: Input symbol alphabet $\{0, 1, \dots, |\Sigma|-1\}$, $|\Sigma|=16$.
- $T: S \times \Sigma \to S$: State transition matrix.
- $E: S \times \Sigma \to \{10, 11, \dots, 10 + |\Sigma| - 1\}$: Observation emission matrix.
- $s_0 \sim \text{Uniform}(S)$: Initial latent state.

At each step $t$:
1. A latent symbol $\sigma_t \sim \text{Uniform}(\Sigma)$ is drawn.
2. An observable token $e_t = E(s_t, \sigma_t)$ is emitted.
3. The latent state updates algebraically: $s_{t+1} = T(s_t, \sigma_t)$.

The model only observes the emitted sequence $(e_1, e_2, \dots, e_T)$ and must implicitly infer the transition dynamics of $T$ to predict $e_{t+1}$.

### 3.2 Formal Splits
- **In-Distribution Training Split ($D_{\text{train}}$)**:
  - $|S| = 8$ latent states, $|\Sigma| = 16$ alphabet, $N_{\text{train}} = 1000$, $T = 128$.
- **In-Distribution Validation Split ($D_{\text{val}}$)**:
  - $|S| = 8$, $|\Sigma| = 16$, $N_{\text{val}} = 200$, $T = 128$, disjoint sequences.
- **In-Distribution Test Split ($D_{\text{test}}$)**:
  - $|S| = 8$, $|\Sigma| = 16$, $N_{\text{test}} = 200$, $T = 128$, disjoint sequences.
- **Out-of-Distribution Test Split ($D_{\text{OOD}}$)**:
  - Doubled state-space: $|S_{\text{OOD}}| = 16$ latent states, $N_{\text{OOD}} = 300$, $T = 128$.
  - Tests capacity scaling when the underlying state-space complexity exceeds training capacity.

---

## 4. Evaluation Metrics & Generalization Gap Formulation

For any candidate architecture $\theta$:

### 4.1 In-Distribution Cross-Entropy Loss ($L_{\text{ID}}$)
$$L_{\text{ID}}(\theta) = -\frac{1}{|D_{\text{val}}|} \sum_{x \in D_{\text{val}}} \frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{1:t})$$

### 4.2 Out-of-Distribution Loss ($L_{\text{OOD}}$)
$$L_{\text{OOD}}(\theta) = -\frac{1}{|D_{\text{OOD}}|} \sum_{x \in D_{\text{OOD}}} \frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{1:t})$$

### 4.3 Structural Target Accuracy ($\text{Acc}_{\text{structural}}$)
Evaluated strictly at closing bracket positions where deterministic stack reduction is required:
$$\text{Acc}_{\text{structural}}(\theta) = \frac{\sum_{x \in D} \sum_{t=1}^{T-1} \mathbb{I}(M_{x, t+1} = 1) \cdot \mathbb{I}(\hat{x}_{t+1} = x_{t+1})}{\sum_{x \in D} \sum_{t=1}^{T-1} \mathbb{I}(M_{x, t+1} = 1)} \times 100\%$$
where $M_{x, t} = 1$ if $x_t$ is a closing bracket, and $\hat{x}_{t+1} = \arg\max_v P_\theta(v \mid x_{1:t})$.

### 4.4 Absolute & Relative Generalization Gaps
$$\text{Gap}(\theta) = L_{\text{OOD}}(\theta) - L_{\text{ID}}(\theta)$$
$$\text{Gap}_{\text{relative}}(\theta) = \frac{L_{\text{OOD}}(\theta) - L_{\text{ID}}(\theta)}{L_{\text{ID}}(\theta)}$$

A robust, generalizing architecture minimizes both $L_{\text{ID}}$ and $\text{Gap}_{\text{relative}}$.
