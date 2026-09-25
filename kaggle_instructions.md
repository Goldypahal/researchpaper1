# Running Experimental Sweeps on Kaggle (Zero Laptop Thermal Load)

This guide explains how to offload all heavy model training, comparative search sweeps, and reasoning ablations to **Kaggle's free high-performance GPUs (NVIDIA T4 / P100)**, completely eliminating CPU overheating on your laptop.

---

## ⚡ Step-by-Step Instructions (Takes ~1 Minute to Start)

### Step 1: Open a New Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100** (Free 30h/week).
   - **Internet**: Toggle to **Internet on** (required to clone the repo and install packages).
3. Under the top menu bar, click **"Add-ons"** -> **"Secrets"**:
   - Add `NVIDIA_API_KEY`, `GROQ_API_KEY`, and/or `MISTRAL_API_KEY` (also supports `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
   - *(Note: Gemini's free tier is restricted to ~20 requests/day, making it unsuitable for multi-seed sweeps. NVIDIA Nemotron, Groq GPT-OSS, and Mistral are prioritized).*
   - The test script will auto-detect and authenticate without exposing your keys in code!

---

### Step 2: Run the Pipeline

You can either upload `kaggle_research_runner.ipynb` directly, or paste the following code into the notebook cells:

#### Cell 1: Environment & GPU Verification
```python
!nvidia-smi
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

#### Cell 2: Clone the Research Repository
```python
import os
if not os.path.exists("researchpaper1"):
    !git clone https://github.com/Goldypahal/researchpaper1.git
    %cd researchpaper1
else:
    %cd researchpaper1
    !git pull origin main
```

#### Cell 3: Install Required Dependencies
```python
!pip install -q optuna scipy matplotlib openai anthropic
```

#### Cell 4: LLM Qualification & Pre-Flight Benchmark (Smoke Test)
Before committing to an expensive multi-seed sweep, qualify the triad of reasoning candidates under the exact same Dyck-4 sequence modeling prompt:
```
                LLM qualification
                          │
          ┌───────────────┼────────────────┐
          ↓               ↓                ↓
       NVIDIA            Groq            Mistral
       Nemotron          GPT-OSS         candidate
          │                │
       Super/Ultra       20B/120B
```

```python
# Evaluates NVIDIA Nemotron -> Groq GPT-OSS -> Mistral Candidate across 3 identical calls:
!python benchmark_providers.py --calls 3
```
*(To test specific models: `!python benchmark_providers.py --nvidia-model nvidia/llama-3.1-nemotron-70b-instruct --groq-model openai/gpt-oss-120b --mistral-model mistral-small-latest`)*

#### Cell 5: Launch the Full Matched-Compute GPU Sweep
Choose your qualified provider and model as a **frozen experimental variable** across all seeds:
```python
# Frozen variable: all seeds [42, 101, 202, 303, 404] evaluated with the exact same chosen provider/model:
!python run_on_kaggle.py --seeds 5 --iterations 10 --steps 50 --provider nvidia --model nvidia/llama-3.1-nemotron-70b-instruct
```
*(Or for Groq: `--provider groq --model openai/gpt-oss-120b`; or for Mistral: `--provider mistral --model mistral-small-latest`)*
*(To study provider dependence, run separate distinct conditions: `--provider groq`, `--provider mistral`, `--provider gemini`)*

#### Cell 6: Save & Download Results
```python
!cp kaggle_experiment_results.zip /kaggle/working/
print("Complete! Download 'kaggle_experiment_results.zip' from the right-hand Output panel.")
```

---

## 📥 How to Bring Results Back to Your Local Repository

1. Once the notebook finishes running, look at the right sidebar under **"Output"** -> `/kaggle/working/`.
2. Click the three dots next to `kaggle_experiment_results.zip` and select **"Download"**.
3. Extract the zip into your local `Research_Paper_1_AI_Improves_AI` directory. All traces, statistical reports, and `EXPERIMENT_REGISTRY.json` entries will immediately sync!

---

## 🚀 Why This Is Substantially Better:
- **Zero Local Heat**: Your laptop CPU/GPU stays at 0% utilization and completely cool.
- **10x–50x Faster Execution**: NVIDIA T4/P100 tensor cores evaluate Transformer architectures in milliseconds per batch rather than seconds per batch on CPU.
- **Permits Large Sample Sweeps ($N=30$ to $50$)**: You can scale seeds from $N=5$ to $N=30$ or $50$ effortlessly in Kaggle without locking up your machine.
