# Running Experimental Sweeps on Kaggle (Zero Laptop Thermal Load)

This guide explains how to execute the autonomous research pipeline on **Kaggle's free high-performance GPUs (dual NVIDIA Tesla T4 16GB)**.

---

## 🏛️ Experimental Architecture: Local-First Scientific Design

```
                PRIMARY SCIENTIFIC EVIDENCE (Experiment A)
                               │
                      Local Open-Weight LLM
              (Qwen2.5-7B-Instruct with 4-bit on T4 GPU)
                               │
                      Reproducible Sweep
                               │
               ┌───────────────┴───────────────┐
               ↓                               ↓
        Classical Search                Semantic Search
      (Random / TPE / GA)              (Local LLM Agent)
               │                               │
               └───────────────┬───────────────┘
                               ↓
                      Statistical Analysis

               SECONDARY ROBUSTNESS EVIDENCE (Experiment B)
                               │
                 Optional Commercial Backends
               (Groq / Mistral / Nemotron / Gemini)
```

### Why Local Open-Weight Execution is Scientifically Superior:
1. **Zero Quota Failure**: No daily request caps (RPD), rate-limits (RPM), or billing friction.
2. **100% Scientific Reproducibility**: Fixed weights, fixed 4-bit quantization, fixed decoding temperature (`0.7`), and deterministic seeds. Any researcher can download the exact same HuggingFace checkpoint and replicate your results.
3. **Dual T4 Acceleration**: Kaggle provides two 16GB Tesla T4 GPUs (32GB VRAM total). A 7B model quantized to 4-bit uses only **~5GB VRAM**, leaving the remaining GPU memory fully available for lightning-fast Transformer candidate evaluations.

---

## ⚡ Step-by-Step Instructions

### Step 1: Open a New Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU T4 x2** (Free 30h/week).
   - **Internet**: Toggle to **Internet on** (required to clone repo and load model weights).
3. *(Optional for Experiment B)*: If running commercial API comparisons under **"Add-ons"** -> **"Secrets"**, you can add `GROQ_API_KEY`, `MISTRAL_API_KEY`, or `NVIDIA_API_KEY`. For Experiment A (primary), **NO API KEYS ARE NEEDED!**

---

### Step 2: Run the Notebook Cells

You can either upload [`kaggle_research_runner.ipynb`](file:///c:/Users/Asus/OneDrive/Desktop/Researchpapers/Research_Paper_1_AI_Improves_AI/kaggle_research_runner.ipynb) directly, or run the following cells:

#### Cell 1: Environment & GPU Verification
```python
!nvidia-smi
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

#### Cell 2: Clone Research Repository
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
# Install search baselines and local HuggingFace inference stack with 4-bit quantization
!pip install -q optuna scipy matplotlib transformers accelerate bitsandbytes
```

#### Cell 4: Primary Experiment — Full GPU Sweep with Local Open-Weight Model
```python
# Primary Scientific Evidence (Experiment A):
# Runs Random vs TPE vs GA vs Local LLM Agent (Qwen2.5-7B 4-bit) across Dyck & FSM:
!python run_on_kaggle.py --seeds 5 --iterations 10 --steps 50 --provider local --model Qwen/Qwen2.5-7B-Instruct --quantization 4bit
```
*(For even faster inference on smaller memory footprints, you can also use `--model Qwen/Qwen2.5-3B-Instruct`)*

#### Cell 5 (Optional): Secondary Robustness Experiment (Commercial Backends)
If testing model-dependence (Experiment B):
```python
# Optional robustness checks using API backends:
# Groq:
# !python run_on_kaggle.py --seeds 5 --iterations 10 --steps 50 --provider groq --model openai/gpt-oss-120b
# Mistral:
# !python run_on_kaggle.py --seeds 5 --iterations 10 --steps 50 --provider mistral --model mistral-small-latest
# NVIDIA Nemotron:
# !python run_on_kaggle.py --seeds 5 --iterations 10 --steps 50 --provider nvidia --model nvidia/llama-3.1-nemotron-70b-instruct
```

#### Cell 6: Save & Download Results
```python
!cp kaggle_experiment_results.zip /kaggle/working/
print("Complete! Download 'kaggle_experiment_results.zip' from the right-hand Kaggle Output sidebar.")
```

---

## 📥 How to Bring Results Back to Your Local Repository

1. Once the notebook finishes running, look at the right sidebar under **"Output"** -> `/kaggle/working/`.
2. Click the three dots next to `kaggle_experiment_results.zip` and select **"Download"**.
3. Extract the zip into your local `Research_Paper_1_AI_Improves_AI` directory. All traces, statistical reports, and `EXPERIMENT_REGISTRY.json` entries will immediately sync!
