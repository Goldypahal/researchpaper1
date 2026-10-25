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
                               ↓
             Automatic 7-Figure Publication Suite
```

### Why Local Open-Weight Execution is Scientifically Superior:
1. **Zero Quota Failure**: No daily request caps (RPD), rate-limits (RPM), or billing friction.
2. **100% Scientific Reproducibility**: Fixed weights, fixed 4-bit quantization, fixed decoding temperature (`0.7`), and deterministic seeds. Any researcher can download the exact same HuggingFace checkpoint and replicate your results.
3. **Dual T4 Acceleration**: Kaggle provides two 16GB Tesla T4 GPUs (32GB VRAM total). A 7B model quantized to 4-bit uses only **~5GB VRAM**, leaving the remaining GPU memory fully available for lightning-fast Transformer candidate evaluations.

---

## ⚡ Option 1: The All-in-One Single Cell (Recommended)

If you want to paste a single cell into a fresh Kaggle notebook and let it run from start to finish without any manual file uploads:

```python
# ==============================================================================
# ALL-IN-ONE KAGGLE RESEARCH RUNNER — RESEARCH PAPER 1
# ==============================================================================
import os, sys, glob, shutil
from IPython.display import Image, display

# 1. Environment & GPU Verification
print(">>> STEP 1: Verifying GPU Acceleration <<<")
!nvidia-smi
import torch
assert torch.cuda.is_available(), "FATAL: GPU not detected! Enable GPU T4 x2 in Kaggle Settings."
print(f"CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}")

# 2. Sync Repository
print("\n>>> STEP 2: Syncing Research Repository <<<")
if not os.path.exists("researchpaper1"):
    !git clone https://github.com/Goldypahal/researchpaper1.git
    %cd researchpaper1
else:
    %cd researchpaper1
    !git pull origin main

# 3. Install Dependencies
print("\n>>> STEP 3: Installing Dependencies <<<")
!pip install -q optuna scipy matplotlib transformers accelerate bitsandbytes

# 4. Optional: Run Phase 4 Noise Floor & Fidelity Calibration (~1.5 hours)
# Measures retrain noise sigma_retrain and rank agreement between 50 and 200 steps
# !python scripts/measure_noise_and_fidelity.py

# 5. Launch Experimental Sweep
# Stage A: Fast Classical Baselines + Hybrid Arms (Random, TPE, GA, Prior-Sampler)
# !python run_on_kaggle.py --seeds 10 --iterations 10 --steps 50 --arms random tpe evolutionary prior_sampler

# Stage B: Full Decomposition Sweep (LLM Named vs Anonymized vs Shuffled)
print("\n>>> STEP 5: Executing Experimental Sweep <<<")
!python run_on_kaggle.py --seeds 10 --iterations 10 --steps 50 --arms random tpe evolutionary llm --llm_modes named_real anonymized_real named_shuffled --provider local --model Qwen/Qwen2.5-7B-Instruct --quantization 4bit

# 6. Run Statistical Decomposition & FWER Corrections
print("\n>>> STEP 6: Generating Statistical Decomposition Report <<<")
!python analysis/generate_decomposition_analysis.py
print("\n>>> STEP 6: Exporting Artifact Archive <<<")
src_zip = "kaggle_experiment_results.zip"
dst_zip = "/kaggle/working/kaggle_experiment_results.zip"
if os.path.exists(src_zip):
    shutil.copy2(src_zip, dst_zip)
    print(f"\n[SUCCESS] Packaged archive ready: {dst_zip} ({os.path.getsize(dst_zip)/(1024**2):.2f} MB)")
    print("Download 'kaggle_experiment_results.zip' from the right-hand Kaggle Output sidebar!")
```

---

## ⚡ Option 2: Step-by-Step Modular Notebook

You can also upload [`kaggle_research_runner.ipynb`](file:///c:/Users/Asus/OneDrive/Desktop/Researchpapers/Research_Paper_1_AI_Improves_AI/kaggle_research_runner.ipynb) directly into Kaggle.

### Settings Checklist:
1. **Accelerator**: Select **GPU T4 x2** (Free 30h/week on Kaggle).
2. **Internet**: Toggle to **Internet on** (required to load HuggingFace weights and git sync).

### Cells Breakdown:
* **Cell 1**: GPU environment check (`!nvidia-smi`, CUDA memory).
* **Cell 2**: Repo sync (`git clone` or `git pull origin main`).
* **Cell 3**: Dependency install (`optuna`, `transformers`, `bitsandbytes`).
* **Cell 4**: Execution command with configurable `--seeds 10` or `--seeds 50`.
* **Cell 5**: Inline publication figure viewer (`IPython.display.Image`).
* **Cell 6**: Export to `/kaggle/working/kaggle_experiment_results.zip`.

---

## 📥 How to Bring Results Back to Your Local Repository

1. Once the notebook finishes running, look at the right sidebar under **"Output"** -> `/kaggle/working/`.
2. Click the three dots next to `kaggle_experiment_results.zip` and select **"Download"**.
3. Extract the zip into your local `Research_Paper_1_AI_Improves_AI` directory.
4. All traces, statistical reports, updated figures, and `EXPERIMENT_REGISTRY.json` entries will immediately sync!
