# Running Experimental Sweeps on Kaggle (Zero Laptop Thermal Load)

This guide explains how to offload all heavy model training, comparative search sweeps, and reasoning ablations to **Kaggle's free high-performance GPUs (NVIDIA T4 / P100)**, completely eliminating CPU overheating on your laptop.

---

## ⚡ Step-by-Step Instructions (Takes ~1 Minute to Start)

### Step 1: Open a New Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100** (Free 30h/week).
   - **Internet**: Toggle to **Internet on** (required to clone the repo and install packages).

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
!pip install -q optuna scipy matplotlib
```

#### Cell 4: Launch the Full Matched-Compute GPU Sweep
```python
# Evaluates Random vs TPE vs GA vs LLM Agent across Dyck-4 & Hidden FSM on GPU
!python run_on_kaggle.py --seeds 5 --iterations 10 --steps 50
```

#### Cell 5: Save & Download Results
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
