# QUICKSTART: Person A Pipeline

## Prerequisites

1. **Download iCub Tactile Braille Dataset**
   - Download from: [public iCub dataset source]
   - Extract to: `data/icub_braille_raw/`
   - Structure:
     ```
     data/icub_braille_raw/
       a/, b/, c/, ..., z/, space/
         0.npy, 1.npy, 2.npy, ...
     ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Run (Recommended)

Run the complete pipeline in one command:

```bash
python run_pipeline.py
```

This will:

- Phase 1: Load & validate iCub dataset
- Phase 2: Build 300-500 word vocabulary
- Phase 3: Generate synthetic noisy corpus
- Phase 4: Train denoising autoencoder
- Phase 5: Train letter-level baseline classifier
- Phase 6: Compute WER/CER baseline metrics

**Estimated time:** 2-4 hours (depends on hardware, GPU available, dataset size)

## Step-by-Step (Alternative)

If you prefer to run each step separately:

### Step 1: Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

- Loads iCub dataset
- Validates 27 characters
- Computes per-character statistics
- Builds word list with train/val/test splits

### Step 2: Corpus Synthesis

```bash
jupyter notebook notebooks/02_synthesis_validation.ipynb
```

- Concatenates letter recordings into words
- Injects realistic boundary noise
- Generates train/val/test splits with (noisy, clean, label)
- Visualizes synthesis with boundary regions

### Step 3: Model Training

```bash
jupyter notebook notebooks/03_dae_training.ipynb
```

- Trains denoising autoencoder on corpus
- Trains letter-level baseline classifier
- Ready for final evaluation

### Step 4: Evaluate Baseline

```bash
python evaluate_baseline.py
```

- Computes WER/CER for condition ① (letter-level classification)
- Shows baseline performance metrics

## Verify Handoff

After pipeline completes, verify all deliverables:

```bash
python verify_handoff.py
```

This checks:

- ✓ All required files present
- ✓ Data integrity (corpus structure, normalization params)
- ✓ Denoiser interface works
- ✓ Ready to hand off to Person B

## Key Artifacts for Person B

After successful completion, Person B should receive:

```
✓ data/corpus.h5              — Synthetic (noisy, clean, label) dataset
✓ data/wordlist_splits.json   — Word assignments to train/val/test
✓ data/splits.json            — Same, with better name for Person B
✓ data/norm_params.json       — Per-channel normalization (mean, std)
✓ config/noise.yaml           — Noise configuration (boundary parameters)
✓ checkpoints/dae_best.pt     — Trained denoising autoencoder weights
✓ models/denoiser.py          — Callable denoise(signal) interface
✓ evaluate.py                 — Shared WER/CER evaluation script
```

## Quick Import Test (Person B)

Person B verifies the handoff works with:

```python
# Test 1: Denoiser interface
from models.denoiser import denoise
import h5py

f = h5py.File("data/corpus.h5", "r")
noisy = f["test/0/noisy"][:]  # (T, 12)
clean = denoise(noisy)         # (T, 12)
f.close()
print(f"✓ Denoiser works: {noisy.shape} -> {clean.shape}")

# Test 2: Normalization params
import json
norm = json.load(open("data/norm_params.json"))
print(f"✓ Normalization loaded: {len(norm['mean'])} channels")

# Test 3: Corpus structure
with h5py.File("data/corpus.h5", "r") as f:
    print(f"✓ Corpus splits: {list(f.keys())}")
    print(f"✓ Train samples: {len(f['train'])}")
```

## Troubleshooting

**Q: "Dataset directory not found"**

- A: Download iCub dataset and extract to `data/icub_braille_raw/`

**Q: "Out of memory during training"**

- A: Reduce batch size in `run_pipeline.py` (e.g., `batch_size=8`)

**Q: "Module import errors"**

- A: Ensure all Python files are in the repository and `__init__.py` exists

**Q: "Training is too slow"**

- A: Ensure you have GPU support (CUDA) and `torch` is using it

**Q: "Corpus generation failed"**

- A: Check that you have enough disk space (~1-2 GB for corpus.h5)

## File Structure

```
.
├── data/
│   ├── icub_braille_raw/      [download dataset here]
│   ├── wordlist.txt            [generated: 300-500 words]
│   ├── wordlist_splits.json    [generated: train/val/test]
│   ├── splits.json             [generated: for Person B]
│   ├── corpus.h5               [generated: synthetic corpus]
│   ├── norm_params.json        [generated: normalization]
│   └── signal_stats.json       [generated: per-char stats]
│
├── config/
│   └── noise.yaml              [boundary noise config]
│
├── models/
│   ├── __init__.py
│   ├── dataset.py              [Module 1]
│   ├── wordlist.py             [Module 2]
│   ├── synthesis.py            [Module 3]
│   ├── dae.py                  [Module 4]
│   ├── baseline.py             [Module 5]
│   └── denoiser.py             [EXPORT: Person B uses this]
│
├── checkpoints/
│   ├── dae_best.pt             [best denoising autoencoder]
│   └── letter_baseline.pt      [best letter classifier]
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthesis_validation.ipynb
│   └── 03_dae_training.ipynb
│
├── run_pipeline.py             [master orchestration script]
├── evaluate.py                 [shared evaluation (WER/CER)]
├── evaluate_baseline.py        [compute baseline metrics]
├── verify_handoff.py           [final verification]
├── requirements.txt            [dependencies]
├── README.md                   [full documentation]
└── QUICKSTART.md               [this file]
```

## Key Concepts

**Three Shared Contracts with Person B:**

1. **Signal Format**
   - Type: float32
   - Shape: (T, 12) — time steps × 12 taxel channels
   - Values: [-1, 1] after normalization
   - Normalization: (signal - mean) / std

2. **Splits & Words**
   - Fixed train/val/test split: 70%/15%/15%
   - Random seed: 42 (reproducible)
   - Word list: 300-500 words, a-z only, length 3-8

3. **Noise Configuration**
   - Boundary blend window: 8 timesteps
   - Noise scale: 15% of local signal std
   - Both modules use same `config/noise.yaml`

## Useful Commands

```bash
# Check everything is working
python verify_handoff.py

# Run full pipeline
python run_pipeline.py

# Compute baseline WER/CER
python evaluate_baseline.py

# Interactive exploration
jupyter notebook

# View generated corpus
python -c "import h5py; f=h5py.File('data/corpus.h5'); print(list(f['train'].keys())[:5]); f.close()"
```

## Next: Person B's Role

Once you hand off these artifacts to Person B, they will:

1. Load noisy corpus with `denoise()` preprocessing
2. Train CNN-LSTM + CTC decoder
3. Evaluate on test set
4. Compare against baseline WER/CER

Good luck! 🚀
