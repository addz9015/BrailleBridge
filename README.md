"""
Person A: Braille Recognition Pipeline - Data & Denoising Module
================================================================

This is the upstream data synthesis and denoising autoencoder module.
Person B (CNN-LSTM + CTC decoder) consumes the outputs from this module.

# PROJECT STRUCTURE

data/
icub_braille_raw/ ← Raw iCub dataset (download separately)
a/, b/, ..., z/, space/
0.npy, 1.npy, ...

wordlist.txt ← Filtered English words (300-500)
wordlist_splits.json ← Train/val/test splits (70/15/15)
splits.json ← Same as above, for Person B
corpus.h5 ← Synthetic (noisy, clean, label) triples
norm_params.json ← Per-channel normalization (mean, std)
signal_stats.json ← Signal statistics per character

config/
noise.yaml ← Boundary noise configuration

models/
dataset.py ← Module 1: Dataset loading
wordlist.py ← Module 2: Word list construction
synthesis.py ← Module 3: Signal synthesis
dae.py ← Module 4: Denoising autoencoder
baseline.py ← Module 5: Letter-level baseline
denoiser.py ← EXPORT: denoise() function for Person B

checkpoints/
dae_best.pt ← Best denoising autoencoder weights
letter_baseline.pt ← Best baseline classifier weights

notebooks/
01_data_exploration.ipynb ← Data loading & validation
02_synthesis_validation.ipynb ← Corpus generation & visualization
03_dae_training.ipynb ← Autoencoder training

root files/
evaluate.py ← Shared evaluation script (jiwer)
evaluate_baseline.py ← Baseline WER/CER computation
run_pipeline.py ← Master orchestration script

app/
backend.py ← Inference backend module (sample generation + decoding)
api.py ← FastAPI deployment backend

frontend/
React + Vite frontend connected to the FastAPI backend

# QUICK START

1. Download iCub Braille Dataset
   ────────────────────────────────
   The iCub tactile Braille dataset is publicly available.
   Download and place in data/icub_braille_raw/

   Directory structure should be:
   data/icub_braille_raw/
   a/
   0.npy (shape: T x 12, float32)
   1.npy
   ...
   b/
   ...
   z/
   space/ (or use ' ' as directory name)

2. Install Dependencies
   ────────────────────
   pip install -r requirements.txt

3. Run Full Pipeline
   ─────────────────
   python run_pipeline.py

   This will:
   - Phase 1: Load & validate iCub dataset
   - Phase 2: Build word list (300-500 words)
   - Phase 3: Generate synthetic corpus with boundary noise
   - Phase 4: Train denoising autoencoder
   - Phase 5: Train letter-level baseline classifier
   - Phase 6: Compute WER/CER for baseline

4. Alternatively, Run Notebooks
   ─────────────────────────────
   jupyter notebook notebooks/01_data_exploration.ipynb
   jupyter notebook notebooks/02_synthesis_validation.ipynb
   jupyter notebook notebooks/03_dae_training.ipynb

5. Launch the demo frontend
   ────────────────────────
   Start the HTTP backend first:
   uvicorn app.api:app --host 127.0.0.1 --port 8000

   Then start the React frontend:
   cd frontend
   npm install
   npm run dev

   The frontend talks to the FastAPI backend over HTTP. Set VITE_API_URL
   if the backend is not running on http://127.0.0.1:8000.

   For public deployment, set API_CORS_ORIGINS to a comma-separated list of
   allowed frontend origins, for example:
   API_CORS_ORIGINS=https://your-app.example,https://www.your-app.example

# HANDOFF INTERFACE FOR PERSON B

Person B imports and uses three key artifacts:

1. Denoising Function
   ──────────────────
   from models.denoiser import denoise

   clean_signal = denoise(noisy_signal) # (T, 12) -> (T, 12)

2. Normalization Parameters
   ───────────────────────
   with open('data/norm_params.json') as f:
   norm = json.load(f)

   # Apply normalization

   normalized = (signal - norm['mean']) / (norm['std'] + 1e-8)

3. Word List & Splits
   ──────────────────
   with open('data/wordlist_splits.json') as f:
   splits = json.load(f)

   train_words = splits['train']
   test_words = splits['test']

4. Synthetic Corpus
   ────────────────
   with h5py.File('data/corpus.h5', 'r') as f:
   noisy = f['train/0/noisy'][:] # (T, 12)
   clean = f['train/0/clean'][:] # (T, 12)
   label = f['train/0'].attrs['label'] # "cat"

# MODULES OVERVIEW

Module 1: Dataset (models/dataset.py)
─────────────────────────────────────

- Loads iCub tactile Braille dataset
- Interface: get_letter(char, idx) -> (T, 12) ndarray
- Validates: all 27 characters, min 20 recordings each
- Computes: per-character signal statistics

Module 2: Word List (models/wordlist.py)
────────────────────────────────────────

- Filters English words: length 3-8, a-z only
- Target: 300-500 words
- Creates deterministic splits: 70/15/15 (seed=42)
- Output: wordlist.txt, wordlist_splits.json

Module 3: Synthesis (models/synthesis.py)
──────────────────────────────────────────

- Concatenates letter recordings into word-length signals
- Injects realistic boundary noise:
  - Blending window width: W=8 timesteps
  - Alpha-decayed interpolation at boundaries
  - Gaussian noise: 15% of local signal std
- Generates: 5 train / 2 val / 2 test samples per word
- Output: corpus.h5 with (noisy, clean, label, boundaries)

Module 4: Denoising Autoencoder (models/dae.py)
────────────────────────────────────────────────

- 1D convolutional autoencoder
- Architecture: Conv layers with stride=2 downsampling
- Training: MSE loss, Adam (lr=1e-3), batch=16
- Early stopping: patience=7 on validation loss
- Variable-length: pad to max T in batch, mask loss
- Export: standalone denoise() function

Module 5: Baseline Classifier (models/baseline.py)
────────────────────────────────────────────────────

- CNN for letter classification (27 classes: a-z + space)
- Input: fixed-length letter recording (T=150)
- Ablation condition ①: classify each letter independently
- Training: cross-entropy, Adam, 80/10/10 split
- Baseline benchmark: establishes floor performance

# Shared Contracts

Signal Format
─────────────

- Type: float32
- Shape: (T, 12) where T is variable (time steps x taxel channels)
- Values: approximately [-1, 1] after normalization
- Normalization: per-channel using train set mean/std

Splits File
───────────

- data/wordlist_splits.json: train/val/test word assignments
- Deterministic split with seed 42
- Never regenerate after Person B starts training

Noise Configuration
───────────────────

- config/noise.yaml: boundary noise parameters
- Both Person A (training) and Person B (evaluation) import this
- Ensures consistent noise model across pipeline

Evaluation Script
─────────────────

- evaluate.py: shared WER/CER computation (jiwer library)
- Used by Person A for baseline results
- Used by Person B for ablation conditions

# FILE CHECKLIST FOR HANDOFF

Before handing to Person B, verify all files exist:

Essential files:
✓ data/corpus.h5
✓ data/wordlist.txt
✓ data/wordlist_splits.json
✓ data/splits.json
✓ data/norm_params.json
✓ config/noise.yaml
✓ checkpoints/dae_best.pt
✓ models/denoiser.py
✓ evaluate.py

Optional but helpful:
✓ data/signal_stats.json
✓ checkpoints/letter_baseline.pt
✓ data/baseline_results.json

# DEPENDENCIES

Core:

- numpy
- scipy
- h5py
- yaml
- torch
- matplotlib (for plotting)

Evaluation:

- jiwer (for WER/CER)

All dependencies listed in requirements.txt

# TROUBLESHOOTING

Dataset not found
─────────────────
Q: "Dataset directory ... not found"
A: Download the iCub tactile Braille dataset and place in data/icub_braille_raw/
Dataset is publicly available (cite the paper)

Too few words in wordlist
─────────────────────────
Q: "Only N words after filtering, minimum 300 required"
A: Try relaxing length constraints or using different word source
Check config in models/wordlist.py

Out of memory during training
──────────────────────────────
Q: CUDA out of memory or CPU memory errors
A: Reduce batch size in run_pipeline.py or notebooks
Try batch_size=8 instead of 16

Module import errors
────────────────────
Q: "No module named 'models.xxx'"
A: Ensure PROJECT_ROOT is correctly set
Verify **init**.py files in all package directories

# RUNNING ACCEPTANCE TESTS

After pipeline completion, verify:

1. Dataset acceptance test
   ───────────────────────
   python -c "
   from models.dataset import get_dataset
   ds = get_dataset('data/icub_braille_raw')
   a = ds.get_letter('a', 0)
   z = ds.get_letter('z', 0)
   print(f'✓ a shape: {a.shape}, z shape: {z.shape}')
   "

2. Wordlist acceptance test
   ────────────────────────
   python -c "
   import json
   with open('data/wordlist_splits.json') as f:
   splits = json.load(f)
   total = len(splits['train']) + len(splits['val']) + len(splits['test'])
   print(f'✓ Total words: {total} (target: 300-500)')
   "

3. Denoiser acceptance test
   ─────────────────────────
   python -c "
   from models.denoiser import denoise
   import h5py, numpy as np
   f = h5py.File('data/corpus.h5')
   noisy = f['test/0/noisy'][:]
   clean = denoise(noisy)
   print(f'✓ Denoised shape: {clean.shape}')
   f.close()
   "

4. Evaluate baseline
   ──────────────────
   python evaluate_baseline.py

# CONTACT / COLLABORATION

Person A Responsibilities:

- Data loading, validation, synthesis
- Corpus generation with boundary noise
- Denoising autoencoder training
- Letter-level baseline
- Handoff of clean signals and normalization

Person B Responsibilities:

- CNN-LSTM + CTC decoder
- Full end-to-end model training
- Word-level prediction
- Final evaluation on noisy corpus

Shared Responsibility:

- Use common splits (seed 42)
- Use common noise config
- Use shared evaluate.py script
- Version control all code

# Papers / References

[Insert citations for:

- iCub tactile Braille dataset
- Boundary noise injection methodology
- Denoising autoencoder architecture
- CTC decoding (Person B)]
