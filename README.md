# BrailleBridge

BrailleBridge is an end-to-end Braille recognition project that combines tactile signal processing, denoising, sequence decoding, and an accessible web interface.

It includes:
- Data synthesis from tactile Braille recordings
- Denoising and decoding pipeline
- FastAPI backend for inference
- React frontend with dot-tap Braille input
- Accessibility features (speech output, spell mode, export bundle)

## What This Repository Contains

### Core pipeline
- `models/dataset.py`: iCub tactile dataset loading and validation
- `models/wordlist.py`: word filtering and split generation
- `models/synthesis.py`: synthetic noisy/clean word-signal generation
- `models/dae.py`: denoising autoencoder training
- `models/denoiser.py`: denoise interface for downstream use
- `models/ctc_model.py`: CTC decoding model components
- `models/pipeline.py`: decoding pipeline helpers

### Training and evaluation
- `train/train_ctc.py`: CTC training
- `train/train_bigram.py`: language model / bigram training
- `eval/run_ablation.py`: ablation script
- `evaluate.py`: evaluation utilities
- `evaluate_baseline.py`: baseline evaluation
- `verify_handoff.py`: artifact checks

### Deployment
- `app/api.py`: FastAPI routes
- `app/backend.py`: inference backend logic
- `frontend/`: React + Vite web app

### Notebooks
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_synthesis_validation.ipynb`
- `notebooks/03_dae_training.ipynb`

## Data Requirements

The project expects iCub tactile Braille raw data at:

`data/icub_braille_raw/`

Expected class folders:
- `a` ... `z`
- `space`

Each recording should be a `.npy` array with shape `(T, 12)`.

## Setup (Consolidated)

### 1. Create and activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".venv\Scripts\Activate.ps1")
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

For frontend:

```powershell
cd frontend
npm install
cd ..
```

### 3. Prepare dataset

Place the iCub tactile Braille data under `data/icub_braille_raw/`.

## Running the Project

### Option A: Run full pipeline

```powershell
python run_pipeline.py
```

### Option B: Run step-by-step with notebooks

```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_synthesis_validation.ipynb
jupyter notebook notebooks/03_dae_training.ipynb
```

### Start backend

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Or with explicit interpreter:

```powershell
& ".venv\Scripts\python.exe" -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

### Start frontend

```powershell
cd frontend
npm run dev
```

If needed, set API URL:

```powershell
$env:VITE_API_URL='http://127.0.0.1:8000'
```

## Verification (Consolidated)

Run these checks after setup/training:

### 1. Module import check

```powershell
python -c "from models.dataset import get_dataset; from models.denoiser import denoise; print('OK')"
```

### 2. Noise config check

```powershell
python -c "import yaml; yaml.safe_load(open('config/noise.yaml')); print('noise.yaml OK')"
```

### 3. Handoff/artifact verification

```powershell
python verify_handoff.py
```

### 4. Baseline evaluation

```powershell
python evaluate_baseline.py
```

## Key Artifacts

After successful runs, typical generated artifacts are:
- `data/corpus.h5`
- `data/wordlist_splits.json`
- `data/splits.json`
- `data/norm_params.json`
- `checkpoints/dae_best.pt`
- `checkpoints/letter_baseline.pt`

## Accessibility Features in Frontend

Dot-tap mode includes:
- Standard Braille dot mapping
- Spoken prediction
- Spell mode (letter-by-letter speech)
- Low-confidence spoken alternatives
- Export bundle with word, Braille symbols, and regenerated synthetic signal JSON

## Troubleshooting

### Dataset not found
Ensure `data/icub_braille_raw/` exists and has the required class folders.

### Import errors
Run commands from project root and ensure virtual environment is active.

### Backend not reachable from frontend
- Confirm backend is running on `127.0.0.1:8000`
- Set `VITE_API_URL` if frontend points elsewhere

### Memory issues during training
Reduce batch size in training configurations/scripts.

## Additional Documents

- `QUICKSTART.md`
- `IMPLEMENTATION_SUMMARY.md`
- `INDEX.md`

## Project Ownership

This project is made by **Advika Nagool**,**Thejovati Narayanan** and **Lakshita Mukti**.
