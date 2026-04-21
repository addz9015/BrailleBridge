"""
Ablation Study — 4 Conditions
Run from project root AFTER training is complete:
    python eval/run_ablation.py

Produces: results/ablation.md
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import json
import numpy as np
import torch
import jiwer

from models.denoiser import denoise
from models.ctc_model import load_ctc_model
from models.pipeline import BrailleDecoder, greedy_decode
from models.dataset import get_dataset
from models.synthesis import SignalSynthesizer

os.makedirs('results', exist_ok=True)


# ─── Load test set ────────────────────────────────────────────────────────────

def load_test_set(h5_path, norm_path, max_samples=0):
    norm = json.load(open(norm_path))
    mean = np.array(norm['mean'], dtype=np.float32)
    std  = np.array(norm['std'],  dtype=np.float32)

    samples = []

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'test' in f:
                keys = sorted(f['test'].keys(), key=lambda x: int(x))
                for key in keys:
                    noisy = f['test'][key]['noisy'][:]
                    clean = f['test'][key]['clean'][:]
                    label_raw = f['test'][key].attrs['label']
                    if isinstance(label_raw, bytes):
                        label = label_raw.decode('utf-8')
                    else:
                        label = str(label_raw)
                    samples.append({'noisy': noisy, 'clean': clean,
                                    'label': label, 'mean': mean, 'std': std})
    except Exception:
        samples = []

    if not samples:
        splits = json.load(open('data/wordlist_splits.json'))
        test_words = splits.get('test', [])
        if max_samples > 0:
            test_words = test_words[:max_samples]

        dataset = get_dataset('data/icub_braille_raw')
        synthesizer = SignalSynthesizer(dataset, 'config/noise.yaml')
        rng = np.random.RandomState(42)

        for word in test_words:
            noisy, clean, _ = synthesizer.synthesize_word(word, rng)
            samples.append({'noisy': noisy, 'clean': clean,
                            'label': word, 'mean': mean, 'std': std})

    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


# ─── Inference helpers ────────────────────────────────────────────────────────

def run_ctc_inference(model, sample, use_denoiser, device):
    if use_denoiser:
        signal = denoise(sample['noisy']).astype(np.float32)
    else:
        signal = sample['noisy'].astype(np.float32)

    mean = sample['mean']
    std  = sample['std']
    signal = (signal - mean) / (std + 1e-8)

    tensor  = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    lengths = torch.tensor([signal.shape[0]], dtype=torch.long)

    with torch.no_grad():
        log_probs, out_lengths = model(tensor.to(device), lengths.to(device))

    return log_probs.squeeze(1).cpu().numpy()


def compute_metrics(predictions, targets):
    wer = jiwer.wer(targets, predictions) * 100
    cer = jiwer.cer(targets, predictions) * 100
    return round(wer, 2), round(cer, 2)


# ─── Main ablation ────────────────────────────────────────────────────────────

def run_ablation():
    H5_PATH    = 'data/corpus.h5'
    NORM_PATH  = 'data/norm_params.json'
    CTC_CKPT   = os.getenv('ABLATION_CTC_CKPT', 'checkpoints/ctc_best.pt').strip()
    BIGRAM_PATH = 'models/bigram_logprobs.npy'
    LM_WEIGHT = float(os.getenv('LM_WEIGHT', '0.3'))
    MAX_SAMPLES = int(os.getenv('ABLATION_MAX_SAMPLES', '0'))
    PROGRESS_EVERY = int(os.getenv('ABLATION_PROGRESS_EVERY', '20'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running ablation on device: {device}\n")

    # Check required files
    for path in [CTC_CKPT, BIGRAM_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run training first.")
            sys.exit(1)

    test_samples = load_test_set(H5_PATH, NORM_PATH, max_samples=MAX_SAMPLES)
    print(f"Test samples: {len(test_samples)}\n")

    model  = load_ctc_model(CTC_CKPT, device=device)
    decoder = BrailleDecoder(CTC_CKPT, BIGRAM_PATH, NORM_PATH, lm_weight=LM_WEIGHT, device=str(device))

    targets = [s['label'] for s in test_samples]

    results = {}

    # ── Condition ② — Word-level, no denoising, greedy ───────────────────────
    print("Running condition ②: Word-level, no denoising...")
    preds2 = []
    for i, s in enumerate(test_samples, start=1):
        logits = run_ctc_inference(model, s, use_denoiser=False, device=device)
        preds2.append(greedy_decode(logits))
        if PROGRESS_EVERY > 0 and i % PROGRESS_EVERY == 0:
            print(f"  cond2 progress: {i}/{len(test_samples)}")
    results['cond2'] = compute_metrics(preds2, targets)
    print(f"  WER={results['cond2'][0]}%  CER={results['cond2'][1]}%")

    # ── Condition ③ — Word-level + denoising, greedy ─────────────────────────
    print("Running condition ③: Word-level + denoising...")
    preds3 = []
    for i, s in enumerate(test_samples, start=1):
        logits = run_ctc_inference(model, s, use_denoiser=True, device=device)
        preds3.append(greedy_decode(logits))
        if PROGRESS_EVERY > 0 and i % PROGRESS_EVERY == 0:
            print(f"  cond3 progress: {i}/{len(test_samples)}")
    results['cond3'] = compute_metrics(preds3, targets)
    print(f"  WER={results['cond3'][0]}%  CER={results['cond3'][1]}%")

    # ── Condition ④ — Full pipeline (denoise + beam search + bigram LM) ──────
    print("Running condition ④: Full pipeline (denoise + beam search + LM)...")
    preds4 = []
    for i, s in enumerate(test_samples, start=1):
        word, _ = decoder.predict(s['noisy'], use_lm=True)
        preds4.append(word)
        if PROGRESS_EVERY > 0 and i % PROGRESS_EVERY == 0:
            print(f"  cond4 progress: {i}/{len(test_samples)}")
    results['cond4'] = compute_metrics(preds4, targets)
    print(f"  WER={results['cond4'][0]}%  CER={results['cond4'][1]}%")

    # ── Condition ① — Letter baseline (fill in from Person A) ────────────────
    # Fill from env vars if available, else leave as TODO in table.
    BASELINE_WER = os.getenv('BASELINE_WER', 'TBD')
    BASELINE_CER = os.getenv('BASELINE_CER', 'TBD')

    # ── Write results table ───────────────────────────────────────────────────
    table = f"""# Ablation Study Results

| Condition | Denoising | Decoder | LM | WER (%) | CER (%) |
|---|---|---|---|---|---|
| ① Letter baseline | ✓ | CNN classifier | ✗ | {BASELINE_WER} | {BASELINE_CER} |
| ② Word, no denoise | ✗ | CNN-LSTM CTC | ✗ | {results['cond2'][0]} | {results['cond2'][1]} |
| ③ Word + denoise | ✓ | CNN-LSTM CTC | ✗ | {results['cond3'][0]} | {results['cond3'][1]} |
| ④ Full pipeline | ✓ | CNN-LSTM CTC | ✓ bigram | {results['cond4'][0]} | {results['cond4'][1]} |

**Notes:**
- WER and CER are percentages (lower is better)
- Condition ① numbers provided by Person A
- LM weight for condition ④: {LM_WEIGHT} (set via LM_WEIGHT env var)
- Test set: {len(test_samples)} samples
"""

    with open('results/ablation.md', 'w', encoding='utf-8') as f:
        f.write(table)

    print("\n" + table)
    print("Saved to results/ablation.md")


if __name__ == '__main__':
    run_ablation()
