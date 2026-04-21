"""
Train the CNN-LSTM + CTC model.

Run from the project root:
    python train/train_ctc.py

Saves best checkpoint to: checkpoints/ctc_best.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import h5py
import json
import numpy as np
import random
import math
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
from tqdm import tqdm

from models.ctc_model import BrailleCTCModel
from models.vocab import char_to_idx, BLANK_IDX
from models.denoiser import denoise


# ─── Dataset ────────────────────────────────────────────────────────────────

class BrailleDataset(Dataset):
    def __init__(self, h5_path, split, norm_params, splits_path,
                 signal_mode='clean', precompute_denoised=False):
        self.h5_path      = h5_path
        self.split        = split
        self.mean         = np.array(norm_params['mean'], dtype=np.float32)
        self.std          = np.array(norm_params['std'],  dtype=np.float32)
        self.signal_mode  = signal_mode
        self.precompute_denoised = precompute_denoised
        self._denoised_cache = {}

        with h5py.File(h5_path, 'r') as f:
            self.keys = sorted(f[split].keys(), key=lambda x: int(x))

        max_words = int(os.getenv('CTC_DEBUG_MAX_WORDS', '0'))
        if max_words <= 0 and os.getenv('CTC_OVERFIT', '0') == '1':
            max_words = 5

        if max_words > 0:
            selected_words = []
            seen_words = set()
            key_labels = {}

            with h5py.File(h5_path, 'r') as f:
                for key in self.keys:
                    label_raw = f[self.split][key].attrs['label']
                    if isinstance(label_raw, bytes):
                        label_str = label_raw.decode('utf-8')
                    else:
                        label_str = str(label_raw)

                    key_labels[key] = label_str
                    if label_str not in seen_words and len(selected_words) < max_words:
                        seen_words.add(label_str)
                        selected_words.append(label_str)

            self.keys = [key for key in self.keys if key_labels[key] in seen_words]
            print(
                f"Using first {len(selected_words)} unique words for {split} "
                f"(CTC_DEBUG_MAX_WORDS)"
            )

        # Optional debug subset for quick overfit/sanity checks.
        max_samples = int(os.getenv('CTC_DEBUG_MAX_SAMPLES', '0'))
        if max_samples > 0:
            self.keys = self.keys[:max_samples]

        if self.precompute_denoised and self.signal_mode in ('dae', 'hybrid'):
            print(f"Precomputing denoised signals for {split} ({len(self.keys)} samples)...")
            with h5py.File(self.h5_path, 'r') as f:
                for key in self.keys:
                    noisy = f[self.split][key]['noisy'][:].astype(np.float32)
                    self._denoised_cache[key] = denoise(noisy).astype(np.float32)

    def _pick_signal(self, key, noisy, clean):
        if self.signal_mode == 'clean':
            return clean.astype(np.float32)
        if self.signal_mode == 'noisy':
            return noisy.astype(np.float32)
        if self.signal_mode == 'dae':
            if key in self._denoised_cache:
                return self._denoised_cache[key]
            den = denoise(noisy.astype(np.float32)).astype(np.float32)
            self._denoised_cache[key] = den
            return den
        if self.signal_mode == 'hybrid':
            # Mix clean and denoised inputs so the model remains robust in both modes.
            if random.random() < 0.5:
                return clean.astype(np.float32)
            if key in self._denoised_cache:
                return self._denoised_cache[key]
            den = denoise(noisy.astype(np.float32)).astype(np.float32)
            self._denoised_cache[key] = den
            return den
        raise ValueError(f"Unsupported signal_mode: {self.signal_mode}")

    def _augment_signal(self, signal):
        if self.split != 'train':
            return signal.astype(np.float32)

        if os.getenv('CTC_ENABLE_AUGMENTATION', '1') != '1':
            return signal.astype(np.float32)

        augmented = signal.astype(np.float32).copy()

        if os.getenv('CTC_AUG_TIME_SHIFT', '1') == '1':
            max_shift = int(os.getenv('CTC_AUG_MAX_SHIFT', '4'))
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                prefix = np.repeat(augmented[:1], shift, axis=0)
                augmented = np.concatenate([prefix, augmented[:-shift]], axis=0)
            elif shift < 0:
                shift_abs = abs(shift)
                suffix = np.repeat(augmented[-1:], shift_abs, axis=0)
                augmented = np.concatenate([augmented[shift_abs:], suffix], axis=0)

        if os.getenv('CTC_AUG_SCALE', '1') == '1':
            global_scale = np.random.uniform(0.90, 1.10)
            channel_scale = np.random.uniform(0.96, 1.04, size=(1, augmented.shape[1])).astype(np.float32)
            augmented = augmented * global_scale * channel_scale

        if os.getenv('CTC_AUG_NOISE', '1') == '1':
            noise_std = float(os.getenv('CTC_AUG_NOISE_STD', '0.03'))
            local_scale = np.std(augmented, axis=0, keepdims=True).astype(np.float32) + 1e-4
            noise = np.random.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
            augmented = augmented + noise * local_scale

        if os.getenv('CTC_AUG_CHANNEL_DROP', '1') == '1' and random.random() < 0.20:
            drop_channel = random.randrange(augmented.shape[1])
            augmented[:, drop_channel] = 0.0

        if os.getenv('CTC_AUG_TIME_MASK', '1') == '1':
            mask_width = int(os.getenv('CTC_AUG_TIME_MASK_WIDTH', '3'))
            if augmented.shape[0] > mask_width + 1 and random.random() < 0.30:
                start = random.randint(0, augmented.shape[0] - mask_width)
                augmented[start:start + mask_width, :] *= np.random.uniform(0.0, 0.25)

        if os.getenv('CTC_AUG_CHANNEL_NOISE', '1') == '1':
            channel_noise = np.random.normal(0.0, 0.01, size=(1, augmented.shape[1])).astype(np.float32)
            augmented = augmented + channel_noise

        return augmented.astype(np.float32)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with h5py.File(self.h5_path, 'r') as f:
            noisy = f[self.split][key]['noisy'][:]
            clean = f[self.split][key]['clean'][:]
            # Use canonical label stored with each sample to avoid split-order mismatch.
            label_raw = f[self.split][key].attrs['label']

        if isinstance(label_raw, bytes):
            label_str = label_raw.decode('utf-8')
        else:
            label_str = str(label_raw)

        signal = self._pick_signal(key, noisy, clean)

        signal = (signal - self.mean) / (self.std + 1e-8)
        signal = self._augment_signal(signal)

        label_indices = [char_to_idx[c] for c in label_str if c in char_to_idx]

        return (
            torch.tensor(signal, dtype=torch.float32),
            torch.tensor(label_indices, dtype=torch.long),
        )
 
def collate_fn(batch):
    """
    Pad signals to the same length within a batch.
    CTC needs actual lengths to ignore padding.
    """
    signals, labels = zip(*batch)

    signal_lengths = torch.tensor([s.shape[0] for s in signals], dtype=torch.long)
    label_lengths  = torch.tensor([len(l) for l in labels],       dtype=torch.long)

    # Pad signals along time axis
    signals_padded = nn.utils.rnn.pad_sequence(signals, batch_first=True)  # (B, T_max, 12)

    # Concatenate labels (CTCLoss expects a flat label tensor)
    labels_cat = torch.cat(labels)

    return signals_padded, labels_cat, signal_lengths, label_lengths


# ─── Greedy decoder (for val CER during training) ───────────────────────────

def greedy_decode(log_probs_batch, output_lengths):
    """
    Quick greedy decode for monitoring CER during training.
    log_probs_batch: (T, B, 28)
    Returns list of predicted strings.
    """
    from models.vocab import vocab
    predictions = []
    # (T, B, 28) → (B, T, 28)
    log_probs_batch = log_probs_batch.permute(1, 0, 2)

    for i, log_probs in enumerate(log_probs_batch):
        length = output_lengths[i].item()
        indices = log_probs[:length].argmax(dim=-1).tolist()

        # Collapse repeats
        collapsed = [indices[0]] if indices else []
        for j in range(1, len(indices)):
            if indices[j] != indices[j-1]:
                collapsed.append(indices[j])

        # Remove blank, map to chars
        chars = [vocab[k] for k in collapsed if k != BLANK_IDX]
        predictions.append(''.join(chars))

    return predictions


def compute_cer(predictions, targets):
    """Simple character error rate."""
    import jiwer
    if not predictions:
        return 1.0
    # jiwer CER treats each string as a sequence of characters
    try:
        return jiwer.cer(targets, predictions)
    except Exception:
        return 1.0


# ─── Training loop ──────────────────────────────────────────────────────────

def train():
    # ── Config ──
    H5_PATH    = 'data/corpus.h5'
    NORM_PATH  = 'data/norm_params.json'
    CKPT_PATH  = os.getenv('CTC_CHECKPOINT_PATH', 'checkpoints/ctc_best.pt').strip()
    BATCH_SIZE = int(os.getenv('CTC_BATCH_SIZE', '16'))
    EPOCHS     = int(os.getenv('CTC_EPOCHS', '80'))
    LR         = float(os.getenv('CTC_LR', '1e-4'))
    use_tqdm   = os.getenv('CTC_NO_TQDM', '0') != '1'
    TRAIN_SIGNAL_MODE = os.getenv('CTC_TRAIN_SIGNAL_MODE', 'hybrid').strip().lower()
    VAL_SIGNAL_MODE   = os.getenv('CTC_VAL_SIGNAL_MODE', 'dae').strip().lower()
    PRECOMPUTE_DENOISED = os.getenv('CTC_PRECOMPUTE_DENOISED', '0') == '1'
    SCHEDULER_NAME = os.getenv('CTC_SCHEDULER', 'onecycle').strip().lower()
    BLANK_PENALTY_WEIGHT = float(os.getenv('CTC_BLANK_PENALTY_WEIGHT', '0.1'))
    BLANK_PENALTY_THRESHOLD = float(os.getenv('CTC_BLANK_PENALTY_THRESHOLD', '0.8'))
    BLANK_TARGET_RATIO = float(os.getenv('CTC_BLANK_TARGET_RATIO', '0.55'))
    BLANK_PENALTY_RAMP_EPOCHS = max(1, int(os.getenv('CTC_BLANK_PENALTY_RAMP_EPOCHS', '8')))
    CHECKPOINT_EVERY = int(os.getenv('CTC_CHECKPOINT_EVERY', '20'))
    GRAD_ACCUM_STEPS = max(1, int(os.getenv('CTC_GRAD_ACCUM_STEPS', '1')))
    MAX_BATCHES_PER_EPOCH = int(os.getenv('CTC_MAX_BATCHES_PER_EPOCH', '0'))
    MAX_VAL_BATCHES = int(os.getenv('CTC_MAX_VAL_BATCHES', '0'))
    MAX_TRAIN_SECONDS = float(os.getenv('CTC_MAX_TRAIN_SECONDS', '0'))
    USE_LORA = os.getenv('CTC_USE_LORA', '0') == '1'
    LORA_RANK = int(os.getenv('CTC_LORA_RANK', '8'))
    LORA_ALPHA = float(os.getenv('CTC_LORA_ALPHA', '16'))
    LORA_DROPOUT = float(os.getenv('CTC_LORA_DROPOUT', '0.05'))
    FREEZE_BASE = os.getenv('CTC_LORA_FREEZE_BASE', '1') == '1'
    INIT_CHECKPOINT = os.getenv('CTC_INIT_CHECKPOINT', '').strip()
    PATIENCE_ES   = int(os.getenv('CTC_EARLY_STOP_PATIENCE', '10'))
    MIN_EPOCHS_BEFORE_ES = int(os.getenv('CTC_MIN_EPOCHS_BEFORE_EARLY_STOP', '0'))
    PATIENCE_LR   = 5    # LR scheduler patience
    GRAD_CLIP     = 5.0

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(os.path.dirname(CKPT_PATH) or 'checkpoints', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──
    norm = json.load(open(NORM_PATH))

    train_ds = BrailleDataset(
        H5_PATH,
        'train',
        norm,
        splits_path='data/splits.json',
        signal_mode=os.getenv('CTC_TRAIN_SIGNAL_MODE', 'dae').strip().lower(),
        precompute_denoised=PRECOMPUTE_DENOISED,
    )
    val_ds = BrailleDataset(
        H5_PATH,
        'val',
        norm,
        splits_path='data/splits.json',
        signal_mode=VAL_SIGNAL_MODE,
        precompute_denoised=PRECOMPUTE_DENOISED,
    )

    if os.getenv('CTC_OVERFIT', '0') == '1':
        # Sanity mode: train/val are identical, tiny subset.
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"Signal modes -> train: {os.getenv('CTC_TRAIN_SIGNAL_MODE', 'dae').strip().lower()} | val: {VAL_SIGNAL_MODE}")

    # ── Model ──
    model     = BrailleCTCModel(use_lora=USE_LORA, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT).to(device)
    if INIT_CHECKPOINT:
        state_dict = torch.load(INIT_CHECKPOINT, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded init checkpoint from {INIT_CHECKPOINT}")
        if missing:
            print(f"  missing keys: {len(missing)}")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)}")
    if USE_LORA and FREEZE_BASE:
        model.freeze_base_model()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, trainable={trainable}/{total}")
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=1e-4)
    scheduler = None

    effective_train_batches = len(train_loader) if MAX_BATCHES_PER_EPOCH <= 0 else min(MAX_BATCHES_PER_EPOCH, len(train_loader))
    optimizer_steps_per_epoch = max(1, math.ceil(effective_train_batches / GRAD_ACCUM_STEPS))

    if SCHEDULER_NAME == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR,
            epochs=EPOCHS,
            steps_per_epoch=optimizer_steps_per_epoch,
            pct_start=float(os.getenv('CTC_ONECYCLE_WARMUP', '0.25')),
            div_factor=float(os.getenv('CTC_ONECYCLE_DIV_FACTOR', '10')),
            final_div_factor=float(os.getenv('CTC_ONECYCLE_FINAL_DIV_FACTOR', '200')),
            anneal_strategy='cos',
        )
    elif SCHEDULER_NAME == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=LR * float(os.getenv('CTC_COSINE_MIN_FACTOR', '0.02')),
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE_LR,
                              factor=0.5)

    print(f"Scheduler: {SCHEDULER_NAME}")
    print(f"Batch controls -> grad_accum={GRAD_ACCUM_STEPS}, max_train_batches={MAX_BATCHES_PER_EPOCH}, max_val_batches={MAX_VAL_BATCHES}")
    if MAX_TRAIN_SECONDS > 0:
        print(f"Time limit -> max_train_seconds={MAX_TRAIN_SECONDS:.1f}")

    # ── Training ──
    best_val_cer  = float('inf')
    epochs_no_imp = 0
    train_start_time = time.monotonic()
    stop_requested = False

    def _time_limit_reached():
        if MAX_TRAIN_SECONDS <= 0:
            return False
        return (time.monotonic() - train_start_time) >= MAX_TRAIN_SECONDS

    for epoch in range(1, EPOCHS + 1):
        if _time_limit_reached():
            print(f"Reached training time limit before epoch {epoch} start.")
            break

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_blank_count = 0
        train_token_count = 0
        blank_penalty_batches = 0
        train_blank_prob_sum = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False) if use_tqdm else train_loader
        optimizer.zero_grad()
        processed_train_batches = 0

        for batch_idx, (signals, labels, sig_lengths, lbl_lengths) in enumerate(train_iter):
            if _time_limit_reached():
                stop_requested = True
                break
            if MAX_BATCHES_PER_EPOCH > 0 and batch_idx >= MAX_BATCHES_PER_EPOCH:
                break

            signals    = signals.to(device)
            labels     = labels.to(device)
            sig_lengths = sig_lengths.to(device)

            log_probs, out_lengths = model(signals, sig_lengths)

            loss = ctc_loss(log_probs, labels, out_lengths, lbl_lengths)

            # Probability-aware blank regularization is more stable than argmax-only checks.
            blank_probs = log_probs.exp()[:, :, BLANK_IDX]
            batch_blank_prob = blank_probs.mean()
            ramp = min(1.0, epoch / BLANK_PENALTY_RAMP_EPOCHS)
            adaptive_blank_weight = BLANK_PENALTY_WEIGHT * ramp
            blank_excess = torch.relu(batch_blank_prob - BLANK_TARGET_RATIO)
            if adaptive_blank_weight > 0:
                loss = loss + adaptive_blank_weight * blank_excess

            argmax_tokens = log_probs.argmax(dim=-1)
            batch_blank_frac = (argmax_tokens == BLANK_IDX).float().mean()
            if batch_blank_frac.item() > BLANK_PENALTY_THRESHOLD:
                loss = loss + BLANK_PENALTY_WEIGHT * batch_blank_frac
                blank_penalty_batches += 1

            train_blank_prob_sum += batch_blank_prob.item()

            train_blank_count += (argmax_tokens == BLANK_IDX).sum().item()
            train_token_count += argmax_tokens.numel()

            (loss / GRAD_ACCUM_STEPS).backward()
            should_step = ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0)
            is_last_batch = (MAX_BATCHES_PER_EPOCH > 0 and batch_idx + 1 >= MAX_BATCHES_PER_EPOCH) or (batch_idx + 1 >= len(train_loader))

            if should_step or is_last_batch:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

                if SCHEDULER_NAME == 'onecycle':
                    scheduler.step()

            train_loss += loss.item()
            processed_train_batches += 1

        if processed_train_batches == 0:
            if _time_limit_reached():
                elapsed = time.monotonic() - train_start_time
                print(f"Reached time limit during train loop ({elapsed:.1f}s elapsed).")
            else:
                print("No train batches processed in this epoch (batching caps). Stopping.")
            break

        train_loss /= max(1, processed_train_batches)
        train_blank_ratio = (train_blank_count / max(1, train_token_count))
        train_blank_prob = train_blank_prob_sum / max(1, processed_train_batches)

        # ── Validate ──
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_tgts  = []
        blank_count = 0
        token_count = 0

        with torch.no_grad():
            processed_val_batches = 0
            for val_batch_idx, (signals, labels, sig_lengths, lbl_lengths) in enumerate(val_loader):
                if _time_limit_reached():
                    stop_requested = True
                    break
                if MAX_VAL_BATCHES > 0 and val_batch_idx >= MAX_VAL_BATCHES:
                    break

                signals    = signals.to(device)
                sig_lengths = sig_lengths.to(device)

                log_probs, out_lengths = model(signals, sig_lengths)

                # Reconstruct individual label strings for CER
                offset = 0
                for l in lbl_lengths:
                    from models.vocab import vocab
                    tgt_chars = [vocab[i] for i in labels[offset:offset+l].tolist()]
                    all_tgts.append(''.join(tgt_chars))
                    offset += l

                preds = greedy_decode(log_probs, out_lengths)
                all_preds.extend(preds)

                # Monitor blank-collapse tendency.
                argmax_tokens = log_probs.argmax(dim=-1)  # (T, B)
                blank_count += (argmax_tokens == BLANK_IDX).sum().item()
                token_count += argmax_tokens.numel()

                # Val loss
                loss = ctc_loss(log_probs, labels.to(device), out_lengths, lbl_lengths)
                val_loss += loss.item()
                processed_val_batches += 1

        if processed_val_batches == 0:
            if _time_limit_reached():
                elapsed = time.monotonic() - train_start_time
                print(f"Reached time limit during validation loop ({elapsed:.1f}s elapsed).")
            else:
                print("No validation batches processed in this epoch (batching caps). Stopping.")
            break

        val_loss /= max(1, processed_val_batches)
        val_cer   = compute_cer(all_preds, all_tgts)
        blank_ratio = (blank_count / max(1, token_count))

        if val_cer > 0.0 and blank_ratio > 0.85:
            print("  [warn] blank collapse risk detected; consider lowering LR or increasing augmentation")

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"train_blank_ratio={train_blank_ratio:.3f} | "
              f"train_blank_prob={train_blank_prob:.3f} | "
              f"val_loss={val_loss:.4f} | val_CER={val_cer:.4f} | blank_ratio={blank_ratio:.3f} | "
              f"blank_penalty_batches={blank_penalty_batches}")

        if SCHEDULER_NAME == 'plateau':
            scheduler.step(val_cer)
        elif SCHEDULER_NAME == 'cosine':
            scheduler.step()

        # ── Checkpoint ──
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            epochs_no_imp = 0
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  [OK] New best val CER: {best_val_cer:.4f} - saved to {CKPT_PATH}")
        else:
            epochs_no_imp += 1
            if epoch >= MIN_EPOCHS_BEFORE_ES and epochs_no_imp >= PATIENCE_ES:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE_ES} epochs)")
                break

        if CHECKPOINT_EVERY > 0 and epoch % CHECKPOINT_EVERY == 0:
            snapshot_path = os.path.join('checkpoints', f'ctc_epoch_{epoch:03d}.pt')
            torch.save(model.state_dict(), snapshot_path)
            print(f"  [OK] Periodic checkpoint saved to {snapshot_path}")

        if stop_requested:
            elapsed = time.monotonic() - train_start_time
            print(f"Stopping after epoch {epoch} due to time limit ({elapsed:.1f}s elapsed).")
            break

    print(f"\nTraining complete. Best val CER: {best_val_cer:.4f}")
    print(f"Checkpoint saved at: {CKPT_PATH}")


if __name__ == '__main__':
    train()
