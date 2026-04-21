"""
Full Inference Pipeline — BrailleDecoder

Usage:
    from models.pipeline import BrailleDecoder
    decoder = BrailleDecoder('checkpoints/ctc_best.pt',
                              'models/bigram_logprobs.npy',
                              'data/norm_params.json')
    word, candidates = decoder.predict(noisy_signal)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json

from models.denoiser import denoise
from models.vocab import vocab, char_to_idx, BLANK_IDX
from models.ctc_model import load_ctc_model


def _edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


# ─── Greedy decoder ──────────────────────────────────────────────────────────

def greedy_decode(logits):
    """
    Args:
        logits: (T, 28) numpy array — raw model output (pre-softmax)
    Returns:
        string
    """
    indices = logits.argmax(axis=-1)          # (T,)

    # Collapse consecutive repeats
    collapsed = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1]:
            collapsed.append(indices[i])

    # Remove blank, map to characters
    chars = [vocab[i] for i in collapsed if i != BLANK_IDX]
    return ''.join(chars)


# ─── Beam search + bigram LM decoder ─────────────────────────────────────────

def beam_search_decode(
    logits,
    bigram_logprobs,
    beam_width=10,
    lm_weight=0.3,
    length_penalty=0.05,
    max_output_len=None,
    frame_logp_cutoff=6.0,
    two_cycle_penalty=1.0,
):
    """
    Args:
        logits:          (T, 28) numpy array — raw model output
        bigram_logprobs: (28, 28) numpy array — log P(c_t | c_{t-1})
        beam_width:      number of candidate sequences to keep
        lm_weight:       how much weight to give the LM vs CTC (tune on val set)

    Returns:
        top_word:   best predicted string
        top3:       list of (sequence, score) for top 3 candidates
    """
    def log_add(a, b):
        if a == -np.inf:
            return b
        if b == -np.inf:
            return a
        m = a if a > b else b
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    probs_sum = torch.exp(logits_tensor).sum(dim=-1).mean().item()
    looks_like_log_probs = bool(torch.max(logits_tensor).item() <= 1e-4 and abs(probs_sum - 1.0) < 0.25)
    if looks_like_log_probs:
        log_probs = logits_tensor.numpy()
    else:
        log_probs = torch.log_softmax(logits_tensor, dim=-1).numpy()  # (T, 28)

    if max_output_len is None:
        max_output_len = min(log_probs.shape[0], 24)

    # Prefix beam state: prefix tuple -> (p_blank, p_nonblank)
    beams = {(): (0.0, -np.inf)}

    top_k = min(log_probs.shape[-1], max(beam_width * 2, beam_width))

    for t in range(log_probs.shape[0]):
        next_beams = {}

        # Prune candidates at each frame for speed/noise control.
        frame_best = np.max(log_probs[t])
        top_chars = [
            idx for idx in np.argsort(log_probs[t])[-top_k:]
            if log_probs[t, idx] >= frame_best - frame_logp_cutoff
        ]

        for prefix, (p_b, p_nb) in beams.items():
            p_total = log_add(p_b, p_nb)

            # Extend with blank.
            nb_p_b, nb_p_nb = next_beams.get(prefix, (-np.inf, -np.inf))
            nb_p_b = log_add(nb_p_b, p_total + log_probs[t, BLANK_IDX])
            next_beams[prefix] = (nb_p_b, nb_p_nb)

            last = prefix[-1] if prefix else None
            prev_for_lm = last if last is not None else char_to_idx[' ']

            for c_idx in top_chars:
                if c_idx == BLANK_IDX:
                    continue
                if len(prefix) >= max_output_len:
                    continue

                p = log_probs[t, c_idx]
                lm = lm_weight * bigram_logprobs[prev_for_lm, c_idx]
                cycle_penalty = 0.0
                if len(prefix) >= 2 and c_idx == prefix[-2] and prefix[-1] != c_idx:
                    cycle_penalty = two_cycle_penalty

                if c_idx == last:
                    # Same char can continue only from blank-ending prefix.
                    new_p_nb = p_b + p + lm - cycle_penalty
                    nb_p_b, nb_p_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    nb_p_nb = log_add(nb_p_nb, new_p_nb)
                    next_beams[prefix] = (nb_p_b, nb_p_nb)
                else:
                    new_prefix = prefix + (c_idx,)
                    new_p_nb = p_total + p + lm - cycle_penalty
                    nb_p_b, nb_p_nb = next_beams.get(new_prefix, (-np.inf, -np.inf))
                    nb_p_nb = log_add(nb_p_nb, new_p_nb)
                    next_beams[new_prefix] = (nb_p_b, nb_p_nb)

        scored = []
        for prefix, (p_b, p_nb) in next_beams.items():
            total_score = log_add(p_b, p_nb)
            length_norm = max(len(prefix), 1) ** length_penalty
            scored.append((prefix, p_b, p_nb, total_score / length_norm))
        scored.sort(key=lambda x: x[3], reverse=True)
        beams = {prefix: (p_b, p_nb) for prefix, p_b, p_nb, _ in scored[:beam_width]}

    if not beams:
        return '', []

    ranked = []
    for prefix, (p_b, p_nb) in beams.items():
        score = log_add(p_b, p_nb)
        word = ''.join(vocab[i] for i in prefix[:max_output_len])
        ranked.append((word, score))
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_word = ranked[0][0]
    return top_word, ranked[:3]


# ─── End-to-end pipeline class ────────────────────────────────────────────────

class BrailleDecoder:
    def __init__(self, ctc_ckpt, bigram_path, norm_path,
                 lm_weight=0.3, beam_width=10, length_penalty=0.05, device='cpu'):
        """
        Args:
            ctc_ckpt:    path to checkpoints/ctc_best.pt
            bigram_path: path to models/bigram_logprobs.npy
            norm_path:   path to data/norm_params.json
            lm_weight:   LM weight for beam search (tuned on val set)
            device:      'cpu' or 'cuda'
        """
        self.device    = device
        self.lm_weight = lm_weight
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.max_output_len = int(os.getenv('DECODER_MAX_OUTPUT_LEN', '24'))
        self.frame_logp_cutoff = float(os.getenv('DECODER_FRAME_LOGP_CUTOFF', '6.0'))
        self.two_cycle_penalty = float(os.getenv('DECODER_TWO_CYCLE_PENALTY', '1.0'))
        self.use_lexicon_constraint = os.getenv('DECODER_LEXICON_CONSTRAIN', '1') == '1'
        self.lexicon_path = os.getenv('DECODER_LEXICON_PATH', 'data/wordlist.txt').strip()
        self.lexicon_words = []

        # Load CTC model
        self.model = load_ctc_model(ctc_ckpt, device=device)
        self.model.eval()

        # Load bigram LM
        self.bigram = np.load(bigram_path)

        # Load normalisation params
        norm = json.load(open(norm_path))
        self.mean = np.array(norm['mean'], dtype=np.float32)
        self.std  = np.array(norm['std'],  dtype=np.float32)

        if self.use_lexicon_constraint and os.path.exists(self.lexicon_path):
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                self.lexicon_words = [line.strip() for line in f if line.strip()]

    def _snap_to_lexicon(self, word):
        if not self.lexicon_words or not word:
            return word

        best = None
        best_key = None
        for w in self.lexicon_words:
            dist = _edit_distance(word, w)
            key = (dist, abs(len(word) - len(w)))
            if best_key is None or key < best_key:
                best_key = key
                best = w
                if dist == 0:
                    break
        return best if best is not None else word

    def _normalise(self, signal):
        return (signal - self.mean) / (self.std + 1e-8)

    def predict_from_clean(self, clean_signal, use_lm=True):
        """
        Decode a pre-denoised signal.

        Args:
            clean_signal: (T, 12) numpy float32
            use_lm:       True = beam search + bigram, False = greedy

        Returns:
            word:       predicted string
            candidates: list of (string, score) top-3 candidates
                        (for greedy, returns [(word, 0.0)])
        """
        # 1. Normalise
        normed = self._normalise(clean_signal)

        # 2. Forward pass through CNN-LSTM
        tensor = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)  # (1, T, 12)
        lengths = torch.tensor([normed.shape[0]], dtype=torch.long)

        with torch.inference_mode():
            log_probs, out_lengths = self.model(tensor.to(self.device), lengths.to(self.device))

        # log_probs: (T/2, 1, 28) → (T/2, 28) numpy
        logits = log_probs.squeeze(1).cpu().numpy()

        # 3. Decode
        if use_lm:
            word, candidates = beam_search_decode(
                logits,
                self.bigram,
                beam_width=self.beam_width,
                lm_weight=self.lm_weight,
                length_penalty=self.length_penalty,
                max_output_len=self.max_output_len,
                frame_logp_cutoff=self.frame_logp_cutoff,
                two_cycle_penalty=self.two_cycle_penalty,
            )
            if self.use_lexicon_constraint:
                snapped = self._snap_to_lexicon(word)
                if snapped:
                    word = snapped
        else:
            word       = greedy_decode(logits)
            candidates = [(word, 0.0)]

        return word, candidates

    def predict(self, noisy_signal, use_lm=True):
        """Full pipeline: denoise -> normalize -> CNN-LSTM -> decode."""
        clean = denoise(noisy_signal)
        return self.predict_from_clean(clean, use_lm=use_lm)


# ─── LM weight tuning helper ──────────────────────────────────────────────────

def tune_lm_weight(decoder, val_h5_path, norm_params):
    """
    Sweep lm_weight over [0.1, 0.2, 0.3, 0.5, 0.7] on the val set.
    Prints CER for each weight. Pick the lowest.

    Run once after training is done:
        from models.pipeline import tune_lm_weight, BrailleDecoder
        decoder = BrailleDecoder(...)
        tune_lm_weight(decoder, 'data/corpus.h5', json.load(open('data/norm_params.json')))
    """
    import h5py
    import jiwer

    mean = np.array(norm_params['mean'], dtype=np.float32)
    std  = np.array(norm_params['std'],  dtype=np.float32)

    with h5py.File(val_h5_path, 'r') as f:
        keys = list(f['val'].keys())
        samples = []
        for k in keys:
            noisy = f['val'][k]['noisy'][:]
            label = f['val'][k]['label'][()]
            if isinstance(label, bytes):
                label = label.decode()
            samples.append((noisy, label))

    print("Tuning LM weight on val set...")
    for lw in [0.1, 0.2, 0.3, 0.5, 0.7]:
        decoder.lm_weight = lw
        preds, tgts = [], []
        for noisy, label in samples:
            word, _ = decoder.predict(noisy, use_lm=True)
            preds.append(word)
            tgts.append(label)
        cer = jiwer.cer(tgts, preds)
        print(f"  lm_weight={lw:.1f}  →  val CER = {cer:.4f}")

    print("Set decoder.lm_weight to whichever gave the lowest CER.")
