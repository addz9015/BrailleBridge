"""
Train the character-level bigram language model.

Run from project root AFTER train_ctc.py:
    python train/train_bigram.py

Saves: models/bigram_logprobs.npy
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from models.vocab import char_to_idx, VOCAB_SIZE


def train_bigram(word_list):
    """
    Build a 28x28 log-probability transition matrix.
    counts[i, j] = how often character j follows character i.
    Laplace smoothing prevents zero probabilities.
    """
    counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)

    for word in word_list:
        word = word.strip().lower()
        if not word:
            continue
        # Pad with space so the model learns word start/end transitions
        padded = ' ' + word + ' '
        for k in range(len(padded) - 1):
            c1 = char_to_idx.get(padded[k])
            c2 = char_to_idx.get(padded[k+1])
            if c1 is not None and c2 is not None:
                counts[c1, c2] += 1

    # Laplace smoothing (+1 to every cell)
    counts += 1.0

    # Normalise each row to get probabilities, then take log
    row_sums = counts.sum(axis=1, keepdims=True)
    probs    = counts / row_sums
    log_probs = np.log(probs)

    return log_probs


if __name__ == '__main__':
    WORDLIST_PATH = 'data/wordlist.txt'
    OUT_PATH      = 'models/bigram_logprobs.npy'

    # Load word list
    with open(WORDLIST_PATH, 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    print(f"Training bigram on {len(words)} words...")

    # Optionally add common English words for a richer LM
    # (the project word list alone is fine for this task)
    try:
        import nltk
        nltk.download('brown', quiet=True)
        from nltk.corpus import brown
        extra = [w.lower() for w in brown.words() if w.isalpha()]
        words = words + extra
        print(f"  + {len(extra)} words from NLTK Brown corpus")
    except Exception:
        print("  (NLTK not available — using project word list only, that's fine)")

    log_probs = train_bigram(words)

    np.save(OUT_PATH, log_probs)
    print(f"Bigram log-prob matrix saved to {OUT_PATH}")
    print(f"Matrix shape: {log_probs.shape}  (should be 28x28)")
