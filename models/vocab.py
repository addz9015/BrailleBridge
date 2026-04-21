"""
Shared Vocabulary
Import this everywhere — decoder, beam search, bigram LM, training.
"""

# 28 tokens total:
#   index 0       = <blank>  (CTC blank token)
#   index 1-26    = a-z
#   index 27      = space
vocab = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz') + [' ']

char_to_idx = {c: i for i, c in enumerate(vocab)}

VOCAB_SIZE = len(vocab)   # 28
BLANK_IDX  = 0
SPACE_IDX  = 27
