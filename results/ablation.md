# Ablation Study Results

| Condition | Denoising | Decoder | LM | WER (%) | CER (%) |
|---|---|---|---|---|---|
| ① Letter baseline | ✓ | CNN classifier | ✗ | TBD | TBD |
| ② Word, no denoise | ✗ | CNN-LSTM CTC | ✗ | 100.0 | 97.74 |
| ③ Word + denoise | ✓ | CNN-LSTM CTC | ✗ | 90.0 | 28.39 |
| ④ Full pipeline | ✓ | CNN-LSTM CTC | ✓ bigram | 40.0 | 24.84 |

**Notes:**
- WER and CER are percentages (lower is better)
- Condition ① numbers provided by Person A
- LM weight for condition ④: 0.0 (set via LM_WEIGHT env var)
- Test set: 50 samples
