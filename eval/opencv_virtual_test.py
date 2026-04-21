"""
OpenCV Virtual Braille Testing Module

Renders a Braille word image, simulates a sliding finger scan to produce
synthetic 12-taxel signals, and feeds them into the existing decoder pipeline.

Run from project root:
    python eval/opencv_virtual_test.py --word cat

Optional visual scan animation:
    python eval/opencv_virtual_test.py --word hello --visualize
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline import BrailleDecoder


# Standard Braille dot patterns for a-z (6-dot cell: 1..6)
BRAILLE_PATTERNS: Dict[str, List[int]] = {
    'a': [1, 0, 0, 0, 0, 0],
    'b': [1, 1, 0, 0, 0, 0],
    'c': [1, 0, 0, 1, 0, 0],
    'd': [1, 0, 0, 1, 1, 0],
    'e': [1, 0, 0, 0, 1, 0],
    'f': [1, 1, 0, 1, 0, 0],
    'g': [1, 1, 0, 1, 1, 0],
    'h': [1, 1, 0, 0, 1, 0],
    'i': [0, 1, 0, 1, 0, 0],
    'j': [0, 1, 0, 1, 1, 0],
    'k': [1, 0, 1, 0, 0, 0],
    'l': [1, 1, 1, 0, 0, 0],
    'm': [1, 0, 1, 1, 0, 0],
    'n': [1, 0, 1, 1, 1, 0],
    'o': [1, 0, 1, 0, 1, 0],
    'p': [1, 1, 1, 1, 0, 0],
    'q': [1, 1, 1, 1, 1, 0],
    'r': [1, 1, 1, 0, 1, 0],
    's': [0, 1, 1, 1, 0, 0],
    't': [0, 1, 1, 1, 1, 0],
    'u': [1, 0, 1, 0, 0, 1],
    'v': [1, 1, 1, 0, 0, 1],
    'w': [0, 1, 0, 1, 1, 1],
    'x': [1, 0, 1, 1, 0, 1],
    'y': [1, 0, 1, 1, 1, 1],
    'z': [1, 0, 1, 0, 1, 1],
    ' ': [0, 0, 0, 0, 0, 0],
}

# Dot coordinates in a 3x2 Braille cell for dots 1..6
DOT_POSITIONS: List[Tuple[int, int]] = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def _sanitize_word(word: str) -> str:
    cleaned = ''.join(ch for ch in word.lower() if ch in BRAILLE_PATTERNS)
    return cleaned.strip()


def render_braille_word(word: str, dot_size: int = 24, cell_padding: int = 16) -> np.ndarray:
    word = _sanitize_word(word)
    if not word:
        raise ValueError("Input word has no valid Braille characters (a-z or space).")

    rows, cols = 3, 2
    cell_w = cols * dot_size + cell_padding
    cell_h = rows * dot_size + cell_padding

    img_w = cell_w * len(word) + cell_padding
    img_h = cell_h + (cell_padding * 2)
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    for char_idx, char in enumerate(word):
        pattern = BRAILLE_PATTERNS.get(char, [0] * 6)
        x_offset = char_idx * cell_w + cell_padding

        for dot_idx, active in enumerate(pattern):
            row, col = DOT_POSITIONS[dot_idx]
            cx = x_offset + col * dot_size + dot_size // 2
            cy = cell_padding + row * dot_size + dot_size // 2

            color = (30, 30, 30) if active else (210, 210, 210)
            cv2.circle(img, (cx, cy), dot_size // 3, color, -1)

    return img


def simulate_finger_scan(img: np.ndarray, window_width: int = 40, step: int = 2) -> np.ndarray:
    signal_frames: List[List[float]] = []
    img_h, img_w = img.shape[:2]

    if window_width >= img_w:
        raise ValueError(f"window_width ({window_width}) must be smaller than image width ({img_w}).")

    for x in range(0, img_w - window_width, step):
        window = img[:, x:x + window_width]

        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY_INV)

        # 12 taxels: 3 rows x 4 cols
        taxel_readings = []
        region_h = img_h // 3
        region_w = window_width // 4

        for r in range(3):
            for c in range(4):
                region = binary[r * region_h:(r + 1) * region_h, c * region_w:(c + 1) * region_w]
                taxel_readings.append(float(region.mean()))

        signal_frames.append(taxel_readings)

    return np.array(signal_frames, dtype=np.float32)


def visualise_scan(img: np.ndarray, window_width: int = 40, step: int = 2, delay: int = 20) -> None:
    img_h, img_w = img.shape[:2]

    for x in range(0, img_w - window_width, step):
        frame = img.copy()
        cv2.rectangle(frame, (x, 0), (x + window_width, img_h), (255, 100, 0), 2)
        cv2.imshow('Braille Finger Scan', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def test_word_virtually(
    word: str,
    decoder: BrailleDecoder,
    window_width: int = 40,
    step: int = 2,
    visualize: bool = False,
    use_lm: bool = True,
    use_decoder_denoiser: bool = False,
) -> Tuple[str, List[Tuple[str, float]]]:
    img = render_braille_word(word)

    if visualize:
        visualise_scan(img, window_width=window_width, step=step)

    synthetic_signal = simulate_finger_scan(img, window_width=window_width, step=step)

    if use_decoder_denoiser:
        predicted_word, candidates = decoder.predict(synthetic_signal, use_lm=use_lm)
    else:
        predicted_word, candidates = decoder.predict_from_clean(synthetic_signal, use_lm=use_lm)

    print(f"Input word:     {word}")
    print(f"Signal shape:   {synthetic_signal.shape}")
    print(f"Predicted word: {predicted_word}")
    print(f"Top 3:          {candidates}")

    return predicted_word, candidates


def build_decoder(device: str = 'cpu', lm_weight: float = 0.0) -> BrailleDecoder:
    return BrailleDecoder(
        ctc_ckpt=os.getenv('VIRTUAL_TEST_CTC_CKPT', 'checkpoints/ctc_full_gpu_safe.pt'),
        bigram_path=os.getenv('VIRTUAL_TEST_BIGRAM_PATH', 'models/bigram_logprobs.npy'),
        norm_path=os.getenv('VIRTUAL_TEST_NORM_PATH', 'data/norm_params.json'),
        lm_weight=lm_weight,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV virtual Braille word tester")
    parser.add_argument('--word', type=str, required=True, help='Word to render and test')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Inference device')
    parser.add_argument('--lm-weight', type=float, default=0.0, help='LM weight for decoder')
    parser.add_argument('--window-width', type=int, default=40, help='Sliding window width')
    parser.add_argument('--step', type=int, default=2, help='Sliding step size')
    parser.add_argument('--visualize', action='store_true', help='Show OpenCV sliding scan window')
    parser.add_argument('--use-lm', action='store_true', help='Enable beam/LM decode path')
    parser.add_argument('--use-decoder-denoiser', action='store_true', help='Use decoder.predict instead of predict_from_clean')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'

    decoder = build_decoder(device=device, lm_weight=args.lm_weight)

    test_word_virtually(
        word=args.word,
        decoder=decoder,
        window_width=args.window_width,
        step=args.step,
        visualize=args.visualize,
        use_lm=args.use_lm,
        use_decoder_denoiser=args.use_decoder_denoiser,
    )


if __name__ == '__main__':
    main()
