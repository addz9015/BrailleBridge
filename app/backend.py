"""
Backend inference service for the Braille demo.

This module centralizes sample generation and model inference so the frontend
can stay thin and only render results.
"""

import os
import json
import sys
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dataset import get_dataset
from models.denoiser import denoise
from models.pipeline import BrailleDecoder
from models.synthesis import SignalSynthesizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


BRAILLE_PATTERNS = {
    "a": [1, 0, 0, 0, 0, 0],
    "b": [1, 1, 0, 0, 0, 0],
    "c": [1, 0, 0, 1, 0, 0],
    "d": [1, 0, 0, 1, 1, 0],
    "e": [1, 0, 0, 0, 1, 0],
    "f": [1, 1, 0, 1, 0, 0],
    "g": [1, 1, 0, 1, 1, 0],
    "h": [1, 1, 0, 0, 1, 0],
    "i": [0, 1, 0, 1, 0, 0],
    "j": [0, 1, 0, 1, 1, 0],
    "k": [1, 0, 1, 0, 0, 0],
    "l": [1, 1, 1, 0, 0, 0],
    "m": [1, 0, 1, 1, 0, 0],
    "n": [1, 0, 1, 1, 1, 0],
    "o": [1, 0, 1, 0, 1, 0],
    "p": [1, 1, 1, 1, 0, 0],
    "q": [1, 1, 1, 1, 1, 0],
    "r": [1, 1, 1, 0, 1, 0],
    "s": [0, 1, 1, 1, 0, 0],
    "t": [0, 1, 1, 1, 1, 0],
    "u": [1, 0, 1, 0, 0, 1],
    "v": [1, 1, 1, 0, 0, 1],
    "w": [0, 1, 0, 1, 1, 1],
    "x": [1, 0, 1, 1, 0, 1],
    "y": [1, 0, 1, 1, 1, 1],
    "z": [1, 0, 1, 0, 1, 1],
    " ": [0, 0, 0, 0, 0, 0],
}

PATTERN_TO_CHAR = {tuple(pattern): ch for ch, pattern in BRAILLE_PATTERNS.items()}


@dataclass(frozen=True)
class SampleMeta:
    sample_id: int
    word: str
    split: str = "test"


class BrailleBackend:
    def __init__(self):
        self.samples_per_word = int(os.getenv("APP_SAMPLES_PER_WORD", "2"))
        self.base_seed = int(os.getenv("APP_SAMPLE_SEED", "42"))
        self.default_lm_weight = float(os.getenv("APP_LM_WEIGHT", "0.3"))
        self.default_beam_width = int(os.getenv("APP_BEAM_WIDTH", "12"))
        self._sample_cache = {}

        self.dataset = get_dataset(os.path.join(DATA_DIR, "icub_braille_raw"))
        self.synthesizer = SignalSynthesizer(self.dataset, os.path.join(CONFIG_DIR, "noise.yaml"))

        with open(os.path.join(DATA_DIR, "wordlist_splits.json"), "r") as handle:
            self.splits = json.load(handle)

        self.test_words = self.splits.get("test", [])
        self.samples = self._build_samples()

        self.decoder = BrailleDecoder(
            ctc_ckpt=os.path.join(CHECKPOINT_DIR, "ctc_best.pt"),
            bigram_path=os.path.join(MODEL_DIR, "bigram_logprobs.npy"),
            norm_path=os.path.join(DATA_DIR, "norm_params.json"),
            lm_weight=self.default_lm_weight,
            beam_width=self.default_beam_width,
            device="cpu",
        )

    def _sanitize_word(self, word: str) -> str:
        cleaned = "".join(ch for ch in word.lower() if ch in BRAILLE_PATTERNS)
        return cleaned.strip()

    def _pattern_to_unicode(self, pattern: List[int]) -> str:
        # Unicode Braille: dots 1..8 map to bits 0..7. We use the 6-dot subset.
        code = 0x2800
        for idx, active in enumerate(pattern[:6]):
            if active:
                code |= 1 << idx
        return chr(code)

    def _sanitize_tap_cells(self, cells: List[List[int]], expected_len: int) -> List[List[int]]:
        normalized = []
        for idx in range(expected_len):
            src = cells[idx] if idx < len(cells) else [0, 0, 0, 0, 0, 0]
            if len(src) != 6:
                raise ValueError(f"Cell {idx} must contain exactly 6 dot values.")
            cleaned = [1 if int(value) else 0 for value in src]
            normalized.append(cleaned)
        return normalized

    def _word_similarity(self, a: str, b: str) -> float:
        return float(SequenceMatcher(None, a, b).ratio())

    def _snap_tapped_word(self, tapped_word: str) -> tuple[str, float]:
        if not self.test_words:
            return tapped_word, 0.0

        best_word = tapped_word
        best_score = -1.0
        for candidate in self.test_words:
            score = self._word_similarity(tapped_word, candidate)
            if score > best_score:
                best_score = score
                best_word = candidate
        return best_word, best_score

    def analyze_tap_cells(self, word: str, cells: List[List[int]]) -> dict:
        cleaned_word = self._sanitize_word(word)
        if not cleaned_word:
            raise ValueError("Word must include at least one valid character (a-z or space).")

        normalized_cells = self._sanitize_tap_cells(cells, expected_len=len(cleaned_word))
        direct_chars = []
        unknown_count = 0
        for pattern in normalized_cells:
            mapped = PATTERN_TO_CHAR.get(tuple(pattern), "?")
            if mapped == "?":
                unknown_count += 1
            direct_chars.append(mapped)

        direct_word = "".join(direct_chars)
        decoded_word = direct_word
        decode_mode = "direct"
        snapped_similarity = 0.0

        if "?" in direct_word:
            fallback_seed = direct_word.replace("?", "")
            decoded_word, snapped_similarity = self._snap_tapped_word(fallback_seed)
            decode_mode = "snapped"

        per_cell = []
        for idx, pattern in enumerate(normalized_cells):
            mapped = direct_chars[idx]
            per_cell.append(
                {
                    "index": idx,
                    "pattern": pattern,
                    "unicode": self._pattern_to_unicode(pattern),
                    "char": mapped,
                    "known": mapped != "?",
                }
            )

        known_ratio = 1.0 - (unknown_count / max(len(normalized_cells), 1))
        target_similarity = self._word_similarity(decoded_word, cleaned_word)
        confidence = float(np.clip((0.65 * known_ratio) + (0.35 * target_similarity), 0.0, 1.0))

        warnings = []
        if unknown_count > 0:
            warnings.append("Some tapped cells did not map to a known 6-dot Braille character.")
        if decode_mode == "snapped":
            warnings.append("Used nearest-word snapping due to unknown tap pattern(s).")

        return {
            "word_input": word,
            "word_sanitized": cleaned_word,
            "predicted_word": decoded_word,
            "direct_word": direct_word,
            "decode_mode": decode_mode,
            "snapped_similarity": snapped_similarity,
            "confidence": confidence,
            "unknown_cells": unknown_count,
            "cells": per_cell,
            "warnings": warnings,
            "correct": decoded_word == cleaned_word,
        }

    def _build_samples(self):
        samples = []
        sample_id = 0
        for word in self.test_words:
            for _ in range(self.samples_per_word):
                samples.append(SampleMeta(sample_id=sample_id, word=word))
                sample_id += 1
        return samples

    def sample_options(self):
        return [f"{sample.sample_id} — '{sample.word}'" for sample in self.samples]

    def get_sample(self, sample_id: int):
        if sample_id < 0 or sample_id >= len(self.samples):
            raise IndexError(f"Sample index {sample_id} out of range")

        if sample_id in self._sample_cache:
            return self._sample_cache[sample_id]

        sample = self.samples[sample_id]
        rng = np.random.RandomState(self.base_seed + sample.sample_id)
        noisy, clean, boundaries = self.synthesizer.synthesize_word(sample.word, rng)

        cached_sample = {
            "sample_id": sample.sample_id,
            "word": sample.word,
            "noisy": noisy,
            "clean": clean,
            "boundaries": boundaries,
        }

        self._sample_cache[sample_id] = cached_sample
        return cached_sample

    def analyze_sample(self, sample_id: int, use_lm: bool = True, lm_weight: float | None = None, beam_width: int | None = None):
        sample = self.get_sample(sample_id)
        noisy = sample["noisy"]
        if "denoised" not in sample:
            sample["denoised"] = denoise(noisy)
        clean = sample["denoised"]

        self.decoder.lm_weight = self.default_lm_weight if lm_weight is None else float(lm_weight)
        self.decoder.beam_width = self.default_beam_width if beam_width is None else int(beam_width)

        greedy_pred, _ = self.decoder.predict_from_clean(clean, use_lm=False)
        lm_pred, candidates = self.decoder.predict_from_clean(clean, use_lm=use_lm)

        return {
            **sample,
            "denoised": clean,
            "greedy_pred": greedy_pred,
            "lm_pred": lm_pred,
            "candidates": candidates,
            "greedy_ok": greedy_pred == sample["word"],
            "lm_ok": lm_pred == sample["word"],
            "lm_weight": self.decoder.lm_weight,
            "beam_width": self.decoder.beam_width,
        }


backend = BrailleBackend()
