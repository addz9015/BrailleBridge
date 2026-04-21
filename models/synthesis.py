"""
Module 3: Signal Synthesis

Concatenates letter recordings into words and injects realistic boundary noise.
Produces (noisy_signal, clean_signal, label, boundaries) tuples.
"""

import numpy as np
import h5py
import json
import logging
from typing import Tuple, List, Dict
import yaml
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalSynthesizer:
    """
    Synthesizes word-level signals by concatenating letter recordings
    and injecting boundary noise.
    """
    
    def __init__(self, dataset_loader, noise_config_path: str = "config/noise.yaml"):
        """
        Initialize synthesizer.
        
        Args:
            dataset_loader: BrailleDatasetLoader instance
            noise_config_path: Path to noise configuration YAML
        """
        self.dataset = dataset_loader
        
        # Load noise config
        with open(noise_config_path, 'r') as f:
            self.noise_config = yaml.safe_load(f)
        
        self.W = self.noise_config.get('blend_window_width', 8)
        self.noise_scale = self.noise_config.get('noise_scale', 0.15)
        self.seed = self.noise_config.get('random_seed', 42)
        
        logger.info(f"Synthesizer initialized:")
        logger.info(f"  Blend window width: {self.W}")
        logger.info(f"  Noise scale: {self.noise_scale}")
        logger.info(f"  Random seed: {self.seed}")
    
    def synthesize_word(self, word: str, rng: np.random.RandomState) -> Tuple:
        """
        Synthesize a single word by concatenating letter recordings
        and injecting boundary noise.
        
        Args:
            word: Word to synthesize (lowercase, a-z and space only)
            rng: Random number generator (RandomState for reproducibility)
        
        Returns:
            Tuple of (noisy_signal, clean_signal, boundaries)
                noisy_signal: (T, 12) float32
                clean_signal: (T, 12) float32
                boundaries: list of ints (start indices of each letter)
        """
        # Collect letters
        letters = []
        for char in word:
            # Get random recording for this character
            available = self.dataset.get_all_letters(char)
            if not available:
                raise ValueError(f"No recordings for character '{char}'")
            idx = rng.randint(0, len(available))
            letters.append(self.dataset.get_letter(char, idx))
        
        # Concatenate clean signal
        clean_signal = np.concatenate(letters, axis=0)  # (T_word, 12)
        
        # Initialize noisy signal as copy of clean
        noisy_signal = clean_signal.copy()
        
        # Track boundaries (start index of each letter)
        boundaries = [0]
        current_pos = 0
        
        # Inject boundary noise at each letter transition
        for letter_idx in range(len(letters) - 1):
            letter_end = current_pos + letters[letter_idx].shape[0]
            next_letter_start = letter_end  # Start of next letter
            
            # Blending window: from (letter_end - W) to letter_end + W
            blend_start = max(0, letter_end - self.W)
            blend_end = min(noisy_signal.shape[0], letter_end + self.W)
            
            # Create blending mask (1 at letter_end, fades to 0)
            blend_len = blend_end - blend_start
            
            for t_idx in range(blend_len):
                t_global = blend_start + t_idx
                
                # Relative position in blend window: 0 = start, 1 = end
                t_rel = t_idx / blend_len if blend_len > 0 else 0.5
                
                # Alpha: 1 at letter_end, 0 at boundaries
                # Simple linear interpolation
                dist_from_boundary = abs(t_rel - (self.W / blend_len))
                alpha = max(0, 1 - dist_from_boundary * 2)
                
                # Get signal before and after boundary
                tail_signal = letters[letter_idx][-1, :]  # Last frame of current letter
                head_signal = letters[letter_idx + 1][0, :]  # First frame of next letter
                
                # Blend
                blended = alpha * tail_signal + (1 - alpha) * head_signal
                
                # Compute local std for noise scaling
                local_region = noisy_signal[max(0, t_global - 5):min(noisy_signal.shape[0], t_global + 5), :]
                local_std = np.std(local_region) if local_region.size > 0 else 1.0
                
                # Add Gaussian noise
                noise = rng.randn(12) * (local_std * self.noise_scale)
                
                # Update noisy signal
                noisy_signal[t_global, :] = blended + noise
            
            # Record next letter boundary
            current_pos = next_letter_start
            boundaries.append(current_pos)
        
        return (
            noisy_signal.astype(np.float32),
            clean_signal.astype(np.float32),
            boundaries
        )
    
    def synthesize_corpus(self, 
                         wordlist: List[str],
                         splits_dict: Dict[str, List[str]],
                         samples_per_word: Dict[str, int] = None,
                         output_path: str = "data/corpus.h5") -> str:
        """
        Synthesize full corpus and save to HDF5.
        
        Args:
            wordlist: Full word list
            splits_dict: Dict with 'train', 'val', 'test' keys
            samples_per_word: Dict with 'train', 'val', 'test' sample counts
                             (default: train=5, val=2, test=2)
            output_path: Path to save HDF5 file
        
        Returns:
            Path to saved HDF5 file
        """
        if samples_per_word is None:
            samples_per_word = {'train': 5, 'val': 2, 'test': 2}
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        rng = np.random.RandomState(self.seed)
        
        logger.info("=" * 60)
        logger.info("SYNTHESIZING CORPUS")
        logger.info("=" * 60)
        
        with h5py.File(output_path, 'w') as h5f:
            split_names = ['train', 'val', 'test']
            
            for split_name in split_names:
                split_words = splits_dict[split_name]
                n_samples = samples_per_word[split_name]
                
                split_group = h5f.create_group(split_name)
                
                logger.info(f"\n{split_name.upper()} SPLIT:")
                logger.info(f"  Words: {len(split_words)}")
                logger.info(f"  Samples per word: {n_samples}")
                logger.info(f"  Total samples: {len(split_words) * n_samples}")
                
                sample_idx = 0
                for word_idx, word in enumerate(split_words):
                    if (word_idx + 1) % max(1, len(split_words) // 10) == 0:
                        logger.info(f"  Progress: {word_idx + 1}/{len(split_words)}")
                    
                    for sample_num in range(n_samples):
                        try:
                            noisy, clean, boundaries = self.synthesize_word(word, rng)
                            
                            # Create group for this sample
                            sample_group = split_group.create_group(str(sample_idx))
                            sample_group.create_dataset('noisy', data=noisy, compression='gzip')
                            sample_group.create_dataset('clean', data=clean, compression='gzip')
                            sample_group.attrs['label'] = word
                            sample_group.create_dataset('boundaries', data=np.array(boundaries))
                            
                            sample_idx += 1
                        
                        except Exception as e:
                            logger.error(f"Error synthesizing word '{word}': {e}")
                            continue
        
        logger.info("\n" + "=" * 60)
        logger.info(f"CORPUS SYNTHESIS COMPLETE")
        logger.info(f"Saved to: {output_path}")
        logger.info("=" * 60)
        
        return output_path


def synthesize_corpus(dataset_loader,
                     wordlist: List[str],
                     splits_dict: Dict[str, List[str]],
                     noise_config_path: str = "config/noise.yaml",
                     output_path: str = "data/corpus.h5",
                     samples_per_word: Dict = None) -> str:
    """
    Convenience function to synthesize corpus.
    
    Args:
        dataset_loader: BrailleDatasetLoader instance
        wordlist: List of words
        splits_dict: Train/val/test split dictionary
        noise_config_path: Path to noise config
        output_path: Output HDF5 path
        samples_per_word: Samples per word per split
    
    Returns:
        Path to saved corpus
    """
    if samples_per_word is None:
        samples_per_word = {'train': 5, 'val': 2, 'test': 2}
    
    synthesizer = SignalSynthesizer(dataset_loader, noise_config_path)
    return synthesizer.synthesize_corpus(wordlist, splits_dict, samples_per_word, output_path)


if __name__ == "__main__":
    # Example usage (requires dataset and wordlist to be available)
    from models.dataset import get_dataset
    from models.wordlist import build_wordlist
    
    # Load dataset
    dataset = get_dataset("data/icub_braille_raw")
    
    # Build wordlist
    wordlist, splits = build_wordlist()
    
    # Synthesize corpus
    corpus_path = synthesize_corpus(dataset, wordlist, splits)
    print(f"Corpus saved to: {corpus_path}")
