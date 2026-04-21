"""
Module 1: iCub Tactile Braille Dataset Loader

Loads and validates the iCub tactile Braille dataset.
Returns spike train recordings across 12 taxel channels.
"""

import numpy as np
import os
import json
from typing import Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All 26 letters + space = 27 classes
ALL_CHARACTERS = 'abcdefghijklmnopqrstuvwxyz '


class BrailleDatasetLoader:
    """
    Loads iCub tactile Braille dataset.
    
    Expected directory structure:
        dataset/
            a/
                0.npy, 1.npy, ...
            b/
                0.npy, 1.npy, ...
            ...
            z/
            space/ (or ' ')
    
    Each file should be a numpy array of shape (T, 12) where:
        T = variable timesteps (typically 50-150)
        12 = taxel channels
    """
    
    def __init__(self, dataset_dir: str = "data/icub_braille_raw"):
        """
        Initialize dataset loader.
        
        Args:
            dataset_dir: Path to the iCub dataset directory
        """
        self.dataset_dir = dataset_dir
        self.data = {}  # {char: [array_0, array_1, ...]}
        self.stats = {}  # {char: {mean, std, min, max, count}}
        self.character_counts = {}
        
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory {dataset_dir} not found. "
                          "You need to download the iCub tactile Braille dataset manually.")
            return
        
        self._load_dataset()
        self._validate_dataset()
        self._compute_statistics()
    
    def _load_dataset(self):
        """Load all character recordings from disk."""
        logger.info(f"Loading dataset from {self.dataset_dir}")
        
        for char in ALL_CHARACTERS:
            # Handle space character specially
            char_dir_name = "space" if char == " " else char
            char_dir = os.path.join(self.dataset_dir, char_dir_name)
            
            if not os.path.isdir(char_dir):
                logger.warning(f"Directory not found for character '{char}': {char_dir}")
                self.data[char] = []
                continue
            
            recordings = []
            for fname in sorted(os.listdir(char_dir)):
                if fname.endswith('.npy'):
                    fpath = os.path.join(char_dir, fname)
                    try:
                        recording = np.load(fpath)
                        # Validate shape: should be (T, 12)
                        if len(recording.shape) != 2 or recording.shape[1] != 12:
                            logger.warning(f"Invalid shape {recording.shape} for {char}/{fname}, "
                                         "expected (T, 12)")
                            continue
                        if not np.isfinite(recording).all():
                            logger.warning(f"NaN or Inf found in {char}/{fname}, skipping")
                            continue
                        recordings.append(recording)
                    except Exception as e:
                        logger.warning(f"Error loading {char}/{fname}: {e}")
                        continue
            
            self.data[char] = recordings
            self.character_counts[char] = len(recordings)
    
    def _validate_dataset(self):
        """Check that all characters have at least 20 recordings."""
        logger.info("Validating dataset...")
        
        problems = []
        for char in ALL_CHARACTERS:
            count = self.character_counts.get(char, 0)
            if count < 20:
                problems.append(f"  Character '{char}': {count} recordings (minimum 20 required)")
            elif count < 30:
                logger.warning(f"Character '{char}': only {count} recordings")
            else:
                logger.info(f"Character '{char}': {count} recordings ✓")
        
        if problems:
            logger.error("Dataset validation FAILED:")
            for problem in problems:
                logger.error(problem)
            raise ValueError("Dataset does not meet minimum requirements (20 recordings per character)")
        
        logger.info("Dataset validation PASSED ✓")
    
    def _compute_statistics(self):
        """Compute mean, std, min, max per character."""
        logger.info("Computing signal statistics...")
        
        for char in ALL_CHARACTERS:
            recordings = self.data.get(char, [])
            if not recordings:
                self.stats[char] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0,
                    'avg_length': 0
                }
                continue
            
            all_signals = np.concatenate(recordings, axis=0)  # (total_T, 12)
            
            self.stats[char] = {
                'mean': float(np.mean(all_signals)),
                'std': float(np.std(all_signals)),
                'min': float(np.min(all_signals)),
                'max': float(np.max(all_signals)),
                'count': len(recordings),
                'avg_length': float(np.mean([r.shape[0] for r in recordings]))
            }
            
            logger.info(f"  {char:3s}: μ={self.stats[char]['mean']:7.3f}, "
                       f"σ={self.stats[char]['std']:6.3f}, "
                       f"avg_T={self.stats[char]['avg_length']:6.1f}, "
                       f"N={self.stats[char]['count']}")
    
    def get_letter(self, char: str, idx: int) -> np.ndarray:
        """
        Get a single letter recording.
        
        Args:
            char: Character (a-z or space)
            idx: Index of recording for this character
        
        Returns:
            numpy array of shape (T, 12)
        
        Raises:
            ValueError: If character not found or index out of range
        """
        if char not in ALL_CHARACTERS:
            raise ValueError(f"Invalid character: '{char}'. Must be a-z or space.")
        
        recordings = self.data.get(char, [])
        if idx < 0 or idx >= len(recordings):
            raise IndexError(f"Character '{char}' has {len(recordings)} recordings, "
                           f"requested index {idx}")
        
        return recordings[idx].astype(np.float32)
    
    def get_all_letters(self, char: str) -> list:
        """Get all recordings for a character."""
        if char not in ALL_CHARACTERS:
            raise ValueError(f"Invalid character: '{char}'. Must be a-z or space.")
        return self.data.get(char, [])
    
    def get_statistics(self) -> Dict:
        """Return signal statistics dictionary."""
        return self.stats
    
    def save_statistics(self, output_path: str):
        """Save statistics to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistics saved to {output_path}")


def get_dataset(dataset_dir: str = "data/icub_braille_raw") -> BrailleDatasetLoader:
    """Factory function to get dataset loader instance."""
    return BrailleDatasetLoader(dataset_dir)
