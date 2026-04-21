"""
Baseline Evaluation Script

Computes WER/CER for condition ①: letter-level classification.
This is the ablation baseline that uses individual letter classifiers.

Usage:
    python evaluate_baseline.py
"""

import os
import sys
import json
import numpy as np
import torch
import h5py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.dataset import get_dataset
from models.baseline import LetterClassifier
from evaluate import evaluate_baseline, print_results

def evaluate_baseline_condition():
    """
    Evaluate baseline condition ①.
    
    Strategy: Use the letter classifier to predict each letter independently,
    then concatenate predictions to form words.
    """
    logger.info("=" * 70)
    logger.info("BASELINE EVALUATION (Condition ①)")
    logger.info("=" * 70)
    
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.h5')
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'letter_baseline.pt')
    
    # Check files exist
    if not os.path.exists(CORPUS_PATH):
        logger.error(f"Corpus not found: {CORPUS_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.info("Run training first: python run_pipeline.py")
        return
    
    if not os.path.exists(os.path.join(DATA_DIR, 'wordlist_splits.json')):
        logger.error("Word list splits not found")
        return
    
    # Load splits
    with open(os.path.join(DATA_DIR, 'wordlist_splits.json'), 'r') as f:
        splits = json.load(f)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LetterClassifier(num_classes=27, fixed_length=150)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {MODEL_PATH}")
    logger.info(f"Using device: {device}")
    
    # Alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    
    # Load corpus and make predictions
    predictions = []
    targets = []
    
    with h5py.File(CORPUS_PATH, 'r') as f:
        test_group = f['test']
        
        logger.info(f"Processing {len(test_group)} test samples...")
        
        for sample_id in sorted(test_group.keys()):
            sample = test_group[sample_id]
            label = sample.attrs['label']
            
            # Predict each letter using letter classifier
            predicted_word = ""
            
            for char in label:
                # Get a random recording of this character from the dataset
                # For simplicity, we'll just use the first recording
                dataset = get_dataset(os.path.join(DATA_DIR, 'icub_braille_raw'))
                letter_recording = dataset.get_letter(char, 0)  # (T, 12)
                
                # Prepare for model: pad/truncate to 150
                T_target = 150
                if letter_recording.shape[0] >= T_target:
                    letter_proc = letter_recording[:T_target, :]
                else:
                    pad_len = T_target - letter_recording.shape[0]
                    letter_proc = np.pad(letter_recording,
                                        ((0, pad_len), (0, 0)),
                                        mode='constant', constant_values=0)
                
                # Forward pass
                letter_tensor = torch.from_numpy(letter_proc).float().to(device).unsqueeze(0).transpose(1, 2)
                with torch.no_grad():
                    output = model(letter_tensor)
                pred_idx = output.argmax(dim=1).item()
                predicted_word += alphabet[pred_idx]
            
            predictions.append(predicted_word)
            targets.append(label)
    
    logger.info(f"Generated predictions for {len(predictions)} words")
    
    # Compute WER/CER
    results = evaluate_baseline(predictions, targets)
    print_results(results, "Condition ① (Letter-Level Baseline)")
    
    # Save results
    results_path = os.path.join(DATA_DIR, 'baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results

if __name__ == "__main__":
    try:
        results = evaluate_baseline_condition()
        if results:
            logger.info(f"\n✓ WER: {results['WER']:.2f}%")
            logger.info(f"✓ CER: {results['CER']:.2f}%")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
