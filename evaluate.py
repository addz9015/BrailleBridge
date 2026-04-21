"""
Shared Evaluation Script

Computes WER (Word Error Rate) and CER (Character Error Rate).
Used by both Person A (for baseline) and Person B (for main conditions).

Requires: jiwer library
"""

import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_wer_cer(predictions: List[str],
                   targets: List[str]) -> Tuple[float, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER).
    
    Args:
        predictions: List of predicted words/sequences
        targets: List of target words/sequences
    
    Returns:
        Tuple of (WER, CER) as percentages [0-100]
    """
    try:
        import jiwer
    except ImportError:
        logger.error("jiwer not installed. Install with: pip install jiwer")
        raise
    
    if len(predictions) != len(targets):
        raise ValueError(f"Predictions and targets must have same length: "
                        f"{len(predictions)} vs {len(targets)}")
    
    # Compute WER
    wer = jiwer.wer(targets, predictions)
    
    # Compute CER
    cer = jiwer.cer(targets, predictions)
    
    return wer * 100, cer * 100


def evaluate_baseline(predictions: List[str],
                     targets: List[str]) -> dict:
    """
    Evaluate baseline condition.
    
    Args:
        predictions: List of predicted sequences
        targets: List of target sequences
    
    Returns:
        Dictionary with WER, CER, and other metrics
    """
    wer, cer = compute_wer_cer(predictions, targets)
    
    # Accuracy (exact match)
    accuracy = sum([1 for p, t in zip(predictions, targets) if p == t]) / len(targets)
    
    results = {
        'WER': wer,
        'CER': cer,
        'accuracy': accuracy * 100,
        'num_samples': len(targets)
    }
    
    return results


def print_results(results: dict, condition_name: str = "Condition"):
    """Print evaluation results nicely."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{condition_name} Results")
    logger.info(f"{'=' * 60}")
    logger.info(f"  WER: {results['WER']:.2f}%")
    logger.info(f"  CER: {results['CER']:.2f}%")
    logger.info(f"  Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"  Samples: {results['num_samples']}")
    logger.info(f"{'=' * 60}\n")


def evaluate_with_confidence(predictions: List[Tuple[str, float]],
                            targets: List[str]) -> dict:
    """
    Evaluate with confidence scores.
    
    Args:
        predictions: List of (prediction, confidence) tuples
        targets: List of target sequences
    
    Returns:
        Results dict with additional confidence analysis
    """
    pred_texts = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    
    results = evaluate_baseline(pred_texts, targets)
    
    # Add confidence statistics
    results['avg_confidence'] = float(np.mean(confidences))
    results['min_confidence'] = float(np.min(confidences))
    results['max_confidence'] = float(np.max(confidences))
    
    return results


if __name__ == "__main__":
    # Example usage
    preds = ["hello", "world", "test"]
    targets = ["hello", "word", "test"]
    
    results = evaluate_baseline(preds, targets)
    print_results(results, "Example")
