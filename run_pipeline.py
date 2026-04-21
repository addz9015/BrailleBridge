"""
Master Training Pipeline Orchestration

This script coordinates all phases of the Person A workflow:
1. Dataset loading and validation
2. Word list construction
3. Synthetic corpus generation
4. Denoising autoencoder training
5. Letter-level baseline classifier training
6. Evaluation and reporting

Usage:
    python run_pipeline.py
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

for d in [DATA_DIR, CHECKPOINT_DIR, CONFIG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
sys.path.insert(0, PROJECT_ROOT)
from models.dataset import get_dataset
from models.wordlist import build_wordlist
from models.synthesis import synthesize_corpus
from models.dae_simple import DenoisingAutoencoder, DenoisingAutoEncoderTrainer
from models.baseline import LetterClassifier, LetterClassifierTrainer
import torch

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

def phase_1_dataset_loading():
    """Phase 1: Load and validate dataset."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Dataset Loading & Validation")
    logger.info("=" * 70)
    
    dataset = get_dataset(os.path.join(DATA_DIR, 'icub_braille_raw'))
    logger.info("✓ Dataset loaded successfully")
    return dataset

def phase_2_wordlist_construction():
    """Phase 2: Build word list and splits."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Word List Construction")
    logger.info("=" * 70)
    
    wordlist, splits = build_wordlist(output_dir=DATA_DIR)
    
    # Also save as splits.json for Person B
    splits_path = os.path.join(DATA_DIR, 'splits.json')
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"✓ Splits also saved to {splits_path}")
    
    return wordlist, splits

def phase_3_corpus_synthesis(dataset, wordlist, splits):
    """Phase 3: Generate synthetic corpus."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Synthetic Corpus Generation")
    logger.info("=" * 70)
    
    corpus_path = os.path.join(DATA_DIR, 'corpus.h5')
    
    samples_per_word = {
        'train': 5,
        'val': 2,
        'test': 2
    }
    
    synthesize_corpus(
        dataset,
        wordlist,
        splits,
        noise_config_path=os.path.join(CONFIG_DIR, 'noise.yaml'),
        output_path=corpus_path,
        samples_per_word=samples_per_word
    )
    
    logger.info(f"✓ Corpus saved to {corpus_path}")
    return corpus_path

def phase_4_dae_training(dataset, corpus_path):
    """Phase 4: Train denoising autoencoder."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Denoising Autoencoder Training")
    logger.info("=" * 70)
    
    import h5py
    
    # Load training data from corpus
    train_loader = []
    val_loader = []
    
    with h5py.File(corpus_path, 'r') as f:
        # Load train set
        for sample_id in sorted(f['train'].keys()):
            noisy = f['train'][sample_id]['noisy'][:]
            clean = f['train'][sample_id]['clean'][:]
            train_loader.append((noisy, clean))
        
        # Load val set
        for sample_id in sorted(f['val'].keys()):
            noisy = f['val'][sample_id]['noisy'][:]
            clean = f['val'][sample_id]['clean'][:]
            val_loader.append((noisy, clean))

    max_train_samples = int(os.getenv('PIPELINE_DAE_MAX_SAMPLES', '0'))
    max_val_samples = int(os.getenv('PIPELINE_DAE_VAL_MAX_SAMPLES', '0'))
    if max_train_samples > 0:
        train_loader = train_loader[:max_train_samples]
        logger.info(f"Using first {len(train_loader)} DAE train samples (PIPELINE_DAE_MAX_SAMPLES)")
    if max_val_samples > 0:
        val_loader = val_loader[:max_val_samples]
        logger.info(f"Using first {len(val_loader)} DAE val samples (PIPELINE_DAE_VAL_MAX_SAMPLES)")
    
    logger.info(f"Train samples: {len(train_loader)}, Val samples: {len(val_loader)}")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = DenoisingAutoencoder()
    trainer = DenoisingAutoEncoderTrainer(model, device=device, learning_rate=1e-3)
    dae_epochs = int(os.getenv('PIPELINE_DAE_EPOCHS', '50'))
    
    # Train
    trainer.fit(
        train_loader,
        val_loader,
        epochs=dae_epochs,
        batch_size=16,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, 'dae_best.pt')
    )
    
    logger.info(f"✓ Best model saved to {os.path.join(CHECKPOINT_DIR, 'dae_best.pt')}")

def phase_5_baseline_training(dataset):
    """Phase 5: Train letter-level baseline classifier."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Letter-Level Baseline Classifier")
    logger.info("=" * 70)
    
    # Prepare letter data with 80/10/10 split
    all_letters = []
    char_to_idx = {}
    
    for idx, char in enumerate('abcdefghijklmnopqrstuvwxyz '):
        char_to_idx[char] = idx
        recordings = dataset.get_all_letters(char)
        for recording in recordings:
            all_letters.append((recording, idx))
    
    logger.info(f"Total letter samples: {len(all_letters)}")
    
    # Split
    n = len(all_letters)
    
    indices = np.random.permutation(n)
    max_letter_samples = int(os.getenv('PIPELINE_BASELINE_MAX_SAMPLES', '0'))
    if max_letter_samples > 0:
        indices = indices[:max_letter_samples]
        logger.info(f"Using random subset of {len(indices)} baseline samples (PIPELINE_BASELINE_MAX_SAMPLES)")

    n = len(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_data = [all_letters[i] for i in indices[:n_train]]
    val_data = [all_letters[i] for i in indices[n_train:n_train + n_val]]
    test_data = [all_letters[i] for i in indices[n_train + n_val:]]
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LetterClassifier(num_classes=27, fixed_length=150)
    trainer = LetterClassifierTrainer(model, device=device, learning_rate=1e-3)
    baseline_epochs = int(os.getenv('PIPELINE_BASELINE_EPOCHS', '50'))
    
    trainer.fit(
        train_data,
        val_data,
        epochs=baseline_epochs,
        batch_size=32,
        patience=10,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    # Test
    test_acc, per_char_acc = trainer.test(test_data)
    
    logger.info(f"\n✓ Baseline Results:")
    logger.info(f"  Overall accuracy: {test_acc * 100:.2f}%")
    logger.info(f"  Saved model to {os.path.join(CHECKPOINT_DIR, 'letter_baseline.pt')}")
    
    return test_acc, per_char_acc

def phase_6_evaluation(splits):
    """Phase 6: Compute WER/CER for baseline condition."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Baseline Evaluation (WER/CER)")
    logger.info("=" * 70)
    
    logger.info("To compute WER/CER, run:")
    logger.info("  python evaluate_baseline.py")
    logger.info("\nThis requires the letter classifier model to make predictions on test words.")

def main():
    """Run full pipeline."""
    logger.info("\n" + "█" * 70)
    logger.info("█ PERSON A: BRAILLE RECOGNITION PIPELINE")
    logger.info("█ Data Synthesis + Denoising Autoencoder")
    logger.info("█" * 70 + "\n")
    
    try:
        # Phase 1: Dataset Loading
        dataset = phase_1_dataset_loading()
        
        # Phase 2: Word List
        wordlist, splits = phase_2_wordlist_construction()
        
        # Phase 3: Corpus Synthesis
        corpus_path = phase_3_corpus_synthesis(dataset, wordlist, splits)
        
        # Phase 4: DAE Training
        phase_4_dae_training(dataset, corpus_path)
        
        # Phase 5: Baseline Training
        test_acc, per_char_acc = phase_5_baseline_training(dataset)
        
        # Phase 6: Evaluation
        phase_6_evaluation(splits)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info("\nHandoff artifacts ready for Person B:")
        logger.info(f"  ✓ {os.path.join(DATA_DIR, 'corpus.h5')}")
        logger.info(f"  ✓ {os.path.join(CHECKPOINT_DIR, 'dae_best.pt')}")
        logger.info(f"  ✓ {os.path.join(DATA_DIR, 'norm_params.json')}")
        logger.info(f"  ✓ {os.path.join(DATA_DIR, 'wordlist_splits.json')}")
        logger.info(f"  ✓ models/denoiser.py (callable interface)")
        logger.info("\n")
        
    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
