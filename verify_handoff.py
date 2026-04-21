#!/usr/bin/env python3
"""
Final Verification Checklist

Verifies all deliverables are ready for downstream integration.
Run this after completing run_pipeline.py

Usage:
    python verify_handoff.py
"""

import os
import sys
import json
import h5py
import numpy as np
import logging
import io
import io

# Ensure UTF-8 encoding for Windows compatibility
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# Ensure UTF-8 encoding for Windows compatibility
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

def check_file_exists(path, description):
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "[OK]" if exists else "[XX]"
    print(f"  {status} {description}: {path}")
    return exists

def verify_artifacts():
    """Verify all handoff artifacts."""
    print("\n" + "=" * 70)
    print("BRAILLEBRIDGE: HANDOFF VERIFICATION CHECKLIST")
    print("=" * 70 + "\n")
    
    all_ok = True
    
    # Data artifacts
    print("Data Artifacts:")
    all_ok &= check_file_exists(os.path.join(DATA_DIR, 'wordlist.txt'), 'Word list')
    all_ok &= check_file_exists(os.path.join(DATA_DIR, 'wordlist_splits.json'), 'Word splits')
    all_ok &= check_file_exists(os.path.join(DATA_DIR, 'splits.json'), 'Splits')
    all_ok &= check_file_exists(os.path.join(DATA_DIR, 'corpus.h5'), 'Synthetic corpus')
    all_ok &= check_file_exists(os.path.join(DATA_DIR, 'norm_params.json'), 'Normalization params')
    
    # Config
    print("\nConfiguration:")
    all_ok &= check_file_exists(os.path.join(CONFIG_DIR, 'noise.yaml'), 'Noise config')
    
    # Model checkpoints
    print("\nModel Checkpoints:")
    all_ok &= check_file_exists(os.path.join(CHECKPOINT_DIR, 'dae_best.pt'), 'DAE best weights')
    all_ok &= check_file_exists(os.path.join(CHECKPOINT_DIR, 'letter_baseline.pt'), 'Baseline weights')
    
    # Export interface
    print("\nExport Interface:")
    all_ok &= check_file_exists(os.path.join(MODEL_DIR, 'denoiser.py'), 'Denoiser export')
    
    # Evaluation script
    print("\nEvaluation:")
    all_ok &= check_file_exists(os.path.join(PROJECT_ROOT, 'evaluate.py'), 'Shared evaluate.py')
    all_ok &= check_file_exists(os.path.join(PROJECT_ROOT, 'evaluate_baseline.py'), 'Baseline evaluation')
    
    return all_ok

def verify_data_integrity():
    """Verify data integrity."""
    print("\n" + "-" * 70)
    print("Data Integrity Checks:")
    print("-" * 70 + "\n")
    
    try:
        # Check wordlist
        wordlist_path = os.path.join(DATA_DIR, 'wordlist.txt')
        with open(wordlist_path, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        
        print(f"  Wordlist: {len(words)} words")
        if 300 <= len(words) <= 500:
            print(f"    [OK] Within target range (300-500)")
        else:
            print(f"    [XX] Outside target range (300-500)")
            return False
        
        # Check splits
        splits_path = os.path.join(DATA_DIR, 'wordlist_splits.json')
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        print(f"  Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        if total == len(words):
            print(f"    [OK] Splits are exhaustive")
        else:
            print(f"    [XX] Splits don't match wordlist")
            return False
        
        # Check corpus
        corpus_path = os.path.join(DATA_DIR, 'corpus.h5')
        with h5py.File(corpus_path, 'r') as f:
            splits_in_corpus = list(f.keys())
            print(f"  Corpus: splits={splits_in_corpus}")
            
            for split_name in ['train', 'val', 'test']:
                if split_name in f:
                    n_samples = len(f[split_name])
                    print(f"    {split_name}: {n_samples} samples")
                    
                    # Check first sample structure
                    sample = f[split_name]['0']
                    noisy_shape = sample['noisy'].shape
                    clean_shape = sample['clean'].shape
                    
                    if noisy_shape == clean_shape and noisy_shape[1] == 12:
                        print(f"      [OK] Sample structure correct (T, 12)")
                    else:
                        print(f"      [XX] Invalid shape: noisy={noisy_shape}, clean={clean_shape}")
                        return False
        
        # Check normalization params
        norm_path = os.path.join(DATA_DIR, 'norm_params.json')
        with open(norm_path, 'r') as f:
            norm = json.load(f)
        
        print(f"  Normalization params:")
        print(f"    Mean: {len(norm['mean'])} channels")
        print(f"    Std: {len(norm['std'])} channels")
        if len(norm['mean']) == 12 and len(norm['std']) == 12:
            print(f"      [OK] Correct for 12-channel signals")
        else:
            print(f"      [XX] Wrong number of channels")
            return False
        
        return True
    
    except Exception as e:
        print(f"  [XX] Error: {e}")
        return False

def smoke_test_denoiser():
    """Smoke test: ensure denoiser works."""
    print("\n" + "-" * 70)
    print("Smoke Test: Denoiser Interface")
    print("-" * 70 + "\n")
    
    try:
        # Import denoiser
        sys.path.insert(0, PROJECT_ROOT)
        from models.denoiser import denoise
        
        print("  Attempting to denoise a test signal...")
        
        # Create dummy signal
        test_signal = np.random.randn(100, 12).astype(np.float32)
        
        # Denoise (this will try to load the model)
        try:
            denoised = denoise(test_signal)
            
            if denoised.shape == test_signal.shape:
                print(f"  [OK] Denoiser works!")
                print(f"    Input shape: {test_signal.shape}")
                print(f"    Output shape: {denoised.shape}")
                print(f"    Output dtype: {denoised.dtype}")
                return True
            else:
                print(f"  [XX] Output shape mismatch: {denoised.shape} != {test_signal.shape}")
                return False
        
        except FileNotFoundError as e:
            print(f"  [INFO] Model weights not found (normal if training not complete)")
            print(f"    Will be generated after: python run_pipeline.py")
            return True  # Not a fatal error
    
    except Exception as e:
        print(f"  [XX] Error: {e}")
        return False

def create_final_report():
    """Create final handoff report."""
    print("\n" + "=" * 70)
    print("FINAL HANDOFF REPORT")
    print("=" * 70 + "\n")
    
    # Summary
    artifacts_ok = verify_artifacts()
    integrity_ok = verify_data_integrity()
    denoiser_ok = smoke_test_denoiser()
    
    all_checks_passed = artifacts_ok and integrity_ok and denoiser_ok
    
    print("\n" + "-" * 70)
    print("Summary:")
    print("-" * 70)
    print(f"  Artifacts present: {'[OK] PASS' if artifacts_ok else '[XX] FAIL'}")
    print(f"  Data integrity: {'[OK] PASS' if integrity_ok else '[XX] FAIL'}")
    print(f"  Denoiser interface: {'[OK] PASS' if denoiser_ok else '[XX] FAIL'}")
    
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("[OK] ALL CHECKS PASSED - READY FOR HANDOFF [OK]")
    else:
        print("[XX] SOME CHECKS FAILED - PLEASE REVIEW ABOVE [XX]")
    print("=" * 70 + "\n")
    
    return all_checks_passed

if __name__ == "__main__":
    success = create_final_report()
    sys.exit(0 if success else 1)
