#!/usr/bin/env python3
"""
Download and convert Zenodo iCub Braille dataset to required format.
Source: https://zenodo.org/records/7050094
"""

import os
import sys
import pickle
import numpy as np
import requests
import zipfile
import shutil
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7050094/files/reading_braille_data.zip"
DATASET_DIR = Path("data/icub_braille_raw")
DOWNLOADS_DIR = Path("data/downloads")

def download_dataset():
    """Download the Zenodo dataset."""
    print("📥 Downloading iCub Braille dataset from Zenodo...")
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = DOWNLOADS_DIR / "reading_braille_data.zip"
    
    if zip_path.exists():
        print(f"✅ Dataset already downloaded at {zip_path}")
        return zip_path
    
    try:
        response = requests.get(ZENODO_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / total_size * 100) if total_size else 0
                    print(f"  Progress: {percent:.1f}%", end='\r')
        
        print(f"\n✅ Downloaded {zip_path.stat().st_size / 1e6:.1f} MB")
        return zip_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

def extract_dataset(zip_path):
    """Extract the downloaded zip file."""
    print("📦 Extracting dataset...")
    extract_dir = DOWNLOADS_DIR / "extracted"
    
    if extract_dir.exists():
        print(f"✅ Already extracted at {extract_dir}")
        return extract_dir
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        print(f"✅ Extracted to {extract_dir}")
        return extract_dir
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)

def convert_pickle_to_npy(extracted_dir):
    """
    Convert pickle files (pandas DataFrame) to individual .npy files per character.
    DataFrame has columns: 'letter', 'repetition', 'taxel_data' (tactile sensor readings)
    """
    print("🔄 Converting pickle data to .npy format...")
    
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find pickle files
    pickle_file = None
    for name in ["data_braille_letters_digits.pkl", "data_braille_letters_th_1.pkl"]:
        candidate = extracted_dir / name
        if candidate.exists():
            pickle_file = candidate
            break
    
    if not pickle_file:
        print(f"❌ No suitable pickle file found in {extracted_dir}")
        sys.exit(1)
    
    print(f"  Loading: {pickle_file.name}")
    
    try:
        import pandas as pd
        
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data
        
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Group by letter and save recordings
        total_samples = 0
        for letter, group in df.groupby('letter'):
            # Normalize letter name to lowercase
            letter_lower = letter.lower().replace('_', '').replace('space', 'space')
            
            char_dir = DATASET_DIR / letter_lower
            char_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (_, row) in enumerate(group.iterrows()):
                taxel_data = row.get('taxel_data')
                
                if taxel_data is None:
                    continue
                
                # Convert to (T, 12) format
                recording = np.asarray(taxel_data, dtype=np.float32)
                
                # Handle different shapes
                if recording.ndim == 1:
                    # Single channel - repeat to 12 channels
                    recording = np.tile(recording[:, None], (1, 12))
                elif recording.ndim == 2:
                    if recording.shape[1] < 12:
                        # Pad channels
                        pad_width = ((0, 0), (0, 12 - recording.shape[1]))
                        recording = np.pad(recording, pad_width, mode='constant')
                    elif recording.shape[1] > 12:
                        # Trim channels
                        recording = recording[:, :12]
                else:
                    print(f"  ⚠ Unexpected shape for {letter} rep {idx}: {recording.shape}")
                    continue
                
                # Normalize to [-1, 1]
                if len(recording) > 0:
                    rmin, rmax = recording.min(), recording.max()
                    if rmax > rmin:
                        recording = 2 * (recording - rmin) / (rmax - rmin) - 1
                    
                    npy_path = char_dir / f"{letter_lower}_{idx:03d}.npy"
                    np.save(npy_path, recording)
                    total_samples += 1
        
        print(f"✅ Created {total_samples} .npy files")
        
    except ImportError:
        print("❌ pandas not installed. Installing...")
        os.system("pip install pandas")
        convert_pickle_to_npy(extracted_dir)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def verify_structure():
    """Verify the created directory structure."""
    print("\n📊 Verifying dataset structure...")
    
    chars = list('abcdefghijklmnopqrstuvwxyz') + ['space']
    total_files = 0
    
    for char in chars:
        char_dir = DATASET_DIR / char
        
        if not char_dir.exists():
            print(f"  ⚠ Missing: {char}")
            continue
        
        npy_files = list(char_dir.glob("*.npy"))
        total_files += len(npy_files)
        print(f"  ✓ {char:>5}: {len(npy_files):>3} recordings")
    
    print(f"\n✅ Total: {total_files} recordings across {len(chars)} characters")
    
    if total_files == 0:
        print("❌ No files found!")
        sys.exit(1)

def main():
    print("=" * 70)
    print("  iCub BRAILLE DATASET SETUP")
    print("  Source: Zenodo (zenodo.org/records/7050094)")
    print("=" * 70)
    
    # Step 1: Download
    zip_path = download_dataset()
    
    # Step 2: Extract
    extracted_dir = extract_dataset(zip_path)
    
    # Step 3: Convert
    convert_pickle_to_npy(extracted_dir)
    
    # Step 4: Verify
    verify_structure()
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print(f"\nDataset is now ready at: {DATASET_DIR.absolute()}")
    print("\nNext steps:")
    print("  1. Run: python run_pipeline.py")
    print("  2. Or open: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("=" * 70)

if __name__ == "__main__":
    main()
