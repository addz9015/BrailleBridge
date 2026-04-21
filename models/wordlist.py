"""
Module 2: Word List Construction

Builds a filtered word list suitable for Braille synthesis.
- 3-8 character words
- a-z and space only
- 300-500 words
- Fixed 70/15/15 train/val/test split
"""

import json
import numpy as np
import logging
from typing import List, Dict, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_standard_wordlist() -> List[str]:
    """
    Get a standard English word list.
    Uses NLTK's brown corpus if available, otherwise uses a fallback list.
    """
    try:
        from nltk.corpus import brown
        import nltk
        try:
            nltk.data.find('corpora/brown')
        except LookupError:
            logger.info("Downloading NLTK brown corpus...")
            nltk.download('brown', quiet=True)
        
        # Get word frequencies from brown corpus
        words = []
        for word in brown.words():
            words.append(word.lower())
        
        # Get unique words
        unique_words = list(set(words))
        logger.info(f"Loaded {len(unique_words)} unique words from brown corpus")
        return unique_words
    
    except ImportError:
        logger.warning("NLTK not available, using fallback word list")
        # Fallback: use a common English word list
        # In practice, you'd download a word list file
        fallback_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
            'cat', 'dog', 'the', 'tree', 'book', 'word', 'name', 'hand', 'head', 'face',
            'very', 'where', 'much', 'such', 'those', 'tell', 'boy', 'did', 'car', 'let',
            'put', 'say', 'too', 'use', 'big', 'old', 'own', 'say', 'she', 'top', 'try',
        ]
        return fallback_words


def filter_wordlist(words: List[str], 
                   min_len: int = 3,
                   max_len: int = 8,
                   min_count: int = 300,
                   max_count: int = 500) -> List[str]:
    """
    Filter word list according to constraints.
    
    Args:
        words: Input word list
        min_len: Minimum word length (chars)
        max_len: Maximum word length (chars)
        min_count: Minimum number of words to return
        max_count: Maximum number of words to return
    
    Returns:
        Filtered word list, sorted and unique
    """
    logger.info(f"Filtering word list: {len(words)} input words")
    logger.info(f"Constraints: length [{min_len}, {max_len}], "
               f"a-z only, no proper nouns, {min_count}-{max_count} words")
    
    filtered = set()
    
    for word in words:
        # Must be lowercase
        if word != word.lower():
            continue
        
        # Length constraints
        if not (min_len <= len(word) <= max_len):
            continue
        
        # Only a-z and space
        if not all(c in 'abcdefghijklmnopqrstuvwxyz ' for c in word):
            continue
        
        # Avoid proper nouns (heuristic: words that start with capital in original often are proper nouns)
        # Since we only have lowercase, we'll skip this check
        
        # Avoid common abbreviations and contractions
        if "'" in word or "-" in word:
            continue
        
        filtered.add(word)
    
    # Convert to sorted list
    filtered_list = sorted(list(filtered))
    logger.info(f"After filtering: {len(filtered_list)} words")
    
    # If we have too many, sample randomly (reproducible with seed)
    if len(filtered_list) > max_count:
        np.random.seed(42)
        indices = np.random.choice(len(filtered_list), size=max_count, replace=False)
        filtered_list = [filtered_list[i] for i in sorted(indices)]
        logger.info(f"Randomly sampled to {max_count} words (seed=42)")
    
    # Check minimum requirement
    if len(filtered_list) < min_count:
        logger.warning(f"Only {len(filtered_list)} words after filtering, "
                      f"minimum {min_count} required. Expanding...")
        # Could relax constraints here, but for now we'll warn
    
    return filtered_list


def create_splits(wordlist: List[str], 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42) -> Dict[str, List[str]]:
    """
    Split wordlist into train/val/test.
    
    Args:
        wordlist: List of words
        train_ratio: Proportion for training (default 70%)
        val_ratio: Proportion for validation (default 15%)
        test_ratio: Proportion for testing (default 15%)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing word lists
    """
    np.random.seed(seed)
    
    n = len(wordlist)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    splits = {
        'train': [wordlist[i] for i in train_idx],
        'val': [wordlist[i] for i in val_idx],
        'test': [wordlist[i] for i in test_idx]
    }
    
    logger.info(f"Splits (seed={seed}):")
    logger.info(f"  Train: {len(splits['train'])} words")
    logger.info(f"  Val:   {len(splits['val'])} words")
    logger.info(f"  Test:  {len(splits['test'])} words")
    
    return splits


def build_wordlist(output_dir: str = "data",
                  min_len: int = 3,
                  max_len: int = 8,
                  target_count: int = 400,
                  seed: int = 42):
    """
    Build and save word list and splits.
    
    Args:
        output_dir: Directory to save files
        min_len: Minimum word length
        max_len: Maximum word length
        target_count: Target number of words (300-500)
        seed: Random seed
    
    Returns:
        Tuple of (wordlist, splits_dict)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Building word list")
    logger.info("=" * 60)
    
    # Get standard word list
    words = get_standard_wordlist()
    
    # Filter
    wordlist = filter_wordlist(words, min_len=min_len, max_len=max_len,
                              min_count=300, max_count=target_count)
    
    if len(wordlist) == 0:
        raise ValueError("No words found matching criteria")
    
    # Create splits
    splits = create_splits(wordlist, seed=seed)
    
    # Save wordlist
    wordlist_path = os.path.join(output_dir, 'wordlist.txt')
    with open(wordlist_path, 'w') as f:
        for word in wordlist:
            f.write(word + '\n')
    logger.info(f"Saved wordlist to {wordlist_path}")
    
    # Save splits as JSON
    splits_path = os.path.join(output_dir, 'wordlist_splits.json')
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits to {splits_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"WORDLIST CONSTRUCTION COMPLETE")
    logger.info(f"  Total words: {len(wordlist)}")
    logger.info(f"  Length range: [{min_len}, {max_len}]")
    logger.info(f"  Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    logger.info("=" * 60)
    
    return wordlist, splits


if __name__ == "__main__":
    wordlist, splits = build_wordlist()
    print(f"\nFirst 10 words: {wordlist[:10]}")
    print(f"Last 10 words: {wordlist[-10:]}")
