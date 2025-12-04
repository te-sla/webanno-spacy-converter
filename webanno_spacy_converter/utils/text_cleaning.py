"""
Text cleaning utilities for transformer-safe text processing.

Transformer models (BERT, RoBERTa, etc.) are sensitive to hidden characters
that can cause unexpected tokenization. This module provides functions to
clean text before conversion to spaCy format.
"""

import re
from typing import Dict, Tuple

# Characters to remove completely (invisible, no semantic meaning for NLP)
CHARS_TO_REMOVE = {
    '\u200b': 'Zero-Width Space (ZWSP)',
    '\u200c': 'Zero-Width Non-Joiner (ZWNJ)',
    '\u200d': 'Zero-Width Joiner (ZWJ)',
    '\u2060': 'Word Joiner',
    '\xad': 'Soft Hyphen',
    '\ufeff': 'BOM/Zero-Width No-Break Space',  # When mid-text
}

# Build regex pattern for all removable characters
_ALL_PROBLEMATIC = ''.join(CHARS_TO_REMOVE.keys())
PROBLEMATIC_CHAR_PATTERN = re.compile(f'[{re.escape(_ALL_PROBLEMATIC)}]')


def clean_text_for_transformers(text: str) -> str:
    """
    Remove hidden/problematic characters that can break transformer tokenization.
    
    Characters removed:
    - Zero-Width Space (ZWSP): \\u200b
    - Zero-Width Non-Joiner (ZWNJ): \\u200c
    - Zero-Width Joiner (ZWJ): \\u200d
    - Word Joiner: \\u2060
    - Soft Hyphen: \\xad
    - BOM (when mid-text): \\ufeff
    
    Args:
        text: Input text that may contain problematic characters
        
    Returns:
        Cleaned text with problematic characters removed
        
    Example:
        >>> clean_text_for_transformers("hello\\u200bworld")
        'helloworld'
    """
    if not text:
        return text
    return PROBLEMATIC_CHAR_PATTERN.sub('', text)


def clean_text_with_stats(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Remove hidden characters and return statistics about what was removed.
    
    Args:
        text: Input text that may contain problematic characters
        
    Returns:
        Tuple of (cleaned_text, dict mapping character names to counts)
        
    Example:
        >>> text, stats = clean_text_with_stats("a\\u200bb\\u200bc")
        >>> text
        'abc'
        >>> stats
        {'Zero-Width Space (ZWSP)': 2}
    """
    if not text:
        return text, {}
    
    removed_counts: Dict[str, int] = {}
    
    for match in PROBLEMATIC_CHAR_PATTERN.finditer(text):
        char = match.group()
        char_name = CHARS_TO_REMOVE.get(char, f'U+{ord(char):04X}')
        removed_counts[char_name] = removed_counts.get(char_name, 0) + 1
    
    cleaned = PROBLEMATIC_CHAR_PATTERN.sub('', text)
    return cleaned, removed_counts


def has_problematic_chars(text: str) -> bool:
    """
    Check if text contains any problematic characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains characters that could break transformer tokenization
    """
    if not text:
        return False
    return bool(PROBLEMATIC_CHAR_PATTERN.search(text))
