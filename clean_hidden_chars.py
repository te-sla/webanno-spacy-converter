#!/usr/bin/env python
"""
Clean hidden/problematic characters from TSV files.

These characters can break transformer models by causing unexpected tokenization:
- Zero-Width Space (ZWSP): \u200b
- Zero-Width Non-Joiner (ZWNJ): \u200c  
- Zero-Width Joiner (ZWJ): \u200d
- BOM (Byte Order Mark): \ufeff (when mid-text, not at file start)
- Soft Hyphen: \xad
- Word Joiner: \u2060
- Zero-Width No-Break Space: \ufeff

Usage:
    python clean_hidden_chars.py                    # Dry run (show what would be cleaned)
    python clean_hidden_chars.py --apply            # Actually clean the files
    python clean_hidden_chars.py --file path.tsv   # Clean specific file
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Characters to remove completely (invisible, no semantic meaning)
CHARS_TO_REMOVE = {
    '\u200b': 'Zero-Width Space (ZWSP)',
    '\u200c': 'Zero-Width Non-Joiner (ZWNJ)',
    '\u200d': 'Zero-Width Joiner (ZWJ)',
    '\u2060': 'Word Joiner',
    '\xad': 'Soft Hyphen',
}

# Characters to remove only when mid-text (BOM at start of file is OK)
CHARS_TO_REMOVE_MIDTEXT = {
    '\ufeff': 'BOM (Byte Order Mark)',
}

# Build regex pattern for all removable characters
ALL_PROBLEMATIC = ''.join(CHARS_TO_REMOVE.keys()) + ''.join(CHARS_TO_REMOVE_MIDTEXT.keys())
PROBLEMATIC_PATTERN = re.compile(f'[{re.escape(ALL_PROBLEMATIC)}]')


def get_char_name(char: str) -> str:
    """Get human-readable name for a character."""
    if char in CHARS_TO_REMOVE:
        return CHARS_TO_REMOVE[char]
    if char in CHARS_TO_REMOVE_MIDTEXT:
        return CHARS_TO_REMOVE_MIDTEXT[char]
    return f'U+{ord(char):04X}'


def clean_content(content: str, preserve_file_bom: bool = True) -> Tuple[str, Dict[str, int]]:
    """
    Clean problematic characters from content.
    
    Args:
        content: The text content to clean
        preserve_file_bom: If True, preserve BOM at very start of file
        
    Returns:
        Tuple of (cleaned_content, dict of removed char counts)
    """
    removed_counts: Dict[str, int] = {}
    
    # Check for BOM at start of file
    has_leading_bom = content.startswith('\ufeff')
    
    # Count what we're removing
    for match in PROBLEMATIC_PATTERN.finditer(content):
        char = match.group()
        char_name = get_char_name(char)
        
        # Skip BOM at position 0 if we're preserving it
        if char == '\ufeff' and match.start() == 0 and preserve_file_bom:
            continue
            
        removed_counts[char_name] = removed_counts.get(char_name, 0) + 1
    
    # Clean the content
    if preserve_file_bom and has_leading_bom:
        # Remove BOM, clean, then restore
        cleaned = PROBLEMATIC_PATTERN.sub('', content[1:])
        cleaned = '\ufeff' + cleaned
    else:
        cleaned = PROBLEMATIC_PATTERN.sub('', content)
    
    return cleaned, removed_counts


def process_file(filepath: Path, dry_run: bool = True) -> Tuple[bool, Dict[str, int]]:
    """
    Process a single file to clean hidden characters.
    
    Args:
        filepath: Path to the file
        dry_run: If True, don't actually modify the file
        
    Returns:
        Tuple of (was_modified, removed_counts)
    """
    try:
        # Try UTF-8 first, fall back to UTF-8 with BOM
        try:
            content = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = filepath.read_text(encoding='utf-8-sig')
        
        cleaned, removed_counts = clean_content(content)
        
        if removed_counts and not dry_run:
            # Write back with same encoding
            filepath.write_text(cleaned, encoding='utf-8')
        
        return bool(removed_counts), removed_counts
        
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")
        return False, {}


def scan_directory(directory: Path, dry_run: bool = True) -> None:
    """Scan and optionally clean all TSV files in directory."""
    
    tsv_files = sorted(directory.rglob("*.tsv"))
    
    if not tsv_files:
        print(f"No TSV files found in {directory}")
        return
    
    mode = "DRY RUN" if dry_run else "CLEANING"
    print(f"\n{mode}: Scanning {len(tsv_files)} files in {directory}...")
    print("=" * 70)
    
    files_with_issues = 0
    total_removed: Dict[str, int] = {}
    
    for filepath in tsv_files:
        was_modified, removed_counts = process_file(filepath, dry_run)
        
        if removed_counts:
            files_with_issues += 1
            print(f"\n📄 {filepath.name}")
            
            for char_name, count in sorted(removed_counts.items()):
                action = "would remove" if dry_run else "removed"
                print(f"  {'🔍' if dry_run else '✅'} {action} {count}x {char_name}")
                total_removed[char_name] = total_removed.get(char_name, 0) + count
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files scanned: {len(tsv_files)}")
    print(f"Files with issues: {files_with_issues}")
    
    if total_removed:
        print(f"\nCharacters {'to remove' if dry_run else 'removed'}:")
        for char_name, count in sorted(total_removed.items(), key=lambda x: -x[1]):
            print(f"  {char_name}: {count}")
        
        if dry_run:
            print(f"\n⚠️  Run with --apply to actually clean the files")
    else:
        print("\n✅ No problematic characters found!")


def main():
    parser = argparse.ArgumentParser(
        description="Clean hidden/problematic characters from TSV files"
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually modify files (default is dry run)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Clean a specific file instead of the default directory'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='conversion/wetransfer_alzir-tsv_2025-12-01_1654',
        help='Directory to scan (default: conversion/wetransfer_alzir-tsv_2025-12-01_1654)'
    )
    
    args = parser.parse_args()
    
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)
        
        mode = "CLEANING" if args.apply else "DRY RUN"
        print(f"\n{mode}: {filepath}")
        
        was_modified, removed_counts = process_file(filepath, dry_run=not args.apply)
        
        if removed_counts:
            for char_name, count in sorted(removed_counts.items()):
                action = "would remove" if not args.apply else "removed"
                print(f"  {'🔍' if not args.apply else '✅'} {action} {count}x {char_name}")
        else:
            print("  ✅ No problematic characters found")
    else:
        directory = Path(args.dir)
        if not directory.exists():
            print(f"Directory not found: {directory}")
            sys.exit(1)
        
        scan_directory(directory, dry_run=not args.apply)


if __name__ == '__main__':
    main()
