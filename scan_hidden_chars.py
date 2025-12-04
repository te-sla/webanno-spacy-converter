"""
Script to scan TSV files for hidden/special characters that might break Transformers.

These include:
- Zero-width characters (ZWJ, ZWNJ, ZWS, etc.)
- Control characters (except newline/tab)
- BOM (Byte Order Mark)
- Unusual Unicode whitespace
- Right-to-left markers
- Other problematic characters
"""
import os
import unicodedata
from pathlib import Path
from collections import defaultdict

# Characters that are problematic for Transformers
PROBLEMATIC_CHARS = {
    '\ufeff': 'BOM (Byte Order Mark)',
    '\u200b': 'Zero-Width Space (ZWSP)',
    '\u200c': 'Zero-Width Non-Joiner (ZWNJ)',
    '\u200d': 'Zero-Width Joiner (ZWJ)',
    '\u200e': 'Left-to-Right Mark (LRM)',
    '\u200f': 'Right-to-Left Mark (RLM)',
    '\u202a': 'Left-to-Right Embedding',
    '\u202b': 'Right-to-Left Embedding',
    '\u202c': 'Pop Directional Formatting',
    '\u202d': 'Left-to-Right Override',
    '\u202e': 'Right-to-Left Override',
    '\u2060': 'Word Joiner',
    '\u2061': 'Function Application',
    '\u2062': 'Invisible Times',
    '\u2063': 'Invisible Separator',
    '\u2064': 'Invisible Plus',
    '\ufffe': 'Invalid Unicode',
    '\uffff': 'Invalid Unicode',
    '\u00a0': 'Non-Breaking Space (NBSP)',
    '\u00ad': 'Soft Hyphen',
    '\u2000': 'En Quad',
    '\u2001': 'Em Quad',
    '\u2002': 'En Space',
    '\u2003': 'Em Space',
    '\u2004': 'Three-Per-Em Space',
    '\u2005': 'Four-Per-Em Space',
    '\u2006': 'Six-Per-Em Space',
    '\u2007': 'Figure Space',
    '\u2008': 'Punctuation Space',
    '\u2009': 'Thin Space',
    '\u200a': 'Hair Space',
    '\u2028': 'Line Separator',
    '\u2029': 'Paragraph Separator',
    '\u202f': 'Narrow No-Break Space',
    '\u205f': 'Medium Mathematical Space',
    '\u3000': 'Ideographic Space',
}

def is_control_char(char):
    """Check if character is a control character (except common ones)."""
    if char in '\n\r\t':
        return False
    category = unicodedata.category(char)
    return category.startswith('C')  # Control, Format, etc.

def scan_file(filepath):
    """Scan a single file for problematic characters."""
    issues = defaultdict(list)  # char -> list of (line_num, context)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                for i, char in enumerate(line):
                    # Check known problematic chars
                    if char in PROBLEMATIC_CHARS:
                        context = line[max(0, i-10):i+10].replace('\n', '\\n')
                        issues[char].append((line_num, context))
                    
                    # Check other control characters
                    elif is_control_char(char) and char not in '\r':
                        context = line[max(0, i-10):i+10].replace('\n', '\\n')
                        issues[char].append((line_num, context))
                        
    except Exception as e:
        return {'error': str(e)}
    
    return dict(issues)

def scan_folder(folder, extensions=('.tsv',), max_examples=3):
    """Scan all files in folder for problematic characters."""
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        return
    
    files = []
    for ext in extensions:
        files.extend(folder_path.glob(f"*{ext}"))
    
    print(f"Scanning {len(files)} files in {folder}...")
    print("=" * 70)
    
    files_with_issues = 0
    total_issues = defaultdict(int)
    
    for fpath in sorted(files):
        issues = scan_file(fpath)
        
        if 'error' in issues:
            print(f"\n❌ ERROR reading {fpath.name}: {issues['error']}")
            continue
        
        if issues:
            files_with_issues += 1
            print(f"\n📄 {fpath.name}")
            
            for char, occurrences in issues.items():
                char_name = PROBLEMATIC_CHARS.get(char, f"Control char U+{ord(char):04X}")
                total_issues[char] += len(occurrences)
                
                print(f"  ⚠️  {char_name}: {len(occurrences)} occurrence(s)")
                
                # Show first few examples
                for line_num, context in occurrences[:max_examples]:
                    print(f"      Line {line_num}: ...{repr(context)}...")
                
                if len(occurrences) > max_examples:
                    print(f"      ... and {len(occurrences) - max_examples} more")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files scanned: {len(files)}")
    print(f"Files with issues: {files_with_issues}")
    
    if total_issues:
        print(f"\nProblematic characters found:")
        for char, count in sorted(total_issues.items(), key=lambda x: -x[1]):
            char_name = PROBLEMATIC_CHARS.get(char, f"Control char U+{ord(char):04X}")
            print(f"  {char_name}: {count} total")
    else:
        print("\n✅ No problematic characters found!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan TSV files for hidden/problematic characters')
    parser.add_argument('folder', nargs='?', 
                        default='conversion/wetransfer_alzir-tsv_2025-12-01_1654',
                        help='Folder to scan')
    parser.add_argument('--ext', default='.tsv', help='File extension to scan')
    parser.add_argument('--examples', type=int, default=3, help='Max examples per issue')
    
    args = parser.parse_args()
    
    scan_folder(args.folder, extensions=(args.ext,), max_examples=args.examples)

if __name__ == "__main__":
    main()
