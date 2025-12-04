"""
Script to find TSV files that contain multi-line #Text= blocks.
These are sentences where the text spans multiple lines (contains actual newlines).
"""
import os
from pathlib import Path

def find_multiline_text_blocks(folder: str, max_files: int = None):
    """Search for TSV files with multi-line #Text= blocks."""
    
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        return []
    
    tsv_files = list(folder_path.glob("*.tsv"))
    print(f"Scanning {len(tsv_files)} TSV files in {folder}...")
    
    results = []
    
    for idx, fpath in enumerate(tsv_files):
        if max_files and idx >= max_files:
            break
            
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {fpath.name}: {e}")
            continue
        
        # Look for #Text= followed by a non-token line
        for i in range(len(lines) - 1):
            curr = lines[i].rstrip('\r\n')
            next_line = lines[i + 1].rstrip('\r\n')
            
            if curr.startswith('#Text='):
                # Check if next line is a continuation (not a token line, not empty, not a comment)
                if next_line.strip():  # Not empty
                    # Token lines start with "digit-digit\t"
                    is_token_line = len(next_line) > 3 and next_line[0].isdigit() and '-' in next_line[:6] and '\t' in next_line
                    is_comment = next_line.startswith('#')
                    
                    if not is_token_line and not is_comment:
                        results.append({
                            'file': fpath.name,
                            'line': i + 1,
                            'text_start': curr[:80],
                            'continuation': next_line[:80],
                        })
                        break  # Found one in this file, move to next
    
    return results

def main():
    folder = "conversion/wetransfer_alzir-tsv_2025-12-01_1654"
    
    print("=" * 70)
    print("Searching for TSV files with multi-line #Text= blocks")
    print("=" * 70)
    
    results = find_multiline_text_blocks(folder)
    
    if results:
        print(f"\nFound {len(results)} files with multi-line text blocks:\n")
        for r in results[:20]:  # Show first 20
            print(f"FILE: {r['file']}")
            print(f"  Line {r['line']}: {r['text_start']}...")
            print(f"  Continues: {r['continuation']}...")
            print()
    else:
        print("\nNo multi-line text blocks found.")
        
    # Always check for newlines in #Text= content
    print("\n" + "=" * 70)
    print("Checking for newlines/special chars in #Text= content...")
    print("=" * 70)
    check_text_content_for_newlines(folder)

def check_text_content_for_newlines(folder: str):
    """Check if any #Text= line contains escaped newlines or special chars."""
    from pathlib import Path
    
    folder_path = Path(folder)
    tsv_files = list(folder_path.glob("*.tsv"))
    
    found_any = False
    
    for fpath in tsv_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('#Text='):
                        text = line[6:].rstrip('\r\n')
                        # Check for various newline representations
                        if '\\n' in text or '\n' in text or '&#10;' in text or '\r' in text:
                            print(f"FILE: {fpath.name} line {line_num}")
                            print(f"  Found special chars in: {repr(text[:100])}")
                            print()
                            found_any = True
                            break
        except Exception as e:
            continue
    
    if not found_any:
        print("No newlines found in any #Text= content.")
        print("\nConclusion: WebAnno TSV in this corpus does NOT have multi-line sentences.")
        print("The 'Whitespace Rigidity' issue is about the GAP BETWEEN sentences when")
        print("they are combined via Doc.from_docs(), not within sentences.")

def check_offset_gaps(folder: str):
    """Check for newlines by analyzing token offset gaps."""
    from pathlib import Path
    
    folder_path = Path(folder)
    tsv_files = list(folder_path.glob("*.tsv"))[:10]  # Check first 10 files
    
    for fpath in tsv_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            continue
        
        lines = content.split('\n')
        current_text = None
        prev_end = None
        
        for line in lines:
            if line.startswith('#Text='):
                current_text = line[6:]  # Text after #Text=
                prev_end = None
            elif line.strip() and line[0].isdigit() and '\t' in line:
                # Token line
                parts = line.split('\t')
                if len(parts) >= 3:
                    pos = parts[1]  # e.g., "0-5"
                    if '-' in pos:
                        try:
                            start, end = map(int, pos.split('-'))
                            
                            # Check gap from previous token
                            if prev_end is not None and current_text:
                                gap_size = start - prev_end
                                if gap_size > 1:  # More than just a single space
                                    gap_content = current_text[prev_end:start] if prev_end < len(current_text) and start <= len(current_text) else ""
                                    if '\n' in gap_content or gap_size > 5:
                                        print(f"FILE: {fpath.name}")
                                        print(f"  Gap of {gap_size} chars between tokens")
                                        print(f"  Gap content: {repr(gap_content[:50])}")
                                        print()
                                        break
                            
                            prev_end = end
                        except ValueError:
                            pass

if __name__ == "__main__":
    main()
