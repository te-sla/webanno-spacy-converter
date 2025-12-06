#!/usr/bin/env python
"""
Find misaligned tokens in spaCy DocBin files.
Shows exactly which documents and tokens have alignment issues.
"""

import spacy
from spacy.tokens import DocBin
from pathlib import Path
import sys


def check_alignment(doc, doc_idx: int) -> list:
    """Check if all tokens in a doc are properly aligned."""
    issues = []
    
    for i, token in enumerate(doc):
        # Reconstruct what the text should be based on token + whitespace
        expected_text = token.text
        if token.whitespace_:
            expected_text += token.whitespace_
        
        # Get actual text from doc at token's character positions
        actual_text = doc.text[token.idx:token.idx + len(token.text)]
        
        if actual_text != token.text:
            issues.append({
                'doc_idx': doc_idx,
                'token_idx': i,
                'token_text': repr(token.text),
                'token_idx_char': token.idx,
                'expected': repr(token.text),
                'actual': repr(actual_text),
                'context': repr(doc.text[max(0, token.idx-10):token.idx+len(token.text)+10])
            })
    
    return issues


def find_misaligned_tokens(spacy_file: str, nlp=None):
    """Find all misaligned tokens in a DocBin file."""
    
    path = Path(spacy_file)
    if not path.exists():
        print(f"File not found: {spacy_file}")
        return
    
    if nlp is None:
        nlp = spacy.blank("sr")
    
    print(f"\nAnalyzing: {spacy_file}")
    print("=" * 70)
    
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"Total documents: {len(docs)}")
    
    all_issues = []
    docs_with_issues = 0
    
    for doc_idx, doc in enumerate(docs):
        issues = check_alignment(doc, doc_idx)
        if issues:
            docs_with_issues += 1
            all_issues.extend(issues)
            
            # Show first few issues per doc
            if docs_with_issues <= 10:  # Limit output
                print(f"\n📄 Doc #{doc_idx} has {len(issues)} misaligned token(s):")
                print(f"   Doc text (first 100 chars): {repr(doc.text[:100])}...")
                
                for issue in issues[:3]:  # Show max 3 per doc
                    print(f"   Token #{issue['token_idx']}: {issue['token_text']}")
                    print(f"      At char {issue['token_idx_char']}")
                    print(f"      Expected: {issue['expected']}")
                    print(f"      Actual:   {issue['actual']}")
                    print(f"      Context:  {issue['context']}")
                
                if len(issues) > 3:
                    print(f"   ... and {len(issues) - 3} more issues in this doc")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total documents: {len(docs)}")
    print(f"Documents with issues: {docs_with_issues}")
    print(f"Total misaligned tokens: {len(all_issues)}")
    
    if all_issues:
        print(f"\n⚠️  {len(all_issues)} misaligned tokens found!")
        
        # Analyze patterns
        print("\nAnalyzing patterns...")
        
        # Check for common patterns
        whitespace_issues = 0
        newline_issues = 0
        empty_issues = 0
        
        for issue in all_issues:
            actual = issue['actual']
            token = issue['token_text']
            
            if '\\n' in actual or '\n' in actual:
                newline_issues += 1
            elif actual.strip() == '' or actual == "''":
                empty_issues += 1
            elif ' ' in actual or '\\t' in actual:
                whitespace_issues += 1
        
        if newline_issues:
            print(f"  - Newline-related: {newline_issues}")
        if whitespace_issues:
            print(f"  - Whitespace-related: {whitespace_issues}")
        if empty_issues:
            print(f"  - Empty/missing text: {empty_issues}")
    else:
        print("\n✅ No misaligned tokens found!")
    
    return all_issues


def main():
    if len(sys.argv) < 2:
        # Default: check the files mentioned in the user's command
        files = [
            "corpora/trsic3-train.spacy",
            "corpora/trsic3-dev.spacy",
        ]
        # Also try local conversion folder
        local_files = list(Path("conversion").glob("*.spacy"))
        if local_files:
            files = [str(f) for f in local_files[:2]]
    else:
        files = sys.argv[1:]
    
    nlp = spacy.blank("sr")
    
    for f in files:
        if Path(f).exists():
            find_misaligned_tokens(f, nlp)
        else:
            print(f"File not found: {f}")


if __name__ == "__main__":
    main()
