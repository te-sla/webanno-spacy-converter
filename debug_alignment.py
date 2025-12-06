#!/usr/bin/env python
"""
Find misaligned tokens using spaCy's own alignment check.
This checks entity spans against token boundaries.
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from pathlib import Path
import sys


def check_entity_alignment(doc, doc_idx: int) -> list:
    """Check if entities are properly aligned to token boundaries."""
    issues = []
    
    for ent in doc.ents:
        # Check if entity start aligns with a token start
        start_aligned = any(token.idx == ent.start_char for token in doc)
        # Check if entity end aligns with a token end
        end_aligned = any(token.idx + len(token.text) == ent.end_char for token in doc)
        
        if not start_aligned or not end_aligned:
            # Find nearest tokens
            start_tokens = [(t.idx, t.text) for t in doc if abs(t.idx - ent.start_char) <= 5]
            end_tokens = [(t.idx + len(t.text), t.text) for t in doc if abs(t.idx + len(t.text) - ent.end_char) <= 5]
            
            issues.append({
                'doc_idx': doc_idx,
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'start_aligned': start_aligned,
                'end_aligned': end_aligned,
                'nearby_start_tokens': start_tokens,
                'nearby_end_tokens': end_tokens,
                'doc_text_around': repr(doc.text[max(0, ent.start_char-20):ent.end_char+20])
            })
    
    return issues


def check_whitespace_alignment(doc, doc_idx: int) -> list:
    """
    Check whitespace alignment - this is what spaCy's debug data actually checks.
    A token is misaligned if doc.text[token.idx:token.idx+len(token)] != token.text
    """
    issues = []
    
    for i, token in enumerate(doc):
        # This is the exact check spaCy uses
        if doc.text[token.idx : token.idx + len(token.text)] != token.text:
            issues.append({
                'doc_idx': doc_idx,
                'token_idx': i,
                'token_text': repr(token.text),
                'char_start': token.idx,
                'char_end': token.idx + len(token.text),
                'expected': repr(token.text),
                'actual': repr(doc.text[token.idx : token.idx + len(token.text)]),
                'context': repr(doc.text[max(0, token.idx-15):token.idx + len(token.text) + 15])
            })
    
    return issues


def analyze_file(spacy_file: str, nlp=None):
    """Analyze a DocBin file for alignment issues."""
    
    path = Path(spacy_file)
    if not path.exists():
        print(f"File not found: {spacy_file}")
        return
    
    if nlp is None:
        nlp = spacy.blank("sr")
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {spacy_file}")
    print('='*70)
    
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"Total documents: {len(docs)}")
    
    # Check whitespace alignment
    print("\n--- Whitespace Alignment Check ---")
    ws_issues = []
    docs_with_ws_issues = 0
    
    for doc_idx, doc in enumerate(docs):
        issues = check_whitespace_alignment(doc, doc_idx)
        if issues:
            docs_with_ws_issues += 1
            ws_issues.extend(issues)
            
            if docs_with_ws_issues <= 5:
                print(f"\n📄 Doc #{doc_idx}:")
                print(f"   Text (first 80): {repr(doc.text[:80])}...")
                for issue in issues[:2]:
                    print(f"   Token #{issue['token_idx']}: {issue['token_text']}")
                    print(f"      Chars {issue['char_start']}-{issue['char_end']}")
                    print(f"      Expected: {issue['expected']}")
                    print(f"      Actual:   {issue['actual']}")
                    print(f"      Context:  {issue['context']}")
    
    print(f"\nWhitespace misaligned tokens: {len(ws_issues)} in {docs_with_ws_issues} docs")
    
    # Check entity alignment
    print("\n--- Entity Alignment Check ---")
    ent_issues = []
    docs_with_ent_issues = 0
    
    for doc_idx, doc in enumerate(docs):
        issues = check_entity_alignment(doc, doc_idx)
        if issues:
            docs_with_ent_issues += 1
            ent_issues.extend(issues)
            
            if docs_with_ent_issues <= 5:
                print(f"\n📄 Doc #{doc_idx}:")
                for issue in issues[:2]:
                    print(f"   Entity: '{issue['entity_text']}' ({issue['entity_label']})")
                    print(f"      Chars {issue['start_char']}-{issue['end_char']}")
                    print(f"      Start aligned: {issue['start_aligned']}, End aligned: {issue['end_aligned']}")
                    print(f"      Context: {issue['doc_text_around']}")
    
    print(f"\nEntity alignment issues: {len(ent_issues)} in {docs_with_ent_issues} docs")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total documents: {len(docs)}")
    print(f"Whitespace misaligned tokens: {len(ws_issues)}")
    print(f"Entity alignment issues: {len(ent_issues)}")
    
    return ws_issues, ent_issues


def main():
    if len(sys.argv) < 2:
        files = [
            "conversion/trsic3-train.spacy",
            "conversion/trsic3-dev.spacy",
        ]
    else:
        files = sys.argv[1:]
    
    nlp = spacy.blank("sr")
    
    for f in files:
        if Path(f).exists():
            analyze_file(f, nlp)
        else:
            print(f"File not found: {f}")


if __name__ == "__main__":
    main()
