#!/usr/bin/env python
"""
Check alignment exactly how spaCy's debug data command does it.
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from pathlib import Path
import sys


def check_spacy_alignment(spacy_file: str):
    """Check alignment the same way spacy debug data does."""
    
    nlp = spacy.blank('sr')
    
    print(f"\nAnalyzing: {spacy_file}")
    print("=" * 70)
    
    doc_bin = DocBin().from_disk(spacy_file)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"Loaded {len(docs)} docs")
    
    misaligned_count = 0
    docs_with_issues = set()
    
    for i, gold_doc in enumerate(docs):
        # This is exactly what spacy does:
        # It creates a new doc from the text, then compares alignment
        predicted_doc = nlp.make_doc(gold_doc.text)
        eg = Example(predicted_doc, gold_doc)
        
        # Check alignment for each token in gold/reference doc
        for j in range(len(eg.reference)):
            ref_token = eg.reference[j]
            # Get aligned token indices in predicted doc
            aligned = eg.alignment.y2x[j]
            
            if len(aligned) == 0:
                # Token not aligned - this is what spacy counts as misaligned
                misaligned_count += 1
                docs_with_issues.add(i)
                
                if misaligned_count <= 20:
                    print(f"\nDoc #{i}: Token #{j} not aligned")
                    print(f"  Token text: {repr(ref_token.text)}")
                    print(f"  Token whitespace: {repr(ref_token.whitespace_)}")
                    print(f"  Char position: {ref_token.idx}")
                    print(f"  Gold doc text (around token): {repr(gold_doc.text[max(0, ref_token.idx-20):ref_token.idx+len(ref_token.text)+20])}")
                    
                    # Show predicted tokenization around this area
                    pred_tokens_near = [t for t in predicted_doc if abs(t.idx - ref_token.idx) < 30]
                    print(f"  Predicted tokens nearby: {[(t.text, t.idx) for t in pred_tokens_near]}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total docs: {len(docs)}")
    print(f"Docs with issues: {len(docs_with_issues)}")
    print(f"Total misaligned tokens: {misaligned_count}")
    
    if misaligned_count > 0:
        print(f"\n⚠️  These are the tokens spaCy's debug data reports as misaligned!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            check_spacy_alignment(f)
    else:
        check_spacy_alignment("conversion/trsic3-train.spacy")
