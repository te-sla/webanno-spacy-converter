#!/usr/bin/env python
"""
Find misaligned tokens using spaCy's actual debug data logic with transformer tokenizer.
"""

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import load_model_from_config, load_config
from pathlib import Path
import sys


def find_misaligned_with_config(config_path: str, train_path: str, dev_path: str):
    """Find misaligned tokens using the actual config (with transformer)."""
    
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    
    # Override paths
    cfg["paths"]["train"] = train_path
    cfg["paths"]["dev"] = dev_path
    
    print("Initializing pipeline from config...")
    nlp = load_model_from_config(cfg, auto_fill=True)
    
    print(f"Pipeline components: {nlp.pipe_names}")
    
    for split_name, split_path in [("train", train_path), ("dev", dev_path)]:
        print(f"\n{'='*70}")
        print(f"Analyzing {split_name}: {split_path}")
        print('='*70)
        
        doc_bin = DocBin().from_disk(split_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        print(f"Total documents: {len(docs)}")
        
        misaligned_count = 0
        docs_with_issues = 0
        
        for doc_idx, gold_doc in enumerate(docs):
            # Re-tokenize with the transformer's tokenizer
            pred_doc = nlp.make_doc(gold_doc.text)
            eg = Example(pred_doc, gold_doc)
            
            doc_issues = []
            # This is the EXACT check spaCy uses (line 829 of debug_data.py):
            # It iterates over predicted tokens and checks x2y alignment
            align = eg.alignment
            for token in pred_doc:
                if token.orth_.isspace():
                    continue
                if align.x2y.lengths[token.i] != 1:
                    doc_issues.append({
                        'token_idx': token.i,
                        'token_text': token.text,
                        'token_ws': token.whitespace_,
                        'char_idx': token.idx,
                        'alignment_length': align.x2y.lengths[token.i]
                    })
                    misaligned_count += 1
            
            if doc_issues:
                docs_with_issues += 1
                
                if docs_with_issues <= 10:  # Show first 10 docs with issues
                    print(f"\n📄 Doc #{doc_idx} ({len(doc_issues)} misaligned tokens):")
                    print(f"   Gold text (first 100): {repr(gold_doc.text[:100])}...")
                    print(f"   Gold tokens: {[(t.text, t.idx) for t in gold_doc[:15]]}...")
                    print(f"   Pred tokens: {[(t.text, t.idx) for t in pred_doc[:15]]}...")
                    
                    for issue in doc_issues[:5]:
                        print(f"   ❌ Pred Token #{issue['token_idx']}: {repr(issue['token_text'])} at char {issue['char_idx']}")
                        print(f"      alignment_length: {issue['alignment_length']} (should be 1)")
                        # Show context
                        ctx_start = max(0, issue['char_idx'] - 15)
                        ctx_end = min(len(gold_doc.text), issue['char_idx'] + len(issue['token_text']) + 15)
                        print(f"      context: {repr(gold_doc.text[ctx_start:ctx_end])}")
                    
                    if len(doc_issues) > 5:
                        print(f"   ... and {len(doc_issues) - 5} more")
        
        print(f"\n{'='*70}")
        print(f"SUMMARY for {split_name}:")
        print(f"  Total docs: {len(docs)}")
        print(f"  Docs with issues: {docs_with_issues}")
        print(f"  Total misaligned tokens: {misaligned_count}")
        
        if misaligned_count > 0:
            print(f"\n⚠️  {misaligned_count} misaligned tokens (this matches spaCy's debug data warning)")


def main():
    config_path = "configs/config_transformer_trsis1.cfg"
    train_path = "conversion/trsic3-train.spacy"
    dev_path = "conversion/trsic3-dev.spacy"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        train_path = sys.argv[2]
    if len(sys.argv) > 3:
        dev_path = sys.argv[3]
    
    find_misaligned_with_config(config_path, train_path, dev_path)


if __name__ == "__main__":
    main()
