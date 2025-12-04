#!/usr/bin/env python
"""
Test that cleaning handles misaligned offsets caused by hidden chars.
This simulates the REAL problem: TSV offsets that don't account for hidden chars.
"""

import spacy
from pathlib import Path
from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import (
    AnnotationSentencesToDocBinConverterV2
)
from webanno_spacy_converter.utils.text_cleaning import has_problematic_chars


def test_alignment_with_bad_offsets():
    """Test that cleaning handles misaligned offsets."""
    
    tsv_path = "test_data/hidden_chars_bad_offsets.tsv"
    
    print("=" * 60)
    print("Testing alignment with MISALIGNED offsets (hidden chars not in offsets)")
    print("=" * 60)
    
    parser = WebAnnoNELParser(tsv_path)
    sentences = parser.parse()
    
    print(f"\nParsed {len(sentences)} sentences")
    
    for i, sent in enumerate(sentences):
        print(f"\nSentence {i+1}:")
        print(f"  Raw text: {repr(sent.text)}")
        print(f"  Has hidden: {has_problematic_chars(sent.text)}")
        print(f"  Tokens:")
        for t in sent.tokens:
            # Check if token text matches what's at the offset
            actual = sent.text[t.start:t.end] if t.end <= len(sent.text) else "OUT OF BOUNDS"
            match = "✅" if actual == t.text else f"❌ actual={repr(actual)}"
            print(f"    {t.start}-{t.end}: {repr(t.text)} {match}")
    
    print("\n" + "=" * 60)
    print("Converting WITH cleaning enabled...")
    print("=" * 60)
    
    nlp = spacy.blank("sr")
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp, sentences_per_doc=1, clean_hidden_chars=True
    )
    
    doc_bin = converter.convert(sentences)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    all_aligned = True
    for i, doc in enumerate(docs):
        reconstructed = "".join(t.text + t.whitespace_ for t in doc)
        aligned = (doc.text == reconstructed)
        all_aligned = all_aligned and aligned
        
        status = "✅" if aligned else "❌"
        print(f"\nDoc {i+1}: {status}")
        print(f"  Text:          {repr(doc.text)}")
        print(f"  Reconstructed: {repr(reconstructed)}")
        print(f"  Tokens: {[t.text for t in doc]}")
    
    print("\n" + "=" * 60)
    if all_aligned:
        print("✅ Alignment preserved even with misaligned source offsets!")
    else:
        print("❌ ALIGNMENT BROKEN - cleaning doesn't fix offset misalignment")
    print("=" * 60)
    
    return all_aligned


if __name__ == "__main__":
    success = test_alignment_with_bad_offsets()
    exit(0 if success else 1)
