#!/usr/bin/env python
"""
Test full pipeline with hidden characters - TSV to spaCy Doc.
Verifies that cleaning doesn't break token alignment.
"""

import spacy
from pathlib import Path
from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import (
    AnnotationSentencesToDocBinConverterV2
)
from webanno_spacy_converter.utils.text_cleaning import (
    clean_text_for_transformers,
    has_problematic_chars
)


def test_pipeline_with_hidden_chars():
    """Test full pipeline: TSV with hidden chars -> spaCy Doc."""
    
    tsv_path = "test_data/hidden_chars_test.tsv"
    
    print("=" * 60)
    print("STEP 1: Parse TSV file")
    print("=" * 60)
    
    parser = WebAnnoNELParser(tsv_path)
    sentences = parser.parse()
    
    print(f"Parsed {len(sentences)} sentences\n")
    
    for i, sent in enumerate(sentences):
        print(f"Sentence {i+1}:")
        print(f"  Text: {repr(sent.text)}")
        print(f"  Has hidden chars: {has_problematic_chars(sent.text)}")
        print(f"  Tokens: {[t.text for t in sent.tokens]}")
        print(f"  Token offsets: {[(t.start, t.end) for t in sent.tokens]}")
        print()
    
    print("=" * 60)
    print("STEP 2: Convert to spaCy (WITH cleaning - V2 default)")
    print("=" * 60)
    
    nlp = spacy.blank("sr")
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp, sentences_per_doc=1, clean_hidden_chars=True
    )
    
    doc_bin = converter.convert(sentences)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"Created {len(docs)} docs\n")
    
    for i, doc in enumerate(docs):
        print(f"Doc {i+1}:")
        print(f"  Text: {repr(doc.text)}")
        print(f"  Has hidden chars: {has_problematic_chars(doc.text)}")
        print(f"  Tokens: {[t.text for t in doc]}")
        
        # Check alignment
        reconstructed = ""
        for j, token in enumerate(doc):
            reconstructed += token.text
            if token.whitespace_:
                reconstructed += token.whitespace_
        
        print(f"  Reconstructed: {repr(reconstructed)}")
        print(f"  Text == Reconstructed: {doc.text == reconstructed}")
        print()
    
    print("=" * 60)
    print("STEP 3: Convert to spaCy (WITHOUT cleaning)")
    print("=" * 60)
    
    converter_no_clean = AnnotationSentencesToDocBinConverterV2(
        nlp, sentences_per_doc=1, clean_hidden_chars=False
    )
    
    # Re-parse since converter may have modified tokens
    parser2 = WebAnnoNELParser(tsv_path)
    sentences = parser2.parse()
    doc_bin_no_clean = converter_no_clean.convert(sentences)
    docs_no_clean = list(doc_bin_no_clean.get_docs(nlp.vocab))
    
    for i, doc in enumerate(docs_no_clean):
        print(f"Doc {i+1}:")
        print(f"  Text: {repr(doc.text)}")
        print(f"  Has hidden chars: {has_problematic_chars(doc.text)}")
        print(f"  Tokens: {[repr(t.text) for t in doc]}")
        print()
    
    print("=" * 60)
    print("STEP 4: Verify alignment after DocBin roundtrip")
    print("=" * 60)
    
    # Save and reload
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".spacy", delete=False) as f:
        temp_path = f.name
    
    # Re-parse and convert
    parser3 = WebAnnoNELParser(tsv_path)
    sentences = parser3.parse()
    doc_bin = converter.convert(sentences)
    doc_bin.to_disk(temp_path)
    
    # Reload
    loaded_bin = spacy.tokens.DocBin().from_disk(temp_path)
    loaded_docs = list(loaded_bin.get_docs(nlp.vocab))
    
    print(f"Loaded {len(loaded_docs)} docs from disk\n")
    
    all_aligned = True
    for i, doc in enumerate(loaded_docs):
        reconstructed = ""
        for token in doc:
            reconstructed += token.text
            if token.whitespace_:
                reconstructed += token.whitespace_
        
        aligned = (doc.text == reconstructed)
        all_aligned = all_aligned and aligned
        
        status = "✅" if aligned else "❌"
        print(f"Doc {i+1}: {status}")
        print(f"  Text:          {repr(doc.text)}")
        print(f"  Reconstructed: {repr(reconstructed)}")
        print()
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("=" * 60)
    if all_aligned:
        print("✅ All docs properly aligned after cleaning and roundtrip!")
    else:
        print("❌ ALIGNMENT ISSUES DETECTED!")
    print("=" * 60)
    
    return all_aligned


if __name__ == "__main__":
    success = test_pipeline_with_hidden_chars()
    exit(0 if success else 1)
