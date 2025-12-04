# import pytest
import spacy
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

def test_conversion_debug_sample():
    # Path to the sample file
    base_path = Path(__file__).parent.parent
    input_file = base_path / "test_data" / "debug_sample.tsv"
    
    assert input_file.exists(), f"Sample file not found at {input_file}"

    # 1. Parse
    parser = WebAnnoNELParser(str(input_file))
    sentences = parser.parse()
    
    assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}"

    # 2. Convert
    try:
        nlp = spacy.blank("sr")
    except ImportError:
        nlp = spacy.blank("xx")

    converter = AnnotationSentencesToDocBinConverterV2(
        nlp, 
        sentences_per_doc=10,
        ner=True,
        nel=True
    )
    doc_bin = converter.convert(sentences)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    doc = docs[0]

    # 3. Verify Entities
    # Expected: SRPSKOGA (DEMO), srpskoga (DEMO), Baji (LOC, Q203344)
    ents = doc.ents
    assert len(ents) == 3, f"Expected 3 entities, got {len(ents)}"
    
    ent_texts = [e.text for e in ents]
    assert "SRPSKOGA" in ent_texts
    assert "Baji" in ent_texts
    
    # Check specific entity attributes
    baji = [e for e in ents if e.text == "Baji"][0]
    assert baji.label_ == "LOC"
    assert baji.kb_id_ == "Q203344"

if __name__ == "__main__":
    test_conversion_debug_sample()
    print("Test passed!")
