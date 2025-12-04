"""
Test specifically for newline handling between sentences.
"""
import spacy
from pathlib import Path

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

def test_multi_sentence_doc():
    """Test what happens when we combine sentences into a single Doc."""
    
    print("=" * 60)
    print("Testing Multi-Sentence Doc (sentences_per_doc=3)")
    print("=" * 60)
    
    tara_file = Path("conversion/wetransfer_alzir-tsv_2025-12-01_1654/Tara.tsv")
    
    if not tara_file.exists():
        print(f"File not found: {tara_file}")
        return
        
    # Parse
    parser = WebAnnoNELParser(str(tara_file))
    sentences = parser.parse()
    
    print(f"Parsed {len(sentences)} sentences")
    
    # Convert with 3 sentences per doc
    nlp = spacy.blank("sr")
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp, 
        sentences_per_doc=3,  # Combine 3 sentences
        ner=True, 
        nel=True
    )
    
    doc_bin = converter.convert(sentences)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"\nCreated {len(docs)} combined docs")
    
    # Check first doc (should have 3 sentences combined)
    if docs:
        doc = docs[0]
        print(f"\n--- First Combined Doc ---")
        print(f"Text (first 200 chars): {repr(doc.text[:200])}")
        print(f"Full text length: {len(doc.text)}")
        print(f"Token count: {len(doc)}")
        
        # Check sentence boundaries
        sent_starts = [i for i, tok in enumerate(doc) if tok.is_sent_start]
        print(f"Sentence start positions (token indices): {sent_starts}")
        
        # Check for newlines
        newline_count = doc.text.count('\n')
        print(f"Newline count in doc text: {newline_count}")
        
        # Check if there are newline tokens
        newline_tokens = [i for i, tok in enumerate(doc) if tok.text == '\n']
        print(f"Newline token positions: {newline_tokens}")
        
        # Show the text with visible markers
        visible_text = doc.text.replace('\n', '⏎\n').replace(' ', '·')
        print(f"\nVisible text (first 300 chars):\n{visible_text[:300]}")

if __name__ == "__main__":
    test_multi_sentence_doc()
