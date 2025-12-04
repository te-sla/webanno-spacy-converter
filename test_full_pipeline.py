"""
Full Pipeline Test: TSV -> Parser -> Converter -> spaCy Doc
Tests the entire conversion with a focus on newline preservation.
"""
import spacy
from pathlib import Path

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_pipeline(tsv_path: str):
    """Test the full pipeline from TSV to spaCy Doc."""
    
    print_separator("STEP 1: Parse TSV File")
    
    # Parse the TSV file
    parser = WebAnnoNELParser(tsv_path)
    sentences = parser.parse()
    
    print(f"Parsed {len(sentences)} sentences from {tsv_path}")
    
    for i, sent in enumerate(sentences):
        print(f"\n--- Sentence {i+1} ---")
        print(f"  Text (repr): {repr(sent.text)}")
        has_nl = "\n" in sent.text
        print(f"  Has newline in text: {has_nl}")
        print(f"  Token count: {len(sent.tokens)}")
        print(f"  Entities: {sent.entities}")
        
        # Show tokens with their offsets
        print(f"  Tokens:")
        for t in sent.tokens:
            # Check if the offset range matches the text
            if sent.text and t.start < len(sent.text) and t.end <= len(sent.text):
                text_slice = sent.text[t.start:t.end]
                match = "✓" if text_slice == t.text else f"✗ (got '{text_slice}')"
            else:
                match = "✗ (out of bounds)"
            print(f"    [{t.start:3d}-{t.end:3d}] '{t.text}' {match}")
    
    print_separator("STEP 2: Convert to spaCy Doc")
    
    nlp = spacy.blank("sr")  # Serbian blank model
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp, 
        sentences_per_doc=1,  # 1 sentence per doc for clarity
        ner=True, 
        nel=True
    )
    
    # Convert each sentence individually to see what happens
    for i, sent in enumerate(sentences):
        print(f"\n--- Converting Sentence {i+1} ---")
        print(f"  Input text (repr): {repr(sent.text)}")
        
        try:
            doc = converter._convert_sentence_to_doc(sent)
            
            print(f"  Output Doc text (repr): {repr(doc.text)}")
            print(f"  Token count: {len(doc)}")
            print(f"  Tokens: {[t.text for t in doc]}")
            
            # Check for newline preservation
            has_newline_in_input = "\n" in sent.text
            has_newline_in_output = "\n" in doc.text
            has_newline_token = any(t.text == "\n" for t in doc)
            
            print(f"  Newline in input text: {has_newline_in_input}")
            print(f"  Newline in output text: {has_newline_in_output}")
            print(f"  Newline as token: {has_newline_token}")
            
            # Check text alignment
            if doc.text == sent.text:
                print(f"  Text Match: ✓ EXACT MATCH")
            else:
                print(f"  Text Match: ✗ MISMATCH")
                print(f"    Expected: {repr(sent.text)}")
                print(f"    Got:      {repr(doc.text)}")
            
            # Check entities
            print(f"  Entities in Doc: {[(e.text, e.label_, e.kb_id_) for e in doc.ents]}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print_separator("STEP 3: Full DocBin Conversion")
    
    try:
        doc_bin = converter.convert(sentences)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        print(f"Created DocBin with {len(docs)} docs")
        
        for i, doc in enumerate(docs):
            print(f"\n--- Doc {i+1} ---")
            print(f"  Text (repr): {repr(doc.text)}")
            print(f"  Tokens: {[t.text for t in doc]}")
            has_nl_doc = "\n" in doc.text
            print(f"  Has newline: {has_nl_doc}")
            print(f"  Entities: {[(e.text, e.label_) for e in doc.ents]}")
            
    except Exception as e:
        print(f"ERROR in DocBin conversion: {e}")
        import traceback
        traceback.print_exc()
    
    print_separator("SUMMARY")
    print("Check the output above to verify:")
    print("1. Parser correctly extracts text with newlines")
    print("2. Token offsets align with the text")
    print("3. Converter preserves newlines (either in text or as tokens)")
    print("4. Entities are correctly aligned")

if __name__ == "__main__":
    # Test with our custom newline TSV
    test_file = Path("test_data/test_newline.tsv")
    
    if test_file.exists():
        test_pipeline(str(test_file))
    else:
        print(f"Test file not found: {test_file}")
        
    # Also test with the Tara.tsv file which has real data
    print("\n\n")
    print("#" * 80)
    print("# TESTING WITH REAL DATA: Tara.tsv")
    print("#" * 80)
    
    tara_file = Path("conversion/wetransfer_alzir-tsv_2025-12-01_1654/Tara.tsv")
    if tara_file.exists():
        test_pipeline(str(tara_file))
    else:
        print(f"Tara.tsv not found at: {tara_file}")
