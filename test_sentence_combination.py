"""
Test the full sentence combination flow with the converter.
"""
import spacy
from spacy.tokens import Doc
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence, AnnotationToken
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

def test_sentence_combination():
    """Test that multiple sentences are combined with proper newline separation."""
    
    print("=" * 70)
    print("Testing Multi-Sentence Combination")
    print("=" * 70)
    
    nlp = spacy.blank("en")
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp, 
        sentences_per_doc=3,  # Combine 3 sentences
        ner=True
    )
    
    # Create 3 mock sentences
    sentences = [
        AnnotationSentence(
            text="Hello world.",
            tokens=[
                AnnotationToken(1, 1, "Hello", 0, 5),
                AnnotationToken(1, 2, "world", 6, 11),
                AnnotationToken(1, 3, ".", 11, 12),
            ],
            entities=[]
        ),
        AnnotationSentence(
            text="How are you?",
            tokens=[
                AnnotationToken(2, 1, "How", 0, 3),
                AnnotationToken(2, 2, "are", 4, 7),
                AnnotationToken(2, 3, "you", 8, 11),
                AnnotationToken(2, 4, "?", 11, 12),
            ],
            entities=[]
        ),
        AnnotationSentence(
            text="I am fine.",
            tokens=[
                AnnotationToken(3, 1, "I", 0, 1),
                AnnotationToken(3, 2, "am", 2, 4),
                AnnotationToken(3, 3, "fine", 5, 9),
                AnnotationToken(3, 4, ".", 9, 10),
            ],
            entities=[]
        ),
    ]
    
    # Convert
    doc_bin = converter.convert(sentences)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"\nCreated {len(docs)} doc(s)")
    
    if docs:
        doc = docs[0]
        print(f"\nCombined Doc:")
        print(f"  Text: {repr(doc.text)}")
        print(f"  Tokens: {[t.text for t in doc]}")
        
        # Count newlines
        newline_count = doc.text.count('\n')
        print(f"  Newline count: {newline_count}")
        
        # Check for newline tokens
        newline_tokens = [t.text for t in doc if t.text == '\n']
        print(f"  Newline tokens: {len(newline_tokens)}")
        
        # Expected: newlines between sentences
        expected_newlines = len(sentences)  # Each sentence ends with \n
        
        if newline_count >= len(sentences) - 1:  # At least one \n between each sentence pair
            print(f"\n✓ SUCCESS: Found {newline_count} newlines (expected at least {len(sentences)-1})")
        else:
            print(f"\n✗ FAILURE: Only {newline_count} newlines (expected at least {len(sentences)-1})")
        
        # Verify sentence boundaries are preserved
        sent_starts = [i for i, t in enumerate(doc) if t.is_sent_start]
        print(f"\n  Sentence starts at token indices: {sent_starts}")
        print(f"  Expected {len(sentences)} sentence boundaries")

def test_docbin_roundtrip_with_sentences():
    """Test that combined sentences survive DocBin save/load."""
    from spacy.tokens import DocBin
    import tempfile
    import os
    
    print("\n" + "=" * 70)
    print("Testing DocBin Roundtrip with Combined Sentences")
    print("=" * 70)
    
    nlp = spacy.blank("en")
    converter = AnnotationSentencesToDocBinConverterV2(nlp, sentences_per_doc=2, ner=True)
    
    sentences = [
        AnnotationSentence(
            text="First sentence.",
            tokens=[
                AnnotationToken(1, 1, "First", 0, 5),
                AnnotationToken(1, 2, "sentence", 6, 14),
                AnnotationToken(1, 3, ".", 14, 15),
            ],
            entities=[]
        ),
        AnnotationSentence(
            text="Second sentence.",
            tokens=[
                AnnotationToken(2, 1, "Second", 0, 6),
                AnnotationToken(2, 2, "sentence", 7, 15),
                AnnotationToken(2, 3, ".", 15, 16),
            ],
            entities=[]
        ),
    ]
    
    doc_bin = converter.convert(sentences)
    original_docs = list(doc_bin.get_docs(nlp.vocab))
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.spacy', delete=False) as f:
        temp_path = f.name
    
    doc_bin.to_disk(temp_path)
    
    # Load back
    loaded_bin = DocBin().from_disk(temp_path)
    loaded_docs = list(loaded_bin.get_docs(nlp.vocab))
    
    os.unlink(temp_path)
    
    print(f"\nOriginal doc text: {repr(original_docs[0].text)}")
    print(f"Loaded doc text:   {repr(loaded_docs[0].text)}")
    
    if original_docs[0].text == loaded_docs[0].text:
        print("\n✓ SUCCESS: Text preserved through DocBin roundtrip!")
    else:
        print("\n✗ FAILURE: Text changed after DocBin roundtrip!")
    
    # Check newlines
    has_newline = '\n' in loaded_docs[0].text
    print(f"\nNewlines preserved after roundtrip: {has_newline}")

if __name__ == "__main__":
    test_sentence_combination()
    test_docbin_roundtrip_with_sentences()
