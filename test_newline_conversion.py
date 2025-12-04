import spacy
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence, AnnotationToken
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

def test_newline_preservation():
    nlp = spacy.blank("en")
    converter = AnnotationSentencesToDocBinConverterV2(nlp, ner=True)
    
    # Create a mock sentence with a newline
    # Text: "Hello\nWorld"
    # Tokens: "Hello" (0-5), "World" (6-11)
    # The gap is "\n" (index 5-6)
    
    text = "Hello\nWorld"
    tokens = [
        AnnotationToken(1, 1, "Hello", 0, 5),
        AnnotationToken(1, 2, "World", 6, 11)
    ]
    
    sent = AnnotationSentence(text, tokens)
    
    # Convert
    doc = converter._convert_sentence_to_doc(sent)
    
    print(f"Original Text: {repr(text)}")
    print(f"Doc Text:      {repr(doc.text)}")
    print(f"Tokens:        {[t.text for t in doc]}")
    
    # Check for newline token
    has_newline_token = any(t.text == "\n" for t in doc)
    
    if has_newline_token:
        print("SUCCESS: Newline token found!")
    else:
        print("FAILURE: Newline token NOT found.")
        
    # Check alignment
    if doc.text == text:
        print("SUCCESS: Text reconstruction matches exactly.")
    else:
        print("FAILURE: Text reconstruction mismatch.")

if __name__ == "__main__":
    test_newline_preservation()
