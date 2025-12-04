import spacy
from spacy.tokens import DocBin, Doc

def test_docbin_whitespace_preservation():
    nlp = spacy.blank("en")
    
    # Case 1: Manual Doc creation with custom whitespace
    words = ["Hello", "world"]
    spaces = [True, False] # "Hello " "world"
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    
    # Force a newline into the whitespace of the first token
    # doc[0].whitespace_ = "\n" # This is read-only on Token
    
    # To set custom whitespace, we must create the Doc with it, OR modify the underlying C-level data?
    # Actually, Doc(spaces=...) only takes booleans.
    # But we can create a Doc from words and then merge? No.
    
    # Let's try to see if we can construct it differently.
    # If we use nlp.make_doc("Hello\nworld"), it tokenizes.
    doc = nlp.make_doc("Hello\nworld")
    # If the tokenizer splits on \n, we get 3 tokens: "Hello", "\n", "world".
    # If we want "Hello" to have "\n" as whitespace, we need to see if that's possible.
    
    print(f"Tokenized 'Hello\\nworld': {[t.text for t in doc]}")
    
    # If we want to force "Hello" to have "\n" as whitespace:
    # We can't easily do that with standard Doc constructor if spaces is boolean.
    # However, let's check if DocBin preserves it IF we somehow got it there.
    
    # Let's try to create a doc where we know whitespace is preserved.
    # If we have a token that is just "\n", that's a token, not whitespace.
    
    # What if we use the 'words' and 'spaces' but we want 'spaces' to be non-boolean?
    # The Doc constructor docs say: spaces: Optional[List[bool]].
    
    # So, strictly speaking, spaCy Docs created via words+spaces CANNOT have custom whitespace characters 
    # UNLESS those characters are tokens themselves.
    
    # Let's verify this hypothesis.
    print("Hypothesis: Doc(spaces=...) only supports ' ' or ''.")
    
    doc2 = Doc(nlp.vocab, words=["Hello", "world"], spaces=[True, False])
    print(f"Doc2 text: {repr(doc2.text)}") # 'Hello world'
    
    # So if the original text was "Hello\nworld", and we want 2 tokens "Hello", "world",
    # we CANNOT represent this in a spaCy Doc without "\n" being a separate token.
    
    # Therefore, if the converter is NOT making \n a separate token, it IS normalizing it to space.
    
    pass

if __name__ == "__main__":
    test_docbin_whitespace_preservation()
