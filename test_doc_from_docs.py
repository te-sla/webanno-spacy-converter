"""
Test what happens when spaCy combines multiple Docs via Doc.from_docs().
This is where the whitespace rigidity issue likely occurs.
"""
import spacy
from spacy.tokens import Doc

def test_doc_from_docs_whitespace():
    """Test how Doc.from_docs() handles whitespace between docs."""
    
    print("=" * 70)
    print("Testing Doc.from_docs() whitespace handling")
    print("=" * 70)
    
    nlp = spacy.blank("en")
    
    # Create two simple docs
    doc1 = Doc(nlp.vocab, words=["Hello", "world", "."], spaces=[True, False, False])
    doc2 = Doc(nlp.vocab, words=["How", "are", "you", "?"], spaces=[True, True, False, False])
    
    print(f"\nDoc 1: {repr(doc1.text)}")
    print(f"Doc 2: {repr(doc2.text)}")
    
    # Combine them
    combined = Doc.from_docs([doc1, doc2])
    
    print(f"\nCombined via Doc.from_docs(): {repr(combined.text)}")
    print(f"Tokens: {[t.text for t in combined]}")
    print(f"Spaces: {[t.whitespace_ for t in combined]}")
    
    # What's between the docs?
    print(f"\nNote: Doc.from_docs() adds a SPACE between docs by default.")
    print(f"This is where newlines get lost - they become spaces.")
    
    # Test with separator parameter
    print("\n" + "-" * 70)
    print("Testing with custom separator (if supported)...")
    
    # Check if Doc.from_docs supports separator
    import inspect
    sig = inspect.signature(Doc.from_docs)
    print(f"Doc.from_docs parameters: {list(sig.parameters.keys())}")
    
    # Test: What if we want a newline between docs?
    print("\n" + "-" * 70)
    print("Testing manual newline insertion...")
    
    # Create doc with trailing newline
    doc1_with_nl = Doc(nlp.vocab, words=["Hello", "world", ".", "\n"], spaces=[True, False, False, False])
    doc2 = Doc(nlp.vocab, words=["How", "are", "you", "?"], spaces=[True, True, False, False])
    
    print(f"\nDoc 1 (with \\n token): {repr(doc1_with_nl.text)}")
    print(f"Doc 2: {repr(doc2.text)}")
    
    combined2 = Doc.from_docs([doc1_with_nl, doc2])
    print(f"\nCombined: {repr(combined2.text)}")
    print(f"Tokens: {[t.text for t in combined2]}")
    
    has_newline = '\n' in combined2.text
    print(f"\nNewline preserved: {has_newline}")
    
    if has_newline:
        print("SUCCESS: Adding \\n as a token preserves it through Doc.from_docs()!")
    else:
        print("FAILURE: Newline was lost even as a token.")

def test_docbin_roundtrip():
    """Test if newlines survive DocBin save/load."""
    from spacy.tokens import DocBin
    import tempfile
    import os
    
    print("\n" + "=" * 70)
    print("Testing DocBin save/load roundtrip")
    print("=" * 70)
    
    nlp = spacy.blank("en")
    
    # Create a doc with a newline token
    doc = Doc(nlp.vocab, words=["Hello", "\n", "World"], spaces=[False, False, False])
    
    print(f"\nOriginal doc text: {repr(doc.text)}")
    print(f"Original tokens: {[t.text for t in doc]}")
    
    # Save to DocBin
    doc_bin = DocBin()
    doc_bin.add(doc)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.spacy', delete=False) as f:
        temp_path = f.name
    
    doc_bin.to_disk(temp_path)
    
    # Load back
    loaded_bin = DocBin().from_disk(temp_path)
    loaded_docs = list(loaded_bin.get_docs(nlp.vocab))
    
    os.unlink(temp_path)
    
    if loaded_docs:
        loaded_doc = loaded_docs[0]
        print(f"\nLoaded doc text: {repr(loaded_doc.text)}")
        print(f"Loaded tokens: {[t.text for t in loaded_doc]}")
        
        if loaded_doc.text == doc.text:
            print("\nSUCCESS: Text matches after DocBin roundtrip!")
        else:
            print("\nFAILURE: Text changed after DocBin roundtrip!")
            print(f"  Expected: {repr(doc.text)}")
            print(f"  Got: {repr(loaded_doc.text)}")
    else:
        print("ERROR: No docs loaded from DocBin")

if __name__ == "__main__":
    test_doc_from_docs_whitespace()
    test_docbin_roundtrip()
