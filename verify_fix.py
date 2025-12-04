import spacy
from spacy.tokens import DocBin
from pathlib import Path

def verify_fix():
    spacy_path = Path("test_data/sr_elexis_debug.spacy")
    nlp = spacy.blank("sr")
    
    doc_bin = DocBin().from_disk(spacy_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    print(f"Loaded {len(docs)} docs.")
    
    newline_tokens = 0
    newline_in_text = 0
    
    for doc in docs:
        if "\n" in doc.text:
            newline_in_text += 1
        
        for token in doc:
            if "\n" in token.text:
                newline_tokens += 1
                
    print(f"Docs with newlines in text: {newline_in_text}")
    print(f"Total newline tokens found: {newline_tokens}")
    
    if newline_tokens > 0:
        print("SUCCESS: Newlines are now preserved as tokens!")
    else:
        print("FAILURE: No newline tokens found. (Maybe the input TSV didn't have newlines?)")

if __name__ == "__main__":
    verify_fix()
