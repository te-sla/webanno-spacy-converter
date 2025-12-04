import spacy
from spacy.tokens import DocBin
from pathlib import Path
import sys
import datetime

def validate_doc(doc, doc_index, filename, report_file):
    errors = []
    
    # 1. Check Sentence Boundaries
    if not doc.has_annotation("SENT_START"):
        errors.append(f"  [SENTENCE_ERROR] Doc does not have sentence boundaries (SENT_START).")

    # 2. Check Token Alignment
    # spaCy DocBin usually guarantees this, but we check for corruption/encoding issues
    for token in doc:
        # Calculate expected text from doc.text using character offsets
        expected_text = doc.text[token.idx : token.idx + len(token)]
        if expected_text != token.text:
            errors.append(f"  [TOKEN_ALIGNMENT_ERROR] Token '{token.text}' (idx: {token.idx}) does not match doc text slice '{expected_text}'.")

    # 3. Check Entity Alignment
    for ent in doc.ents:
        # Check if entity text matches doc text slice
        expected_ent_text = doc.text[ent.start_char : ent.end_char]
        if expected_ent_text != ent.text:
            errors.append(f"  [ENTITY_TEXT_ERROR] Entity '{ent.text}' ({ent.label_}) text mismatch. Doc slice: '{expected_ent_text}'.")
        
        # Check alignment with tokens (DocBin stores entities as token indices, so this checks internal consistency)
        start_token = doc[ent.start]
        if ent.start_char != start_token.idx:
             errors.append(f"  [ENTITY_TOKEN_START_ERROR] Entity '{ent.text}' start_char ({ent.start_char}) != start_token.idx ({start_token.idx}).")
        
        end_token = doc[ent.end - 1]
        end_token_end_char = end_token.idx + len(end_token)
        # Note: ent.end_token is the last token in the span. ent.end is the index AFTER the last token.
        # But ent.end_char is character offset of the end.
        # Let's verify the character span covers the tokens exactly.
        # The span includes tokens from ent.start to ent.end - 1.
        
        # Recalculate end char from tokens
        calculated_end_char = end_token.idx + len(end_token)
        if ent.end_char != calculated_end_char:
             errors.append(f"  [ENTITY_TOKEN_END_ERROR] Entity '{ent.text}' end_char ({ent.end_char}) != calculated token end ({calculated_end_char}).")

    # 4. Check Whitespace Rigidity
    # We want to know if the doc preserves newlines and complex whitespace, or if everything was normalized to single spaces.
    # This is critical for Transformers (RoBERTa) which rely on exact whitespace (e.g. newlines).
    
    # Check if the document text contains any newlines
    if "\n" not in doc.text and len(doc) > 10: # Skip very short snippets
        # This is a heuristic: most real documents should have at least one newline (e.g. between paragraphs)
        # If absolutely no newlines are found, it MIGHT indicate that newlines were normalized to spaces.
        # We won't call it an ERROR, but a WARNING.
        pass # We'll just log statistics in the main loop instead of per-doc errors to avoid noise.

    # Check for "suspicious" whitespace reconstruction
    # spaCy's Doc(spaces=...) only supports boolean (space or no space). 
    # If the original text had "\n", and we passed spaces=[True], it becomes " ".
    # To preserve "\n", the newline must be its own token OR we must use a custom tokenizer/reconstruction.
    # Let's check if we have any tokens that ARE newlines, or if we have tokens with custom whitespace.
    
    has_newline_tokens = any("\n" in token.text for token in doc)
    has_newline_whitespace = any("\n" in token.whitespace_ for token in doc)
    
    if not has_newline_tokens and not has_newline_whitespace and len(doc) > 50:
         errors.append(f"  [WHITESPACE_WARNING] Doc has >50 tokens but NO newlines in text or whitespace. Potential normalization?")

    if errors:
        report_file.write(f"FILE: {filename} | DOC ID: {doc_index}\n")
        # Print a snippet of the text for context (first 100 chars)
        snippet = doc.text[:100].replace('\n', ' ') + "..."
        report_file.write(f"CONTEXT: {snippet}\n")
        for err in errors:
            report_file.write(f"{err}\n")
        report_file.write("-" * 40 + "\n")
        return len(errors)
    return 0

def main():
    # Initialize spaCy (blank is fine for loading DocBin, we just need the vocab)
    # We try to load a language if possible, otherwise blank 'xx' (multi-language)
    try:
        nlp = spacy.blank("sr") # Serbian seems to be the target language based on filenames
    except:
        nlp = spacy.blank("xx")

    conversion_dir = Path("conversion")
    report_path = Path("validation_report.txt")
    
    spacy_files = list(conversion_dir.glob("*.spacy"))
    
    if not spacy_files:
        print(f"No .spacy files found in {conversion_dir.absolute()}")
        return

    print(f"Found {len(spacy_files)} .spacy files. Starting validation...")
    
    total_errors = 0
    files_with_errors = 0
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Validation Report - {datetime.datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        
        for spacy_file in spacy_files:
            print(f"Checking {spacy_file.name}...", end="", flush=True)
            try:
                doc_bin = DocBin().from_disk(spacy_file)
                docs = list(doc_bin.get_docs(nlp.vocab))
                
                file_errors = 0
                for i, doc in enumerate(docs):
                    file_errors += validate_doc(doc, i, spacy_file.name, f)
                
                if file_errors > 0:
                    print(f" FOUND {file_errors} ISSUES")
                    files_with_errors += 1
                    total_errors += file_errors
                else:
                    print(" OK")
                    
            except Exception as e:
                print(f" CRASHED: {e}")
                f.write(f"FILE: {spacy_file.name} | CRITICAL ERROR: Failed to load or process file.\n")
                f.write(f"Details: {str(e)}\n")
                f.write("-" * 40 + "\n")
                files_with_errors += 1

    print("\n" + "=" * 60)
    print(f"Validation Complete.")
    print(f"Total Files Checked: {len(spacy_files)}")
    print(f"Files with Errors:   {files_with_errors}")
    print(f"Total Issues Found:  {total_errors}")
    print(f"Detailed report written to: {report_path.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
