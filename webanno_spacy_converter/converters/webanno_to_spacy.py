import logging
from typing import List, Optional
from spacy.tokens import DocBin, Doc, Span
from spacy.util import filter_spans
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.utils.text_cleaning import clean_text_for_transformers

logger = logging.getLogger(__name__)

class AnnotationSentencesToDocBinConverter:
    """
    Converts a list of AnnotationSentence objects into a spaCy DocBin.
    
    This converter is primarily used for static embedding pipelines (e.g., Tok2Vec)
    where exact whitespace preservation is less critical.

    Attributes:
        nlp: The spaCy language pipeline.
        sentences_per_doc (int): Number of sentences to combine into a single Doc.
    """

    def __init__(self, nlp, sentences_per_doc: int = 3, clean_hidden_chars: bool = False):
        self.nlp = nlp
        self.sentences_per_doc = sentences_per_doc
        self.clean_hidden_chars = clean_hidden_chars
        # Default attribute values (can be overridden by subclasses)
        self.tag_layer: Optional[str] = None
        self.lemma_layer: Optional[str] = None
        self.ner: bool = False
        self.nel: bool = False

    def convert(self, sentences: List[AnnotationSentence]) -> DocBin:
        """
        Convert AnnotationSentences into a DocBin.

        Args:
            sentences (List[AnnotationSentence]): List of annotated sentences.

        Returns:
            DocBin: The resulting DocBin object.
        """
        doc_bin = DocBin(store_user_data=True)
        batch: List[Doc] = []

        for sent in sentences:
            doc = self._convert_sentence_to_doc(sent)
            
            batch.append(doc)
            if len(batch) == self.sentences_per_doc:
                combined = Doc.from_docs(batch)
                doc_bin.add(combined)
                batch = []

        if batch:
            combined = Doc.from_docs(batch)
            doc_bin.add(combined)

        return doc_bin

    def _convert_sentence_to_doc(self, sent: AnnotationSentence) -> Doc:
        """
        Convert a single AnnotationSentence to a spaCy Doc using exact token alignment.
        Preserves newlines by inserting them as tokens if they appear in gaps.
        
        This method is optimized for Transformer pipelines (e.g., RoBERTa) which are sensitive
        to exact whitespace and layout preservation. It ensures that newlines are treated
        as tokens so they are preserved in the DocBin format.

        Args:
            sent (AnnotationSentence): The annotated sentence.

        Returns:
            Doc: The spaCy Doc with entities and kb_ids set.
        """
        original_tokens = sent.tokens
        
        # Clean hidden characters if enabled (for transformer compatibility)
        sentence_text = sent.text
        if self.clean_hidden_chars and sentence_text:
            sentence_text = clean_text_for_transformers(sentence_text)
            # Also clean token texts
            for token in original_tokens:
                token.text = clean_text_for_transformers(token.text)
        
        # Filter out empty tokens (can happen after cleaning or from bad source data)
        # Keep track of original indices for entity mapping
        filtered_tokens = []
        original_to_filtered = {}  # Maps original index to filtered index
        for i, token in enumerate(original_tokens):
            if token.text and token.text.strip():  # Skip empty or whitespace-only tokens
                original_to_filtered[i] = len(filtered_tokens)
                filtered_tokens.append(token)
            else:
                logger.warning(f"Skipping empty token at position {token.start}-{token.end} in sentence: '{sentence_text[:50]}...'")
        
        original_tokens = filtered_tokens
        
        # Lists to build the Doc
        words = []
        spaces = []
        tags = []
        lemmas = []
        
        # Map original token index to new token index (for entity alignment)
        # original_index -> new_index
        token_map = {}
        
        if sentence_text and len(original_tokens) > 0:
            cursor = 0
            
            # Handle leading whitespace/newlines before first token
            first_start = sentence_text.find(original_tokens[0].text, 0)
            if first_start > 0:
                leading_gap = sentence_text[0:first_start]
                if "\n" in leading_gap:
                    # Insert newline token
                    words.append("\n")
                    spaces.append(False) # No space after newline usually
                    tags.append("_SP") # Standard spaCy tag for space tokens
                    lemmas.append("\n")
            
            # Iterate through tokens
            for i, token in enumerate(original_tokens):
                # Add the current token
                token_map[i] = len(words)
                words.append(token.text)
                
                # Get tag/lemma
                tags.append(token.layers.get(self.tag_layer, "_") if self.tag_layer else None)
                lemmas.append(token.layers.get(self.lemma_layer, "_") if self.lemma_layer else None)
                
                # Determine what comes next
                current_end = token.end # Character offset end
                
                # Find where this token actually ends in text (to be safe)
                # But we trust token.end from parser usually.
                # Let's use the gap logic.
                
                # Calculate gap to next token
                if i < len(original_tokens) - 1:
                    next_token = original_tokens[i+1]
                    gap_start = current_end
                    gap_end = next_token.start
                    
                    # If we have text, we can check the gap content
                    # Note: sentence_text might be larger than tokens cover
                    if gap_end > gap_start:
                        gap_text = sentence_text[gap_start:gap_end]
                        
                        if "\n" in gap_text:
                            # Gap contains newline -> Insert newline token
                            spaces.append(False) # Space for the current token is False because next is \n
                            
                            words.append("\n")
                            spaces.append(False) # Space after \n? Usually False unless \n \n
                            tags.append("_SP")
                            lemmas.append("\n")
                        elif len(gap_text.strip()) == 0 and len(gap_text) > 0:
                            # Just whitespace (spaces/tabs) -> Use spaces=True
                            spaces.append(True)
                        else:
                            # Gap has content but no newline? (Shouldn't happen if tokenization is complete)
                            # Fallback to space
                            spaces.append(True)
                    else:
                        # No gap
                        spaces.append(False)
                else:
                    # Last token - add trailing newline for sentence separation
                    # This ensures Doc.from_docs() preserves newlines between sentences
                    spaces.append(False)
                    
                    # Add trailing newline token
                    words.append("\n")
                    spaces.append(False)
                    tags.append("_SP")
                    lemmas.append("\n")
                    
        else:
            # Fallback if no text available (unlikely for this use case)
            words = sent.get_token_texts()
            spaces = [True] * len(words)
            if spaces: spaces[-1] = False
            token_map = {i: i for i in range(len(words))}
            tags = [t.layers.get(self.tag_layer, "_") for t in original_tokens] if self.tag_layer else None
            lemmas = [t.layers.get(self.lemma_layer, "_") for t in original_tokens] if self.lemma_layer else None

        # 3. Create the Doc Manually
        # Filter out any remaining empty words (safety check)
        if not words or all(w == "" for w in words):
            logger.warning(f"Sentence has no valid tokens after filtering: '{sent.text[:50]}...'")
            # Create a minimal doc with just a newline to avoid errors
            words = ["\n"]
            spaces = [False]
            tags = ["_SP"] if self.tag_layer else None
            lemmas = ["\n"] if self.lemma_layer else None
        
        # Note: We pass None for tags/lemmas if they aren't provided to let spaCy handle defaults
        # Filter None from tags/lemmas if lists were created with Nones
        final_tags = tags if self.tag_layer else None
        final_lemmas = lemmas if self.lemma_layer else None
        
        doc = Doc(self.nlp.vocab, words=words, spaces=spaces, tags=final_tags, lemmas=final_lemmas)
        
        if len(doc) > 0:
            doc[0].is_sent_start = True
            for i in range(1, len(doc)):
                doc[i].is_sent_start = False

        # 4. Handle Entities (NER) using Token Mapping
        if self.ner:
            spans: List[Span] = []
            
            for start_char, end_char, label, qid in sent.entities:
                # Find original token indices
                # We need to find which original tokens cover this char range
                # Then map those to new token indices
                
                orig_start_idx = -1
                orig_end_idx = -1
                
                # Simple search (can be optimized)
                for i, t in enumerate(original_tokens):
                    if t.start == start_char:
                        orig_start_idx = i
                    if t.end == end_char:
                        orig_end_idx = i + 1 # Exclusive
                
                if orig_start_idx != -1 and orig_end_idx != -1:
                    # Map to new indices
                    new_start = token_map[orig_start_idx]
                    # The end index in original tokens is exclusive. 
                    # We need the index of the last included token to map it.
                    last_included_token_idx = orig_end_idx - 1
                    new_end = token_map[last_included_token_idx] + 1
                    
                    span = Span(doc, new_start, new_end, label=label)
                    if self.nel:
                        span.kb_id_ = qid if qid != "*" else "NIL"
                    spans.append(span)
                else:
                    # Fallback: try char_span on the new doc
                    # This works because we inserted \n as tokens, so char offsets should be preserved
                    # (assuming \n token has length 1)
                    span = doc.char_span(start_char, end_char, label=label, alignment_mode="expand")
                    if span:
                        if self.nel:
                            span.kb_id_ = qid if qid != "*" else "NIL"
                        spans.append(span)
                    else:
                        logger.warning(
                            f"Could not align entity '{label}' at {start_char}:{end_char} "
                            f"in sentence: '{sent.text[:20]}...'"
                        )

            # Handle overlapping spans
            final_spans = filter_spans(spans)
            if len(final_spans) < len(spans):
                logger.warning(
                    f"Dropped {len(spans) - len(final_spans)} overlapping entities in sentence: '{sent.text[:20]}...'"
                )

            doc.ents = final_spans
            
        return doc

class AnnotationSentencesToDocBinConverterV2(AnnotationSentencesToDocBinConverter):
    """
    Converts a list of AnnotationSentence objects into a spaCy DocBin.
    
    This converter is designed for Transformer pipelines (e.g., RoBERTa, BERT) which are
    highly sensitive to whitespace and token alignment. It ensures "Whitespace Rigidity"
    by treating newlines as explicit tokens to preserve the exact layout of the original text.

    Compatible with spaCy v3 + Transformers.
    """

    def __init__(
        self, nlp, sentences_per_doc: int = 3,
        tag_layer: Optional[str] = None, lemma_layer: Optional[str] = None,
        ner: bool = False, nel: bool = False,
        clean_hidden_chars: bool = True,
    ):
        """
        Initialize the V2 converter for transformer pipelines.
        
        Args:
            nlp: The spaCy language pipeline.
            sentences_per_doc: Number of sentences to combine into a single Doc.
            tag_layer: Layer name for POS tags in WebAnno annotations.
            lemma_layer: Layer name for lemmas in WebAnno annotations.
            ner: Whether to include NER annotations.
            nel: Whether to include NEL (entity linking) annotations.
            clean_hidden_chars: Whether to remove hidden characters (ZWSP, BOM, etc.)
                that can break transformer tokenization. Default True for V2.
        """
        super().__init__(nlp, sentences_per_doc, clean_hidden_chars=clean_hidden_chars)
        self.tag_layer = tag_layer
        self.lemma_layer = lemma_layer
        self.ner = ner
        self.nel = nel





