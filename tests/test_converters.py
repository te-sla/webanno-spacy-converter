"""
Tests for WebAnno to spaCy converters.

These tests ensure that:
1. AnnotationSentences are correctly converted to spaCy Docs
2. Entities are preserved with correct labels and QIDs
3. Token alignment is maintained
4. POS tags and lemmas are transferred correctly
5. Multiple sentences are combined into Docs correctly
6. Edge cases (empty sentences, overlapping entities) are handled
"""

import pytest
import spacy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.models.annotation_token import AnnotationToken
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.converters.webanno_to_spacy import (
    AnnotationSentencesToDocBinConverter,
    AnnotationSentencesToDocBinConverterV2
)


@pytest.fixture
def nlp():
    """Create a blank spaCy model for Serbian."""
    return spacy.blank("sr")


def create_simple_sentence(text="Hello world .", tokens_data=None):
    """Helper to create a simple AnnotationSentence."""
    if tokens_data is None:
        tokens_data = [
            (1, 1, "Hello", 0, 5, {}),
            (1, 2, "world", 6, 11, {}),
            (1, 3, ".", 12, 13, {}),
        ]
    
    tokens = [
        AnnotationToken(sent_idx, tok_idx, text, start, end, layers)
        for sent_idx, tok_idx, text, start, end, layers in tokens_data
    ]
    
    return AnnotationSentence(text=text, tokens=tokens, entities=[])


def create_sentence_with_entity(
    text="Belgrade is nice .",
    entity_text="Belgrade",
    entity_label="LOC",
    entity_qid="Q3711"
):
    """Helper to create a sentence with a single entity."""
    tokens = [
        AnnotationToken(1, 1, "Belgrade", 0, 8, {"value": "LOC", "identifier": "Q3711"}),
        AnnotationToken(1, 2, "is", 9, 11, {}),
        AnnotationToken(1, 3, "nice", 12, 16, {}),
        AnnotationToken(1, 4, ".", 17, 18, {}),
    ]
    
    entities = [(0, 8, entity_label, entity_qid)]
    
    return AnnotationSentence(text=text, tokens=tokens, entities=entities)


def create_multi_token_entity_sentence():
    """Helper to create a sentence with multi-token entity."""
    tokens = [
        AnnotationToken(1, 1, "New", 0, 3, {"value": "LOC[1]", "identifier": "Q60[1]"}),
        AnnotationToken(1, 2, "York", 4, 8, {"value": "LOC[1]", "identifier": "Q60[1]"}),
        AnnotationToken(1, 3, "City", 9, 13, {"value": "LOC[1]", "identifier": "Q60[1]"}),
        AnnotationToken(1, 4, "is", 14, 16, {}),
        AnnotationToken(1, 5, "great", 17, 22, {}),
        AnnotationToken(1, 6, ".", 23, 24, {}),
    ]
    
    # Multi-token entity spans all three name tokens
    entities = [(0, 13, "LOC", "Q60")]
    
    return AnnotationSentence(
        text="New York City is great .",
        tokens=tokens,
        entities=entities
    )


class TestConverterBasic:
    """Basic conversion tests."""
    
    def test_convert_single_sentence(self, nlp):
        """Test converting a single sentence to spaCy Doc."""
        sentence = create_simple_sentence()
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Check tokens are present
        assert len(doc) >= 3  # At least our 3 tokens
        
        # Check text content
        token_texts = [t.text for t in doc if t.text.strip()]
        assert "Hello" in token_texts
        assert "world" in token_texts
    
    def test_convert_multiple_sentences(self, nlp):
        """Test converting multiple sentences combined into one Doc."""
        sentences = [
            create_simple_sentence("Hello world .", [
                (1, 1, "Hello", 0, 5, {}),
                (1, 2, "world", 6, 11, {}),
                (1, 3, ".", 12, 13, {}),
            ]),
            create_simple_sentence("Goodbye all .", [
                (2, 1, "Goodbye", 0, 7, {}),
                (2, 2, "all", 8, 11, {}),
                (2, 3, ".", 12, 13, {}),
            ]),
        ]
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=2,  # Combine 2 sentences per doc
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert(sentences)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1  # Both sentences in one doc
    
    def test_sentences_split_into_multiple_docs(self, nlp):
        """Test that sentences are split into multiple docs based on sentences_per_doc."""
        sentences = [
            create_simple_sentence(f"Sentence {i} .", [
                (i, 1, "Sentence", 0, 8, {}),
                (i, 2, str(i), 9, 10, {}),
                (i, 3, ".", 11, 12, {}),
            ])
            for i in range(1, 6)  # 5 sentences
        ]
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=2,  # 2 sentences per doc
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert(sentences)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        # 5 sentences / 2 per doc = 3 docs (2 + 2 + 1)
        assert len(docs) == 3


class TestConverterEntities:
    """Tests for entity conversion."""
    
    def test_convert_single_token_entity(self, nlp):
        """Test converting a sentence with single-token entity."""
        sentence = create_sentence_with_entity()
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=True,
            nel=True
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Check entity
        assert len(doc.ents) == 1
        ent = doc.ents[0]
        assert ent.text == "Belgrade"
        assert ent.label_ == "LOC"
        assert ent.kb_id_ == "Q3711"
    
    def test_convert_multi_token_entity(self, nlp):
        """Test converting a sentence with multi-token entity."""
        sentence = create_multi_token_entity_sentence()
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=True,
            nel=True
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Check entity
        assert len(doc.ents) >= 1
        
        # Find the LOC entity
        loc_ents = [e for e in doc.ents if e.label_ == "LOC"]
        assert len(loc_ents) >= 1
    
    def test_convert_multiple_entities(self, nlp):
        """Test converting a sentence with multiple entities."""
        tokens = [
            AnnotationToken(1, 1, "Belgrade", 0, 8, {}),
            AnnotationToken(1, 2, "and", 9, 12, {}),
            AnnotationToken(1, 3, "Serbia", 13, 19, {}),
            AnnotationToken(1, 4, ".", 20, 21, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Belgrade and Serbia .",
            tokens=tokens,
            entities=[
                (0, 8, "LOC", "Q3711"),   # Belgrade
                (13, 19, "LOC", "Q403"),  # Serbia
            ]
        )
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=True,
            nel=True
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Should have 2 entities
        assert len(doc.ents) == 2
        
        qids = {e.kb_id_ for e in doc.ents}
        assert "Q3711" in qids
        assert "Q403" in qids
    
    def test_entity_nil_link(self, nlp):
        """Test that entities without Wikidata link get NIL kb_id."""
        tokens = [
            AnnotationToken(1, 1, "Unknown", 0, 7, {}),
            AnnotationToken(1, 2, ".", 8, 9, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Unknown .",
            tokens=tokens,
            entities=[(0, 7, "PER", "*")]  # asterisk = no link
        )
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=True,
            nel=True
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        assert len(doc.ents) == 1
        assert doc.ents[0].kb_id_ == "NIL"


class TestConverterPOSAndLemma:
    """Tests for POS tag and lemma conversion."""
    
    def test_convert_with_pos_tags(self, nlp):
        """Test converting with POS tags."""
        tokens = [
            AnnotationToken(1, 1, "Belgrade", 0, 8, {"coarseValue": "PROPN"}),
            AnnotationToken(1, 2, "is", 9, 11, {"coarseValue": "AUX"}),
            AnnotationToken(1, 3, "nice", 12, 16, {"coarseValue": "ADJ"}),
            AnnotationToken(1, 4, ".", 17, 18, {"coarseValue": "PUNCT"}),
        ]
        
        sentence = AnnotationSentence(
            text="Belgrade is nice .",
            tokens=tokens,
            entities=[]
        )
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            tag_layer="coarseValue",
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Check POS tags - find tokens by text
        for token in doc:
            if token.text == "Belgrade":
                assert token.tag_ == "PROPN"
            elif token.text == "is":
                assert token.tag_ == "AUX"
            elif token.text == "nice":
                assert token.tag_ == "ADJ"
    
    def test_convert_with_lemmas(self, nlp):
        """Test converting with lemmas."""
        tokens = [
            AnnotationToken(1, 1, "running", 0, 7, {"value_4": "run"}),
            AnnotationToken(1, 2, "fast", 8, 12, {"value_4": "fast"}),
            AnnotationToken(1, 3, ".", 13, 14, {"value_4": "."}),
        ]
        
        sentence = AnnotationSentence(
            text="running fast .",
            tokens=tokens,
            entities=[]
        )
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            lemma_layer="value_4",
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Check lemmas
        for token in doc:
            if token.text == "running":
                assert token.lemma_ == "run"


class TestConverterSentenceBoundaries:
    """Tests for sentence boundary handling."""
    
    def test_sentence_start_markers(self, nlp):
        """Test that sentence start markers are set correctly."""
        sentences = [
            create_simple_sentence("First sentence .", [
                (1, 1, "First", 0, 5, {}),
                (1, 2, "sentence", 6, 14, {}),
                (1, 3, ".", 15, 16, {}),
            ]),
            create_simple_sentence("Second sentence .", [
                (2, 1, "Second", 0, 6, {}),
                (2, 2, "sentence", 7, 15, {}),
                (2, 3, ".", 16, 17, {}),
            ]),
        ]
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=2,  # Both in one doc
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert(sentences)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Count sentence starts
        sent_starts = [t for t in doc if t.is_sent_start]
        assert len(sent_starts) >= 1  # At least the first token


class TestConverterEdgeCases:
    """Edge case tests for converter."""
    
    def test_empty_sentence_list(self, nlp):
        """Test handling of empty sentence list."""
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert([])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 0
    
    def test_sentence_with_newline(self, nlp):
        """Test handling of sentences with newlines."""
        tokens = [
            AnnotationToken(1, 1, "Line", 0, 4, {}),
            AnnotationToken(1, 2, "one", 5, 8, {}),
            AnnotationToken(1, 3, ".", 8, 9, {}),
            # Newline would be at position 9-10
            AnnotationToken(1, 4, "Line", 11, 15, {}),
            AnnotationToken(1, 5, "two", 16, 19, {}),
            AnnotationToken(1, 6, ".", 19, 20, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Line one .\nLine two .",
            tokens=tokens,
            entities=[]
        )
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=False,
            nel=False
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
    
    def test_unicode_text(self, nlp):
        """Test handling of Unicode text (Cyrillic)."""
        tokens = [
            AnnotationToken(1, 1, "Београд", 0, 7, {}),
            AnnotationToken(1, 2, "је", 8, 10, {}),
            AnnotationToken(1, 3, "леп", 11, 14, {}),
            AnnotationToken(1, 4, ".", 15, 16, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Београд је леп .",
            tokens=tokens,
            entities=[(0, 7, "LOC", "Q3711")]
        )
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=True,
            nel=True
        )
        
        doc_bin = converter.convert([sentence])
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Check entity
        assert len(doc.ents) == 1
        assert doc.ents[0].text == "Београд"


class TestConverterV2SpecificFeatures:
    """Tests specific to V2 converter (transformer compatibility)."""
    
    def test_clean_hidden_chars_enabled(self, nlp):
        """Test that hidden character cleaning is enabled by default in V2."""
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=False,
            nel=False
        )
        
        assert converter.clean_hidden_chars is True
    
    def test_clean_hidden_chars_disabled(self, nlp):
        """Test that hidden character cleaning can be disabled."""
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=False,
            nel=False,
            clean_hidden_chars=False
        )
        
        assert converter.clean_hidden_chars is False


class TestDocBinSerialization:
    """Tests for DocBin serialization and deserialization."""
    
    def test_docbin_roundtrip(self, nlp, tmp_path):
        """Test saving and loading DocBin."""
        sentence = create_sentence_with_entity()
        
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=1,
            ner=True,
            nel=True
        )
        
        doc_bin = converter.convert([sentence])
        
        # Save to file
        output_path = tmp_path / "test.spacy"
        doc_bin.to_disk(str(output_path))
        
        # Load back
        loaded_doc_bin = spacy.tokens.DocBin().from_disk(str(output_path))
        docs = list(loaded_doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # Verify entity survived roundtrip
        assert len(doc.ents) == 1
        assert doc.ents[0].label_ == "LOC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
