"""
Tests for data models: AnnotationToken, AnnotationSentence, and MWE models.

These tests ensure that:
1. Models correctly store and retrieve data
2. Factory methods work correctly
3. Helper methods return expected values
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.models.annotation_token import AnnotationToken
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.models.sentence_with_mwes import MultiWordExpression, AnnotatedSentenceWithMWEs


class TestAnnotationToken:
    """Tests for AnnotationToken model."""
    
    def test_create_basic_token(self):
        """Test basic token creation."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="Hello",
            start=0,
            end=5,
            layers={}
        )
        
        assert token.sentence_index == 1
        assert token.token_index == 1
        assert token.text == "Hello"
        assert token.start == 0
        assert token.end == 5
        assert token.layers == {}
    
    def test_token_with_layers(self):
        """Test token with annotation layers."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="Belgrade",
            start=0,
            end=8,
            layers={
                "POS": "PROPN",
                "lemma": "Belgrade",
                "value": "LOC",
                "identifier": "Q3711"
            }
        )
        
        assert token.layers["POS"] == "PROPN"
        assert token.layers["value"] == "LOC"
        assert token.layers["identifier"] == "Q3711"
    
    def test_get_layer_existing(self):
        """Test get_layer method with existing layer."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="test",
            start=0,
            end=4,
            layers={"POS": "NOUN"}
        )
        
        assert token.get_layer("POS") == "NOUN"
    
    def test_get_layer_missing(self):
        """Test get_layer method with missing layer."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="test",
            start=0,
            end=4,
            layers={}
        )
        
        assert token.get_layer("POS") is None
    
    def test_add_layer(self):
        """Test add_layer method."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="test",
            start=0,
            end=4,
            layers={}
        )
        
        token.add_layer("POS", "NOUN")
        assert token.layers["POS"] == "NOUN"
    
    def test_add_layer_overwrites(self):
        """Test that add_layer overwrites existing layer."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="test",
            start=0,
            end=4,
            layers={"POS": "NOUN"}
        )
        
        token.add_layer("POS", "VERB")
        assert token.layers["POS"] == "VERB"
    
    def test_has_annotation_true(self):
        """Test has_annotation returns True for existing layer."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="test",
            start=0,
            end=4,
            layers={"POS": "NOUN"}
        )
        
        assert token.has_annotation("POS") is True
    
    def test_has_annotation_false(self):
        """Test has_annotation returns False for missing layer."""
        token = AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="test",
            start=0,
            end=4,
            layers={}
        )
        
        assert token.has_annotation("POS") is False


class TestAnnotationSentence:
    """Tests for AnnotationSentence model."""
    
    def test_create_basic_sentence(self):
        """Test basic sentence creation."""
        tokens = [
            AnnotationToken(1, 1, "Hello", 0, 5, {}),
            AnnotationToken(1, 2, "world", 6, 11, {}),
            AnnotationToken(1, 3, ".", 11, 12, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Hello world.",
            tokens=tokens,
            entities=[]
        )
        
        assert sentence.text == "Hello world."
        assert len(sentence.tokens) == 3
        assert len(sentence.entities) == 0
    
    def test_sentence_with_entities(self):
        """Test sentence with named entities."""
        tokens = [
            AnnotationToken(1, 1, "Belgrade", 0, 8, {"value": "LOC", "identifier": "Q3711"}),
            AnnotationToken(1, 2, "is", 9, 11, {}),
            AnnotationToken(1, 3, "nice", 12, 16, {}),
            AnnotationToken(1, 4, ".", 16, 17, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Belgrade is nice.",
            tokens=tokens,
            entities=[(0, 8, "LOC", "Q3711")]
        )
        
        assert len(sentence.entities) == 1
        start, end, label, qid = sentence.entities[0]
        assert start == 0
        assert end == 8
        assert label == "LOC"
        assert qid == "Q3711"
    
    def test_get_token_texts(self):
        """Test get_token_texts returns list of token texts."""
        tokens = [
            AnnotationToken(1, 1, "Hello", 0, 5, {}),
            AnnotationToken(1, 2, "world", 6, 11, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Hello world",
            tokens=tokens,
            entities=[]
        )
        
        texts = sentence.get_token_texts()
        assert texts == ["Hello", "world"]
    
    def test_get_entity_spans(self):
        """Test get_entity_spans returns entity text."""
        tokens = [
            AnnotationToken(1, 1, "Belgrade", 0, 8, {}),
            AnnotationToken(1, 2, "is", 9, 11, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Belgrade is",
            tokens=tokens,
            entities=[(0, 8, "LOC", "Q3711")]
        )
        
        spans = sentence.get_entity_spans()
        assert spans == ["Belgrade"]
    
    def test_multi_token_entity_span(self):
        """Test entity span extraction for multi-token entities."""
        tokens = [
            AnnotationToken(1, 1, "New", 0, 3, {}),
            AnnotationToken(1, 2, "York", 4, 8, {}),
            AnnotationToken(1, 3, "City", 9, 13, {}),
        ]
        
        sentence = AnnotationSentence(
            text="New York City",
            tokens=tokens,
            entities=[(0, 13, "LOC", "Q60")]
        )
        
        spans = sentence.get_entity_spans()
        assert spans == ["New York City"]
    
    def test_repr(self):
        """Test __repr__ method."""
        tokens = [AnnotationToken(1, 1, "Test", 0, 4, {})]
        sentence = AnnotationSentence(
            text="Test",
            tokens=tokens,
            entities=[(0, 4, "MISC", "*")]
        )
        
        repr_str = repr(sentence)
        assert "AnnotationSentence" in repr_str
        assert "1 tokens" in repr_str
        assert "1 entities" in repr_str


class TestMultiWordExpression:
    """Tests for MultiWordExpression model."""
    
    def test_create_mwe(self):
        """Test basic MWE creation."""
        mwe = MultiWordExpression(
            lemma="New York",
            token_count=2,
            token_indices=[0, 1],
            type="LOC",
            group_id="1"
        )
        
        assert mwe.lemma == "New York"
        assert mwe.token_count == 2
        assert mwe.token_indices == [0, 1]
        assert mwe.type == "LOC"
        assert mwe.group_id == "1"
    
    def test_mwe_defaults(self):
        """Test MWE default values."""
        mwe = MultiWordExpression(
            lemma="test",
            token_count=1,
            token_indices=[0]
        )
        
        assert mwe.type == ""
        assert mwe.group_id == ""


class TestAnnotatedSentenceWithMWEs:
    """Tests for AnnotatedSentenceWithMWEs model."""
    
    def test_create_sentence_with_mwes(self):
        """Test sentence creation with MWEs."""
        tokens = [
            AnnotationToken(1, 1, "New", 0, 3, {}),
            AnnotationToken(1, 2, "York", 4, 8, {}),
            AnnotationToken(1, 3, "is", 9, 11, {}),
            AnnotationToken(1, 4, "nice", 12, 16, {}),
        ]
        
        mwes = [
            MultiWordExpression(
                lemma="New York",
                token_count=2,
                token_indices=[0, 1],
                type="LOC",
                group_id="1"
            )
        ]
        
        sentence = AnnotatedSentenceWithMWEs(
            text="New York is nice",
            tokens=tokens,
            entities=[],
            mwes=mwes
        )
        
        assert len(sentence.mwes) == 1
        assert sentence.mwes[0].lemma == "New York"
    
    def test_get_expression_spans(self):
        """Test get_expression_spans returns MWE text."""
        tokens = [
            AnnotationToken(1, 1, "New", 0, 3, {}),
            AnnotationToken(1, 2, "York", 4, 8, {}),
            AnnotationToken(1, 3, "City", 9, 13, {}),
        ]
        
        mwes = [
            MultiWordExpression(
                lemma="New York City",
                token_count=3,
                token_indices=[0, 1, 2],
                type="LOC",
                group_id="1"
            )
        ]
        
        sentence = AnnotatedSentenceWithMWEs(
            text="New York City",
            tokens=tokens,
            entities=[],
            mwes=mwes
        )
        
        spans = sentence.get_expression_spans()
        assert spans == ["New York City"]
    
    def test_non_contiguous_mwe(self):
        """Test non-contiguous MWE (e.g., phrasal verb with object in between)."""
        # "pick ... up" with object in between
        tokens = [
            AnnotationToken(1, 1, "pick", 0, 4, {}),
            AnnotationToken(1, 2, "it", 5, 7, {}),
            AnnotationToken(1, 3, "up", 8, 10, {}),
        ]
        
        mwes = [
            MultiWordExpression(
                lemma="pick up",
                token_count=2,
                token_indices=[0, 2],  # Non-contiguous!
                type="VERB",
                group_id="1"
            )
        ]
        
        sentence = AnnotatedSentenceWithMWEs(
            text="pick it up",
            tokens=tokens,
            entities=[],
            mwes=mwes
        )
        
        spans = sentence.get_expression_spans()
        assert spans == ["pick up"]  # Should join tokens at indices 0 and 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
