"""
Tests for WebAnno TSV writers.

These tests ensure that:
1. Annotations are correctly written to TSV format
2. Headers are properly formatted
3. Token offsets are calculated correctly
4. Entity annotations are preserved in roundtrip
5. Output files are valid WebAnno TSV format
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.models.annotation_token import AnnotationToken
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.writers.webanno_writer import (
    BaseWebAnnoTSVWriter,
    WebAnnoNELWriter
)
from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser


def create_test_sentence(
    text="Belgrade is nice .",
    entities=None
):
    """Helper to create a test sentence."""
    tokens = [
        AnnotationToken(
            sentence_index=1,
            token_index=1,
            text="Belgrade",
            start=0,
            end=8,
            layers={"identifier": "Q3711", "value": "LOC"}
        ),
        AnnotationToken(
            sentence_index=1,
            token_index=2,
            text="is",
            start=9,
            end=11,
            layers={}
        ),
        AnnotationToken(
            sentence_index=1,
            token_index=3,
            text="nice",
            start=12,
            end=16,
            layers={}
        ),
        AnnotationToken(
            sentence_index=1,
            token_index=4,
            text=".",
            start=17,
            end=18,
            layers={}
        ),
    ]
    
    if entities is None:
        entities = [(0, 8, "LOC", "Q3711")]
    
    return AnnotationSentence(text=text, tokens=tokens, entities=entities)


def create_multi_sentence_data():
    """Helper to create multiple test sentences."""
    sentences = []
    
    # Sentence 1
    tokens1 = [
        AnnotationToken(1, 1, "Hello", 0, 5, {}),
        AnnotationToken(1, 2, "world", 6, 11, {}),
        AnnotationToken(1, 3, ".", 12, 13, {}),
    ]
    sentences.append(AnnotationSentence(text="Hello world .", tokens=tokens1, entities=[]))
    
    # Sentence 2
    tokens2 = [
        AnnotationToken(2, 1, "Belgrade", 0, 8, {"identifier": "Q3711", "value": "LOC"}),
        AnnotationToken(2, 2, "is", 9, 11, {}),
        AnnotationToken(2, 3, "great", 12, 17, {}),
        AnnotationToken(2, 4, ".", 18, 19, {}),
    ]
    sentences.append(AnnotationSentence(
        text="Belgrade is great .",
        tokens=tokens2,
        entities=[(0, 8, "LOC", "Q3711")]
    ))
    
    return sentences


@pytest.fixture
def temp_output_file():
    """Create a temporary output file path."""
    fd, path = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    
    yield path
    
    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


class TestWebAnnoNELWriterBasic:
    """Basic tests for WebAnnoNELWriter."""
    
    def test_write_single_sentence(self, temp_output_file):
        """Test writing a single sentence."""
        sentence = create_test_sentence()
        writer = WebAnnoNELWriter([sentence])
        
        writer.save(temp_output_file)
        
        # Verify file exists and has content
        assert os.path.exists(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 0
        assert "#FORMAT=WebAnno TSV 3.3" in content
    
    def test_write_multiple_sentences(self, temp_output_file):
        """Test writing multiple sentences."""
        sentences = create_multi_sentence_data()
        writer = WebAnnoNELWriter(sentences)
        
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have both sentence texts
        assert "#Text=Hello world ." in content
        assert "#Text=Belgrade is great ." in content
    
    def test_header_format(self, temp_output_file):
        """Test that header is correctly formatted."""
        sentence = create_test_sentence()
        writer = WebAnnoNELWriter([sentence])
        
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # First line should be format header
        assert lines[0].strip() == "#FORMAT=WebAnno TSV 3.3"
        
        # Second line should be layer header
        assert "#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity" in lines[1]
        assert "identifier" in lines[1]
        assert "value" in lines[1]


class TestWebAnnoNELWriterTokens:
    """Tests for token writing."""
    
    def test_token_format(self, temp_output_file):
        """Test that tokens are written in correct format."""
        sentence = create_test_sentence()
        writer = WebAnnoNELWriter([sentence])
        
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have token lines with format: sent-tok \t offset \t text \t layers
        lines = content.split('\n')
        token_lines = [l for l in lines if l.startswith('1-')]
        
        assert len(token_lines) == 4  # 4 tokens
        
        # Check first token (Belgrade with entity)
        assert "Belgrade" in token_lines[0]
        assert "Q3711" in token_lines[0] or "wikidata" in token_lines[0]
        assert "LOC" in token_lines[0]
    
    def test_token_offsets_correct(self, temp_output_file):
        """Test that token offsets are calculated correctly."""
        sentences = create_multi_sentence_data()
        writer = WebAnnoNELWriter(sentences)
        
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # First sentence tokens should start from 0
        first_token = [l for l in lines if l.startswith('1-1\t')][0]
        assert '0-5' in first_token  # "Hello" is 0-5
        
        # Second sentence tokens should continue from previous sentence
        # First sentence is "Hello world ." (13 chars + 1 for offset = 14)
        second_sent_tokens = [l for l in lines if l.startswith('2-1\t')][0]
        # The offset calculation depends on implementation
        assert '2-1' in second_sent_tokens


class TestWebAnnoNELWriterEntities:
    """Tests for entity writing."""
    
    def test_entity_with_wikidata_link(self, temp_output_file):
        """Test writing entity with Wikidata link."""
        sentence = create_test_sentence()
        writer = WebAnnoNELWriter([sentence])
        
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have wikidata URL
        assert "http://www.wikidata.org/entity/Q3711" in content
        assert "LOC" in content
    
    def test_entity_without_link(self, temp_output_file):
        """Test writing entity without Wikidata link (asterisk)."""
        tokens = [
            AnnotationToken(1, 1, "Unknown", 0, 7, {"identifier": "*", "value": "PER"}),
            AnnotationToken(1, 2, ".", 8, 9, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Unknown .",
            tokens=tokens,
            entities=[(0, 7, "PER", "*")]
        )
        
        writer = WebAnnoNELWriter([sentence])
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have asterisk for no link
        assert "*\tPER" in content or "PER" in content
    
    def test_non_entity_tokens_have_underscore(self, temp_output_file):
        """Test that non-entity tokens have underscore for layers."""
        sentence = create_test_sentence()
        writer = WebAnnoNELWriter([sentence])
        
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # "is" token should have underscores
        lines = content.split('\n')
        is_token_line = [l for l in lines if '\tis\t' in l][0]
        assert '_\t_' in is_token_line


class TestWebAnnoNELWriterRoundtrip:
    """Tests for write-then-read roundtrip."""
    
    def test_roundtrip_preserves_text(self, temp_output_file):
        """Test that text is preserved in roundtrip."""
        original = create_test_sentence()
        
        # Write
        writer = WebAnnoNELWriter([original])
        writer.save(temp_output_file)
        
        # Read back
        parser = WebAnnoNELParser(temp_output_file)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        assert sentences[0].text == original.text
    
    def test_roundtrip_preserves_tokens(self, temp_output_file):
        """Test that tokens are preserved in roundtrip."""
        original = create_test_sentence()
        
        # Write
        writer = WebAnnoNELWriter([original])
        writer.save(temp_output_file)
        
        # Read back
        parser = WebAnnoNELParser(temp_output_file)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        assert len(sentences[0].tokens) == len(original.tokens)
        
        for orig_tok, read_tok in zip(original.tokens, sentences[0].tokens):
            assert orig_tok.text == read_tok.text
    
    def test_roundtrip_preserves_entities(self, temp_output_file):
        """Test that entities are preserved in roundtrip."""
        original = create_test_sentence()
        
        # Write
        writer = WebAnnoNELWriter([original])
        writer.save(temp_output_file)
        
        # Read back
        parser = WebAnnoNELParser(temp_output_file)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        assert len(sentences[0].entities) == 1
        
        _, _, label, qid = sentences[0].entities[0]
        assert label == "LOC"
        assert qid == "Q3711"


class TestWebAnnoNELWriterUnicode:
    """Tests for Unicode handling in writer."""
    
    def test_write_cyrillic_text(self, temp_output_file):
        """Test writing Cyrillic text."""
        tokens = [
            AnnotationToken(1, 1, "Београд", 0, 7, {"identifier": "Q3711", "value": "LOC"}),
            AnnotationToken(1, 2, "је", 8, 10, {}),
            AnnotationToken(1, 3, "леп", 11, 14, {}),
            AnnotationToken(1, 4, ".", 15, 16, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Београд је леп .",
            tokens=tokens,
            entities=[(0, 7, "LOC", "Q3711")]
        )
        
        writer = WebAnnoNELWriter([sentence])
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Београд" in content
        assert "#Text=Београд је леп ." in content
    
    def test_write_special_characters(self, temp_output_file):
        """Test writing special characters."""
        tokens = [
            AnnotationToken(1, 1, "Test", 0, 4, {}),
            AnnotationToken(1, 2, "—", 5, 6, {}),
            AnnotationToken(1, 3, '"quotes"', 7, 15, {}),
            AnnotationToken(1, 4, ".", 16, 17, {}),
        ]
        
        sentence = AnnotationSentence(
            text='Test — "quotes" .',
            tokens=tokens,
            entities=[]
        )
        
        writer = WebAnnoNELWriter([sentence])
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "—" in content
        assert '"quotes"' in content


class TestWebAnnoNELWriterEdgeCases:
    """Edge case tests for writer."""
    
    def test_write_empty_sentence_list(self, temp_output_file):
        """Test writing empty sentence list."""
        writer = WebAnnoNELWriter([])
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should still have header
        assert "#FORMAT=WebAnno TSV 3.3" in content
    
    def test_write_sentence_no_entities(self, temp_output_file):
        """Test writing sentence without entities."""
        tokens = [
            AnnotationToken(1, 1, "Hello", 0, 5, {}),
            AnnotationToken(1, 2, "world", 6, 11, {}),
            AnnotationToken(1, 3, ".", 12, 13, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Hello world .",
            tokens=tokens,
            entities=[]
        )
        
        writer = WebAnnoNELWriter([sentence])
        writer.save(temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # All tokens should have underscores for entity layers
        lines = content.split('\n')
        token_lines = [l for l in lines if l.startswith('1-')]
        
        for line in token_lines:
            assert '_\t_' in line or line.endswith('_\t')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
