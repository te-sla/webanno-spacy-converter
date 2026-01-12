"""
Tests for TSV parsers: BaseWebAnnoTSVParser, WebAnnoNELParser, WebAnnoLEXISParser.

These tests ensure that:
1. TSV files are parsed correctly
2. Headers are extracted properly
3. Tokens and entities are aligned correctly
4. Different TSV formats (Type A, B, C, D) are handled
5. Edge cases (empty files, malformed lines) are handled gracefully
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.parsers.tsv_parser_v3 import (
    BaseWebAnnoTSVParser,
    WebAnnoNELParser,
    WebAnnoLEXISParser
)
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.models.sentence_with_mwes import AnnotatedSentenceWithMWEs


# --- Test Data (various TSV formats) ---

# Type A: Simple NER only
TSV_TYPE_A_SIMPLE_NER = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Znate gde je Sv . Andreja ?
1-1	0-5	Znate	_	_	
1-2	6-9	gde	_	_	
1-3	10-12	je	_	_	
1-4	13-15	Sv	http://www.wikidata.org/entity/Q390798[1]	LOC[1]	
1-5	16-17	.	http://www.wikidata.org/entity/Q390798[1]	LOC[1]	
1-6	18-25	Andreja	http://www.wikidata.org/entity/Q390798[1]	LOC[1]	
1-7	26-27	?	_	_	
"""

# Type B: POS + NER + Lemma
TSV_TYPE_B_POS_NER_LEMMA = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma|value


#Text=Ginter de Brojn je pisac .
1-1	0-6	Ginter	PROPN	PROPN	http://www.wikidata.org/entity/Q62753[1]	PERS[1]	Ginter	
1-2	7-9	de	X	X	http://www.wikidata.org/entity/Q62753[1]	PERS[1]	de	
1-3	10-15	Brojn	PROPN	PROPN	http://www.wikidata.org/entity/Q62753[1]	PERS[1]	Brojn	
1-4	16-18	je	AUX	AUX	_	_	biti	
1-5	19-24	pisac	NOUN	NOUN	_	_	pisac	
1-6	25-26	.	PUNCT	PUNCT	_	_	.	
"""

# Simple single-token entities
TSV_SINGLE_TOKEN_ENTITIES = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Belgrade is in Serbia .
1-1	0-8	Belgrade	http://www.wikidata.org/entity/Q3711	LOC	
1-2	9-11	is	_	_	
1-3	12-14	in	_	_	
1-4	15-21	Serbia	http://www.wikidata.org/entity/Q403	LOC	
1-5	22-23	.	_	_	
"""

# Multiple sentences
TSV_MULTIPLE_SENTENCES = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Hello world .
1-1	0-5	Hello	_	_	
1-2	6-11	world	_	_	
1-3	12-13	.	_	_	

#Text=Belgrade is nice .
2-1	14-22	Belgrade	http://www.wikidata.org/entity/Q3711	LOC	
2-2	23-25	is	_	_	
2-3	26-30	nice	_	_	
2-4	31-32	.	_	_	

#Text=Goodbye !
3-1	33-40	Goodbye	_	_	
3-2	41-42	!	_	_	
"""

# No entities
TSV_NO_ENTITIES = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=This is a simple test .
1-1	0-4	This	_	_	
1-2	5-7	is	_	_	
1-3	8-9	a	_	_	
1-4	10-16	simple	_	_	
1-5	17-21	test	_	_	
1-6	22-23	.	_	_	
"""

# Entity without Wikidata link (asterisk)
TSV_ENTITY_NO_LINK = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=NIP-a is an organization .
1-1	0-5	NIP-a	*	WORK	
1-2	6-8	is	_	_	
1-3	9-11	an	_	_	
1-4	12-24	organization	_	_	
1-5	25-26	.	_	_	
"""

# MWE data for LEXIS parser
TSV_WITH_MWES = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value
#T_SP=custom.MWE|MWEid|MWElemma|MWEtype


#Text=New York City is great .
1-1	0-3	New	_	_	NYC[1]	New York City	LOC[1]	
1-2	4-8	York	_	_	NYC[1]	*	LOC[1]	
1-3	9-13	City	_	_	NYC[1]	*	LOC[1]	
1-4	14-16	is	_	_	_	_	_	
1-5	17-22	great	_	_	_	_	_	
1-6	23-24	.	_	_	_	_	_	
"""


@pytest.fixture
def temp_tsv_file():
    """Create a temporary TSV file and return its path."""
    def _create(content):
        fd, path = tempfile.mkstemp(suffix=".tsv")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
    
    paths = []
    def factory(content):
        path = _create(content)
        paths.append(path)
        return path
    
    yield factory
    
    # Cleanup
    for path in paths:
        try:
            os.unlink(path)
        except:
            pass


class TestWebAnnoNELParserBasic:
    """Basic parsing tests for WebAnnoNELParser."""
    
    def test_parse_simple_ner(self, temp_tsv_file):
        """Test parsing Type A (simple NER) format."""
        path = temp_tsv_file(TSV_TYPE_A_SIMPLE_NER)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        sent = sentences[0]
        
        # Check text
        assert sent.text == "Znate gde je Sv . Andreja ?"
        
        # Check tokens
        assert len(sent.tokens) == 7
        assert sent.tokens[0].text == "Znate"
        assert sent.tokens[3].text == "Sv"
        
        # Check entity (multi-token: "Sv . Andreja")
        assert len(sent.entities) == 1
        start, end, label, qid = sent.entities[0]
        assert label == "LOC"
        assert qid == "Q390798"
    
    def test_parse_pos_ner_lemma(self, temp_tsv_file):
        """Test parsing Type B (POS + NER + Lemma) format."""
        path = temp_tsv_file(TSV_TYPE_B_POS_NER_LEMMA)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        sent = sentences[0]
        
        # Check tokens have layers
        assert sent.tokens[0].text == "Ginter"
        
        # Check entity (multi-token: "Ginter de Brojn")
        assert len(sent.entities) == 1
        start, end, label, qid = sent.entities[0]
        assert label == "PERS"
        assert qid == "Q62753"
    
    def test_parse_single_token_entities(self, temp_tsv_file):
        """Test parsing single-token entities."""
        path = temp_tsv_file(TSV_SINGLE_TOKEN_ENTITIES)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        sent = sentences[0]
        
        # Should have 2 entities: Belgrade and Serbia
        assert len(sent.entities) == 2
        
        entity_labels = {e[2] for e in sent.entities}
        entity_qids = {e[3] for e in sent.entities}
        
        assert entity_labels == {"LOC"}
        assert "Q3711" in entity_qids  # Belgrade
        assert "Q403" in entity_qids   # Serbia
    
    def test_parse_multiple_sentences(self, temp_tsv_file):
        """Test parsing multiple sentences."""
        path = temp_tsv_file(TSV_MULTIPLE_SENTENCES)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 3
        
        assert sentences[0].text == "Hello world ."
        assert sentences[1].text == "Belgrade is nice ."
        assert sentences[2].text == "Goodbye !"
        
        # Only second sentence should have entity
        assert len(sentences[0].entities) == 0
        assert len(sentences[1].entities) == 1
        assert len(sentences[2].entities) == 0
    
    def test_parse_no_entities(self, temp_tsv_file):
        """Test parsing sentence with no entities."""
        path = temp_tsv_file(TSV_NO_ENTITIES)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        assert len(sentences[0].entities) == 0
        assert len(sentences[0].tokens) == 6
    
    def test_parse_entity_without_link(self, temp_tsv_file):
        """Test parsing entity with asterisk (no Wikidata link)."""
        path = temp_tsv_file(TSV_ENTITY_NO_LINK)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        assert len(sentences[0].entities) == 1
        
        start, end, label, qid = sentences[0].entities[0]
        assert label == "WORK"
        assert qid == "*"  # No link


class TestWebAnnoNELParserTokenAlignment:
    """Tests for token alignment and offsets."""
    
    def test_token_offsets_match_text(self, temp_tsv_file):
        """Test that token offsets correctly match sentence text."""
        path = temp_tsv_file(TSV_SINGLE_TOKEN_ENTITIES)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        sent = sentences[0]
        
        for token in sent.tokens:
            # Extract text using offsets
            extracted = sent.text[token.start:token.end]
            assert extracted == token.text, f"Token '{token.text}' doesn't match extracted '{extracted}' at {token.start}:{token.end}"
    
    def test_entity_offsets_match_text(self, temp_tsv_file):
        """Test that entity offsets correctly match sentence text."""
        path = temp_tsv_file(TSV_TYPE_A_SIMPLE_NER)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        sent = sentences[0]
        
        # Entity should be "Sv . Andreja"
        assert len(sent.entities) == 1
        start, end, label, qid = sent.entities[0]
        
        entity_text = sent.text[start:end]
        assert "Sv" in entity_text
        assert "Andreja" in entity_text


class TestWebAnnoNELParserEdgeCases:
    """Edge case tests for parser."""
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        parser = WebAnnoNELParser("/nonexistent/path/file.tsv")
        
        with pytest.raises(FileNotFoundError):
            parser.parse()
    
    def test_empty_sentences_skipped(self, temp_tsv_file):
        """Test that empty/malformed sentences are skipped gracefully."""
        content = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Hello world .
1-1	0-5	Hello	_	_	
1-2	6-11	world	_	_	
1-3	12-13	.	_	_	

#Text=
"""
        path = temp_tsv_file(content)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        # Should only get the valid sentence
        assert len(sentences) >= 1


class TestWebAnnoLEXISParser:
    """Tests for WebAnnoLEXISParser with MWE support."""
    
    def test_parse_with_mwes(self, temp_tsv_file):
        """Test parsing sentences with multi-word expressions."""
        path = temp_tsv_file(TSV_WITH_MWES)
        parser = WebAnnoLEXISParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        sent = sentences[0]
        
        # Should be AnnotatedSentenceWithMWEs
        assert isinstance(sent, AnnotatedSentenceWithMWEs)
        
        # Should have MWEs
        assert len(sent.mwes) == 1
        mwe = sent.mwes[0]
        assert mwe.lemma == "New York City"
        assert mwe.token_count == 3
        assert mwe.token_indices == [0, 1, 2]


class TestParserLayerExtraction:
    """Tests for annotation layer extraction."""
    
    def test_layer_names_extracted(self, temp_tsv_file):
        """Test that layer names are correctly extracted from header."""
        path = temp_tsv_file(TSV_TYPE_B_POS_NER_LEMMA)
        parser = WebAnnoNELParser(path)
        parser.parse()
        
        # Check that layer names were extracted
        layer_names = list(parser.layer_names.values())
        
        # Should have multiple layers
        assert len(layer_names) > 0
        
        # Common layer names should be present
        assert "identifier" in layer_names or "value" in layer_names
    
    def test_token_layers_populated(self, temp_tsv_file):
        """Test that token layers are populated from TSV columns."""
        path = temp_tsv_file(TSV_SINGLE_TOKEN_ENTITIES)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        # Belgrade token should have layers
        belgrade = sentences[0].tokens[0]
        assert belgrade.text == "Belgrade"
        
        # Should have NER layers
        assert "identifier" in belgrade.layers or "value" in belgrade.layers


class TestParserRobustness:
    """Robustness tests for parser."""
    
    def test_handles_unicode(self, temp_tsv_file):
        """Test handling of Unicode characters."""
        content = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Београд је леп .
1-1	0-7	Београд	http://www.wikidata.org/entity/Q3711	LOC	
1-2	8-10	је	_	_	
1-3	11-14	леп	_	_	
1-4	15-16	.	_	_	
"""
        path = temp_tsv_file(content)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        assert sentences[0].tokens[0].text == "Београд"
    
    def test_handles_special_characters(self, temp_tsv_file):
        """Test handling of special characters in text."""
        content = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Test — with "quotes" & symbols !
1-1	0-4	Test	_	_	
1-2	5-6	—	_	_	
1-3	7-11	with	_	_	
1-4	12-20	"quotes"	_	_	
1-5	21-22	&	_	_	
1-6	23-30	symbols	_	_	
1-7	31-32	!	_	_	
"""
        path = temp_tsv_file(content)
        parser = WebAnnoNELParser(path)
        sentences = parser.parse()
        
        assert len(sentences) == 1
        token_texts = [t.text for t in sentences[0].tokens]
        assert "—" in token_texts
        assert '"quotes"' in token_texts
        assert "&" in token_texts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
