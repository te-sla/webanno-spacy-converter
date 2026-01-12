"""
Tests for Cyrillic ↔ Latin transliteration in the conversion pipeline.

These tests verify that:
1. Text (sentence, tokens, lemmas) is correctly transliterated
2. Codes (POS tags, entity labels, QIDs) are NOT transliterated
3. Entity alignment is preserved after transliteration
4. Both directions work: Latin→Cyrillic and Cyrillic→Latin
"""

import pytest
import spacy
import cyrtranslit
from pathlib import Path
import sys
import copy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.models.annotation_token import AnnotationToken


# --- Test Data ---
# Sample TSV content with Latin text and named entities (from itsrner_0-sr.tsv)
SAMPLE_TSV_LATIN = """\
#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value


#Text=Veoma mlada se udaje za Stefana Karačija i uspešno upravlja radnjom .
1-1	0-5	Veoma	_	_
1-2	6-11	mlada	_	_
1-3	12-14	se	_	_
1-4	15-20	udaje	_	_
1-5	21-23	za	_	_
1-6	24-31	Stefana	http://www.wikidata.org/entity/Q122382232[1]	PERS[1]
1-7	32-40	Karačija	http://www.wikidata.org/entity/Q122382232[1]	PERS[1]
1-8	41-42	i	_	_
1-9	43-50	uspešno	_	_
1-10	51-59	upravlja	_	_
1-11	60-67	radnjom	_	_
1-12	68-69	.	_	_

#Text=Elena počinje da piše o Pizi gde upoznaje Pjetra Ajrotu .
2-1	70-75	Elena	http://www.wikidata.org/entity/Q117323535	PERS
2-2	76-83	počinje	_	_
2-3	84-86	da	_	_
2-4	87-91	piše	_	_
2-5	92-93	o	_	_
2-6	94-98	Pizi	http://www.wikidata.org/entity/Q13375	LOC
2-7	99-102	gde	_	_
2-8	103-111	upoznaje	_	_
2-9	112-118	Pjetra	http://www.wikidata.org/entity/Q122397996[2]	PERS[2]
2-10	119-125	Ajrotu	http://www.wikidata.org/entity/Q122397996[2]	PERS[2]
2-11	126-127	.	_	_

#Text=On živi u Beogradu sa porodicom Petrović .
3-1	128-130	On	_	_
3-2	131-135	živi	_	_
3-3	136-137	u	_	_
3-4	138-146	Beogradu	http://www.wikidata.org/entity/Q3711	LOC
3-5	147-149	sa	_	_
3-6	150-159	porodicom	_	_
3-7	160-168	Petrović	http://www.wikidata.org/entity/Q12345	PERS
3-8	169-170	.	_	_

"""

# Expected Cyrillic equivalents
EXPECTED_CYRILLIC_TOKENS = {
    "Veoma": "Веома",
    "mlada": "млада",
    "se": "се",
    "udaje": "удаје",
    "za": "за",
    "Stefana": "Стефана",
    "Karačija": "Карачија",
    "Elena": "Елена",
    "počinje": "почиње",
    "piše": "пише",
    "Pizi": "Пизи",
    "gde": "где",
    "upoznaje": "упознаје",
    "Pjetra": "Пјетра",
    "Ajrotu": "Ајроту",
    "Beogradu": "Београду",
    "Petrović": "Петровић",
}


class MockGUITransliterator:
    """
    Mock class that replicates the GUI's transliteration methods.
    This allows us to test the transliteration logic without tkinter.
    """
    
    def transliterate_text(self, text: str, mode: str, lang: str) -> str:
        """Transliterate text between Cyrillic and Latin scripts."""
        if mode == "none" or not text:
            return text
        
        try:
            if mode == "to_latin":
                return cyrtranslit.to_latin(text, lang)
            elif mode == "to_cyrillic":
                return cyrtranslit.to_cyrillic(text, lang)
        except Exception:
            pass
        
        return text
    
    def transliterate_sentence(self, sentence: AnnotationSentence, mode: str, lang: str, lemma_layer: str = None):
        """
        Transliterate all text content in an AnnotationSentence.
        
        Transliterates: sentence.text, token.text, token.layers[lemma_layer]
        Does NOT transliterate: POS tags, entity labels, QIDs
        """
        if mode == "none":
            return
        
        # Transliterate sentence text
        if sentence.text:
            sentence.text = self.transliterate_text(sentence.text, mode, lang)
        
        # Transliterate each token's text and lemma
        for token in sentence.tokens:
            if token.text:
                token.text = self.transliterate_text(token.text, mode, lang)
            
            if lemma_layer and lemma_layer in token.layers:
                lemma = token.layers[lemma_layer]
                if lemma and lemma != "_":
                    token.layers[lemma_layer] = self.transliterate_text(lemma, mode, lang)


@pytest.fixture
def sample_tsv_file(tmp_path):
    """Create a temporary TSV file with sample Latin text."""
    tsv_file = tmp_path / "sample_latin.tsv"
    tsv_file.write_text(SAMPLE_TSV_LATIN, encoding="utf-8")
    return tsv_file


@pytest.fixture
def nlp():
    """Create a blank spaCy model."""
    return spacy.blank("sr")


@pytest.fixture
def transliterator():
    """Create a mock transliterator."""
    return MockGUITransliterator()


class TestTransliterationBasic:
    """Basic transliteration tests."""
    
    def test_cyrtranslit_to_cyrillic(self):
        """Test that cyrtranslit converts Latin to Cyrillic correctly."""
        assert cyrtranslit.to_cyrillic("Beograd", "sr") == "Београд"
        assert cyrtranslit.to_cyrillic("Stefan", "sr") == "Стефан"
        assert cyrtranslit.to_cyrillic("Elena", "sr") == "Елена"
    
    def test_cyrtranslit_to_latin(self):
        """Test that cyrtranslit converts Cyrillic to Latin correctly."""
        assert cyrtranslit.to_latin("Београд", "sr") == "Beograd"
        assert cyrtranslit.to_latin("Стефан", "sr") == "Stefan"
        assert cyrtranslit.to_latin("Елена", "sr") == "Elena"
    
    def test_mixed_text_to_latin(self):
        """Test that mixed text is normalized to Latin."""
        mixed = "Beograd Београд"
        result = cyrtranslit.to_latin(mixed, "sr")
        assert result == "Beograd Beograd"
    
    def test_mixed_text_to_cyrillic(self):
        """Test that mixed text is normalized to Cyrillic."""
        mixed = "Beograd Београд"
        result = cyrtranslit.to_cyrillic(mixed, "sr")
        assert result == "Београд Београд"


class TestSentenceTransliteration:
    """Test transliteration of AnnotationSentence objects."""
    
    def test_transliterate_sentence_to_cyrillic(self, sample_tsv_file, transliterator):
        """Test that Latin sentences are correctly transliterated to Cyrillic."""
        # Parse the TSV
        parser = WebAnnoNELParser(str(sample_tsv_file))
        sentences = parser.parse()
        
        assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}"
        
        # Make a deep copy to preserve original
        original_sentences = copy.deepcopy(sentences)
        
        # Transliterate to Cyrillic
        for sent in sentences:
            transliterator.transliterate_sentence(sent, "to_cyrillic", "sr", None)
        
        # Verify first sentence text is transliterated
        assert "Веома" in sentences[0].text
        assert "Стефана" in sentences[0].text
        assert "Карачија" in sentences[0].text
        
        # Verify tokens are transliterated
        first_sent_tokens = {t.text for t in sentences[0].tokens}
        assert "Веома" in first_sent_tokens
        assert "Стефана" in first_sent_tokens
        
        # Verify original Latin text is gone (fully converted)
        assert "Veoma" not in sentences[0].text
        assert "Stefana" not in sentences[0].text
    
    def test_transliterate_sentence_to_latin(self, transliterator):
        """Test that Cyrillic text is correctly transliterated to Latin."""
        # Create a Cyrillic sentence manually
        tokens = [
            AnnotationToken(1, 1, "Београд", 0, 7, {}),
            AnnotationToken(1, 2, "је", 8, 10, {}),
            AnnotationToken(1, 3, "главни", 11, 17, {}),
            AnnotationToken(1, 4, "град", 18, 22, {}),
            AnnotationToken(1, 5, ".", 23, 24, {}),
        ]
        sentence = AnnotationSentence(
            text="Београд је главни град .",
            tokens=tokens,
            entities=[(0, 7, "LOC", "Q3711")]
        )
        
        # Transliterate to Latin
        transliterator.transliterate_sentence(sentence, "to_latin", "sr", None)
        
        # Verify
        assert sentence.text == "Beograd je glavni grad ."
        assert sentence.tokens[0].text == "Beograd"
        assert sentence.tokens[1].text == "je"
    
    def test_entity_labels_not_transliterated(self, sample_tsv_file, transliterator):
        """Test that entity labels (PERS, LOC) are NOT transliterated."""
        parser = WebAnnoNELParser(str(sample_tsv_file))
        sentences = parser.parse()
        
        # Get original entity labels
        original_labels = []
        for sent in sentences:
            for ent in sent.entities:
                original_labels.append(ent[2])  # label is at index 2
        
        # Transliterate
        for sent in sentences:
            transliterator.transliterate_sentence(sent, "to_cyrillic", "sr", None)
        
        # Verify labels are unchanged
        new_labels = []
        for sent in sentences:
            for ent in sent.entities:
                new_labels.append(ent[2])
        
        assert original_labels == new_labels
        assert "PERS" in new_labels
        assert "LOC" in new_labels
    
    def test_entity_qids_not_transliterated(self, sample_tsv_file, transliterator):
        """Test that entity QIDs (Wikidata IDs) are NOT transliterated."""
        parser = WebAnnoNELParser(str(sample_tsv_file))
        sentences = parser.parse()
        
        # Get original QIDs
        original_qids = []
        for sent in sentences:
            for ent in sent.entities:
                original_qids.append(ent[3])  # qid is at index 3
        
        # Transliterate
        for sent in sentences:
            transliterator.transliterate_sentence(sent, "to_cyrillic", "sr", None)
        
        # Verify QIDs are unchanged
        new_qids = []
        for sent in sentences:
            for ent in sent.entities:
                new_qids.append(ent[3])
        
        assert original_qids == new_qids
        # Check specific QIDs are preserved
        qid_set = set(new_qids)
        assert "Q122382232" in qid_set or any("Q122382232" in str(q) for q in qid_set)


class TestFullPipelineWithTransliteration:
    """Test the full conversion pipeline with transliteration."""
    
    def test_convert_to_spacy_with_cyrillic_transliteration(self, sample_tsv_file, nlp, transliterator):
        """Test full pipeline: parse → transliterate → convert to spaCy."""
        # 1. Parse
        parser = WebAnnoNELParser(str(sample_tsv_file))
        sentences = parser.parse()
        
        # 2. Transliterate to Cyrillic
        for sent in sentences:
            transliterator.transliterate_sentence(sent, "to_cyrillic", "sr", None)
        
        # 3. Convert to spaCy
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=10,
            ner=True,
            nel=True
        )
        doc_bin = converter.convert(sentences)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        assert len(docs) == 1
        doc = docs[0]
        
        # 4. Verify entities exist and have correct labels
        assert len(doc.ents) > 0, "No entities found in converted doc"
        
        # Check entity labels are preserved (not transliterated)
        ent_labels = {ent.label_ for ent in doc.ents}
        assert "PERS" in ent_labels or "LOC" in ent_labels
        
        # Check that text is in Cyrillic
        doc_text = doc.text
        # Should contain Cyrillic characters
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in doc_text)
        assert has_cyrillic, f"Expected Cyrillic text, got: {doc_text[:100]}"
    
    def test_entity_text_matches_after_transliteration(self, sample_tsv_file, nlp, transliterator):
        """Test that entity text correctly reflects transliterated form."""
        # Parse
        parser = WebAnnoNELParser(str(sample_tsv_file))
        sentences = parser.parse()
        
        # Transliterate to Cyrillic
        for sent in sentences:
            transliterator.transliterate_sentence(sent, "to_cyrillic", "sr", None)
        
        # Convert
        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=10,
            ner=True,
            nel=True
        )
        doc_bin = converter.convert(sentences)
        docs = list(doc_bin.get_docs(nlp.vocab))
        doc = docs[0]
        
        # Each entity text should be in Cyrillic (if it was Latin originally)
        for ent in doc.ents:
            # Entity text should match what's in the doc
            assert ent.text == doc.text[ent.start_char:ent.end_char]
            
            # If entity text contains letters, it should be Cyrillic
            if any(c.isalpha() for c in ent.text):
                # Check it's not pure Latin (should have been transliterated)
                has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in ent.text)
                # Allow for punctuation-only entities or special cases
                if ent.text.strip(".,:;!?"):
                    assert has_cyrillic, f"Entity '{ent.text}' should be in Cyrillic"


class TestTransliterationModeNone:
    """Test that 'none' mode preserves original text."""
    
    def test_mode_none_preserves_text(self, sample_tsv_file, transliterator):
        """Test that mode='none' does not change anything."""
        parser = WebAnnoNELParser(str(sample_tsv_file))
        sentences = parser.parse()
        
        # Store original texts
        original_texts = [sent.text for sent in sentences]
        original_tokens = [[t.text for t in sent.tokens] for sent in sentences]
        
        # Apply transliteration with mode="none"
        for sent in sentences:
            transliterator.transliterate_sentence(sent, "none", "sr", None)
        
        # Verify nothing changed
        for i, sent in enumerate(sentences):
            assert sent.text == original_texts[i]
            for j, token in enumerate(sent.tokens):
                assert token.text == original_tokens[i][j]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
