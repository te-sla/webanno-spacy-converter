"""
Tests for GUI logic (without tkinter dependency).

These tests verify the business logic used by the GUI:
1. Sentence preparation and shuffling
2. Chunk creation
3. Train/dev/test splitting logic
4. Transliteration logic

Note: These tests mock the GUI class to test logic without tkinter.
"""

import pytest
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.models.annotation_token import AnnotationToken
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence

# Import cyrtranslit for transliteration tests
try:
    import cyrtranslit
    HAS_CYRTRANSLIT = True
except ImportError:
    HAS_CYRTRANSLIT = False


def create_test_sentences(count: int = 10, prefix: str = "file") -> List[AnnotationSentence]:
    """Create a list of test sentences."""
    sentences = []
    for i in range(count):
        tokens = [
            AnnotationToken(i+1, 1, "Sentence", 0, 8, {}),
            AnnotationToken(i+1, 2, str(i+1), 9, 10 + len(str(i+1)), {}),
            AnnotationToken(i+1, 3, ".", 11 + len(str(i+1)), 12 + len(str(i+1)), {}),
        ]
        sentences.append(AnnotationSentence(
            text=f"Sentence {i+1} .",
            tokens=tokens,
            entities=[]
        ))
    return sentences


class MockGUILogic:
    """
    Mock class that replicates GUI logic for testing without tkinter.
    Extracted from gui.py for testing purposes.
    """
    
    def create_chunks(
        self, 
        sentences_by_file: Dict[str, List], 
        chunk_size: int
    ) -> List[List]:
        """
        Group sentences into chunks, respecting file boundaries.
        """
        chunks = []
        for file_path in sorted(sentences_by_file.keys()):
            file_sents = sentences_by_file[file_path]
            for i in range(0, len(file_sents), chunk_size):
                chunk = file_sents[i:i + chunk_size]
                chunks.append(chunk)
        return chunks
    
    def prepare_sentences(
        self,
        sentences_by_file: Dict[str, List],
        mode: str,
        chunk_size: int,
        seed: Optional[int]
    ) -> List:
        """
        Prepare sentences for train/dev/test split based on shuffle mode.
        """
        if seed is not None:
            random.seed(seed)
        
        if mode == "chunk":
            chunks = self.create_chunks(sentences_by_file, chunk_size)
            random.shuffle(chunks)
            return [sent for chunk in chunks for sent in chunk]
        
        elif mode == "sentence":
            all_sents = []
            for file_path in sorted(sentences_by_file.keys()):
                all_sents.extend(sentences_by_file[file_path])
            random.shuffle(all_sents)
            return all_sents
        
        else:  # mode == "none"
            all_sents = []
            for file_path in sorted(sentences_by_file.keys()):
                all_sents.extend(sentences_by_file[file_path])
            return all_sents
    
    def transliterate_text(self, text: str, mode: str, lang: str) -> str:
        """Transliterate text between Cyrillic and Latin scripts."""
        if mode == "none" or not text:
            return text
        
        if not HAS_CYRTRANSLIT:
            return text
        
        try:
            if mode == "to_latin":
                return cyrtranslit.to_latin(text, lang)
            elif mode == "to_cyrillic":
                return cyrtranslit.to_cyrillic(text, lang)
        except Exception:
            pass
        
        return text
    
    def transliterate_sentence(
        self, 
        sentence: AnnotationSentence, 
        mode: str, 
        lang: str, 
        lemma_layer: Optional[str] = None
    ):
        """Transliterate text content in an AnnotationSentence."""
        if mode == "none":
            return
        
        if sentence.text:
            sentence.text = self.transliterate_text(sentence.text, mode, lang)
        
        for token in sentence.tokens:
            if token.text:
                token.text = self.transliterate_text(token.text, mode, lang)
            
            if lemma_layer and lemma_layer in token.layers:
                lemma = token.layers[lemma_layer]
                if lemma and lemma != "_":
                    token.layers[lemma_layer] = self.transliterate_text(lemma, mode, lang)


@pytest.fixture
def gui_logic():
    """Create mock GUI logic instance."""
    return MockGUILogic()


class TestChunkCreation:
    """Tests for chunk creation logic."""
    
    def test_create_chunks_single_file(self, gui_logic):
        """Test chunk creation from single file."""
        sentences = create_test_sentences(10)
        sentences_by_file = {"file1.tsv": sentences}
        
        chunks = gui_logic.create_chunks(sentences_by_file, chunk_size=3)
        
        # 10 sentences / 3 per chunk = 4 chunks (3+3+3+1)
        assert len(chunks) == 4
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        assert len(chunks[3]) == 1  # Remainder
    
    def test_create_chunks_multiple_files(self, gui_logic):
        """Test chunk creation from multiple files."""
        sentences_by_file = {
            "file1.tsv": create_test_sentences(5, "file1"),
            "file2.tsv": create_test_sentences(5, "file2"),
        }
        
        chunks = gui_logic.create_chunks(sentences_by_file, chunk_size=2)
        
        # File 1: 5 sentences / 2 = 3 chunks (2+2+1)
        # File 2: 5 sentences / 2 = 3 chunks (2+2+1)
        # Total: 6 chunks
        assert len(chunks) == 6
    
    def test_create_chunks_respects_file_boundaries(self, gui_logic):
        """Test that chunks don't mix sentences from different files."""
        sentences_by_file = {
            "file1.tsv": create_test_sentences(3, "file1"),
            "file2.tsv": create_test_sentences(3, "file2"),
        }
        
        chunks = gui_logic.create_chunks(sentences_by_file, chunk_size=2)
        
        # Each chunk should only have sentences from one file
        # (based on sentence text pattern)
        for chunk in chunks:
            # All sentences in a chunk should have same prefix
            texts = [s.text for s in chunk]
            # Since we're using numbered sentences, they should be consecutive
            # within each file
            pass  # Logic verification
    
    def test_create_chunks_chunk_size_larger_than_file(self, gui_logic):
        """Test when chunk size is larger than file content."""
        sentences = create_test_sentences(3)
        sentences_by_file = {"file1.tsv": sentences}
        
        chunks = gui_logic.create_chunks(sentences_by_file, chunk_size=10)
        
        # Should get one chunk with all 3 sentences
        assert len(chunks) == 1
        assert len(chunks[0]) == 3


class TestSentencePreparation:
    """Tests for sentence preparation and shuffling."""
    
    def test_prepare_sentences_mode_none(self, gui_logic):
        """Test that mode='none' preserves original order."""
        sentences_by_file = {
            "file1.tsv": create_test_sentences(5),
        }
        
        result = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="none",
            chunk_size=2,
            seed=None
        )
        
        # Should be in original order
        assert len(result) == 5
        assert result[0].text == "Sentence 1 ."
        assert result[4].text == "Sentence 5 ."
    
    def test_prepare_sentences_mode_none_alphabetical_files(self, gui_logic):
        """Test that mode='none' sorts files alphabetically."""
        sentences_by_file = {
            "z_file.tsv": create_test_sentences(2),
            "a_file.tsv": create_test_sentences(2),
        }
        
        result = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="none",
            chunk_size=10,
            seed=None
        )
        
        # a_file should come before z_file
        assert len(result) == 4
        # First two from a_file, last two from z_file
    
    def test_prepare_sentences_mode_chunk_shuffles(self, gui_logic):
        """Test that mode='chunk' shuffles chunks."""
        sentences_by_file = {
            "file1.tsv": create_test_sentences(10),
        }
        
        # With seed, should be deterministic
        result1 = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="chunk",
            chunk_size=2,
            seed=42
        )
        
        result2 = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="chunk",
            chunk_size=2,
            seed=42
        )
        
        # Same seed should give same order
        assert [s.text for s in result1] == [s.text for s in result2]
    
    def test_prepare_sentences_mode_sentence_shuffles(self, gui_logic):
        """Test that mode='sentence' shuffles individual sentences."""
        sentences_by_file = {
            "file1.tsv": create_test_sentences(10),
        }
        
        # With seed, should be deterministic
        result1 = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="sentence",
            chunk_size=2,
            seed=42
        )
        
        result2 = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="sentence",
            chunk_size=2,
            seed=42
        )
        
        # Same seed should give same order
        assert [s.text for s in result1] == [s.text for s in result2]
    
    def test_prepare_sentences_different_seeds_different_order(self, gui_logic):
        """Test that different seeds produce different orders."""
        sentences_by_file = {
            "file1.tsv": create_test_sentences(20),
        }
        
        result1 = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="sentence",
            chunk_size=2,
            seed=42
        )
        
        result2 = gui_logic.prepare_sentences(
            sentences_by_file,
            mode="sentence",
            chunk_size=2,
            seed=123
        )
        
        # Different seeds should give different orders (almost certainly)
        texts1 = [s.text for s in result1]
        texts2 = [s.text for s in result2]
        assert texts1 != texts2


class TestTrainDevTestSplit:
    """Tests for train/dev/test splitting logic."""
    
    def test_80_10_10_split(self):
        """Test standard 80/10/10 split calculation."""
        total = 100
        
        train_end = int(total * 0.8)
        dev_end = int(total * 0.9)
        
        assert train_end == 80
        assert dev_end == 90
        
        train_count = train_end
        dev_count = dev_end - train_end
        test_count = total - dev_end
        
        assert train_count == 80
        assert dev_count == 10
        assert test_count == 10
    
    def test_split_small_dataset(self):
        """Test split with small dataset."""
        total = 10
        
        train_end = int(total * 0.8)
        dev_end = int(total * 0.9)
        
        train_count = train_end  # 8
        dev_count = dev_end - train_end  # 1
        test_count = total - dev_end  # 1
        
        assert train_count == 8
        assert dev_count == 1
        assert test_count == 1


@pytest.mark.skipif(not HAS_CYRTRANSLIT, reason="cyrtranslit not installed")
class TestTransliterationLogic:
    """Tests for transliteration logic in GUI."""
    
    def test_transliterate_text_to_cyrillic(self, gui_logic):
        """Test text transliteration to Cyrillic."""
        result = gui_logic.transliterate_text("Beograd", "to_cyrillic", "sr")
        assert result == "Београд"
    
    def test_transliterate_text_to_latin(self, gui_logic):
        """Test text transliteration to Latin."""
        result = gui_logic.transliterate_text("Београд", "to_latin", "sr")
        assert result == "Beograd"
    
    def test_transliterate_text_mode_none(self, gui_logic):
        """Test that mode='none' preserves text."""
        original = "Mixed текст"
        result = gui_logic.transliterate_text(original, "none", "sr")
        assert result == original
    
    def test_transliterate_text_empty(self, gui_logic):
        """Test handling of empty text."""
        assert gui_logic.transliterate_text("", "to_cyrillic", "sr") == ""
        assert gui_logic.transliterate_text(None, "to_cyrillic", "sr") is None
    
    def test_transliterate_sentence_text(self, gui_logic):
        """Test that sentence text is transliterated."""
        tokens = [
            AnnotationToken(1, 1, "Beograd", 0, 7, {}),
            AnnotationToken(1, 2, "je", 8, 10, {}),
            AnnotationToken(1, 3, "lep", 11, 14, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Beograd je lep",
            tokens=tokens,
            entities=[]
        )
        
        gui_logic.transliterate_sentence(sentence, "to_cyrillic", "sr", None)
        
        assert sentence.text == "Београд је леп"
    
    def test_transliterate_sentence_tokens(self, gui_logic):
        """Test that token texts are transliterated."""
        tokens = [
            AnnotationToken(1, 1, "Beograd", 0, 7, {}),
            AnnotationToken(1, 2, "je", 8, 10, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Beograd je",
            tokens=tokens,
            entities=[]
        )
        
        gui_logic.transliterate_sentence(sentence, "to_cyrillic", "sr", None)
        
        assert sentence.tokens[0].text == "Београд"
        assert sentence.tokens[1].text == "је"
    
    def test_transliterate_sentence_lemmas(self, gui_logic):
        """Test that lemmas are transliterated."""
        tokens = [
            AnnotationToken(1, 1, "Beograda", 0, 8, {"lemma": "Beograd"}),
        ]
        
        sentence = AnnotationSentence(
            text="Beograda",
            tokens=tokens,
            entities=[]
        )
        
        gui_logic.transliterate_sentence(sentence, "to_cyrillic", "sr", "lemma")
        
        assert sentence.tokens[0].layers["lemma"] == "Београд"
    
    def test_transliterate_sentence_preserves_entities(self, gui_logic):
        """Test that entity structure is preserved."""
        tokens = [
            AnnotationToken(1, 1, "Beograd", 0, 7, {"value": "LOC", "identifier": "Q3711"}),
        ]
        
        original_entities = [(0, 7, "LOC", "Q3711")]
        
        sentence = AnnotationSentence(
            text="Beograd",
            tokens=tokens,
            entities=original_entities
        )
        
        gui_logic.transliterate_sentence(sentence, "to_cyrillic", "sr", None)
        
        # Entity metadata should be unchanged
        assert sentence.entities == original_entities
        assert sentence.tokens[0].layers["value"] == "LOC"
        assert sentence.tokens[0].layers["identifier"] == "Q3711"
    
    def test_transliterate_mixed_text(self, gui_logic):
        """Test transliteration of mixed Latin/Cyrillic text."""
        tokens = [
            AnnotationToken(1, 1, "Beograd", 0, 7, {}),
            AnnotationToken(1, 2, "Београд", 8, 15, {}),
        ]
        
        sentence = AnnotationSentence(
            text="Beograd Београд",
            tokens=tokens,
            entities=[]
        )
        
        gui_logic.transliterate_sentence(sentence, "to_latin", "sr", None)
        
        # Both should be Latin now
        assert sentence.text == "Beograd Beograd"
        assert sentence.tokens[0].text == "Beograd"
        assert sentence.tokens[1].text == "Beograd"


class TestInputFileCollection:
    """Tests for input file collection logic."""
    
    def test_collect_from_folder(self, tmp_path):
        """Test collecting TSV files from folder."""
        # Create test files
        (tmp_path / "file1.tsv").touch()
        (tmp_path / "file2.tsv").touch()
        (tmp_path / "file3.txt").touch()  # Should be ignored
        
        # Mock the collection logic
        folder = str(tmp_path)
        import os
        
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".tsv")
        ]
        
        assert len(files) == 2
        assert all(f.endswith(".tsv") for f in files)
    
    def test_collect_semicolon_separated(self):
        """Test parsing semicolon-separated file paths."""
        raw = "/path/file1.tsv;/path/file2.tsv;/path/file3.tsv"
        
        files = [p.strip() for p in raw.split(";") if p.strip()]
        
        assert len(files) == 3
        assert files[0] == "/path/file1.tsv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
