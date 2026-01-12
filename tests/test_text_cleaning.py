"""
Tests for text cleaning utilities.

These tests ensure that:
1. Hidden characters (ZWSP, BOM, etc.) are properly removed
2. Normal text is preserved
3. Statistics are correctly reported
4. Edge cases (empty strings, None) are handled
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from webanno_spacy_converter.utils.text_cleaning import (
    clean_text_for_transformers,
    clean_text_with_stats,
    has_problematic_chars,
    CHARS_TO_REMOVE
)


class TestCleanTextForTransformers:
    """Tests for clean_text_for_transformers function."""
    
    def test_remove_zero_width_space(self):
        """Test removal of Zero-Width Space (ZWSP)."""
        text = "hello\u200bworld"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_remove_zero_width_non_joiner(self):
        """Test removal of Zero-Width Non-Joiner (ZWNJ)."""
        text = "hello\u200cworld"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_remove_zero_width_joiner(self):
        """Test removal of Zero-Width Joiner (ZWJ)."""
        text = "hello\u200dworld"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_remove_word_joiner(self):
        """Test removal of Word Joiner."""
        text = "hello\u2060world"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_remove_soft_hyphen(self):
        """Test removal of Soft Hyphen."""
        text = "hello\xadworld"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_remove_bom(self):
        """Test removal of BOM (mid-text)."""
        text = "hello\ufeffworld"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_remove_multiple_hidden_chars(self):
        """Test removal of multiple different hidden characters."""
        text = "a\u200bb\u200cc\u200dd\u2060e\xadf\ufeffg"
        result = clean_text_for_transformers(text)
        assert result == "abcdefg"
    
    def test_remove_consecutive_hidden_chars(self):
        """Test removal of consecutive hidden characters."""
        text = "hello\u200b\u200b\u200bworld"
        result = clean_text_for_transformers(text)
        assert result == "helloworld"
    
    def test_preserve_normal_text(self):
        """Test that normal text is preserved unchanged."""
        text = "Hello, World! This is a test."
        result = clean_text_for_transformers(text)
        assert result == text
    
    def test_preserve_normal_whitespace(self):
        """Test that normal whitespace (spaces, tabs, newlines) is preserved."""
        text = "Hello World\n\tNew line and tab"
        result = clean_text_for_transformers(text)
        assert result == text
    
    def test_preserve_unicode(self):
        """Test that regular Unicode characters are preserved."""
        text = "Привет мир 日本語 🎉"
        result = clean_text_for_transformers(text)
        assert result == text
    
    def test_empty_string(self):
        """Test handling of empty string."""
        result = clean_text_for_transformers("")
        assert result == ""
    
    def test_none_input(self):
        """Test handling of None input."""
        result = clean_text_for_transformers(None)
        assert result is None
    
    def test_only_hidden_chars(self):
        """Test string containing only hidden characters."""
        text = "\u200b\u200c\u200d"
        result = clean_text_for_transformers(text)
        assert result == ""


class TestCleanTextWithStats:
    """Tests for clean_text_with_stats function."""
    
    def test_returns_cleaned_text(self):
        """Test that function returns cleaned text."""
        text = "hello\u200bworld"
        cleaned, stats = clean_text_with_stats(text)
        assert cleaned == "helloworld"
    
    def test_returns_correct_stats(self):
        """Test that function returns correct statistics."""
        text = "a\u200bb\u200bc"
        cleaned, stats = clean_text_with_stats(text)
        
        assert "Zero-Width Space (ZWSP)" in stats
        assert stats["Zero-Width Space (ZWSP)"] == 2
    
    def test_multiple_char_types_stats(self):
        """Test statistics for multiple character types."""
        text = "a\u200bb\u200cc"  # ZWSP and ZWNJ
        cleaned, stats = clean_text_with_stats(text)
        
        assert len(stats) == 2
        assert stats.get("Zero-Width Space (ZWSP)") == 1
        assert stats.get("Zero-Width Non-Joiner (ZWNJ)") == 1
    
    def test_no_hidden_chars_empty_stats(self):
        """Test that clean text returns empty stats."""
        text = "Hello World"
        cleaned, stats = clean_text_with_stats(text)
        
        assert cleaned == text
        assert stats == {}
    
    def test_empty_string_stats(self):
        """Test empty string returns empty stats."""
        cleaned, stats = clean_text_with_stats("")
        
        assert cleaned == ""
        assert stats == {}


class TestHasProblematicChars:
    """Tests for has_problematic_chars function."""
    
    def test_detects_zwsp(self):
        """Test detection of Zero-Width Space."""
        assert has_problematic_chars("hello\u200bworld") is True
    
    def test_detects_zwnj(self):
        """Test detection of Zero-Width Non-Joiner."""
        assert has_problematic_chars("hello\u200cworld") is True
    
    def test_detects_zwj(self):
        """Test detection of Zero-Width Joiner."""
        assert has_problematic_chars("hello\u200dworld") is True
    
    def test_detects_word_joiner(self):
        """Test detection of Word Joiner."""
        assert has_problematic_chars("hello\u2060world") is True
    
    def test_detects_soft_hyphen(self):
        """Test detection of Soft Hyphen."""
        assert has_problematic_chars("hello\xadworld") is True
    
    def test_detects_bom(self):
        """Test detection of BOM."""
        assert has_problematic_chars("hello\ufeffworld") is True
    
    def test_clean_text_returns_false(self):
        """Test that clean text returns False."""
        assert has_problematic_chars("Hello World!") is False
    
    def test_unicode_text_returns_false(self):
        """Test that normal Unicode returns False."""
        assert has_problematic_chars("Привет мир") is False
    
    def test_empty_string_returns_false(self):
        """Test that empty string returns False."""
        assert has_problematic_chars("") is False
    
    def test_none_returns_false(self):
        """Test that None returns False."""
        assert has_problematic_chars(None) is False


class TestCharsToRemoveConstant:
    """Tests for the CHARS_TO_REMOVE constant."""
    
    def test_contains_expected_chars(self):
        """Test that constant contains all expected characters."""
        expected_chars = [
            '\u200b',  # ZWSP
            '\u200c',  # ZWNJ
            '\u200d',  # ZWJ
            '\u2060',  # Word Joiner
            '\xad',    # Soft Hyphen
            '\ufeff',  # BOM
        ]
        
        for char in expected_chars:
            assert char in CHARS_TO_REMOVE, f"Missing character: {repr(char)}"
    
    def test_all_chars_have_descriptions(self):
        """Test that all characters have descriptions."""
        for char, description in CHARS_TO_REMOVE.items():
            assert isinstance(description, str)
            assert len(description) > 0


class TestRealWorldCases:
    """Tests based on real-world problematic text samples."""
    
    def test_copied_from_pdf(self):
        """Test text that might be copied from PDF with hidden chars."""
        # PDFs often have soft hyphens for line breaks
        text = "docu\xadment\xadation"
        result = clean_text_for_transformers(text)
        assert result == "documentation"
    
    def test_web_copy_paste(self):
        """Test text copied from web with ZWSP."""
        # Some websites insert ZWSP for word-breaking hints
        text = "super\u200bcali\u200bfragi\u200blistic"
        result = clean_text_for_transformers(text)
        assert result == "supercalifragilistic"
    
    def test_bom_at_start(self):
        """Test file that starts with BOM."""
        text = "\ufeffHello World"
        result = clean_text_for_transformers(text)
        assert result == "Hello World"
    
    def test_mixed_visible_and_hidden(self):
        """Test realistic mixed content."""
        text = "The\u200b quick\u200c brown\u200d fox"
        result = clean_text_for_transformers(text)
        assert result == "The quick brown fox"


class TestTransformerCompatibility:
    """Tests to ensure cleaned text works with transformer models."""
    
    def test_cleaned_text_tokenizable(self):
        """Test that cleaned text can be processed by basic tokenization."""
        text = "hello\u200bworld\u200ctest\u200dcase"
        cleaned = clean_text_for_transformers(text)
        
        # Basic split should work
        tokens = cleaned.split()
        assert len(tokens) == 1  # Should be one word now
        assert tokens[0] == "helloworldtestcase"
    
    def test_preserves_sentence_structure(self):
        """Test that sentence structure is preserved."""
        text = "First sentence\u200b. Second sentence\u200b."
        cleaned = clean_text_for_transformers(text)
        
        # Should still have two sentences
        assert cleaned.count(".") == 2
        assert "First sentence" in cleaned
        assert "Second sentence" in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
