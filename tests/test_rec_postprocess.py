import os
import sys

import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from ppocr.postprocess.rec_postprocess import BaseRecLabelDecode


class TestBaseRecLabelDecode:
    """Tests for BaseRecLabelDecode.get_word_info() method."""

    @pytest.fixture
    def decoder(self):
        """Create a BaseRecLabelDecode instance for testing."""
        return BaseRecLabelDecode()

    def test_get_word_info_with_german_accented_chars(self, decoder):
        """Test that German words with accented characters are not split."""
        text = "Grüßen"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1, "German word should not be split"
        assert "".join(word_list[0]) == "Grüßen"
        assert state_list[0] == "en&num"

    def test_get_word_info_with_longer_german_word(self, decoder):
        """Test longer German words with umlauts remain intact."""
        text = "ungewöhnlichen"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1, "German word should not be split"
        assert "".join(word_list[0]) == "ungewöhnlichen"
        assert state_list[0] == "en&num"

    def test_get_word_info_with_french_accented_chars(self, decoder):
        """Test French words with accented characters."""
        text = "café"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1, "French word should not be split"
        assert "".join(word_list[0]) == "café"

    def test_get_word_info_underscore_as_splitter(self, decoder):
        """Test that underscores are treated as word splitters."""
        text = "hello_world"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2, "Underscore should split words"
        assert "".join(word_list[0]) == "hello"
        assert "".join(word_list[1]) == "world"

    def test_get_word_info_with_mixed_content(self, decoder):
        """Test mixed content with spaces and accented characters."""
        text = "Grüßen Sie"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2, "Should have two words separated by space"
        assert "".join(word_list[0]) == "Grüßen"
        assert "".join(word_list[1]) == "Sie"

    def test_get_word_info_with_french_apostrophe(self, decoder):
        """Test French words with apostrophes like n'êtes."""
        text = "n'êtes"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        # Apostrophe should keep words connected in French context
        assert len(word_list) == 1, "French apostrophe should connect words"
        assert "".join(word_list[0]) == "n'êtes"

    def test_get_word_info_with_ascii_only(self, decoder):
        """Test backward compatibility with ASCII-only text."""
        text = "hello world"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2
        assert "".join(word_list[0]) == "hello"
        assert "".join(word_list[1]) == "world"

    def test_get_word_info_with_numbers(self, decoder):
        """Test that numbers are properly handled."""
        text = "VGG-16"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1, "Hyphenated word-number should stay together"
        assert "".join(word_list[0]) == "VGG-16"

    def test_get_word_info_with_floating_point(self, decoder):
        """Test floating point numbers stay together."""
        text = "price 3.14"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2
        assert "".join(word_list[0]) == "price"
        assert "".join(word_list[1]) == "3.14"

    def test_get_word_info_with_chinese(self, decoder):
        """Test Chinese characters are properly grouped."""
        text = "你好啊"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "你好啊"
        assert state_list[0] == "cn"
