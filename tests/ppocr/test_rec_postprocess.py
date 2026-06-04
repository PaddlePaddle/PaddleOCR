# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_base_rec_label_decode():
    spec = importlib.util.spec_from_file_location(
        "ppocr.postprocess.rec_postprocess",
        REPO_ROOT / "ppocr" / "postprocess" / "rec_postprocess.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.BaseRecLabelDecode


BaseRecLabelDecode = _load_base_rec_label_decode()


class TestBaseRecLabelDecode:
    """Tests for BaseRecLabelDecode.get_word_info()."""

    @pytest.fixture
    def decoder(self):
        return BaseRecLabelDecode()

    def test_get_word_info_with_german_accented_chars(self, decoder):
        text = "Grüßen"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "Grüßen"
        assert state_list[0] == "en&num"

    def test_get_word_info_with_longer_german_word(self, decoder):
        text = "ungewöhnlichen"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "ungewöhnlichen"
        assert state_list[0] == "en&num"

    def test_get_word_info_with_french_accented_chars(self, decoder):
        text = "café"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "café"

    def test_get_word_info_underscore_as_splitter(self, decoder):
        text = "hello_world"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2
        assert "".join(word_list[0]) == "hello"
        assert "".join(word_list[1]) == "world"

    def test_get_word_info_with_mixed_content(self, decoder):
        text = "Grüßen Sie"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2
        assert "".join(word_list[0]) == "Grüßen"
        assert "".join(word_list[1]) == "Sie"

    def test_get_word_info_with_french_apostrophe(self, decoder):
        text = "n'êtes"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "n'êtes"

    @pytest.mark.parametrize(
        "text,expected_word_count,expected_joined_words",
        [
            ("été", 1, ["été"]),
            ("français", 1, ["français"]),
            ("élève", 1, ["élève"]),
            ("à demain", 2, ["à", "demain"]),
        ],
    )
    def test_get_word_info_french_accented_words(
        self, decoder, text, expected_word_count, expected_joined_words
    ):
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == expected_word_count
        assert ["".join(word) for word in word_list] == expected_joined_words
        assert all(state == "en&num" for state in state_list)

    def test_get_word_info_french_complex_sentence(self, decoder):
        text = "C'était très français"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert ["".join(word) for word in word_list] == [
            "C'était",
            "très",
            "français",
        ]
        assert state_list == ["en&num", "en&num", "en&num"]

    def test_get_word_info_with_ascii_only(self, decoder):
        text = "hello world"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2
        assert "".join(word_list[0]) == "hello"
        assert "".join(word_list[1]) == "world"

    def test_get_word_info_with_numbers(self, decoder):
        text = "VGG-16"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "VGG-16"

    def test_get_word_info_with_floating_point(self, decoder):
        text = "price 3.14"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 2
        assert "".join(word_list[0]) == "price"
        assert "".join(word_list[1]) == "3.14"

    def test_get_word_info_with_chinese(self, decoder):
        text = "你好啊"
        selection = np.ones(len(text), dtype=bool)
        word_list, _, state_list = decoder.get_word_info(text, selection)
        assert len(word_list) == 1
        assert "".join(word_list[0]) == "你好啊"
        assert state_list[0] == "cn"
