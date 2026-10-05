# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the RTL (reverse=True) word-box column alignment in rec_postprocess.

The decoder reverses arabic text with pred_reverse before get_word_info, so the
selection columns must be reordered by the same token mapping. These tests load
the module directly with a paddle stub to avoid the full paddle runtime.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_rec_postprocess():
    paddle_stub = types.ModuleType("paddle")
    nn_stub = types.ModuleType("paddle.nn")
    functional_stub = types.ModuleType("paddle.nn.functional")
    paddle_stub.nn = nn_stub
    nn_stub.functional = functional_stub
    for name, mod in (
        ("paddle", paddle_stub),
        ("paddle.nn", nn_stub),
        ("paddle.nn.functional", functional_stub),
    ):
        sys.modules.setdefault(name, mod)

    spec = importlib.util.spec_from_file_location(
        "ppocr.postprocess.rec_postprocess",
        REPO_ROOT / "ppocr" / "postprocess" / "rec_postprocess.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_decoder(tmp_path, reverse):
    dict_file = tmp_path / ("arabic_dict.txt" if reverse else "dict.txt")
    chars = "abcdefghijklmnopqrstuvwxyz" + "ابحرم"
    dict_file.write_text("\n".join(chars) + "\n", encoding="utf-8")
    return _import_rec_postprocess().BaseRecLabelDecode(
        character_dict_path=str(dict_file), use_space_char=True
    )


# "Hello مرحبا": LTR run "Hello " (indices 0-5, space joins the run) then five
# arabic letters (indices 6-10). pred_reverse keeps the LTR run whole, so the
# reversed order is the five arabic letters, then "Hello ".
ARABIC_TEXT = "Hello مرحبا"
REVERSED_CHARS = [10, 9, 8, 7, 6, 0, 1, 2, 3, 4, 5]


def _selection_with_leading_blanks(n_valid, n_blank):
    selection = np.array([False] * n_blank + [True] * n_valid)
    return selection


class TestReverseSelection:
    def test_columns_follow_pred_reverse_units(self, tmp_path):
        decoder = _make_decoder(tmp_path, reverse=True)
        selection = _selection_with_leading_blanks(len(ARABIC_TEXT), 3)

        selection_re = decoder.reverse_selection(ARABIC_TEXT, selection)

        expected = np.arange(3, 3 + len(ARABIC_TEXT))[REVERSED_CHARS]
        assert selection_re.tolist() == expected.tolist()

    def test_reversed_text_matches_selection_order(self, tmp_path):
        decoder = _make_decoder(tmp_path, reverse=True)
        text_re = decoder.pred_reverse(ARABIC_TEXT)
        assert list(text_re) == [ARABIC_TEXT[i] for i in REVERSED_CHARS]


class TestRtlWordBoxAlignment:
    @staticmethod
    def _flat(cols):
        return [c for group in cols for c in group]

    def test_rtl_columns_match_original_positions(self, tmp_path):
        decoder = _make_decoder(tmp_path, reverse=True)
        selection = _selection_with_leading_blanks(len(ARABIC_TEXT), 3)
        selection_re = decoder.reverse_selection(ARABIC_TEXT, selection)
        text_re = decoder.pred_reverse(ARABIC_TEXT)

        _, word_col_list, _ = decoder.get_word_info(text_re, selection_re)

        # The reported columns must be the original feature columns of each
        # non-separator character, in the reversed text's reading order. The
        # arabic run and "Hello" land in separate word groups; the space
        # splits them and its column is not reported.
        expected = [
            c
            for c in np.arange(3, 3 + len(ARABIC_TEXT))[REVERSED_CHARS].tolist()
            if c != 8
        ]
        assert self._flat(word_col_list) == expected

    def test_ltr_behavior_unchanged(self, tmp_path):
        decoder = _make_decoder(tmp_path, reverse=False)
        selection = _selection_with_leading_blanks(len(ARABIC_TEXT), 3)

        _, word_col_list, _ = decoder.get_word_info(ARABIC_TEXT, selection)

        assert self._flat(word_col_list) == [
            c for c in range(3, 3 + len(ARABIC_TEXT)) if c != 8
        ]

    def test_blank_interleaving_keeps_column_mapping(self, tmp_path):
        decoder = _make_decoder(tmp_path, reverse=True)
        # Blanks in the middle as well: columns 3-7 ("Hello") and 9-13
        # (arabic) survive; the space at column 8 is dropped.
        selection = np.array([False] * 3 + [True] * 5 + [False] + [True] * 5 + [False])
        kept = [3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
        # Feature columns 3-7 decode to "Hello", column 8 is a blank, 9-13
        # decode to the five arabic letters.
        kept_text = "Helloمرحبا"
        text_re = decoder.pred_reverse(kept_text)

        selection_re = decoder.reverse_selection(kept_text, selection)

        # "Helloمرحبا" (no space): units are "Hello" and five single arabic
        # letters, reversed to the letters first then "Hello".
        kept_map = [9, 8, 7, 6, 5, 0, 1, 2, 3, 4]
        expected = np.array(kept)[kept_map]
        assert selection_re.tolist() == expected.tolist()
        _, word_col_list, _ = decoder.get_word_info(text_re, selection_re)
        assert self._flat(word_col_list) == expected.tolist()
