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

"""Tests for clamping Word Heading 7-9 styles to Markdown H6 (issue #18411)."""

import re

import pytest

docx = pytest.importorskip("docx")

from paddleocr._doc2md.converters.docx import (  # noqa: E402
    _detect_heading_level,
    _get_body_font_size,
)


def _make_docx_with_deep_headings():
    doc = docx.Document()
    for level in range(1, 10):
        doc.add_paragraph(f"Section level {level}", style=f"Heading {level}")
    doc.add_paragraph("Body text.")
    return doc


@pytest.mark.parametrize("level", range(1, 10))
def test_detect_heading_level_is_clamped(level):
    doc = _make_docx_with_deep_headings()
    para = doc.paragraphs[level - 1]
    assert para.style.name == f"Heading {level}"
    assert _detect_heading_level(para, _get_body_font_size(doc)) == min(level, 6)


def test_docx_to_markdown_emits_at_most_six_hashes(tmp_path):
    from paddleocr._doc2md.converters.docx import DocxConverter

    path = tmp_path / "deep_headings.docx"
    _make_docx_with_deep_headings().save(str(path))

    markdown = DocxConverter().convert_file(path).markdown

    # Word's built-in Heading 4/6/8 styles are italic, so match on the level
    # prefix only.
    levels = {}
    for line in markdown.splitlines():
        match = re.match(r"^(#+) \*?Section level (\d)", line)
        if match:
            levels[int(match.group(2))] = len(match.group(1))
    assert levels == {level: min(level, 6) for level in range(1, 10)}
