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

import pytest
from pptx import Presentation

from paddleocr import doc2md_convert


@pytest.mark.parametrize(
    ("note", "expected_lines"),
    [
        ("", []),
        ("One reminder", ["> **Notes**: One reminder"]),
        (
            "First paragraph\n\nSecond paragraph\nFinal reminder",
            [
                "> **Notes**: First paragraph",
                ">",
                "> Second paragraph",
                "> Final reminder",
            ],
        ),
    ],
    ids=["empty", "single-line", "multiline"],
)
def test_pptx_speaker_notes_stay_in_block_quote(tmp_path, note, expected_lines):
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.notes_slide.notes_text_frame.text = note
    deck = tmp_path / "notes.pptx"
    presentation.save(str(deck))

    markdown = doc2md_convert(str(deck)).markdown
    assert markdown.strip().splitlines() == expected_lines
