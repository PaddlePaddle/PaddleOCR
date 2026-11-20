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

from paddleocr import doc2md_convert

pptx = pytest.importorskip("pptx")
from pptx.util import Inches  # noqa: E402


@pytest.mark.parametrize(
    "notes_kind, expected_notes",
    [
        ("missing_body", ""),
        ("absent", ""),
        ("present", "\n\n\n> **Notes**: Speaker text"),
    ],
)
def test_pptx_notes_body_is_optional(tmp_path, notes_kind, expected_notes):
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    shape = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(2))
    shape.text = "Slide content"
    if notes_kind == "missing_body":
        element = slide.notes_slide.notes_placeholder._element
        element.getparent().remove(element)
    elif notes_kind == "present":
        slide.notes_slide.notes_text_frame.text = "Speaker text"
    source = tmp_path / "notes.pptx"
    presentation.save(str(source))

    reopened_slide = pptx.Presentation(str(source)).slides[0]
    if notes_kind == "missing_body":
        assert reopened_slide.has_notes_slide
        assert reopened_slide.notes_slide.notes_text_frame is None

    assert doc2md_convert(source).markdown == "- Slide content" + expected_notes
