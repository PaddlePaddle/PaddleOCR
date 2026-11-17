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


def test_textbox_preserves_soft_line_breaks(tmp_path):
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    frame = slide.shapes.add_textbox(
        Inches(1), Inches(1), Inches(5), Inches(2)
    ).text_frame
    paragraph = frame.paragraphs[0]
    first = paragraph.add_run()
    first.text = "Before"
    first.font.bold = True
    paragraph.add_line_break()
    paragraph.add_line_break()
    linked = paragraph.add_run()
    linked.text = "After"
    linked.hyperlink.address = "https://example.com/"
    paragraph.add_run().text = " line break"
    frame.add_paragraph().text = "Separate paragraph"
    frame.add_paragraph().add_line_break()
    source = tmp_path / "line-breaks.pptx"
    presentation.save(str(source))

    assert doc2md_convert(source).markdown == (
        "- **Before**<br><br>[After](https://example.com/) line break\n"
        "- Separate paragraph"
    )
