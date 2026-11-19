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

docx = pytest.importorskip("docx")
html = pytest.importorskip("lxml.html")


@pytest.mark.parametrize(
    "text", ["Region <ROI> and tooth", "Literal &copy; <br> marker", "Plain text"]
)
def test_docx_table_preserves_literal_text(tmp_path, text):
    document = docx.Document()
    document.add_paragraph("Body first\nBody second")
    paragraph = document.add_table(rows=1, cols=1).cell(0, 0).paragraphs[0]
    run = paragraph.add_run(text)
    run.bold = True
    run.add_break()
    paragraph.add_run("tail").italic = True
    source = tmp_path / "literal.docx"
    document.save(str(source))

    markdown = doc2md_convert(source).markdown

    assert markdown.startswith("Body first<br>\nBody second\n\n")
    table = html.fromstring(markdown[markdown.index("<table>") :])
    cell = table.xpath(".//td")[0]
    assert cell.text_content() == text + "\ntail"
    assert cell.xpath("./b")[0].text == text
    assert cell.xpath("./i")[0].text == "tail"
    assert len(cell.xpath(".//br")) == 1
