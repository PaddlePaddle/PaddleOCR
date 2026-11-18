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
from docx.oxml import OxmlElement  # noqa: E402


@pytest.mark.parametrize("in_table", [False, True], ids=["body", "table"])
def test_docx_math_with_plain_text(tmp_path, in_table):
    document = docx.Document()
    if in_table:
        paragraph = document.add_table(rows=1, cols=1).cell(0, 0).paragraphs[0]
    else:
        paragraph = document.add_paragraph()
    paragraph.add_run("before ")
    equation = OxmlElement("m:oMath")
    run = OxmlElement("m:r")
    text = OxmlElement("m:t")
    text.text = "x"
    run.append(text)
    equation.append(run)
    paragraph._p.append(equation)
    paragraph.add_run(" after")
    source = tmp_path / "equation.docx"
    document.save(str(source))

    markdown = doc2md_convert(source).markdown

    expected = "before $x$ after"
    if in_table:
        expected = f"<table>\n<tr>\n<td>{expected}</td>\n</tr>\n</table>"
    assert markdown == expected
