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
from docx.oxml import parse_xml  # noqa: E402


def _add_text_box(paragraph, text, shape_id):
    paragraph.add_run()._r.append(
        parse_xml(
            f"""
            <mc:AlternateContent
                xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
                xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
                xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
              <mc:Choice Requires="wps">
                <w:drawing>
                  <wp:inline>
                    <wp:extent cx="914400" cy="457200"/>
                    <wp:docPr id="{shape_id}" name="Text Box {shape_id}"/>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
                        <wps:wsp>
                          <wps:cNvSpPr txBox="1"/>
                          <wps:spPr>
                            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                          </wps:spPr>
                          <wps:txbx>
                            <w:txbxContent><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:txbxContent>
                          </wps:txbx>
                          <wps:bodyPr/>
                        </wps:wsp>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing>
              </mc:Choice>
            </mc:AlternateContent>
            """
        )
    )


@pytest.mark.parametrize(
    "options", [{}, {"extract_drawings": False}], ids=["default", "disabled"]
)
def test_extract_drawings_applies_to_word_table_text_boxes(tmp_path, options):
    document = docx.Document()
    _add_text_box(document.add_paragraph("Body text"), "Body box", 1)
    cell = document.add_table(rows=1, cols=1).cell(0, 0)
    cell.text = "Cell text"
    _add_text_box(cell.paragraphs[0], "Cell box", 2)
    source = tmp_path / "drawings.docx"
    document.save(str(source))

    markdown = doc2md_convert(source, **options).markdown

    assert "Body text" in markdown
    if options:
        assert "Body box" not in markdown
        assert "<td>Cell text</td>" in markdown
    else:
        assert "> Body box" in markdown
        assert "<td>Cell text<br>[Cell box]</td>" in markdown
