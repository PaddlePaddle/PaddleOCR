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


docx = pytest.importorskip("docx")
from docx.shared import Inches
from lxml import html
from PIL import Image

from paddleocr import doc2md_convert


@pytest.mark.parametrize("nested", [False, True])
def test_docx_table_block_order(tmp_path, nested):
    """Preserve paragraph and table order through multiple nesting levels.

    Args:
        tmp_path: Temporary directory for the generated DOCX document.
        nested: Whether to add two nested tables.

    Returns:
        None.
    """
    document = docx.Document()
    outer = document.add_table(rows=1, cols=1).cell(0, 0)
    outer.paragraphs[0].text = "Outer before"
    expected = ["Outer before"]
    if nested:
        inner = outer.add_table(rows=1, cols=1).cell(0, 0)
        inner.paragraphs[0].text = "Inner before"
        inner.add_table(rows=1, cols=1).cell(0, 0).text = "Deep value"
        inner.add_paragraph("Inner after")
        expected += ["Inner before", "Deep value", "Inner after"]
    outer.add_paragraph("Outer after")
    expected.append("Outer after")
    source = tmp_path / "ordered.docx"
    document.save(source)

    converted = html.document_fromstring(doc2md_convert(source).markdown)
    assert [text.strip() for text in converted.itertext() if text.strip()] == expected
    assert len(converted.xpath("//table")) == (3 if nested else 1)


def test_docx_nested_merged_cells_are_emitted_once(tmp_path):
    """Preserve nested cell spans without duplicating their contents.

    Args:
        tmp_path: Temporary directory for the generated DOCX document.

    Returns:
        None.
    """
    document = docx.Document()
    outer = document.add_table(rows=1, cols=2)
    cell = outer.cell(0, 0).merge(outer.cell(0, 1))
    inner = cell.add_table(rows=2, cols=2)
    inner.cell(0, 0).merge(inner.cell(1, 0)).text = "Shared entry"
    inner.cell(0, 1).text = "First right"
    inner.cell(1, 1).text = "Second right"
    source = tmp_path / "merged.docx"
    document.save(source)

    converted = html.document_fromstring(doc2md_convert(source).markdown)
    assert [text.strip() for text in converted.itertext() if text.strip()] == [
        "Shared entry",
        "First right",
        "Second right",
    ]
    assert len(converted.xpath("//table")) == 2
    assert len(converted.xpath("//td[@colspan='2']")) == 1
    assert len(converted.xpath("//th[@rowspan='2']")) == 1


def test_docx_nested_images_share_output_resources(tmp_path):
    """Save document and nested-table images with distinct resource names.

    Args:
        tmp_path: Temporary directory for generated documents and output.

    Returns:
        None.
    """
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "blue").save(second)
    document = docx.Document()
    document.add_paragraph().add_run().add_picture(str(first), width=Inches(0.25))
    outer = document.add_table(rows=1, cols=1).cell(0, 0)
    inner = outer.add_table(rows=1, cols=1).cell(0, 0)
    inner.paragraphs[0].add_run().add_picture(str(second), width=Inches(0.25))
    source = tmp_path / "images.docx"
    document.save(source)
    output = tmp_path / "converted" / "report.md"

    result = doc2md_convert(source, output=output)
    expected = {
        "images/image1.png": first.read_bytes(),
        "images/image2.png": second.read_bytes(),
    }
    assert result.images == expected
    assert html.document_fromstring(result.markdown).xpath("//img/@src") == list(
        expected
    )
    assert output.read_text(encoding="utf-8") == result.markdown
    for name, data in expected.items():
        assert (output.parent / name).read_bytes() == data
