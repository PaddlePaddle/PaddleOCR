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

from copy import deepcopy

import pytest
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn

from paddleocr import doc2md_convert


@pytest.mark.parametrize(
    ("style_name", "expected"),
    [
        ("List Number", "1. First\n2. Second"),
        ("List Bullet", "- First\n- Second"),
    ],
    ids=("numbered", "bulleted"),
)
def test_doc2md_list_from_paragraph_style(tmp_path, style_name, expected):
    document = Document()
    document.add_paragraph("First", style=style_name)
    document.add_paragraph("Second", style=style_name)
    path = tmp_path / "styled-list.docx"
    document.save(str(path))

    assert doc2md_convert(str(path)).markdown == expected


def test_doc2md_list_from_base_style_and_direct_override(tmp_path):
    document = Document()
    style = document.styles.add_style("Review bullet", WD_STYLE_TYPE.PARAGRAPH)
    style.base_style = document.styles["List Bullet"]
    document.add_paragraph("Inherited bullet", style=style)

    # Direct numId=0 cancels numbering inherited from a paragraph style.
    suppressed = document.add_paragraph("Plain paragraph", style=style)
    suppressed._p.get_or_add_pPr().get_or_add_numPr().get_or_add_numId().val = 0

    # Direct numbering takes precedence over a paragraph's bullet style.
    ordered = document.add_paragraph("Direct number", style=style)
    ordered_id = document.styles["List Number"].element.pPr.numPr.numId.val
    ordered._p.get_or_add_pPr().get_or_add_numPr().get_or_add_numId().val = ordered_id
    path = tmp_path / "mixed-numbering.docx"
    document.save(str(path))

    assert doc2md_convert(str(path)).markdown == (
        "- Inherited bullet\n\nPlain paragraph\n\n1. Direct number"
    )


def test_doc2md_style_selects_linked_numbering_level(tmp_path):
    document = Document()
    style = document.styles["List Number"]
    numbering = document.part.numbering_part.element
    num_id = style.element.pPr.numPr.numId.val
    num = numbering.num_having_numId(num_id)
    abstract_id = num.abstractNumId.val
    abstract = numbering.find(
        f"{qn('w:abstractNum')}[@{qn('w:abstractNumId')}='{abstract_id}']"
    )
    abstract.find(qn("w:multiLevelType")).set(qn("w:val"), "multilevel")
    parent_level = abstract.find(qn("w:lvl"))
    child_level = deepcopy(parent_level)
    parent_level.remove(parent_level.find(qn("w:pStyle")))
    child_level.set(qn("w:ilvl"), "1")
    child_level.find(qn("w:lvlText")).set(qn("w:val"), "%2.")
    abstract.append(child_level)

    parent = document.add_paragraph("Parent")
    parent._p.get_or_add_pPr().get_or_add_numPr().get_or_add_numId().val = num_id
    style.element.pPr.numPr.get_or_add_ilvl().val = 0
    document.add_paragraph("Nested number", style=style)
    path = tmp_path / "style-level.docx"
    document.save(str(path))

    assert doc2md_convert(str(path)).markdown == "1. Parent\n    1. Nested number"
