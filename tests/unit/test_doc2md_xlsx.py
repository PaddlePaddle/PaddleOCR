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

import importlib.util
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.py38_incompatible
openpyxl = pytest.importorskip("openpyxl")
html = pytest.importorskip("lxml.html")


@pytest.fixture(scope="module")
def doc2md():
    # Load the real office converter without importing PaddleX inference modules.
    package = Path(__file__).resolve().parents[2] / "paddleocr" / "_doc2md"
    spec = importlib.util.spec_from_file_location(
        "_paddleocr_doc2md_xlsx_test",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("style", ["plain", "bold-italic", "hyperlink"])
@pytest.mark.parametrize(
    "value",
    [
        "Normal finding",
        "Region <ROI> & adjacent tooth",
        "<b>literal annotation</b>",
        "Literal &copy; marker",
    ],
)
def test_xlsx_preserves_literal_cell_text(doc2md, tmp_path, style, value):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Finding", "Note"])
    sheet.append([value, "unchanged"])
    cell = sheet["A2"]
    if style == "bold-italic":
        cell.font = openpyxl.styles.Font(bold=True, italic=True)
    elif style == "hyperlink":
        cell.hyperlink = "https://example.org/find?q=ROI&view=full"
    source = tmp_path / "findings.xlsx"
    workbook.save(source)
    workbook.close()

    result = doc2md.convert(source)
    document = html.fromstring(result.markdown)
    rows = document.xpath("//table/tr")
    assert len(rows) == 2
    assert all(len(row) == 2 for row in rows)
    converted_cell = rows[1][0]
    assert converted_cell.text_content() == value
    assert rows[1][1].text_content() == "unchanged"
    if style == "bold-italic":
        assert len(converted_cell.xpath("./i/b")) == 1
        assert len(converted_cell.xpath(".//*")) == 2
    elif style == "hyperlink":
        assert converted_cell.xpath("./a/@href") == [cell.hyperlink.target]
        assert len(converted_cell.xpath(".//*")) == 1
    else:
        assert len(converted_cell) == 0
