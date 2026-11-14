# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from io import BytesIO
from zipfile import ZipFile

import pytest

openpyxl = pytest.importorskip("openpyxl")
PILImage = pytest.importorskip("PIL.Image")
from lxml import etree
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image

from paddleocr import doc2md_convert


@pytest.fixture
def workbook_with_chartsheet(tmp_path):
    workbook = openpyxl.Workbook()
    sales = workbook.active
    sales.title = "Sales"
    sales.append(["Quarter", "Revenue"])
    sales.append(["Q1", 27])
    chart = BarChart()
    chart.add_data(Reference(sales, min_col=2, min_row=1, max_row=2))
    workbook.create_chartsheet("Revenue chart", 0).add_chart(chart)
    image = BytesIO()
    PILImage.new("RGB", (8, 8), "blue").save(image, format="PNG")
    sales.add_image(Image(image), "C1")
    notes = workbook.create_sheet("Notes")
    notes.append(["Comment"])
    notes.append(["Quarterly review"])
    path = tmp_path / "report.xlsx"
    workbook.save(path)
    workbook.close()
    return path


@pytest.mark.parametrize(
    ("options", "expected_headings"),
    [
        ({}, ["## Sales", "## Notes"]),
        ({"sheet_name": "Sales"}, ["## Sales"]),
        ({"sheet_name": "Revenue chart"}, []),
        ({"sheet_name": "Missing"}, []),
    ],
)
def test_xlsx_chartsheet_is_skipped(
    workbook_with_chartsheet, options, expected_headings
):
    result = doc2md_convert(workbook_with_chartsheet, **options)

    assert [
        line for line in result.markdown.splitlines() if line.startswith("## ")
    ] == expected_headings
    assert ("<td>27</td>" in result.markdown) == ("## Sales" in expected_headings)
    assert ("<td>Quarterly review</td>" in result.markdown) == (
        "## Notes" in expected_headings
    )
    assert result.metadata["sheet_names"] == ["Revenue chart", "Sales", "Notes"]
    assert result.metadata["sheet_count"] == 3


def test_xlsx_chartsheet_does_not_shift_worksheet_formula(workbook_with_chartsheet):
    with ZipFile(workbook_with_chartsheet) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    sales_rels = etree.fromstring(members["xl/worksheets/_rels/sheet1.xml.rels"])
    (drawing_rel,) = sales_rels
    drawing_path = drawing_rel.attrib["Target"].lstrip("/")
    drawing = etree.fromstring(members[drawing_path])
    # openpyxl writes drawing anchors but cannot author OMML text boxes.
    # Add the OMML payload to the drawing linked from the Sales worksheet.
    drawing.append(
        etree.fromstring(
            b"<mc:AlternateContent "
            b'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            b'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            b'<mc:Choice Requires="m"><a:p><m:oMath><m:r><m:t>42</m:t></m:r>'
            b"</m:oMath></a:p></mc:Choice><mc:Fallback/></mc:AlternateContent>"
        )
    )
    members[drawing_path] = etree.tostring(drawing)

    with ZipFile(workbook_with_chartsheet, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    sales = doc2md_convert(workbook_with_chartsheet, sheet_name="Sales")
    notes = doc2md_convert(workbook_with_chartsheet, sheet_name="Notes")

    assert "\n$$\n42\n$$" in sales.markdown
    assert "\n$$\n42\n$$" not in notes.markdown
