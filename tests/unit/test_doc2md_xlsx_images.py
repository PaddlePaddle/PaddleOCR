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

from io import BytesIO

import pytest

openpyxl = pytest.importorskip("openpyxl")
PILImage = pytest.importorskip("PIL.Image")
from lxml import html
from openpyxl.drawing.image import Image
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, TwoCellAnchor

from paddleocr import doc2md_convert


@pytest.mark.parametrize("max_rows", [None, 1])
def test_xlsx_keeps_images_anchored_inside_merged_cells(tmp_path, max_rows):
    """Keep images from visible anchor rows in the merged HTML cell.

    Args:
        tmp_path: Temporary directory for the workbook and conversion output.
        max_rows: Optional limit that excludes the second anchor row.
    """
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Quarterly report"
    sheet.merge_cells("A1:B2")
    payloads = []
    for color in ("red", "green", "blue", "yellow"):
        buffer = BytesIO()
        PILImage.new("RGB", (8, 8), color).save(buffer, format="PNG")
        payloads.append(buffer.getvalue())

    sheet.add_image(Image(BytesIO(payloads[0])), "A1")
    sheet.add_image(Image(BytesIO(payloads[1])), "B1")
    lower_image = Image(BytesIO(payloads[2]))
    lower_image.anchor = TwoCellAnchor(
        _from=AnchorMarker(col=1, row=1), to=AnchorMarker(col=2, row=2)
    )
    sheet.add_image(lower_image)
    sheet.add_image(Image(BytesIO(payloads[3])), "C1")
    source = tmp_path / "report.xlsx"
    workbook.save(source)
    output = tmp_path / "converted" / "report.md"

    result = doc2md_convert(source, output=output, max_rows=max_rows)
    table = html.fromstring(result.markdown)
    merged, neighbor = table.xpath("//table/tr[1]/th")
    assert merged.get("colspan") == "2"
    assert merged.get("rowspan") == ("2" if max_rows is None else None)
    assert merged.text_content() == "Quarterly report"
    merged_refs = merged.xpath("./img/@src")
    other_refs = neighbor.xpath("./img/@src")
    assert [result.images[ref] for ref in merged_refs] == (
        payloads[:3] if max_rows is None else payloads[:2]
    )
    assert [result.images[ref] for ref in other_refs] == payloads[3:]
    assert len(result.images) == len(merged_refs) + len(other_refs)
    for ref, data in result.images.items():
        assert (output.parent / ref).read_bytes() == data
