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
from lxml import html
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as WorksheetImage
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
from openpyxl.drawing.xdr import XDRPositiveSize2D
from openpyxl.utils.cell import column_index_from_string
from openpyxl.utils.units import pixels_to_EMU
from PIL import Image

from paddleocr import doc2md_convert


@pytest.mark.parametrize(
    "group_end", [None, "C", "E"], ids=["separate", "range", "clipped-range"]
)
def test_xlsx_image_width_uses_every_column_in_a_width_range(tmp_path, group_end):
    book = Workbook()
    sheet = book.active
    sheet.append(["First", "Second", "Third"])
    for column in ("A", "B", "C"):
        sheet.column_dimensions[column].width = 20
    if group_end is not None:
        sheet.column_dimensions.group("A", group_end)

    png = BytesIO()
    Image.new("RGB", (210, 20), "blue").save(png, format="PNG")
    image_data = png.getvalue()
    picture = WorksheetImage(BytesIO(image_data))
    picture.anchor = OneCellAnchor(
        _from=AnchorMarker(col=0, row=1),
        ext=XDRPositiveSize2D(cx=pixels_to_EMU(210), cy=pixels_to_EMU(20)),
    )
    sheet.add_image(picture)
    path = tmp_path / "widths.xlsx"
    book.save(path)

    reopened = load_workbook(path)
    try:
        saved_sheet = reopened.active
        assert saved_sheet.max_column == 3
        if group_end is not None:
            column = saved_sheet.column_dimensions["A"]
            assert (column.min, column.max) == (1, column_index_from_string(group_end))
            assert list(saved_sheet.column_dimensions) == ["A"]
    finally:
        reopened.close()

    result = doc2md_convert(str(path))
    images = html.fromstring(result.markdown).xpath(".//img")
    assert len(images) == 1
    image = images[0]
    assert result.images[image.get("src")] == image_data
    # With three width-20 columns at 7 px per character, 210 px is half the sheet.
    assert image.get("width") == "50%"
