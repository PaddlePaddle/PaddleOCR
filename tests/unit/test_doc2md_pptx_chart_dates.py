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


from datetime import date
from io import BytesIO

import pytest
from lxml import html
from openpyxl import load_workbook
from openpyxl.utils.datetime import CALENDAR_MAC_1904
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from paddleocr import doc2md_convert


@pytest.mark.parametrize("date_system", ["1900", "1904", "1904-implicit"])
def test_pptx_chart_date_axis(tmp_path, date_system):
    dates = [date(2024, 1, 1), date(2024, 6, 1)]
    data = CategoryChartData()
    data.categories = dates
    data.add_series("Measurements", (12, 34))
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(6), Inches(3), data
    ).chart
    if date_system != "1900":
        date1904 = chart._element.get_or_add_date1904()
        if date_system == "1904":
            date1904.val = True
        else:
            date1904.attrib.pop("val", None)
        points = chart._element.xpath(".//c:ser/c:cat/c:numRef/c:numCache/c:pt/c:v")
        assert len(points) == len(dates)
        for point, day in zip(points, dates):
            point.text = str((day - date(1904, 1, 1)).days)

        # Match the chart cache by storing the source dates with the same epoch.
        chart_workbook = chart.part.chart_workbook
        workbook = load_workbook(BytesIO(chart_workbook.xlsx_part.blob))
        workbook.epoch = CALENDAR_MAC_1904
        with BytesIO() as buffer:
            workbook.save(buffer)
            chart_workbook.update_from_xlsx_blob(buffer.getvalue())

    path = tmp_path / "chart.pptx"
    presentation.save(str(path))

    markdown = doc2md_convert(str(path)).markdown
    rows = html.fromstring(markdown).xpath(".//tbody/tr")
    assert [row.xpath("./td/text()") for row in rows] == [
        ["2024-01-01", "12.0"],
        ["2024-06-01", "34.0"],
    ]
