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

pptx = pytest.importorskip("pptx")
from lxml import html
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from paddleocr import doc2md_convert


@pytest.mark.parametrize(
    ("series_names", "axis_title", "expected_names"),
    [
        (("Revenue", "Operating cost"), None, ("Revenue", "Operating cost")),
        (("Revenue", "Operating cost"), "USD", ("Revenue", "Operating cost")),
        (("", ""), None, ("Series1", "Series2")),
        (("", ""), "USD", ("USD", "USD")),
    ],
)
def test_pptx_chart_series_names(tmp_path, series_names, axis_title, expected_names):
    """Preserve chart series labels and the unnamed-series fallbacks.

    Args:
        tmp_path: Temporary directory for the generated PowerPoint file.
        series_names: The two series names to store in the chart.
        axis_title: Optional value-axis title used for unnamed series.
        expected_names: The expected table column names.

    Returns:
        None.
    """
    document = pptx.Presentation()
    slide = document.slides.add_slide(document.slide_layouts[6])
    data = CategoryChartData()
    data.categories = ["Q1", "Q2"]
    data.add_series(series_names[0], [100, 120])
    data.add_series(series_names[1], [60, 70])
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        Inches(1),
        Inches(1),
        Inches(6),
        Inches(4),
        data,
    ).chart
    if axis_title is not None:
        chart.value_axis.has_title = True
        chart.value_axis.axis_title.text_frame.text = axis_title
    source = tmp_path / "quarterly.pptx"
    document.save(source)

    table = html.fromstring(doc2md_convert(source).markdown)
    assert [cell.text_content() for cell in table.xpath(".//thead/tr/th")] == [
        "",
        *expected_names,
    ]
    assert [
        [cell.text_content() for cell in row.xpath("./td")]
        for row in table.xpath(".//tbody/tr")
    ] == [["Q1", "100.0", "60.0"], ["Q2", "120.0", "70.0"]]
