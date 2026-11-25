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
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from paddleocr import doc2md_convert


@pytest.mark.parametrize(
    ("group_widths", "expected"),
    [((), "20%"), ((4,), "40%"), ((3, 6), "60%")],
    ids=["ungrouped", "resized-group", "resized-nested-groups"],
)
def test_pptx_picture_width_in_resized_groups(tmp_path, group_widths, expected):
    image = BytesIO()
    Image.new("RGB", (16, 16), "blue").save(image, format="PNG")
    image.seek(0)
    presentation = Presentation()
    presentation.slide_width = Inches(10)
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])

    groups = []
    shapes = slide.shapes
    for _ in group_widths:
        group = shapes.add_group_shape()
        groups.append(group)
        shapes = group.shapes
    shapes.add_picture(image, 0, 0, width=Inches(2), height=Inches(2))

    # Resize inside-out. Reset the parent coordinate box to its current children
    # before resizing the parent itself, as when editing a group before scaling it.
    for group, width in zip(reversed(groups), group_widths):
        group._element.recalculate_extents()
        group.width = Inches(width)

    deck = tmp_path / "width.pptx"
    presentation.save(str(deck))
    result = doc2md_convert(str(deck))
    image_tag = html.fragment_fromstring(result.markdown)
    assert image_tag.tag == "img"
    assert result.images[image_tag.get("src")] == image.getvalue()
    assert image_tag.get("width") == expected
