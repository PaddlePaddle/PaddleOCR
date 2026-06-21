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
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sorted_boxes(monkeypatch):
    """Load ``tools/infer/predict_system.py`` in isolation.

    ``predict_system`` imports the full inference stack (paddle, cv2, ...) at
    module scope, but ``sorted_boxes`` itself only needs numpy. Stub the heavy
    modules so the pure ordering function can be exercised without those deps.
    """

    def _stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    _stub("cv2")
    pil = _stub("PIL")
    pil.Image = _stub("PIL.Image")
    _stub("tools")
    _stub("tools.infer")
    _stub(
        "tools.infer.utility",
        draw_ocr_box_txt=None,
        get_rotate_crop_image=None,
        get_minarea_rect_crop=None,
        slice_generator=None,
        merge_fragmented=None,
    )
    _stub("tools.infer.predict_rec")
    _stub("tools.infer.predict_det")
    _stub("tools.infer.predict_cls")
    _stub("ppocr")
    _stub("ppocr.utils")
    _stub("ppocr.utils.utility", get_image_file_list=None, check_and_read=None)
    _stub(
        "ppocr.utils.logging",
        get_logger=lambda *a, **k: logging.getLogger("predict_system_test"),
    )

    spec = importlib.util.spec_from_file_location(
        "predict_system_under_test",
        REPO_ROOT / "tools" / "infer" / "predict_system.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.sorted_boxes


def _line_boxes(rows, h=26, w=80, hgap=40, vgap=18):
    """Axis-aligned word boxes on ``rows`` text lines.

    Lines are document-width (many words per line) with normal line spacing,
    matching real forms where a slightly tilted line spans many times the old
    10 px row threshold. Each box carries an id ``row * 100 + col`` so the
    correct reading order is exactly ``sorted(ids)``.
    """
    boxes, ids = [], []
    for r, ncol in enumerate(rows):
        y = r * (h + vgap)
        for c in range(ncol):
            x = c * (w + hgap)
            boxes.append(
                np.array(
                    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    dtype=np.float32,
                )
            )
            ids.append(r * 100 + c)
    return np.array(boxes, dtype=np.float32), ids


def _rotate(boxes, deg, center=None):
    if center is None:
        center = boxes.reshape(-1, 2).mean(axis=0)
    t = np.deg2rad(deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    out = boxes.copy().astype(np.float32)
    out[..., 0] -= center[0]
    out[..., 1] -= center[1]
    out = out @ rot.T
    out[..., 0] += center[0]
    out[..., 1] += center[1]
    return out


def _recovered_order(sorted_boxes, boxes, ids):
    """Run sorted_boxes and map each returned quad back to its id."""
    order = []
    for quad in sorted_boxes(boxes):
        dist = np.linalg.norm(boxes[:, 0, :] - quad[0], axis=1)
        order.append(ids[int(dist.argmin())])
    return order


def test_sorted_boxes_upright_reading_order(sorted_boxes):
    # 3 document-width lines, upright -> strict top-to-bottom, left-to-right.
    boxes, ids = _line_boxes([10, 10, 10])
    assert _recovered_order(sorted_boxes, boxes, ids) == sorted(ids)


def test_sorted_boxes_robust_to_mild_skew(sorted_boxes):
    # The same page rotated by a few degrees must keep the same reading order.
    # On document-width lines the previous fixed-10px implementation
    # interleaves the lines once the per-line vertical span exceeds 10 px.
    boxes, ids = _line_boxes([10, 10, 10])
    expected = sorted(ids)
    for deg in (3, 6, 10, -5):
        skewed = _rotate(boxes, deg)
        assert (
            _recovered_order(sorted_boxes, skewed, ids) == expected
        ), f"reading order broke at {deg} degrees skew"
