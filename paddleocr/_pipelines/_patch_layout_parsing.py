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

"""
Patches for paddlex layout parsing utilities to fix:
- Integer overflow in calculate_overlap_ratio when bounding box coordinates
  are large (e.g. after doc_unwarping), causing RuntimeWarning and incorrect
  overlap calculations.
- ValueError in calculate_minimum_enclosing_bbox when the bounding box list
  is empty, which can happen when overflow causes all overlap matches to fail.

See: https://github.com/PaddlePaddle/PaddleOCR/issues/17503
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_patched = False


def _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="union"):
    """
    Calculate the overlap ratio between two bounding boxes.

    This version casts coordinates to float64 before computing areas to
    prevent integer overflow when bounding box coordinates are large
    (e.g. after document unwarping).
    """
    bbox1 = np.array(bbox1, dtype=np.float64)
    bbox2 = np.array(bbox2, dtype=np.float64)

    x_min_inter = np.maximum(bbox1[0], bbox2[0])
    y_min_inter = np.maximum(bbox1[1], bbox2[1])
    x_max_inter = np.minimum(bbox1[2], bbox2[2])
    y_max_inter = np.minimum(bbox1[3], bbox2[3])

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    inter_area = inter_width * inter_height

    bbox1_area = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    bbox2_area = abs((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = np.minimum(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = np.maximum(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def _fixed_calculate_minimum_enclosing_bbox(bboxes):
    """
    Calculate the minimum enclosing bounding box for a list of bounding boxes.

    This version returns a zero-area bounding box at the origin instead of
    raising ValueError when the list is empty, allowing the caller to
    continue without crashing.
    """
    if not bboxes:
        logger.debug(
            "calculate_minimum_enclosing_bbox received an empty list; "
            "returning a degenerate bounding box"
        )
        return np.array([0, 0, 0, 0])

    bboxes_array = np.array(bboxes)

    min_x = np.min(bboxes_array[:, 0])
    min_y = np.min(bboxes_array[:, 1])
    max_x = np.max(bboxes_array[:, 2])
    max_y = np.max(bboxes_array[:, 3])

    return np.array([min_x, min_y, max_x, max_y])


def apply_patches():
    """
    Apply patches to paddlex layout parsing utilities to fix integer overflow
    and empty bounding box errors.

    This function is idempotent and safe to call multiple times.
    """
    global _patched
    if _patched:
        return

    try:
        import paddlex.inference.pipelines.layout_parsing.utils as lp_utils
        import paddlex.inference.pipelines.layout_parsing.pipeline_v2 as lp_pipeline
    except ImportError:
        logger.debug("paddlex layout parsing modules not available; skipping patches")
        return

    # Patch the utils module
    lp_utils.calculate_overlap_ratio = _fixed_calculate_overlap_ratio
    lp_utils.calculate_minimum_enclosing_bbox = _fixed_calculate_minimum_enclosing_bbox

    # Also patch the references imported directly into pipeline_v2, since
    # Python binds names at import time
    if hasattr(lp_pipeline, "calculate_overlap_ratio"):
        lp_pipeline.calculate_overlap_ratio = _fixed_calculate_overlap_ratio
    if hasattr(lp_pipeline, "calculate_minimum_enclosing_bbox"):
        lp_pipeline.calculate_minimum_enclosing_bbox = (
            _fixed_calculate_minimum_enclosing_bbox
        )

    _patched = True
    logger.debug("Applied layout parsing patches for issue #17503")
