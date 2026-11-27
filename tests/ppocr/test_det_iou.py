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
from shapely.geometry import Polygon

from ppocr.metrics.eval_det_iou import DetectionIoUEvaluator


def box(left, right, ignore=False):
    return {
        "points": [(left, 0), (right, 0), (right, 10), (left, 10)],
        "ignore": ignore,
    }


def test_first_eligible_detection_wins():
    gt = [box(0, 10), box(3, 13)]
    left, shared = box(0, 9), box(1, 11)
    evaluator = DetectionIoUEvaluator(iou_constraint=0.65)
    assert evaluator.evaluate_image(gt, [shared, left])["detMatched"] == 1
    assert evaluator.evaluate_image(gt, [left, shared])["detMatched"] == 2


def test_ignore_coverage_precedes_matching_and_uses_strict_threshold():
    gt = [box(0, 10), box(0, 5, ignore=True)]
    result = DetectionIoUEvaluator().evaluate_image(gt, [box(0, 10), box(0, 4)])
    assert result == {"gtCare": 1, "detCare": 1, "detMatched": 1}


def test_overlapping_bounds_require_exact_polygon_iou():
    gt = [{"points": [(0, 0), (10, 0), (0, 10)], "ignore": False}]
    det = [{"points": [(10, 10), (10, 1), (1, 10)]}]
    assert DetectionIoUEvaluator(iou_constraint=0).evaluate_image(gt, det) == {
        "gtCare": 1,
        "detCare": 1,
        "detMatched": 0,
    }


@pytest.mark.parametrize("threshold, expected", [(0, 0), (-0.1, 1)])
def test_disjoint_detections(threshold, expected):
    evaluator = DetectionIoUEvaluator(iou_constraint=threshold)
    assert evaluator.evaluate_image([box(0, 2)], [box(4, 6)])["detMatched"] == expected


def test_invalid_polygons_and_empty_inputs():
    invalid = {"points": [(0, 0), (2, 2), (0, 2), (2, 0)], "ignore": False}
    evaluator = DetectionIoUEvaluator()
    assert evaluator.evaluate_image([invalid, box(0, 1)], [invalid]) == {
        "gtCare": 1,
        "detCare": 0,
        "detMatched": 0,
    }
    assert evaluator.evaluate_image([], [box(0, 1)]) == {
        "gtCare": 0,
        "detCare": 1,
        "detMatched": 0,
    }


def test_disjoint_regions_skip_polygon_intersections(monkeypatch):
    def unexpected_intersection(*args, **kwargs):
        pytest.fail("disjoint bounds should skip exact intersection")

    monkeypatch.setattr(Polygon, "intersection", unexpected_intersection)
    gt = [box(0, 2, ignore=True), box(3, 5)]
    assert DetectionIoUEvaluator().evaluate_image(gt, [box(7, 10)]) == {
        "gtCare": 1,
        "detCare": 1,
        "detMatched": 0,
    }
