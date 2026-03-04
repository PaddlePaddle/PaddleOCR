"""Tests for the layout parsing patches (issue #17503).

These tests verify the fixed functions directly without importing the
full paddleocr package (which requires paddlex).
"""

import importlib
import sys
import numpy as np
import pytest


# Import the patch module directly to avoid triggering the full paddleocr
# import chain which requires paddlex
def _import_patch_module():
    """Import _patch_layout_parsing without triggering paddleocr.__init__."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "paddleocr._pipelines._patch_layout_parsing",
        "paddleocr/_pipelines/_patch_layout_parsing.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_patch_mod = _import_patch_module()
_fixed_calculate_overlap_ratio = _patch_mod._fixed_calculate_overlap_ratio
_fixed_calculate_minimum_enclosing_bbox = (
    _patch_mod._fixed_calculate_minimum_enclosing_bbox
)


class TestFixedCalculateOverlapRatio:
    """Tests for the overflow-safe calculate_overlap_ratio."""

    def test_normal_overlap(self):
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="union")
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500.0 / 17500.0
        assert abs(ratio - expected) < 1e-6

    def test_no_overlap(self):
        bbox1 = [0, 0, 50, 50]
        bbox2 = [100, 100, 200, 200]
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="union")
        assert ratio == 0.0

    def test_complete_overlap(self):
        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 100, 100]
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="union")
        assert abs(ratio - 1.0) < 1e-6

    def test_large_coordinates_no_overflow(self):
        """Verify no integer overflow with large coordinate values.

        This is the primary bug from issue #17503. With int32 arithmetic,
        inter_width * inter_height would overflow for large images
        (e.g. after document unwarping).
        """
        # Coordinates large enough to cause int32 overflow when multiplied
        bbox1 = np.array([0, 0, 50000, 50000], dtype=np.int32)
        bbox2 = np.array([0, 0, 50000, 50000], dtype=np.int32)
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="union")
        # Should be 1.0 (identical boxes), not corrupted by overflow
        assert abs(ratio - 1.0) < 1e-6

    def test_large_coordinates_partial_overlap(self):
        """Verify correct overlap ratio with large coordinates."""
        bbox1 = np.array([0, 0, 60000, 60000], dtype=np.int32)
        bbox2 = np.array([30000, 30000, 90000, 90000], dtype=np.int32)
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="small")
        # Intersection: 30000x30000 = 900_000_000
        # Small box area: 60000*60000 = 3_600_000_000
        expected = 900_000_000.0 / 3_600_000_000.0
        assert abs(ratio - expected) < 1e-6

    def test_mode_small(self):
        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 200, 200]
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="small")
        # Intersection = 100*100 = 10000, small area = 10000
        assert abs(ratio - 1.0) < 1e-6

    def test_mode_large(self):
        bbox1 = [0, 0, 100, 100]
        bbox2 = [0, 0, 200, 200]
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="large")
        # Intersection = 100*100 = 10000, large area = 40000
        expected = 10000.0 / 40000.0
        assert abs(ratio - expected) < 1e-6

    def test_zero_area_bbox(self):
        bbox1 = [0, 0, 0, 0]
        bbox2 = [0, 0, 100, 100]
        ratio = _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="union")
        assert ratio == 0.0

    def test_invalid_mode_raises(self):
        bbox1 = [0, 0, 100, 100]
        bbox2 = [50, 50, 150, 150]
        with pytest.raises(ValueError, match="Invalid mode"):
            _fixed_calculate_overlap_ratio(bbox1, bbox2, mode="invalid")


class TestFixedCalculateMinimumEnclosingBbox:
    """Tests for the empty-safe calculate_minimum_enclosing_bbox."""

    def test_single_bbox(self):
        result = _fixed_calculate_minimum_enclosing_bbox([[10, 20, 30, 40]])
        np.testing.assert_array_equal(result, [10, 20, 30, 40])

    def test_multiple_bboxes(self):
        bboxes = [[10, 20, 30, 40], [5, 15, 35, 45]]
        result = _fixed_calculate_minimum_enclosing_bbox(bboxes)
        np.testing.assert_array_equal(result, [5, 15, 35, 45])

    def test_empty_list_returns_degenerate_bbox(self):
        """Verify empty list returns a degenerate bbox instead of raising.

        This is the secondary fix from issue #17503. The original code
        raises ValueError("The list of bounding boxes is empty.").
        """
        result = _fixed_calculate_minimum_enclosing_bbox([])
        assert result is not None
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_none_input_returns_degenerate_bbox(self):
        result = _fixed_calculate_minimum_enclosing_bbox(None)
        assert result is not None
        np.testing.assert_array_equal(result, [0, 0, 0, 0])
