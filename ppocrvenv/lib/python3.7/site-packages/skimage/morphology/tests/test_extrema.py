import math
import unittest

import numpy as np
import pytest
from numpy.testing import assert_equal
from pytest import raises, warns

from skimage._shared.testing import expected_warnings
from skimage.morphology import extrema


eps = 1e-12


def diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = ((a - b) ** 2).sum()
    return math.sqrt(t)


class TestExtrema():

    def test_saturated_arithmetic(self):
        """Adding/subtracting a constant and clipping"""
        # Test for unsigned integer
        data = np.array([[250, 251, 5, 5],
                         [100, 200, 253, 252],
                         [4, 10, 1, 3]],
                        dtype=np.uint8)
        # adding the constant
        img_constant_added = extrema._add_constant_clip(data, 4)
        expected = np.array([[254, 255, 9, 9],
                             [104, 204, 255, 255],
                             [8, 14, 5, 7]],
                            dtype=np.uint8)
        error = diff(img_constant_added, expected)
        assert error < eps
        img_constant_subtracted = extrema._subtract_constant_clip(data, 4)
        expected = np.array([[246, 247, 1, 1],
                             [96, 196, 249, 248],
                             [0, 6, 0, 0]],
                            dtype=np.uint8)
        error = diff(img_constant_subtracted, expected)
        assert error < eps

        # Test for signed integer
        data = np.array([[32767, 32766],
                         [-32768, -32767]],
                        dtype=np.int16)
        img_constant_added = extrema._add_constant_clip(data, 1)
        expected = np.array([[32767, 32767],
                             [-32767, -32766]],
                            dtype=np.int16)
        error = diff(img_constant_added, expected)
        assert error < eps
        img_constant_subtracted = extrema._subtract_constant_clip(data, 1)
        expected = np.array([[32766, 32765],
                             [-32768, -32768]],
                            dtype=np.int16)
        error = diff(img_constant_subtracted, expected)
        assert error < eps

    def test_h_maxima(self):
        """h-maxima for various data types"""

        data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11],
                         [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
                         [13, 15, 40, 40, 18, 18, 18, 60, 60, 15],
                         [14, 16, 40, 40, 19, 19, 19, 60, 60, 16],
                         [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                         [15, 16, 18, 19, 19, 20, 19, 19, 18, 16],
                         [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                         [14, 16, 80, 80, 19, 19, 19, 100, 100, 16],
                         [13, 15, 80, 80, 18, 18, 18, 100, 100, 15],
                         [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]],
                        dtype=np.uint8)

        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
            data = data.astype(dtype)
            out = extrema.h_maxima(data, 40)

            error = diff(expected_result, out)
            assert error < eps

    def test_h_minima(self):
        """h-minima for various data types"""

        data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11],
                         [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
                         [13, 15, 40, 40, 18, 18, 18, 60, 60, 15],
                         [14, 16, 40, 40, 19, 19, 19, 60, 60, 16],
                         [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                         [15, 16, 18, 19, 19, 20, 19, 19, 18, 16],
                         [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                         [14, 16, 80, 80, 19, 19, 19, 100, 100, 16],
                         [13, 15, 80, 80, 18, 18, 18, 100, 100, 15],
                         [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]],
                        dtype=np.uint8)
        data = 100 - data
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
            data = data.astype(dtype)
            out = extrema.h_minima(data, 40)

            error = diff(expected_result, out)
            assert error < eps
            assert out.dtype == expected_result.dtype

    def test_extrema_float(self):
        """specific tests for float type"""
        data = np.array([[0.10, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14,
                          0.14, 0.13, 0.11],
                         [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16,
                          0.16, 0.15, 0.13],
                         [0.13, 0.15, 0.40, 0.40, 0.18, 0.18, 0.18,
                          0.60, 0.60, 0.15],
                         [0.14, 0.16, 0.40, 0.40, 0.19, 0.19, 0.19,
                          0.60, 0.60, 0.16],
                         [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19,
                          0.19, 0.18, 0.16],
                         [0.15, 0.182, 0.18, 0.19, 0.204, 0.20, 0.19,
                          0.19, 0.18, 0.16],
                         [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19,
                          0.19, 0.18, 0.16],
                         [0.14, 0.16, 0.80, 0.80, 0.19, 0.19, 0.19,
                          1.0, 1.0, 0.16],
                         [0.13, 0.15, 0.80, 0.80, 0.18, 0.18, 0.18,
                          1.0, 1.0, 0.15],
                         [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16,
                          0.16, 0.15, 0.13]],
                        dtype=np.float32)
        inverted_data = 1.0 - data

        out = extrema.h_maxima(data, 0.003)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)

        error = diff(expected_result, out)
        assert error < eps

        out = extrema.h_minima(inverted_data, 0.003)
        error = diff(expected_result, out)
        assert error < eps

    def test_h_maxima_float_image(self):
        """specific tests for h-maxima float image type"""
        w = 10
        x, y = np.mgrid[0:w, 0:w]
        data = 20 - 0.2 * ((x - w / 2) ** 2 + (y - w / 2) ** 2)
        data[2:4, 2:4] = 40
        data[2:4, 7:9] = 60
        data[7:9, 2:4] = 80
        data[7:9, 7:9] = 100
        data = data.astype(np.float32)

        expected_result = np.zeros_like(data)
        expected_result[(data > 19.9)] = 1.0

        for h in [1.0e-12, 1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1, 0.1]:
            out = extrema.h_maxima(data, h)
            error = diff(expected_result, out)
            assert error < eps

    def test_h_maxima_float_h(self):
        """specific tests for h-maxima float h parameter"""
        data = np.array([[0, 0, 0, 0, 0],
                         [0, 3, 3, 3, 0],
                         [0, 3, 4, 3, 0],
                         [0, 3, 3, 3, 0],
                         [0, 0, 0, 0, 0]], dtype=np.uint8)

        h_vals = np.linspace(1.0, 2.0, 100)
        failures = 0

        for h in h_vals:
            if h % 1 != 0:
                msgs = ['possible precision loss converting image']
            else:
                msgs = []

            with expected_warnings(msgs):
                maxima = extrema.h_maxima(data, h)

            if (maxima[2, 2] == 0):
                failures += 1

        assert (failures == 0)

    def test_h_maxima_large_h(self):
        """test that h-maxima works correctly for large h"""
        data = np.array([[10, 10, 10, 10, 10],
                         [10, 13, 13, 13, 10],
                         [10, 13, 14, 13, 10],
                         [10, 13, 13, 13, 10],
                         [10, 10, 10, 10, 10]], dtype=np.uint8)

        maxima = extrema.h_maxima(data, 5)
        assert (np.sum(maxima) == 0)

        data = np.array([[10, 10, 10, 10, 10],
                         [10, 13, 13, 13, 10],
                         [10, 13, 14, 13, 10],
                         [10, 13, 13, 13, 10],
                         [10, 10, 10, 10, 10]], dtype=np.float32)

        maxima = extrema.h_maxima(data, 5.0)
        assert (np.sum(maxima) == 0)

    def test_h_minima_float_image(self):
        """specific tests for h-minima float image type"""
        w = 10
        x, y = np.mgrid[0:w, 0:w]
        data = 180 + 0.2 * ((x - w / 2) ** 2 + (y - w / 2) ** 2)
        data[2:4, 2:4] = 160
        data[2:4, 7:9] = 140
        data[7:9, 2:4] = 120
        data[7:9, 7:9] = 100
        data = data.astype(np.float32)

        expected_result = np.zeros_like(data)
        expected_result[(data < 180.1)] = 1.0

        for h in [1.0e-12, 1.0e-6, 1.0e-3, 1.0e-2, 1.0e-1, 0.1]:
            out = extrema.h_minima(data, h)
            error = diff(expected_result, out)
            assert error < eps

    def test_h_minima_float_h(self):
        """specific tests for h-minima float h parameter"""
        data = np.array([[4, 4, 4, 4, 4],
                         [4, 1, 1, 1, 4],
                         [4, 1, 0, 1, 4],
                         [4, 1, 1, 1, 4],
                         [4, 4, 4, 4, 4]], dtype=np.uint8)

        h_vals = np.linspace(1.0, 2.0, 100)
        failures = 0
        for h in h_vals:
            if h % 1 != 0:
                msgs = ['possible precision loss converting image']
            else:
                msgs = []

            with expected_warnings(msgs):
                minima = extrema.h_minima(data, h)

            if (minima[2, 2] == 0):
                failures += 1

        assert (failures == 0)

    def test_h_minima_large_h(self):
        """test that h-minima works correctly for large h"""
        data = np.array([[14, 14, 14, 14, 14],
                         [14, 11, 11, 11, 14],
                         [14, 11, 10, 11, 14],
                         [14, 11, 11, 11, 14],
                         [14, 14, 14, 14, 14]], dtype=np.uint8)

        maxima = extrema.h_minima(data, 5)
        assert (np.sum(maxima) == 0)

        data = np.array([[14, 14, 14, 14, 14],
                         [14, 11, 11, 11, 14],
                         [14, 11, 10, 11, 14],
                         [14, 11, 11, 11, 14],
                         [14, 14, 14, 14, 14]], dtype=np.float32)

        maxima = extrema.h_minima(data, 5.0)
        assert (np.sum(maxima) == 0)


class TestLocalMaxima(unittest.TestCase):
    """Some tests for local_minima are included as well."""

    supported_dtypes = [
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.int8, np.int16, np.int32, np.int64,
        np.float32, np.float64
    ]
    image = np.array(
        [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 3, 3, 0, 0, 4, 0, 2, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0, 3, 0, 0, 0],
         [0, 2, 0, 1, 0, 2, 1, 0, 0, 0, 0, 3, 0, 0, 0],
         [0, 0, 2, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0]],
        dtype=np.uint8
    )
    # Connectivity 2, maxima can touch border, returned with default values
    expected_default = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
        dtype=bool
    )
    # Connectivity 1 (cross), maxima can touch border
    expected_cross = np.array(
        [[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
        dtype=bool
    )

    def test_empty(self):
        """Test result with empty image."""
        result = extrema.local_maxima(np.array([[]]), indices=False)
        assert result.size == 0
        assert result.dtype == bool
        assert result.shape == (1, 0)

        result = extrema.local_maxima(np.array([]), indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].size == 0
        assert result[0].dtype == np.intp

        result = extrema.local_maxima(np.array([[]]), indices=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].size == 0
        assert result[0].dtype == np.intp
        assert result[1].size == 0
        assert result[1].dtype == np.intp

    def test_dtypes(self):
        """Test results with default configuration for all supported dtypes."""
        for dtype in self.supported_dtypes:
            result = extrema.local_maxima(self.image.astype(dtype))
            assert result.dtype == bool
            assert_equal(result, self.expected_default)

    def test_dtypes_old(self):
        """
        Test results with default configuration and data copied from old unit
        tests for all supported dtypes.
        """
        data = np.array(
            [[10, 11, 13, 14, 14, 15, 14, 14, 13, 11],
             [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
             [13, 15, 40, 40, 18, 18, 18, 60, 60, 15],
             [14, 16, 40, 40, 19, 19, 19, 60, 60, 16],
             [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
             [15, 16, 18, 19, 19, 20, 19, 19, 18, 16],
             [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
             [14, 16, 80, 80, 19, 19, 19, 100, 100, 16],
             [13, 15, 80, 80, 18, 18, 18, 100, 100, 15],
             [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]],
            dtype=np.uint8
        )
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )
        for dtype in self.supported_dtypes:
            image = data.astype(dtype)
            result = extrema.local_maxima(image)
            assert result.dtype == bool
            assert_equal(result, expected)

    def test_connectivity(self):
        """Test results if footprint is a scalar."""
        # Connectivity 1: generates cross shaped footprint
        result_conn1 = extrema.local_maxima(self.image, connectivity=1)
        assert result_conn1.dtype == bool
        assert_equal(result_conn1, self.expected_cross)

        # Connectivity 2: generates square shaped footprint
        result_conn2 = extrema.local_maxima(self.image, connectivity=2)
        assert result_conn2.dtype == bool
        assert_equal(result_conn2, self.expected_default)

        # Connectivity 3: generates square shaped footprint
        result_conn3 = extrema.local_maxima(self.image, connectivity=3)
        assert result_conn3.dtype == bool
        assert_equal(result_conn3, self.expected_default)

    def test_footprint(self):
        """Test results if footprint is given."""
        footprint_cross = np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        result_footprint_cross = extrema.local_maxima(
            self.image, footprint=footprint_cross)
        assert result_footprint_cross.dtype == bool
        assert_equal(result_footprint_cross, self.expected_cross)

        for footprint in [
            ((True,) * 3,) * 3,
            np.ones((3, 3), dtype=np.float64),
            np.ones((3, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=bool),
        ]:
            # Test different dtypes for footprint which expects a boolean array
            # but will accept and convert other types if possible
            result_footprint_square = extrema.local_maxima(
                self.image, footprint=footprint
            )
            assert result_footprint_square.dtype == bool
            assert_equal(result_footprint_square, self.expected_default)

        footprint_x = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
        expected_footprint_x = np.array(
            [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]],
            dtype=bool
        )
        result_footprint_x = extrema.local_maxima(self.image,
                                                  footprint=footprint_x)
        assert result_footprint_x.dtype == bool
        assert_equal(result_footprint_x, expected_footprint_x)

    def test_indices(self):
        """Test output if indices of peaks are desired."""
        # Connectivity 1
        expected_conn1 = np.nonzero(self.expected_cross)
        result_conn1 = extrema.local_maxima(self.image, connectivity=1,
                                            indices=True)
        assert_equal(result_conn1, expected_conn1)

        # Connectivity 2
        expected_conn2 = np.nonzero(self.expected_default)
        result_conn2 = extrema.local_maxima(self.image, connectivity=2,
                                            indices=True)
        assert_equal(result_conn2, expected_conn2)

    def test_allow_borders(self):
        """Test maxima detection at the image border."""
        # Use connectivity 1 to allow many maxima, only filtering at border is
        # of interest
        result_with_boder = extrema.local_maxima(
            self.image, connectivity=1, allow_borders=True)
        assert result_with_boder.dtype == bool
        assert_equal(result_with_boder, self.expected_cross)

        expected_without_border = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
             [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )
        result_without_border = extrema.local_maxima(
            self.image, connectivity=1, allow_borders=False)
        assert result_with_boder.dtype == bool
        assert_equal(result_without_border, expected_without_border)

    def test_nd(self):
        """Test one- and three-dimensional case."""
        # One-dimension
        x_1d = np.array([1, 1, 0, 1, 2, 3, 0, 2, 1, 2, 0])
        expected_1d = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                               dtype=bool)
        result_1d = extrema.local_maxima(x_1d)
        assert result_1d.dtype == bool
        assert_equal(result_1d, expected_1d)

        # 3-dimensions (adapted from old unit test)
        x_3d = np.zeros((8, 8, 8), dtype=np.uint8)
        expected_3d = np.zeros((8, 8, 8), dtype=bool)
        # first maximum: only one pixel
        x_3d[1, 1:3, 1:3] = 100
        x_3d[2, 2, 2] = 200
        x_3d[3, 1:3, 1:3] = 100
        expected_3d[2, 2, 2] = 1
        # second maximum: three pixels in z-direction
        x_3d[5:8, 1, 1] = 200
        expected_3d[5:8, 1, 1] = 1
        # third: two maxima in 0 and 3.
        x_3d[0, 5:8, 5:8] = 200
        x_3d[1, 6, 6] = 100
        x_3d[2, 5:7, 5:7] = 200
        x_3d[0:3, 5:8, 5:8] += 50
        expected_3d[0, 5:8, 5:8] = 1
        expected_3d[2, 5:7, 5:7] = 1
        # four : one maximum in the corner of the square
        x_3d[6:8, 6:8, 6:8] = 200
        x_3d[7, 7, 7] = 255
        expected_3d[7, 7, 7] = 1
        result_3d = extrema.local_maxima(x_3d)
        assert result_3d.dtype == bool
        assert_equal(result_3d, expected_3d)

    def test_constant(self):
        """Test behaviour for 'flat' images."""
        const_image = np.full((7, 6), 42, dtype=np.uint8)
        expected = np.zeros((7, 6), dtype=np.uint8)
        for dtype in self.supported_dtypes:
            const_image = const_image.astype(dtype)
            # test for local maxima
            result = extrema.local_maxima(const_image)
            assert result.dtype == bool
            assert_equal(result, expected)
            # test for local minima
            result = extrema.local_minima(const_image)
            assert result.dtype == bool
            assert_equal(result, expected)

    def test_extrema_float(self):
        """Specific tests for float type."""
        # Copied from old unit test for local_maxma
        image = np.array(
            [[0.10, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.11],
             [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13],
             [0.13, 0.15, 0.40, 0.40, 0.18, 0.18, 0.18, 0.60, 0.60, 0.15],
             [0.14, 0.16, 0.40, 0.40, 0.19, 0.19, 0.19, 0.60, 0.60, 0.16],
             [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16],
             [0.15, 0.182, 0.18, 0.19, 0.204, 0.20, 0.19, 0.19, 0.18, 0.16],
             [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16],
             [0.14, 0.16, 0.80, 0.80, 0.19, 0.19, 0.19, 1.0, 1.0, 0.16],
             [0.13, 0.15, 0.80, 0.80, 0.18, 0.18, 0.18, 1.0, 1.0, 0.15],
             [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13]],
            dtype=np.float32
        )
        inverted_image = 1.0 - image
        expected_result = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=bool
        )

        # Test for local maxima with automatic step calculation
        result = extrema.local_maxima(image)
        assert result.dtype == bool
        assert_equal(result, expected_result)

        # Test for local minima with automatic step calculation
        result = extrema.local_minima(inverted_image)
        assert result.dtype == bool
        assert_equal(result, expected_result)

    def test_exceptions(self):
        """Test if input validation triggers correct exceptions."""
        # Mismatching number of dimensions
        with raises(ValueError, match="number of dimensions"):
            extrema.local_maxima(
                self.image, footprint=np.ones((3, 3, 3), dtype=bool))
        with raises(ValueError, match="number of dimensions"):
            extrema.local_maxima(
                self.image, footprint=np.ones((3,), dtype=bool))

        # All dimensions in footprint must be of size 3
        with raises(ValueError, match="dimension size"):
            extrema.local_maxima(
                self.image, footprint=np.ones((2, 3), dtype=bool))
        with raises(ValueError, match="dimension size"):
            extrema.local_maxima(
                self.image, footprint=np.ones((5, 5), dtype=bool))

        with raises(TypeError, match="float16 which is not supported"):
            extrema.local_maxima(np.empty(1, dtype=np.float16))

    def test_small_array(self):
        """Test output for arrays with dimension smaller 3.

        If any dimension of an array is smaller than 3 and `allow_borders` is
        false a footprint, which has at least 3 elements in each
        dimension, can't be applied. This is an implementation detail so
        `local_maxima` should still return valid output (see gh-3261).

        If `allow_borders` is true the array is padded internally and there is
        no problem.
        """
        warning_msg = "maxima can't exist .* any dimension smaller 3 .*"
        x = np.array([0, 1])
        extrema.local_maxima(x, allow_borders=True)  # no warning
        with warns(UserWarning, match=warning_msg):
            result = extrema.local_maxima(x, allow_borders=False)
        assert_equal(result, [0, 0])
        assert result.dtype == bool

        x = np.array([[1, 2], [2, 2]])
        extrema.local_maxima(x, allow_borders=True, indices=True)  # no warning
        with warns(UserWarning, match=warning_msg):
            result = extrema.local_maxima(x, allow_borders=False, indices=True)
        assert_equal(result, np.zeros((2, 0), dtype=np.intp))
        assert result[0].dtype == np.intp
        assert result[1].dtype == np.intp


@pytest.mark.parametrize(
    'function',
    ['local_maxima', 'local_minima', 'h_minima', 'h_maxima']
)
def test_selem_kwarg_deprecation(function):
    img = np.zeros((16, 16))
    args = (20,) if function.startswith('h_') else ()
    with expected_warnings(["`selem` is a deprecated argument name"]):
        getattr(extrema, function)(img, *args, selem=np.ones((3, 3)))
