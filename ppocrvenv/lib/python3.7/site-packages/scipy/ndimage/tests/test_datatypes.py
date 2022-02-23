""" Testing data types for ndimage calls
"""
import sys

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
import pytest

from scipy import ndimage


def test_map_coordinates_dts():
    # check that ndimage accepts different data types for interpolation
    data = np.array([[4, 1, 3, 2],
                     [7, 6, 8, 5],
                     [3, 5, 3, 6]])
    shifted_data = np.array([[0, 0, 0, 0],
                             [0, 4, 1, 3],
                             [0, 7, 6, 8]])
    idx = np.indices(data.shape)
    dts = (np.uint8, np.uint16, np.uint32, np.uint64,
           np.int8, np.int16, np.int32, np.int64,
           np.intp, np.uintp, np.float32, np.float64)
    for order in range(0, 6):
        for data_dt in dts:
            these_data = data.astype(data_dt)
            for coord_dt in dts:
                # affine mapping
                mat = np.eye(2, dtype=coord_dt)
                off = np.zeros((2,), dtype=coord_dt)
                out = ndimage.affine_transform(these_data, mat, off)
                assert_array_almost_equal(these_data, out)
                # map coordinates
                coords_m1 = idx.astype(coord_dt) - 1
                coords_p10 = idx.astype(coord_dt) + 10
                out = ndimage.map_coordinates(these_data, coords_m1, order=order)
                assert_array_almost_equal(out, shifted_data)
                # check constant fill works
                out = ndimage.map_coordinates(these_data, coords_p10, order=order)
                assert_array_almost_equal(out, np.zeros((3,4)))
            # check shift and zoom
            out = ndimage.shift(these_data, 1)
            assert_array_almost_equal(out, shifted_data)
            out = ndimage.zoom(these_data, 1)
            assert_array_almost_equal(these_data, out)


@pytest.mark.xfail(not sys.platform == 'darwin', reason="runs only on darwin")
def test_uint64_max():
    # Test interpolation respects uint64 max.  Reported to fail at least on
    # win32 (due to the 32 bit visual C compiler using signed int64 when
    # converting between uint64 to double) and Debian on s390x.
    # Interpolation is always done in double precision floating point, so
    # we use the largest uint64 value for which int(float(big)) still fits
    # in a uint64.
    big = 2**64 - 1025
    arr = np.array([big, big, big], dtype=np.uint64)
    # Tests geometric transform (map_coordinates, affine_transform)
    inds = np.indices(arr.shape) - 0.1
    x = ndimage.map_coordinates(arr, inds)
    assert_(x[1] == int(float(big)))
    assert_(x[2] == int(float(big)))
    # Tests zoom / shift
    x = ndimage.shift(arr, 0.1)
    assert_(x[1] == int(float(big)))
    assert_(x[2] == int(float(big)))
