
import numpy as np
from skimage.util import crop
from skimage._shared.testing import (assert_array_equal, assert_equal)


def test_multi_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2), (2, 1)))
    assert_array_equal(out[0], [7, 8])
    assert_array_equal(out[-1], [32, 33])
    assert_equal(out.shape, (6, 2))


def test_pair_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, (1, 2))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert_equal(out.shape, (6, 2))


def test_pair_tuple_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2),))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert_equal(out.shape, (6, 2))


def test_int_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, 1)
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert_equal(out.shape, (7, 3))


def test_int_tuple_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, (1,))
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert_equal(out.shape, (7, 3))


def test_copy_crop():
    arr = np.arange(45).reshape(9, 5)
    out0 = crop(arr, 1, copy=True)
    assert out0.flags.c_contiguous
    out0[0, 0] = 100
    assert not np.any(arr == 100)
    assert not np.may_share_memory(arr, out0)

    out1 = crop(arr, 1)
    out1[0, 0] = 100
    assert arr[1, 1] == 100
    assert np.may_share_memory(arr, out1)


def test_zero_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, 0)
    assert out.shape == (9, 5)


def test_np_int_crop():
    arr = np.arange(45).reshape(9, 5)
    out1 = crop(arr, np.int64(1))
    out2 = crop(arr, np.int32(1))
    assert_array_equal(out1, out2)
    assert out1.shape == (7, 3)
