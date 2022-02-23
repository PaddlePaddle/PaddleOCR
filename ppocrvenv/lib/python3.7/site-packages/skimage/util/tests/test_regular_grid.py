import numpy as np
from skimage.util import regular_grid
from skimage._shared.testing import assert_equal


def test_regular_grid_full():
    ar = np.zeros((2, 2))
    g = regular_grid(ar, 25)
    assert_equal(g, [slice(None, None, None), slice(None, None, None)])
    ar[g] = 1
    assert_equal(ar.size, ar.sum())


def test_regular_grid_2d_8():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 8)
    assert_equal(g, [slice(5.0, None, 10.0), slice(5.0, None, 10.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 8)


def test_regular_grid_2d_32():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 32)
    assert_equal(g, [slice(2.0, None, 5.0), slice(2.0, None, 5.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 32)


def test_regular_grid_3d_8():
    ar = np.zeros((3, 20, 40))
    g = regular_grid(ar.shape, 8)
    assert_equal(g, [slice(1.0, None, 3.0), slice(5.0, None, 10.0),
                     slice(5.0, None, 10.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 8)
