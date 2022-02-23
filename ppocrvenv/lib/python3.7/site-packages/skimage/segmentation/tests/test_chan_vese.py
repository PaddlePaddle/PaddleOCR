import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.segmentation import chan_vese


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_chan_vese_flat_level_set(dtype):
    # because the algorithm evolves the level set around the
    # zero-level, it the level-set has no zero level, the algorithm
    # will not produce results in theory. However, since a continuous
    # approximation of the delta function is used, the algorithm
    # still affects the entirety of the level-set. Therefore with
    # infinite time, the segmentation will still converge.
    img = np.zeros((10, 10), dtype=dtype)
    img[3:6, 3:6] = 1
    ls = np.full((10, 10), 1000, dtype=dtype)
    result = chan_vese(img, mu=0.0, tol=1e-3, init_level_set=ls)
    assert_array_equal(result.astype(float), np.ones((10, 10)))
    result = chan_vese(img, mu=0.0, tol=1e-3, init_level_set=-ls)
    assert_array_equal(result.astype(float), np.zeros((10, 10)))


def test_chan_vese_small_disk_level_set():
    img = np.zeros((10, 10))
    img[3:6, 3:6] = 1
    result = chan_vese(img, mu=0.0, tol=1e-3, init_level_set="small disk")
    assert_array_equal(result.astype(float), img)


def test_chan_vese_simple_shape():
    img = np.zeros((10, 10))
    img[3:6, 3:6] = 1
    result = chan_vese(img, mu=0.0, tol=1e-8).astype(float)
    assert_array_equal(result, img)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.float16, np.float32, np.float64]
)
def test_chan_vese_extended_output(dtype):
    img = np.zeros((10, 10), dtype=dtype)
    img[3:6, 3:6] = 1
    result = chan_vese(img, mu=0.0, tol=1e-8, extended_output=True)
    float_dtype = _supported_float_type(dtype)
    assert result[1].dtype == float_dtype
    assert all(arr.dtype == float_dtype for arr in result[2])
    assert_array_equal(len(result), 3)


def test_chan_vese_remove_noise():
    ref = np.zeros((10, 10))
    ref[1:6, 1:6] = np.array([[0, 1, 1, 1, 0],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [0, 1, 1, 1, 0]])
    img = ref.copy()
    img[8, 3] = 1
    result = chan_vese(img, mu=0.3, tol=1e-3, max_num_iter=100, dt=10,
                       init_level_set="disk").astype(float)
    assert_array_equal(result, ref)


def test_chan_vese_incorrect_image_type():
    img = np.zeros((10, 10, 3))
    ls = np.zeros((10, 9))
    with pytest.raises(ValueError):
        chan_vese(img, mu=0.0, init_level_set=ls)


def test_chan_vese_gap_closing():
    ref = np.zeros((20, 20))
    ref[8:15, :] = np.ones((7, 20))
    img = ref.copy()
    img[:, 6] = np.zeros((20))
    result = chan_vese(img, mu=0.7, tol=1e-3, max_num_iter=1000, dt=1000,
                       init_level_set="disk").astype(float)
    assert_array_equal(result, ref)


def test_chan_vese_max_iter_deprecation():
    img = np.zeros((20, 20))
    with expected_warnings(["`max_iter` is a deprecated argument"]):
        chan_vese(img, max_iter=10)


def test_chan_vese_incorrect_level_set():
    img = np.zeros((10, 10))
    ls = np.zeros((10, 9))
    with pytest.raises(ValueError):
        chan_vese(img, mu=0.0, init_level_set=ls)
    with pytest.raises(ValueError):
        chan_vese(img, mu=0.0, init_level_set="a")


def test_chan_vese_blank_image():
    img = np.zeros((10, 10))
    level_set = np.random.rand(10, 10)
    ref = level_set > 0
    result = chan_vese(img, mu=0.0, tol=0.0, init_level_set=level_set)
    assert_array_equal(result, ref)
