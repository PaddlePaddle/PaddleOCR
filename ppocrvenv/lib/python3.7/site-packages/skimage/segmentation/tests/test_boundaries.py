import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from skimage._shared.utils import _supported_float_type
from skimage.segmentation import find_boundaries, mark_boundaries


white = (1, 1, 1)


def test_find_boundaries():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_find_boundaries_bool():
    image = np.zeros((5, 5), dtype=bool)
    image[2:5, 2:5] = True

    ref = np.array([[False, False, False, False, False],
                    [False, False,  True,  True,  True],
                    [False,  True,  True,  True,  True],
                    [False,  True,  True, False, False],
                    [False,  True,  True, False, False]], dtype=bool)
    result = find_boundaries(image)
    assert_array_equal(result, ref)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.float16, np.float32, np.float64]
)
def test_mark_boundaries(dtype):
    image = np.zeros((10, 10), dtype=dtype)
    label_image = np.zeros((10, 10), dtype=np.uint8)
    label_image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    assert marked.dtype == _supported_float_type(dtype)
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)

    ref = np.array([[0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
                    [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                    [2, 1, 1, 2, 0, 2, 1, 1, 2, 0],
                    [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
                    [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
                    [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    marked = mark_boundaries(image, label_image, color=white,
                             outline_color=(2, 2, 2), mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)


def test_mark_boundaries_bool():
    image = np.zeros((10, 10), dtype=bool)
    label_image = np.zeros((10, 10), dtype=np.uint8)
    label_image[2:7, 2:7] = 1

    ref = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    marked = mark_boundaries(image, label_image, color=white, mode='thick')
    result = np.mean(marked, axis=-1)
    assert_array_equal(result, ref)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_mark_boundaries_subpixel(dtype):
    labels = np.array([[0, 0, 0, 0],
                       [0, 0, 5, 0],
                       [0, 1, 5, 0],
                       [0, 0, 5, 0],
                       [0, 0, 0, 0]], dtype=np.uint8)
    np.random.seed(0)
    image = np.round(np.random.rand(*labels.shape), 2)
    image = image.astype(dtype, copy=False)
    marked = mark_boundaries(image, labels, color=white, mode='subpixel')
    assert marked.dtype == _supported_float_type(dtype)
    marked_proj = np.round(np.mean(marked, axis=-1), 2)

    ref_result = np.array(
        [[ 0.55,  0.63,  0.72,  0.69,  0.6 ,  0.55,  0.54],
         [ 0.45,  0.58,  0.72,  1.  ,  1.  ,  1.  ,  0.69],
         [ 0.42,  0.54,  0.65,  1.  ,  0.44,  1.  ,  0.89],
         [ 0.69,  1.  ,  1.  ,  1.  ,  0.69,  1.  ,  0.83],
         [ 0.96,  1.  ,  0.38,  1.  ,  0.79,  1.  ,  0.53],
         [ 0.89,  1.  ,  1.  ,  1.  ,  0.38,  1.  ,  0.16],
         [ 0.57,  0.78,  0.93,  1.  ,  0.07,  1.  ,  0.09],
         [ 0.2 ,  0.52,  0.92,  1.  ,  1.  ,  1.  ,  0.54],
         [ 0.02,  0.35,  0.83,  0.9 ,  0.78,  0.81,  0.87]])
    assert_allclose(marked_proj, ref_result, atol=0.01)


@pytest.mark.parametrize('mode', ['thick', 'inner', 'outer', 'subpixel'])
def test_boundaries_constant_image(mode):
    """A constant-valued image has not boundaries."""
    ones = np.ones((8, 8), dtype=int)
    b = find_boundaries(ones, mode=mode)
    assert np.all(b == 0)
