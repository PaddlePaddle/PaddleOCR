import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage

from skimage._shared.testing import expected_warnings
from skimage.filters import median, rank


@pytest.fixture
def image():
    return np.array([[1, 2, 3, 2, 1],
                     [1, 1, 2, 2, 3],
                     [3, 2, 1, 2, 1],
                     [3, 2, 1, 1, 1],
                     [1, 2, 1, 2, 3]],
                    dtype=np.uint8)


@pytest.mark.parametrize(
    "mode, cval, behavior, warning_type",
    [('nearest', 0.0, 'ndimage', None),
     ('constant', 0.0, 'rank', UserWarning),
     ('nearest', 0.0, 'rank', None),
     ('nearest', 0.0, 'ndimage', None)]
)
def test_median_warning(image, mode, cval, behavior, warning_type):

    if warning_type:
        with pytest.warns(warning_type):
            median(image, mode=mode, behavior=behavior)
    else:
        median(image, mode=mode, behavior=behavior)


def test_selem_kwarg_deprecation(image):
    with expected_warnings(["`selem` is a deprecated argument name"]):
        median(image, selem=None)


@pytest.mark.parametrize(
    "behavior, func, params",
    [('ndimage', ndimage.median_filter, {'size': (3, 3)}),
     ('rank', rank.median, {'footprint': np.ones((3, 3), dtype=np.uint8)})]
)
def test_median_behavior(image, behavior, func, params):
    assert_allclose(median(image, behavior=behavior), func(image, **params))


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.float32, np.float64]
)
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior='ndimage')
    assert median_image.dtype == dtype


def test_median_error_ndim():
    img = np.random.randint(0, 10, size=(5, 5, 5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        median(img, behavior='rank')


@pytest.mark.parametrize(
    "img, behavior",
    [(np.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'rank'),
     (np.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'ndimage'),
     (np.random.randint(0, 10, size=(3, 3, 3), dtype=np.uint8), 'ndimage')]
)
def test_median(img, behavior):
    median(img, behavior=behavior)
