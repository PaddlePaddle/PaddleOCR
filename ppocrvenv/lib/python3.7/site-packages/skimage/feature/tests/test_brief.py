import pytest
import numpy as np

from skimage._shared.testing import assert_array_equal
from skimage import data
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing


def test_color_image_unsupported_error():
    """Brief descriptors can be evaluated on gray-scale images only."""
    img = np.zeros((20, 20, 3))
    keypoints = np.asarray([[7, 5], [11, 13]])
    with testing.raises(ValueError):
        BRIEF().extract(img, keypoints)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'int'])
def test_normal_mode(dtype):
    """Verify the computed BRIEF descriptors with expected for normal mode."""
    img = data.coins().astype(dtype)

    keypoints = corner_peaks(corner_harris(img), min_distance=5,
                             threshold_abs=0, threshold_rel=0.1)

    extractor = BRIEF(descriptor_size=8, sigma=2)

    extractor.extract(img, keypoints[:8])

    expected = np.array([[1, 1, 1, 0, 1, 1, 0, 1],
                         [0, 1, 1, 0, 1, 1, 0, 0],
                         [1, 1, 1, 0, 1, 1, 0, 1],
                         [0, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 1, 0, 1, 1, 0, 0],
                         [0, 1, 1, 0, 1, 1, 1, 0],
                         [1, 1, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 0, 1, 1, 0]], dtype=bool)

    assert_array_equal(extractor.descriptors, expected)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'int'])
def test_uniform_mode(dtype):
    """Verify the computed BRIEF descriptors with expected for uniform mode."""
    img = data.coins().astype(dtype)

    keypoints = corner_peaks(corner_harris(img), min_distance=5,
                             threshold_abs=0, threshold_rel=0.1)

    extractor = BRIEF(descriptor_size=8, sigma=2, mode='uniform')

    extractor.extract(img, keypoints[:8])

    expected = np.array([[0, 1, 0, 1, 0, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 0, 0, 1, 1, 1],
                         [1, 0, 1, 0, 1, 0, 1, 1],
                         [0, 0, 1, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 1, 0, 1],
                         [0, 1, 0, 0, 0, 1, 1, 1],
                         [1, 0, 1, 1, 1, 0, 0, 1]], dtype=bool)

    assert_array_equal(extractor.descriptors, expected)


def test_unsupported_mode():
    with testing.raises(ValueError):
        BRIEF(mode='foobar')


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'int'])
def test_border(dtype):
    img = np.zeros((100, 100), dtype=dtype)
    keypoints = np.array([[1, 1], [20, 20], [50, 50], [80, 80]])

    extractor = BRIEF(patch_size=41)
    extractor.extract(img, keypoints)

    assert extractor.descriptors.shape[0] == 3
    assert_array_equal(extractor.mask, (False, True, True, True))
