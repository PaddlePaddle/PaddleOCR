import numpy as np
from skimage.measure import shannon_entropy

from skimage._shared.testing import assert_almost_equal


def test_shannon_ones():
    img = np.ones((10, 10))
    res = shannon_entropy(img, base=np.e)
    assert_almost_equal(res, 0.0)


def test_shannon_all_unique():
    img = np.arange(64)
    res = shannon_entropy(img, base=2)
    assert_almost_equal(res, np.log(64) / np.log(2))
