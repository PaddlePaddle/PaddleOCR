import numpy as np
from skimage.segmentation import quickshift

from skimage._shared import testing
from skimage._shared.testing import (assert_greater, test_parallel,
                                     assert_equal, assert_array_equal)

@test_parallel()
@testing.parametrize('dtype', [np.float32, np.float64])
def test_grey(dtype):
    rnd = np.random.default_rng(0)
    img = np.zeros((20, 21))
    img[:10, 10:] = 0.2
    img[10:, :10] = 0.4
    img[10:, 10:] = 0.6
    img += 0.05 * rnd.normal(size=img.shape)
    img = img.astype(dtype, copy=False)
    seg = quickshift(img, kernel_size=2, max_dist=3, random_seed=0,
                     convert2lab=False, sigma=0)
    # we expect 4 segments:
    assert_equal(len(np.unique(seg)), 4)
    # that mostly respect the 4 regions:
    for i in range(4):
        hist = np.histogram(img[seg == i], bins=[0, 0.1, 0.3, 0.5, 1])[0]
        assert_greater(hist[i], 20)


@testing.parametrize('dtype', [np.float32, np.float64])
@testing.parametrize('channel_axis', [-3, -2, -1, 0, 1, 2])
def test_color(dtype, channel_axis):
    rnd = np.random.default_rng(583428449)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    img = img.astype(dtype, copy=False)

    img = np.moveaxis(img, source=-1, destination=channel_axis)
    seg = quickshift(img, random_seed=0, max_dist=30, kernel_size=10, sigma=0,
                     channel_axis=channel_axis)
    # we expect 4 segments:
    assert_equal(len(np.unique(seg)), 4)
    assert_array_equal(seg[:10, :10], 1)
    assert_array_equal(seg[10:, :10], 3)
    assert_array_equal(seg[:10, 10:], 0)
    assert_array_equal(seg[10:, 10:], 2)

    seg2 = quickshift(img, kernel_size=1, max_dist=2, random_seed=0,
                      convert2lab=False, sigma=0,
                      channel_axis=channel_axis)
    # very oversegmented:
    assert len(np.unique(seg2)) > 10
    # still don't cross lines
    assert (seg2[9, :] != seg2[10, :]).all()
    assert (seg2[:, 9] != seg2[:, 10]).all()
