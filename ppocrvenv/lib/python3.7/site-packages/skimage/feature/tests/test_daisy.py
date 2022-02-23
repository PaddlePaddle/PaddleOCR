import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal

from skimage import data
from skimage import img_as_float
from skimage.feature import daisy


def test_daisy_color_image_unsupported_error():
    img = np.zeros((20, 20, 3))
    with pytest.raises(ValueError):
        daisy(img)


def test_daisy_desc_dims():
    img = img_as_float(data.astronaut()[:128, :128].mean(axis=2))
    rings = 2
    histograms = 4
    orientations = 3
    descs = daisy(img, rings=rings, histograms=histograms,
                  orientations=orientations)
    assert(descs.shape[2] == (rings * histograms + 1) * orientations)

    rings = 4
    histograms = 5
    orientations = 13
    descs = daisy(img, rings=rings, histograms=histograms,
                  orientations=orientations)
    assert(descs.shape[2] == (rings * histograms + 1) * orientations)


def test_descs_shape():
    img = img_as_float(data.astronaut()[:256, :256].mean(axis=2))
    radius = 20
    step = 8
    descs = daisy(img, radius=radius, step=step)
    assert(descs.shape[0] == ceil((img.shape[0] - radius * 2) / float(step)))
    assert(descs.shape[1] == ceil((img.shape[1] - radius * 2) / float(step)))

    img = img[:-1, :-2]
    radius = 5
    step = 3
    descs = daisy(img, radius=radius, step=step)
    assert(descs.shape[0] == ceil((img.shape[0] - radius * 2) / float(step)))
    assert(descs.shape[1] == ceil((img.shape[1] - radius * 2) / float(step)))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_daisy_sigmas_and_radii(dtype):
    img = data.astronaut()[:64, :64].mean(axis=2).astype(dtype, copy=False)
    sigmas = [1, 2, 3]
    radii = [1, 2]
    descs = daisy(img, sigmas=sigmas, ring_radii=radii)
    assert descs.dtype == img.dtype


def test_daisy_incompatible_sigmas_and_radii():
    img = img_as_float(data.astronaut()[:64, :64].mean(axis=2))
    sigmas = [1, 2]
    radii = [1, 2]
    with pytest.raises(ValueError):
        daisy(img, sigmas=sigmas, ring_radii=radii)


def test_daisy_normalization():
    img = img_as_float(data.astronaut()[:64, :64].mean(axis=2))

    descs = daisy(img, normalization='l1')
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            assert_almost_equal(np.sum(descs[i, j, :]), 1)
    descs_ = daisy(img)
    assert_almost_equal(descs, descs_)

    descs = daisy(img, normalization='l2')
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            assert_almost_equal(sqrt(np.sum(descs[i, j, :] ** 2)), 1)

    orientations = 8
    descs = daisy(img, orientations=orientations, normalization='daisy')
    desc_dims = descs.shape[2]
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            for k in range(0, desc_dims, orientations):
                assert_almost_equal(sqrt(np.sum(
                    descs[i, j, k:k + orientations] ** 2)), 1)

    img = np.zeros((50, 50))
    descs = daisy(img, normalization='off')
    for i in range(descs.shape[0]):
        for j in range(descs.shape[1]):
            assert_almost_equal(np.sum(descs[i, j, :]), 0)

    with pytest.raises(ValueError):
        daisy(img, normalization='does_not_exist')


def test_daisy_visualization():
    img = img_as_float(data.astronaut()[:32, :32].mean(axis=2))
    descs, descs_img = daisy(img, visualize=True)
    assert(descs_img.shape == (32, 32, 3))
