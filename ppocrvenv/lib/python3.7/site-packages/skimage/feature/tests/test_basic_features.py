import pytest
import numpy as np

from skimage._shared.testing import expected_warnings
from skimage.feature import multiscale_basic_features


@pytest.mark.parametrize('edges', (False, True))
@pytest.mark.parametrize('texture', (False, True))
def test_multiscale_basic_features_gray(edges, texture):
    img = np.zeros((20, 20))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    features = multiscale_basic_features(img, edges=edges, texture=texture)

    n_sigmas = 6
    intensity = True
    assert features.shape[-1] == (
        n_sigmas * (int(intensity) + int(edges) + 2 * int(texture))
    )
    assert features.shape[:-1] == img.shape[:]


@pytest.mark.parametrize('edges', (False, True))
@pytest.mark.parametrize('texture', (False, True))
def test_multiscale_basic_features_rgb(edges, texture):
    img = np.zeros((20, 20, 3))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        features = multiscale_basic_features(img, edges=edges, texture=texture,
                                             multichannel=True)

    n_sigmas = 6
    intensity = True
    assert features.shape[-1] == (
        3 * n_sigmas * (int(intensity) + int(edges) + 2 * int(texture))
    )
    assert features.shape[:-1] == img.shape[:-1]


def test_multiscale_basic_features_deprecated_multichannel():
    img = np.zeros((10, 10, 5))
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    n_sigmas = 2
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        features = multiscale_basic_features(img, sigma_min=1, sigma_max=2,
                                             multichannel=True)
    assert features.shape[-1] == 5 * n_sigmas * 4
    assert features.shape[:-1] == img.shape[:-1]

    # repeat prior test, but check for positional multichannel warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        multiscale_basic_features(img, True, sigma_min=1, sigma_max=2)
    assert features.shape[-1] == 5 * n_sigmas * 4
    assert features.shape[:-1] == img.shape[:-1]

    # Consider last axis as spatial dimension
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2)
    assert features.shape[-1] == n_sigmas * 5
    assert features.shape[:-1] == img.shape


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1, -2])
def test_multiscale_basic_features_channel_axis(channel_axis):
    num_channels = 5
    shape_spatial = (10, 10)
    ndim = len(shape_spatial)
    shape = tuple(
        np.insert(shape_spatial, channel_axis % (ndim + 1), num_channels)
    )
    img = np.zeros(shape)
    img[:10] = 1
    img += 0.05 * np.random.randn(*img.shape)
    n_sigmas = 2

    # features for all channels are concatenated along the last axis
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2,
                                         channel_axis=channel_axis)
    assert features.shape[-1] == 5 * n_sigmas * 4
    assert features.shape[:-1] == np.moveaxis(img, channel_axis, -1).shape[:-1]

    # Consider channel_axis as spatial dimension
    features = multiscale_basic_features(img, sigma_min=1, sigma_max=2)
    assert features.shape[-1] == n_sigmas * 5
    assert features.shape[:-1] == img.shape
