import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.filters import difference_of_gaussians, gaussian


def test_negative_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1
    with pytest.raises(ValueError):
        gaussian(a, sigma=-1.0)
    with pytest.raises(ValueError):
        gaussian(a, sigma=[-1.0, 1.0])
    with pytest.raises(ValueError):
        gaussian(a, sigma=np.asarray([-1.0, 1.0]))


def test_null_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert np.all(gaussian(a, 0, preserve_range=True) == a)


def test_default_sigma():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert_array_equal(
        gaussian(a, preserve_range=True),
        gaussian(a, preserve_range=True, sigma=1)
    )


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64]
)
def test_image_dtype(dtype):
    a = np.zeros((3, 3), dtype=dtype)
    assert gaussian(a).dtype == _supported_float_type(a.dtype)


def test_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    gaussian_a = gaussian(a, preserve_range=True, sigma=1, mode='reflect')
    assert gaussian_a.std() < a.std()


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_multichannel(channel_axis):
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    a = np.moveaxis(a, -1, channel_axis)
    gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect', preserve_range=True,
                              channel_axis=channel_axis)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    spatial_axes = tuple(
        [ax for ax in range(a.ndim) if ax != channel_axis % a.ndim]
    )
    assert np.allclose(a.mean(axis=spatial_axes),
                       gaussian_rgb_a.mean(axis=spatial_axes))

    if channel_axis % a.ndim == 2:
        # Test legacy behavior equivalent to old (multichannel = None)
        with expected_warnings(['multichannel']):
            gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect',
                                      preserve_range=True)

        # Check that the mean value is conserved in each channel
        # (color channels are not mixed together)
        assert np.allclose(a.mean(axis=spatial_axes),
                           gaussian_rgb_a.mean(axis=spatial_axes))
    # Iterable sigma
    gaussian_rgb_a = gaussian(a, sigma=[1, 2], mode='reflect',
                              channel_axis=channel_axis,
                              preserve_range=True)
    assert np.allclose(a.mean(axis=spatial_axes),
                       gaussian_rgb_a.mean(axis=spatial_axes))


def test_deprecated_multichannel():
    a = np.zeros((5, 5, 3))
    a[1, 1] = np.arange(1, 4)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        gaussian_rgb_a = gaussian(a, sigma=1, mode='reflect',
                                  multichannel=True)
    # Check that the mean value is conserved in each channel
    # (color channels are not mixed together)
    assert np.allclose(a.mean(axis=(0, 1)), gaussian_rgb_a.mean(axis=(0, 1)))

    # check positional multichannel argument warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        gaussian_rgb_a = gaussian(a, 1, None, 'reflect', 0, True)


def test_preserve_range():
    """Test preserve_range parameter."""
    ones = np.ones((2, 2), dtype=np.int64)
    filtered_ones = gaussian(ones, preserve_range=False)
    assert np.all(filtered_ones == filtered_ones[0, 0])
    assert filtered_ones[0, 0] < 1e-10

    filtered_preserved = gaussian(ones, preserve_range=True)
    assert np.all(filtered_preserved == 1.)

    img = np.array([[10.0, -10.0], [-4, 3]], dtype=np.float32)
    gaussian(img, 1)


def test_1d_ok():
    """Testing Gaussian Filter for 1D array.
    With any array consisting of positive integers and only one zero - it
    should filter all values to be greater than 0.1
    """
    nums = np.arange(7)
    filtered = gaussian(nums, preserve_range=True)
    assert np.all(filtered > 0.1)


def test_4d_ok():
    img = np.zeros((5,) * 4)
    img[2, 2, 2, 2] = 1
    res = gaussian(img, 1, mode='reflect', preserve_range=True)
    assert np.allclose(res.sum(), 1)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64]
)
def test_preserve_output(dtype):
    image = np.arange(9, dtype=dtype).reshape((3, 3))
    output = np.zeros_like(image, dtype=dtype)
    gaussian_image = gaussian(image, sigma=1, output=output,
                              preserve_range=True)
    assert gaussian_image is output


def test_output_error():
    image = np.arange(9, dtype=np.float32).reshape((3, 3))
    output = np.zeros_like(image, dtype=np.uint8)
    with pytest.raises(ValueError):
        gaussian(image, sigma=1, output=output,
                 preserve_range=True)


@pytest.mark.parametrize("s", [1, (2, 3)])
@pytest.mark.parametrize("s2", [4, (5, 6)])
@pytest.mark.parametrize("channel_axis", [None, 0, 1, -1])
def test_difference_of_gaussians(s, s2, channel_axis):
    image = np.random.rand(10, 10)
    if channel_axis is not None:
        n_channels = 5
        image = np.stack((image,) * n_channels, channel_axis)
    im1 = gaussian(image, s, preserve_range=True, channel_axis=channel_axis)
    im2 = gaussian(image, s2, preserve_range=True, channel_axis=channel_axis)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2, channel_axis=channel_axis)
    assert np.allclose(dog, dog2)


@pytest.mark.parametrize("s", [1, (1, 2)])
def test_auto_sigma2(s):
    image = np.random.rand(10, 10)
    im1 = gaussian(image, s, preserve_range=True)
    s2 = 1.6 * np.array(s)
    im2 = gaussian(image, s2, preserve_range=True)
    dog = im1 - im2
    dog2 = difference_of_gaussians(image, s, s2)
    assert np.allclose(dog, dog2)


def test_dog_invalid_sigma_dims():
    image = np.ones((5, 5, 3))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 2))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, 1, (3, 4))
    with pytest.raises(ValueError):
        with expected_warnings(["`multichannel` is a deprecated argument"]):
            difference_of_gaussians(image, (1, 2, 3), multichannel=True)
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 2, 3), channel_axis=-1)


def test_dog_invalid_sigma2():
    image = np.ones((3, 3))
    with pytest.raises(ValueError):
        difference_of_gaussians(image, 3, 2)
    with pytest.raises(ValueError):
        difference_of_gaussians(image, (1, 5), (2, 4))
