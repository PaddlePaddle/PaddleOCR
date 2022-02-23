import functools
import itertools

import numpy as np
import pytest
import pywt
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_warns)

from skimage import color, data, img_as_float, restoration
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration._denoise import _wavelet_threshold

try:
    import dask  # noqa
except ImportError:
    DASK_NOT_INSTALLED_WARNING = 'The optional dask dependency is not installed'
else:
    DASK_NOT_INSTALLED_WARNING = None


np.random.seed(1234)


astro = img_as_float(data.astronaut()[:128, :128])
astro_gray = color.rgb2gray(astro)
checkerboard_gray = img_as_float(data.checkerboard())
checkerboard = color.gray2rgb(checkerboard_gray)
# versions with one odd-sized dimension
astro_gray_odd = astro_gray[:, :-1]
astro_odd = astro[:, :-1]


float_dtypes = [np.float16, np.float32, np.float64]
try:
    float_dtypes += [np.float128]
except AttributeError:
    pass


@pytest.mark.parametrize('dtype', float_dtypes)
def test_denoise_tv_chambolle_2d(dtype):
    # astronaut image
    img = astro_gray.astype(dtype, copy=True)
    # add noise to astronaut
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    # clip noise so that it does not exceed allowed range for float images.
    img = np.clip(img, 0, 1)
    # denoise
    denoised_astro = restoration.denoise_tv_chambolle(img, weight=0.1)
    assert denoised_astro.dtype == _supported_float_type(img)

    from scipy import ndimage as ndi

    # Convert to a floating point type supported by scipy.ndimage
    float_dtype = _supported_float_type(img)
    img = img.astype(float_dtype, copy=False)

    grad = ndi.morphological_gradient(img, size=((3, 3)))
    grad_denoised = ndi.morphological_gradient(denoised_astro, size=((3, 3)))
    # test if the total variation has decreased
    assert grad_denoised.dtype == float_dtype
    assert np.sqrt((grad_denoised**2).sum()) < np.sqrt((grad**2).sum())


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_denoise_tv_chambolle_multichannel(channel_axis):
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)

    img = np.moveaxis(astro, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(img, weight=0.1,
                                                channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis, axis=channel_axis % img.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)

    # tile astronaut subset to generate 3D+channels data
    astro3 = np.tile(astro[:64, :64, np.newaxis, :], [1, 1, 2, 1])
    # modify along tiled dimension to give non-zero gradient on 3rd axis
    astro3[:, :, 0, :] = 2*astro3[:, :, 0, :]
    denoised0 = restoration.denoise_tv_chambolle(astro3[..., 0], weight=0.1)

    astro3 = np.moveaxis(astro3, -1, channel_axis)
    denoised = restoration.denoise_tv_chambolle(astro3, weight=0.1,
                                                channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis,
                            axis=channel_axis % astro3.ndim)
    assert_array_equal(denoised[_at(0)], denoised0)


def test_denoise_tv_chambolle_multichannel_deprecation():
    denoised0 = restoration.denoise_tv_chambolle(astro[..., 0], weight=0.1)

    with expected_warnings(["`multichannel` is a deprecated argument"]):
        denoised = restoration.denoise_tv_chambolle(astro, weight=0.1,
                                                    multichannel=True)

    # providing multichannel argument positionally also warns
    with expected_warnings(["Providing the `multichannel` argument"]):
        denoised = restoration.denoise_tv_chambolle(astro, 0.1, 2e-4, 200,
                                                    True)

    assert_array_equal(denoised[..., 0], denoised0)


def test_denoise_tv_chambolle_float_result_range():
    # astronaut image
    img = astro_gray
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_chambolle(int_astro,
                                                          weight=0.1)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_astro.dtype == float
    assert np.max(denoised_int_astro) <= 1.0
    assert np.min(denoised_int_astro) >= 0.0


def test_denoise_tv_chambolle_3d():
    """Apply the TV denoising algorithm on a 3D image representing a sphere."""
    x, y, z = np.ogrid[0:40, 0:40, 0:40]
    mask = (x - 22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    mask = 100 * mask.astype(float)
    mask += 60
    mask += 20 * np.random.rand(*mask.shape)
    mask[mask < 0] = 0
    mask[mask > 255] = 255
    res = restoration.denoise_tv_chambolle(mask.astype(np.uint8), weight=0.1)
    assert res.dtype == float
    assert res.std() * 255 < mask.std()


def test_denoise_tv_chambolle_1d():
    """Apply the TV denoising algorithm on a 1D sinusoid."""
    x = 125 + 100*np.sin(np.linspace(0, 8*np.pi, 1000))
    x += 20 * np.random.rand(x.size)
    x = np.clip(x, 0, 255)
    res = restoration.denoise_tv_chambolle(x.astype(np.uint8), weight=0.1)
    assert res.dtype == float
    assert res.std() * 255 < x.std()


def test_denoise_tv_chambolle_4d():
    """ TV denoising for a 4D input."""
    im = 255 * np.random.rand(8, 8, 8, 8)
    res = restoration.denoise_tv_chambolle(im.astype(np.uint8), weight=0.1)
    assert res.dtype == float
    assert res.std() * 255 < im.std()


def test_denoise_tv_chambolle_weighting():
    # make sure a specified weight gives consistent results regardless of
    # the number of input image dimensions
    rstate = np.random.default_rng(1234)
    img2d = astro_gray.copy()
    img2d += 0.15 * rstate.standard_normal(img2d.shape)
    img2d = np.clip(img2d, 0, 1)

    # generate 4D image by tiling
    img4d = np.tile(img2d[..., None, None], (1, 1, 2, 2))

    w = 0.2
    denoised_2d = restoration.denoise_tv_chambolle(img2d, weight=w)
    denoised_4d = restoration.denoise_tv_chambolle(img4d, weight=w)
    assert structural_similarity(denoised_2d,
                                 denoised_4d[:, :, 0, 0]) > 0.99


def test_denoise_tv_bregman_2d():
    img = checkerboard_gray.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=10)
    out2 = restoration.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_tv_bregman_float_result_range():
    # astronaut image
    img = astro_gray.copy()
    int_astro = np.multiply(img, 255).astype(np.uint8)
    assert np.max(int_astro) > 1
    denoised_int_astro = restoration.denoise_tv_bregman(int_astro, weight=60.0)
    # test if the value range of output float data is within [0.0:1.0]
    assert denoised_int_astro.dtype == float
    assert np.max(denoised_int_astro) <= 1.0
    assert np.min(denoised_int_astro) >= 0.0


def test_denoise_tv_bregman_3d():
    img = checkerboard.copy()
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=10)
    out2 = restoration.denoise_tv_bregman(img, weight=5)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_denoise_tv_bregman_3d_multichannel(channel_axis):
    img_astro = astro.copy()
    denoised0 = restoration.denoise_tv_bregman(img_astro[..., 0], weight=60.0)
    img_astro = np.moveaxis(img_astro, -1, channel_axis)
    denoised = restoration.denoise_tv_bregman(img_astro, weight=60.0,
                                              channel_axis=channel_axis)
    _at = functools.partial(slice_at_axis,
                            axis=channel_axis % img_astro.ndim)
    assert_array_equal(denoised0, denoised[_at(0)])


def test_denoise_tv_bregman_3d_multichannel_deprecation():
    img_astro = astro.copy()
    denoised0 = restoration.denoise_tv_bregman(img_astro[..., 0], weight=60.0)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        denoised = restoration.denoise_tv_bregman(img_astro, weight=60.0,
                                                  multichannel=True)

    assert_array_equal(denoised0, denoised[..., 0])


def test_denoise_tv_bregman_max_iter_deprecation():
    with expected_warnings(["`max_iter` is a deprecated argument"]):
        restoration.denoise_tv_bregman(astro_gray, weight=60.0, max_iter=5)


def test_denoise_tv_bregman_multichannel():
    img = checkerboard_gray.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_tv_bregman(img, weight=60.0)
    out2 = restoration.denoise_tv_bregman(img, weight=60.0, channel_axis=-1)

    assert_array_equal(out1, out2)


def test_denoise_bilateral_null():
    img = np.zeros((50, 50))
    out = restoration.denoise_bilateral(img)

    # image full of zeros should return identity
    assert_array_equal(out, img)


def test_denoise_bilateral_negative():
    img = -np.ones((50, 50))
    out = restoration.denoise_bilateral(img)

    # image with only negative values should be ok
    assert_array_equal(out, img)


def test_denoise_bilateral_negative2():
    img = np.ones((50, 50))
    img[2, 2] = 2

    out1 = restoration.denoise_bilateral(img)
    out2 = restoration.denoise_bilateral(img - 10)  # contains negative values

    # 2 images with a given offset should give the same result (with the same
    # offset)
    assert_array_almost_equal(out1, out2 + 10)


def test_denoise_bilateral_2d():
    img = checkerboard_gray.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=10, channel_axis=None)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                         sigma_spatial=20, channel_axis=None)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_pad():
    """This test checks if the bilateral filter is returning an image
    correctly padded."""
    img = img_as_float(data.chelsea())[100:200, 100:200]
    img_bil = restoration.denoise_bilateral(img, sigma_color=0.1,
                                            sigma_spatial=10,
                                            channel_axis=-1)
    condition_padding = np.count_nonzero(np.isclose(img_bil,
                                                    0,
                                                    atol=0.001))
    assert_array_equal(condition_padding, 0)


@pytest.mark.parametrize('dtype', [np.float32, np.double])
def test_denoise_bilateral_types(dtype):
    img = checkerboard_gray.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1).astype(dtype)

    # check that we can process multiple float types
    restoration.denoise_bilateral(img, sigma_color=0.1,
                                  sigma_spatial=10, channel_axis=None)


@pytest.mark.parametrize('dtype', [np.float32, np.double])
def test_denoise_bregman_types(dtype):
    img = checkerboard_gray.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1).astype(dtype)

    # check that we can process multiple float types
    restoration.denoise_tv_bregman(img, weight=5)


def test_denoise_bilateral_zeros():
    img = np.zeros((10, 10))
    assert_array_equal(img,
                       restoration.denoise_bilateral(img, channel_axis=None))


def test_denoise_bilateral_constant():
    img = np.ones((10, 10)) * 5
    assert_array_equal(img,
                       restoration.denoise_bilateral(img, channel_axis=None))


@pytest.mark.parametrize('channel_axis', [0, 1, -1])
def test_denoise_bilateral_color(channel_axis):
    img = checkerboard.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    img = np.moveaxis(img, -1, channel_axis)
    out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                         sigma_spatial=10,
                                         channel_axis=channel_axis)
    out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                         sigma_spatial=20,
                                         channel_axis=channel_axis)
    img = np.moveaxis(img, channel_axis, -1)
    out1 = np.moveaxis(out1, channel_axis, -1)
    out2 = np.moveaxis(out2, channel_axis, -1)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_multichannel_deprecation():
    img = checkerboard.copy()[:50, :50]
    # add some random noise
    img += 0.5 * img.std() * np.random.rand(*img.shape)
    img = np.clip(img, 0, 1)

    with expected_warnings(["`multichannel` is a deprecated argument"]):
        out1 = restoration.denoise_bilateral(img, sigma_color=0.1,
                                             sigma_spatial=10,
                                             multichannel=True)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        out2 = restoration.denoise_bilateral(img, sigma_color=0.2,
                                             sigma_spatial=20,
                                             multichannel=True)

    # make sure noise is reduced in the checkerboard cells
    assert img[30:45, 5:15].std() > out1[30:45, 5:15].std()
    assert out1[30:45, 5:15].std() > out2[30:45, 5:15].std()


def test_denoise_bilateral_3d_grayscale():
    img = np.ones((50, 50, 3))
    with pytest.raises(ValueError):
        restoration.denoise_bilateral(img, channel_axis=None)


def test_denoise_bilateral_3d_multichannel():
    img = np.ones((50, 50, 50))
    with expected_warnings(["grayscale"]):
        result = restoration.denoise_bilateral(img, channel_axis=-1)

    assert_array_equal(result, img)


def test_denoise_bilateral_multidimensional():
    img = np.ones((10, 10, 10, 10))
    with pytest.raises(ValueError):
        restoration.denoise_bilateral(img, channel_axis=None)
    with pytest.raises(ValueError):
        restoration.denoise_bilateral(img, channel_axis=-1)


def test_denoise_bilateral_nan():
    img = np.full((50, 50), np.NaN)
    # This is in fact an optional warning for our test suite.
    # Python 3.5 will not trigger a warning.
    with expected_warnings([r'invalid|\A\Z']):
        out = restoration.denoise_bilateral(img, channel_axis=None)
    assert_array_equal(img, out)


@pytest.mark.parametrize('fast_mode', [False, True])
def test_denoise_nl_means_2d(fast_mode):
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.
    sigma = 0.3
    img += sigma * np.random.standard_normal(img.shape)
    img_f32 = img.astype('float32')
    for s in [sigma, 0]:
        denoised = restoration.denoise_nl_means(img, 7, 5, 0.2,
                                                fast_mode=fast_mode,
                                                channel_axis=None,
                                                sigma=s)
        # make sure noise is reduced
        assert img.std() > denoised.std()

        denoised_f32 = restoration.denoise_nl_means(img_f32, 7, 5, 0.2,
                                                    fast_mode=fast_mode,
                                                    channel_axis=None,
                                                    sigma=s)
        # make sure noise is reduced
        assert img.std() > denoised_f32.std()

        # Sheck single precision result
        assert np.allclose(denoised_f32, denoised, atol=1e-2)


@pytest.mark.parametrize('fast_mode', [False, True])
@pytest.mark.parametrize('n_channels', [2, 3, 6])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_denoise_nl_means_2d_multichannel(fast_mode, n_channels, dtype):
    # reduce image size because nl means is slow
    img = np.copy(astro[:50, :50])
    img = np.concatenate((img, ) * 2, )  # 6 channels
    img = img.astype(dtype)

    # add some random noise
    sigma = 0.1
    imgn = img + sigma * np.random.standard_normal(img.shape)
    imgn = np.clip(imgn, 0, 1)
    imgn = imgn.astype(dtype)

    for s in [sigma, 0]:
        psnr_noisy = peak_signal_noise_ratio(
            img[..., :n_channels], imgn[..., :n_channels])
        denoised = restoration.denoise_nl_means(imgn[..., :n_channels],
                                                3, 5, h=0.75 * sigma,
                                                fast_mode=fast_mode,
                                                channel_axis=-1,
                                                sigma=s)
        psnr_denoised = peak_signal_noise_ratio(
            denoised[..., :n_channels], img[..., :n_channels])

        # make sure noise is reduced
        assert psnr_denoised > psnr_noisy


def test_denoise_nl_means_2d_multichannel_deprecated():
    # reduce image size because nl means is slow
    img = np.copy(astro[:50, :50])

    # add some random noise
    sigma = 0.1
    imgn = img + sigma * np.random.standard_normal(img.shape)
    imgn = np.clip(imgn, 0, 1)

    psnr_noisy = peak_signal_noise_ratio(img, imgn)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        denoised = restoration.denoise_nl_means(imgn,
                                                3, 5, h=0.75 * sigma,
                                                multichannel=True,
                                                sigma=sigma)
    psnr_denoised = peak_signal_noise_ratio(denoised, img)

    # make sure noise is reduced
    assert psnr_denoised > psnr_noisy

    # providing multichannel argument positionally also warns
    with expected_warnings(["Providing the `multichannel` argument"]):
        restoration.denoise_nl_means(imgn, 3, 5, 0.75 * sigma, True,
                                     sigma=sigma)


@pytest.mark.parametrize('fast_mode', [False, True])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_denoise_nl_means_3d(fast_mode, dtype):
    img = np.zeros((12, 12, 8), dtype=dtype)
    img[5:-5, 5:-5, 2:-2] = 1.
    sigma = 0.3
    imgn = img + sigma * np.random.standard_normal(img.shape)
    imgn = imgn.astype(dtype)
    psnr_noisy = peak_signal_noise_ratio(img, imgn)
    for s in [sigma, 0]:
        denoised = restoration.denoise_nl_means(imgn, 3, 4, h=0.75 * sigma,
                                                fast_mode=fast_mode,
                                                channel_axis=None, sigma=s)
        # make sure noise is reduced
        assert peak_signal_noise_ratio(img, denoised) > psnr_noisy


@pytest.mark.parametrize('fast_mode', [False, True])
@pytest.mark.parametrize('dtype', ['float64', 'float32', 'float16'])
@pytest.mark.parametrize('channel_axis', [0, -1])
def test_denoise_nl_means_multichannel(fast_mode, dtype, channel_axis):
    # for true 3D data, 3D denoising is better than denoising as 2D+channels

    # synthetic 3d volume
    img = data.binary_blobs(length=32, n_dim=3, seed=5)
    img = img[:, :24, :16].astype(dtype, copy=False)

    sigma = 0.2
    rng = np.random.default_rng(5)
    imgn = img + sigma * rng.standard_normal(img.shape)
    imgn = imgn.astype(dtype)

    # test 3D denoising (channel_axis = None)
    denoised_ok_multichannel = restoration.denoise_nl_means(
        imgn.copy(), 3, 2, h=0.6 * sigma, sigma=sigma, fast_mode=fast_mode,
        channel_axis=None)

    # set a channel axis: one dimension is (incorrectly) considered "channels"
    imgn = np.moveaxis(imgn, -1, channel_axis)
    denoised_wrong_multichannel = restoration.denoise_nl_means(
        imgn.copy(), 3, 2, h=0.6 * sigma, sigma=sigma, fast_mode=fast_mode,
        channel_axis=channel_axis
    )
    denoised_wrong_multichannel = np.moveaxis(
        denoised_wrong_multichannel, channel_axis, -1
    )

    img = img.astype(denoised_wrong_multichannel.dtype)
    psnr_wrong = peak_signal_noise_ratio(img, denoised_wrong_multichannel)
    psnr_ok = peak_signal_noise_ratio(img, denoised_ok_multichannel)
    assert psnr_ok > psnr_wrong


def test_denoise_nl_means_4d():
    rng = np.random.default_rng(5)
    img = np.zeros((10, 10, 8, 5))
    img[2:-2, 2:-2, 2:-2, :2] = 0.5
    img[2:-2, 2:-2, 2:-2, 2:] = 1.
    sigma = 0.3
    imgn = img + sigma * rng.standard_normal(img.shape)

    nlmeans_kwargs = dict(patch_size=3, patch_distance=2, h=0.3 * sigma,
                          sigma=sigma, fast_mode=True)

    psnr_noisy = peak_signal_noise_ratio(img, imgn, data_range=1.)

    # denoise by looping over 3D slices
    denoised_3d = np.zeros_like(imgn)
    for ch in range(img.shape[-1]):
        denoised_3d[..., ch] = restoration.denoise_nl_means(
            imgn[..., ch],
            channel_axis=None,
            **nlmeans_kwargs)
    psnr_3d = peak_signal_noise_ratio(img, denoised_3d, data_range=1.)
    assert psnr_3d > psnr_noisy

    # denoise as 4D
    denoised_4d = restoration.denoise_nl_means(imgn,
                                               channel_axis=None,
                                               **nlmeans_kwargs)
    psnr_4d = peak_signal_noise_ratio(img, denoised_4d, data_range=1.)
    assert psnr_4d > psnr_3d

    # denoise as 3D + channels instead
    denoised_3dmc = restoration.denoise_nl_means(imgn,
                                                 channel_axis=-1,
                                                 **nlmeans_kwargs)
    psnr_3dmc = peak_signal_noise_ratio(img, denoised_3dmc, data_range=1.)
    assert psnr_3dmc > psnr_3d


def test_denoise_nl_means_4d_multichannel():
    img = np.zeros((8, 8, 8, 4, 4))
    img[2:-2, 2:-2, 2:-2, 1:-1, :] = 1.
    sigma = 0.3
    imgn = img + sigma * np.random.randn(*img.shape)

    psnr_noisy = peak_signal_noise_ratio(img, imgn, data_range=1.)

    denoised_4dmc = restoration.denoise_nl_means(imgn, 3, 3, h=0.35 * sigma,
                                                 fast_mode=True,
                                                 channel_axis=-1,
                                                 sigma=sigma)
    psnr_4dmc = peak_signal_noise_ratio(img, denoised_4dmc, data_range=1.)
    assert psnr_4dmc > psnr_noisy


def test_denoise_nl_means_wrong_dimension():
    # 1D not implemented
    img = np.zeros((5, ))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None)

    img = np.zeros((5, 3))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1)

    # 3D + channels only for fast mode
    img = np.zeros((5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1, fast_mode=False)

    # 4D only for fast mode
    img = np.zeros((5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None, fast_mode=False)

    # 4D + channels only for fast mode
    img = np.zeros((5, 5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=-1, fast_mode=False)

    # 5D not implemented
    img = np.zeros((5, 5, 5, 5, 5))
    with pytest.raises(NotImplementedError):
        restoration.denoise_nl_means(img, channel_axis=None)


@pytest.mark.parametrize('fast_mode', [False, True])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_no_denoising_for_small_h(fast_mode, dtype):
    img = np.zeros((40, 40))
    img[10:-10, 10:-10] = 1.
    img += 0.3 * np.random.standard_normal(img.shape)
    img = img.astype(dtype)
    # very small h should result in no averaging with other patches
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01,
                                            fast_mode=fast_mode,
                                            channel_axis=None)
    assert np.allclose(denoised, img)
    denoised = restoration.denoise_nl_means(img, 7, 5, 0.01,
                                            fast_mode=fast_mode,
                                            channel_axis=None)
    assert np.allclose(denoised, img)


@pytest.mark.parametrize('fast_mode', [False, True])
def test_denoise_nl_means_2d_dtype(fast_mode):
    img = np.zeros((40, 40), dtype=int)
    img_f32 = img.astype('float32')
    img_f64 = img.astype('float64')

    assert restoration.denoise_nl_means(
        img, fast_mode=fast_mode).dtype == 'float64'

    assert restoration.denoise_nl_means(
        img_f32, fast_mode=fast_mode).dtype == img_f32.dtype

    assert restoration.denoise_nl_means(
        img_f64, fast_mode=fast_mode).dtype == img_f64.dtype


@pytest.mark.parametrize('fast_mode', [False, True])
def test_denoise_nl_means_3d_dtype(fast_mode):
    img = np.zeros((12, 12, 8), dtype=int)
    img_f32 = img.astype('float32')
    img_f64 = img.astype('float64')

    assert restoration.denoise_nl_means(
        img, patch_distance=2, fast_mode=fast_mode).dtype == 'float64'

    assert restoration.denoise_nl_means(
        img_f32, patch_distance=2, fast_mode=fast_mode).dtype == img_f32.dtype

    assert restoration.denoise_nl_means(
        img_f64, patch_distance=2, fast_mode=fast_mode).dtype == img_f64.dtype


@pytest.mark.parametrize(
    'img, multichannel, convert2ycbcr',
    [(astro_gray, False, False),
     (astro_gray_odd, False, False),
     (astro_odd, True, False),
     (astro_odd, True, True)]
)
def test_wavelet_denoising(img, multichannel, convert2ycbcr):
    rstate = np.random.default_rng(1234)
    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)

    channel_axis = -1 if multichannel else None

    # Verify that SNR is improved when true sigma is used
    denoised = restoration.denoise_wavelet(noisy, sigma=sigma,
                                           channel_axis=channel_axis,
                                           convert2ycbcr=convert2ycbcr,
                                           rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy

    # Verify that SNR is improved with internally estimated sigma
    denoised = restoration.denoise_wavelet(noisy,
                                           channel_axis=channel_axis,
                                           convert2ycbcr=convert2ycbcr,
                                           rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy

    # SNR is improved less with 1 wavelet level than with the default.
    denoised_1 = restoration.denoise_wavelet(noisy,
                                             channel_axis=channel_axis,
                                             wavelet_levels=1,
                                             convert2ycbcr=convert2ycbcr,
                                             rescale_sigma=True)
    psnr_denoised_1 = peak_signal_noise_ratio(img, denoised_1)
    assert psnr_denoised > psnr_denoised_1
    assert psnr_denoised_1 > psnr_noisy

    # Test changing noise_std (higher threshold, so less energy in signal)
    res1 = restoration.denoise_wavelet(noisy, sigma=2 * sigma,
                                       channel_axis=channel_axis,
                                       rescale_sigma=True)
    res2 = restoration.denoise_wavelet(noisy, sigma=sigma,
                                       channel_axis=channel_axis,
                                       rescale_sigma=True)
    assert np.sum(res1**2) <= np.sum(res2**2)


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
@pytest.mark.parametrize('convert2ycbcr', [False, True])
def test_wavelet_denoising_channel_axis(channel_axis, convert2ycbcr):
    rstate = np.random.default_rng(1234)
    sigma = 0.1
    img = astro_odd
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)

    img = np.moveaxis(img, -1, channel_axis)
    noisy = np.moveaxis(noisy, -1, channel_axis)

    # Verify that SNR is improved when true sigma is used
    denoised = restoration.denoise_wavelet(noisy, sigma=sigma,
                                           channel_axis=channel_axis,
                                           convert2ycbcr=convert2ycbcr,
                                           rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy


def test_wavelet_denoising_deprecated():
    rstate = np.random.default_rng(1234)
    sigma = 0.1
    img = astro_odd
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)

    with expected_warnings(["`multichannel` is a deprecated argument"]):
        # Verify that SNR is improved when true sigma is used
        denoised = restoration.denoise_wavelet(noisy, sigma=sigma,
                                               multichannel=True,
                                               rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy

    # providing multichannel argument positionally also warns
    with expected_warnings(["Providing the `multichannel` argument"]):
        restoration.denoise_wavelet(noisy, sigma, 'db1', 'soft', None, True,
                                    rescale_sigma=True)


@pytest.mark.parametrize(
    'case, dtype, convert2ycbcr, estimate_sigma',
    itertools.product(
        ['1d', '2d multichannel'],
        [np.float16, np.float32, np.float64, np.int16, np.uint8],
        [True, False],
        [True, False])
)
def test_wavelet_denoising_scaling(case, dtype, convert2ycbcr,
                                   estimate_sigma):
    """Test cases for images without prescaling via img_as_float."""
    rstate = np.random.default_rng(1234)

    if case == '1d':
        # 1D single-channel in range [0, 255]
        x = np.linspace(0, 255, 1024)
    elif case == '2d multichannel':
        # 2D multichannel in range [0, 255]
        x = data.astronaut()[:64, :64]
    x = x.astype(dtype)

    # add noise and clip to original signal range
    sigma = 25.
    noisy = x + sigma * rstate.standard_normal(x.shape)
    noisy = np.clip(noisy, x.min(), x.max())
    noisy = noisy.astype(x.dtype)

    channel_axis = -1 if x.shape[-1] == 3 else None

    if estimate_sigma:
        sigma_est = restoration.estimate_sigma(noisy,
                                               channel_axis=channel_axis)
    else:
        sigma_est = None

    if convert2ycbcr and channel_axis is None:
        # YCbCr requires multichannel == True
        with pytest.raises(ValueError):
            denoised = restoration.denoise_wavelet(noisy,
                                                   sigma=sigma_est,
                                                   wavelet='sym4',
                                                   channel_axis=channel_axis,
                                                   convert2ycbcr=convert2ycbcr,
                                                   rescale_sigma=True)
        return

    denoised = restoration.denoise_wavelet(noisy, sigma=sigma_est,
                                           wavelet='sym4',
                                           channel_axis=channel_axis,
                                           convert2ycbcr=convert2ycbcr,
                                           rescale_sigma=True)
    assert denoised.dtype == _supported_float_type(noisy)

    data_range = x.max() - x.min()
    psnr_noisy = peak_signal_noise_ratio(x, noisy, data_range=data_range)
    clipped = np.dtype(dtype).kind != 'f'
    if not clipped:
        psnr_denoised = peak_signal_noise_ratio(x, denoised,
                                                data_range=data_range)

        # output's max value is not substantially smaller than x's
        assert denoised.max() > 0.9 * x.max()
    else:
        # have to compare to x_as_float in integer input cases
        x_as_float = img_as_float(x)
        f_data_range = x_as_float.max() - x_as_float.min()
        psnr_denoised = peak_signal_noise_ratio(x_as_float, denoised,
                                                data_range=f_data_range)

        # output has been clipped to expected range
        assert denoised.max() <= 1.0
        if np.dtype(dtype).kind == 'u':
            assert denoised.min() >= 0
        else:
            assert denoised.min() >= -1

    assert psnr_denoised > psnr_noisy


def test_wavelet_threshold():
    rstate = np.random.default_rng(1234)

    img = astro_gray
    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)

    # employ a single, user-specified threshold instead of BayesShrink sigmas
    denoised = _wavelet_threshold(noisy, wavelet='db1', method=None,
                                  threshold=sigma)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy

    # either method or threshold must be defined
    with pytest.raises(ValueError):
        _wavelet_threshold(noisy, wavelet='db1', method=None, threshold=None)

    # warns if a threshold is provided in a case where it would be ignored
    with expected_warnings(["Thresholding method "]):
        _wavelet_threshold(noisy, wavelet='db1', method='BayesShrink',
                           threshold=sigma)


@pytest.mark.parametrize(
    'rescale_sigma, method, ndim',
    itertools.product(
        [True, False],
        ['VisuShrink', 'BayesShrink'],
        range(1, 5)
    )
)
def test_wavelet_denoising_nd(rescale_sigma, method, ndim):
    rstate = np.random.default_rng(1234)
    # Generate a very simple test image
    if ndim < 3:
        img = 0.2*np.ones((128, )*ndim)
    else:
        img = 0.2*np.ones((16, )*ndim)
    img[(slice(5, 13), ) * ndim] = 0.8

    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)

    # Mark H. 2018.08:
    #   The issue arises because when ndim in [1, 2]
    #   ``waverecn`` calls ``_match_coeff_dims``
    #   Which includes a numpy 1.15 deprecation.
    #   for larger number of dimensions _match_coeff_dims isn't called
    #   for some reason.
    # Verify that SNR is improved with internally estimated sigma
    denoised = restoration.denoise_wavelet(
        noisy, method=method,
        rescale_sigma=rescale_sigma)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy


def test_wavelet_invalid_method():
    with pytest.raises(ValueError):
        restoration.denoise_wavelet(np.ones(16), method='Unimplemented',
                                    rescale_sigma=True)


@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_wavelet_denoising_levels(rescale_sigma):
    rstate = np.random.default_rng(1234)
    ndim = 2
    N = 256
    wavelet = 'db1'
    # Generate a very simple test image
    img = 0.2*np.ones((N, )*ndim)
    img[(slice(5, 13), ) * ndim] = 0.8

    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)

    denoised = restoration.denoise_wavelet(noisy, wavelet=wavelet,
                                           rescale_sigma=rescale_sigma)
    denoised_1 = restoration.denoise_wavelet(noisy, wavelet=wavelet,
                                             wavelet_levels=1,
                                             rescale_sigma=rescale_sigma)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    psnr_denoised_1 = peak_signal_noise_ratio(img, denoised_1)

    # multi-level case should outperform single level case
    assert psnr_denoised > psnr_denoised_1 > psnr_noisy

    # invalid number of wavelet levels results in a ValueError or UserWarning
    max_level = pywt.dwt_max_level(np.min(img.shape),
                                   pywt.Wavelet(wavelet).dec_len)
    # exceeding max_level raises a UserWarning in PyWavelets >= 1.0.0
    with expected_warnings([
            'all coefficients will experience boundary effects']):
        restoration.denoise_wavelet(
            noisy, wavelet=wavelet, wavelet_levels=max_level + 1,
            rescale_sigma=rescale_sigma)

    with pytest.raises(ValueError):
        restoration.denoise_wavelet(
            noisy,
            wavelet=wavelet, wavelet_levels=-1,
            rescale_sigma=rescale_sigma)


def test_estimate_sigma_gray():
    rstate = np.random.default_rng(1234)
    # astronaut image
    img = astro_gray.copy()
    sigma = 0.1
    # add noise to astronaut
    img += sigma * rstate.standard_normal(img.shape)

    sigma_est = restoration.estimate_sigma(img, channel_axis=None)
    assert_array_almost_equal(sigma, sigma_est, decimal=2)


def test_estimate_sigma_masked_image():
    # Verify computation on an image with a large, noise-free border.
    # (zero regions will be masked out by _sigma_est_dwt to avoid returning
    #  sigma = 0)
    rstate = np.random.default_rng(1234)
    # uniform image
    img = np.zeros((128, 128))
    center_roi = (slice(32, 96), slice(32, 96))
    img[center_roi] = 0.8
    sigma = 0.1

    img[center_roi] = sigma * rstate.standard_normal(img[center_roi].shape)

    sigma_est = restoration.estimate_sigma(img, channel_axis=None)
    assert_array_almost_equal(sigma, sigma_est, decimal=1)


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_estimate_sigma_color(channel_axis):
    rstate = np.random.default_rng(1234)
    # astronaut image
    img = astro.copy()
    sigma = 0.1
    # add noise to astronaut
    img += sigma * rstate.standard_normal(img.shape)
    img = np.moveaxis(img, -1, channel_axis)

    sigma_est = restoration.estimate_sigma(img, channel_axis=channel_axis,
                                           average_sigmas=True)
    assert_array_almost_equal(sigma, sigma_est, decimal=2)

    sigma_list = restoration.estimate_sigma(img, channel_axis=channel_axis,
                                            average_sigmas=False)
    assert_array_equal(len(sigma_list), img.shape[channel_axis])
    assert_array_almost_equal(sigma_list[0], sigma_est, decimal=2)

    if channel_axis % img.ndim == 2:
        # default channel_axis=None should raise a warning about last axis size
        assert_warns(UserWarning, restoration.estimate_sigma, img)


def test_estimate_sigma_color_deprecated_multichannel():
    rstate = np.random.default_rng(1234)
    # astronaut image
    img = astro.copy()
    sigma = 0.1
    # add noise to astronaut
    img += sigma * rstate.standard_normal(img.shape)

    with expected_warnings(["`multichannel` is a deprecated argument"]):
        sigma_est = restoration.estimate_sigma(img, multichannel=True,
                                               average_sigmas=True)
    assert_array_almost_equal(sigma, sigma_est, decimal=2)

    # providing multichannel argument positionally also warns
    with expected_warnings(["Providing the `multichannel` argument"]):
        sigma_est = restoration.estimate_sigma(img, True, True)
    assert_array_almost_equal(sigma, sigma_est, decimal=2)


@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_wavelet_denoising_args(rescale_sigma):
    """
    Some of the functions inside wavelet denoising throw an error the wrong
    arguments are passed. This protects against that and verifies that all
    arguments can be passed.
    """
    img = astro
    noisy = img.copy() + 0.1 * np.random.standard_normal(img.shape)

    for convert2ycbcr in [True, False]:
        for multichannel in [True, False]:
            channel_axis = -1 if multichannel else None
            if convert2ycbcr and not multichannel:
                with pytest.raises(ValueError):
                    restoration.denoise_wavelet(noisy,
                                                convert2ycbcr=convert2ycbcr,
                                                channel_axis=channel_axis,
                                                rescale_sigma=rescale_sigma)
                continue
            for sigma in [0.1, [0.1, 0.1, 0.1], None]:
                if (not multichannel and not convert2ycbcr) or \
                        (isinstance(sigma, list) and not multichannel):
                    continue
                restoration.denoise_wavelet(noisy, sigma=sigma,
                                            convert2ycbcr=convert2ycbcr,
                                            channel_axis=channel_axis,
                                            rescale_sigma=rescale_sigma)


@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_denoise_wavelet_biorthogonal(rescale_sigma):
    """Biorthogonal wavelets should raise a warning during thresholding."""
    img = astro_gray
    assert_warns(UserWarning, restoration.denoise_wavelet, img,
                 wavelet='bior2.2', channel_axis=None,
                 rescale_sigma=rescale_sigma)


@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_cycle_spinning_multichannel(rescale_sigma):
    sigma = 0.1
    rstate = np.random.default_rng(1234)

    for channel_axis in -1, None:
        if channel_axis is not None:
            img = astro
            # can either omit or be 0 along the channels axis
            valid_shifts = [1, (0, 1), (1, 0), (1, 1), (1, 1, 0)]
            # can either omit or be 1 on channels axis.
            valid_steps = [1, 2, (1, 2), (1, 2, 1)]
            # too few or too many shifts or non-zero shift on channels
            invalid_shifts = [(1, 1, 2), (1, ), (1, 1, 0, 1)]
            # too few or too many shifts or any shifts <= 0
            invalid_steps = [(1, ), (1, 1, 1, 1), (0, 1), (-1, -1)]
        else:
            img = astro_gray
            valid_shifts = [1, (0, 1), (1, 0), (1, 1)]
            valid_steps = [1, 2, (1, 2)]
            invalid_shifts = [(1, 1, 2), (1, )]
            invalid_steps = [(1, ), (1, 1, 1), (0, 1), (-1, -1)]

        noisy = img.copy() + 0.1 * rstate.standard_normal(img.shape)

        denoise_func = restoration.denoise_wavelet
        func_kw = dict(sigma=sigma, channel_axis=channel_axis,
                       rescale_sigma=rescale_sigma)

        # max_shifts=0 is equivalent to just calling denoise_func
        with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
            dn_cc = restoration.cycle_spin(noisy, denoise_func, max_shifts=0,
                                           func_kw=func_kw,
                                           channel_axis=channel_axis)
            dn = denoise_func(noisy, **func_kw)
        assert_array_equal(dn, dn_cc)

        # denoising with cycle spinning will give better PSNR than without
        for max_shifts in valid_shifts:
            with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
                dn_cc = restoration.cycle_spin(noisy, denoise_func,
                                               max_shifts=max_shifts,
                                               func_kw=func_kw,
                                               channel_axis=channel_axis)
            psnr = peak_signal_noise_ratio(img, dn)
            psnr_cc = peak_signal_noise_ratio(img, dn_cc)
            assert psnr_cc > psnr

        for shift_steps in valid_steps:
            with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
                dn_cc = restoration.cycle_spin(noisy, denoise_func,
                                               max_shifts=2,
                                               shift_steps=shift_steps,
                                               func_kw=func_kw,
                                               channel_axis=channel_axis)
            psnr = peak_signal_noise_ratio(img, dn)
            psnr_cc = peak_signal_noise_ratio(img, dn_cc)
            assert psnr_cc > psnr

        for max_shifts in invalid_shifts:
            with pytest.raises(ValueError):
                dn_cc = restoration.cycle_spin(noisy, denoise_func,
                                               max_shifts=max_shifts,
                                               func_kw=func_kw,
                                               channel_axis=channel_axis)
        for shift_steps in invalid_steps:
            with pytest.raises(ValueError):
                dn_cc = restoration.cycle_spin(noisy, denoise_func,
                                               max_shifts=2,
                                               shift_steps=shift_steps,
                                               func_kw=func_kw,
                                               channel_axis=channel_axis)


def test_cycle_spinning_num_workers():
    img = astro_gray
    sigma = 0.1
    rstate = np.random.default_rng(1234)
    noisy = img.copy() + 0.1 * rstate.standard_normal(img.shape)

    denoise_func = restoration.denoise_wavelet
    func_kw = dict(sigma=sigma, channel_axis=-1, rescale_sigma=True)

    # same results are expected whether using 1 worker or multiple workers
    dn_cc1 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1,
                                    func_kw=func_kw, channel_axis=None,
                                    num_workers=1)
    with expected_warnings([DASK_NOT_INSTALLED_WARNING]):
        dn_cc2 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1,
                                        func_kw=func_kw, channel_axis=None,
                                        num_workers=4)
        dn_cc3 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1,
                                        func_kw=func_kw, channel_axis=None,
                                        num_workers=None)
    assert_array_almost_equal(dn_cc1, dn_cc2)
    assert_array_almost_equal(dn_cc1, dn_cc3)


def test_cycle_spinning_num_workers_deprecated_multichannel():
    img = astro_gray[:32, :32]
    sigma = 0.1
    rstate = np.random.default_rng(1234)
    noisy = img.copy() + 0.1 * rstate.standard_normal(img.shape)

    denoise_func = restoration.denoise_wavelet

    func_kw = dict(sigma=sigma, channel_axis=-1, rescale_sigma=True)

    mc_warn_str = "`multichannel` is a deprecated argument"

    # same results are expected whether using 1 worker or multiple workers
    with expected_warnings([mc_warn_str]):
        dn_cc1 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1,
                                        func_kw=func_kw, multichannel=False,
                                        num_workers=1)

    if DASK_NOT_INSTALLED_WARNING is None:
        exp_warn = [mc_warn_str]
    else:
        exp_warn = [mc_warn_str, DASK_NOT_INSTALLED_WARNING]
    with expected_warnings(exp_warn):
        dn_cc2 = restoration.cycle_spin(noisy, denoise_func, max_shifts=1,
                                        func_kw=func_kw, multichannel=False,
                                        num_workers=2)
    assert_array_almost_equal(dn_cc1, dn_cc2)

    # providing multichannel argument positionally also warns
    mc_warn_str = "Providing the `multichannel` argument"
    if DASK_NOT_INSTALLED_WARNING is None:
        exp_warn = [mc_warn_str]
    else:
        exp_warn = [mc_warn_str, DASK_NOT_INSTALLED_WARNING]

    with expected_warnings(exp_warn):
        restoration.cycle_spin(noisy, denoise_func, 1, 1, None, False)
