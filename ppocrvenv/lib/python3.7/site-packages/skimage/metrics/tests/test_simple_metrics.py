import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal

from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.metrics import (peak_signal_noise_ratio, normalized_root_mse,
                             mean_squared_error, normalized_mutual_information)


np.random.seed(5)
cam = data.camera()
sigma = 20.0
cam_noisy = np.clip(cam + sigma * np.random.randn(*cam.shape), 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)


def test_PSNR_vs_IPOL():
    """ Tests vs. imdiff result from the following IPOL article and code:
    https://www.ipol.im/pub/art/2011/g_lmii/.

    Notes
    -----
    To generate p_IPOL, we need a local copy of cam_noisy:

    >>> from skimage import io
    >>> io.imsave('/tmp/cam_noisy.png', cam_noisy)

    Then, we use the following command:
    $ ./imdiff -m psnr <path to camera.png>/camera.png /tmp/cam_noisy.png

    Values for current data.camera() calculated by Gregory Lee on Sep, 2020.
    Available at:
    https://github.com/scikit-image/scikit-image/pull/4913#issuecomment-700653165
    """
    p_IPOL = 22.409353363576034
    p = peak_signal_noise_ratio(cam, cam_noisy)
    assert_almost_equal(p, p_IPOL, decimal=4)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_PSNR_float(dtype):
    p_uint8 = peak_signal_noise_ratio(cam, cam_noisy)
    camf = (cam / 255.).astype(dtype, copy=False)
    camf_noisy = (cam_noisy / 255.).astype(dtype, copy=False)
    p_float64 = peak_signal_noise_ratio(camf, camf_noisy, data_range=1)
    assert p_float64.dtype == np.float64
    decimal = 3 if dtype == np.float16 else 5
    assert_almost_equal(p_uint8, p_float64, decimal=decimal)

    # mixed precision inputs
    p_mixed = peak_signal_noise_ratio(cam / 255., np.float32(cam_noisy / 255.),
                                      data_range=1)

    assert_almost_equal(p_mixed, p_float64, decimal=decimal)

    # mismatched dtype results in a warning if data_range is unspecified
    with expected_warnings(['Inputs have mismatched dtype']):
        p_mixed = peak_signal_noise_ratio(cam / 255.,
                                          np.float32(cam_noisy / 255.))
    assert_almost_equal(p_mixed, p_float64, decimal=decimal)

    # mismatched dtype results in a warning if data_range is unspecified
    with expected_warnings(['Inputs have mismatched dtype']):
        p_mixed = peak_signal_noise_ratio(cam / 255.,
                                          np.float32(cam_noisy / 255.))
    assert_almost_equal(p_mixed, p_float64, decimal=decimal)


def test_PSNR_errors():
    # shape mismatch
    with pytest.raises(ValueError):
        peak_signal_noise_ratio(cam, cam[:-1, :])


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_NRMSE(dtype):
    x = np.ones(4, dtype=dtype)
    y = np.asarray([0., 2., 2., 2.], dtype=dtype)
    nrmse = normalized_root_mse(y, x, normalization='mean')
    assert nrmse.dtype == np.float64
    assert_equal(nrmse, 1 / np.mean(y))
    assert_equal(normalized_root_mse(y, x, normalization='euclidean'),
                 1 / np.sqrt(3))
    assert_equal(normalized_root_mse(y, x, normalization='min-max'),
                 1 / (y.max() - y.min()))

    # mixed precision inputs are allowed
    assert_almost_equal(normalized_root_mse(y, np.float32(x),
                                            normalization='min-max'),
                        1 / (y.max() - y.min()))


def test_NRMSE_no_int_overflow():
    camf = cam.astype(np.float32)
    cam_noisyf = cam_noisy.astype(np.float32)
    assert_almost_equal(mean_squared_error(cam, cam_noisy),
                        mean_squared_error(camf, cam_noisyf))
    assert_almost_equal(normalized_root_mse(cam, cam_noisy),
                        normalized_root_mse(camf, cam_noisyf))


def test_NRMSE_errors():
    x = np.ones(4)
    # shape mismatch
    with pytest.raises(ValueError):
        normalized_root_mse(x[:-1], x)
    # invalid normalization name
    with pytest.raises(ValueError):
        normalized_root_mse(x, x, normalization='foo')


def test_nmi():
    assert_almost_equal(normalized_mutual_information(cam, cam), 2)
    assert (normalized_mutual_information(cam, cam_noisy)
            < normalized_mutual_information(cam, cam))


def test_nmi_different_sizes():
    assert normalized_mutual_information(cam[:, :400], cam[:400, :]) > 1


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_nmi_random(dtype):
    rng = np.random.default_rng()
    random1 = rng.random((100, 100)).astype(dtype)
    random2 = rng.random((100, 100)).astype(dtype)
    nmi = normalized_mutual_information(random1, random2, bins=10)
    assert nmi.dtype == np.float64
    assert_almost_equal(nmi, 1, decimal=2)


def test_nmi_random_3d():
    random1, random2 = np.random.random((2, 10, 100, 100))
    assert_almost_equal(
        normalized_mutual_information(random1, random2, bins=10),
        1,
        decimal=2,
    )
