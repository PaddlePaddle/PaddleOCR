import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal

from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.metrics import structural_similarity

np.random.seed(5)
cam = data.camera()
sigma = 20.0
cam_noisy = np.clip(cam + sigma * np.random.randn(*cam.shape), 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)

np.random.seed(1234)


def test_structural_similarity_patch_range():
    N = 51
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    assert(structural_similarity(X, Y, win_size=N) < 0.1)
    assert_equal(structural_similarity(X, X, win_size=N), 1)


def test_structural_similarity_image():
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    S0 = structural_similarity(X, X, win_size=3)
    assert_equal(S0, 1)

    S1 = structural_similarity(X, Y, win_size=3)
    assert(S1 < 0.3)

    S2 = structural_similarity(X, Y, win_size=11, gaussian_weights=True)
    assert(S2 < 0.3)

    mssim0, S3 = structural_similarity(X, Y, full=True)
    assert_equal(S3.shape, X.shape)
    mssim = structural_similarity(X, Y)
    assert_equal(mssim0, mssim)

    # structural_similarity of image with itself should be 1.0
    assert_equal(structural_similarity(X, X), 1.0)


# Because we are forcing a random seed state, it is probably good to test
# against a few seeds in case on seed gives a particularly bad example
@pytest.mark.parametrize('seed', [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_structural_similarity_grad(seed, dtype):
    N = 60
    # NOTE: This test is known to randomly fail on some systems (Mac OS X 10.6)
    #       And when testing tests in parallel. Therefore, we choose a few
    #       seeds that are known to work.
    #       The likely cause of this failure is that we are setting a hard
    #       threshold on the value of the gradient. Often the computed gradient
    #       is only slightly larger than what was measured.
    rnd = np.random.default_rng(seed)
    X = rnd.random((N, N)).astype(dtype, copy=False) * 255
    Y = rnd.random((N, N)).astype(dtype, copy=False) * 255

    f = structural_similarity(X, Y, data_range=255)
    g = structural_similarity(X, Y, data_range=255, gradient=True)

    assert f < 0.05

    assert g[0] < 0.05
    assert np.all(g[1] < 0.05)

    mssim, grad, s = structural_similarity(
        X, Y, data_range=255, gradient=True, full=True)
    assert s.dtype == _supported_float_type(dtype)
    assert grad.dtype == _supported_float_type(dtype)
    assert np.all(grad < 0.05)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64]
)
def test_structural_similarity_dtype(dtype):
    N = 30
    X = np.random.rand(N, N)
    Y = np.random.rand(N, N)
    if np.dtype(dtype).kind in 'iub':
        X = (X * 255).astype(np.uint8)
        Y = (X * 255).astype(np.uint8)
    else:
        X = X.astype(dtype, copy=False)
        Y = Y.astype(dtype, copy=False)

    S1 = structural_similarity(X, Y)
    assert S1.dtype == np.float64

    assert S1 < 0.1


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_structural_similarity_multichannel(channel_axis):
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    S1 = structural_similarity(X, Y, win_size=3)

    # replicate across three channels.  should get identical value
    Xc = np.tile(X[..., np.newaxis], (1, 1, 3))
    Yc = np.tile(Y[..., np.newaxis], (1, 1, 3))

    # move channels from last position to specified channel_axis
    Xc, Yc = (np.moveaxis(_arr, -1, channel_axis) for _arr in (Xc, Yc))

    S2 = structural_similarity(Xc, Yc, channel_axis=channel_axis, win_size=3)
    assert_almost_equal(S1, S2)

    # full case should return an image as well
    m, S3 = structural_similarity(Xc, Yc, channel_axis=channel_axis, full=True)
    assert_equal(S3.shape, Xc.shape)

    # gradient case
    m, grad = structural_similarity(Xc, Yc, channel_axis=channel_axis,
                                    gradient=True)
    assert_equal(grad.shape, Xc.shape)

    # full and gradient case
    m, grad, S3 = structural_similarity(Xc, Yc,
                                        channel_axis=channel_axis,
                                        full=True,
                                        gradient=True)
    assert_equal(grad.shape, Xc.shape)
    assert_equal(S3.shape, Xc.shape)

    # fail if win_size exceeds any non-channel dimension
    with pytest.raises(ValueError):
        structural_similarity(Xc, Yc, win_size=7, channel_axis=None)


def test_structural_similarity_multichannel_deprecated():
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    S1 = structural_similarity(X, Y, win_size=3)

    # replicate across three channels.  should get identical value
    Xc = np.tile(X[..., np.newaxis], (1, 1, 3))
    Yc = np.tile(Y[..., np.newaxis], (1, 1, 3))
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        S2 = structural_similarity(Xc, Yc, multichannel=True, win_size=3)
    assert_almost_equal(S1, S2)


@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
def test_structural_similarity_nD(dtype):
    # test 1D through 4D on small random arrays
    N = 10
    for ndim in range(1, 5):
        xsize = [N, ] * 5
        X = (np.random.rand(*xsize) * 255).astype(dtype)
        Y = (np.random.rand(*xsize) * 255).astype(dtype)

        mssim = structural_similarity(X, Y, win_size=3)
        assert mssim.dtype == np.float64
        assert mssim < 0.05


def test_structural_similarity_multichannel_chelsea():
    # color image example
    Xc = data.chelsea()
    sigma = 15.0
    Yc = np.clip(Xc + sigma * np.random.randn(*Xc.shape), 0, 255)
    Yc = Yc.astype(Xc.dtype)

    # multichannel result should be mean of the individual channel results
    mssim = structural_similarity(Xc, Yc, channel_axis=-1)
    mssim_sep = [structural_similarity(
        Yc[..., c], Xc[..., c]) for c in range(Xc.shape[-1])]
    assert_almost_equal(mssim, np.mean(mssim_sep))

    # structural_similarity of image with itself should be 1.0
    assert_equal(structural_similarity(Xc, Xc, channel_axis=-1), 1.0)


def test_gaussian_structural_similarity_vs_IPOL():
    """ Tests vs. imdiff result from the following IPOL article and code:
    https://www.ipol.im/pub/art/2011/g_lmii/.

    Notes
    -----
    To generate mssim_IPOL, we need a local copy of cam_noisy:

    >>> from skimage import io
    >>> io.imsave('/tmp/cam_noisy.png', cam_noisy)

    Then, we use the following command:
    $ ./imdiff -m mssim <path to camera.png>/camera.png /tmp/cam_noisy.png

    Values for current data.camera() calculated by Gregory Lee on Sep, 2020.
    Available at:
    https://github.com/scikit-image/scikit-image/pull/4913#issuecomment-700653165
    """
    mssim_IPOL = 0.357959091663361
    mssim = structural_similarity(cam, cam_noisy, gaussian_weights=True,
                                  use_sample_covariance=False)
    assert_almost_equal(mssim, mssim_IPOL, decimal=3)


def test_mssim_vs_legacy():
    # check that ssim with default options matches skimage 0.17 result
    mssim_skimage_0pt17 = 0.3674518327910367
    mssim = structural_similarity(cam, cam_noisy)
    assert_almost_equal(mssim, mssim_skimage_0pt17)


def test_mssim_mixed_dtype():
    mssim = structural_similarity(cam, cam_noisy)
    with expected_warnings(['Inputs have mismatched dtype']):
        mssim_mixed = structural_similarity(cam, cam_noisy.astype(np.float32))
    assert_almost_equal(mssim, mssim_mixed)

    # no warning when user supplies data_range
    mssim_mixed = structural_similarity(
        cam, cam_noisy.astype(np.float32), data_range=255)
    assert_almost_equal(mssim, mssim_mixed)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_structural_similarity_small_image(dtype):
    X = np.zeros((5, 5), dtype=dtype)
    # structural_similarity can be computed for small images if win_size is
    # a) odd and b) less than or equal to the images' smaller side
    assert_equal(structural_similarity(X, X, win_size=3), 1.0)
    assert_equal(structural_similarity(X, X, win_size=5), 1.0)
    # structural_similarity errors for small images if user doesn't specify
    # win_size
    with pytest.raises(ValueError):
        structural_similarity(X, X)


def test_invalid_input():
    # size mismatch
    X = np.zeros((9, 9), dtype=np.double)
    Y = np.zeros((8, 8), dtype=np.double)
    with pytest.raises(ValueError):
        structural_similarity(X, Y)
    # win_size exceeds image extent
    with pytest.raises(ValueError):
        structural_similarity(X, X, win_size=X.shape[0] + 1)
    # some kwarg inputs must be non-negative
    with pytest.raises(ValueError):
        structural_similarity(X, X, K1=-0.1)
    with pytest.raises(ValueError):
        structural_similarity(X, X, K2=-0.1)
    with pytest.raises(ValueError):
        structural_similarity(X, X, sigma=-1.0)
