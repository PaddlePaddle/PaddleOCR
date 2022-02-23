
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack

# scipy.fft wraps the fftpack versions but with normalized inverse transforms.
# So, the forward transforms and definitions are already thoroughly tested in
# fftpack/test_real_transforms.py


@pytest.mark.parametrize("forward, backward", [(dct, idct), (dst, idst)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 10, 16])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
def test_identity_1d(forward, backward, type, n, axis, norm):
    # Test the identity f^-1(f(x)) == x
    x = np.random.rand(n, n)

    y = forward(x, type, axis=axis, norm=norm)
    z = backward(y, type, axis=axis, norm=norm)
    assert_allclose(z, x)

    pad = [(0, 0)] * 2
    pad[axis] = (0, 4)

    y2 = np.pad(y, pad, mode='edge')
    z2 = backward(y2, type, n, axis, norm)
    assert_allclose(z2, x)


@pytest.mark.parametrize("forward, backward", [(dct, idct), (dst, idst)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize("overwrite_x", [True, False])
def test_identity_1d_overwrite(forward, backward, type, dtype, axis, norm,
                               overwrite_x):
    # Test the identity f^-1(f(x)) == x
    x = np.random.rand(7, 8)
    x_orig = x.copy()

    y = forward(x, type, axis=axis, norm=norm, overwrite_x=overwrite_x)
    y_orig = y.copy()
    z = backward(y, type, axis=axis, norm=norm, overwrite_x=overwrite_x)
    if not overwrite_x:
        assert_allclose(z, x, rtol=1e-6, atol=1e-6)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)
    else:
        assert_allclose(z, x_orig, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("forward, backward", [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("shape, axes",
                         [
                             ((4, 4), 0),
                             ((4, 4), 1),
                             ((4, 4), None),
                             ((4, 4), (0, 1)),
                             ((10, 12), None),
                             ((10, 12), (0, 1)),
                             ((4, 5, 6), None),
                             ((4, 5, 6), 1),
                             ((4, 5, 6), (0, 2)),
                         ])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
def test_identity_nd(forward, backward, type, shape, axes, norm):
    # Test the identity f^-1(f(x)) == x

    x = np.random.random(shape)

    if axes is not None:
        shape = np.take(shape, axes)

    y = forward(x, type, axes=axes, norm=norm)
    z = backward(y, type, axes=axes, norm=norm)
    assert_allclose(z, x)

    if axes is None:
        pad = [(0, 4)] * x.ndim
    elif isinstance(axes, int):
        pad = [(0, 0)] * x.ndim
        pad[axes] = (0, 4)
    else:
        pad = [(0, 0)] * x.ndim

        for a in axes:
            pad[a] = (0, 4)

    y2 = np.pad(y, pad, mode='edge')
    z2 = backward(y2, type, shape, axes, norm)
    assert_allclose(z2, x)


@pytest.mark.parametrize("forward, backward", [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("shape, axes",
                         [
                             ((4, 5), 0),
                             ((4, 5), 1),
                             ((4, 5), None),
                         ])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize("overwrite_x", [False, True])
def test_identity_nd_overwrite(forward, backward, type, shape, axes, dtype,
                               norm, overwrite_x):
    # Test the identity f^-1(f(x)) == x

    x = np.random.random(shape).astype(dtype)
    x_orig = x.copy()

    if axes is not None:
        shape = np.take(shape, axes)

    y = forward(x, type, axes=axes, norm=norm)
    y_orig = y.copy()
    z = backward(y, type, axes=axes, norm=norm)
    if overwrite_x:
        assert_allclose(z, x_orig, rtol=1e-6, atol=1e-6)
    else:
        assert_allclose(z, x, rtol=1e-6, atol=1e-6)
        assert_array_equal(x, x_orig)
        assert_array_equal(y, y_orig)


@pytest.mark.parametrize("func", ['dct', 'dst', 'dctn', 'dstn'])
@pytest.mark.parametrize("type", [1, 2, 3, 4])
@pytest.mark.parametrize("norm", [None, 'backward', 'ortho', 'forward'])
def test_fftpack_equivalience(func, type, norm):
    x = np.random.rand(8, 16)
    fft_res = getattr(fft, func)(x, type, norm=norm)
    fftpack_res = getattr(fftpack, func)(x, type, norm=norm)

    assert_allclose(fft_res, fftpack_res)
