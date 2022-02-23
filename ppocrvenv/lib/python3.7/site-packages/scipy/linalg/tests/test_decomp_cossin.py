import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose

from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs

REAL_DTYPES = (np.float32, np.float64)
COMPLEX_DTYPES = (np.complex64, np.complex128)
DTYPES = REAL_DTYPES + COMPLEX_DTYPES


@pytest.mark.parametrize('dtype_', DTYPES)
@pytest.mark.parametrize('m, p, q',
                         [
                             (2, 1, 1),
                             (3, 2, 1),
                             (3, 1, 2),
                             (4, 2, 2),
                             (4, 1, 2),
                             (40, 12, 20),
                             (40, 30, 1),
                             (40, 1, 30),
                             (100, 50, 1),
                             (100, 50, 50),
                         ])
@pytest.mark.parametrize('swap_sign', [True, False])
def test_cossin(dtype_, m, p, q, swap_sign):
    seed(1234)
    if dtype_ in COMPLEX_DTYPES:
        x = np.array(unitary_group.rvs(m), dtype=dtype_)
    else:
        x = np.array(ortho_group.rvs(m), dtype=dtype_)

    u, cs, vh = cossin(x, p, q,
                       swap_sign=swap_sign)
    assert_allclose(x, u @ cs @ vh, rtol=0., atol=m*1e3*np.finfo(dtype_).eps)
    assert u.dtype == dtype_
    # Test for float32 or float 64
    assert cs.dtype == np.real(u).dtype
    assert vh.dtype == dtype_

    u, cs, vh = cossin([x[:p, :q], x[:p, q:], x[p:, :q], x[p:, q:]],
                       swap_sign=swap_sign)
    assert_allclose(x, u @ cs @ vh, rtol=0., atol=m*1e3*np.finfo(dtype_).eps)
    assert u.dtype == dtype_
    assert cs.dtype == np.real(u).dtype
    assert vh.dtype == dtype_

    _, cs2, vh2 = cossin(x, p, q,
                         compute_u=False,
                         swap_sign=swap_sign)
    assert_allclose(cs, cs2, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(vh, vh2, rtol=0., atol=10*np.finfo(dtype_).eps)

    u2, cs2, _ = cossin(x, p, q,
                        compute_vh=False,
                        swap_sign=swap_sign)
    assert_allclose(u, u2, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(cs, cs2, rtol=0., atol=10*np.finfo(dtype_).eps)

    _, cs2, _ = cossin(x, p, q,
                       compute_u=False,
                       compute_vh=False,
                       swap_sign=swap_sign)
    assert_allclose(cs, cs2, rtol=0., atol=10*np.finfo(dtype_).eps)


def test_cossin_mixed_types():
    seed(1234)
    x = np.array(ortho_group.rvs(4), dtype=np.float64)
    u, cs, vh = cossin([x[:2, :2],
                        np.array(x[:2, 2:], dtype=np.complex128),
                        x[2:, :2],
                        x[2:, 2:]])

    assert u.dtype == np.complex128
    assert cs.dtype == np.float64
    assert vh.dtype == np.complex128
    assert_allclose(x, u @ cs @ vh, rtol=0.,
                    atol=1e4 * np.finfo(np.complex128).eps)


def test_cossin_error_incorrect_subblocks():
    with pytest.raises(ValueError, match="be due to missing p, q arguments."):
        cossin(([1, 2], [3, 4, 5], [6, 7], [8, 9, 10]))


def test_cossin_error_empty_subblocks():
    with pytest.raises(ValueError, match="x11.*empty"):
        cossin(([], [], [], []))
    with pytest.raises(ValueError, match="x12.*empty"):
        cossin(([1, 2], [], [6, 7], [8, 9, 10]))
    with pytest.raises(ValueError, match="x21.*empty"):
        cossin(([1, 2], [3, 4, 5], [], [8, 9, 10]))
    with pytest.raises(ValueError, match="x22.*empty"):
        cossin(([1, 2], [3, 4, 5], [2], []))


def test_cossin_error_missing_partitioning():
    with pytest.raises(ValueError, match=".*exactly four arrays.* got 2"):
        cossin(unitary_group.rvs(2))

    with pytest.raises(ValueError, match=".*might be due to missing p, q"):
        cossin(unitary_group.rvs(4))


def test_cossin_error_non_iterable():
    with pytest.raises(ValueError, match="containing the subblocks of X"):
        cossin(12j)


def test_cossin_error_non_square():
    with pytest.raises(ValueError, match="only supports square"):
        cossin(np.array([[1, 2]]), 1, 1)

def test_cossin_error_partitioning():
    x = np.array(ortho_group.rvs(4), dtype=np.float64)
    with pytest.raises(ValueError, match="invalid p=0.*0<p<4.*"):
        cossin(x, 0, 1)
    with pytest.raises(ValueError, match="invalid p=4.*0<p<4.*"):
        cossin(x, 4, 1)
    with pytest.raises(ValueError, match="invalid q=-2.*0<q<4.*"):
        cossin(x, 1, -2)
    with pytest.raises(ValueError, match="invalid q=5.*0<q<4.*"):
        cossin(x, 1, 5)


@pytest.mark.parametrize("dtype_", DTYPES)
def test_cossin_separate(dtype_):
    seed(1234)
    m, p, q = 250, 80, 170

    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'
    X = ortho_group.rvs(m) if pfx == 'or' else unitary_group.rvs(m)
    X = np.array(X, dtype=dtype_)

    drv, dlw = get_lapack_funcs((pfx + 'csd', pfx + 'csd_lwork'),[X])
    lwval = _compute_lwork(dlw, m, p, q)
    lwvals = {'lwork': lwval} if pfx == 'or' else dict(zip(['lwork',
                                                            'lrwork'],
                                                           lwval))

    *_, theta, u1, u2, v1t, v2t, _ = \
        drv(X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:], **lwvals)

    (u1_2, u2_2), theta2, (v1t_2, v2t_2) = cossin(X, p, q, separate=True)

    assert_allclose(u1_2, u1, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(u2_2, u2, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(v1t_2, v1t, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(v2t_2, v2t, rtol=0., atol=10*np.finfo(dtype_).eps)
    assert_allclose(theta2, theta, rtol=0., atol=10*np.finfo(dtype_).eps)
