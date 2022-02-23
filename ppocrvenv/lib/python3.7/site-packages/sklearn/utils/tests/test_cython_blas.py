import pytest

import numpy as np

from sklearn.utils._testing import assert_allclose
from sklearn.utils._cython_blas import _dot_memview
from sklearn.utils._cython_blas import _asum_memview
from sklearn.utils._cython_blas import _axpy_memview
from sklearn.utils._cython_blas import _nrm2_memview
from sklearn.utils._cython_blas import _copy_memview
from sklearn.utils._cython_blas import _scal_memview
from sklearn.utils._cython_blas import _rotg_memview
from sklearn.utils._cython_blas import _rot_memview
from sklearn.utils._cython_blas import _gemv_memview
from sklearn.utils._cython_blas import _ger_memview
from sklearn.utils._cython_blas import _gemm_memview
from sklearn.utils._cython_blas import RowMajor, ColMajor
from sklearn.utils._cython_blas import Trans, NoTrans


def _numpy_to_cython(dtype):
    cython = pytest.importorskip("cython")
    if dtype == np.float32:
        return cython.float
    elif dtype == np.float64:
        return cython.double


RTOL = {np.float32: 1e-6, np.float64: 1e-12}
ORDER = {RowMajor: "C", ColMajor: "F"}


def _no_op(x):
    return x


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dot(dtype):
    dot = _dot_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(10).astype(dtype, copy=False)

    expected = x.dot(y)
    actual = dot(x, y)

    assert_allclose(actual, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_asum(dtype):
    asum = _asum_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)

    expected = np.abs(x).sum()
    actual = asum(x)

    assert_allclose(actual, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_axpy(dtype):
    axpy = _axpy_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(10).astype(dtype, copy=False)
    alpha = 2.5

    expected = alpha * x + y
    axpy(alpha, x, y)

    assert_allclose(y, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nrm2(dtype):
    nrm2 = _nrm2_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)

    expected = np.linalg.norm(x)
    actual = nrm2(x)

    assert_allclose(actual, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_copy(dtype):
    copy = _copy_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = np.empty_like(x)

    expected = x.copy()
    copy(x, y)

    assert_allclose(y, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scal(dtype):
    scal = _scal_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)
    alpha = 2.5

    expected = alpha * x
    scal(alpha, x)

    assert_allclose(x, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotg(dtype):
    rotg = _rotg_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    a = dtype(rng.randn())
    b = dtype(rng.randn())
    c, s = 0.0, 0.0

    def expected_rotg(a, b):
        roe = a if abs(a) > abs(b) else b
        if a == 0 and b == 0:
            c, s, r, z = (1, 0, 0, 0)
        else:
            r = np.sqrt(a ** 2 + b ** 2) * (1 if roe >= 0 else -1)
            c, s = a / r, b / r
            z = s if roe == a else (1 if c == 0 else 1 / c)
        return r, z, c, s

    expected = expected_rotg(a, b)
    actual = rotg(a, b, c, s)

    assert_allclose(actual, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rot(dtype):
    rot = _rot_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(10).astype(dtype, copy=False)
    c = dtype(rng.randn())
    s = dtype(rng.randn())

    expected_x = c * x + s * y
    expected_y = c * y - s * x

    rot(x, y, c, s)

    assert_allclose(x, expected_x)
    assert_allclose(y, expected_y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "opA, transA", [(_no_op, NoTrans), (np.transpose, Trans)], ids=["NoTrans", "Trans"]
)
@pytest.mark.parametrize("order", [RowMajor, ColMajor], ids=["RowMajor", "ColMajor"])
def test_gemv(dtype, opA, transA, order):
    gemv = _gemv_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    A = np.asarray(
        opA(rng.random_sample((20, 10)).astype(dtype, copy=False)), order=ORDER[order]
    )
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(20).astype(dtype, copy=False)
    alpha, beta = 2.5, -0.5

    expected = alpha * opA(A).dot(x) + beta * y
    gemv(transA, alpha, A, x, beta, y)

    assert_allclose(y, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", [RowMajor, ColMajor], ids=["RowMajor", "ColMajor"])
def test_ger(dtype, order):
    ger = _ger_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    x = rng.random_sample(10).astype(dtype, copy=False)
    y = rng.random_sample(20).astype(dtype, copy=False)
    A = np.asarray(
        rng.random_sample((10, 20)).astype(dtype, copy=False), order=ORDER[order]
    )
    alpha = 2.5

    expected = alpha * np.outer(x, y) + A
    ger(alpha, x, y, A)

    assert_allclose(A, expected, rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "opB, transB", [(_no_op, NoTrans), (np.transpose, Trans)], ids=["NoTrans", "Trans"]
)
@pytest.mark.parametrize(
    "opA, transA", [(_no_op, NoTrans), (np.transpose, Trans)], ids=["NoTrans", "Trans"]
)
@pytest.mark.parametrize("order", [RowMajor, ColMajor], ids=["RowMajor", "ColMajor"])
def test_gemm(dtype, opA, transA, opB, transB, order):
    gemm = _gemm_memview[_numpy_to_cython(dtype)]

    rng = np.random.RandomState(0)
    A = np.asarray(
        opA(rng.random_sample((30, 10)).astype(dtype, copy=False)), order=ORDER[order]
    )
    B = np.asarray(
        opB(rng.random_sample((10, 20)).astype(dtype, copy=False)), order=ORDER[order]
    )
    C = np.asarray(
        rng.random_sample((30, 20)).astype(dtype, copy=False), order=ORDER[order]
    )
    alpha, beta = 2.5, -0.5

    expected = alpha * opA(A).dot(opB(B)) + beta * C
    gemm(transA, transB, alpha, A, B, beta, C)

    assert_allclose(C, expected, rtol=RTOL[dtype])
