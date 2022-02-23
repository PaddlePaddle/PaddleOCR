import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy.sparse.linalg.isolve import minres
from scipy.linalg import norm

from pytest import raises as assert_raises
from .test_iterative import assert_normclose


def get_sample_problem():
    # A random 10 x 10 symmetric matrix
    np.random.seed(1234)
    matrix = np.random.rand(10, 10)
    matrix = matrix + matrix.T
    # A random vector of length 10
    vector = np.random.rand(10)
    return matrix, vector


def test_singular():
    A, b = get_sample_problem()
    A[0, ] = 0
    b[0] = 0
    xp, info = minres(A, b)
    assert_equal(info, 0)
    assert_normclose(A.dot(xp), b, tol=1e-5)


def test_x0_is_used_by():
    A, b = get_sample_problem()
    # Random x0 to feed minres
    np.random.seed(12345)
    x0 = np.random.rand(10)
    trace = []

    def trace_iterates(xk):
        trace.append(xk)
    minres(A, b, x0=x0, callback=trace_iterates)
    trace_with_x0 = trace

    trace = []
    minres(A, b, callback=trace_iterates)
    assert_(not np.array_equal(trace_with_x0[0], trace[0]))


def test_shift():
    A, b = get_sample_problem()
    shift = 0.5
    shifted_A = A - shift * np.eye(10)
    x1, info1 = minres(A, b, shift=shift)
    x2, info2 = minres(shifted_A, b)
    assert_equal(info1, 0)
    assert_allclose(x1, x2, rtol=1e-5)


def test_asymmetric_fail():
    """Asymmetric matrix should raise `ValueError` when check=True"""
    A, b = get_sample_problem()
    A[1, 2] = 1
    A[2, 1] = 2
    with assert_raises(ValueError):
        xp, info = minres(A, b, check=True)


def test_minres_non_default_x0():
    np.random.seed(1234)
    tol = 10**(-6)
    a = np.random.randn(5, 5)
    a = np.dot(a, a.T)
    b = np.random.randn(5)
    c = np.random.randn(5)
    x = minres(a, b, x0=c, tol=tol)[0]
    assert norm(a.dot(x) - b) < tol


def test_minres_precond_non_default_x0():
    np.random.seed(12345)
    tol = 10**(-6)
    a = np.random.randn(5, 5)
    a = np.dot(a, a.T)
    b = np.random.randn(5)
    c = np.random.randn(5)
    m = np.random.randn(5, 5)
    m = np.dot(m, m.T)
    x = minres(a, b, M=m, x0=c, tol=tol)[0]
    assert norm(a.dot(x) - b) < tol


def test_minres_precond_exact_x0():
    np.random.seed(1234)
    tol = 10**(-6)
    a = np.eye(10)
    b = np.ones(10)
    c = np.ones(10)
    m = np.random.randn(10, 10)
    m = np.dot(m, m.T)
    x = minres(a, b, M=m, x0=c, tol=tol)[0]
    assert norm(a.dot(x) - b) < tol
