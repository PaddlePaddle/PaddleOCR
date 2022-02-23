import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose, assert_equal)
from scipy.linalg import polar, eigh


diag2 = np.array([[2, 0], [0, 3]])
a13 = np.array([[1, 2, 2]])

precomputed_cases = [
    [[[0]], 'right', [[1]], [[0]]],
    [[[0]], 'left', [[1]], [[0]]],
    [[[9]], 'right', [[1]], [[9]]],
    [[[9]], 'left', [[1]], [[9]]],
    [diag2, 'right', np.eye(2), diag2],
    [diag2, 'left', np.eye(2), diag2],
    [a13, 'right', a13/norm(a13[0]), a13.T.dot(a13)/norm(a13[0])],
]

verify_cases = [
    [[1, 2], [3, 4]],
    [[1, 2, 3]],
    [[1], [2], [3]],
    [[1, 2, 3], [3, 4, 0]],
    [[1, 2], [3, 4], [5, 5]],
    [[1, 2], [3, 4+5j]],
    [[1, 2, 3j]],
    [[1], [2], [3j]],
    [[1, 2, 3+2j], [3, 4-1j, -4j]],
    [[1, 2], [3-2j, 4+0.5j], [5, 5]],
    [[10000, 10, 1], [-1, 2, 3j], [0, 1, 2]],
]


def check_precomputed_polar(a, side, expected_u, expected_p):
    # Compare the result of the polar decomposition to a
    # precomputed result.
    u, p = polar(a, side=side)
    assert_allclose(u, expected_u, atol=1e-15)
    assert_allclose(p, expected_p, atol=1e-15)


def verify_polar(a):
    # Compute the polar decomposition, and then verify that
    # the result has all the expected properties.
    product_atol = np.sqrt(np.finfo(float).eps)

    aa = np.asarray(a)
    m, n = aa.shape

    u, p = polar(a, side='right')
    assert_equal(u.shape, (m, n))
    assert_equal(p.shape, (n, n))
    # a = up
    assert_allclose(u.dot(p), a, atol=product_atol)
    if m >= n:
        assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)
    else:
        assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)
    # p is Hermitian positive semidefinite.
    assert_allclose(p.conj().T, p)
    evals = eigh(p, eigvals_only=True)
    nonzero_evals = evals[abs(evals) > 1e-14]
    assert_((nonzero_evals >= 0).all())

    u, p = polar(a, side='left')
    assert_equal(u.shape, (m, n))
    assert_equal(p.shape, (m, m))
    # a = pu
    assert_allclose(p.dot(u), a, atol=product_atol)
    if m >= n:
        assert_allclose(u.conj().T.dot(u), np.eye(n), atol=1e-15)
    else:
        assert_allclose(u.dot(u.conj().T), np.eye(m), atol=1e-15)
    # p is Hermitian positive semidefinite.
    assert_allclose(p.conj().T, p)
    evals = eigh(p, eigvals_only=True)
    nonzero_evals = evals[abs(evals) > 1e-14]
    assert_((nonzero_evals >= 0).all())


def test_precomputed_cases():
    for a, side, expected_u, expected_p in precomputed_cases:
        check_precomputed_polar(a, side, expected_u, expected_p)


def test_verify_cases():
    for a in verify_cases:
        verify_polar(a)

