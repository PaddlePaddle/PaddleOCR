"""Test functions for linalg._solve_toeplitz module
"""
import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose

import pytest
from pytest import raises as assert_raises


def test_solve_equivalence():
    # For toeplitz matrices, solve_toeplitz() should be equivalent to solve().
    random = np.random.RandomState(1234)
    for n in (1, 2, 3, 10):
        c = random.randn(n)
        if random.rand() < 0.5:
            c = c + 1j * random.randn(n)
        r = random.randn(n)
        if random.rand() < 0.5:
            r = r + 1j * random.randn(n)
        y = random.randn(n)
        if random.rand() < 0.5:
            y = y + 1j * random.randn(n)

        # Check equivalence when both the column and row are provided.
        actual = solve_toeplitz((c,r), y)
        desired = solve(toeplitz(c, r=r), y)
        assert_allclose(actual, desired)

        # Check equivalence when the column is provided but not the row.
        actual = solve_toeplitz(c, b=y)
        desired = solve(toeplitz(c), y)
        assert_allclose(actual, desired)


def test_multiple_rhs():
    random = np.random.RandomState(1234)
    c = random.randn(4)
    r = random.randn(4)
    for offset in [0, 1j]:
        for yshape in ((4,), (4, 3), (4, 3, 2)):
            y = random.randn(*yshape) + offset
            actual = solve_toeplitz((c,r), b=y)
            desired = solve(toeplitz(c, r=r), y)
            assert_equal(actual.shape, yshape)
            assert_equal(desired.shape, yshape)
            assert_allclose(actual, desired)
            
            
def test_native_list_arguments():
    c = [1,2,4,7]
    r = [1,3,9,12]
    y = [5,1,4,2]
    actual = solve_toeplitz((c,r), y)
    desired = solve(toeplitz(c, r=r), y)
    assert_allclose(actual, desired)


def test_zero_diag_error():
    # The Levinson-Durbin implementation fails when the diagonal is zero.
    random = np.random.RandomState(1234)
    n = 4
    c = random.randn(n)
    r = random.randn(n)
    y = random.randn(n)
    c[0] = 0
    assert_raises(np.linalg.LinAlgError,
        solve_toeplitz, (c, r), b=y)


def test_wikipedia_counterexample():
    # The Levinson-Durbin implementation also fails in other cases.
    # This example is from the talk page of the wikipedia article.
    random = np.random.RandomState(1234)
    c = [2, 2, 1]
    y = random.randn(3)
    assert_raises(np.linalg.LinAlgError, solve_toeplitz, c, b=y)


def test_reflection_coeffs():
    # check that that the partial solutions are given by the reflection
    # coefficients

    random = np.random.RandomState(1234)
    y_d = random.randn(10)
    y_z = random.randn(10) + 1j
    reflection_coeffs_d = [1]
    reflection_coeffs_z = [1]
    for i in range(2, 10):
        reflection_coeffs_d.append(solve_toeplitz(y_d[:(i-1)], b=y_d[1:i])[-1])
        reflection_coeffs_z.append(solve_toeplitz(y_z[:(i-1)], b=y_z[1:i])[-1])

    y_d_concat = np.concatenate((y_d[-2:0:-1], y_d[:-1]))
    y_z_concat = np.concatenate((y_z[-2:0:-1].conj(), y_z[:-1]))
    _, ref_d = levinson(y_d_concat, b=y_d[1:])
    _, ref_z = levinson(y_z_concat, b=y_z[1:])

    assert_allclose(reflection_coeffs_d, ref_d[:-1])
    assert_allclose(reflection_coeffs_z, ref_z[:-1])


@pytest.mark.xfail(reason='Instability of Levinson iteration')
def test_unstable():
    # this is a "Gaussian Toeplitz matrix", as mentioned in Example 2 of
    # I. Gohbert, T. Kailath and V. Olshevsky "Fast Gaussian Elimination with
    # Partial Pivoting for Matrices with Displacement Structure"
    # Mathematics of Computation, 64, 212 (1995), pp 1557-1576
    # which can be unstable for levinson recursion.

    # other fast toeplitz solvers such as GKO or Burg should be better.
    random = np.random.RandomState(1234)
    n = 100
    c = 0.9 ** (np.arange(n)**2)
    y = random.randn(n)

    solution1 = solve_toeplitz(c, b=y)
    solution2 = solve(toeplitz(c), y)

    assert_allclose(solution1, solution2)

