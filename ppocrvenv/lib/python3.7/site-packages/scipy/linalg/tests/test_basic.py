import itertools
import warnings

import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
                   float32)
import numpy.linalg as linalg
from numpy.random import random

from numpy.testing import (assert_equal, assert_almost_equal, assert_,
                           assert_array_almost_equal, assert_allclose,
                           assert_array_equal, suppress_warnings)
import pytest
from pytest import raises as assert_raises

from scipy.linalg import (solve, inv, det, lstsq, pinv, pinv2, pinvh, norm,
                          solve_banded, solveh_banded, solve_triangular,
                          solve_circulant, circulant, LinAlgError, block_diag,
                          matrix_balance, qr, LinAlgWarning)

from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64

REAL_DTYPES = (np.float32, np.float64, np.longdouble)
COMPLEX_DTYPES = (np.complex64, np.complex128, np.clongdouble)
DTYPES = REAL_DTYPES + COMPLEX_DTYPES


def _eps_cast(dtyp):
    """Get the epsilon for dtype, possibly downcast to BLAS types."""
    dt = dtyp
    if dt == np.longdouble:
        dt = np.float64
    elif dt == np.clongdouble:
        dt = np.complex128
    return np.finfo(dt).eps


class TestSolveBanded:

    def test_real(self):
        a = array([[1.0, 20, 0, 0],
                   [-30, 4, 6, 0],
                   [2, 1, 20, 2],
                   [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2],
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2, -1, 0, 0]])
        l, u = 2, 1
        b4 = array([10.0, 0.0, 2.0, 14.0])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1],
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((l, u), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_complex(self):
        a = array([[1.0, 20, 0, 0],
                   [-30, 4, 6, 0],
                   [2j, 1, 20, 2j],
                   [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2j],
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2j, -1, 0, 0]])
        l, u = 2, 1
        b4 = array([10.0, 0.0, 2.0, 14.0j])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1],
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],
                       [0, 0, 0, 1j],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((l, u), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_tridiag_real(self):
        ab = array([[0.0, 20, 6, 2],
                   [1, 4, 20, 14],
                   [-30, 1, 7, 0]])
        a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(
                                                                ab[2, :-1], -1)
        b4 = array([10.0, 0.0, 2.0, 14.0])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1],
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((1, 1), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_tridiag_complex(self):
        ab = array([[0.0, 20, 6, 2j],
                   [1, 4, 20, 14],
                   [-30, 1, 7, 0]])
        a = np.diag(ab[0, 1:], 1) + np.diag(ab[1, :], 0) + np.diag(
                                                               ab[2, :-1], -1)
        b4 = array([10.0, 0.0, 2.0, 14.0j])
        b4by1 = b4.reshape(-1, 1)
        b4by2 = array([[2, 1],
                       [-30, 4],
                       [2, 3],
                       [1, 3]])
        b4by4 = array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0],
                       [0, 1, 0, 0]])
        for b in [b4, b4by1, b4by2, b4by4]:
            x = solve_banded((1, 1), ab, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_check_finite(self):
        a = array([[1.0, 20, 0, 0],
                   [-30, 4, 6, 0],
                   [2, 1, 20, 2],
                   [0, -1, 7, 14]])
        ab = array([[0.0, 20, 6, 2],
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2, -1, 0, 0]])
        l, u = 2, 1
        b4 = array([10.0, 0.0, 2.0, 14.0])
        x = solve_banded((l, u), ab, b4, check_finite=False)
        assert_array_almost_equal(dot(a, x), b4)

    def test_bad_shape(self):
        ab = array([[0.0, 20, 6, 2],
                    [1, 4, 20, 14],
                    [-30, 1, 7, 0],
                    [2, -1, 0, 0]])
        l, u = 2, 1
        bad = array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 4)
        assert_raises(ValueError, solve_banded, (l, u), ab, bad)
        assert_raises(ValueError, solve_banded, (l, u), ab, [1.0, 2.0])

        # Values of (l,u) are not compatible with ab.
        assert_raises(ValueError, solve_banded, (1, 1), ab, [1.0, 2.0])

    def test_1x1(self):
        b = array([[1., 2., 3.]])
        x = solve_banded((1, 1), [[0], [2], [0]], b)
        assert_array_equal(x, [[0.5, 1.0, 1.5]])
        assert_equal(x.dtype, np.dtype('f8'))
        assert_array_equal(b, [[1.0, 2.0, 3.0]])

    def test_native_list_arguments(self):
        a = [[1.0, 20, 0, 0],
             [-30, 4, 6, 0],
             [2, 1, 20, 2],
             [0, -1, 7, 14]]
        ab = [[0.0, 20, 6, 2],
              [1, 4, 20, 14],
              [-30, 1, 7, 0],
              [2, -1, 0, 0]]
        l, u = 2, 1
        b = [10.0, 0.0, 2.0, 14.0]
        x = solve_banded((l, u), ab, b)
        assert_array_almost_equal(dot(a, x), b)


class TestSolveHBanded:

    def test_01_upper(self):
        # Solve
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        # with the RHS as a 1D array.
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0, 2.0])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_upper(self):
        # Solve
        # [ 4 1 2 0]     [1 6]
        # [ 1 4 1 2] X = [4 2]
        # [ 2 1 4 1]     [1 6]
        # [ 0 2 1 4]     [2 1]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([[1.0, 6.0],
                   [4.0, 2.0],
                   [1.0, 6.0],
                   [2.0, 1.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_03_upper(self):
        # Solve
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        # with the RHS as a 2D array with shape (3,1).
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0, 2.0]).reshape(-1, 1)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, array([0., 1., 0., 0.]).reshape(-1, 1))

    def test_01_lower(self):
        # Solve
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        #
        ab = array([[4.0, 4.0, 4.0, 4.0],
                    [1.0, 1.0, 1.0, -99],
                    [2.0, 2.0, 0.0, 0.0]])
        b = array([1.0, 4.0, 1.0, 2.0])
        x = solveh_banded(ab, b, lower=True)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_lower(self):
        # Solve
        # [ 4 1 2 0]     [1 6]
        # [ 1 4 1 2] X = [4 2]
        # [ 2 1 4 1]     [1 6]
        # [ 0 2 1 4]     [2 1]
        #
        ab = array([[4.0, 4.0, 4.0, 4.0],
                    [1.0, 1.0, 1.0, -99],
                    [2.0, 2.0, 0.0, 0.0]])
        b = array([[1.0, 6.0],
                   [4.0, 2.0],
                   [1.0, 6.0],
                   [2.0, 1.0]])
        x = solveh_banded(ab, b, lower=True)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_01_float32(self):
        # Solve
        # [ 4 1 2 0]     [1]
        # [ 1 4 1 2] X = [4]
        # [ 2 1 4 1]     [1]
        # [ 0 2 1 4]     [2]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]], dtype=float32)
        b = array([1.0, 4.0, 1.0, 2.0], dtype=float32)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])

    def test_02_float32(self):
        # Solve
        # [ 4 1 2 0]     [1 6]
        # [ 1 4 1 2] X = [4 2]
        # [ 2 1 4 1]     [1 6]
        # [ 0 2 1 4]     [2 1]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, 1.0, 1.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0]], dtype=float32)
        b = array([[1.0, 6.0],
                   [4.0, 2.0],
                   [1.0, 6.0],
                   [2.0, 1.0]], dtype=float32)
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0],
                          [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_01_complex(self):
        # Solve
        # [ 4 -j  2  0]     [2-j]
        # [ j  4 -j  2] X = [4-j]
        # [ 2  j  4 -j]     [4+j]
        # [ 0  2  j  4]     [2+j]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, -1.0j, -1.0j, -1.0j],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([2-1.0j, 4.0-1j, 4+1j, 2+1j])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 1.0, 0.0])

    def test_02_complex(self):
        # Solve
        # [ 4 -j  2  0]     [2-j 2+4j]
        # [ j  4 -j  2] X = [4-j -1-j]
        # [ 2  j  4 -j]     [4+j 4+2j]
        # [ 0  2  j  4]     [2+j j]
        #
        ab = array([[0.0, 0.0, 2.0, 2.0],
                    [-99, -1.0j, -1.0j, -1.0j],
                    [4.0, 4.0, 4.0, 4.0]])
        b = array([[2-1j, 2+4j],
                   [4.0-1j, -1-1j],
                   [4.0+1j, 4+2j],
                   [2+1j, 1j]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0j],
                          [1.0, 0.0],
                          [1.0, 1.0],
                          [0.0, 0.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_upper(self):
        # Solve
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        # with the RHS as a 1D array.
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_upper(self):
        # Solve
        # [ 4 1 0]     [1 4]
        # [ 1 4 1] X = [4 2]
        # [ 0 1 4]     [1 4]
        #
        ab = array([[-99, 1.0, 1.0],
                    [4.0, 4.0, 4.0]])
        b = array([[1.0, 4.0],
                   [4.0, 2.0],
                   [1.0, 4.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_03_upper(self):
        # Solve
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        # with the RHS as a 2D array with shape (3,1).
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0]).reshape(-1, 1)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, array([0.0, 1.0, 0.0]).reshape(-1, 1))

    def test_tridiag_01_lower(self):
        # Solve
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        #
        ab = array([[4.0, 4.0, 4.0],
                    [1.0, 1.0, -99]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b, lower=True)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_lower(self):
        # Solve
        # [ 4 1 0]     [1 4]
        # [ 1 4 1] X = [4 2]
        # [ 0 1 4]     [1 4]
        #
        ab = array([[4.0, 4.0, 4.0],
                    [1.0, 1.0, -99]])
        b = array([[1.0, 4.0],
                   [4.0, 2.0],
                   [1.0, 4.0]])
        x = solveh_banded(ab, b, lower=True)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_float32(self):
        # Solve
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        #
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]], dtype=float32)
        b = array([1.0, 4.0, 1.0], dtype=float32)
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_tridiag_02_float32(self):
        # Solve
        # [ 4 1 0]     [1 4]
        # [ 1 4 1] X = [4 2]
        # [ 0 1 4]     [1 4]
        #
        ab = array([[-99, 1.0, 1.0],
                    [4.0, 4.0, 4.0]], dtype=float32)
        b = array([[1.0, 4.0],
                   [4.0, 2.0],
                   [1.0, 4.0]], dtype=float32)
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0],
                          [1.0, 0.0],
                          [0.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_tridiag_01_complex(self):
        # Solve
        # [ 4 -j 0]     [ -j]
        # [ j 4 -j] X = [4-j]
        # [ 0 j  4]     [4+j]
        #
        ab = array([[-99, -1.0j, -1.0j], [4.0, 4.0, 4.0]])
        b = array([-1.0j, 4.0-1j, 4+1j])
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 1.0])

    def test_tridiag_02_complex(self):
        # Solve
        # [ 4 -j 0]     [ -j    4j]
        # [ j 4 -j] X = [4-j  -1-j]
        # [ 0 j  4]     [4+j   4  ]
        #
        ab = array([[-99, -1.0j, -1.0j],
                    [4.0, 4.0, 4.0]])
        b = array([[-1j, 4.0j],
                   [4.0-1j, -1.0-1j],
                   [4.0+1j, 4.0]])
        x = solveh_banded(ab, b)
        expected = array([[0.0, 1.0j],
                          [1.0, 0.0],
                          [1.0, 1.0]])
        assert_array_almost_equal(x, expected)

    def test_check_finite(self):
        # Solve
        # [ 4 1 0]     [1]
        # [ 1 4 1] X = [4]
        # [ 0 1 4]     [1]
        # with the RHS as a 1D array.
        ab = array([[-99, 1.0, 1.0], [4.0, 4.0, 4.0]])
        b = array([1.0, 4.0, 1.0])
        x = solveh_banded(ab, b, check_finite=False)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0])

    def test_bad_shapes(self):
        ab = array([[-99, 1.0, 1.0],
                    [4.0, 4.0, 4.0]])
        b = array([[1.0, 4.0],
                   [4.0, 2.0]])
        assert_raises(ValueError, solveh_banded, ab, b)
        assert_raises(ValueError, solveh_banded, ab, [1.0, 2.0])
        assert_raises(ValueError, solveh_banded, ab, [1.0])

    def test_1x1(self):
        x = solveh_banded([[1]], [[1, 2, 3]])
        assert_array_equal(x, [[1.0, 2.0, 3.0]])
        assert_equal(x.dtype, np.dtype('f8'))

    def test_native_list_arguments(self):
        # Same as test_01_upper, using python's native list.
        ab = [[0.0, 0.0, 2.0, 2.0],
              [-99, 1.0, 1.0, 1.0],
              [4.0, 4.0, 4.0, 4.0]]
        b = [1.0, 4.0, 1.0, 2.0]
        x = solveh_banded(ab, b)
        assert_array_almost_equal(x, [0.0, 1.0, 0.0, 0.0])


class TestSolve:
    def setup_method(self):
        np.random.seed(1234)

    def test_20Feb04_bug(self):
        a = [[1, 1], [1.0, 0]]  # ok
        x0 = solve(a, [1, 0j])
        assert_array_almost_equal(dot(a, x0), [1, 0])

        # gives failure with clapack.zgesv(..,rowmajor=0)
        a = [[1, 1], [1.2, 0]]
        b = [1, 0j]
        x0 = solve(a, b)
        assert_array_almost_equal(dot(a, x0), [1, 0])

    def test_simple(self):
        a = [[1, 20], [-30, 4]]
        for b in ([[1, 0], [0, 1]],
                  [1, 0],
                  [[2, 1], [-30, 4]]
                  ):
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_complex(self):
        a = array([[5, 2], [2j, 4]], 'D')
        for b in ([1j, 0],
                  [[1j, 1j], [0, 2]],
                  [1, 0j],
                  array([1, 0], 'D'),
                  ):
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_pos(self):
        a = [[2, 3], [3, 5]]
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]],
                      [1, 0]
                      ):
                x = solve(a, b, assume_a='pos', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    def test_simple_pos_complexb(self):
        a = [[5, 2], [2, 4]]
        for b in ([1j, 0],
                  [[1j, 1j], [0, 2]],
                  ):
            x = solve(a, b, assume_a='pos')
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_sym(self):
        a = [[2, 3], [3, -5]]
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]],
                      [1, 0]
                      ):
                x = solve(a, b, assume_a='sym', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

    def test_simple_sym_complexb(self):
        a = [[5, 2], [2, -4]]
        for b in ([1j, 0],
                  [[1j, 1j],[0, 2]]
                  ):
            x = solve(a, b, assume_a='sym')
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_sym_complex(self):
        a = [[5, 2+1j], [2+1j, -4]]
        for b in ([1j, 0],
                  [1, 0],
                  [[1j, 1j], [0, 2]]
                  ):
            x = solve(a, b, assume_a='sym')
            assert_array_almost_equal(dot(a, x), b)

    def test_simple_her_actuallysym(self):
        a = [[2, 3], [3, -5]]
        for lower in [0, 1]:
            for b in ([[1, 0], [0, 1]],
                      [1, 0],
                      [1j, 0],
                      ):
                x = solve(a, b, assume_a='her', lower=lower)
                assert_array_almost_equal(dot(a, x), b)

            
    def test_simple_her(self):
        a = [[5, 2+1j], [2-1j, -4]]
        for b in ([1j, 0],
                  [1, 0],
                  [[1j, 1j], [0, 2]]
                  ):
            x = solve(a, b, assume_a='her')
            assert_array_almost_equal(dot(a, x), b)

            

    def test_nils_20Feb04(self):
        n = 2
        A = random([n, n])+random([n, n])*1j
        X = zeros((n, n), 'D')
        Ainv = inv(A)
        R = identity(n)+identity(n)*0j
        for i in arange(0, n):
            r = R[:, i]
            X[:, i] = solve(A, r)
        assert_array_almost_equal(X, Ainv)

    def test_random(self):

        n = 20
        a = random([n, n])
        for i in range(n):
            a[i, i] = 20*(.1+a[i, i])
        for i in range(4):
            b = random([n, 3])
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_random_complex(self):
        n = 20
        a = random([n, n]) + 1j * random([n, n])
        for i in range(n):
            a[i, i] = 20*(.1+a[i, i])
        for i in range(2):
            b = random([n, 3])
            x = solve(a, b)
            assert_array_almost_equal(dot(a, x), b)

    def test_random_sym(self):
        n = 20
        a = random([n, n])
        for i in range(n):
            a[i, i] = abs(20*(.1+a[i, i]))
            for j in range(i):
                a[i, j] = a[j, i]
        for i in range(4):
            b = random([n])
            x = solve(a, b, sym_pos=1)
            assert_array_almost_equal(dot(a, x), b)

    def test_random_sym_complex(self):
        n = 20
        a = random([n, n])
        a = a + 1j*random([n, n])
        for i in range(n):
            a[i, i] = abs(20*(.1+a[i, i]))
            for j in range(i):
                a[i, j] = conjugate(a[j, i])
        b = random([n])+2j*random([n])
        for i in range(2):
            x = solve(a, b, sym_pos=1)
            assert_array_almost_equal(dot(a, x), b)

    def test_check_finite(self):
        a = [[1, 20], [-30, 4]]
        for b in ([[1, 0], [0, 1]], [1, 0],
                  [[2, 1], [-30, 4]]):
            x = solve(a, b, check_finite=False)
            assert_array_almost_equal(dot(a, x), b)

    def test_scalar_a_and_1D_b(self):
        a = 1
        b = [1, 2, 3]
        x = solve(a, b)
        assert_array_almost_equal(x.ravel(), b)
        assert_(x.shape == (3,), 'Scalar_a_1D_b test returned wrong shape')

    def test_simple2(self):
        a = np.array([[1.80, 2.88, 2.05, -0.89],
                      [525.00, -295.00, -95.00, -380.00],
                      [1.58, -2.69, -2.90, -1.04],
                      [-1.11, -0.66, -0.59, 0.80]])

        b = np.array([[9.52, 18.47],
                      [2435.00, 225.00],
                      [0.77, -13.28],
                      [-6.22, -6.21]])

        x = solve(a, b)
        assert_array_almost_equal(x, np.array([[1., -1, 3, -5],
                                               [3, 2, 4, 1]]).T)

    def test_simple_complex2(self):
        a = np.array([[-1.34+2.55j, 0.28+3.17j, -6.39-2.20j, 0.72-0.92j],
                      [-1.70-14.10j, 33.10-1.50j, -1.50+13.40j, 12.90+13.80j],
                      [-3.29-2.39j, -1.91+4.42j, -0.14-1.35j, 1.72+1.35j],
                      [2.41+0.39j, -0.56+1.47j, -0.83-0.69j, -1.96+0.67j]])

        b = np.array([[26.26+51.78j, 31.32-6.70j],
                      [64.30-86.80j, 158.60-14.20j],
                      [-5.75+25.31j, -2.15+30.19j],
                      [1.16+2.57j, -2.56+7.55j]])

        x = solve(a, b)
        assert_array_almost_equal(x, np. array([[1+1.j, -1-2.j],
                                                [2-3.j, 5+1.j],
                                                [-4-5.j, -3+4.j],
                                                [6.j, 2-3.j]]))

    def test_hermitian(self):
        # An upper triangular matrix will be used for hermitian matrix a
        a = np.array([[-1.84, 0.11-0.11j, -1.78-1.18j, 3.91-1.50j],
                      [0, -4.63, -1.84+0.03j, 2.21+0.21j],
                      [0, 0, -8.87, 1.58-0.90j],
                      [0, 0, 0, -1.36]])
        b = np.array([[2.98-10.18j, 28.68-39.89j],
                      [-9.58+3.88j, -24.79-8.40j],
                      [-0.77-16.05j, 4.23-70.02j],
                      [7.79+5.48j, -35.39+18.01j]])
        res = np.array([[2.+1j, -8+6j],
                        [3.-2j, 7-2j],
                        [-1+2j, -1+5j],
                        [1.-1j, 3-4j]])
        x = solve(a, b, assume_a='her')
        assert_array_almost_equal(x, res)
        # Also conjugate a and test for lower triangular data
        x = solve(a.conj().T, b, assume_a='her', lower=True)
        assert_array_almost_equal(x, res)

    def test_pos_and_sym(self):
        A = np.arange(1, 10).reshape(3, 3)
        x = solve(np.tril(A)/9, np.ones(3), assume_a='pos')
        assert_array_almost_equal(x, [9., 1.8, 1.])
        x = solve(np.tril(A)/9, np.ones(3), assume_a='sym')
        assert_array_almost_equal(x, [9., 1.8, 1.])

    def test_singularity(self):
        a = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 1],
                      [1, 1, 1, 0, 0, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0, 0, 1, 0, 1],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        b = np.arange(9)[:, None]
        assert_raises(LinAlgError, solve, a, b)

    def test_ill_condition_warning(self):
        a = np.array([[1, 1], [1+1e-16, 1-1e-16]])
        b = np.ones(2)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert_raises(LinAlgWarning, solve, a, b)

    def test_empty_rhs(self):
        a = np.eye(2)
        b = [[], []]
        x = solve(a, b)
        assert_(x.size == 0, 'Returned array is not empty')
        assert_(x.shape == (2, 0), 'Returned empty array shape is wrong')

    def test_multiple_rhs(self):
        a = np.eye(2)
        b = np.random.rand(2, 3, 4)
        x = solve(a, b)
        assert_array_almost_equal(x, b)

    def test_transposed_keyword(self):
        A = np.arange(9).reshape(3, 3) + 1
        x = solve(np.tril(A)/9, np.ones(3), transposed=True)
        assert_array_almost_equal(x, [1.2, 0.2, 1])
        x = solve(np.tril(A)/9, np.ones(3), transposed=False)
        assert_array_almost_equal(x, [9, -5.4, -1.2])

    def test_transposed_notimplemented(self):
        a = np.eye(3).astype(complex)
        with assert_raises(NotImplementedError):
            solve(a, a, transposed=True)

    def test_nonsquare_a(self):
        assert_raises(ValueError, solve, [1, 2], 1)

    def test_size_mismatch_with_1D_b(self):
        assert_array_almost_equal(solve(np.eye(3), np.ones(3)), np.ones(3))
        assert_raises(ValueError, solve, np.eye(3), np.ones(4))

    def test_assume_a_keyword(self):
        assert_raises(ValueError, solve, 1, 1, assume_a='zxcv')

    @pytest.mark.skip(reason="Failure on OS X (gh-7500), "
                             "crash on Windows (gh-8064)")
    def test_all_type_size_routine_combinations(self):
        sizes = [10, 100]
        assume_as = ['gen', 'sym', 'pos', 'her']
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        for size, assume_a, dtype in itertools.product(sizes, assume_as,
                                                       dtypes):
            is_complex = dtype in (np.complex64, np.complex128)
            if assume_a == 'her' and not is_complex:
                continue

            err_msg = ("Failed for size: {}, assume_a: {},"
                       "dtype: {}".format(size, assume_a, dtype))

            a = np.random.randn(size, size).astype(dtype)
            b = np.random.randn(size).astype(dtype)
            if is_complex:
                a = a + (1j*np.random.randn(size, size)).astype(dtype)

            if assume_a == 'sym':  # Can still be complex but only symmetric
                a = a + a.T
            elif assume_a == 'her':  # Handle hermitian matrices here instead
                a = a + a.T.conj()
            elif assume_a == 'pos':
                a = a.conj().T.dot(a) + 0.1*np.eye(size)

            tol = 1e-12 if dtype in (np.float64, np.complex128) else 1e-6

            if assume_a in ['gen', 'sym', 'her']:
                # We revert the tolerance from before
                #   4b4a6e7c34fa4060533db38f9a819b98fa81476c
                if dtype in (np.float32, np.complex64):
                    tol *= 10

            x = solve(a, b, assume_a=assume_a)
            assert_allclose(a.dot(x), b,
                            atol=tol * size,
                            rtol=tol * size,
                            err_msg=err_msg)

            if assume_a == 'sym' and dtype not in (np.complex64,
                                                   np.complex128):
                x = solve(a, b, assume_a=assume_a, transposed=True)
                assert_allclose(a.dot(x), b,
                                atol=tol * size,
                                rtol=tol * size,
                                err_msg=err_msg)


class TestSolveTriangular:

    def test_simple(self):
        """
        solve_triangular on a simple 2x2 matrix.
        """
        A = array([[1, 0], [1, 2]])
        b = [1, 1]
        sol = solve_triangular(A, b, lower=True)
        assert_array_almost_equal(sol, [1, 0])

        # check that it works also for non-contiguous matrices
        sol = solve_triangular(A.T, b, lower=False)
        assert_array_almost_equal(sol, [.5, .5])

        # and that it gives the same result as trans=1
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [.5, .5])

        b = identity(2)
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [[1., -.5], [0, 0.5]])

    def test_simple_complex(self):
        """
        solve_triangular on a simple 2x2 complex matrix
        """
        A = array([[1+1j, 0], [1j, 2]])
        b = identity(2)
        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [[.5-.5j, -.25-.25j], [0, 0.5]])

        # check other option combinations with complex rhs
        b = np.diag([1+1j, 1+2j])
        sol = solve_triangular(A, b, lower=True, trans=0)
        assert_array_almost_equal(sol, [[1, 0], [-0.5j, 0.5+1j]])

        sol = solve_triangular(A, b, lower=True, trans=1)
        assert_array_almost_equal(sol, [[1, 0.25-0.75j], [0, 0.5+1j]])

        sol = solve_triangular(A, b, lower=True, trans=2)
        assert_array_almost_equal(sol, [[1j, -0.75-0.25j], [0, 0.5+1j]])

        sol = solve_triangular(A.T, b, lower=False, trans=0)
        assert_array_almost_equal(sol, [[1, 0.25-0.75j], [0, 0.5+1j]])

        sol = solve_triangular(A.T, b, lower=False, trans=1)
        assert_array_almost_equal(sol, [[1, 0], [-0.5j, 0.5+1j]])

        sol = solve_triangular(A.T, b, lower=False, trans=2)
        assert_array_almost_equal(sol, [[1j, 0], [-0.5, 0.5+1j]])

    def test_check_finite(self):
        """
        solve_triangular on a simple 2x2 matrix.
        """
        A = array([[1, 0], [1, 2]])
        b = [1, 1]
        sol = solve_triangular(A, b, lower=True, check_finite=False)
        assert_array_almost_equal(sol, [1, 0])


class TestInv:
    def setup_method(self):
        np.random.seed(1234)

    def test_simple(self):
        a = [[1, 2], [3, 4]]
        a_inv = inv(a)
        assert_array_almost_equal(dot(a, a_inv), np.eye(2))
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        a_inv = inv(a)
        assert_array_almost_equal(dot(a, a_inv), np.eye(3))

    def test_random(self):
        n = 20
        for i in range(4):
            a = random([n, n])
            for i in range(n):
                a[i, i] = 20*(.1+a[i, i])
            a_inv = inv(a)
            assert_array_almost_equal(dot(a, a_inv),
                                      identity(n))

    def test_simple_complex(self):
        a = [[1, 2], [3, 4j]]
        a_inv = inv(a)
        assert_array_almost_equal(dot(a, a_inv), [[1, 0], [0, 1]])

    def test_random_complex(self):
        n = 20
        for i in range(4):
            a = random([n, n])+2j*random([n, n])
            for i in range(n):
                a[i, i] = 20*(.1+a[i, i])
            a_inv = inv(a)
            assert_array_almost_equal(dot(a, a_inv),
                                      identity(n))

    def test_check_finite(self):
        a = [[1, 2], [3, 4]]
        a_inv = inv(a, check_finite=False)
        assert_array_almost_equal(dot(a, a_inv), [[1, 0], [0, 1]])


class TestDet:
    def setup_method(self):
        np.random.seed(1234)

    def test_simple(self):
        a = [[1, 2], [3, 4]]
        a_det = det(a)
        assert_almost_equal(a_det, -2.0)

    def test_simple_complex(self):
        a = [[1, 2], [3, 4j]]
        a_det = det(a)
        assert_almost_equal(a_det, -6+4j)

    def test_random(self):
        basic_det = linalg.det
        n = 20
        for i in range(4):
            a = random([n, n])
            d1 = det(a)
            d2 = basic_det(a)
            assert_almost_equal(d1, d2)

    def test_random_complex(self):
        basic_det = linalg.det
        n = 20
        for i in range(4):
            a = random([n, n]) + 2j*random([n, n])
            d1 = det(a)
            d2 = basic_det(a)
            assert_allclose(d1, d2, rtol=1e-13)

    def test_check_finite(self):
        a = [[1, 2], [3, 4]]
        a_det = det(a, check_finite=False)
        assert_almost_equal(a_det, -2.0)


def direct_lstsq(a, b, cmplx=0):
    at = transpose(a)
    if cmplx:
        at = conjugate(at)
    a1 = dot(at, a)
    b1 = dot(at, b)
    return solve(a1, b1)


class TestLstsq:

    lapack_drivers = ('gelsd', 'gelss', 'gelsy', None)

    def setup_method(self):
        np.random.seed(1234)

    def test_simple_exact(self):
        for dtype in REAL_DTYPES:
            a = np.array([[1, 20], [-30, 4]], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    for bt in (((1, 0), (0, 1)), (1, 0),
                               ((2, 1), (-30, 4))):
                        # Store values in case they are overwritten
                        # later
                        a1 = a.copy()
                        b = np.array(bt, dtype=dtype)
                        b1 = b.copy()
                        out = lstsq(a1, b1,
                                    lapack_driver=lapack_driver,
                                    overwrite_a=overwrite,
                                    overwrite_b=overwrite)
                        x = out[0]
                        r = out[2]
                        assert_(r == 2,
                                'expected efficient rank 2, got %s' % r)
                        assert_allclose(dot(a, x), b,
                                        atol=25 * _eps_cast(a1.dtype),
                                        rtol=25 * _eps_cast(a1.dtype),
                                        err_msg="driver: %s" % lapack_driver)

    def test_simple_overdet(self):
        for dtype in REAL_DTYPES:
            a = np.array([[1, 2], [4, 5], [3, 4]], dtype=dtype)
            b = np.array([1, 2, 3], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    # Store values in case they are overwritten later
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                overwrite_a=overwrite,
                                overwrite_b=overwrite)
                    x = out[0]
                    if lapack_driver == 'gelsy':
                        residuals = np.sum((b - a.dot(x))**2)
                    else:
                        residuals = out[1]
                    r = out[2]
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    assert_allclose(abs((dot(a, x) - b)**2).sum(axis=0),
                                    residuals,
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)
                    assert_allclose(x, (-0.428571428571429, 0.85714285714285),
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)

    def test_simple_overdet_complex(self):
        for dtype in COMPLEX_DTYPES:
            a = np.array([[1+2j, 2], [4, 5], [3, 4]], dtype=dtype)
            b = np.array([1, 2+4j, 3], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    # Store values in case they are overwritten later
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                overwrite_a=overwrite,
                                overwrite_b=overwrite)

                    x = out[0]
                    if lapack_driver == 'gelsy':
                        res = b - a.dot(x)
                        residuals = np.sum(res * res.conj())
                    else:
                        residuals = out[1]
                    r = out[2]
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    assert_allclose(abs((dot(a, x) - b)**2).sum(axis=0),
                                    residuals,
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)
                    assert_allclose(
                                x, (-0.4831460674157303 + 0.258426966292135j,
                                    0.921348314606741 + 0.292134831460674j),
                                rtol=25 * _eps_cast(a1.dtype),
                                atol=25 * _eps_cast(a1.dtype),
                                err_msg="driver: %s" % lapack_driver)

    def test_simple_underdet(self):
        for dtype in REAL_DTYPES:
            a = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            b = np.array([1, 2], dtype=dtype)
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    # Store values in case they are overwritten later
                    a1 = a.copy()
                    b1 = b.copy()
                    out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                overwrite_a=overwrite,
                                overwrite_b=overwrite)

                    x = out[0]
                    r = out[2]
                    assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                    assert_allclose(x, (-0.055555555555555, 0.111111111111111,
                                        0.277777777777777),
                                    rtol=25 * _eps_cast(a1.dtype),
                                    atol=25 * _eps_cast(a1.dtype),
                                    err_msg="driver: %s" % lapack_driver)

    def test_random_exact(self):
        for dtype in REAL_DTYPES:
            for n in (20, 200):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(random([n, n]), dtype=dtype)
                        for i in range(n):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(4):
                            b = np.asarray(random([n, 3]), dtype=dtype)
                            # Store values in case they are overwritten later
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1,
                                        lapack_driver=lapack_driver,
                                        overwrite_a=overwrite,
                                        overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == n, 'expected efficient rank %s, '
                                    'got %s' % (n, r))
                            if dtype is np.float32:
                                assert_allclose(
                                          dot(a, x), b,
                                          rtol=500 * _eps_cast(a1.dtype),
                                          atol=500 * _eps_cast(a1.dtype),
                                          err_msg="driver: %s" % lapack_driver)
                            else:
                                assert_allclose(
                                          dot(a, x), b,
                                          rtol=1000 * _eps_cast(a1.dtype),
                                          atol=1000 * _eps_cast(a1.dtype),
                                          err_msg="driver: %s" % lapack_driver)

    def test_random_complex_exact(self):
        for dtype in COMPLEX_DTYPES:
            for n in (20, 200):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(random([n, n]) + 1j*random([n, n]),
                                       dtype=dtype)
                        for i in range(n):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(2):
                            b = np.asarray(random([n, 3]), dtype=dtype)
                            # Store values in case they are overwritten later
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1, lapack_driver=lapack_driver,
                                        overwrite_a=overwrite,
                                        overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == n, 'expected efficient rank %s, '
                                    'got %s' % (n, r))
                            if dtype is np.complex64:
                                assert_allclose(
                                          dot(a, x), b,
                                          rtol=400 * _eps_cast(a1.dtype),
                                          atol=400 * _eps_cast(a1.dtype),
                                          err_msg="driver: %s" % lapack_driver)
                            else:
                                assert_allclose(
                                          dot(a, x), b,
                                          rtol=1000 * _eps_cast(a1.dtype),
                                          atol=1000 * _eps_cast(a1.dtype),
                                          err_msg="driver: %s" % lapack_driver)

    def test_random_overdet(self):
        for dtype in REAL_DTYPES:
            for (n, m) in ((20, 15), (200, 2)):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(random([n, m]), dtype=dtype)
                        for i in range(m):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(4):
                            b = np.asarray(random([n, 3]), dtype=dtype)
                            # Store values in case they are overwritten later
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1,
                                        lapack_driver=lapack_driver,
                                        overwrite_a=overwrite,
                                        overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == m, 'expected efficient rank %s, '
                                    'got %s' % (m, r))
                            assert_allclose(
                                          x, direct_lstsq(a, b, cmplx=0),
                                          rtol=25 * _eps_cast(a1.dtype),
                                          atol=25 * _eps_cast(a1.dtype),
                                          err_msg="driver: %s" % lapack_driver)

    def test_random_complex_overdet(self):
        for dtype in COMPLEX_DTYPES:
            for (n, m) in ((20, 15), (200, 2)):
                for lapack_driver in TestLstsq.lapack_drivers:
                    for overwrite in (True, False):
                        a = np.asarray(random([n, m]) + 1j*random([n, m]),
                                       dtype=dtype)
                        for i in range(m):
                            a[i, i] = 20 * (0.1 + a[i, i])
                        for i in range(2):
                            b = np.asarray(random([n, 3]), dtype=dtype)
                            # Store values in case they are overwritten
                            # later
                            a1 = a.copy()
                            b1 = b.copy()
                            out = lstsq(a1, b1,
                                        lapack_driver=lapack_driver,
                                        overwrite_a=overwrite,
                                        overwrite_b=overwrite)
                            x = out[0]
                            r = out[2]
                            assert_(r == m, 'expected efficient rank %s, '
                                    'got %s' % (m, r))
                            assert_allclose(
                                      x, direct_lstsq(a, b, cmplx=1),
                                      rtol=25 * _eps_cast(a1.dtype),
                                      atol=25 * _eps_cast(a1.dtype),
                                      err_msg="driver: %s" % lapack_driver)

    def test_check_finite(self):
        with suppress_warnings() as sup:
            # On (some) OSX this tests triggers a warning (gh-7538)
            sup.filter(RuntimeWarning,
                       "internal gelsd driver lwork query error,.*"
                       "Falling back to 'gelss' driver.")

        at = np.array(((1, 20), (-30, 4)))
        for dtype, bt, lapack_driver, overwrite, check_finite in \
            itertools.product(REAL_DTYPES,
                              (((1, 0), (0, 1)), (1, 0), ((2, 1), (-30, 4))),
                              TestLstsq.lapack_drivers,
                              (True, False),
                              (True, False)):

            a = at.astype(dtype)
            b = np.array(bt, dtype=dtype)
            # Store values in case they are overwritten
            # later
            a1 = a.copy()
            b1 = b.copy()
            out = lstsq(a1, b1, lapack_driver=lapack_driver,
                        check_finite=check_finite, overwrite_a=overwrite,
                        overwrite_b=overwrite)
            x = out[0]
            r = out[2]
            assert_(r == 2, 'expected efficient rank 2, got %s' % r)
            assert_allclose(dot(a, x), b,
                            rtol=25 * _eps_cast(a.dtype),
                            atol=25 * _eps_cast(a.dtype),
                            err_msg="driver: %s" % lapack_driver)

    def test_zero_size(self):
        for a_shape, b_shape in (((0, 2), (0,)),
                                 ((0, 4), (0, 2)),
                                 ((4, 0), (4,)),
                                 ((4, 0), (4, 2))):
            b = np.ones(b_shape)
            x, residues, rank, s = lstsq(np.zeros(a_shape), b)
            assert_equal(x, np.zeros((a_shape[1],) + b_shape[1:]))
            residues_should_be = (np.empty((0,)) if a_shape[1]
                                  else np.linalg.norm(b, axis=0)**2)
            assert_equal(residues, residues_should_be)
            assert_(rank == 0, 'expected rank 0')
            assert_equal(s, np.empty((0,)))


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestPinv:
    def setup_method(self):
        np.random.seed(1234)

    def test_simple_real(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        a_pinv = pinv(a)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))
        a_pinv = pinv2(a)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    def test_simple_complex(self):
        a = (array([[1, 2, 3], [4, 5, 6], [7, 8, 10]],
             dtype=float) + 1j * array([[10, 8, 7], [6, 5, 4], [3, 2, 1]],
                                       dtype=float))
        a_pinv = pinv(a)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))
        a_pinv = pinv2(a)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    def test_simple_singular(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        a_pinv = pinv(a)
        a_pinv2 = pinv2(a)
        assert_array_almost_equal(a_pinv, a_pinv2)

    def test_simple_cols(self):
        a = array([[1, 2, 3], [4, 5, 6]], dtype=float)
        a_pinv = pinv(a)
        a_pinv2 = pinv2(a)
        assert_array_almost_equal(a_pinv, a_pinv2)

    def test_simple_rows(self):
        a = array([[1, 2], [3, 4], [5, 6]], dtype=float)
        a_pinv = pinv(a)
        a_pinv2 = pinv2(a)
        assert_array_almost_equal(a_pinv, a_pinv2)

    def test_check_finite(self):
        a = array([[1, 2, 3], [4, 5, 6.], [7, 8, 10]])
        a_pinv = pinv(a, check_finite=False)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))
        a_pinv = pinv2(a, check_finite=False)
        assert_array_almost_equal(dot(a, a_pinv), np.eye(3))

    def test_native_list_argument(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a_pinv = pinv(a)
        a_pinv2 = pinv2(a)
        assert_array_almost_equal(a_pinv, a_pinv2)

    def test_atol_rtol(self):
        n = 12
        # get a random ortho matrix for shuffling
        q, _ = qr(np.random.rand(n, n))
        a_m = np.arange(35.0).reshape(7,5)
        a = a_m.copy()
        a[0,0] = 0.001
        atol = 1e-5
        rtol = 0.05
        # svds of a_m is ~ [116.906, 4.234, tiny, tiny, tiny]
        # svds of a is ~ [116.906, 4.234, 4.62959e-04, tiny, tiny]
        # Just abs cutoff such that we arrive at a_modified
        a_p = pinv(a_m, atol=atol, rtol=0.)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        # Now adiff1 should be around atol value while adiff2 should be
        # relatively tiny
        assert_allclose(np.linalg.norm(adiff1), 5e-4, atol=5.e-4)
        assert_allclose(np.linalg.norm(adiff2), 5e-14, atol=5.e-14)

        # Now do the same but remove another sv ~4.234 via rtol
        a_p = pinv(a_m, atol=atol, rtol=rtol)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        assert_allclose(np.linalg.norm(adiff1), 4.233, rtol=0.01)
        assert_allclose(np.linalg.norm(adiff2), 4.233, rtol=0.01)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
class TestPinvSymmetric:

    def setup_method(self):
        np.random.seed(1234)

    def test_simple_real(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        a = np.dot(a, a.T)
        a_pinv = pinvh(a)
        assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))

    def test_nonpositive(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        a = np.dot(a, a.T)
        u, s, vt = np.linalg.svd(a)
        s[0] *= -1
        a = np.dot(u * s, vt)  # a is now symmetric non-positive and singular
        a_pinv = pinv2(a)
        a_pinvh = pinvh(a)
        assert_array_almost_equal(a_pinv, a_pinvh)

    def test_simple_complex(self):
        a = (array([[1, 2, 3], [4, 5, 6], [7, 8, 10]],
             dtype=float) + 1j * array([[10, 8, 7], [6, 5, 4], [3, 2, 1]],
                                       dtype=float))
        a = np.dot(a, a.conj().T)
        a_pinv = pinvh(a)
        assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))

    def test_native_list_argument(self):
        a = array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        a = np.dot(a, a.T)
        a_pinv = pinvh(a.tolist())
        assert_array_almost_equal(np.dot(a, a_pinv), np.eye(3))

    def test_atol_rtol(self):
        n = 12
        # get a random ortho matrix for shuffling
        q, _ = qr(np.random.rand(n, n))
        a = np.diag([4, 3, 2, 1, 0.99e-4, 0.99e-5] + [0.99e-6]*(n-6))
        a = q.T @ a @ q
        a_m = np.diag([4, 3, 2, 1, 0.99e-4, 0.] + [0.]*(n-6))
        a_m = q.T @ a_m @ q
        atol = 1e-5
        rtol = (4.01e-4 - 4e-5)/4
        # Just abs cutoff such that we arrive at a_modified
        a_p = pinvh(a, atol=atol, rtol=0.)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        # Now adiff1 should dance around atol value since truncation
        # while adiff2 should be relatively tiny
        assert_allclose(norm(adiff1), atol, rtol=0.1)
        assert_allclose(norm(adiff2), 1e-12, atol=1e-11)

        # Now do the same but through rtol cancelling atol value
        a_p = pinvh(a, atol=atol, rtol=rtol)
        adiff1 = a @ a_p @ a - a
        adiff2 = a_m @ a_p @ a_m - a_m
        # adiff1 and adiff2 should be elevated to ~1e-4 due to mismatch
        assert_allclose(norm(adiff1), 1e-4, rtol=0.1)
        assert_allclose(norm(adiff2), 1e-4, rtol=0.1)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@pytest.mark.parametrize('scale', (1e-20, 1., 1e20))
@pytest.mark.parametrize('pinv_', (pinv, pinvh, pinv2))
def test_auto_rcond(scale, pinv_):
    x = np.array([[1, 0], [0, 1e-10]]) * scale
    expected = np.diag(1. / np.diag(x))
    x_inv = pinv_(x)
    assert_allclose(x_inv, expected)


class TestVectorNorms:

    def test_types(self):
        for dtype in np.typecodes['AllFloat']:
            x = np.array([1, 2, 3], dtype=dtype)
            tol = max(1e-15, np.finfo(dtype).eps.real * 20)
            assert_allclose(norm(x), np.sqrt(14), rtol=tol)
            assert_allclose(norm(x, 2), np.sqrt(14), rtol=tol)

        for dtype in np.typecodes['Complex']:
            x = np.array([1j, 2j, 3j], dtype=dtype)
            tol = max(1e-15, np.finfo(dtype).eps.real * 20)
            assert_allclose(norm(x), np.sqrt(14), rtol=tol)
            assert_allclose(norm(x, 2), np.sqrt(14), rtol=tol)

    def test_overflow(self):
        # unlike numpy's norm, this one is
        # safer on overflow
        a = array([1e20], dtype=float32)
        assert_almost_equal(norm(a), a)

    def test_stable(self):
        # more stable than numpy's norm
        a = array([1e4] + [1]*10000, dtype=float32)
        try:
            # snrm in double precision; we obtain the same as for float64
            # -- large atol needed due to varying blas implementations
            assert_allclose(norm(a) - 1e4, 0.5, atol=1e-2)
        except AssertionError:
            # snrm implemented in single precision, == np.linalg.norm result
            msg = ": Result should equal either 0.0 or 0.5 (depending on " \
                  "implementation of snrm2)."
            assert_almost_equal(norm(a) - 1e4, 0.0, err_msg=msg)

    def test_zero_norm(self):
        assert_equal(norm([1, 0, 3], 0), 2)
        assert_equal(norm([1, 2, 3], 0), 3)

    def test_axis_kwd(self):
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        assert_allclose(norm(a, axis=1), [[3.60555128, 4.12310563]] * 2)
        assert_allclose(norm(a, 1, axis=1), [[5.] * 2] * 2)

    def test_keepdims_kwd(self):
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        b = norm(a, axis=1, keepdims=True)
        assert_allclose(b, [[[3.60555128, 4.12310563]]] * 2)
        assert_(b.shape == (2, 1, 2))
        assert_allclose(norm(a, 1, axis=2, keepdims=True), [[[3.], [7.]]] * 2)

    @pytest.mark.skipif(not HAS_ILP64, reason="64-bit BLAS required")
    def test_large_vector(self):
        check_free_memory(free_mb=17000)
        x = np.zeros([2**31], dtype=np.float64)
        x[-1] = 1
        res = norm(x)
        del x
        assert_allclose(res, 1.0)


class TestMatrixNorms:

    def test_matrix_norms(self):
        # Not all of these are matrix norms in the most technical sense.
        np.random.seed(1234)
        for n, m in (1, 1), (1, 3), (3, 1), (4, 4), (4, 5), (5, 4):
            for t in np.single, np.double, np.csingle, np.cdouble, np.int64:
                A = 10 * np.random.randn(n, m).astype(t)
                if np.issubdtype(A.dtype, np.complexfloating):
                    A = (A + 10j * np.random.randn(n, m)).astype(t)
                    t_high = np.cdouble
                else:
                    t_high = np.double
                for order in (None, 'fro', 1, -1, 2, -2, np.inf, -np.inf):
                    actual = norm(A, ord=order)
                    desired = np.linalg.norm(A, ord=order)
                    # SciPy may return higher precision matrix norms.
                    # This is a consequence of using LAPACK.
                    if not np.allclose(actual, desired):
                        desired = np.linalg.norm(A.astype(t_high), ord=order)
                        assert_allclose(actual, desired)

    def test_axis_kwd(self):
        a = np.array([[[2, 1], [3, 4]]] * 2, 'd')
        b = norm(a, ord=np.inf, axis=(1, 0))
        c = norm(np.swapaxes(a, 0, 1), ord=np.inf, axis=(0, 1))
        d = norm(a, ord=1, axis=(0, 1))
        assert_allclose(b, c)
        assert_allclose(c, d)
        assert_allclose(b, d)
        assert_(b.shape == c.shape == d.shape)
        b = norm(a, ord=1, axis=(1, 0))
        c = norm(np.swapaxes(a, 0, 1), ord=1, axis=(0, 1))
        d = norm(a, ord=np.inf, axis=(0, 1))
        assert_allclose(b, c)
        assert_allclose(c, d)
        assert_allclose(b, d)
        assert_(b.shape == c.shape == d.shape)

    def test_keepdims_kwd(self):
        a = np.arange(120, dtype='d').reshape(2, 3, 4, 5)
        b = norm(a, ord=np.inf, axis=(1, 0), keepdims=True)
        c = norm(a, ord=1, axis=(0, 1), keepdims=True)
        assert_allclose(b, c)
        assert_(b.shape == c.shape)


class TestOverwrite:
    def test_solve(self):
        assert_no_overwrite(solve, [(3, 3), (3,)])

    def test_solve_triangular(self):
        assert_no_overwrite(solve_triangular, [(3, 3), (3,)])

    def test_solve_banded(self):
        assert_no_overwrite(lambda ab, b: solve_banded((2, 1), ab, b),
                            [(4, 6), (6,)])

    def test_solveh_banded(self):
        assert_no_overwrite(solveh_banded, [(2, 6), (6,)])

    def test_inv(self):
        assert_no_overwrite(inv, [(3, 3)])

    def test_det(self):
        assert_no_overwrite(det, [(3, 3)])

    def test_lstsq(self):
        assert_no_overwrite(lstsq, [(3, 2), (3,)])

    def test_pinv(self):
        assert_no_overwrite(pinv, [(3, 3)])

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_pinv2(self):
        assert_no_overwrite(pinv2, [(3, 3)])

    def test_pinvh(self):
        assert_no_overwrite(pinvh, [(3, 3)])


class TestSolveCirculant:

    def test_basic1(self):
        c = np.array([1, 2, 3, 5])
        b = np.array([1, -1, 1, 0])
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_basic2(self):
        # b is a 2-d matrix.
        c = np.array([1, 2, -3, -5])
        b = np.arange(12).reshape(4, 3)
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_basic3(self):
        # b is a 3-d matrix.
        c = np.array([1, 2, -3, -5])
        b = np.arange(24).reshape(4, 3, 2)
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_complex(self):
        # Complex b and c
        c = np.array([1+2j, -3, 4j, 5])
        b = np.arange(8).reshape(4, 2) + 0.5j
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_random_b_and_c(self):
        # Random b and c
        np.random.seed(54321)
        c = np.random.randn(50)
        b = np.random.randn(50)
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)

    def test_singular(self):
        # c gives a singular circulant matrix.
        c = np.array([1, 1, 0, 0])
        b = np.array([1, 2, 3, 4])
        x = solve_circulant(c, b, singular='lstsq')
        y, res, rnk, s = lstsq(circulant(c), b)
        assert_allclose(x, y)
        assert_raises(LinAlgError, solve_circulant, x, y)

    def test_axis_args(self):
        # Test use of caxis, baxis and outaxis.

        # c has shape (2, 1, 4)
        c = np.array([[[-1, 2.5, 3, 3.5]], [[1, 6, 6, 6.5]]])

        # b has shape (3, 4)
        b = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [1, -1, 0, 0]])

        x = solve_circulant(c, b, baxis=1)
        assert_equal(x.shape, (4, 2, 3))
        expected = np.empty_like(x)
        expected[:, 0, :] = solve(circulant(c[0]), b.T)
        expected[:, 1, :] = solve(circulant(c[1]), b.T)
        assert_allclose(x, expected)

        x = solve_circulant(c, b, baxis=1, outaxis=-1)
        assert_equal(x.shape, (2, 3, 4))
        assert_allclose(np.rollaxis(x, -1), expected)

        # np.swapaxes(c, 1, 2) has shape (2, 4, 1); b.T has shape (4, 3).
        x = solve_circulant(np.swapaxes(c, 1, 2), b.T, caxis=1)
        assert_equal(x.shape, (4, 2, 3))
        assert_allclose(x, expected)

    def test_native_list_arguments(self):
        # Same as test_basic1 using python's native list.
        c = [1, 2, 3, 5]
        b = [1, -1, 1, 0]
        x = solve_circulant(c, b)
        y = solve(circulant(c), b)
        assert_allclose(x, y)


class TestMatrix_Balance:

    def test_string_arg(self):
        assert_raises(ValueError, matrix_balance, 'Some string for fail')

    def test_infnan_arg(self):
        assert_raises(ValueError, matrix_balance,
                      np.array([[1, 2], [3, np.inf]]))
        assert_raises(ValueError, matrix_balance,
                      np.array([[1, 2], [3, np.nan]]))

    def test_scaling(self):
        _, y = matrix_balance(np.array([[1000, 1], [1000, 0]]))
        # Pre/post LAPACK 3.5.0 gives the same result up to an offset
        # since in each case col norm is x1000 greater and
        # 1000 / 32 ~= 1 * 32 hence balanced with 2 ** 5.
        assert_allclose(int(np.diff(np.log2(np.diag(y)))), 5)

    def test_scaling_order(self):
        A = np.array([[1, 0, 1e-4], [1, 1, 1e-2], [1e4, 1e2, 1]])
        x, y = matrix_balance(A)
        assert_allclose(solve(y, A).dot(y), x)

    def test_separate(self):
        _, (y, z) = matrix_balance(np.array([[1000, 1], [1000, 0]]),
                                   separate=1)
        assert_equal(int(np.diff(np.log2(y))), 5)
        assert_allclose(z, np.arange(2))

    def test_permutation(self):
        A = block_diag(np.ones((2, 2)), np.tril(np.ones((2, 2))),
                       np.ones((3, 3)))
        x, (y, z) = matrix_balance(A, separate=1)
        assert_allclose(y, np.ones_like(y))
        assert_allclose(z, np.array([0, 1, 6, 5, 4, 3, 2]))

    def test_perm_and_scaling(self):
        # Matrix with its diagonal removed
        cases = (  # Case 0
                 np.array([[0., 0., 0., 0., 0.000002],
                           [0., 0., 0., 0., 0.],
                           [2., 2., 0., 0., 0.],
                           [2., 2., 0., 0., 0.],
                           [0., 0., 0.000002, 0., 0.]]),
                 #  Case 1 user reported GH-7258
                 np.array([[-0.5, 0., 0., 0.],
                           [0., -1., 0., 0.],
                           [1., 0., -0.5, 0.],
                           [0., 1., 0., -1.]]),
                 #  Case 2 user reported GH-7258
                 np.array([[-3., 0., 1., 0.],
                           [-1., -1., -0., 1.],
                           [-3., -0., -0., 0.],
                           [-1., -0., 1., -1.]])
                 )

        for A in cases:
            x, y = matrix_balance(A)
            x, (s, p) = matrix_balance(A, separate=1)
            ip = np.empty_like(p)
            ip[p] = np.arange(A.shape[0])
            assert_allclose(y, np.diag(s)[ip, :])
            assert_allclose(solve(y, A).dot(y), x)
