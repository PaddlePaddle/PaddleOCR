""" Test functions for the sparse.linalg.eigen.lobpcg module
"""
import itertools
import platform

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_allclose, assert_array_less)

import pytest

from numpy import ones, r_, diag
from numpy.random import rand
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg.eigen.lobpcg import lobpcg

def ElasticRod(n):
    """Build the matrices for the generalized eigenvalue problem of the
    fixed-free elastic rod vibration model.
    """
    L = 1.0
    le = L/n
    rho = 7.85e3
    S = 1.e-4
    E = 2.1e11
    mass = rho*S*le/6.
    k = E*S/le
    A = k*(diag(r_[2.*ones(n-1), 1])-diag(ones(n-1), 1)-diag(ones(n-1), -1))
    B = mass*(diag(r_[4.*ones(n-1), 2])+diag(ones(n-1), 1)+diag(ones(n-1), -1))
    return A, B


def MikotaPair(n):
    """Build a pair of full diagonal matrices for the generalized eigenvalue
    problem. The Mikota pair acts as a nice test since the eigenvalues are the
    squares of the integers n, n=1,2,...
    """
    x = np.arange(1, n+1)
    B = diag(1./x)
    y = np.arange(n-1, 0, -1)
    z = np.arange(2*n-1, 0, -2)
    A = diag(z)-diag(y, -1)-diag(y, 1)
    return A, B


def compare_solutions(A, B, m):
    """Check eig vs. lobpcg consistency.
    """
    n = A.shape[0]
    np.random.seed(0)
    V = rand(n, m)
    X = orth(V)
    eigvals, _ = lobpcg(A, X, B=B, tol=1e-5, maxiter=30, largest=False)
    eigvals.sort()
    w, _ = eig(A, b=B)
    w.sort()
    assert_almost_equal(w[:int(m/2)], eigvals[:int(m/2)], decimal=2)


def test_Small():
    A, B = ElasticRod(10)
    compare_solutions(A, B, 10)
    A, B = MikotaPair(10)
    compare_solutions(A, B, 10)


def test_ElasticRod():
    A, B = ElasticRod(100)
    compare_solutions(A, B, 20)


def test_MikotaPair():
    A, B = MikotaPair(100)
    compare_solutions(A, B, 20)


def test_regression():
    """Check the eigenvalue of the identity matrix is one.
    """
    # https://mail.python.org/pipermail/scipy-user/2010-October/026944.html
    n = 10
    X = np.ones((n, 1))
    A = np.identity(n)
    w, _ = lobpcg(A, X)
    assert_allclose(w, [1])


def test_diagonal():
    """Check for diagonal matrices.
    """
    # This test was moved from '__main__' in lobpcg.py.
    # Coincidentally or not, this is the same eigensystem
    # required to reproduce arpack bug
    # https://forge.scilab.org/p/arpack-ng/issues/1397/
    # even using the same n=100.

    np.random.seed(1234)

    # The system of interest is of size n x n.
    n = 100

    # We care about only m eigenpairs.
    m = 4

    # Define the generalized eigenvalue problem Av = cBv
    # where (c, v) is a generalized eigenpair,
    # and where we choose A to be the diagonal matrix whose entries are 1..n
    # and where B is chosen to be the identity matrix.
    vals = np.arange(1, n+1, dtype=float)
    A = diags([vals], [0], (n, n))
    B = eye(n)

    # Let the preconditioner M be the inverse of A.
    M = diags([1./vals], [0], (n, n))

    # Pick random initial vectors.
    X = np.random.rand(n, m)

    # Require that the returned eigenvectors be in the orthogonal complement
    # of the first few standard basis vectors.
    m_excluded = 3
    Y = np.eye(n, m_excluded)

    eigvals, vecs = lobpcg(A, X, B, M=M, Y=Y, tol=1e-4, maxiter=40, largest=False)

    assert_allclose(eigvals, np.arange(1+m_excluded, 1+m_excluded+m))
    _check_eigen(A, eigvals, vecs, rtol=1e-3, atol=1e-3)


def _check_eigen(M, w, V, rtol=1e-8, atol=1e-14):
    """Check if the eigenvalue residual is small.
    """
    mult_wV = np.multiply(w, V)
    dot_MV = M.dot(V)
    assert_allclose(mult_wV, dot_MV, rtol=rtol, atol=atol)


def _check_fiedler(n, p):
    """Check the Fiedler vector computation.
    """
    # This is not necessarily the recommended way to find the Fiedler vector.
    np.random.seed(1234)
    col = np.zeros(n)
    col[1] = 1
    A = toeplitz(col)
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Compute the full eigendecomposition using tricks, e.g.
    # http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    tmp = np.pi * np.arange(n) / n
    analytic_w = 2 * (1 - np.cos(tmp))
    analytic_V = np.cos(np.outer(np.arange(n) + 1/2, tmp))
    _check_eigen(L, analytic_w, analytic_V)
    # Compute the full eigendecomposition using eigh.
    eigh_w, eigh_V = eigh(L)
    _check_eigen(L, eigh_w, eigh_V)
    # Check that the first eigenvalue is near zero and that the rest agree.
    assert_array_less(np.abs([eigh_w[0], analytic_w[0]]), 1e-14)
    assert_allclose(eigh_w[1:], analytic_w[1:])

    # Check small lobpcg eigenvalues.
    X = analytic_V[:, :p]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_array_less(np.abs(np.min(lobpcg_w)), 1e-14)
    assert_allclose(np.sort(lobpcg_w)[1:], analytic_w[1:p])

    # Check large lobpcg eigenvalues.
    X = analytic_V[:, -p:]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=True)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_allclose(np.sort(lobpcg_w), analytic_w[-p:])

    # Look for the Fiedler vector using good but not exactly correct guesses.
    fiedler_guess = np.concatenate((np.ones(n//2), -np.ones(n-n//2)))
    X = np.vstack((np.ones(n), fiedler_guess)).T
    lobpcg_w, _ = lobpcg(L, X, largest=False)
    # Mathematically, the smaller eigenvalue should be zero
    # and the larger should be the algebraic connectivity.
    lobpcg_w = np.sort(lobpcg_w)
    assert_allclose(lobpcg_w, analytic_w[:2], atol=1e-14)


def test_fiedler_small_8():
    """Check the dense workaround path for small matrices.
    """
    # This triggers the dense path because 8 < 2*5.
    _check_fiedler(8, 2)


def test_fiedler_large_12():
    """Check the dense workaround path avoided for non-small matrices.
    """
    # This does not trigger the dense path, because 2*5 <= 12.
    _check_fiedler(12, 2)


def test_hermitian():
    """Check complex-value Hermitian cases.
    """
    np.random.seed(1234)

    sizes = [3, 10, 50]
    ks = [1, 3, 10, 50]
    gens = [True, False]

    for size, k, gen in itertools.product(sizes, ks, gens):
        if k > size:
            continue

        H = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
        H = 10 * np.eye(size) + H + H.T.conj()

        X = np.random.rand(size, k)

        if not gen:
            B = np.eye(size)
            w, v = lobpcg(H, X, maxiter=5000)
            w0, _ = eigh(H)
        else:
            B = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
            B = 10 * np.eye(size) + B.dot(B.T.conj())
            w, v = lobpcg(H, X, B, maxiter=5000, largest=False)
            w0, _ = eigh(H, B)

        for wx, vx in zip(w, v.T):
            # Check eigenvector
            assert_allclose(np.linalg.norm(H.dot(vx) - B.dot(vx) * wx)
                            / np.linalg.norm(H.dot(vx)),
                            0, atol=5e-4, rtol=0)

            # Compare eigenvalues
            j = np.argmin(abs(w0 - wx))
            assert_allclose(wx, w0[j], rtol=1e-4)


# The n=5 case tests the alternative small matrix code path that uses eigh().
@pytest.mark.parametrize('n, atol', [(20, 1e-3), (5, 1e-8)])
def test_eigs_consistency(n, atol):
    """Check eigs vs. lobpcg consistency.
    """
    vals = np.arange(1, n+1, dtype=np.float64)
    A = spdiags(vals, 0, n, n)
    np.random.seed(345678)
    X = np.random.rand(n, 2)
    lvals, lvecs = lobpcg(A, X, largest=True, maxiter=100)
    vals, _ = eigs(A, k=2)

    _check_eigen(A, lvals, lvecs, atol=atol, rtol=0)
    assert_allclose(np.sort(vals), np.sort(lvals), atol=1e-14)


def test_verbosity(tmpdir):
    """Check that nonzero verbosity level code runs.
    """
    A, B = ElasticRod(100)
    n = A.shape[0]
    m = 20
    np.random.seed(0)
    V = rand(n, m)
    X = orth(V)
    _, _ = lobpcg(A, X, B=B, tol=1e-5, maxiter=30, largest=False,
                  verbosityLevel=9)


@pytest.mark.xfail(platform.machine() == 'ppc64le',
                   reason="fails on ppc64le")
def test_tolerance_float32():
    """Check lobpcg for attainable tolerance in float32.
    """
    np.random.seed(1234)
    n = 50
    m = 3
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = np.random.randn(n, m)
    X = X.astype(np.float32)
    eigvals, _ = lobpcg(A, X, tol=1e-9, maxiter=50, verbosityLevel=0)
    assert_allclose(eigvals, -np.arange(1, 1 + m), atol=1e-5)


def test_random_initial_float32():
    """Check lobpcg in float32 for specific initial.
    """
    np.random.seed(3)
    n = 50
    m = 4
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = np.random.rand(n, m)
    X = X.astype(np.float32)
    eigvals, _ = lobpcg(A, X, tol=1e-3, maxiter=50, verbosityLevel=1)
    assert_allclose(eigvals, -np.arange(1, 1 + m), atol=1e-2)


def test_maxit_None():
    """Check lobpcg if maxit=None runs 20 iterations (the default)
    by checking the size of the iteration history output, which should
    be the number of iterations plus 2 (initial and final values).
    """
    np.random.seed(1566950023)
    n = 50
    m = 4
    vals = -np.arange(1, n + 1)
    A = diags([vals], [0], (n, n))
    A = A.astype(np.float32)
    X = np.random.randn(n, m)
    X = X.astype(np.float32)
    _, _, l_h = lobpcg(A, X, tol=1e-8, maxiter=20, retLambdaHistory=True)
    assert_allclose(np.shape(l_h)[0], 20+2)


@pytest.mark.slow
def test_diagonal_data_types():
    """Check lobpcg for diagonal matrices for all matrix types.
    """
    np.random.seed(1234)
    n = 40
    m = 4
    # Define the generalized eigenvalue problem Av = cBv
    # where (c, v) is a generalized eigenpair,
    # and where we choose A  and B to be diagonal.
    vals = np.arange(1, n + 1)

    list_sparse_format = ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil']
    sparse_formats = len(list_sparse_format)
    for s_f_i, s_f in enumerate(list_sparse_format):

        As64 = diags([vals * vals], [0], (n, n), format=s_f)
        As32 = As64.astype(np.float32)
        Af64 = As64.toarray()
        Af32 = Af64.astype(np.float32)
        listA = [Af64, As64, Af32, As32]

        Bs64 = diags([vals], [0], (n, n), format=s_f)
        Bf64 = Bs64.toarray()
        listB = [Bf64, Bs64]

        # Define the preconditioner function as LinearOperator.
        Ms64 = diags([1./vals], [0], (n, n), format=s_f)

        def Ms64precond(x):
            return Ms64 @ x
        Ms64precondLO = LinearOperator(matvec=Ms64precond,
                                    matmat=Ms64precond,
                                    shape=(n, n), dtype=float)
        Mf64 = Ms64.toarray()

        def Mf64precond(x):
            return Mf64 @ x
        Mf64precondLO = LinearOperator(matvec=Mf64precond,
                                    matmat=Mf64precond,
                                    shape=(n, n), dtype=float)
        Ms32 = Ms64.astype(np.float32)

        def Ms32precond(x):
            return Ms32 @ x
        Ms32precondLO = LinearOperator(matvec=Ms32precond,
                                    matmat=Ms32precond,
                                    shape=(n, n), dtype=np.float32)
        Mf32 = Ms32.toarray()

        def Mf32precond(x):
            return Mf32 @ x
        Mf32precondLO = LinearOperator(matvec=Mf32precond,
                                    matmat=Mf32precond,
                                    shape=(n, n), dtype=np.float32)
        listM = [None, Ms64precondLO, Mf64precondLO,
                 Ms32precondLO, Mf32precondLO]

        # Setup matrix of the initial approximation to the eigenvectors
        # (cannot be sparse array).
        Xf64 = np.random.rand(n, m)
        Xf32 = Xf64.astype(np.float32)
        listX = [Xf64, Xf32]

        # Require that the returned eigenvectors be in the orthogonal complement
        # of the first few standard basis vectors (cannot be sparse array).
        m_excluded = 3
        Yf64 = np.eye(n, m_excluded, dtype=float)
        Yf32 = np.eye(n, m_excluded, dtype=np.float32)
        listY = [Yf64, Yf32]

        tests = list(itertools.product(listA, listB, listM, listX, listY))
        # This is one of the slower tests because there are >1,000 configs
        # to test here, instead of checking product of all input, output types
        # test each configuration for the first sparse format, and then
        # for one additional sparse format. this takes 2/7=30% as long as
        # testing all configurations for all sparse formats.
        if s_f_i > 0:
            tests = tests[s_f_i - 1::sparse_formats-1]

        for A, B, M, X, Y in tests:
            eigvals, _ = lobpcg(A, X, B=B, M=M, Y=Y, tol=1e-4,
                                maxiter=100, largest=False)
            assert_allclose(eigvals,
                            np.arange(1 + m_excluded, 1 + m_excluded + m))
