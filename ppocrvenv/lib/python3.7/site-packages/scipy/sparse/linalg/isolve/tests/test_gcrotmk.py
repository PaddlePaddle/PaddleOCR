#!/usr/bin/env python
"""Tests for the linalg.isolve.gcrotmk module
"""

from numpy.testing import (assert_, assert_allclose, assert_equal,
                           suppress_warnings)

import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand

from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg.isolve import gcrotmk, gmres


Am = csr_matrix(array([[-2,1,0,0,0,9],
                       [1,-2,1,0,5,0],
                       [0,1,-2,1,0,0],
                       [0,0,1,-2,1,0],
                       [0,3,0,1,-2,1],
                       [1,0,0,0,1,-2]]))
b = array([1,2,3,4,5,6])
count = [0]


def matvec(v):
    count[0] += 1
    return Am*v


A = LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)


def do_solve(**kw):
    count[0] = 0
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, ".*called without specifying.*")
        x0, flag = gcrotmk(A, b, x0=zeros(A.shape[0]), tol=1e-14, **kw)
    count_0 = count[0]
    assert_(allclose(A*x0, b, rtol=1e-12, atol=1e-12), norm(A*x0-b))
    return x0, count_0


class TestGCROTMK:
    def test_preconditioner(self):
        # Check that preconditioning works
        pc = splu(Am.tocsc())
        M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)

        x0, count_0 = do_solve()
        x1, count_1 = do_solve(M=M)

        assert_equal(count_1, 3)
        assert_(count_1 < count_0/2)
        assert_(allclose(x1, x0, rtol=1e-14))

    def test_arnoldi(self):
        np.random.seed(1)

        A = eye(2000) + rand(2000, 2000, density=5e-4)
        b = np.random.rand(2000)

        # The inner arnoldi should be equivalent to gmres
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x0, flag0 = gcrotmk(A, b, x0=zeros(A.shape[0]), m=15, k=0, maxiter=1)
            x1, flag1 = gmres(A, b, x0=zeros(A.shape[0]), restart=15, maxiter=1)

        assert_equal(flag0, 1)
        assert_equal(flag1, 1)
        assert np.linalg.norm(A.dot(x0) - b) > 1e-3

        assert_allclose(x0, x1)

    def test_cornercase(self):
        np.random.seed(1234)

        # Rounding error may prevent convergence with tol=0 --- ensure
        # that the return values in this case are correct, and no
        # exceptions are raised

        for n in [3, 5, 10, 100]:
            A = 2*eye(n)

            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                b = np.ones(n)
                x, info = gcrotmk(A, b, maxiter=10)
                assert_equal(info, 0)
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                x, info = gcrotmk(A, b, tol=0, maxiter=10)
                if info == 0:
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                b = np.random.rand(n)
                x, info = gcrotmk(A, b, maxiter=10)
                assert_equal(info, 0)
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                x, info = gcrotmk(A, b, tol=0, maxiter=10)
                if info == 0:
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

    def test_nans(self):
        A = eye(3, format='lil')
        A[1,1] = np.nan
        b = np.ones(3)

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x, info = gcrotmk(A, b, tol=0, maxiter=10)
            assert_equal(info, 1)

    def test_truncate(self):
        np.random.seed(1234)
        A = np.random.rand(30, 30) + np.eye(30)
        b = np.random.rand(30)

        for truncate in ['oldest', 'smallest']:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                x, info = gcrotmk(A, b, m=10, k=10, truncate=truncate, tol=1e-4,
                                  maxiter=200)
            assert_equal(info, 0)
            assert_allclose(A.dot(x) - b, 0, atol=1e-3)

    def test_CU(self):
        for discard_C in (True, False):
            # Check that C,U behave as expected
            CU = []
            x0, count_0 = do_solve(CU=CU, discard_C=discard_C)
            assert_(len(CU) > 0)
            assert_(len(CU) <= 6)

            if discard_C:
                for c, u in CU:
                    assert_(c is None)

            # should converge immediately
            x1, count_1 = do_solve(CU=CU, discard_C=discard_C)
            if discard_C:
                assert_equal(count_1, 2 + len(CU))
            else:
                assert_equal(count_1, 3)
            assert_(count_1 <= count_0/2)
            assert_allclose(x1, x0, atol=1e-14)

    def test_denormals(self):
        # Check that no warnings are emitted if the matrix contains
        # numbers for which 1/x has no float representation, and that
        # the solver behaves properly.
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A *= 100 * np.nextafter(0, 1)

        b = np.array([1, 1])

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            xp, info = gcrotmk(A, b)

        if info == 0:
            assert_allclose(A.dot(xp), b)
