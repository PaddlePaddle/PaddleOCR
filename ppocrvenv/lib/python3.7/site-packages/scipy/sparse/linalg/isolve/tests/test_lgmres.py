"""Tests for the linalg.isolve.lgmres module
"""

from numpy.testing import (assert_, assert_allclose, assert_equal,
                           suppress_warnings)

import pytest
from platform import python_implementation

import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand

from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg.isolve import lgmres, gmres


Am = csr_matrix(array([[-2, 1, 0, 0, 0, 9],
                       [1, -2, 1, 0, 5, 0],
                       [0, 1, -2, 1, 0, 0],
                       [0, 0, 1, -2, 1, 0],
                       [0, 3, 0, 1, -2, 1],
                       [1, 0, 0, 0, 1, -2]]))
b = array([1, 2, 3, 4, 5, 6])
count = [0]


def matvec(v):
    count[0] += 1
    return Am*v


A = LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)


def do_solve(**kw):
    count[0] = 0
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, ".*called without specifying.*")
        x0, flag = lgmres(A, b, x0=zeros(A.shape[0]),
                          inner_m=6, tol=1e-14, **kw)
    count_0 = count[0]
    assert_(allclose(A*x0, b, rtol=1e-12, atol=1e-12), norm(A*x0-b))
    return x0, count_0


class TestLGMRES:
    def test_preconditioner(self):
        # Check that preconditioning works
        pc = splu(Am.tocsc())
        M = LinearOperator(matvec=pc.solve, shape=A.shape, dtype=A.dtype)

        x0, count_0 = do_solve()
        x1, count_1 = do_solve(M=M)

        assert_(count_1 == 3)
        assert_(count_1 < count_0/2)
        assert_(allclose(x1, x0, rtol=1e-14))

    def test_outer_v(self):
        # Check that the augmentation vectors behave as expected

        outer_v = []
        x0, count_0 = do_solve(outer_k=6, outer_v=outer_v)
        assert_(len(outer_v) > 0)
        assert_(len(outer_v) <= 6)

        x1, count_1 = do_solve(outer_k=6, outer_v=outer_v,
                               prepend_outer_v=True)
        assert_(count_1 == 2, count_1)
        assert_(count_1 < count_0/2)
        assert_(allclose(x1, x0, rtol=1e-14))

        # ---

        outer_v = []
        x0, count_0 = do_solve(outer_k=6, outer_v=outer_v,
                               store_outer_Av=False)
        assert_(array([v[1] is None for v in outer_v]).all())
        assert_(len(outer_v) > 0)
        assert_(len(outer_v) <= 6)

        x1, count_1 = do_solve(outer_k=6, outer_v=outer_v,
                               prepend_outer_v=True)
        assert_(count_1 == 3, count_1)
        assert_(count_1 < count_0/2)
        assert_(allclose(x1, x0, rtol=1e-14))

    @pytest.mark.skipif(python_implementation() == 'PyPy',
                        reason="Fails on PyPy CI runs. See #9507")
    def test_arnoldi(self):
        np.random.seed(1234)

        A = eye(2000) + rand(2000, 2000, density=5e-4)
        b = np.random.rand(2000)

        # The inner arnoldi should be equivalent to gmres
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x0, flag0 = lgmres(A, b, x0=zeros(A.shape[0]),
                               inner_m=15, maxiter=1)
            x1, flag1 = gmres(A, b, x0=zeros(A.shape[0]),
                              restart=15, maxiter=1)

        assert_equal(flag0, 1)
        assert_equal(flag1, 1)
        norm = np.linalg.norm(A.dot(x0) - b)
        assert_(norm > 1e-4)
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
                x, info = lgmres(A, b, maxiter=10)
                assert_equal(info, 0)
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                x, info = lgmres(A, b, tol=0, maxiter=10)
                if info == 0:
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                b = np.random.rand(n)
                x, info = lgmres(A, b, maxiter=10)
                assert_equal(info, 0)
                assert_allclose(A.dot(x) - b, 0, atol=1e-14)

                x, info = lgmres(A, b, tol=0, maxiter=10)
                if info == 0:
                    assert_allclose(A.dot(x) - b, 0, atol=1e-14)

    def test_nans(self):
        A = eye(3, format='lil')
        A[1, 1] = np.nan
        b = np.ones(3)

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x, info = lgmres(A, b, tol=0, maxiter=10)
            assert_equal(info, 1)

    def test_breakdown_with_outer_v(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([1, 2])

        x = np.linalg.solve(A, b)
        v0 = np.array([1, 0])

        # The inner iteration should converge to the correct solution,
        # since it's in the outer vector list
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            xp, info = lgmres(A, b, outer_v=[(v0, None), (x, None)], maxiter=1)

        assert_allclose(xp, x, atol=1e-12)

    def test_breakdown_underdetermined(self):
        # Should find LSQ solution in the Krylov span in one inner
        # iteration, despite solver breakdown from nilpotent A.
        A = np.array([[0, 1, 1, 1],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]], dtype=float)

        bs = [
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 0]),
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 0, 0]),
        ]

        for b in bs:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                xp, info = lgmres(A, b, maxiter=1)
            resp = np.linalg.norm(A.dot(xp) - b)

            K = np.c_[b, A.dot(b), A.dot(A.dot(b)), A.dot(A.dot(A.dot(b)))]
            y, _, _, _ = np.linalg.lstsq(A.dot(K), b, rcond=-1)
            x = K.dot(y)
            res = np.linalg.norm(A.dot(x) - b)

            assert_allclose(resp, res, err_msg=repr(b))

    def test_denormals(self):
        # Check that no warnings are emitted if the matrix contains
        # numbers for which 1/x has no float representation, and that
        # the solver behaves properly.
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A *= 100 * np.nextafter(0, 1)

        b = np.array([1, 1])

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            xp, info = lgmres(A, b)

        if info == 0:
            assert_allclose(A.dot(xp), b)

