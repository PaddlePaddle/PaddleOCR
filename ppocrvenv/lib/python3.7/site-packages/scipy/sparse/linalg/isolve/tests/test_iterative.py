""" Test functions for the sparse.linalg.isolve module
"""

import itertools
import platform
import sys
import numpy as np

from numpy.testing import (assert_equal, assert_array_equal,
     assert_, assert_allclose, suppress_warnings)
import pytest
from pytest import raises as assert_raises

from numpy import zeros, arange, array, ones, eye, iscomplexobj
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning

from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg.isolve import cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk

# TODO check that method preserve shape and type
# TODO test both preconditioner methods


class Case:
    def __init__(self, name, A, b=None, skip=None, nonconvergence=None):
        self.name = name
        self.A = A
        if b is None:
            self.b = arange(A.shape[0], dtype=float)
        else:
            self.b = b
        if skip is None:
            self.skip = []
        else:
            self.skip = skip
        if nonconvergence is None:
            self.nonconvergence = []
        else:
            self.nonconvergence = nonconvergence

    def __repr__(self):
        return "<%s>" % self.name


class IterativeParams:
    def __init__(self):
        # list of tuples (solver, symmetric, positive_definite )
        solvers = [cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk]
        sym_solvers = [minres, cg]
        posdef_solvers = [cg]
        real_solvers = [minres]

        self.solvers = solvers

        # list of tuples (A, symmetric, positive_definite )
        self.cases = []

        # Symmetric and Positive Definite
        N = 40
        data = ones((3,N))
        data[0,:] = 2
        data[1,:] = -1
        data[2,:] = -1
        Poisson1D = spdiags(data, [0,-1,1], N, N, format='csr')
        self.Poisson1D = Case("poisson1d", Poisson1D)
        self.cases.append(Case("poisson1d", Poisson1D))
        # note: minres fails for single precision
        self.cases.append(Case("poisson1d", Poisson1D.astype('f'),
                               skip=[minres]))

        # Symmetric and Negative Definite
        self.cases.append(Case("neg-poisson1d", -Poisson1D,
                               skip=posdef_solvers))
        # note: minres fails for single precision
        self.cases.append(Case("neg-poisson1d", (-Poisson1D).astype('f'),
                               skip=posdef_solvers + [minres]))

        # Symmetric and Indefinite
        data = array([[6, -5, 2, 7, -1, 10, 4, -3, -8, 9]],dtype='d')
        RandDiag = spdiags(data, [0], 10, 10, format='csr')
        self.cases.append(Case("rand-diag", RandDiag, skip=posdef_solvers))
        self.cases.append(Case("rand-diag", RandDiag.astype('f'),
                               skip=posdef_solvers))

        # Random real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        self.cases.append(Case("rand", data, skip=posdef_solvers+sym_solvers))
        self.cases.append(Case("rand", data.astype('f'),
                               skip=posdef_solvers+sym_solvers))

        # Random symmetric real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        data = data + data.T
        self.cases.append(Case("rand-sym", data, skip=posdef_solvers))
        self.cases.append(Case("rand-sym", data.astype('f'),
                               skip=posdef_solvers))

        # Random pos-def symmetric real
        np.random.seed(1234)
        data = np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-sym-pd", data))
        # note: minres fails for single precision
        self.cases.append(Case("rand-sym-pd", data.astype('f'),
                               skip=[minres]))

        # Random complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        self.cases.append(Case("rand-cmplx", data,
                               skip=posdef_solvers+sym_solvers+real_solvers))
        self.cases.append(Case("rand-cmplx", data.astype('F'),
                               skip=posdef_solvers+sym_solvers+real_solvers))

        # Random hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        data = data + data.T.conj()
        self.cases.append(Case("rand-cmplx-herm", data,
                               skip=posdef_solvers+real_solvers))
        self.cases.append(Case("rand-cmplx-herm", data.astype('F'),
                               skip=posdef_solvers+real_solvers))

        # Random pos-def hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(9, 9) + 1j*np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-cmplx-sym-pd", data, skip=real_solvers))
        self.cases.append(Case("rand-cmplx-sym-pd", data.astype('F'),
                               skip=real_solvers))

        # Non-symmetric and Positive Definite
        #
        # cgs, qmr, and bicg fail to converge on this one
        #   -- algorithmic limitation apparently
        data = ones((2,10))
        data[0,:] = 2
        data[1,:] = -1
        A = spdiags(data, [0,-1], 10, 10, format='csr')
        self.cases.append(Case("nonsymposdef", A,
                               skip=sym_solvers+[cgs, qmr, bicg]))
        self.cases.append(Case("nonsymposdef", A.astype('F'),
                               skip=sym_solvers+[cgs, qmr, bicg]))

        # Symmetric, non-pd, hitting cgs/bicg/bicgstab/qmr breakdown
        A = np.array([[0, 0, 0, 0, 0, 1, -1, -0, -0, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -1, -0, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -0, -1, -0, -0],
                      [0, 0, 0, 0, 0, 2, -0, -0, -0, -1, -0],
                      [0, 0, 0, 0, 0, 1, -0, -0, -0, -0, -1],
                      [1, 2, 2, 2, 1, 0, -0, -0, -0, -0, -0],
                      [-1, 0, 0, 0, 0, 0, -1, -0, -0, -0, -0],
                      [0, -1, 0, 0, 0, 0, -0, -1, -0, -0, -0],
                      [0, 0, -1, 0, 0, 0, -0, -0, -1, -0, -0],
                      [0, 0, 0, -1, 0, 0, -0, -0, -0, -1, -0],
                      [0, 0, 0, 0, -1, 0, -0, -0, -0, -0, -1]], dtype=float)
        b = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
        assert (A == A.T).all()
        self.cases.append(Case("sym-nonpd", A, b,
                               skip=posdef_solvers,
                               nonconvergence=[cgs,bicg,bicgstab,qmr]))


params = IterativeParams()


def check_maxiter(solver, case):
    A = case.A
    tol = 1e-12

    b = case.b
    x0 = 0*b

    residuals = []

    def callback(x):
        residuals.append(norm(b - case.A*x))

    x, info = solver(A, b, x0=x0, tol=tol, maxiter=1, callback=callback)

    assert_equal(len(residuals), 1)
    assert_equal(info, 1)


def test_maxiter():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            check_maxiter(solver, case)


def assert_normclose(a, b, tol=1e-8):
    residual = norm(a - b)
    tolerance = tol*norm(b)
    msg = "residual (%g) not smaller than tolerance %g" % (residual, tolerance)
    assert_(residual < tolerance, msg=msg)


def check_convergence(solver, case):
    A = case.A

    if A.dtype.char in "dD":
        tol = 1e-8
    else:
        tol = 1e-2

    b = case.b
    x0 = 0*b

    x, info = solver(A, b, x0=x0, tol=tol)

    assert_array_equal(x0, 0*b)  # ensure that x0 is not overwritten
    if solver not in case.nonconvergence:
        assert_equal(info,0)
        assert_normclose(A.dot(x), b, tol=tol)
    else:
        assert_(info != 0)
        assert_(np.linalg.norm(A.dot(x) - b) <= np.linalg.norm(b))


def test_convergence():
    for solver in params.solvers:
        for case in params.cases:
            if solver in case.skip:
                continue
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning, ".*called without specifying.*")
                check_convergence(solver, case)


def check_precond_dummy(solver, case):
    tol = 1e-8

    def identity(b,which=None):
        """trivial preconditioner"""
        return b

    A = case.A

    M,N = A.shape
    spdiags([1.0/A.diagonal()], [0], M, N)

    b = case.b
    x0 = 0*b

    precond = LinearOperator(A.shape, identity, rmatvec=identity)

    if solver is qmr:
        x, info = solver(A, b, M1=precond, M2=precond, x0=x0, tol=tol)
    else:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol)
    assert_equal(info,0)
    assert_normclose(A.dot(x), b, tol)

    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity

    x, info = solver(A, b, x0=x0, tol=tol)
    assert_equal(info,0)
    assert_normclose(A*x, b, tol=tol)


def test_precond_dummy():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            check_precond_dummy(solver, case)


def check_precond_inverse(solver, case):
    tol = 1e-8

    def inverse(b,which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.todense()
        return np.linalg.solve(A, b)

    def rinverse(b,which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.todense()
        return np.linalg.solve(A.T, b)

    matvec_count = [0]

    def matvec(b):
        matvec_count[0] += 1
        return case.A.dot(b)

    def rmatvec(b):
        matvec_count[0] += 1
        return case.A.T.dot(b)

    b = case.b
    x0 = 0*b

    A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
    precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)

    # Solve with preconditioner
    matvec_count = [0]
    x, info = solver(A, b, M=precond, x0=x0, tol=tol)

    assert_equal(info, 0)
    assert_normclose(case.A.dot(x), b, tol)

    # Solution should be nearly instant
    assert_(matvec_count[0] <= 3, repr(matvec_count))


def test_precond_inverse():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        if solver is qmr:
            continue
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            check_precond_inverse(solver, case)


def test_gmres_basic():
    A = np.vander(np.arange(10) + 1)[:, ::-1]
    b = np.zeros(10)
    b[0] = 1
    np.linalg.solve(A, b)

    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, ".*called without specifying.*")
        x_gm, err = gmres(A, b, restart=5, maxiter=1)

    assert_allclose(x_gm[0], 0.359, rtol=1e-2)


def test_reentrancy():
    non_reentrant = [cg, cgs, bicg, bicgstab, gmres, qmr]
    reentrant = [lgmres, minres, gcrotmk]
    for solver in reentrant + non_reentrant:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            _check_reentrancy(solver, solver in reentrant)


def _check_reentrancy(solver, is_reentrant):
    def matvec(x):
        A = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
        y, info = solver(A, x)
        assert_equal(info, 0)
        return y
    b = np.array([1, 1./2, 1./3])
    op = LinearOperator((3, 3), matvec=matvec, rmatvec=matvec,
                        dtype=b.dtype)

    if not is_reentrant:
        assert_raises(RuntimeError, solver, op, b)
    else:
        y, info = solver(op, b)
        assert_equal(info, 0)
        assert_allclose(y, [1, 1, 1])


@pytest.mark.parametrize("solver", [cg, cgs, bicg, bicgstab, gmres, qmr, lgmres, gcrotmk])
def test_atol(solver):
    # TODO: minres. It didn't historically use absolute tolerances, so
    # fixing it is less urgent.

    np.random.seed(1234)
    A = np.random.rand(10, 10)
    A = A.dot(A.T) + 10 * np.eye(10)
    b = 1e3 * np.random.rand(10)
    b_norm = np.linalg.norm(b)

    tols = np.r_[0, np.logspace(np.log10(1e-10), np.log10(1e2), 7), np.inf]

    # Check effect of badly scaled preconditioners
    M0 = np.random.randn(10, 10)
    M0 = M0.dot(M0.T)
    Ms = [None, 1e-6 * M0, 1e6 * M0]

    for M, tol, atol in itertools.product(Ms, tols, tols):
        if tol == 0 and atol == 0:
            continue

        if solver is qmr:
            if M is not None:
                M = aslinearoperator(M)
                M2 = aslinearoperator(np.eye(10))
            else:
                M2 = None
            x, info = solver(A, b, M1=M, M2=M2, tol=tol, atol=atol)
        else:
            x, info = solver(A, b, M=M, tol=tol, atol=atol)
        assert_equal(info, 0)

        residual = A.dot(x) - b
        err = np.linalg.norm(residual)
        atol2 = tol * b_norm
        assert_(err <= max(atol, atol2))


@pytest.mark.parametrize("solver", [cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk])
def test_zero_rhs(solver):
    np.random.seed(1234)
    A = np.random.rand(10, 10)
    A = A.dot(A.T) + 10 * np.eye(10)

    b = np.zeros(10)
    tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

    for tol in tols:
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")

            x, info = solver(A, b, tol=tol)
            assert_equal(info, 0)
            assert_allclose(x, 0, atol=1e-15)

            x, info = solver(A, b, tol=tol, x0=ones(10))
            assert_equal(info, 0)
            assert_allclose(x, 0, atol=tol)

            if solver is not minres:
                x, info = solver(A, b, tol=tol, atol=0, x0=ones(10))
                if info == 0:
                    assert_allclose(x, 0)

                x, info = solver(A, b, tol=tol, atol=tol)
                assert_equal(info, 0)
                assert_allclose(x, 0, atol=1e-300)

                x, info = solver(A, b, tol=tol, atol=0)
                assert_equal(info, 0)
                assert_allclose(x, 0, atol=1e-300)


@pytest.mark.parametrize("solver", [
    pytest.param(gmres, marks=pytest.mark.xfail(platform.machine() == 'aarch64'
                                                and sys.version_info[1] == 9,
                                                reason="gh-13019")),
    qmr,
    pytest.param(lgmres, marks=pytest.mark.xfail(platform.machine() == 'ppc64le',
                                                 reason="fails on ppc64le")),
    pytest.param(cgs, marks=pytest.mark.xfail),
    pytest.param(bicg, marks=pytest.mark.xfail),
    pytest.param(bicgstab, marks=pytest.mark.xfail),
    pytest.param(gcrotmk, marks=pytest.mark.xfail)])
def test_maxiter_worsening(solver):
    # Check error does not grow (boundlessly) with increasing maxiter.
    # This can occur due to the solvers hitting close to breakdown,
    # which they should detect and halt as necessary.
    # cf. gh-9100

    # Singular matrix, rhs numerically not in range
    A = np.array([[-0.1112795288033378, 0, 0, 0.16127952880333685],
                  [0, -0.13627952880333782+6.283185307179586j, 0, 0],
                  [0, 0, -0.13627952880333782-6.283185307179586j, 0],
                  [0.1112795288033368, 0j, 0j, -0.16127952880333785]])
    v = np.ones(4)
    best_error = np.inf
    tol = 7 if platform.machine() == 'aarch64' else 5

    for maxiter in range(1, 20):
        x, info = solver(A, v, maxiter=maxiter, tol=1e-8, atol=0)

        if info == 0:
            assert_(np.linalg.norm(A.dot(x) - v) <= 1e-8*np.linalg.norm(v))

        error = np.linalg.norm(A.dot(x) - v)
        best_error = min(best_error, error)

        # Check with slack
        assert_(error <= tol*best_error)


@pytest.mark.parametrize("solver", [cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk])
def test_x0_working(solver):
    # Easy problem
    np.random.seed(1)
    n = 10
    A = np.random.rand(n, n)
    A = A.dot(A.T)
    b = np.random.rand(n)
    x0 = np.random.rand(n)

    if solver is minres:
        kw = dict(tol=1e-6)
    else:
        kw = dict(atol=0, tol=1e-6)

    x, info = solver(A, b, **kw)
    assert_equal(info, 0)
    assert_(np.linalg.norm(A.dot(x) - b) <= 1e-6*np.linalg.norm(b))

    x, info = solver(A, b, x0=x0, **kw)
    assert_equal(info, 0)
    assert_(np.linalg.norm(A.dot(x) - b) <= 1e-6*np.linalg.norm(b))


#------------------------------------------------------------------------------

class TestQMR:
    def test_leftright_precond(self):
        """Check that QMR works with left and right preconditioners"""

        from scipy.sparse.linalg.dsolve import splu
        from scipy.sparse.linalg.interface import LinearOperator

        n = 100

        dat = ones(n)
        A = spdiags([-2*dat, 4*dat, -dat], [-1,0,1],n,n)
        b = arange(n,dtype='d')

        L = spdiags([-dat/2, dat], [-1,0], n, n)
        U = spdiags([4*dat, -dat], [0,1], n, n)

        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, "splu requires CSC matrix format")
            L_solver = splu(L)
            U_solver = splu(U)

        def L_solve(b):
            return L_solver.solve(b)

        def U_solve(b):
            return U_solver.solve(b)

        def LT_solve(b):
            return L_solver.solve(b,'T')

        def UT_solve(b):
            return U_solver.solve(b,'T')

        M1 = LinearOperator((n,n), matvec=L_solve, rmatvec=LT_solve)
        M2 = LinearOperator((n,n), matvec=U_solve, rmatvec=UT_solve)

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x,info = qmr(A, b, tol=1e-8, maxiter=15, M1=M1, M2=M2)

        assert_equal(info,0)
        assert_normclose(A*x, b, tol=1e-8)


class TestGMRES:
    def test_callback(self):

        def store_residual(r, rvec):
            rvec[rvec.nonzero()[0].max()+1] = r

        # Define, A,b
        A = csr_matrix(array([[-2,1,0,0,0,0],[1,-2,1,0,0,0],[0,1,-2,1,0,0],[0,0,1,-2,1,0],[0,0,0,1,-2,1],[0,0,0,0,1,-2]]))
        b = ones((A.shape[0],))
        maxiter = 1
        rvec = zeros(maxiter+1)
        rvec[0] = 1.0
        callback = lambda r:store_residual(r, rvec)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x,flag = gmres(A, b, x0=zeros(A.shape[0]), tol=1e-16, maxiter=maxiter, callback=callback)

        # Expected output from SciPy 1.0.0
        assert_allclose(rvec, array([1.0, 0.81649658092772603]), rtol=1e-10)

        # Test preconditioned callback
        M = 1e-3 * np.eye(A.shape[0])
        rvec = zeros(maxiter+1)
        rvec[0] = 1.0
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            x, flag = gmres(A, b, M=M, tol=1e-16, maxiter=maxiter, callback=callback)

        # Expected output from SciPy 1.0.0 (callback has preconditioned residual!)
        assert_allclose(rvec, array([1.0, 1e-3 * 0.81649658092772603]), rtol=1e-10)

    def test_abi(self):
        # Check we don't segfault on gmres with complex argument
        A = eye(2)
        b = ones(2)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            r_x, r_info = gmres(A, b)
            r_x = r_x.astype(complex)

            x, info = gmres(A.astype(complex), b.astype(complex))

        assert_(iscomplexobj(x))
        assert_allclose(r_x, x)
        assert_(r_info == info)

    def test_atol_legacy(self):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")

            # Check the strange legacy behavior: the tolerance is interpreted
            # as atol, but only for the initial residual
            A = eye(2)
            b = 1e-6 * ones(2)
            x, info = gmres(A, b, tol=1e-5)
            assert_array_equal(x, np.zeros(2))

            A = eye(2)
            b = ones(2)
            x, info = gmres(A, b, tol=1e-5)
            assert_(np.linalg.norm(A.dot(x) - b) <= 1e-5*np.linalg.norm(b))
            assert_allclose(x, b, atol=0, rtol=1e-8)

            rndm = np.random.RandomState(12345)
            A = rndm.rand(30, 30)
            b = 1e-6 * ones(30)
            x, info = gmres(A, b, tol=1e-7, restart=20)
            assert_(np.linalg.norm(A.dot(x) - b) > 1e-7)

        A = eye(2)
        b = 1e-10 * ones(2)
        x, info = gmres(A, b, tol=1e-8, atol=0)
        assert_(np.linalg.norm(A.dot(x) - b) <= 1e-8*np.linalg.norm(b))

    def test_defective_precond_breakdown(self):
        # Breakdown due to defective preconditioner
        M = np.eye(3)
        M[2,2] = 0

        b = np.array([0, 1, 1])
        x = np.array([1, 0, 0])
        A = np.diag([2, 3, 4])

        x, info = gmres(A, b, x0=x, M=M, tol=1e-15, atol=0)

        # Should not return nans, nor terminate with false success
        assert_(not np.isnan(x).any())
        if info == 0:
            assert_(np.linalg.norm(A.dot(x) - b) <= 1e-15*np.linalg.norm(b))

        # The solution should be OK outside null space of M
        assert_allclose(M.dot(A.dot(x)), M.dot(b))

    def test_defective_matrix_breakdown(self):
        # Breakdown due to defective matrix
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        b = np.array([1, 0, 1])
        x, info = gmres(A, b, tol=1e-8, atol=0)

        # Should not return nans, nor terminate with false success
        assert_(not np.isnan(x).any())
        if info == 0:
            assert_(np.linalg.norm(A.dot(x) - b) <= 1e-8*np.linalg.norm(b))

        # The solution should be OK outside null space of A
        assert_allclose(A.dot(A.dot(x)), A.dot(b))

    def test_callback_type(self):
        # The legacy callback type changes meaning of 'maxiter'
        np.random.seed(1)
        A = np.random.rand(20, 20)
        b = np.random.rand(20)

        cb_count = [0]

        def pr_norm_cb(r):
            cb_count[0] += 1
            assert_(isinstance(r, float))

        def x_cb(x):
            cb_count[0] += 1
            assert_(isinstance(x, np.ndarray))

        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, ".*called without specifying.*")
            # 2 iterations is not enough to solve the problem
            cb_count = [0]
            x, info = gmres(A, b, tol=1e-6, atol=0, callback=pr_norm_cb, maxiter=2, restart=50)
            assert info == 2
            assert cb_count[0] == 2

        # With `callback_type` specified, no warning should be raised
        cb_count = [0]
        x, info = gmres(A, b, tol=1e-6, atol=0, callback=pr_norm_cb, maxiter=2, restart=50,
                        callback_type='legacy')
        assert info == 2
        assert cb_count[0] == 2

        # 2 restart cycles is enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, tol=1e-6, atol=0, callback=pr_norm_cb, maxiter=2, restart=50,
                        callback_type='pr_norm')
        assert info == 0
        assert cb_count[0] > 2

        # 2 restart cycles is enough to solve the problem
        cb_count = [0]
        x, info = gmres(A, b, tol=1e-6, atol=0, callback=x_cb, maxiter=2, restart=50,
                        callback_type='x')
        assert info == 0
        assert cb_count[0] == 2

    def test_callback_x_monotonic(self):
        # Check that callback_type='x' gives monotonic norm decrease
        np.random.seed(1)
        A = np.random.rand(20, 20) + np.eye(20)
        b = np.random.rand(20)

        prev_r = [np.inf]
        count = [0]

        def x_cb(x):
            r = np.linalg.norm(A.dot(x) - b)
            assert r <= prev_r[0]
            prev_r[0] = r
            count[0] += 1

        x, info = gmres(A, b, tol=1e-6, atol=0, callback=x_cb, maxiter=20, restart=10,
                        callback_type='x')
        assert info == 20
        assert count[0] == 21
        x_cb(x)
