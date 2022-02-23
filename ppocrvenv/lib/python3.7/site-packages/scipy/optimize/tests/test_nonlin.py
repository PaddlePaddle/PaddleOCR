""" Unit tests for nonlinear solvers
Author: Ondrej Certik
May 2007
"""
from numpy.testing import assert_
import pytest

from scipy.optimize import nonlin, root
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np

from .test_minpack import pressure_network

SOLVERS = {'anderson': nonlin.anderson, 'diagbroyden': nonlin.diagbroyden,
           'linearmixing': nonlin.linearmixing, 'excitingmixing': nonlin.excitingmixing,
           'broyden1': nonlin.broyden1, 'broyden2': nonlin.broyden2,
           'krylov': nonlin.newton_krylov}
MUST_WORK = {'anderson': nonlin.anderson, 'broyden1': nonlin.broyden1,
             'broyden2': nonlin.broyden2, 'krylov': nonlin.newton_krylov}

#-------------------------------------------------------------------------------
# Test problems
#-------------------------------------------------------------------------------


def F(x):
    x = np.asarray(x).T
    d = diag([3,2,1.5,1,0.5])
    c = 0.01
    f = -d @ x - c * float(x.T @ x) * x
    return f


F.xin = [1,1,1,1,1]
F.KNOWN_BAD = {}


def F2(x):
    return x


F2.xin = [1,2,3,4,5,6]
F2.KNOWN_BAD = {'linearmixing': nonlin.linearmixing,
                'excitingmixing': nonlin.excitingmixing}


def F2_lucky(x):
    return x


F2_lucky.xin = [0,0,0,0,0,0]
F2_lucky.KNOWN_BAD = {}


def F3(x):
    A = np.array([[-2, 1, 0.], [1, -2, 1], [0, 1, -2]])
    b = np.array([1, 2, 3.])
    return A @ x - b


F3.xin = [1,2,3]
F3.KNOWN_BAD = {}


def F4_powell(x):
    A = 1e4
    return [A*x[0]*x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - (1 + 1/A)]


F4_powell.xin = [-1, -2]
F4_powell.KNOWN_BAD = {'linearmixing': nonlin.linearmixing,
                       'excitingmixing': nonlin.excitingmixing,
                       'diagbroyden': nonlin.diagbroyden}


def F5(x):
    return pressure_network(x, 4, np.array([.5, .5, .5, .5]))


F5.xin = [2., 0, 2, 0]
F5.KNOWN_BAD = {'excitingmixing': nonlin.excitingmixing,
                'linearmixing': nonlin.linearmixing,
                'diagbroyden': nonlin.diagbroyden}


def F6(x):
    x1, x2 = x
    J0 = np.array([[-4.256, 14.7],
                [0.8394989, 0.59964207]])
    v = np.array([(x1 + 3) * (x2**5 - 7) + 3*6,
                  np.sin(x2 * np.exp(x1) - 1)])
    return -np.linalg.solve(J0, v)


F6.xin = [-0.5, 1.4]
F6.KNOWN_BAD = {'excitingmixing': nonlin.excitingmixing,
                'linearmixing': nonlin.linearmixing,
                'diagbroyden': nonlin.diagbroyden}


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------


class TestNonlin:
    """
    Check the Broyden methods for a few test problems.

    broyden1, broyden2, and newton_krylov must succeed for
    all functions. Some of the others don't -- tests in KNOWN_BAD are skipped.

    """

    def _check_nonlin_func(self, f, func, f_tol=1e-2):
        x = func(f, f.xin, f_tol=f_tol, maxiter=200, verbose=0)
        assert_(np.absolute(f(x)).max() < f_tol)

    def _check_root(self, f, method, f_tol=1e-2):
        res = root(f, f.xin, method=method,
                   options={'ftol': f_tol, 'maxiter': 200, 'disp': 0})
        assert_(np.absolute(res.fun).max() < f_tol)

    @pytest.mark.xfail
    def _check_func_fail(self, *a, **kw):
        pass

    def test_problem_nonlin(self):
        for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
            for func in SOLVERS.values():
                if func in f.KNOWN_BAD.values():
                    if func in MUST_WORK.values():
                        self._check_func_fail(f, func)
                    continue
                self._check_nonlin_func(f, func)

    def test_tol_norm_called(self):
        # Check that supplying tol_norm keyword to nonlin_solve works
        self._tol_norm_used = False

        def local_norm_func(x):
            self._tol_norm_used = True
            return np.absolute(x).max()

        nonlin.newton_krylov(F, F.xin, f_tol=1e-2, maxiter=200, verbose=0,
             tol_norm=local_norm_func)
        assert_(self._tol_norm_used)

    def test_problem_root(self):
        for f in [F, F2, F2_lucky, F3, F4_powell, F5, F6]:
            for meth in SOLVERS:
                if meth in f.KNOWN_BAD:
                    if meth in MUST_WORK:
                        self._check_func_fail(f, meth)
                    continue
                self._check_root(f, meth)


class TestSecant:
    """Check that some Jacobian approximations satisfy the secant condition"""

    xs = [np.array([1,2,3,4,5], float),
          np.array([2,3,4,5,1], float),
          np.array([3,4,5,1,2], float),
          np.array([4,5,1,2,3], float),
          np.array([9,1,9,1,3], float),
          np.array([0,1,9,1,3], float),
          np.array([5,5,7,1,1], float),
          np.array([1,2,7,5,1], float),]
    fs = [x**2 - 1 for x in xs]

    def _check_secant(self, jac_cls, npoints=1, **kw):
        """
        Check that the given Jacobian approximation satisfies secant
        conditions for last `npoints` points.
        """
        jac = jac_cls(**kw)
        jac.setup(self.xs[0], self.fs[0], None)
        for j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            jac.update(x, f)

            for k in range(min(npoints, j+1)):
                dx = self.xs[j-k+1] - self.xs[j-k]
                df = self.fs[j-k+1] - self.fs[j-k]
                assert_(np.allclose(dx, jac.solve(df)))

            # Check that the `npoints` secant bound is strict
            if j >= npoints:
                dx = self.xs[j-npoints+1] - self.xs[j-npoints]
                df = self.fs[j-npoints+1] - self.fs[j-npoints]
                assert_(not np.allclose(dx, jac.solve(df)))

    def test_broyden1(self):
        self._check_secant(nonlin.BroydenFirst)

    def test_broyden2(self):
        self._check_secant(nonlin.BroydenSecond)

    def test_broyden1_update(self):
        # Check that BroydenFirst update works as for a dense matrix
        jac = nonlin.BroydenFirst(alpha=0.1)
        jac.setup(self.xs[0], self.fs[0], None)

        B = np.identity(5) * (-1/0.1)

        for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            df = f - self.fs[last_j]
            dx = x - self.xs[last_j]
            B += (df - dot(B, dx))[:,None] * dx[None,:] / dot(dx, dx)
            jac.update(x, f)
            assert_(np.allclose(jac.todense(), B, rtol=1e-10, atol=1e-13))

    def test_broyden2_update(self):
        # Check that BroydenSecond update works as for a dense matrix
        jac = nonlin.BroydenSecond(alpha=0.1)
        jac.setup(self.xs[0], self.fs[0], None)

        H = np.identity(5) * (-0.1)

        for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
            df = f - self.fs[last_j]
            dx = x - self.xs[last_j]
            H += (dx - dot(H, df))[:,None] * df[None,:] / dot(df, df)
            jac.update(x, f)
            assert_(np.allclose(jac.todense(), inv(H), rtol=1e-10, atol=1e-13))

    def test_anderson(self):
        # Anderson mixing (with w0=0) satisfies secant conditions
        # for the last M iterates, see [Ey]_
        #
        # .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).
        self._check_secant(nonlin.Anderson, M=3, w0=0, npoints=3)


class TestLinear:
    """Solve a linear equation;
    some methods find the exact solution in a finite number of steps"""

    def _check(self, jac, N, maxiter, complex=False, **kw):
        np.random.seed(123)

        A = np.random.randn(N, N)
        if complex:
            A = A + 1j*np.random.randn(N, N)
        b = np.random.randn(N)
        if complex:
            b = b + 1j*np.random.randn(N)

        def func(x):
            return dot(A, x) - b

        sol = nonlin.nonlin_solve(func, np.zeros(N), jac, maxiter=maxiter,
                                  f_tol=1e-6, line_search=None, verbose=0)
        assert_(np.allclose(dot(A, sol), b, atol=1e-6))

    def test_broyden1(self):
        # Broyden methods solve linear systems exactly in 2*N steps
        self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, False)
        self._check(nonlin.BroydenFirst(alpha=1.0), 20, 41, True)

    def test_broyden2(self):
        # Broyden methods solve linear systems exactly in 2*N steps
        self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, False)
        self._check(nonlin.BroydenSecond(alpha=1.0), 20, 41, True)

    def test_anderson(self):
        # Anderson is rather similar to Broyden, if given enough storage space
        self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, False)
        self._check(nonlin.Anderson(M=50, alpha=1.0), 20, 29, True)

    def test_krylov(self):
        # Krylov methods solve linear systems exactly in N inner steps
        self._check(nonlin.KrylovJacobian, 20, 2, False, inner_m=10)
        self._check(nonlin.KrylovJacobian, 20, 2, True, inner_m=10)


class TestJacobianDotSolve:
    """Check that solve/dot methods in Jacobian approximations are consistent"""

    def _func(self, x):
        return x**2 - 1 + np.dot(self.A, x)

    def _check_dot(self, jac_cls, complex=False, tol=1e-6, **kw):
        np.random.seed(123)

        N = 7

        def rand(*a):
            q = np.random.rand(*a)
            if complex:
                q = q + 1j*np.random.rand(*a)
            return q

        def assert_close(a, b, msg):
            d = abs(a - b).max()
            f = tol + abs(b).max()*tol
            if d > f:
                raise AssertionError('%s: err %g' % (msg, d))

        self.A = rand(N, N)

        # initialize
        x0 = np.random.rand(N)
        jac = jac_cls(**kw)
        jac.setup(x0, self._func(x0), self._func)

        # check consistency
        for k in range(2*N):
            v = rand(N)

            if hasattr(jac, '__array__'):
                Jd = np.array(jac)
                if hasattr(jac, 'solve'):
                    Gv = jac.solve(v)
                    Gv2 = np.linalg.solve(Jd, v)
                    assert_close(Gv, Gv2, 'solve vs array')
                if hasattr(jac, 'rsolve'):
                    Gv = jac.rsolve(v)
                    Gv2 = np.linalg.solve(Jd.T.conj(), v)
                    assert_close(Gv, Gv2, 'rsolve vs array')
                if hasattr(jac, 'matvec'):
                    Jv = jac.matvec(v)
                    Jv2 = np.dot(Jd, v)
                    assert_close(Jv, Jv2, 'dot vs array')
                if hasattr(jac, 'rmatvec'):
                    Jv = jac.rmatvec(v)
                    Jv2 = np.dot(Jd.T.conj(), v)
                    assert_close(Jv, Jv2, 'rmatvec vs array')

            if hasattr(jac, 'matvec') and hasattr(jac, 'solve'):
                Jv = jac.matvec(v)
                Jv2 = jac.solve(jac.matvec(Jv))
                assert_close(Jv, Jv2, 'dot vs solve')

            if hasattr(jac, 'rmatvec') and hasattr(jac, 'rsolve'):
                Jv = jac.rmatvec(v)
                Jv2 = jac.rmatvec(jac.rsolve(Jv))
                assert_close(Jv, Jv2, 'rmatvec vs rsolve')

            x = rand(N)
            jac.update(x, self._func(x))

    def test_broyden1(self):
        self._check_dot(nonlin.BroydenFirst, complex=False)
        self._check_dot(nonlin.BroydenFirst, complex=True)

    def test_broyden2(self):
        self._check_dot(nonlin.BroydenSecond, complex=False)
        self._check_dot(nonlin.BroydenSecond, complex=True)

    def test_anderson(self):
        self._check_dot(nonlin.Anderson, complex=False)
        self._check_dot(nonlin.Anderson, complex=True)

    def test_diagbroyden(self):
        self._check_dot(nonlin.DiagBroyden, complex=False)
        self._check_dot(nonlin.DiagBroyden, complex=True)

    def test_linearmixing(self):
        self._check_dot(nonlin.LinearMixing, complex=False)
        self._check_dot(nonlin.LinearMixing, complex=True)

    def test_excitingmixing(self):
        self._check_dot(nonlin.ExcitingMixing, complex=False)
        self._check_dot(nonlin.ExcitingMixing, complex=True)

    def test_krylov(self):
        self._check_dot(nonlin.KrylovJacobian, complex=False, tol=1e-3)
        self._check_dot(nonlin.KrylovJacobian, complex=True, tol=1e-3)


class TestNonlinOldTests:
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """

    def test_broyden1(self):
        x = nonlin.broyden1(F,F.xin,iter=12,alpha=1)
        assert_(nonlin.norm(x) < 1e-9)
        assert_(nonlin.norm(F(x)) < 1e-9)

    def test_broyden2(self):
        x = nonlin.broyden2(F,F.xin,iter=12,alpha=1)
        assert_(nonlin.norm(x) < 1e-9)
        assert_(nonlin.norm(F(x)) < 1e-9)

    def test_anderson(self):
        x = nonlin.anderson(F,F.xin,iter=12,alpha=0.03,M=5)
        assert_(nonlin.norm(x) < 0.33)

    def test_linearmixing(self):
        x = nonlin.linearmixing(F,F.xin,iter=60,alpha=0.5)
        assert_(nonlin.norm(x) < 1e-7)
        assert_(nonlin.norm(F(x)) < 1e-7)

    def test_exciting(self):
        x = nonlin.excitingmixing(F,F.xin,iter=20,alpha=0.5)
        assert_(nonlin.norm(x) < 1e-5)
        assert_(nonlin.norm(F(x)) < 1e-5)

    def test_diagbroyden(self):
        x = nonlin.diagbroyden(F,F.xin,iter=11,alpha=1)
        assert_(nonlin.norm(x) < 1e-8)
        assert_(nonlin.norm(F(x)) < 1e-8)

    def test_root_broyden1(self):
        res = root(F, F.xin, method='broyden1',
                   options={'nit': 12, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-9)
        assert_(nonlin.norm(res.fun) < 1e-9)

    def test_root_broyden2(self):
        res = root(F, F.xin, method='broyden2',
                   options={'nit': 12, 'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-9)
        assert_(nonlin.norm(res.fun) < 1e-9)

    def test_root_anderson(self):
        res = root(F, F.xin, method='anderson',
                   options={'nit': 12,
                            'jac_options': {'alpha': 0.03, 'M': 5}})
        assert_(nonlin.norm(res.x) < 0.33)

    def test_root_linearmixing(self):
        res = root(F, F.xin, method='linearmixing',
                   options={'nit': 60,
                            'jac_options': {'alpha': 0.5}})
        assert_(nonlin.norm(res.x) < 1e-7)
        assert_(nonlin.norm(res.fun) < 1e-7)

    def test_root_excitingmixing(self):
        res = root(F, F.xin, method='excitingmixing',
                   options={'nit': 20,
                            'jac_options': {'alpha': 0.5}})
        assert_(nonlin.norm(res.x) < 1e-5)
        assert_(nonlin.norm(res.fun) < 1e-5)

    def test_root_diagbroyden(self):
        res = root(F, F.xin, method='diagbroyden',
                   options={'nit': 11,
                            'jac_options': {'alpha': 1}})
        assert_(nonlin.norm(res.x) < 1e-8)
        assert_(nonlin.norm(res.fun) < 1e-8)
