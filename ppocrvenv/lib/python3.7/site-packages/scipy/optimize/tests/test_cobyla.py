import math
import numpy as np

from numpy.testing import assert_allclose, assert_

from scipy.optimize import fmin_cobyla, minimize


class TestCobyla:
    def setup_method(self):
        self.x0 = [4.95, 0.66]
        self.solution = [math.sqrt(25 - (2.0/3)**2), 2.0/3]
        self.opts = {'disp': False, 'rhobeg': 1, 'tol': 1e-5,
                     'maxiter': 100}

    def fun(self, x):
        return x[0]**2 + abs(x[1])**3

    def con1(self, x):
        return x[0]**2 + x[1]**2 - 25

    def con2(self, x):
        return -self.con1(x)

    def test_simple(self):
        # use disp=True as smoke test for gh-8118
        x = fmin_cobyla(self.fun, self.x0, [self.con1, self.con2], rhobeg=1,
                        rhoend=1e-5, maxfun=100, disp=True)
        assert_allclose(x, self.solution, atol=1e-4)

    def test_minimize_simple(self):
        # Minimize with method='COBYLA'
        cons = ({'type': 'ineq', 'fun': self.con1},
                {'type': 'ineq', 'fun': self.con2})
        sol = minimize(self.fun, self.x0, method='cobyla', constraints=cons,
                       options=self.opts)
        assert_allclose(sol.x, self.solution, atol=1e-4)
        assert_(sol.success, sol.message)
        assert_(sol.maxcv < 1e-5, sol)
        assert_(sol.nfev < 70, sol)
        assert_(sol.fun < self.fun(self.solution) + 1e-3, sol)

    def test_minimize_constraint_violation(self):
        np.random.seed(1234)
        pb = np.random.rand(10, 10)
        spread = np.random.rand(10)

        def p(w):
            return pb.dot(w)

        def f(w):
            return -(w * spread).sum()

        def c1(w):
            return 500 - abs(p(w)).sum()

        def c2(w):
            return 5 - abs(p(w).sum())

        def c3(w):
            return 5 - abs(p(w)).max()

        cons = ({'type': 'ineq', 'fun': c1},
                {'type': 'ineq', 'fun': c2},
                {'type': 'ineq', 'fun': c3})
        w0 = np.zeros((10, 1))
        sol = minimize(f, w0, method='cobyla', constraints=cons,
                       options={'catol': 1e-6})
        assert_(sol.maxcv > 1e-6)
        assert_(not sol.success)


def test_vector_constraints():
    # test that fmin_cobyla and minimize can take a combination
    # of constraints, some returning a number and others an array
    def fun(x):
        return (x[0] - 1)**2 + (x[1] - 2.5)**2

    def fmin(x):
        return fun(x) - 1

    def cons1(x):
        a = np.array([[1, -2, 2], [-1, -2, 6], [-1, 2, 2]])
        return np.array([a[i, 0] * x[0] + a[i, 1] * x[1] +
                         a[i, 2] for i in range(len(a))])

    def cons2(x):
        return x     # identity, acts as bounds x > 0

    x0 = np.array([2, 0])
    cons_list = [fun, cons1, cons2]

    xsol = [1.4, 1.7]
    fsol = 0.8

    # testing fmin_cobyla
    sol = fmin_cobyla(fun, x0, cons_list, rhoend=1e-5)
    assert_allclose(sol, xsol, atol=1e-4)

    sol = fmin_cobyla(fun, x0, fmin, rhoend=1e-5)
    assert_allclose(fun(sol), 1, atol=1e-4)

    # testing minimize
    constraints = [{'type': 'ineq', 'fun': cons} for cons in cons_list]
    sol = minimize(fun, x0, constraints=constraints, tol=1e-5)
    assert_allclose(sol.x, xsol, atol=1e-4)
    assert_(sol.success, sol.message)
    assert_allclose(sol.fun, fsol, atol=1e-4)

    constraints = {'type': 'ineq', 'fun': fmin}
    sol = minimize(fun, x0, constraints=constraints, tol=1e-5)
    assert_allclose(sol.fun, 1, atol=1e-4)

