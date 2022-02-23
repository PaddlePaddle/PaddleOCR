"""
Unit tests for optimization routines from _root.py.
"""
from numpy.testing import assert_
from pytest import raises as assert_raises
import numpy as np

from scipy.optimize import root


class TestRoot:
    def test_tol_parameter(self):
        # Check that the minimize() tol= argument does something
        def func(z):
            x, y = z
            return np.array([x**3 - 1, y**3 - 1])

        def dfunc(z):
            x, y = z
            return np.array([[3*x**2, 0], [0, 3*y**2]])

        for method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson',
                       'diagbroyden', 'krylov']:
            if method in ('linearmixing', 'excitingmixing'):
                # doesn't converge
                continue

            if method in ('hybr', 'lm'):
                jac = dfunc
            else:
                jac = None

            sol1 = root(func, [1.1,1.1], jac=jac, tol=1e-4, method=method)
            sol2 = root(func, [1.1,1.1], jac=jac, tol=0.5, method=method)
            msg = "%s: %s vs. %s" % (method, func(sol1.x), func(sol2.x))
            assert_(sol1.success, msg)
            assert_(sol2.success, msg)
            assert_(abs(func(sol1.x)).max() < abs(func(sol2.x)).max(),
                    msg)

    def test_tol_norm(self):

        def norm(x):
            return abs(x[0])

        for method in ['excitingmixing',
                       'diagbroyden',
                       'linearmixing',
                       'anderson',
                       'broyden1',
                       'broyden2',
                       'krylov']:

            root(np.zeros_like, np.zeros(2), method=method,
                options={"tol_norm": norm})

    def test_minimize_scalar_coerce_args_param(self):
        # github issue #3503
        def func(z, f=1):
            x, y = z
            return np.array([x**3 - 1, y**3 - f])
        root(func, [1.1, 1.1], args=1.5)

    def test_f_size(self):
        # gh8320
        # check that decreasing the size of the returned array raises an error
        # and doesn't segfault
        class fun:
            def __init__(self):
                self.count = 0

            def __call__(self, x):
                self.count += 1

                if not (self.count % 5):
                    ret = x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0
                else:
                    ret = ([x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0,
                           0.5 * (x[1] - x[0]) ** 3 + x[1]])

                return ret

        F = fun()
        with assert_raises(ValueError):
            root(F, [0.1, 0.0], method='lm')
