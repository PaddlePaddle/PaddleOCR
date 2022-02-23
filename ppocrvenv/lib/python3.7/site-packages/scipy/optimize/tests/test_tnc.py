"""
Unit tests for TNC optimization routine from tnc.py
"""

from numpy.testing import assert_allclose, assert_equal

import numpy as np
from math import pow

from scipy import optimize
from scipy.sparse.sputils import matrix


class TestTnc:
    """TNC non-linear optimization.

    These tests are taken from Prof. K. Schittkowski's test examples
    for constrained non-linear programming.

    http://www.uni-bayreuth.de/departments/math/~kschittkowski/home.htm

    """
    def setup_method(self):
        # options for minimize
        self.opts = {'disp': False, 'maxfun': 200}

    # objective functions and Jacobian for each test
    def f1(self, x, a=100.0):
        return a * pow((x[1] - pow(x[0], 2)), 2) + pow(1.0 - x[0], 2)

    def g1(self, x, a=100.0):
        dif = [0, 0]
        dif[1] = 2 * a * (x[1] - pow(x[0], 2))
        dif[0] = -2.0 * (x[0] * (dif[1] - 1.0) + 1.0)
        return dif

    def fg1(self, x, a=100.0):
        return self.f1(x, a), self.g1(x, a)

    def f3(self, x):
        return x[1] + pow(x[1] - x[0], 2) * 1.0e-5

    def g3(self, x):
        dif = [0, 0]
        dif[0] = -2.0 * (x[1] - x[0]) * 1.0e-5
        dif[1] = 1.0 - dif[0]
        return dif

    def fg3(self, x):
        return self.f3(x), self.g3(x)

    def f4(self, x):
        return pow(x[0] + 1.0, 3) / 3.0 + x[1]

    def g4(self, x):
        dif = [0, 0]
        dif[0] = pow(x[0] + 1.0, 2)
        dif[1] = 1.0
        return dif

    def fg4(self, x):
        return self.f4(x), self.g4(x)

    def f5(self, x):
        return np.sin(x[0] + x[1]) + pow(x[0] - x[1], 2) - \
                1.5 * x[0] + 2.5 * x[1] + 1.0

    def g5(self, x):
        dif = [0, 0]
        v1 = np.cos(x[0] + x[1])
        v2 = 2.0*(x[0] - x[1])

        dif[0] = v1 + v2 - 1.5
        dif[1] = v1 - v2 + 2.5
        return dif

    def fg5(self, x):
        return self.f5(x), self.g5(x)

    def f38(self, x):
        return (100.0 * pow(x[1] - pow(x[0], 2), 2) +
                pow(1.0 - x[0], 2) + 90.0 * pow(x[3] - pow(x[2], 2), 2) +
                pow(1.0 - x[2], 2) + 10.1 * (pow(x[1] - 1.0, 2) +
                                             pow(x[3] - 1.0, 2)) +
                19.8 * (x[1] - 1.0) * (x[3] - 1.0)) * 1.0e-5

    def g38(self, x):
        dif = [0, 0, 0, 0]
        dif[0] = (-400.0 * x[0] * (x[1] - pow(x[0], 2)) -
                  2.0 * (1.0 - x[0])) * 1.0e-5
        dif[1] = (200.0 * (x[1] - pow(x[0], 2)) + 20.2 * (x[1] - 1.0) +
                  19.8 * (x[3] - 1.0)) * 1.0e-5
        dif[2] = (- 360.0 * x[2] * (x[3] - pow(x[2], 2)) -
                  2.0 * (1.0 - x[2])) * 1.0e-5
        dif[3] = (180.0 * (x[3] - pow(x[2], 2)) + 20.2 * (x[3] - 1.0) +
                  19.8 * (x[1] - 1.0)) * 1.0e-5
        return dif

    def fg38(self, x):
        return self.f38(x), self.g38(x)

    def f45(self, x):
        return 2.0 - x[0] * x[1] * x[2] * x[3] * x[4] / 120.0

    def g45(self, x):
        dif = [0] * 5
        dif[0] = - x[1] * x[2] * x[3] * x[4] / 120.0
        dif[1] = - x[0] * x[2] * x[3] * x[4] / 120.0
        dif[2] = - x[0] * x[1] * x[3] * x[4] / 120.0
        dif[3] = - x[0] * x[1] * x[2] * x[4] / 120.0
        dif[4] = - x[0] * x[1] * x[2] * x[3] / 120.0
        return dif

    def fg45(self, x):
        return self.f45(x), self.g45(x)

    # tests
    # minimize with method=TNC
    def test_minimize_tnc1(self):
        x0, bnds = [-2, 1], ([-np.inf, None], [-1.5, None])
        xopt = [1, 1]
        iterx = []  # to test callback

        res = optimize.minimize(self.f1, x0, method='TNC', jac=self.g1,
                                bounds=bnds, options=self.opts,
                                callback=iterx.append)
        assert_allclose(res.fun, self.f1(xopt), atol=1e-8)
        assert_equal(len(iterx), res.nit)

    def test_minimize_tnc1b(self):
        x0, bnds = matrix([-2, 1]), ([-np.inf, None],[-1.5, None])
        xopt = [1, 1]
        x = optimize.minimize(self.f1, x0, method='TNC',
                              bounds=bnds, options=self.opts).x
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-4)

    def test_minimize_tnc1c(self):
        x0, bnds = [-2, 1], ([-np.inf, None],[-1.5, None])
        xopt = [1, 1]
        x = optimize.minimize(self.fg1, x0, method='TNC',
                              jac=True, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8)

    def test_minimize_tnc2(self):
        x0, bnds = [-2, 1], ([-np.inf, None], [1.5, None])
        xopt = [-1.2210262419616387, 1.5]
        x = optimize.minimize(self.f1, x0, method='TNC',
                              jac=self.g1, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8)

    def test_minimize_tnc3(self):
        x0, bnds = [10, 1], ([-np.inf, None], [0.0, None])
        xopt = [0, 0]
        x = optimize.minimize(self.f3, x0, method='TNC',
                              jac=self.g3, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f3(x), self.f3(xopt), atol=1e-8)

    def test_minimize_tnc4(self):
        x0,bnds = [1.125, 0.125], [(1, None), (0, None)]
        xopt = [1, 0]
        x = optimize.minimize(self.f4, x0, method='TNC',
                              jac=self.g4, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f4(x), self.f4(xopt), atol=1e-8)

    def test_minimize_tnc5(self):
        x0, bnds = [0, 0], [(-1.5, 4),(-3, 3)]
        xopt = [-0.54719755119659763, -1.5471975511965976]
        x = optimize.minimize(self.f5, x0, method='TNC',
                              jac=self.g5, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f5(x), self.f5(xopt), atol=1e-8)

    def test_minimize_tnc38(self):
        x0, bnds = np.array([-3, -1, -3, -1]), [(-10, 10)]*4
        xopt = [1]*4
        x = optimize.minimize(self.f38, x0, method='TNC',
                              jac=self.g38, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f38(x), self.f38(xopt), atol=1e-8)

    def test_minimize_tnc45(self):
        x0, bnds = [2] * 5, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        xopt = [1, 2, 3, 4, 5]
        x = optimize.minimize(self.f45, x0, method='TNC',
                              jac=self.g45, bounds=bnds,
                              options=self.opts).x
        assert_allclose(self.f45(x), self.f45(xopt), atol=1e-8)

    # fmin_tnc
    def test_tnc1(self):
        fg, x, bounds = self.fg1, [-2, 1], ([-np.inf, None], [-1.5, None])
        xopt = [1, 1]

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, args=(100.0, ),
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc1b(self):
        x, bounds = [-2, 1], ([-np.inf, None], [-1.5, None])
        xopt = [1, 1]

        x, nf, rc = optimize.fmin_tnc(self.f1, x, approx_grad=True,
                                      bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-4,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc1c(self):
        x, bounds = [-2, 1], ([-np.inf, None], [-1.5, None])
        xopt = [1, 1]

        x, nf, rc = optimize.fmin_tnc(self.f1, x, fprime=self.g1,
                                      bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc2(self):
        fg, x, bounds = self.fg1, [-2, 1], ([-np.inf, None], [1.5, None])
        xopt = [-1.2210262419616387, 1.5]

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc3(self):
        fg, x, bounds = self.fg3, [10, 1], ([-np.inf, None], [0.0, None])
        xopt = [0, 0]

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f3(x), self.f3(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc4(self):
        fg, x, bounds = self.fg4, [1.125, 0.125], [(1, None), (0, None)]
        xopt = [1, 0]

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f4(x), self.f4(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc5(self):
        fg, x, bounds = self.fg5, [0, 0], [(-1.5, 4),(-3, 3)]
        xopt = [-0.54719755119659763, -1.5471975511965976]

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f5(x), self.f5(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc38(self):
        fg, x, bounds = self.fg38, np.array([-3, -1, -3, -1]), [(-10, 10)]*4
        xopt = [1]*4

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f38(x), self.f38(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])

    def test_tnc45(self):
        fg, x, bounds = self.fg45, [2] * 5, [(0, 1), (0, 2), (0, 3),
                                             (0, 4), (0, 5)]
        xopt = [1, 2, 3, 4, 5]

        x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
                                      messages=optimize.tnc.MSG_NONE,
                                      maxfun=200)

        assert_allclose(self.f45(x), self.f45(xopt), atol=1e-8,
                        err_msg="TNC failed with status: " +
                                optimize.tnc.RCSTRINGS[rc])
