"""Regression tests for optimize.

"""
import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises as assert_raises

import scipy.optimize


class TestRegression:

    def test_newton_x0_is_0(self):
        # Regression test for gh-1601
        tgt = 1
        res = scipy.optimize.newton(lambda x: x - 1, 0)
        assert_almost_equal(res, tgt)

    def test_newton_integers(self):
        # Regression test for gh-1741
        root = scipy.optimize.newton(lambda x: x**2 - 1, x0=2,
                                    fprime=lambda x: 2*x)
        assert_almost_equal(root, 1.0)

    def test_lmdif_errmsg(self):
        # This shouldn't cause a crash on Python 3
        class SomeError(Exception):
            pass
        counter = [0]

        def func(x):
            counter[0] += 1
            if counter[0] < 3:
                return x**2 - np.array([9, 10, 11])
            else:
                raise SomeError()
        assert_raises(SomeError,
                      scipy.optimize.leastsq,
                      func, [1, 2, 3])

