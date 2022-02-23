import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
        assert_allclose)
from scipy.special import logit, expit


class TestLogit:
    def check_logit_out(self, dtype, expected):
        a = np.linspace(0,1,10)
        a = np.array(a, dtype=dtype)
        with np.errstate(divide='ignore'):
            actual = logit(a)

        assert_almost_equal(actual, expected)

        assert_equal(actual.dtype, np.dtype(dtype))

    def test_float32(self):
        expected = np.array([-np.inf, -2.07944155,
                            -1.25276291, -0.69314718,
                            -0.22314353, 0.22314365,
                            0.6931473, 1.25276303,
                            2.07944155, np.inf], dtype=np.float32)
        self.check_logit_out('f4', expected)

    def test_float64(self):
        expected = np.array([-np.inf, -2.07944154,
                            -1.25276297, -0.69314718,
                            -0.22314355, 0.22314355,
                            0.69314718, 1.25276297,
                            2.07944154, np.inf])
        self.check_logit_out('f8', expected)

    def test_nan(self):
        expected = np.array([np.nan]*4)
        with np.errstate(invalid='ignore'):
            actual = logit(np.array([-3., -2., 2., 3.]))

        assert_equal(expected, actual)


class TestExpit:
    def check_expit_out(self, dtype, expected):
        a = np.linspace(-4,4,10)
        a = np.array(a, dtype=dtype)
        actual = expit(a)
        assert_almost_equal(actual, expected)
        assert_equal(actual.dtype, np.dtype(dtype))

    def test_float32(self):
        expected = np.array([0.01798621, 0.04265125,
                            0.09777259, 0.20860852,
                            0.39068246, 0.60931754,
                            0.79139149, 0.9022274,
                            0.95734876, 0.98201376], dtype=np.float32)
        self.check_expit_out('f4',expected)

    def test_float64(self):
        expected = np.array([0.01798621, 0.04265125,
                            0.0977726, 0.20860853,
                            0.39068246, 0.60931754,
                            0.79139147, 0.9022274,
                            0.95734875, 0.98201379])
        self.check_expit_out('f8', expected)

    def test_large(self):
        for dtype in (np.float32, np.float64, np.longdouble):
            for n in (88, 89, 709, 710, 11356, 11357):
                n = np.array(n, dtype=dtype)
                assert_allclose(expit(n), 1.0, atol=1e-20)
                assert_allclose(expit(-n), 0.0, atol=1e-20)
                assert_equal(expit(n).dtype, dtype)
                assert_equal(expit(-n).dtype, dtype)
