import numpy as np
import scipy.special as sc
from numpy.testing import assert_almost_equal, assert_array_equal


class TestPdtr:
    def test(self):
        val = sc.pdtr(0, 1)
        assert_almost_equal(val, np.exp(-1))

    def test_m_zero(self):
        val = sc.pdtr([0, 1, 2], 0)
        assert_array_equal(val, [1, 1, 1])

    def test_rounding(self):
        double_val = sc.pdtr([0.1, 1.1, 2.1], 1.0)
        int_val = sc.pdtr([0, 1, 2], 1.0)
        assert_array_equal(double_val, int_val)

    def test_inf(self):
        val = sc.pdtr(np.inf, 1.0)
        assert_almost_equal(val, 1.0)

    def test_domain(self):
        val = sc.pdtr(-1.1, 1.0)
        assert np.isnan(val)

class TestPdtrc:
    def test_value(self):
        val = sc.pdtrc(0, 1)
        assert_almost_equal(val, 1 - np.exp(-1))

    def test_m_zero(self):
        val = sc.pdtrc([0, 1, 2], 0.0)
        assert_array_equal(val, [0, 0, 0])

    def test_rounding(self):
        double_val = sc.pdtrc([0.1, 1.1, 2.1], 1.0)
        int_val = sc.pdtrc([0, 1, 2], 1.0)
        assert_array_equal(double_val, int_val)

    def test_inf(self):
        val = sc.pdtrc(np.inf, 1.0)
        assert_almost_equal(val, 0.0)

    def test_domain(self):
        val = sc.pdtrc(-1.1, 1.0)
        assert np.isnan(val)
