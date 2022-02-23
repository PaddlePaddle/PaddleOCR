import numpy as np
import scipy.special as sc
import pytest
from numpy.testing import assert_allclose, assert_array_equal, suppress_warnings


class TestBdtr:
    def test(self):
        val = sc.bdtr(0, 1, 0.5)
        assert_allclose(val, 0.5)

    def test_sum_is_one(self):
        val = sc.bdtr([0, 1, 2], 2, 0.5)
        assert_array_equal(val, [0.25, 0.75, 1.0])

    def test_rounding(self):
        double_val = sc.bdtr([0.1, 1.1, 2.1], 2, 0.5)
        int_val = sc.bdtr([0, 1, 2], 2, 0.5)
        assert_array_equal(double_val, int_val)

    @pytest.mark.parametrize('k, n, p', [
        (np.inf, 2, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 2, np.inf)
    ])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            val = sc.bdtr(k, n, p)
        assert np.isnan(val)

    def test_domain(self):
        val = sc.bdtr(-1.1, 1, 0.5)
        assert np.isnan(val)


class TestBdtrc:
    def test_value(self):
        val = sc.bdtrc(0, 1, 0.5)
        assert_allclose(val, 0.5)

    def test_sum_is_one(self):
        val = sc.bdtrc([0, 1, 2], 2, 0.5)
        assert_array_equal(val, [0.75, 0.25, 0.0])

    def test_rounding(self):
        double_val = sc.bdtrc([0.1, 1.1, 2.1], 2, 0.5)
        int_val = sc.bdtrc([0, 1, 2], 2, 0.5)
        assert_array_equal(double_val, int_val)

    @pytest.mark.parametrize('k, n, p', [
        (np.inf, 2, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 2, np.inf)
    ])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            val = sc.bdtrc(k, n, p)
        assert np.isnan(val)

    def test_domain(self):
        val = sc.bdtrc(-1.1, 1, 0.5)
        val2 = sc.bdtrc(2.1, 1, 0.5)
        assert np.isnan(val2)
        assert_allclose(val, 1.0)

    def test_bdtr_bdtrc_sum_to_one(self):
        bdtr_vals = sc.bdtr([0, 1, 2], 2, 0.5)
        bdtrc_vals = sc.bdtrc([0, 1, 2], 2, 0.5)
        vals = bdtr_vals + bdtrc_vals
        assert_allclose(vals, [1.0, 1.0, 1.0])


class TestBdtri:
    def test_value(self):
        val = sc.bdtri(0, 1, 0.5)
        assert_allclose(val, 0.5)

    def test_sum_is_one(self):
        val = sc.bdtri([0, 1], 2, 0.5)
        actual = np.asarray([1 - 1/np.sqrt(2), 1/np.sqrt(2)])
        assert_allclose(val, actual)

    def test_rounding(self):
        double_val = sc.bdtri([0.1, 1.1], 2, 0.5)
        int_val = sc.bdtri([0, 1], 2, 0.5)
        assert_allclose(double_val, int_val)

    @pytest.mark.parametrize('k, n, p', [
        (np.inf, 2, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 2, np.inf)
    ])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            val = sc.bdtri(k, n, p)
        assert np.isnan(val)

    @pytest.mark.parametrize('k, n, p', [
        (-1.1, 1, 0.5),
        (2.1, 1, 0.5)
    ])
    def test_domain(self, k, n, p):
        val = sc.bdtri(k, n, p)
        assert np.isnan(val)

    def test_bdtr_bdtri_roundtrip(self):
        bdtr_vals = sc.bdtr([0, 1, 2], 2, 0.5)
        roundtrip_vals = sc.bdtri([0, 1, 2], 2, bdtr_vals)
        assert_allclose(roundtrip_vals, [0.5, 0.5, np.nan])
