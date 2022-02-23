import numpy as np
from numpy.testing import assert_equal, assert_allclose

import scipy.special as sc


def test_symmetries():
    np.random.seed(1234)
    a, h = np.random.rand(100), np.random.rand(100)
    assert_equal(sc.owens_t(h, a), sc.owens_t(-h, a))
    assert_equal(sc.owens_t(h, a), -sc.owens_t(h, -a))


def test_special_cases():
    assert_equal(sc.owens_t(5, 0), 0)
    assert_allclose(sc.owens_t(0, 5), 0.5*np.arctan(5)/np.pi,
                    rtol=5e-14)
    # Target value is 0.5*Phi(5)*(1 - Phi(5)) for Phi the CDF of the
    # standard normal distribution
    assert_allclose(sc.owens_t(5, 1), 1.4332574485503512543e-07,
                    rtol=5e-14)


def test_nans():
    assert_equal(sc.owens_t(20, np.nan), np.nan)
    assert_equal(sc.owens_t(np.nan, 20), np.nan)
    assert_equal(sc.owens_t(np.nan, np.nan), np.nan)


def test_infs():
    h = 1
    res = 0.5*sc.erfc(h/np.sqrt(2))
    assert_allclose(sc.owens_t(h, np.inf), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -np.inf), -res, rtol=5e-14)

    assert_equal(sc.owens_t(np.inf, 1), 0)
    assert_equal(sc.owens_t(-np.inf, 1), 0)

    assert_equal(sc.owens_t(np.inf, np.inf), 0)
    assert_equal(sc.owens_t(-np.inf, np.inf), 0)
    assert_equal(sc.owens_t(np.inf, -np.inf), -0.0)
    assert_equal(sc.owens_t(-np.inf, -np.inf), -0.0)
