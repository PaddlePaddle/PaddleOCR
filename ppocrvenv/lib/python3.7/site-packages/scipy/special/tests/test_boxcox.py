import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from scipy.special import boxcox, boxcox1p, inv_boxcox, inv_boxcox1p


# There are more tests of boxcox and boxcox1p in test_mpmath.py.

def test_boxcox_basic():
    x = np.array([0.5, 1, 2, 4])

    # lambda = 0  =>  y = log(x)
    y = boxcox(x, 0)
    assert_almost_equal(y, np.log(x))

    # lambda = 1  =>  y = x - 1
    y = boxcox(x, 1)
    assert_almost_equal(y, x - 1)

    # lambda = 2  =>  y = 0.5*(x**2 - 1)
    y = boxcox(x, 2)
    assert_almost_equal(y, 0.5*(x**2 - 1))

    # x = 0 and lambda > 0  =>  y = -1 / lambda
    lam = np.array([0.5, 1, 2])
    y = boxcox(0, lam)
    assert_almost_equal(y, -1.0 / lam)

def test_boxcox_underflow():
    x = 1 + 1e-15
    lmbda = 1e-306
    y = boxcox(x, lmbda)
    assert_allclose(y, np.log(x), rtol=1e-14)


def test_boxcox_nonfinite():
    # x < 0  =>  y = nan
    x = np.array([-1, -1, -0.5])
    y = boxcox(x, [0.5, 2.0, -1.5])
    assert_equal(y, np.array([np.nan, np.nan, np.nan]))

    # x = 0 and lambda <= 0  =>  y = -inf
    x = 0
    y = boxcox(x, [-2.5, 0])
    assert_equal(y, np.array([-np.inf, -np.inf]))


def test_boxcox1p_basic():
    x = np.array([-0.25, -1e-20, 0, 1e-20, 0.25, 1, 3])

    # lambda = 0  =>  y = log(1+x)
    y = boxcox1p(x, 0)
    assert_almost_equal(y, np.log1p(x))

    # lambda = 1  =>  y = x
    y = boxcox1p(x, 1)
    assert_almost_equal(y, x)

    # lambda = 2  =>  y = 0.5*((1+x)**2 - 1) = 0.5*x*(2 + x)
    y = boxcox1p(x, 2)
    assert_almost_equal(y, 0.5*x*(2 + x))

    # x = -1 and lambda > 0  =>  y = -1 / lambda
    lam = np.array([0.5, 1, 2])
    y = boxcox1p(-1, lam)
    assert_almost_equal(y, -1.0 / lam)


def test_boxcox1p_underflow():
    x = np.array([1e-15, 1e-306])
    lmbda = np.array([1e-306, 1e-18])
    y = boxcox1p(x, lmbda)
    assert_allclose(y, np.log1p(x), rtol=1e-14)


def test_boxcox1p_nonfinite():
    # x < -1  =>  y = nan
    x = np.array([-2, -2, -1.5])
    y = boxcox1p(x, [0.5, 2.0, -1.5])
    assert_equal(y, np.array([np.nan, np.nan, np.nan]))

    # x = -1 and lambda <= 0  =>  y = -inf
    x = -1
    y = boxcox1p(x, [-2.5, 0])
    assert_equal(y, np.array([-np.inf, -np.inf]))


def test_inv_boxcox():
    x = np.array([0., 1., 2.])
    lam = np.array([0., 1., 2.])
    y = boxcox(x, lam)
    x2 = inv_boxcox(y, lam)
    assert_almost_equal(x, x2)

    x = np.array([0., 1., 2.])
    lam = np.array([0., 1., 2.])
    y = boxcox1p(x, lam)
    x2 = inv_boxcox1p(y, lam)
    assert_almost_equal(x, x2)


def test_inv_boxcox1p_underflow():
    x = 1e-15
    lam = 1e-306
    y = inv_boxcox1p(x, lam)
    assert_allclose(y, x, rtol=1e-14)

