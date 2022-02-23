import numpy as np
from numpy.testing import assert_allclose, assert_equal

from scipy.stats._tukeylambda_stats import (tukeylambda_variance,
                                            tukeylambda_kurtosis)


def test_tukeylambda_stats_known_exact():
    """Compare results with some known exact formulas."""
    # Some exact values of the Tukey Lambda variance and kurtosis:
    # lambda   var      kurtosis
    #   0     pi**2/3     6/5     (logistic distribution)
    #  0.5    4 - pi    (5/3 - pi/2)/(pi/4 - 1)**2 - 3
    #   1      1/3       -6/5     (uniform distribution on (-1,1))
    #   2      1/12      -6/5     (uniform distribution on (-1/2, 1/2))

    # lambda = 0
    var = tukeylambda_variance(0)
    assert_allclose(var, np.pi**2 / 3, atol=1e-12)
    kurt = tukeylambda_kurtosis(0)
    assert_allclose(kurt, 1.2, atol=1e-10)

    # lambda = 0.5
    var = tukeylambda_variance(0.5)
    assert_allclose(var, 4 - np.pi, atol=1e-12)
    kurt = tukeylambda_kurtosis(0.5)
    desired = (5./3 - np.pi/2) / (np.pi/4 - 1)**2 - 3
    assert_allclose(kurt, desired, atol=1e-10)

    # lambda = 1
    var = tukeylambda_variance(1)
    assert_allclose(var, 1.0 / 3, atol=1e-12)
    kurt = tukeylambda_kurtosis(1)
    assert_allclose(kurt, -1.2, atol=1e-10)

    # lambda = 2
    var = tukeylambda_variance(2)
    assert_allclose(var, 1.0 / 12, atol=1e-12)
    kurt = tukeylambda_kurtosis(2)
    assert_allclose(kurt, -1.2, atol=1e-10)


def test_tukeylambda_stats_mpmath():
    """Compare results with some values that were computed using mpmath."""
    a10 = dict(atol=1e-10, rtol=0)
    a12 = dict(atol=1e-12, rtol=0)
    data = [
        # lambda        variance              kurtosis
        [-0.1, 4.78050217874253547, 3.78559520346454510],
        [-0.0649, 4.16428023599895777, 2.52019675947435718],
        [-0.05, 3.93672267890775277, 2.13129793057777277],
        [-0.001, 3.30128380390964882, 1.21452460083542988],
        [0.001, 3.27850775649572176, 1.18560634779287585],
        [0.03125, 2.95927803254615800, 0.804487555161819980],
        [0.05, 2.78281053405464501, 0.611604043886644327],
        [0.0649, 2.65282386754100551, 0.476834119532774540],
        [1.2, 0.242153920578588346, -1.23428047169049726],
        [10.0, 0.00095237579757703597, 2.37810697355144933],
        [20.0, 0.00012195121951131043, 7.37654321002709531],
    ]

    for lam, var_expected, kurt_expected in data:
        var = tukeylambda_variance(lam)
        assert_allclose(var, var_expected, **a12)
        kurt = tukeylambda_kurtosis(lam)
        assert_allclose(kurt, kurt_expected, **a10)

    # Test with vector arguments (most of the other tests are for single
    # values).
    lam, var_expected, kurt_expected = zip(*data)
    var = tukeylambda_variance(lam)
    assert_allclose(var, var_expected, **a12)
    kurt = tukeylambda_kurtosis(lam)
    assert_allclose(kurt, kurt_expected, **a10)


def test_tukeylambda_stats_invalid():
    """Test values of lambda outside the domains of the functions."""
    lam = [-1.0, -0.5]
    var = tukeylambda_variance(lam)
    assert_equal(var, np.array([np.nan, np.inf]))

    lam = [-1.0, -0.25]
    kurt = tukeylambda_kurtosis(lam)
    assert_equal(kurt, np.array([np.nan, np.inf]))

