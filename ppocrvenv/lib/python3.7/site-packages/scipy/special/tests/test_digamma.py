import numpy as np
from numpy import pi, log, sqrt
from numpy.testing import assert_, assert_equal

from scipy.special._testutils import FuncData
import scipy.special as sc

# Euler-Mascheroni constant
euler = 0.57721566490153286


def test_consistency():
    # Make sure the implementation of digamma for real arguments
    # agrees with the implementation of digamma for complex arguments.

    # It's all poles after -1e16
    x = np.r_[-np.logspace(15, -30, 200), np.logspace(-30, 300, 200)]
    dataset = np.vstack((x + 0j, sc.digamma(x))).T
    FuncData(sc.digamma, dataset, 0, 1, rtol=5e-14, nan_ok=True).check()


def test_special_values():
    # Test special values from Gauss's digamma theorem. See
    #
    # https://en.wikipedia.org/wiki/Digamma_function

    dataset = [(1, -euler),
               (0.5, -2*log(2) - euler),
               (1/3, -pi/(2*sqrt(3)) - 3*log(3)/2 - euler),
               (1/4, -pi/2 - 3*log(2) - euler),
               (1/6, -pi*sqrt(3)/2 - 2*log(2) - 3*log(3)/2 - euler),
               (1/8, -pi/2 - 4*log(2) - (pi + log(2 + sqrt(2)) - log(2 - sqrt(2)))/sqrt(2) - euler)]

    dataset = np.asarray(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()


def test_nonfinite():
    pts = [0.0, -0.0, np.inf]
    std = [-np.inf, np.inf, np.inf]
    assert_equal(sc.digamma(pts), std)
    assert_(all(np.isnan(sc.digamma([-np.inf, -1]))))
