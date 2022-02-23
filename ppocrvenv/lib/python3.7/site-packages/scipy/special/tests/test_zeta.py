import scipy.special as sc
import numpy as np
from numpy.testing import assert_equal, assert_allclose


def test_zeta():
    assert_allclose(sc.zeta(2,2), np.pi**2/6 - 1, rtol=1e-12)


def test_zetac():
    # Expected values in the following were computed using Wolfram
    # Alpha's `Zeta[x] - 1`
    x = [-2.1, 0.8, 0.9999, 9, 50, 75]
    desired = [
        -0.9972705002153750,
        -5.437538415895550,
        -10000.42279161673,
        0.002008392826082214,
        8.881784210930816e-16,
        2.646977960169853e-23,
    ]
    assert_allclose(sc.zetac(x), desired, rtol=1e-12)


def test_zetac_special_cases():
    assert sc.zetac(np.inf) == 0
    assert np.isnan(sc.zetac(-np.inf))
    assert sc.zetac(0) == -1.5
    assert sc.zetac(1.0) == np.inf

    assert_equal(sc.zetac([-2, -50, -100]), -1)


def test_riemann_zeta_special_cases():
    assert np.isnan(sc.zeta(np.nan))
    assert sc.zeta(np.inf) == 1
    assert sc.zeta(0) == -0.5

    # Riemann zeta is zero add negative even integers.
    assert_equal(sc.zeta([-2, -4, -6, -8, -10]), 0)

    assert_allclose(sc.zeta(2), np.pi**2/6, rtol=1e-12)
    assert_allclose(sc.zeta(4), np.pi**4/90, rtol=1e-12)


def test_riemann_zeta_avoid_overflow():
    s = -260.00000000001
    desired = -5.6966307844402683127e+297  # Computed with Mpmath
    assert_allclose(sc.zeta(s), desired, atol=0, rtol=5e-14)
