# Reference MPMATH implementation:
#
# import mpmath
# from mpmath import nsum
#
# def Wright_Series_MPMATH(a, b, z, dps=50, method='r+s+e', steps=[1000]):
#    """Compute Wright' generalized Bessel function as Series.
#
#    This uses mpmath for arbitrary precision.
#    """
#    with mpmath.workdps(dps):
#        res = nsum(lambda k: z**k/mpmath.fac(k) * mpmath.rgamma(a*k+b),
#                          [0, mpmath.inf],
#                          tol=dps, method=method, steps=steps
#                          )
#
#    return res

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

import scipy.special as sc
from scipy.special import rgamma, wright_bessel


@pytest.mark.parametrize('a', [0, 1e-6, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [0, 1e-6, 0.1, 0.5, 1, 10])
def test_wright_bessel_zero(a, b):
    """Test at x = 0."""
    assert_equal(wright_bessel(a, b, 0.), rgamma(b))


@pytest.mark.parametrize('b', [0, 1e-6, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('x', [0, 1e-6, 0.1, 0.5, 1])
def test_wright_bessel_iv(b, x):
    """Test relation of wright_bessel and modified bessel function iv.

    iv(z) = (1/2*z)**v * Phi(1, v+1; 1/4*z**2).
    See https://dlmf.nist.gov/10.46.E2
    """
    if x != 0:
        v = b - 1
        wb = wright_bessel(1, v + 1, x**2 / 4.)
        # Note: iv(v, x) has precision of less than 1e-12 for some cases
        # e.g v=1-1e-6 and x=1e-06)
        assert_allclose(np.power(x / 2., v) * wb,
                        sc.iv(v, x),
                        rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('a', [0, 1e-6, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [1, 1 + 1e-3, 2, 5, 10])
@pytest.mark.parametrize('x', [0, 1e-6, 0.1, 0.5, 1, 5, 10, 100])
def test_wright_functional(a, b, x):
    """Test functional relation of wright_bessel.

    Phi(a, b-1, z) = a*z*Phi(a, b+a, z) + (b-1)*Phi(a, b, z)

    Note that d/dx Phi(a, b, x) = Phi(a, b-1, x)
    See Eq. (22) of
    B. Stankovic, On the Function of E. M. Wright,
    Publ. de l' Institut Mathematique, Beograd,
    Nouvelle S`er. 10 (1970), 113-124.
    """
    assert_allclose(wright_bessel(a, b - 1, x),
                    a * x * wright_bessel(a, b + a, x)
                    + (b - 1) * wright_bessel(a, b, x),
                    rtol=1e-8, atol=1e-8)


# grid of rows [a, b, x, value, accuracy] that do not reach 1e-11 accuracy
# see output of:
# cd scipy/scipy/_precompute
# python wright_bessel_data.py
grid_a_b_x_value_acc = np.array([
    [0.1, 100.0, 709.7827128933841, 8.026353022981087e+34, 2e-8],
    [0.5, 10.0, 709.7827128933841, 2.680788404494657e+48, 9e-8],
    [0.5, 10.0, 1000.0, 2.005901980702872e+64, 1e-8],
    [0.5, 100.0, 1000.0, 3.4112367580445246e-117, 6e-8],
    [1.0, 20.0, 100000.0, 1.7717158630699857e+225, 3e-11],
    [1.0, 100.0, 100000.0, 1.0269334596230763e+22, np.nan],
    [1.0000000000000222, 20.0, 100000.0, 1.7717158630001672e+225, 3e-11],
    [1.0000000000000222, 100.0, 100000.0, 1.0269334595866202e+22, np.nan],
    [1.5, 0.0, 500.0, 15648961196.432373, 3e-11],
    [1.5, 2.220446049250313e-14, 500.0, 15648961196.431465, 3e-11],
    [1.5, 1e-10, 500.0, 15648961192.344728, 3e-11],
    [1.5, 1e-05, 500.0, 15648552437.334162, 3e-11],
    [1.5, 0.1, 500.0, 12049870581.10317, 2e-11],
    [1.5, 20.0, 100000.0, 7.81930438331405e+43, 3e-9],
    [1.5, 100.0, 100000.0, 9.653370857459075e-130, np.nan],
    ])


@pytest.mark.xfail
@pytest.mark.parametrize(
    'a, b, x, phi',
    grid_a_b_x_value_acc[:, :4].tolist())
def test_wright_data_grid_failures(a, b, x, phi):
    """Test cases of test_data that do not reach relative accuracy of 1e-11"""
    assert_allclose(wright_bessel(a, b, x), phi, rtol=1e-11)


@pytest.mark.parametrize(
    'a, b, x, phi, accuracy',
    grid_a_b_x_value_acc.tolist())
def test_wright_data_grid_less_accurate(a, b, x, phi, accuracy):
    """Test cases of test_data that do not reach relative accuracy of 1e-11

    Here we test for reduced accuracy or even nan.
    """
    if np.isnan(accuracy):
        assert np.isnan(wright_bessel(a, b, x))
    else:
        assert_allclose(wright_bessel(a, b, x), phi, rtol=accuracy)
