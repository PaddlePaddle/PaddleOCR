import numpy as np
from numpy.testing import assert_equal, assert_allclose, suppress_warnings

from scipy.special._ufuncs import _sinpi as sinpi
from scipy.special._ufuncs import _cospi as cospi


def test_integer_real_part():
    x = np.arange(-100, 101)
    y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
    x, y = np.meshgrid(x, y)
    z = x + 1j*y
    # In the following we should be *exactly* right
    res = sinpi(z)
    assert_equal(res.real, 0.0)
    res = cospi(z)
    assert_equal(res.imag, 0.0)


def test_half_integer_real_part():
    x = np.arange(-100, 101) + 0.5
    y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
    x, y = np.meshgrid(x, y)
    z = x + 1j*y
    # In the following we should be *exactly* right
    res = sinpi(z)
    assert_equal(res.imag, 0.0)
    res = cospi(z)
    assert_equal(res.real, 0.0)


def test_intermediate_overlow():
    # Make sure we avoid overflow in situations where cosh/sinh would
    # overflow but the product with sin/cos would not
    sinpi_pts = [complex(1 + 1e-14, 227),
                 complex(1e-35, 250),
                 complex(1e-301, 445)]
    # Data generated with mpmath
    sinpi_std = [complex(-8.113438309924894e+295, -np.inf),
                 complex(1.9507801934611995e+306, np.inf),
                 complex(2.205958493464539e+306, np.inf)]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "invalid value encountered in multiply")
        for p, std in zip(sinpi_pts, sinpi_std):
            assert_allclose(sinpi(p), std)

    # Test for cosine, less interesting because cos(0) = 1.
    p = complex(0.5 + 1e-14, 227)
    std = complex(-8.113438309924894e+295, -np.inf)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "invalid value encountered in multiply")
        assert_allclose(cospi(p), std)


def test_zero_sign():
    y = sinpi(-0.0)
    assert y == 0.0
    assert np.signbit(y)

    y = sinpi(0.0)
    assert y == 0.0
    assert not np.signbit(y)

    y = cospi(0.5)
    assert y == 0.0
    assert not np.signbit(y)
