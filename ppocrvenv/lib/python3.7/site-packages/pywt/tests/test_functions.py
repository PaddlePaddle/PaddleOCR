#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_almost_equal, assert_allclose

import pywt


def test_centrfreq():
    # db1 is Haar function, frequency=1
    w = pywt.Wavelet('db1')
    expected = 1
    result = pywt.central_frequency(w, precision=12)
    assert_almost_equal(result, expected, decimal=3)
    # db2, frequency=2/3
    w = pywt.Wavelet('db2')
    expected = 2/3.
    result = pywt.central_frequency(w, precision=12)
    assert_almost_equal(result, expected)


def test_scal2frq_scale():
    scale = 2
    w = pywt.Wavelet('db1')
    expected = 1. / scale
    result = pywt.scale2frequency(w, scale, precision=12)
    assert_almost_equal(result, expected, decimal=3)


def test_intwave_orthogonal():
    w = pywt.Wavelet('db1')
    int_psi, x = pywt.integrate_wavelet(w, precision=12)
    ix = x < 0.5
    # For x < 0.5, the integral is equal to x
    assert_allclose(int_psi[ix], x[ix])
    # For x > 0.5, the integral is equal to (1 - x)
    # Ignore last point here, there x > 1 and something goes wrong
    assert_allclose(int_psi[~ix][:-1], 1 - x[~ix][:-1], atol=1e-10)
