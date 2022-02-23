#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_allclose, assert_, assert_raises

import pywt


def test_upcoef_reconstruct():
    data = np.arange(3)
    a = pywt.downcoef('a', data, 'haar')
    d = pywt.downcoef('d', data, 'haar')

    rec = (pywt.upcoef('a', a, 'haar', take=3) +
           pywt.upcoef('d', d, 'haar', take=3))
    assert_allclose(rec, data)


def test_downcoef_multilevel():
    rstate = np.random.RandomState(1234)
    r = rstate.randn(16)
    nlevels = 3
    # calling with level=1 nlevels times
    a1 = r.copy()
    for i in range(nlevels):
        a1 = pywt.downcoef('a', a1, 'haar', level=1)
    # call with level=nlevels once
    a3 = pywt.downcoef('a', r, 'haar', level=nlevels)
    assert_allclose(a1, a3)


def test_downcoef_complex():
    rstate = np.random.RandomState(1234)
    r = rstate.randn(16) + 1j * rstate.randn(16)
    nlevels = 3
    a = pywt.downcoef('a', r, 'haar', level=nlevels)
    a_ref = pywt.downcoef('a', r.real, 'haar', level=nlevels)
    a_ref = a_ref + 1j * pywt.downcoef('a', r.imag, 'haar', level=nlevels)
    assert_allclose(a, a_ref)


def test_downcoef_errs():
    # invalid part string (not 'a' or 'd')
    assert_raises(ValueError, pywt.downcoef, 'f', np.ones(16), 'haar')


def test_compare_downcoef_coeffs():
    rstate = np.random.RandomState(1234)
    r = rstate.randn(16)
    # compare downcoef against wavedec outputs
    for nlevels in [1, 2, 3]:
        for wavelet in pywt.wavelist():
            if wavelet in ['cmor', 'shan', 'fbsp']:
                # skip these CWT families to avoid warnings
                continue
            wavelet = pywt.DiscreteContinuousWavelet(wavelet)
            if isinstance(wavelet, pywt.Wavelet):
                max_level = pywt.dwt_max_level(r.size, wavelet.dec_len)
                if nlevels <= max_level:
                    a = pywt.downcoef('a', r, wavelet, level=nlevels)
                    d = pywt.downcoef('d', r, wavelet, level=nlevels)
                    coeffs = pywt.wavedec(r, wavelet, level=nlevels)
                    assert_allclose(a, coeffs[0])
                    assert_allclose(d, coeffs[1])


def test_upcoef_multilevel():
    rstate = np.random.RandomState(1234)
    r = rstate.randn(4)
    nlevels = 3
    # calling with level=1 nlevels times
    a1 = r.copy()
    for i in range(nlevels):
        a1 = pywt.upcoef('a', a1, 'haar', level=1)
    # call with level=nlevels once
    a3 = pywt.upcoef('a', r, 'haar', level=nlevels)
    assert_allclose(a1, a3)


def test_upcoef_complex():
    rstate = np.random.RandomState(1234)
    r = rstate.randn(4) + 1j*rstate.randn(4)
    nlevels = 3
    a = pywt.upcoef('a', r, 'haar', level=nlevels)
    a_ref = pywt.upcoef('a', r.real, 'haar', level=nlevels)
    a_ref = a_ref + 1j*pywt.upcoef('a', r.imag, 'haar', level=nlevels)
    assert_allclose(a, a_ref)


def test_upcoef_errs():
    # invalid part string (not 'a' or 'd')
    assert_raises(ValueError, pywt.upcoef, 'f', np.ones(4), 'haar')


def test_upcoef_and_downcoef_1d_only():
    # upcoef and downcoef raise a ValueError if data.ndim > 1d
    for ndim in [2, 3]:
        data = np.ones((8, )*ndim)
        assert_raises(ValueError, pywt.downcoef, 'a', data, 'haar')
        assert_raises(ValueError, pywt.upcoef, 'a', data, 'haar')


def test_wavelet_repr():
    from pywt._extensions import _pywt
    wavelet = _pywt.Wavelet('sym8')

    repr_wavelet = eval(wavelet.__repr__())

    assert_(wavelet.__repr__() == repr_wavelet.__repr__())


def test_dwt_max_level():
    assert_(pywt.dwt_max_level(16, 2) == 4)
    assert_(pywt.dwt_max_level(16, 8) == 1)
    assert_(pywt.dwt_max_level(16, 9) == 1)
    assert_(pywt.dwt_max_level(16, 10) == 0)
    assert_(pywt.dwt_max_level(16, np.int8(10)) == 0)
    assert_(pywt.dwt_max_level(16, 10.) == 0)
    assert_(pywt.dwt_max_level(16, 18) == 0)

    # accepts discrete Wavelet object or string as well
    assert_(pywt.dwt_max_level(32, pywt.Wavelet('sym5')) == 1)
    assert_(pywt.dwt_max_level(32, 'sym5') == 1)

    # string input that is not a discrete wavelet
    assert_raises(ValueError, pywt.dwt_max_level, 16, 'mexh')

    # filter_len must be an integer >= 2
    assert_raises(ValueError, pywt.dwt_max_level, 16, 1)
    assert_raises(ValueError, pywt.dwt_max_level, 16, -1)
    assert_raises(ValueError, pywt.dwt_max_level, 16, 3.3)


def test_ContinuousWavelet_errs():
    assert_raises(ValueError, pywt.ContinuousWavelet, 'qwertz')


def test_ContinuousWavelet_repr():
    from pywt._extensions import _pywt
    wavelet = _pywt.ContinuousWavelet('gaus2')

    repr_wavelet = eval(wavelet.__repr__())

    assert_(wavelet.__repr__() == repr_wavelet.__repr__())


def test_wavelist():
    for name in pywt.wavelist(family='coif'):
        assert_(name.startswith('coif'))

    assert_('cgau7' in pywt.wavelist(kind='continuous'))
    assert_('sym20' in pywt.wavelist(kind='discrete'))
    assert_(len(pywt.wavelist(kind='continuous')) +
            len(pywt.wavelist(kind='discrete')) ==
            len(pywt.wavelist(kind='all')))

    assert_raises(ValueError, pywt.wavelist, kind='foobar')


def test_wavelet_errormsgs():
    try:
        pywt.Wavelet('gaus1')
    except ValueError as e:
        assert_(e.args[0].startswith('The `Wavelet` class'))
    try:
        pywt.Wavelet('cmord')
    except ValueError as e:
        assert_(e.args[0] == "Invalid wavelet name 'cmord'.")
