import warnings

import numpy as np
from numpy.testing import assert_warns, assert_array_equal

import pywt


def test_intwave_deprecation():
    wavelet = pywt.Wavelet('db3')
    assert_warns(DeprecationWarning, pywt.intwave, wavelet)


def test_centrfrq_deprecation():
    wavelet = pywt.Wavelet('db3')
    assert_warns(DeprecationWarning, pywt.centrfrq, wavelet)


def test_scal2frq_deprecation():
    wavelet = pywt.Wavelet('db3')
    assert_warns(DeprecationWarning, pywt.scal2frq, wavelet, 1)


def test_orthfilt_deprecation():
    assert_warns(DeprecationWarning, pywt.orthfilt, range(6))


def test_integrate_wave_tuple():
    sig = [0, 1, 2, 3]
    xgrid = [0, 1, 2, 3]
    assert_warns(DeprecationWarning, pywt.integrate_wavelet, (sig, xgrid))


old_modes = ['zpd',
             'cpd',
             'sym',
             'ppd',
             'sp1',
             'per',
             ]


def test_MODES_from_object_deprecation():
    for mode in old_modes:
        assert_warns(DeprecationWarning, pywt.Modes.from_object, mode)


def test_MODES_attributes_deprecation():
    def get_mode(Modes, name):
        return getattr(Modes, name)

    for mode in old_modes:
        assert_warns(DeprecationWarning, get_mode, pywt.Modes, mode)


def test_MODES_deprecation_new():
    def use_MODES_new():
        return pywt.MODES.symmetric

    assert_warns(DeprecationWarning, use_MODES_new)


def test_MODES_deprecation_old():
    def use_MODES_old():
        return pywt.MODES.sym

    assert_warns(DeprecationWarning, use_MODES_old)


def test_MODES_deprecation_getattr():
    def use_MODES_new():
        return getattr(pywt.MODES, 'symmetric')

    assert_warns(DeprecationWarning, use_MODES_new)


def test_mode_equivalence():
    old_new = [('zpd', 'zero'),
               ('cpd', 'constant'),
               ('sym', 'symmetric'),
               ('ppd', 'periodic'),
               ('sp1', 'smooth'),
               ('per', 'periodization')]
    x = np.arange(8.)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        for old, new in old_new:
            assert_array_equal(pywt.dwt(x, 'db2', mode=old),
                               pywt.dwt(x, 'db2', mode=new))
