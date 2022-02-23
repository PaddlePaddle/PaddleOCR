#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

import pywt


def test_available_modes():
    modes = ['zero', 'constant', 'symmetric', 'periodic', 'smooth',
             'periodization', 'reflect', 'antisymmetric', 'antireflect']
    assert_equal(pywt.Modes.modes, modes)
    assert_equal(pywt.Modes.from_object('constant'), 2)


def test_invalid_modes():
    x = np.arange(4)
    assert_raises(ValueError, pywt.dwt, x, 'db2', 'unknown')
    assert_raises(ValueError, pywt.dwt, x, 'db2', -1)
    assert_raises(ValueError, pywt.dwt, x, 'db2', 9)
    assert_raises(TypeError, pywt.dwt, x, 'db2', None)

    assert_raises(ValueError, pywt.Modes.from_object, 'unknown')
    assert_raises(ValueError, pywt.Modes.from_object, -1)
    assert_raises(ValueError, pywt.Modes.from_object, 9)
    assert_raises(TypeError, pywt.Modes.from_object, None)


def test_dwt_idwt_allmodes():
    # Test that :func:`dwt` and :func:`idwt` can be performed using every mode
    x = [1, 2, 1, 5, -1, 8, 4, 6]
    dwt_results = {
        'zero': ([-0.03467518, 1.73309178, 3.40612438, 6.32928585, 6.95094948],
                 [-0.12940952, -2.15599552, -5.95034847, -1.21545369,
                 -1.8625013]),
        'constant': ([1.28480404, 1.73309178, 3.40612438, 6.32928585,
                      7.51935555],
                     [-0.48296291, -2.15599552, -5.95034847, -1.21545369,
                      0.25881905]),
        'symmetric': ([1.76776695, 1.73309178, 3.40612438, 6.32928585,
                       7.77817459],
                      [-0.61237244, -2.15599552, -5.95034847, -1.21545369,
                       1.22474487]),
        'reflect': ([2.12132034, 1.73309178, 3.40612438, 6.32928585,
                     6.81224877],
                    [-0.70710678, -2.15599552, -5.95034847, -1.21545369,
                     -2.38013939]),
        'periodic': ([6.9162743, 1.73309178, 3.40612438, 6.32928585,
                      6.9162743],
                     [-1.99191082, -2.15599552, -5.95034847, -1.21545369,
                      -1.99191082]),
        'smooth': ([-0.51763809, 1.73309178, 3.40612438, 6.32928585,
                    7.45000519],
                   [0, -2.15599552, -5.95034847, -1.21545369, 0]),
        'periodization': ([4.053172, 3.05257099, 2.85381112, 8.42522221],
                          [0.18946869, 4.18258152, 4.33737503, 2.60428326]),
        'antisymmetric': ([-1.83711731, 1.73309178, 3.40612438, 6.32928585,
                           6.12372436],
                          [0.353553391, -2.15599552, -5.95034847, -1.21545369,
                           -4.94974747]),
        'antireflect': ([0.44828774, 1.73309178, 3.40612438, 6.32928585,
                         8.22646233],
                        [-0.25881905, -2.15599552, -5.95034847, -1.21545369,
                         2.89777748])
    }

    for mode in pywt.Modes.modes:
        cA, cD = pywt.dwt(x, 'db2', mode)
        assert_allclose(cA, dwt_results[mode][0], rtol=1e-7, atol=1e-8)
        assert_allclose(cD, dwt_results[mode][1], rtol=1e-7, atol=1e-8)
        assert_allclose(pywt.idwt(cA, cD, 'db2', mode), x, rtol=1e-10)


def test_dwt_short_input_allmodes():
    # some test cases where the input is shorter than the DWT filter
    x = [1, 3, 2]
    wavelet = 'db2'
    # manually pad each end by the filter size (4 for 'db2' used here)
    padded_x = {'zero': [0, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0],
                'constant': [1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2],
                'symmetric': [2, 2, 3, 1, 1, 3, 2, 2, 3, 1, 1],
                'reflect': [1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2],
                'periodic': [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
                'smooth': [-7, -5, -3, -1, 1, 3, 2, 1, 0, -1, -2],
                'antisymmetric': [2, -2, -3, -1, 1, 3, 2, -2, -3, -1, 1],
                'antireflect': [1, -1, 0, -1, 1, 3, 2, 1, 3, 5, 4],
                }
    for mode, xpad in padded_x.items():
        # DWT of the manually padded array.  will discard edges later so
        # symmetric mode used here doesn't matter.
        cApad, cDpad = pywt.dwt(xpad, wavelet, mode='symmetric')

        # central region of the padded output (unaffected by mode  )
        expected_result = (cApad[2:-2], cDpad[2:-2])

        cA, cD = pywt.dwt(x, wavelet, mode)
        assert_allclose(cA, expected_result[0], rtol=1e-7, atol=1e-8)
        assert_allclose(cD, expected_result[1], rtol=1e-7, atol=1e-8)


def test_default_mode():
    # The default mode should be 'symmetric'
    x = [1, 2, 1, 5, -1, 8, 4, 6]
    cA, cD = pywt.dwt(x, 'db2')
    cA2, cD2 = pywt.dwt(x, 'db2', mode='symmetric')
    assert_allclose(cA, cA2)
    assert_allclose(cD, cD2)
    assert_allclose(pywt.idwt(cA, cD, 'db2'), x)
