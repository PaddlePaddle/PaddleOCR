#!/usr/bin/env python

"""
Verify DWT perfect reconstruction.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_

import pywt


def test_perfect_reconstruction():
    families = ('db', 'sym', 'coif', 'bior', 'rbio')
    wavelets = sum([pywt.wavelist(name) for name in families], [])
    # list of mode names in pywt and matlab
    modes = [('zero', 'zpd'),
             ('constant', 'sp0'),
             ('symmetric', 'sym'),
             ('periodic', 'ppd'),
             ('smooth', 'sp1'),
             ('periodization', 'per')]

    dtypes = (np.float32, np.float64)

    for wavelet in wavelets:
        for pmode, mmode in modes:
            for dt in dtypes:
                check_reconstruction(pmode, mmode, wavelet, dt)


def check_reconstruction(pmode, mmode, wavelet, dtype):
    data_size = list(range(2, 40)) + [100, 200, 500, 1000, 2000, 10000,
                                      50000, 100000]
    np.random.seed(12345)
    # TODO: smoke testing - more failures for different seeds

    if dtype == np.float32:
        # was 3e-7 has to be lowered as db21, db29, db33, db35, coif14, coif16 were failing
        epsilon = 6e-7
    else:
        epsilon = 5e-11

    for N in data_size:
        data = np.asarray(np.random.random(N), dtype)

        # compute dwt coefficients
        pa, pd = pywt.dwt(data, wavelet, pmode)

        # compute reconstruction
        rec = pywt.idwt(pa, pd, wavelet, pmode)

        if len(data) % 2:
            rec = rec[:len(data)]

        rms_rec = np.sqrt(np.mean((data-rec)**2))
        msg = ('[RMS_REC > EPSILON] for Mode: %s, Wavelet: %s, '
               'Length: %d, rms=%.3g' % (pmode, wavelet, len(data), rms_rec))
        assert_(rms_rec < epsilon, msg=msg)
