"""
Tests used to verify running PyWavelets transforms in parallel via
concurrent.futures.ThreadPoolExecutor does not raise errors.
"""

from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
from functools import partial
from numpy.testing import assert_array_equal, assert_allclose
from pywt._pytest import uses_futures, futures, max_workers

import pywt


def _assert_all_coeffs_equal(coefs1, coefs2):
    # return True only if all coefficients of SWT or DWT match over all levels
    if len(coefs1) != len(coefs2):
        return False
    for (c1, c2) in zip(coefs1, coefs2):
        if isinstance(c1, tuple):
            # for swt, swt2, dwt, dwt2, wavedec, wavedec2
            for a1, a2 in zip(c1, c2):
                assert_array_equal(a1, a2)
        elif isinstance(c1, dict):
            # for swtn, dwtn, wavedecn
            for k, v in c1.items():
                assert_array_equal(v, c2[k])
        else:
            return False
    return True


@uses_futures
def test_concurrent_swt():
    # tests error-free concurrent operation (see gh-288)
    # swt on 1D data calls the Cython swt
    # other cases call swt_axes
    with warnings.catch_warnings():
        # can remove catch_warnings once the swt2 FutureWarning is removed
        warnings.simplefilter('ignore', FutureWarning)
        for swt_func, x in zip([pywt.swt, pywt.swt2, pywt.swtn],
                               [np.ones(8), np.eye(16), np.eye(16)]):
            transform = partial(swt_func, wavelet='haar', level=3)
            for _ in range(10):
                arrs = [x.copy() for _ in range(100)]
                with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    results = list(ex.map(transform, arrs))

        # validate result from  one of the concurrent runs
        expected_result = transform(x)
        _assert_all_coeffs_equal(expected_result, results[-1])


@uses_futures
def test_concurrent_wavedec():
    # wavedec on 1D data calls the Cython dwt_single
    # other cases call dwt_axis
    for wavedec_func, x in zip([pywt.wavedec, pywt.wavedec2, pywt.wavedecn],
                               [np.ones(8), np.eye(16), np.eye(16)]):
        transform = partial(wavedec_func, wavelet='haar', level=1)
        for _ in range(10):
            arrs = [x.copy() for _ in range(100)]
            with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(transform, arrs))

        # validate result from  one of the concurrent runs
        expected_result = transform(x)
        _assert_all_coeffs_equal(expected_result, results[-1])


@uses_futures
def test_concurrent_dwt():
    # dwt on 1D data calls the Cython dwt_single
    # other cases call dwt_axis
    for dwt_func, x in zip([pywt.dwt, pywt.dwt2, pywt.dwtn],
                           [np.ones(8), np.eye(16), np.eye(16)]):
        transform = partial(dwt_func, wavelet='haar')
        for _ in range(10):
            arrs = [x.copy() for _ in range(100)]
            with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(transform, arrs))

        # validate result from  one of the concurrent runs
        expected_result = transform(x)
        _assert_all_coeffs_equal([expected_result, ], [results[-1], ])


@uses_futures
def test_concurrent_cwt():
    atol = rtol = 1e-14
    time, sst = pywt.data.nino()
    dt = time[1]-time[0]
    transform = partial(pywt.cwt, scales=np.arange(1, 4), wavelet='cmor1.5-1',
                        sampling_period=dt)
    for _ in range(10):
        arrs = [sst.copy() for _ in range(50)]
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(transform, arrs))

    # validate result from  one of the concurrent runs
    expected_result = transform(sst)
    for a1, a2 in zip(expected_result, results[-1]):
        assert_allclose(a1, a2, atol=atol, rtol=rtol)
