"""
Test used to verify PyWavelets Continuous Wavelet Transform computation
accuracy against MathWorks Wavelet Toolbox.
"""

from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
import pytest
from numpy.testing import assert_

import pywt
from pywt._pytest import (uses_pymatbridge, uses_precomputed, size_set,
                          matlab_result_dict_cwt)

families = ('gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor')
wavelets = sum([pywt.wavelist(name) for name in families], [])


def _get_data_sizes(w):
    """ Return the sizes to test for wavelet w. """
    if size_set == 'full':
        data_sizes = list(range(100, 101)) + \
            [100, 200, 500, 1000, 50000]
    else:
        data_sizes = (1000, 1000 + 1)
    return data_sizes


def _get_scales(w):
    """ Return the scales to test for wavelet w. """
    if size_set == 'full':
        scales = (1, np.arange(1, 3), np.arange(1, 4), np.arange(1, 5))
    else:
        scales = (1, np.arange(1, 3))
    return scales


@uses_pymatbridge  # skip this case if precomputed results are used instead
@pytest.mark.slow
def test_accuracy_pymatbridge_cwt():
    Matlab = pytest.importorskip("pymatbridge.Matlab")
    mlab = Matlab()
    rstate = np.random.RandomState(1234)
    # max RMSE (was 1.0e-10, is reduced to 5.0e-5 due to different coefficents)
    epsilon = 1e-15
    epsilon_psi = 1e-15
    mlab.start()
    try:
        for wavelet in wavelets:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                w = pywt.ContinuousWavelet(wavelet)
            if np.any((wavelet == np.array(['shan', 'cmor'])),axis=0):
                mlab.set_variable('wavelet', wavelet+str(w.bandwidth_frequency)+'-'+str(w.center_frequency))
            elif wavelet == 'fbsp':
                mlab.set_variable('wavelet', wavelet+str(w.fbsp_order)+'-'+str(w.bandwidth_frequency)+'-'+str(w.center_frequency))
            else:
                mlab.set_variable('wavelet', wavelet)
            mlab_code = ("psi = wavefun(wavelet,10)")
            res = mlab.run_code(mlab_code)
            psi = np.asarray(mlab.get_variable('psi'))
            _check_accuracy_psi(w, psi, wavelet, epsilon_psi)
            for N in _get_data_sizes(w):
                data = rstate.randn(N)
                mlab.set_variable('data', data)
                for scales in _get_scales(w):
                    coefs = _compute_matlab_result(data, wavelet, scales, mlab)
                    _check_accuracy(data, w, scales, coefs, wavelet, epsilon)

    finally:
        mlab.stop()


@uses_precomputed  # skip this case if pymatbridge + Matlab are being used
@pytest.mark.slow
def test_accuracy_precomputed_cwt():
    # Keep this specific random seed to match the precomputed Matlab result.
    rstate = np.random.RandomState(1234)
    # has to be improved
    epsilon = 2e-15
    epsilon32 = 1e-5
    epsilon_psi = 1e-15
    for wavelet in wavelets:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            w = pywt.ContinuousWavelet(wavelet)
            w32 = pywt.ContinuousWavelet(wavelet,dtype=np.float32)
        psi = _load_matlab_result_psi(wavelet)
        _check_accuracy_psi(w, psi, wavelet, epsilon_psi)

        for N in _get_data_sizes(w):
            data = rstate.randn(N)
            data32 = data.astype(np.float32)
            scales_count = 0
            for scales in _get_scales(w):
                scales_count += 1
                coefs = _load_matlab_result(data, wavelet, scales_count)
                _check_accuracy(data, w, scales, coefs, wavelet, epsilon)
                _check_accuracy(data32, w32, scales, coefs, wavelet, epsilon32)


def _compute_matlab_result(data, wavelet, scales, mlab):
    """ Compute the result using MATLAB.

    This function assumes that the Matlab variables `wavelet` and `data` have
    already been set externally.
    """
    mlab.set_variable('scales', scales)
    mlab_code = ("coefs = cwt(data, scales, wavelet)")
    res = mlab.run_code(mlab_code)
    if not res['success']:
        raise RuntimeError("Matlab failed to execute the provided code. "
                           "Check that the wavelet toolbox is installed.")
    # need np.asarray because sometimes the output is a single float64
    coefs = np.asarray(mlab.get_variable('coefs'))
    return coefs


def _load_matlab_result(data, wavelet, scales):
    """ Load the precomputed result.
    """
    N = len(data)
    coefs_key = '_'.join([str(scales), wavelet, str(N), 'coefs'])
    if (coefs_key not in matlab_result_dict_cwt):
        raise KeyError(
            "Precompted Matlab result not found for wavelet: "
            "{0}, mode: {1}, size: {2}".format(wavelet, scales, N))
    coefs = matlab_result_dict_cwt[coefs_key]
    return coefs


def _load_matlab_result_psi(wavelet):
    """ Load the precomputed result.
    """
    psi_key = '_'.join([wavelet, 'psi'])
    if (psi_key not in matlab_result_dict_cwt):
        raise KeyError(
            "Precompted Matlab psi result not found for wavelet: "
            "{0}}".format(wavelet))
    psi = matlab_result_dict_cwt[psi_key]
    return psi


def _check_accuracy(data, w, scales, coefs, wavelet, epsilon):
    # PyWavelets result
    coefs_pywt, freq = pywt.cwt(data, scales, w)

    # coefs from Matlab are from R2012a which is missing the complex conjugate
    # as shown in Eq. 2 of Torrence and Compo. We take the complex conjugate of
    # the precomputed Matlab result to account for this.
    coefs = np.conj(coefs)

    # calculate error measures
    err = coefs_pywt - coefs
    rms = np.real(np.sqrt(np.mean(np.conj(err) * err)))

    msg = ('[RMS > EPSILON] for Scale: %s, Wavelet: %s, '
           'Length: %d, rms=%.3g' % (scales, wavelet, len(data), rms))
    assert_(rms < epsilon, msg=msg)


def _check_accuracy_psi(w, psi, wavelet, epsilon):
    # PyWavelets result
    psi_pywt, x = w.wavefun(length=1024)

    # calculate error measures
    err = psi_pywt.flatten() - psi.flatten()
    rms = np.real(np.sqrt(np.mean(np.conj(err) * err)))

    msg = ('[RMS > EPSILON] for  Wavelet: %s, '
           'rms=%.3g' % (wavelet, rms))
    assert_(rms < epsilon, msg=msg)
