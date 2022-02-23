# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2016 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See COPYING for license details.

"""
Other wavelet related functions.
"""

from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from numpy.fft import fft

from ._extensions._pywt import DiscreteContinuousWavelet, Wavelet, ContinuousWavelet


__all__ = ["integrate_wavelet", "central_frequency", "scale2frequency", "qmf",
           "orthogonal_filter_bank",
           "intwave", "centrfrq", "scal2frq", "orthfilt"]


_DEPRECATION_MSG = ("`{old}` has been renamed to `{new}` and will "
                    "be removed in a future version of pywt.")


def _integrate(arr, step):
    integral = np.cumsum(arr)
    integral *= step
    return integral


def intwave(*args, **kwargs):
    msg = _DEPRECATION_MSG.format(old='intwave', new='integrate_wavelet')
    warnings.warn(msg, DeprecationWarning)
    return integrate_wavelet(*args, **kwargs)


def centrfrq(*args, **kwargs):
    msg = _DEPRECATION_MSG.format(old='centrfrq', new='central_frequency')
    warnings.warn(msg, DeprecationWarning)
    return central_frequency(*args, **kwargs)


def scal2frq(*args, **kwargs):
    msg = _DEPRECATION_MSG.format(old='scal2frq', new='scale2frequency')
    warnings.warn(msg, DeprecationWarning)
    return scale2frequency(*args, **kwargs)


def orthfilt(*args, **kwargs):
    msg = _DEPRECATION_MSG.format(old='orthfilt', new='orthogonal_filter_bank')
    warnings.warn(msg, DeprecationWarning)
    return orthogonal_filter_bank(*args, **kwargs)


def integrate_wavelet(wavelet, precision=8):
    """
    Integrate `psi` wavelet function from -Inf to x using the rectangle
    integration method.

    Parameters
    ----------
    wavelet : Wavelet instance or str
        Wavelet to integrate.  If a string, should be the name of a wavelet.
    precision : int, optional
        Precision that will be used for wavelet function
        approximation computed with the wavefun(level=precision)
        Wavelet's method (default: 8).

    Returns
    -------
    [int_psi, x] :
        for orthogonal wavelets
    [int_psi_d, int_psi_r, x] :
        for other wavelets


    Examples
    --------
    >>> from pywt import Wavelet, integrate_wavelet
    >>> wavelet1 = Wavelet('db2')
    >>> [int_psi, x] = integrate_wavelet(wavelet1, precision=5)
    >>> wavelet2 = Wavelet('bior1.3')
    >>> [int_psi_d, int_psi_r, x] = integrate_wavelet(wavelet2, precision=5)

    """
    # FIXME: this function should really use scipy.integrate.quad

    if type(wavelet) in (tuple, list):
        msg = ("Integration of a general signal is deprecated "
               "and will be removed in a future version of pywt.")
        warnings.warn(msg, DeprecationWarning)
    elif not isinstance(wavelet, (Wavelet, ContinuousWavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)

    if type(wavelet) in (tuple, list):
        psi, x = np.asarray(wavelet[0]), np.asarray(wavelet[1])
        step = x[1] - x[0]
        return _integrate(psi, step), x

    functions_approximations = wavelet.wavefun(precision)

    if len(functions_approximations) == 2:      # continuous wavelet
        psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    elif len(functions_approximations) == 3:    # orthogonal wavelet
        phi, psi, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi, step), x

    else:                                       # biorthogonal wavelet
        phi_d, psi_d, phi_r, psi_r, x = functions_approximations
        step = x[1] - x[0]
        return _integrate(psi_d, step), _integrate(psi_r, step), x


def central_frequency(wavelet, precision=8):
    """
    Computes the central frequency of the `psi` wavelet function.

    Parameters
    ----------
    wavelet : Wavelet instance, str or tuple
        Wavelet to integrate.  If a string, should be the name of a wavelet.
    precision : int, optional
        Precision that will be used for wavelet function
        approximation computed with the wavefun(level=precision)
        Wavelet's method (default: 8).

    Returns
    -------
    scalar

    """

    if not isinstance(wavelet, (Wavelet, ContinuousWavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)

    functions_approximations = wavelet.wavefun(precision)

    if len(functions_approximations) == 2:
        psi, x = functions_approximations
    else:
        # (psi, x)   for (phi, psi, x)
        # (psi_d, x) for (phi_d, psi_d, phi_r, psi_r, x)
        psi, x = functions_approximations[1], functions_approximations[-1]

    domain = float(x[-1] - x[0])
    assert domain > 0

    index = np.argmax(abs(fft(psi)[1:])) + 2
    if index > len(psi) / 2:
        index = len(psi) - index + 2

    return 1.0 / (domain / (index - 1))


def scale2frequency(wavelet, scale, precision=8):
    """

    Parameters
    ----------
    wavelet : Wavelet instance or str
        Wavelet to integrate.  If a string, should be the name of a wavelet.
    scale : scalar
    precision : int, optional
        Precision that will be used for wavelet function approximation computed
        with ``wavelet.wavefun(level=precision)``.  Default is 8.

    Returns
    -------
    freq : scalar

    """
    return central_frequency(wavelet, precision=precision) / scale


def qmf(filt):
    """
    Returns the Quadrature Mirror Filter(QMF).

    The magnitude response of QMF is mirror image about `pi/2` of that of the
    input filter.

    Parameters
    ----------
    filt : array_like
        Input filter for which QMF needs to be computed.

    Returns
    -------
    qm_filter : ndarray
        Quadrature mirror of the input filter.

    """
    qm_filter = np.array(filt)[::-1]
    qm_filter[1::2] = -qm_filter[1::2]
    return qm_filter


def orthogonal_filter_bank(scaling_filter):
    """
    Returns the orthogonal filter bank.

    The orthogonal filter bank consists of the HPFs and LPFs at
    decomposition and reconstruction stage for the input scaling filter.

    Parameters
    ----------
    scaling_filter : array_like
        Input scaling filter (father wavelet).

    Returns
    -------
    orth_filt_bank : tuple of 4 ndarrays
        The orthogonal filter bank of the input scaling filter in the order :
        1] Decomposition LPF
        2] Decomposition HPF
        3] Reconstruction LPF
        4] Reconstruction HPF

    """
    if not (len(scaling_filter) % 2 == 0):
        raise ValueError("`scaling_filter` length has to be even.")

    scaling_filter = np.asarray(scaling_filter, dtype=np.float64)

    rec_lo = np.sqrt(2) * scaling_filter / np.sum(scaling_filter)
    dec_lo = rec_lo[::-1]

    rec_hi = qmf(rec_lo)
    dec_hi = rec_hi[::-1]

    orth_filt_bank = (dec_lo, dec_hi, rec_lo, rec_hi)
    return orth_filt_bank
