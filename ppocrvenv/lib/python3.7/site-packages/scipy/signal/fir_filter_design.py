# -*- coding: utf-8 -*-
"""Functions for FIR filter design."""

from math import ceil, log
import operator
import warnings

import numpy as np
from numpy.fft import irfft, fft, ifft
from scipy.special import sinc
from scipy.linalg import (toeplitz, hankel, solve, LinAlgError, LinAlgWarning,
                          lstsq)

from . import sigtools

__all__ = ['kaiser_beta', 'kaiser_atten', 'kaiserord',
           'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase']


def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    if nyq is None and fs is None:
        fs = 2
    elif nyq is not None:
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        fs = 2*nyq
    return fs


# Some notes on function parameters:
#
# `cutoff` and `width` are given as numbers between 0 and 1.  These are
# relative frequencies, expressed as a fraction of the Nyquist frequency.
# For example, if the Nyquist frequency is 2 KHz, then width=0.15 is a width
# of 300 Hz.
#
# The `order` of a FIR filter is one less than the number of taps.
# This is a potential source of confusion, so in the following code,
# we will always use the number of taps as the parameterization of
# the 'size' of the filter. The "number of taps" means the number
# of coefficients, which is the same as the length of the impulse
# response of the filter.


def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.

    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.

    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.

    Examples
    --------
    Suppose we want to design a lowpass filter, with 65 dB attenuation
    in the stop band.  The Kaiser window parameter to be used in the
    window method is computed by `kaiser_beta(65)`:

    >>> from scipy.signal import kaiser_beta
    >>> kaiser_beta(65)
    6.20426

    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta


def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.

    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:

        a = 2.285 * (N - 1) * pi * width + 7.95

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter,
        expressed as a fraction of the Nyquist frequency.

    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.

    See Also
    --------
    kaiserord, kaiser_beta

    Examples
    --------
    Suppose we want to design a FIR filter using the Kaiser window method
    that will have 211 taps and a transition width of 9 Hz for a signal that
    is sampled at 480 Hz. Expressed as a fraction of the Nyquist frequency,
    the width is 9/(0.5*480) = 0.0375. The approximate attenuation (in dB)
    is computed as follows:

    >>> from scipy.signal import kaiser_atten
    >>> kaiser_atten(211, 0.0375)
    64.48099630593983

    """
    a = 2.285 * (numtaps - 1) * np.pi * width + 7.95
    return a


def kaiserord(ripple, width):
    """
    Determine the filter window parameters for the Kaiser window method.

    The parameters returned by this function are generally used to create
    a finite impulse response filter using the window method, with either
    `firwin` or `firwin2`.

    Parameters
    ----------
    ripple : float
        Upper bound for the deviation (in dB) of the magnitude of the
        filter's frequency response from that of the desired filter (not
        including frequencies in any transition intervals). That is, if w
        is the frequency expressed as a fraction of the Nyquist frequency,
        A(w) is the actual frequency response of the filter and D(w) is the
        desired frequency response, the design requirement is that::

            abs(A(w) - D(w))) < 10**(-ripple/20)

        for 0 <= w <= 1 and w not in a transition interval.
    width : float
        Width of transition region, normalized so that 1 corresponds to pi
        radians / sample. That is, the frequency is expressed as a fraction
        of the Nyquist frequency.

    Returns
    -------
    numtaps : int
        The length of the Kaiser window.
    beta : float
        The beta parameter for the Kaiser window.

    See Also
    --------
    kaiser_beta, kaiser_atten

    Notes
    -----
    There are several ways to obtain the Kaiser window:

    - ``signal.windows.kaiser(numtaps, beta, sym=True)``
    - ``signal.get_window(beta, numtaps)``
    - ``signal.get_window(('kaiser', beta), numtaps)``

    The empirical equations discovered by Kaiser are used.

    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", pp.475-476.

    Examples
    --------
    We will use the Kaiser window method to design a lowpass FIR filter
    for a signal that is sampled at 1000 Hz.

    We want at least 65 dB rejection in the stop band, and in the pass
    band the gain should vary no more than 0.5%.

    We want a cutoff frequency of 175 Hz, with a transition between the
    pass band and the stop band of 24 Hz. That is, in the band [0, 163],
    the gain varies no more than 0.5%, and in the band [187, 500], the
    signal is attenuated by at least 65 dB.

    >>> from scipy.signal import kaiserord, firwin, freqz
    >>> import matplotlib.pyplot as plt
    >>> fs = 1000.0
    >>> cutoff = 175
    >>> width = 24

    The Kaiser method accepts just a single parameter to control the pass
    band ripple and the stop band rejection, so we use the more restrictive
    of the two. In this case, the pass band ripple is 0.005, or 46.02 dB,
    so we will use 65 dB as the design parameter.

    Use `kaiserord` to determine the length of the filter and the
    parameter for the Kaiser window.

    >>> numtaps, beta = kaiserord(65, width/(0.5*fs))
    >>> numtaps
    167
    >>> beta
    6.20426

    Use `firwin` to create the FIR filter.

    >>> taps = firwin(numtaps, cutoff, window=('kaiser', beta),
    ...               scale=False, nyq=0.5*fs)

    Compute the frequency response of the filter.  ``w`` is the array of
    frequencies, and ``h`` is the corresponding complex array of frequency
    responses.

    >>> w, h = freqz(taps, worN=8000)
    >>> w *= 0.5*fs/np.pi  # Convert w to Hz.

    Compute the deviation of the magnitude of the filter's response from
    that of the ideal lowpass filter. Values in the transition region are
    set to ``nan``, so they won't appear in the plot.

    >>> ideal = w < cutoff  # The "ideal" frequency response.
    >>> deviation = np.abs(np.abs(h) - ideal)
    >>> deviation[(w > cutoff - 0.5*width) & (w < cutoff + 0.5*width)] = np.nan

    Plot the deviation. A close look at the left end of the stop band shows
    that the requirement for 65 dB attenuation is violated in the first lobe
    by about 0.125 dB. This is not unusual for the Kaiser window method.

    >>> plt.plot(w, 20*np.log10(np.abs(deviation)))
    >>> plt.xlim(0, 0.5*fs)
    >>> plt.ylim(-90, -60)
    >>> plt.grid(alpha=0.25)
    >>> plt.axhline(-65, color='r', ls='--', alpha=0.3)
    >>> plt.xlabel('Frequency (Hz)')
    >>> plt.ylabel('Deviation from ideal (dB)')
    >>> plt.title('Lowpass Filter Frequency Response')
    >>> plt.show()

    """
    A = abs(ripple)  # in case somebody is confused as to what's meant
    if A < 8:
        # Formula for N is not valid in this range.
        raise ValueError("Requested maximum ripple attentuation %f is too "
                         "small for the Kaiser formula." % A)
    beta = kaiser_beta(A)

    # Kaiser's formula (as given in Oppenheim and Schafer) is for the filter
    # order, so we have to add 1 to get the number of taps.
    numtaps = (A - 7.95) / 2.285 / (np.pi * width) + 1

    return int(ceil(numtaps)), beta


def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True,
           scale=True, nyq=None, fs=None):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter. The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1-D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`. The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design. In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `scipy.signal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        If True, the gain at the frequency 0 (i.e., the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).

        .. versionadded:: 1.3.0
           Support for string arguments.
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise

    nyq : float, optional
        *Deprecated. Use `fs` instead.* This is the Nyquist frequency.
        Each frequency in `cutoff` must be between 0 and `nyq`. Default
        is 1.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to ``fs/2``, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See Also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> from scipy import signal
    >>> numtaps = 3
    >>> f = 0.1
    >>> signal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> signal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> signal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> signal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> signal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> signal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> signal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """  # noqa: E501
    # The major enhancements to this function added in November 2010 were
    # developed by Tom Krauss (see ticket #902).

    nyq = 0.5 * _get_fs(fs, nyq)

    cutoff = np.atleast_1d(cutoff) / float(nyq)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError("The cutoff argument must be at most "
                         "one-dimensional.")
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError("Invalid cutoff frequency: frequencies must be "
                         "greater than 0 and less than fs/2.")
    if np.any(np.diff(cutoff) <= 0):
        raise ValueError("Invalid cutoff frequencies: the frequencies "
                         "must be strictly increasing.")

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ('kaiser', beta)

    if isinstance(pass_zero, str):
        if pass_zero in ('bandstop', 'lowpass'):
            if pass_zero == 'lowpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if '
                                     'pass_zero=="lowpass", got %s'
                                     % (cutoff.shape,))
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if '
                                 'pass_zero=="bandstop", got %s'
                                 % (cutoff.shape,))
            pass_zero = True
        elif pass_zero in ('bandpass', 'highpass'):
            if pass_zero == 'highpass':
                if cutoff.size != 1:
                    raise ValueError('cutoff must have one element if '
                                     'pass_zero=="highpass", got %s'
                                     % (cutoff.shape,))
            elif cutoff.size <= 1:
                raise ValueError('cutoff must have at least two elements if '
                                 'pass_zero=="bandpass", got %s'
                                 % (cutoff.shape,))
            pass_zero = False
        else:
            raise ValueError('pass_zero must be True, False, "bandpass", '
                             '"lowpass", "highpass", or "bandstop", got '
                             '%s' % (pass_zero,))
    pass_zero = bool(operator.index(pass_zero))  # ensure bool-like

    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("A filter with an even number of coefficients must "
                         "have zero response at the Nyquist frequency.")

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = np.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2-D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    h = 0
    for left, right in bands:
        h += right * sinc(right * m)
        h -= left * sinc(left * m)

    # Get and apply the window function.
    from .signaltools import get_window
    win = get_window(window, numtaps, fftbins=False)
    h *= win

    # Now handle scaling if desired.
    if scale:
        # Get the first passband.
        left, right = bands[0]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = 0.5 * (left + right)
        c = np.cos(np.pi * m * scale_frequency)
        s = np.sum(h * c)
        h /= s

    return h


# Original version of firwin2 from scipy ticket #457, submitted by "tash".
#
# Rewritten by Warren Weckesser, 2010.

def firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', nyq=None,
            antisymmetric=False, fs=None):
    """
    FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    nyq : float, optional
        *Deprecated. Use `fs` instead.* This is the Nyquist frequency.
        Each frequency in `freq` must be between 0 and `nyq`.  Default is 1.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``. Default is 2.

    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.

    See also
    --------
    firls
    firwin
    minimum_phase
    remez

    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.

    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:

       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced

    Magnitude response of all but type I filters are subjects to following
    constraints:

       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency

    .. versionadded:: 0.9.0

    References
    ----------
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)

    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm

    Examples
    --------
    A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and
    that decreases linearly on [0.5, 1.0] from 1 to 0:

    >>> from scipy import signal
    >>> taps = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    >>> print(taps[72:78])
    [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]

    """
    nyq = 0.5 * _get_fs(fs, nyq)

    if len(freq) != len(gain):
        raise ValueError('freq and gain must be of same length.')

    if nfreqs is not None and numtaps >= nfreqs:
        raise ValueError(('ntaps must be less than nfreqs, but firwin2 was '
                          'called with ntaps=%d and nfreqs=%s') %
                         (numtaps, nfreqs))

    if freq[0] != 0 or freq[-1] != nyq:
        raise ValueError('freq must start with 0 and end with fs/2.')
    d = np.diff(freq)
    if (d < 0).any():
        raise ValueError('The values in freq must be nondecreasing.')
    d2 = d[:-1] + d[1:]
    if (d2 == 0).any():
        raise ValueError('A value in freq must not occur more than twice.')
    if freq[1] == 0:
        raise ValueError('Value 0 must not be repeated in freq')
    if freq[-2] == nyq:
        raise ValueError('Value fs/2 must not be repeated in freq')

    if antisymmetric:
        if numtaps % 2 == 0:
            ftype = 4
        else:
            ftype = 3
    else:
        if numtaps % 2 == 0:
            ftype = 2
        else:
            ftype = 1

    if ftype == 2 and gain[-1] != 0.0:
        raise ValueError("A Type II filter must have zero gain at the "
                         "Nyquist frequency.")
    elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
        raise ValueError("A Type III filter must have zero gain at zero "
                         "and Nyquist frequencies.")
    elif ftype == 4 and gain[0] != 0.0:
        raise ValueError("A Type IV filter must have zero gain at zero "
                         "frequency.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))

    if (d == 0).any():
        # Tweak any repeated values in freq so that interp works.
        freq = np.array(freq, copy=True)
        eps = np.finfo(float).eps * nyq
        for k in range(len(freq) - 1):
            if freq[k] == freq[k + 1]:
                freq[k] = freq[k] - eps
                freq[k + 1] = freq[k + 1] + eps
        # Check if freq is strictly increasing after tweak
        d = np.diff(freq)
        if (d <= 0).any():
            raise ValueError("freq cannot contain numbers that are too close "
                             "(within eps * (fs/2): "
                             "{}) to a repeated value".format(eps))

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = np.linspace(0.0, nyq, nfreqs)
    fx = np.interp(x, freq, gain)

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = np.exp(-(numtaps - 1) / 2. * 1.j * np.pi * x / nyq)
    if ftype > 2:
        shift *= 1j

    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = irfft(fx2)

    if window is not None:
        # Create the window to apply to the filter coefficients.
        from .signaltools import get_window
        wind = get_window(window, numtaps, fftbins=False)
    else:
        wind = 1

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind

    if ftype == 3:
        out[out.size // 2] = 0.0

    return out


def remez(numtaps, bands, desired, weight=None, Hz=None, type='bandpass',
          maxiter=25, grid_density=16, fs=None):
    """
    Calculate the minimax optimal filter using the Remez exchange algorithm.

    Calculate the filter-coefficients for the finite impulse response
    (FIR) filter whose transfer function minimizes the maximum error
    between the desired gain and the realized gain in the specified
    frequency bands using the Remez exchange algorithm.

    Parameters
    ----------
    numtaps : int
        The desired number of taps in the filter. The number of taps is
        the number of terms in the filter, or the filter order plus one.
    bands : array_like
        A monotonic sequence containing the band edges.
        All elements must be non-negative and less than half the sampling
        frequency as given by `fs`.
    desired : array_like
        A sequence half the size of bands containing the desired gain
        in each of the specified bands.
    weight : array_like, optional
        A relative weighting to give to each band region. The length of
        `weight` has to be half the length of `bands`.
    Hz : scalar, optional
        *Deprecated.  Use `fs` instead.*
        The sampling frequency in Hz. Default is 1.
    type : {'bandpass', 'differentiator', 'hilbert'}, optional
        The type of filter:

          * 'bandpass' : flat response in bands. This is the default.

          * 'differentiator' : frequency proportional response in bands.

          * 'hilbert' : filter with odd symmetry, that is, type III
                        (for even order) or type IV (for odd order)
                        linear phase filters.

    maxiter : int, optional
        Maximum number of iterations of the algorithm. Default is 25.
    grid_density : int, optional
        Grid density. The dense grid used in `remez` is of size
        ``(numtaps + 1) * grid_density``. Default is 16.
    fs : float, optional
        The sampling frequency of the signal.  Default is 1.

    Returns
    -------
    out : ndarray
        A rank-1 array containing the coefficients of the optimal
        (in a minimax sense) filter.

    See Also
    --------
    firls
    firwin
    firwin2
    minimum_phase

    References
    ----------
    .. [1] J. H. McClellan and T. W. Parks, "A unified approach to the
           design of optimum FIR linear phase digital filters",
           IEEE Trans. Circuit Theory, vol. CT-20, pp. 697-701, 1973.
    .. [2] J. H. McClellan, T. W. Parks and L. R. Rabiner, "A Computer
           Program for Designing Optimum FIR Linear Phase Digital
           Filters", IEEE Trans. Audio Electroacoust., vol. AU-21,
           pp. 506-525, 1973.

    Examples
    --------
    In these examples `remez` gets used creating a bandpass, bandstop, lowpass
    and highpass filter. The used parameters are the filter order, an array
    with according frequency boundaries, the desired attenuation values and the
    sampling frequency. Using `freqz` the corresponding frequency response
    gets calculated and plotted.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> def plot_response(fs, w, h, title):
    ...     "Utility function to plot response functions"
    ...     fig = plt.figure()
    ...     ax = fig.add_subplot(111)
    ...     ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ...     ax.set_ylim(-40, 5)
    ...     ax.set_xlim(0, 0.5*fs)
    ...     ax.grid(True)
    ...     ax.set_xlabel('Frequency (Hz)')
    ...     ax.set_ylabel('Gain (dB)')
    ...     ax.set_title(title)

    This example shows a steep low pass transition according to the small
    transition width and high filter order:

    >>> fs = 22050.0       # Sample rate, Hz
    >>> cutoff = 8000.0    # Desired cutoff frequency, Hz
    >>> trans_width = 100  # Width of transition from pass band to stop band, Hz
    >>> numtaps = 400      # Size of the FIR filter.
    >>> taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
    >>> w, h = signal.freqz(taps, [1], worN=2000)
    >>> plot_response(fs, w, h, "Low-pass Filter")

    This example shows a high pass filter:

    >>> fs = 22050.0       # Sample rate, Hz
    >>> cutoff = 2000.0    # Desired cutoff frequency, Hz
    >>> trans_width = 250  # Width of transition from pass band to stop band, Hz
    >>> numtaps = 125      # Size of the FIR filter.
    >>> taps = signal.remez(numtaps, [0, cutoff - trans_width, cutoff, 0.5*fs],
    ...                     [0, 1], Hz=fs)
    >>> w, h = signal.freqz(taps, [1], worN=2000)
    >>> plot_response(fs, w, h, "High-pass Filter")

    For a signal sampled with 22 kHz a bandpass filter with a pass band of 2-5
    kHz gets calculated using the Remez algorithm. The transition width is 260
    Hz and the filter order 10:

    >>> fs = 22000.0         # Sample rate, Hz
    >>> band = [2000, 5000]  # Desired pass band, Hz
    >>> trans_width = 260    # Width of transition from pass band to stop band, Hz
    >>> numtaps = 10        # Size of the FIR filter.
    >>> edges = [0, band[0] - trans_width, band[0], band[1],
    ...          band[1] + trans_width, 0.5*fs]
    >>> taps = signal.remez(numtaps, edges, [0, 1, 0], Hz=fs)
    >>> w, h = signal.freqz(taps, [1], worN=2000)
    >>> plot_response(fs, w, h, "Band-pass Filter")

    It can be seen that for this bandpass filter, the low order leads to higher
    ripple and less steep transitions. There is very low attenuation in the
    stop band and little overshoot in the pass band.  Of course the desired
    gain can be better approximated with a higher filter order.

    The next example shows a bandstop filter. Because of the high filter order
    the transition is quite steep:

    >>> fs = 20000.0         # Sample rate, Hz
    >>> band = [6000, 8000]  # Desired stop band, Hz
    >>> trans_width = 200    # Width of transition from pass band to stop band, Hz
    >>> numtaps = 175        # Size of the FIR filter.
    >>> edges = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
    >>> taps = signal.remez(numtaps, edges, [1, 0, 1], Hz=fs)
    >>> w, h = signal.freqz(taps, [1], worN=2000)
    >>> plot_response(fs, w, h, "Band-stop Filter")

    >>> plt.show()

    """
    if Hz is None and fs is None:
        fs = 1.0
    elif Hz is not None:
        if fs is not None:
            raise ValueError("Values cannot be given for both 'Hz' and 'fs'.")
        fs = Hz

    # Convert type
    try:
        tnum = {'bandpass': 1, 'differentiator': 2, 'hilbert': 3}[type]
    except KeyError as e:
        raise ValueError("Type must be 'bandpass', 'differentiator', "
                         "or 'hilbert'") from e

    # Convert weight
    if weight is None:
        weight = [1] * len(desired)

    bands = np.asarray(bands).copy()
    return sigtools._remez(numtaps, bands, desired, weight, tnum, fs,
                           maxiter, grid_density)


def firls(numtaps, bands, desired, weight=None, nyq=None, fs=None):
    """
    FIR filter design using least-squares error minimization.

    Calculate the filter coefficients for the linear-phase finite
    impulse response (FIR) filter which has the best approximation
    to the desired frequency response described by `bands` and
    `desired` in the least squares sense (i.e., the integral of the
    weighted mean-squared error within the specified bands is
    minimized).

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter. `numtaps` must be odd.
    bands : array_like
        A monotonic nondecreasing sequence containing the band edges in
        Hz. All elements must be non-negative and less than or equal to
        the Nyquist frequency given by `nyq`.
    desired : array_like
        A sequence the same size as `bands` containing the desired gain
        at the start and end point of each band.
    weight : array_like, optional
        A relative weighting to give to each band region when solving
        the least squares problem. `weight` has to be half the size of
        `bands`.
    nyq : float, optional
        *Deprecated. Use `fs` instead.*
        Nyquist frequency. Each frequency in `bands` must be between 0
        and `nyq` (inclusive). Default is 1.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `bands`
        must be between 0 and ``fs/2`` (inclusive). Default is 2.

    Returns
    -------
    coeffs : ndarray
        Coefficients of the optimal (in a least squares sense) FIR filter.

    See also
    --------
    firwin
    firwin2
    minimum_phase
    remez

    Notes
    -----
    This implementation follows the algorithm given in [1]_.
    As noted there, least squares design has multiple advantages:

        1. Optimal in a least-squares sense.
        2. Simple, non-iterative method.
        3. The general solution can obtained by solving a linear
           system of equations.
        4. Allows the use of a frequency dependent weighting function.

    This function constructs a Type I linear phase FIR filter, which
    contains an odd number of `coeffs` satisfying for :math:`n < numtaps`:

    .. math:: coeffs(n) = coeffs(numtaps - 1 - n)

    The odd number of coefficients and filter symmetry avoid boundary
    conditions that could otherwise occur at the Nyquist and 0 frequencies
    (e.g., for Type II, III, or IV variants).

    .. versionadded:: 0.18

    References
    ----------
    .. [1] Ivan Selesnick, Linear-Phase Fir Filter Design By Least Squares.
           OpenStax CNX. Aug 9, 2005.
           http://cnx.org/contents/eb1ecb35-03a9-4610-ba87-41cd771c95f2@7

    Examples
    --------
    We want to construct a band-pass filter. Note that the behavior in the
    frequency ranges between our stop bands and pass bands is unspecified,
    and thus may overshoot depending on the parameters of our filter:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> fig, axs = plt.subplots(2)
    >>> fs = 10.0  # Hz
    >>> desired = (0, 0, 1, 1, 0, 0)
    >>> for bi, bands in enumerate(((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 4.5, 5))):
    ...     fir_firls = signal.firls(73, bands, desired, fs=fs)
    ...     fir_remez = signal.remez(73, bands, desired[::2], fs=fs)
    ...     fir_firwin2 = signal.firwin2(73, bands, desired, fs=fs)
    ...     hs = list()
    ...     ax = axs[bi]
    ...     for fir in (fir_firls, fir_remez, fir_firwin2):
    ...         freq, response = signal.freqz(fir)
    ...         hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
    ...     for band, gains in zip(zip(bands[::2], bands[1::2]),
    ...                            zip(desired[::2], desired[1::2])):
    ...         ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
    ...     if bi == 0:
    ...         ax.legend(hs, ('firls', 'remez', 'firwin2'),
    ...                   loc='lower center', frameon=False)
    ...     else:
    ...         ax.set_xlabel('Frequency (Hz)')
    ...     ax.grid(True)
    ...     ax.set(title='Band-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')
    ...
    >>> fig.tight_layout()
    >>> plt.show()

    """  # noqa
    nyq = 0.5 * _get_fs(fs, nyq)

    numtaps = int(numtaps)
    if numtaps % 2 == 0 or numtaps < 1:
        raise ValueError("numtaps must be odd and >= 1")
    M = (numtaps-1) // 2

    # normalize bands 0->1 and make it 2 columns
    nyq = float(nyq)
    if nyq <= 0:
        raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
    bands = np.asarray(bands).flatten() / nyq
    if len(bands) % 2 != 0:
        raise ValueError("bands must contain frequency pairs.")
    if (bands < 0).any() or (bands > 1).any():
        raise ValueError("bands must be between 0 and 1 relative to Nyquist")
    bands.shape = (-1, 2)

    # check remaining params
    desired = np.asarray(desired).flatten()
    if bands.size != desired.size:
        raise ValueError("desired must have one entry per frequency, got %s "
                         "gains for %s frequencies."
                         % (desired.size, bands.size))
    desired.shape = (-1, 2)
    if (np.diff(bands) <= 0).any() or (np.diff(bands[:, 0]) < 0).any():
        raise ValueError("bands must be monotonically nondecreasing and have "
                         "width > 0.")
    if (bands[:-1, 1] > bands[1:, 0]).any():
        raise ValueError("bands must not overlap.")
    if (desired < 0).any():
        raise ValueError("desired must be non-negative.")
    if weight is None:
        weight = np.ones(len(desired))
    weight = np.asarray(weight).flatten()
    if len(weight) != len(desired):
        raise ValueError("weight must be the same size as the number of "
                         "band pairs (%s)." % (len(bands),))
    if (weight < 0).any():
        raise ValueError("weight must be non-negative.")

    # Set up the linear matrix equation to be solved, Qa = b

    # We can express Q(k,n) = 0.5 Q1(k,n) + 0.5 Q2(k,n)
    # where Q1(k,n)=q(k-n) and Q2(k,n)=q(k+n), i.e. a Toeplitz plus Hankel.

    # We omit the factor of 0.5 above, instead adding it during coefficient
    # calculation.

    # We also omit the 1/π from both Q and b equations, as they cancel
    # during solving.

    # We have that:
    #     q(n) = 1/π ∫W(ω)cos(nω)dω (over 0->π)
    # Using our nomalization ω=πf and with a constant weight W over each
    # interval f1->f2 we get:
    #     q(n) = W∫cos(πnf)df (0->1) = Wf sin(πnf)/πnf
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = np.arange(numtaps)[:, np.newaxis, np.newaxis]
    q = np.dot(np.diff(np.sinc(bands * n) * bands, axis=2)[:, :, 0], weight)

    # Now we assemble our sum of Toeplitz and Hankel
    Q1 = toeplitz(q[:M+1])
    Q2 = hankel(q[:M+1], q[M:])
    Q = Q1 + Q2

    # Now for b(n) we have that:
    #     b(n) = 1/π ∫ W(ω)D(ω)cos(nω)dω (over 0->π)
    # Using our normalization ω=πf and with a constant weight W over each
    # interval and a linear term for D(ω) we get (over each f1->f2 interval):
    #     b(n) = W ∫ (mf+c)cos(πnf)df
    #          = f(mf+c)sin(πnf)/πnf + mf**2 cos(nπf)/(πnf)**2
    # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
    n = n[:M + 1]  # only need this many coefficients here
    # Choose m and c such that we are at the start and end weights
    m = (np.diff(desired, axis=1) / np.diff(bands, axis=1))
    c = desired[:, [0]] - bands[:, [0]] * m
    b = bands * (m*bands + c) * np.sinc(bands * n)
    # Use L'Hospital's rule here for cos(nπf)/(πnf)**2 @ n=0
    b[0] -= m * bands * bands / 2.
    b[1:] += m * np.cos(n[1:] * np.pi * bands) / (np.pi * n[1:]) ** 2
    b = np.dot(np.diff(b, axis=2)[:, :, 0], weight)

    # Now we can solve the equation
    try:  # try the fast way
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            a = solve(Q, b, sym_pos=True, check_finite=False)
        for ww in w:
            if (ww.category == LinAlgWarning and
                    str(ww.message).startswith('Ill-conditioned matrix')):
                raise LinAlgError(str(ww.message))
    except LinAlgError:  # in case Q is rank deficient
        # This is faster than pinvh, even though we don't explicitly use
        # the symmetry here. gelsy was faster than gelsd and gelss in
        # some non-exhaustive tests.
        a = lstsq(Q, b, lapack_driver='gelsy')[0]

    # make coefficients symmetric (linear phase)
    coeffs = np.hstack((a[:0:-1], 2 * a[0], a[1:]))
    return coeffs


def _dhtm(mag):
    """Compute the modified 1-D discrete Hilbert transform

    Parameters
    ----------
    mag : ndarray
        The magnitude spectrum. Should be 1-D with an even length, and
        preferably a fast length for FFT/IFFT.
    """
    # Adapted based on code by Niranjan Damera-Venkata,
    # Brian L. Evans and Shawn R. McCaslin (see refs for `minimum_phase`)
    sig = np.zeros(len(mag))
    # Leave Nyquist and DC at 0, knowing np.abs(fftfreq(N)[midpt]) == 0.5
    midpt = len(mag) // 2
    sig[1:midpt] = 1
    sig[midpt+1:] = -1
    # eventually if we want to support complex filters, we will need a
    # np.abs() on the mag inside the log, and should remove the .real
    recon = ifft(mag * np.exp(fft(sig * ifft(np.log(mag))))).real
    return recon


def minimum_phase(h, method='homomorphic', n_fft=None):
    """Convert a linear-phase FIR filter to minimum phase

    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.
    method : {'hilbert', 'homomorphic'}
        The method to use:

            'homomorphic' (default)
                This method [4]_ [5]_ works best with filters with an
                odd number of taps, and the resulting minimum phase filter
                will have a magnitude response that approximates the square
                root of the the original filter's magnitude response.

            'hilbert'
                This method [1]_ is designed to be used with equiripple
                filters (e.g., from `remez`) with unity or zero gain
                regions.

    n_fft : int
        The number of points to use for the FFT. Should be at least a
        few times larger than the signal length (see Notes).

    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(length(h) + 1) // 2``.

    See Also
    --------
    firwin
    firwin2
    remez

    Notes
    -----
    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection
    of an FFT length to estimate the complex cepstrum of the filter.

    In the case of the Hilbert method, the deviation from the ideal
    spectrum ``epsilon`` is related to the number of stopband zeros
    ``n_stop`` and FFT length ``n_fft`` as::

        epsilon = 2. * n_stop / n_fft

    For example, with 100 stopband zeros and a FFT length of 2048,
    ``epsilon = 0.0976``. If we conservatively assume that the number of
    stopband zeros is one less than the filter length, we can take the FFT
    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::

        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))

    This gives reasonable results for both the Hilbert and homomorphic
    methods, and gives the value used when ``n_fft=None``.

    Alternative implementations exist for creating minimum-phase filters,
    including zero inversion [2]_ and spectral factorization [3]_ [4]_.
    For more information, see:

        http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters

    Examples
    --------
    Create an optimal linear-phase filter, then convert it to minimum phase:

    >>> from scipy.signal import remez, minimum_phase, freqz, group_delay
    >>> import matplotlib.pyplot as plt
    >>> freq = [0, 0.2, 0.3, 1.0]
    >>> desired = [1, 0]
    >>> h_linear = remez(151, freq, desired, Hz=2.)

    Convert it to minimum phase:

    >>> h_min_hom = minimum_phase(h_linear, method='homomorphic')
    >>> h_min_hil = minimum_phase(h_linear, method='hilbert')

    Compare the three filters:

    >>> fig, axs = plt.subplots(4, figsize=(4, 8))
    >>> for h, style, color in zip((h_linear, h_min_hom, h_min_hil),
    ...                            ('-', '-', '--'), ('k', 'r', 'c')):
    ...     w, H = freqz(h)
    ...     w, gd = group_delay((h, 1))
    ...     w /= np.pi
    ...     axs[0].plot(h, color=color, linestyle=style)
    ...     axs[1].plot(w, np.abs(H), color=color, linestyle=style)
    ...     axs[2].plot(w, 20 * np.log10(np.abs(H)), color=color, linestyle=style)
    ...     axs[3].plot(w, gd, color=color, linestyle=style)
    >>> for ax in axs:
    ...     ax.grid(True, color='0.5')
    ...     ax.fill_between(freq[1:3], *ax.get_ylim(), color='#ffeeaa', zorder=1)
    >>> axs[0].set(xlim=[0, len(h_linear) - 1], ylabel='Amplitude', xlabel='Samples')
    >>> axs[1].legend(['Linear', 'Min-Hom', 'Min-Hil'], title='Phase')
    >>> for ax, ylim in zip(axs[1:], ([0, 1.1], [-150, 10], [-60, 60])):
    ...     ax.set(xlim=[0, 1], ylim=ylim, xlabel='Frequency')
    >>> axs[1].set(ylabel='Magnitude')
    >>> axs[2].set(ylabel='Magnitude (dB)')
    >>> axs[3].set(ylabel='Group delay')
    >>> plt.tight_layout()

    References
    ----------
    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and
           complex minimum phase digital FIR filters," Acoustics, Speech,
           and Signal Processing, 1999. Proceedings., 1999 IEEE International
           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.
           :doi:`10.1109/ICASSP.1999.756179`
    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR
           filters by direct factorization," Signal Processing,
           vol. 10, no. 4, pp. 369-383, Jun. 1986.
    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in
           Handbook for Digital Signal Processing, chapter 4,
           New York: Wiley-Interscience, 1993.
    .. [4] J. S. Lim, Advanced Topics in Signal Processing.
           Englewood Cliffs, N.J.: Prentice Hall, 1988.
    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,
           "Discrete-Time Signal Processing," 2nd edition.
           Upper Saddle River, N.J.: Prentice Hall, 1999.
    """  # noqa
    h = np.asarray(h)
    if np.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1-D and at least 2 samples long')
    n_half = len(h) // 2
    if not np.allclose(h[-n_half:][::-1], h[:n_half]):
        warnings.warn('h does not appear to by symmetric, conversion may '
                      'fail', RuntimeWarning)
    if not isinstance(method, str) or method not in \
            ('homomorphic', 'hilbert',):
        raise ValueError('method must be "homomorphic" or "hilbert", got %r'
                         % (method,))
    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
    n_fft = int(n_fft)
    if n_fft < len(h):
        raise ValueError('n_fft must be at least len(h)==%s' % len(h))
    if method == 'hilbert':
        w = np.arange(n_fft) * (2 * np.pi / n_fft * n_half)
        H = np.real(fft(h, n_fft) * np.exp(1j * w))
        dp = max(H) - 1
        ds = 0 - min(H)
        S = 4. / (np.sqrt(1+dp+ds) + np.sqrt(1-dp+ds)) ** 2
        H += ds
        H *= S
        H = np.sqrt(H, out=H)
        H += 1e-10  # ensure that the log does not explode
        h_minimum = _dhtm(H)
    else:  # method == 'homomorphic'
        # zero-pad; calculate the DFT
        h_temp = np.abs(fft(h, n_fft))
        # take 0.25*log(|H|**2) = 0.5*log(|H|)
        h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
        np.log(h_temp, out=h_temp)
        h_temp *= 0.5
        # IDFT
        h_temp = ifft(h_temp).real
        # multiply pointwise by the homomorphic filter
        # lmin[n] = 2u[n] - d[n]
        win = np.zeros(n_fft)
        win[0] = 1
        stop = (len(h) + 1) // 2
        win[1:stop] = 2
        if len(h) % 2:
            win[stop] = 1
        h_temp *= win
        h_temp = ifft(np.exp(fft(h_temp)))
        h_minimum = h_temp.real
    n_out = n_half + len(h) % 2
    return h_minimum[:n_out]
