"""Filter design."""
import math
import operator
import warnings

import numpy
import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
                   resize, pi, absolute, logspace, r_, sqrt, tan, log10,
                   arctan, arcsinh, sin, exp, cosh, arccosh, ceil, conjugate,
                   zeros, sinh, append, concatenate, prod, ones, full, array,
                   mintypecode)
from numpy.polynomial.polynomial import polyval as npp_polyval
from numpy.polynomial.polynomial import polyvalfromroots

from scipy import special, optimize, fft as sp_fft
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.optimize import root_scalar


__all__ = ['findfreqs', 'freqs', 'freqz', 'tf2zpk', 'zpk2tf', 'normalize',
           'lp2lp', 'lp2hp', 'lp2bp', 'lp2bs', 'bilinear', 'iirdesign',
           'iirfilter', 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel',
           'band_stop_obj', 'buttord', 'cheb1ord', 'cheb2ord', 'ellipord',
           'buttap', 'cheb1ap', 'cheb2ap', 'ellipap', 'besselap',
           'BadCoefficients', 'freqs_zpk', 'freqz_zpk',
           'tf2sos', 'sos2tf', 'zpk2sos', 'sos2zpk', 'group_delay',
           'sosfreqz', 'iirnotch', 'iirpeak', 'bilinear_zpk',
           'lp2lp_zpk', 'lp2hp_zpk', 'lp2bp_zpk', 'lp2bs_zpk',
           'gammatone', 'iircomb']


class BadCoefficients(UserWarning):
    """Warning about badly conditioned filter coefficients"""
    pass


abs = absolute


def _is_int_type(x):
    """
    Check if input is of a scalar integer type (so ``5`` and ``array(5)`` will
    pass, while ``5.0`` and ``array([5])`` will fail.
    """
    if np.ndim(x) != 0:
        # Older versions of NumPy did not raise for np.array([1]).__index__()
        # This is safe to remove when support for those versions is dropped
        return False
    try:
        operator.index(x)
    except TypeError:
        return False
    else:
        return True


def findfreqs(num, den, N, kind='ba'):
    """
    Find array of frequencies for computing the response of an analog filter.

    Parameters
    ----------
    num, den : array_like, 1-D
        The polynomial coefficients of the numerator and denominator of the
        transfer function of the filter or LTI system, where the coefficients
        are ordered from highest to lowest degree. Or, the roots  of the
        transfer function numerator and denominator (i.e., zeroes and poles).
    N : int
        The length of the array to be computed.
    kind : str {'ba', 'zp'}, optional
        Specifies whether the numerator and denominator are specified by their
        polynomial coefficients ('ba'), or their roots ('zp').

    Returns
    -------
    w : (N,) ndarray
        A 1-D array of frequencies, logarithmically spaced.

    Examples
    --------
    Find a set of nine frequencies that span the "interesting part" of the
    frequency response for the filter with the transfer function

        H(s) = s / (s^2 + 8s + 25)

    >>> from scipy import signal
    >>> signal.findfreqs([1, 0], [1, 8, 25], N=9)
    array([  1.00000000e-02,   3.16227766e-02,   1.00000000e-01,
             3.16227766e-01,   1.00000000e+00,   3.16227766e+00,
             1.00000000e+01,   3.16227766e+01,   1.00000000e+02])
    """
    if kind == 'ba':
        ep = atleast_1d(roots(den)) + 0j
        tz = atleast_1d(roots(num)) + 0j
    elif kind == 'zp':
        ep = atleast_1d(den) + 0j
        tz = atleast_1d(num) + 0j
    else:
        raise ValueError("input must be one of {'ba', 'zp'}")

    if len(ep) == 0:
        ep = atleast_1d(-1000) + 0j

    ez = r_['-1',
            numpy.compress(ep.imag >= 0, ep, axis=-1),
            numpy.compress((abs(tz) < 1e5) & (tz.imag >= 0), tz, axis=-1)]

    integ = abs(ez) < 1e-10
    hfreq = numpy.around(numpy.log10(numpy.max(3 * abs(ez.real + integ) +
                                               1.5 * ez.imag)) + 0.5)
    lfreq = numpy.around(numpy.log10(0.1 * numpy.min(abs(real(ez + integ)) +
                                                     2 * ez.imag)) - 0.5)

    w = logspace(lfreq, hfreq, N)
    return w


def freqs(b, a, worN=200, plot=None):
    """
    Compute frequency response of analog filter.

    Given the M-order numerator `b` and N-order denominator `a` of an analog
    filter, compute its frequency response::

             b[0]*(jw)**M + b[1]*(jw)**(M-1) + ... + b[M]
     H(w) = ----------------------------------------------
             a[0]*(jw)**N + a[1]*(jw)**(N-1) + ... + a[N]

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqs`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqz : Compute the frequency response of a digital filter.

    Notes
    -----
    Using Matplotlib's "plot" function as the callable for `plot` produces
    unexpected results, this plots the real part of the complex transfer
    function, not the magnitude. Try ``lambda w, h: plot(w, abs(h))``.

    Examples
    --------
    >>> from scipy.signal import freqs, iirfilter

    >>> b, a = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1')

    >>> w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid()
    >>> plt.show()

    """
    if worN is None:
        # For backwards compatibility
        w = findfreqs(b, a, 200)
    elif _is_int_type(worN):
        w = findfreqs(b, a, worN)
    else:
        w = atleast_1d(worN)

    s = 1j * w
    h = polyval(b, s) / polyval(a, s)
    if plot is not None:
        plot(w, h)

    return w, h


def freqs_zpk(z, p, k, worN=200):
    """
    Compute frequency response of analog filter.

    Given the zeros `z`, poles `p`, and gain `k` of a filter, compute its
    frequency response::

                (jw-z[0]) * (jw-z[1]) * ... * (jw-z[-1])
     H(w) = k * ----------------------------------------
                (jw-p[0]) * (jw-p[1]) * ... * (jw-p[-1])

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If None, then compute at 200 frequencies around the interesting parts
        of the response curve (determined by pole-zero locations). If a single
        integer, then compute at that many frequencies. Otherwise, compute the
        response at the angular frequencies (e.g., rad/s) given in `worN`.

    Returns
    -------
    w : ndarray
        The angular frequencies at which `h` was computed.
    h : ndarray
        The frequency response.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqz : Compute the frequency response of a digital filter in TF form
    freqz_zpk : Compute the frequency response of a digital filter in ZPK form

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    >>> from scipy.signal import freqs_zpk, iirfilter

    >>> z, p, k = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1',
    ...                     output='zpk')

    >>> w, h = freqs_zpk(z, p, k, worN=np.logspace(-1, 2, 1000))

    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude response [dB]')
    >>> plt.grid()
    >>> plt.show()

    """
    k = np.asarray(k)
    if k.size > 1:
        raise ValueError('k must be a single scalar gain')

    if worN is None:
        # For backwards compatibility
        w = findfreqs(z, p, 200, kind='zp')
    elif _is_int_type(worN):
        w = findfreqs(z, p, worN, kind='zp')
    else:
        w = worN

    w = atleast_1d(w)
    s = 1j * w
    num = polyvalfromroots(s, z)
    den = polyvalfromroots(s, p)
    h = k * num/den
    return w, h


def freqz(b, a=1, worN=512, whole=False, plot=None, fs=2*pi, include_nyquist=False):
    """
    Compute the frequency response of a digital filter.

    Given the M-order numerator `b` and N-order denominator `a` of a digital
    filter, compute its frequency response::

                 jw                 -jw              -jwM
        jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
     H(e  ) = ------ = -----------------------------------
                 jw                 -jw              -jwN
              A(e  )    a[0] + a[1]e    + ... + a[N]e

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    a : array_like
        Denominator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512). This is a convenient alternative to::

            np.linspace(0, fs if whole else fs/2, N, endpoint=include_nyquist)

        Using a number that is fast for FFT computations can result in
        faster computations (see Notes).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if worN is array_like.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqz`.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0
    include_nyquist : bool, optional
        If `whole` is False and `worN` is an integer, setting `include_nyquist` to True
        will include the last frequency (Nyquist frequency) and is otherwise ignored.

        .. versionadded:: 1.5.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz_zpk
    sosfreqz

    Notes
    -----
    Using Matplotlib's :func:`matplotlib.pyplot.plot` function as the callable
    for `plot` produces unexpected results, as this plots the real part of the
    complex transfer function, not the magnitude.
    Try ``lambda w, h: plot(w, np.abs(h))``.

    A direct computation via (R)FFT is used to compute the frequency response
    when the following conditions are met:

    1. An integer value is given for `worN`.
    2. `worN` is fast to compute via FFT (i.e.,
       `next_fast_len(worN) <scipy.fft.next_fast_len>` equals `worN`).
    3. The denominator coefficients are a single value (``a.shape[0] == 1``).
    4. `worN` is at least as long as the numerator coefficients
       (``worN >= b.shape[0]``).
    5. If ``b.ndim > 1``, then ``b.shape[-1] == 1``.

    For long FIR filters, the FFT approach can have lower error and be much
    faster than the equivalent direct polynomial calculation.

    Examples
    --------
    >>> from scipy import signal
    >>> b = signal.firwin(80, 0.5, window=('kaiser', 8))
    >>> w, h = signal.freqz(b)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax1 = plt.subplots()
    >>> ax1.set_title('Digital filter frequency response')

    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [rad/sample]')

    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> ax2.plot(w, angles, 'g')
    >>> ax2.set_ylabel('Angle (radians)', color='g')
    >>> ax2.grid()
    >>> ax2.axis('tight')
    >>> plt.show()

    Broadcasting Examples

    Suppose we have two FIR filters whose coefficients are stored in the
    rows of an array with shape (2, 25). For this demonstration, we'll
    use random data:

    >>> rng = np.random.default_rng()
    >>> b = rng.random((2, 25))

    To compute the frequency response for these two filters with one call
    to `freqz`, we must pass in ``b.T``, because `freqz` expects the first
    axis to hold the coefficients. We must then extend the shape with a
    trivial dimension of length 1 to allow broadcasting with the array
    of frequencies.  That is, we pass in ``b.T[..., np.newaxis]``, which has
    shape (25, 2, 1):

    >>> w, h = signal.freqz(b.T[..., np.newaxis], worN=1024)
    >>> w.shape
    (1024,)
    >>> h.shape
    (2, 1024)

    Now, suppose we have two transfer functions, with the same numerator
    coefficients ``b = [0.5, 0.5]``. The coefficients for the two denominators
    are stored in the first dimension of the 2-D array  `a`::

        a = [   1      1  ]
            [ -0.25, -0.5 ]

    >>> b = np.array([0.5, 0.5])
    >>> a = np.array([[1, 1], [-0.25, -0.5]])

    Only `a` is more than 1-D. To make it compatible for
    broadcasting with the frequencies, we extend it with a trivial dimension
    in the call to `freqz`:

    >>> w, h = signal.freqz(b, a[..., np.newaxis], worN=1024)
    >>> w.shape
    (1024,)
    >>> h.shape
    (2, 1024)

    """
    b = atleast_1d(b)
    a = atleast_1d(a)

    if worN is None:
        # For backwards compatibility
        worN = 512

    h = None

    if _is_int_type(worN):
        N = operator.index(worN)
        del worN
        if N < 0:
            raise ValueError('worN must be nonnegative, got %s' % (N,))
        lastpoint = 2 * pi if whole else pi
        # if include_nyquist is true and whole is false, w should include end point
        w = np.linspace(0, lastpoint, N, endpoint=include_nyquist and not whole)
        if (a.size == 1 and N >= b.shape[0] and
                sp_fft.next_fast_len(N) == N and
                (b.ndim == 1 or (b.shape[-1] == 1))):
            # if N is fast, 2 * N will be fast, too, so no need to check
            n_fft = N if whole else N * 2
            if np.isrealobj(b) and np.isrealobj(a):
                fft_func = sp_fft.rfft
            else:
                fft_func = sp_fft.fft
            h = fft_func(b, n=n_fft, axis=0)[:N]
            h /= a
            if fft_func is sp_fft.rfft and whole:
                # exclude DC and maybe Nyquist (no need to use axis_reverse
                # here because we can build reversal with the truncation)
                stop = -1 if n_fft % 2 == 1 else -2
                h_flip = slice(stop, 0, -1)
                h = np.concatenate((h, h[h_flip].conj()))
            if b.ndim > 1:
                # Last axis of h has length 1, so drop it.
                h = h[..., 0]
                # Rotate the first axis of h to the end.
                h = np.rollaxis(h, 0, h.ndim)
    else:
        w = atleast_1d(worN)
        del worN
        w = 2*pi*w/fs

    if h is None:  # still need to compute using freqs w
        zm1 = exp(-1j * w)
        h = (npp_polyval(zm1, b, tensor=False) /
             npp_polyval(zm1, a, tensor=False))

    w = w*fs/(2*pi)

    if plot is not None:
        plot(w, h)

    return w, h


def freqz_zpk(z, p, k, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in ZPK form.

    Given the Zeros, Poles and Gain of a digital filter, compute its frequency
    response:

    :math:`H(z)=k \prod_i (z - Z[i]) / \prod_j (z - P[j])`

    where :math:`k` is the `gain`, :math:`Z` are the `zeros` and :math:`P` are
    the `poles`.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqs : Compute the frequency response of an analog filter in TF form
    freqs_zpk : Compute the frequency response of an analog filter in ZPK form
    freqz : Compute the frequency response of a digital filter in TF form

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    Design a 4th-order digital Butterworth filter with cut-off of 100 Hz in a
    system with sample rate of 1000 Hz, and plot the frequency response:

    >>> from scipy import signal
    >>> z, p, k = signal.butter(4, 100, output='zpk', fs=1000)
    >>> w, h = signal.freqz_zpk(z, p, k, fs=1000)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(1, 1, 1)
    >>> ax1.set_title('Digital filter frequency response')

    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [Hz]')
    >>> ax1.grid()

    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> ax2.plot(w, angles, 'g')
    >>> ax2.set_ylabel('Angle [radians]', color='g')

    >>> plt.axis('tight')
    >>> plt.show()

    """
    z, p = map(atleast_1d, (z, p))

    if whole:
        lastpoint = 2 * pi
    else:
        lastpoint = pi

    if worN is None:
        # For backwards compatibility
        w = numpy.linspace(0, lastpoint, 512, endpoint=False)
    elif _is_int_type(worN):
        w = numpy.linspace(0, lastpoint, worN, endpoint=False)
    else:
        w = atleast_1d(worN)
        w = 2*pi*w/fs

    zm1 = exp(1j * w)
    h = k * polyvalfromroots(zm1, z) / polyvalfromroots(zm1, p)

    w = w*fs/(2*pi)

    return w, h


def group_delay(system, w=512, whole=False, fs=2*pi):
    r"""Compute the group delay of a digital filter.

    The group delay measures by how many samples amplitude envelopes of
    various spectral components of a signal are delayed by a filter.
    It is formally defined as the derivative of continuous (unwrapped) phase::

               d        jw
     D(w) = - -- arg H(e)
              dw

    Parameters
    ----------
    system : tuple of array_like (b, a)
        Numerator and denominator coefficients of a filter transfer function.
    w : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).

        If an array_like, compute the delay at the frequencies given. These
        are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which group delay was computed, in the same units
        as `fs`.  By default, `w` is normalized to the range [0, pi)
        (radians/sample).
    gd : ndarray
        The group delay.

    Notes
    -----
    The similar function in MATLAB is called `grpdelay`.

    If the transfer function :math:`H(z)` has zeros or poles on the unit
    circle, the group delay at corresponding frequencies is undefined.
    When such a case arises the warning is raised and the group delay
    is set to 0 at those frequencies.

    For the details of numerical computation of the group delay refer to [1]_.

    .. versionadded:: 0.16.0

    See Also
    --------
    freqz : Frequency response of a digital filter

    References
    ----------
    .. [1] Richard G. Lyons, "Understanding Digital Signal Processing,
           3rd edition", p. 830.

    Examples
    --------
    >>> from scipy import signal
    >>> b, a = signal.iirdesign(0.1, 0.3, 5, 50, ftype='cheby1')
    >>> w, gd = signal.group_delay((b, a))

    >>> import matplotlib.pyplot as plt
    >>> plt.title('Digital filter group delay')
    >>> plt.plot(w, gd)
    >>> plt.ylabel('Group delay [samples]')
    >>> plt.xlabel('Frequency [rad/sample]')
    >>> plt.show()

    """
    if w is None:
        # For backwards compatibility
        w = 512

    if _is_int_type(w):
        if whole:
            w = np.linspace(0, 2 * pi, w, endpoint=False)
        else:
            w = np.linspace(0, pi, w, endpoint=False)
    else:
        w = np.atleast_1d(w)
        w = 2*pi*w/fs

    b, a = map(np.atleast_1d, system)
    c = np.convolve(b, a[::-1])
    cr = c * np.arange(c.size)
    z = np.exp(-1j * w)
    num = np.polyval(cr[::-1], z)
    den = np.polyval(c[::-1], z)
    singular = np.absolute(den) < 10 * EPSILON
    if np.any(singular):
        warnings.warn(
            "The group delay is singular at frequencies [{0}], setting to 0".
            format(", ".join("{0:.3f}".format(ws) for ws in w[singular]))
        )

    gd = np.zeros_like(w)
    gd[~singular] = np.real(num[~singular] / den[~singular]) - a.size + 1

    w = w*fs/(2*pi)

    return w, gd


def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


def sosfreqz(sos, worN=512, whole=False, fs=2*pi):
    r"""
    Compute the frequency response of a digital filter in SOS format.

    Given `sos`, an array with shape (n, 6) of second order sections of
    a digital filter, compute the frequency response of the system function::

               B0(z)   B1(z)         B{n-1}(z)
        H(z) = ----- * ----- * ... * ---------
               A0(z)   A1(z)         A{n-1}(z)

    for z = exp(omega*1j), where B{k}(z) and A{k}(z) are numerator and
    denominator of the transfer function of the k-th second order section.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).  Using a number that is fast for FFT computations can result
        in faster computations (see Notes of `freqz`).

        If an array_like, compute the response at the frequencies given (must
        be 1-D). These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).

        .. versionadded:: 1.2.0

    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.

    See Also
    --------
    freqz, sosfilt

    Notes
    -----
    .. versionadded:: 0.19.0

    Examples
    --------
    Design a 15th-order bandpass filter in SOS format.

    >>> from scipy import signal
    >>> sos = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='sos')

    Compute the frequency response at 1500 points from DC to Nyquist.

    >>> w, h = signal.sosfreqz(sos, worN=1500)

    Plot the response.

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    >>> plt.ylim(-75, 5)
    >>> plt.grid(True)
    >>> plt.yticks([0, -20, -40, -60])
    >>> plt.ylabel('Gain [dB]')
    >>> plt.title('Frequency Response')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(w/np.pi, np.angle(h))
    >>> plt.grid(True)
    >>> plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    ...            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    >>> plt.ylabel('Phase [rad]')
    >>> plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    >>> plt.show()

    If the same filter is implemented as a single transfer function,
    numerical error corrupts the frequency response:

    >>> b, a = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='ba')
    >>> w, h = signal.freqz(b, a, worN=1500)
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    >>> plt.ylim(-75, 5)
    >>> plt.grid(True)
    >>> plt.yticks([0, -20, -40, -60])
    >>> plt.ylabel('Gain [dB]')
    >>> plt.title('Frequency Response')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(w/np.pi, np.angle(h))
    >>> plt.grid(True)
    >>> plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    ...            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    >>> plt.ylabel('Phase [rad]')
    >>> plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    >>> plt.show()

    """

    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    h = 1.
    for row in sos:
        w, rowh = freqz(row[:3], row[3:], worN=worN, whole=whole, fs=fs)
        h *= rowh
    return w, h


def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.

    The 1-D input vector `z` is split up into its complex (`zc`) and real (`zr`)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.

    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxpair

    Examples
    --------
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """

    z = atleast_1d(z)
    if z.size == 0:
        return z, z
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')

    if tol is None:
        # Get tolerance from dtype of input
        tol = 100 * np.finfo((1.0 * z).dtype).eps

    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    z = z[np.lexsort((abs(z.imag), z.real))]

    # Split reals from conjugate pairs
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # Input is entirely real
        return array([]), zr

    # Split positive and negative halves of conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = numpy.diff(concatenate(([0], same_real, [0])))
    run_starts = numpy.nonzero(diffs > 0)[0]
    run_stops = numpy.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[np.lexsort([abs(chunk.imag)])]

    # Check that negatives match positives
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    zc = (zp + zn.conj()) / 2

    return zc, zr


def _cplxpair(z, tol=None):
    """
    Sort into pairs of complex conjugates.

    Complex conjugates in `z` are sorted by increasing real part. In each
    pair, the number with negative imaginary part appears first.

    If pairs have identical real parts, they are sorted by increasing
    imaginary magnitude.

    Two complex numbers are considered a conjugate pair if their real and
    imaginary parts differ in magnitude by less than ``tol * abs(z)``.  The
    pairs are forced to be exact complex conjugates by averaging the positive
    and negative values.

    Purely real numbers are also sorted, but placed after the complex
    conjugate pairs. A number is considered real if its imaginary part is
    smaller than `tol` times the magnitude of the number.

    Parameters
    ----------
    z : array_like
        1-D input array to be sorted.
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)

    Returns
    -------
    y : ndarray
        Complex conjugate pairs followed by real numbers.

    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.

    See Also
    --------
    _cplxreal

    Examples
    --------
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> z = _cplxpair(a)
    >>> print(z)
    [ 1.-1.j  1.+1.j  2.-1.j  2.+1.j  2.-1.j  2.+1.j  2.-2.j  2.+2.j  1.+0.j
      3.+0.j  4.+0.j]
    """

    z = atleast_1d(z)
    if z.size == 0 or np.isrealobj(z):
        return np.sort(z)

    if z.ndim != 1:
        raise ValueError('z must be 1-D')

    zc, zr = _cplxreal(z, tol)

    # Interleave complex values and their conjugates, with negative imaginary
    # parts first in each pair
    zc = np.dstack((zc.conj(), zc)).flatten()
    z = np.append(zc, zr)
    return z


def tf2zpk(b, a):
    r"""Return zero, pole, gain (z, p, k) representation from a numerator,
    denominator representation of a linear filter.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    If some values of `b` are too close to 0, they are removed. In that case,
    a BadCoefficients warning is emitted.

    The `b` and `a` arrays are interpreted as coefficients for positive,
    descending powers of the transfer function variable. So the inputs
    :math:`b = [b_0, b_1, ..., b_M]` and :math:`a =[a_0, a_1, ..., a_N]`
    can represent an analog filter of the form:

    .. math::

        H(s) = \frac
        {b_0 s^M + b_1 s^{(M-1)} + \cdots + b_M}
        {a_0 s^N + a_1 s^{(N-1)} + \cdots + a_N}

    or a discrete-time filter of the form:

    .. math::

        H(z) = \frac
        {b_0 z^M + b_1 z^{(M-1)} + \cdots + b_M}
        {a_0 z^N + a_1 z^{(N-1)} + \cdots + a_N}

    This "positive powers" form is found more commonly in controls
    engineering. If `M` and `N` are equal (which is true for all filters
    generated by the bilinear transform), then this happens to be equivalent
    to the "negative powers" discrete-time form preferred in DSP:

    .. math::

        H(z) = \frac
        {b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}
        {a_0 + a_1 z^{-1} + \cdots + a_N z^{-N}}

    Although this is true for common filters, remember that this is not true
    in the general case. If `M` and `N` are not equal, the discrete-time
    transfer function coefficients must first be converted to the "positive
    powers" form before finding the poles and zeros.

    """
    b, a = normalize(b, a)
    b = (b + 0.0) / a[0]
    a = (a + 0.0) / a[0]
    k = b[0]
    b /= b[0]
    z = roots(b)
    p = roots(a)
    return z, p, k


def zpk2tf(z, p, k):
    """
    Return polynomial transfer function representation from zeros and poles

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    """
    z = atleast_1d(z)
    k = atleast_1d(k)
    if len(z.shape) > 1:
        temp = poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * poly(z[i])
    else:
        b = k * poly(z)
    a = atleast_1d(poly(p))

    # Use real output if possible. Copied from numpy.poly, since
    # we can't depend on a specific version of numpy.
    if issubclass(b.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = numpy.asarray(z, complex)
        pos_roots = numpy.compress(roots.imag > 0, roots)
        neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if numpy.all(numpy.sort_complex(neg_roots) ==
                         numpy.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = numpy.asarray(p, complex)
        pos_roots = numpy.compress(roots.imag > 0, roots)
        neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if numpy.all(numpy.sort_complex(neg_roots) ==
                         numpy.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a


def tf2sos(b, a, pairing='nearest'):
    """
    Return second-order sections from transfer function representation

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    pairing : {'nearest', 'keep_odd'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See `zpk2sos`.

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    zpk2sos, sosfilt

    Notes
    -----
    It is generally discouraged to convert from TF to SOS format, since doing
    so usually will not improve numerical precision errors. Instead, consider
    designing filters in ZPK format and converting directly to SOS. TF is
    converted to SOS by first converting to ZPK format, then converting
    ZPK to SOS.

    .. versionadded:: 0.16.0
    """
    return zpk2sos(*tf2zpk(b, a), pairing=pairing)


def sos2tf(sos):
    """
    Return a single transfer function from a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.

    Notes
    -----
    .. versionadded:: 0.16.0
    """
    sos = np.asarray(sos)
    result_type = sos.dtype
    if result_type.kind in 'bui':
        result_type = np.float64

    b = np.array([1], dtype=result_type)
    a = np.array([1], dtype=result_type)
    n_sections = sos.shape[0]
    for section in range(n_sections):
        b = np.polymul(b, sos[section, :3])
        a = np.polymul(a, sos[section, 3:])
    return b, a


def sos2zpk(sos):
    """
    Return zeros, poles, and gain of a series of second-order sections

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    The number of zeros and poles returned will be ``n_sections * 2``
    even if some of these are (effectively) zero.

    .. versionadded:: 0.16.0
    """
    sos = np.asarray(sos)
    n_sections = sos.shape[0]
    z = np.zeros(n_sections*2, np.complex128)
    p = np.zeros(n_sections*2, np.complex128)
    k = 1.
    for section in range(n_sections):
        zpk = tf2zpk(sos[section, :3], sos[section, 3:])
        z[2*section:2*section+len(zpk[0])] = zpk[0]
        p[2*section:2*section+len(zpk[1])] = zpk[1]
        k *= zpk[2]
    return z, p, k


def _nearest_real_complex_idx(fro, to, which):
    """Get the next closest real or complex element based on distance"""
    assert which in ('real', 'complex')
    order = np.argsort(np.abs(fro - to))
    mask = np.isreal(fro[order])
    if which == 'complex':
        mask = ~mask
    return order[np.nonzero(mask)[0][0]]


def zpk2sos(z, p, k, pairing='nearest'):
    """
    Return second-order sections from zeros, poles, and gain of a system

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {'nearest', 'keep_odd'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See Notes below.

    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.

    See Also
    --------
    sosfilt

    Notes
    -----
    The algorithm used to convert ZPK to SOS format is designed to
    minimize errors due to numerical precision issues. The pairing
    algorithm attempts to minimize the peak gain of each biquadratic
    section. This is done by pairing poles with the nearest zeros, starting
    with the poles closest to the unit circle.

    *Algorithms*

    The current algorithms are designed specifically for use with digital
    filters. (The output coefficients are not correct for analog filters.)

    The steps in the ``pairing='nearest'`` and ``pairing='keep_odd'``
    algorithms are mostly shared. The ``nearest`` algorithm attempts to
    minimize the peak gain, while ``'keep_odd'`` minimizes peak gain under
    the constraint that odd-order systems should retain one section
    as first order. The algorithm steps and are as follows:

    As a pre-processing step, add poles or zeros to the origin as
    necessary to obtain the same number of poles and zeros for pairing.
    If ``pairing == 'nearest'`` and there are an odd number of poles,
    add an additional pole and a zero at the origin.

    The following steps are then iterated over until no more poles or
    zeros remain:

    1. Take the (next remaining) pole (complex or real) closest to the
       unit circle to begin a new filter section.

    2. If the pole is real and there are no other remaining real poles [#]_,
       add the closest real zero to the section and leave it as a first
       order section. Note that after this step we are guaranteed to be
       left with an even number of real poles, complex poles, real zeros,
       and complex zeros for subsequent pairing iterations.

    3. Else:

        1. If the pole is complex and the zero is the only remaining real
           zero*, then pair the pole with the *next* closest zero
           (guaranteed to be complex). This is necessary to ensure that
           there will be a real zero remaining to eventually create a
           first-order section (thus keeping the odd order).

        2. Else pair the pole with the closest remaining zero (complex or
           real).

        3. Proceed to complete the second-order section by adding another
           pole and zero to the current pole and zero in the section:

            1. If the current pole and zero are both complex, add their
               conjugates.

            2. Else if the pole is complex and the zero is real, add the
               conjugate pole and the next closest real zero.

            3. Else if the pole is real and the zero is complex, add the
               conjugate zero and the real pole closest to those zeros.

            4. Else (we must have a real pole and real zero) add the next
               real pole closest to the unit circle, and then add the real
               zero closest to that pole.

    .. [#] This conditional can only be met for specific odd-order inputs
           with the ``pairing == 'keep_odd'`` method.

    .. versionadded:: 0.16.0

    Examples
    --------

    Design a 6th order low-pass elliptic digital filter for a system with a
    sampling rate of 8000 Hz that has a pass-band corner frequency of
    1000 Hz. The ripple in the pass-band should not exceed 0.087 dB, and
    the attenuation in the stop-band should be at least 90 dB.

    In the following call to `signal.ellip`, we could use ``output='sos'``,
    but for this example, we'll use ``output='zpk'``, and then convert to SOS
    format with `zpk2sos`:

    >>> from scipy import signal
    >>> z, p, k = signal.ellip(6, 0.087, 90, 1000/(0.5*8000), output='zpk')

    Now convert to SOS format.

    >>> sos = signal.zpk2sos(z, p, k)

    The coefficients of the numerators of the sections:

    >>> sos[:, :3]
    array([[ 0.0014154 ,  0.00248707,  0.0014154 ],
           [ 1.        ,  0.72965193,  1.        ],
           [ 1.        ,  0.17594966,  1.        ]])

    The symmetry in the coefficients occurs because all the zeros are on the
    unit circle.

    The coefficients of the denominators of the sections:

    >>> sos[:, 3:]
    array([[ 1.        , -1.32543251,  0.46989499],
           [ 1.        , -1.26117915,  0.6262586 ],
           [ 1.        , -1.25707217,  0.86199667]])

    The next example shows the effect of the `pairing` option.  We have a
    system with three poles and three zeros, so the SOS array will have
    shape (2, 6). The means there is, in effect, an extra pole and an extra
    zero at the origin in the SOS representation.

    >>> z1 = np.array([-1, -0.5-0.5j, -0.5+0.5j])
    >>> p1 = np.array([0.75, 0.8+0.1j, 0.8-0.1j])

    With ``pairing='nearest'`` (the default), we obtain

    >>> signal.zpk2sos(z1, p1, 1)
    array([[ 1.  ,  1.  ,  0.5 ,  1.  , -0.75,  0.  ],
           [ 1.  ,  1.  ,  0.  ,  1.  , -1.6 ,  0.65]])

    The first section has the zeros {-0.5-0.05j, -0.5+0.5j} and the poles
    {0, 0.75}, and the second section has the zeros {-1, 0} and poles
    {0.8+0.1j, 0.8-0.1j}. Note that the extra pole and zero at the origin
    have been assigned to different sections.

    With ``pairing='keep_odd'``, we obtain:

    >>> signal.zpk2sos(z1, p1, 1, pairing='keep_odd')
    array([[ 1.  ,  1.  ,  0.  ,  1.  , -0.75,  0.  ],
           [ 1.  ,  1.  ,  0.5 ,  1.  , -1.6 ,  0.65]])

    The extra pole and zero at the origin are in the same section.
    The first section is, in effect, a first-order section.

    """
    # TODO in the near future:
    # 1. Add SOS capability to `filtfilt`, `freqz`, etc. somehow (#3259).
    # 2. Make `decimate` use `sosfilt` instead of `lfilter`.
    # 3. Make sosfilt automatically simplify sections to first order
    #    when possible. Note this might make `sosfiltfilt` a bit harder (ICs).
    # 4. Further optimizations of the section ordering / pole-zero pairing.
    # See the wiki for other potential issues.

    valid_pairings = ['nearest', 'keep_odd']
    if pairing not in valid_pairings:
        raise ValueError('pairing must be one of %s, not %s'
                         % (valid_pairings, pairing))
    if len(z) == len(p) == 0:
        return array([[k, 0., 0., 1., 0., 0.]])

    # ensure we have the same number of poles and zeros, and make copies
    p = np.concatenate((p, np.zeros(max(len(z) - len(p), 0))))
    z = np.concatenate((z, np.zeros(max(len(p) - len(z), 0))))
    n_sections = (max(len(p), len(z)) + 1) // 2
    sos = zeros((n_sections, 6))

    if len(p) % 2 == 1 and pairing == 'nearest':
        p = np.concatenate((p, [0.]))
        z = np.concatenate((z, [0.]))
    assert len(p) == len(z)

    # Ensure we have complex conjugate pairs
    # (note that _cplxreal only gives us one element of each complex pair):
    z = np.concatenate(_cplxreal(z))
    p = np.concatenate(_cplxreal(p))

    p_sos = np.zeros((n_sections, 2), np.complex128)
    z_sos = np.zeros_like(p_sos)
    for si in range(n_sections):
        # Select the next "worst" pole
        p1_idx = np.argmin(np.abs(1 - np.abs(p)))
        p1 = p[p1_idx]
        p = np.delete(p, p1_idx)

        # Pair that pole with a zero

        if np.isreal(p1) and np.isreal(p).sum() == 0:
            # Special case to set a first-order section
            z1_idx = _nearest_real_complex_idx(z, p1, 'real')
            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)
            p2 = z2 = 0
        else:
            if not np.isreal(p1) and np.isreal(z).sum() == 1:
                # Special case to ensure we choose a complex zero to pair
                # with so later (setting up a first-order section)
                z1_idx = _nearest_real_complex_idx(z, p1, 'complex')
                assert not np.isreal(z[z1_idx])
            else:
                # Pair the pole with the closest zero (real or complex)
                z1_idx = np.argmin(np.abs(p1 - z))
            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)

            # Now that we have p1 and z1, figure out what p2 and z2 need to be
            if not np.isreal(p1):
                if not np.isreal(z1):  # complex pole, complex zero
                    p2 = p1.conj()
                    z2 = z1.conj()
                else:  # complex pole, real zero
                    p2 = p1.conj()
                    z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                    z2 = z[z2_idx]
                    assert np.isreal(z2)
                    z = np.delete(z, z2_idx)
            else:
                if not np.isreal(z1):  # real pole, complex zero
                    z2 = z1.conj()
                    p2_idx = _nearest_real_complex_idx(p, z1, 'real')
                    p2 = p[p2_idx]
                    assert np.isreal(p2)
                else:  # real pole, real zero
                    # pick the next "worst" pole to use
                    idx = np.nonzero(np.isreal(p))[0]
                    assert len(idx) > 0
                    p2_idx = idx[np.argmin(np.abs(np.abs(p[idx]) - 1))]
                    p2 = p[p2_idx]
                    # find a real zero to match the added pole
                    assert np.isreal(p2)
                    z2_idx = _nearest_real_complex_idx(z, p2, 'real')
                    z2 = z[z2_idx]
                    assert np.isreal(z2)
                    z = np.delete(z, z2_idx)
                p = np.delete(p, p2_idx)
        p_sos[si] = [p1, p2]
        z_sos[si] = [z1, z2]
    assert len(p) == len(z) == 0  # we've consumed all poles and zeros
    del p, z

    # Construct the system, reversing order so the "worst" are last
    p_sos = np.reshape(p_sos[::-1], (n_sections, 2))
    z_sos = np.reshape(z_sos[::-1], (n_sections, 2))
    gains = np.ones(n_sections, np.array(k).dtype)
    gains[0] = k
    for si in range(n_sections):
        x = zpk2tf(z_sos[si], p_sos[si], gains[si])
        sos[si] = np.concatenate(x)
    return sos


def _align_nums(nums):
    """Aligns the shapes of multiple numerators.

    Given an array of numerator coefficient arrays [[a_1, a_2,...,
    a_n],..., [b_1, b_2,..., b_m]], this function pads shorter numerator
    arrays with zero's so that all numerators have the same length. Such
    alignment is necessary for functions like 'tf2ss', which needs the
    alignment when dealing with SIMO transfer functions.

    Parameters
    ----------
    nums: array_like
        Numerator or list of numerators. Not necessarily with same length.

    Returns
    -------
    nums: array
        The numerator. If `nums` input was a list of numerators then a 2-D
        array with padded zeros for shorter numerators is returned. Otherwise
        returns ``np.asarray(nums)``.
    """
    try:
        # The statement can throw a ValueError if one
        # of the numerators is a single digit and another
        # is array-like e.g. if nums = [5, [1, 2, 3]]
        nums = asarray(nums)

        if not np.issubdtype(nums.dtype, np.number):
            raise ValueError("dtype of numerator is non-numeric")

        return nums

    except ValueError:
        nums = [np.atleast_1d(num) for num in nums]
        max_width = max(num.size for num in nums)

        # pre-allocate
        aligned_nums = np.zeros((len(nums), max_width))

        # Create numerators with padded zeros
        for index, num in enumerate(nums):
            aligned_nums[index, -num.size:] = num

        return aligned_nums


def normalize(b, a):
    """Normalize numerator/denominator of a continuous-time transfer function.

    If values of `b` are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    Parameters
    ----------
    b: array_like
        Numerator of the transfer function. Can be a 2-D array to normalize
        multiple transfer functions.
    a: array_like
        Denominator of the transfer function. At most 1-D.

    Returns
    -------
    num: array
        The numerator of the normalized transfer function. At least a 1-D
        array. A 2-D array if the input `num` is a 2-D array.
    den: 1-D array
        The denominator of the normalized transfer function.

    Notes
    -----
    Coefficients for both the numerator and denominator should be specified in
    descending exponent order (e.g., ``s^2 + 3s + 5`` would be represented as
    ``[1, 3, 5]``).
    """
    num, den = b, a

    den = np.atleast_1d(den)
    num = np.atleast_2d(_align_nums(num))

    if den.ndim != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if num.ndim > 2:
        raise ValueError("Numerator polynomial must be rank-1 or"
                         " rank-2 array.")
    if np.all(den == 0):
        raise ValueError("Denominator must have at least on nonzero element.")

    # Trim leading zeros in denominator, leave at least one.
    den = np.trim_zeros(den, 'f')

    # Normalize transfer function
    num, den = num / den[0], den / den[0]

    # Count numerator columns that are all zero
    leading_zeros = 0
    for col in num.T:
        if np.allclose(col, 0, atol=1e-14):
            leading_zeros += 1
        else:
            break

    # Trim leading zeros of numerator
    if leading_zeros > 0:
        warnings.warn("Badly conditioned filter coefficients (numerator): the "
                      "results may be meaningless", BadCoefficients)
        # Make sure at least one column remains
        if leading_zeros == num.shape[1]:
            leading_zeros -= 1
        num = num[:, leading_zeros:]

    # Squeeze first dimension if singular
    if num.shape[0] == 1:
        num = num[0, :]

    return num, den


def lp2lp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g. rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed low-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed low-pass filter.

    See Also
    --------
    lp2hp, lp2bp, lp2bs, bilinear
    lp2lp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    Examples
    --------

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> lp2 = signal.lti(*signal.lp2lp(lp.num, lp.den, 2))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_lp2, p_lp2 = lp2.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_lp2, label='Transformed Lowpass')
    >>> plt.semilogx()
    >>> plt.grid()
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.legend()

    """
    a, b = map(atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    M = max((d, n))
    pwo = pow(wo, numpy.arange(M - 1, -1, -1))
    start1 = max((n - d, 0))
    start2 = max((d - n, 0))
    b = b * pwo[start1] / pwo[start2:]
    a = a * pwo[start1] / pwo[start1:]
    return normalize(b, a)


def lp2hp(b, a, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency, in
    transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed high-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed high-pass filter.

    See Also
    --------
    lp2lp, lp2bp, lp2bs, bilinear
    lp2hp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> hp = signal.lti(*signal.lp2hp(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_hp, p_hp = hp.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_hp, label='Highpass')
    >>> plt.semilogx()
    >>> plt.grid()
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.legend()

    """
    a, b = map(atleast_1d, (a, b))
    try:
        wo = float(wo)
    except TypeError:
        wo = float(wo[0])
    d = len(a)
    n = len(b)
    if wo != 1:
        pwo = pow(wo, numpy.arange(max((d, n))))
    else:
        pwo = numpy.ones(max((d, n)), b.dtype.char)
    if d >= n:
        outa = a[::-1] * pwo
        outb = resize(b, (d,))
        outb[n:] = 0.0
        outb[:n] = b[::-1] * pwo[:n]
    else:
        outb = b[::-1] * pwo
        outa = resize(a, (n,))
        outa[d:] = 0.0
        outa[:d] = a[::-1] * pwo[:d]

    return normalize(outb, outa)


def lp2bp(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-pass filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-pass filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bs, bilinear
    lp2bp_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.0])
    >>> bp = signal.lti(*signal.lp2bp(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_bp, p_bp = bp.bode(w)

    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_bp, label='Bandpass')
    >>> plt.semilogx()
    >>> plt.grid()
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.legend()
    """

    a, b = map(atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = mintypecode((a, b))
    ma = max([N, D])
    Np = N + ma
    Dp = D + ma
    bprime = numpy.empty(Np + 1, artype)
    aprime = numpy.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * b[N - i] * (wosq) ** (i - k) / bw ** i
        bprime[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, i + 1):
                if ma - i + 2 * k == j:
                    val += comb(i, k) * a[D - i] * (wosq) ** (i - k) / bw ** i
        aprime[Dp - j] = val

    return normalize(bprime, aprime)


def lp2bs(b, a, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, in transfer function ('ba') representation.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    b : array_like
        Numerator polynomial coefficients of the transformed band-stop filter.
    a : array_like
        Denominator polynomial coefficients of the transformed band-stop filter.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, bilinear
    lp2bs_zpk

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> lp = signal.lti([1.0], [1.0, 1.5])
    >>> bs = signal.lti(*signal.lp2bs(lp.num, lp.den))
    >>> w, mag_lp, p_lp = lp.bode()
    >>> w, mag_bs, p_bs = bs.bode(w)
    >>> plt.plot(w, mag_lp, label='Lowpass')
    >>> plt.plot(w, mag_bs, label='Bandstop')
    >>> plt.semilogx()
    >>> plt.grid()
    >>> plt.xlabel('Frequency [rad/s]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.legend()
    """
    a, b = map(atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = mintypecode((a, b))
    M = max([N, D])
    Np = M + M
    Dp = M + M
    bprime = numpy.empty(Np + 1, artype)
    aprime = numpy.empty(Dp + 1, artype)
    wosq = wo * wo
    for j in range(Np + 1):
        val = 0.0
        for i in range(0, N + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * b[N - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        bprime[Np - j] = val
    for j in range(Dp + 1):
        val = 0.0
        for i in range(0, D + 1):
            for k in range(0, M - i + 1):
                if i + 2 * k == j:
                    val += (comb(M - i, k) * a[D - i] *
                            (wosq) ** (M - i - k) * bw ** i)
        aprime[Dp - j] = val

    return normalize(bprime, aprime)


def bilinear(b, a, fs=1.0):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    b : array_like
        Numerator of the analog filter transfer function.
    a : array_like
        Denominator of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Numerator of the transformed digital filter transfer function.
    p : ndarray
        Denominator of the transformed digital filter transfer function.

    See Also
    --------
    lp2lp, lp2hp, lp2bp, lp2bs
    bilinear_zpk

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass',
    ...                                   analog=True))
    >>> filtz = signal.lti(*signal.bilinear(filts.num, filts.den, fs))
    >>> wz, hz = signal.freqz(filtz.num, filtz.den)
    >>> ws, hs = signal.freqs(filts.num, filts.den, worN=fs*wz)

    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.grid()
    """
    fs = float(fs)
    a, b = map(atleast_1d, (a, b))
    D = len(a) - 1
    N = len(b) - 1
    artype = float
    M = max([N, D])
    Np = M
    Dp = M
    bprime = numpy.empty(Np + 1, artype)
    aprime = numpy.empty(Dp + 1, artype)
    for j in range(Np + 1):
        val = 0.0
        for i in range(N + 1):
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += (comb(i, k) * comb(M - i, l) * b[N - i] *
                                pow(2 * fs, i) * (-1) ** k)
        bprime[j] = real(val)
    for j in range(Dp + 1):
        val = 0.0
        for i in range(D + 1):
            for k in range(i + 1):
                for l in range(M - i + 1):
                    if k + l == j:
                        val += (comb(i, k) * comb(M - i, l) * a[D - i] *
                                pow(2 * fs, i) * (-1) ** k)
        aprime[j] = real(val)

    return normalize(bprime, aprime)


def _validate_gpass_gstop(gpass, gstop):

    if gpass <= 0.0:
        raise ValueError("gpass should be larger than 0.0")
    elif gstop <= 0.0:
        raise ValueError("gstop should be larger than 0.0")
    elif gpass > gstop:
        raise ValueError("gpass should be smaller than gstop")


def iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip', output='ba',
              fs=None):
    """Complete IIR digital and analog filter design.

    Given passband and stopband frequencies and gains, construct an analog or
    digital IIR filter of minimum order for a given basic type. Return the
    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order
    sections ('sos') form.

    Parameters
    ----------
    wp, ws : float or array like, shape (2,)
        Passband and stopband edge frequencies. Possible values are scalars
        (for lowpass and highpass filters) or ranges (for bandpass and bandstop
        filters).
        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
        Note, that for bandpass and bandstop filters passband must lie strictly
        inside stopband or vice versa.
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies

    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.ticker

    >>> wp = 0.2
    >>> ws = 0.3
    >>> gpass = 1
    >>> gstop = 40

    >>> system = signal.iirdesign(wp, ws, gpass, gstop)
    >>> w, h = signal.freqz(*system)

    >>> fig, ax1 = plt.subplots()
    >>> ax1.set_title('Digital filter frequency response')
    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [rad/sample]')
    >>> ax1.grid()
    >>> ax1.set_ylim([-120, 20])
    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> ax2.plot(w, angles, 'g')
    >>> ax2.set_ylabel('Angle (radians)', color='g')
    >>> ax2.grid()
    >>> ax2.axis('tight')
    >>> ax2.set_ylim([-6, 1])
    >>> nticks = 8
    >>> ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    >>> ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

    """
    try:
        ordfunc = filter_dict[ftype][1]
    except KeyError as e:
        raise ValueError("Invalid IIR filter type: %s" % ftype) from e
    except IndexError as e:
        raise ValueError(("%s does not have order selection. Use "
                          "iirfilter function.") % ftype) from e

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)

    if wp.shape[0] != ws.shape[0] or wp.shape not in [(1,), (2,)]:
        raise ValueError("wp and ws must have one or two elements each, and"
                         "the same shape, got %s and %s"
                         % (wp.shape, ws.shape))
    if wp.shape[0] == 2:
        if wp[0] < 0 or ws[0] < 0:
            raise ValueError("Values for wp, ws can't be negative")
        elif 1 < wp[1] or 1 < ws[1]:
            raise ValueError("Values for wp, ws can't be larger than 1")
        elif not((ws[0] < wp[0] and wp[1] < ws[1]) or
            (wp[0] < ws[0] and ws[1] < wp[1])):
            raise ValueError("Passband must lie strictly inside stopband"
                         " or vice versa")

    band_type = 2 * (len(wp) - 1)
    band_type += 1
    if wp[0] >= ws[0]:
        band_type += 1

    btype = {1: 'lowpass', 2: 'highpass',
             3: 'bandstop', 4: 'bandpass'}[band_type]

    N, Wn = ordfunc(wp, ws, gpass, gstop, analog=analog, fs=fs)
    return iirfilter(N, Wn, rp=gpass, rs=gstop, analog=analog, btype=btype,
                     ftype=ftype, output=output, fs=fs)


def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.

    Design an Nth-order digital or analog filter and return the filter
    coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:

            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    output : {'ba', 'zpk', 'sos'}, optional
        Filter form of the output:

            - second-order sections (recommended): 'sos'
            - numerator/denominator (default)    : 'ba'
            - pole-zero                          : 'zpk'

        In general the second-order sections ('sos') form  is
        recommended because inferring the coefficients for the
        numerator/denominator form ('ba') suffers from numerical
        instabilities. For reasons of backward compatibility the default
        form is the numerator/denominator form ('ba'), where the 'b'
        and the 'a' in 'ba' refer to the commonly used names of the
        coefficients used.

        Note: Using the second-order sections form ('sos') is sometimes
        associated with additional computational costs: for
        data-intense use cases it is therefore recommended to also
        investigate the numerator/denominator form ('ba').

    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec

    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Generate a 17th-order Chebyshev II analog bandpass filter from 50 Hz to
    200 Hz and plot the frequency response:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.iirfilter(17, [2*np.pi*50, 2*np.pi*200], rs=60,
    ...                         btype='band', analog=True, ftype='cheby2')
    >>> w, h = signal.freqs(b, a, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    Create a digital filter with the same properties, in a system with
    sampling rate of 2000 Hz, and plot the frequency response. (Second-order
    sections implementation is required to ensure stability of a filter of
    this order):

    >>> sos = signal.iirfilter(17, [50, 200], rs=60, btype='band',
    ...                        analog=False, ftype='cheby2', fs=2000,
    ...                        output='sos')
    >>> w, h = signal.sosfreqz(sos, 2000, fs=2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()

    """
    ftype, btype, output = [x.lower() for x in (ftype, btype, output)]
    Wn = asarray(Wn)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = 2*Wn/fs

    try:
        btype = band_dict[btype]
    except KeyError as e:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype) from e

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError as e:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype) from e

    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)

    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    elif typefunc == besselap:
        z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    elif typefunc == ellipap:
        if rs is None or rp is None:
            raise ValueError("Both rp and rs must be provided to design an "
                             "elliptic filter.")
        z, p, k = typefunc(N, rp, rs)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # Pre-warp frequencies for digital filter design
    if not analog:
        if numpy.any(Wn <= 0) or numpy.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies "
                                 "must be 0 < Wn < fs/2 (fs={} -> fs/2={})".format(fs, fs/2))
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if numpy.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn for lowpass or highpass filter')

        if btype == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = sqrt(warped[0] * warped[1])
        except IndexError as e:
            raise ValueError('Wn must specify start and stop frequencies for bandpass or bandstop '
                             'filter') from e

        if btype == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k)


def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk
    bilinear

    Notes
    -----
    .. versionadded:: 1.1.0

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass', analog=True,
    ...                                   output='zpk'))
    >>> filtz = signal.lti(*signal.bilinear_zpk(filts.zeros, filts.poles,
    ...                                         filts.gain, fs))
    >>> wz, hz = signal.freqz_zpk(filtz.zeros, filtz.poles, filtz.gain)
    >>> ws, hs = signal.freqs_zpk(filts.zeros, filts.poles, filts.gain,
    ...                           worN=fs*wz)
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)),
    ...              label=r'$|H_z(e^{j \omega})|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)),
    ...              label=r'$|H(j \omega)|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.grid()
    """
    z = atleast_1d(z)
    p = atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = append(z_z, -ones(degree))

    # Compensate for gain change
    k_z = k * real(prod(fs2 - z) / prod(fs2 - p))

    return z_z, p_z, k_z


def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.

    See Also
    --------
    lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2lp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s}{\omega_0}

    .. versionadded:: 1.1.0

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp


def lp2hp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.

    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{\omega_0}{s}

    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.

    .. versionadded:: 1.1.0

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = append(z_hp, zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * real(prod(-z) / prod(-p))

    return z_hp, p_hp, k_hp


def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear
    lp2bp

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}

    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.

    .. versionadded:: 1.1.0

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = concatenate((z_lp + sqrt(z_lp**2 - wo**2),
                        z_lp - sqrt(z_lp**2 - wo**2)))
    p_bp = concatenate((p_lp + sqrt(p_lp**2 - wo**2),
                        p_lp - sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = append(z_bp, zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp


def lp2bs_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandstop filter.

    Return an analog band-stop filter with center frequency `wo` and
    stopband width `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired stopband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired stopband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-stop filter transfer function.
    p : ndarray
        Poles of the transformed band-stop filter transfer function.
    k : float
        System gain of the transformed band-stop filter.

    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, bilinear
    lp2bs

    Notes
    -----
    This is derived from the s-plane substitution

    .. math:: s \rightarrow \frac{s \cdot \mathrm{BW}}{s^2 + {\omega_0}^2}

    This is the "wideband" transformation, producing a stopband with
    geometric (log frequency) symmetry about `wo`.

    .. versionadded:: 1.1.0

    """
    z = atleast_1d(z)
    p = atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Invert to a highpass filter with desired bandwidth
    z_hp = (bw/2) / z
    p_hp = (bw/2) / p

    # Square root needs to produce complex result, not NaN
    z_hp = z_hp.astype(complex)
    p_hp = p_hp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bs = concatenate((z_hp + sqrt(z_hp**2 - wo**2),
                        z_hp - sqrt(z_hp**2 - wo**2)))
    p_bs = concatenate((p_hp + sqrt(p_hp**2 - wo**2),
                        p_hp - sqrt(p_hp**2 - wo**2)))

    # Move any zeros that were at infinity to the center of the stopband
    z_bs = append(z_bs, full(degree, +1j*wo))
    z_bs = append(z_bs, full(degree, -1j*wo))

    # Cancel out gain change caused by inversion
    k_bs = k * real(prod(-z) / prod(-p))

    return z_bs, p_bs, k_bs


def butter(N, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Butterworth digital and analog filter design.

    Design an Nth-order digital or analog Butterworth filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        The critical frequency or frequencies. For lowpass and highpass
        filters, Wn is a scalar; for bandpass and bandstop filters,
        Wn is a length-2 sequence.

        For a Butterworth filter, this is the point at which the gain
        drops to 1/sqrt(2) that of the passband (the "-3 dB point").

        For digital filters, `Wn` are in the same units as `fs`.  By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    buttord, buttap

    Notes
    -----
    The Butterworth filter has maximally flat frequency response in the
    passband.

    The ``'sos'`` output parameter was added in 0.16.0.

    If the transfer function form ``[b, a]`` is requested, numerical
    problems can occur since the conversion between roots and
    the polynomial coefficients is a numerically sensitive operation,
    even for N >= 4. It is recommended to work with the SOS
    representation.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth filter frequency response')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 15 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='butter', fs=fs)


def cheby1(N, rp, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type I digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type I filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type I filters, this is the point in the transition band at which
        the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    cheb1ord, cheb1ap

    Notes
    -----
    The Chebyshev type I filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the passband and increased ringing in the step response.

    Type I filters roll off faster than Type II (`cheby2`), but Type II
    filters do not have any ripple in the passband.

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.cheby1(4, 5, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type I frequency response (rp=5)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-5, color='green') # rp
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.cheby1(10, 1, 15, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 15 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='cheby1', fs=fs)


def cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Chebyshev type II digital and analog filter design.

    Design an Nth-order digital or analog Chebyshev type II filter and
    return the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For Type II filters, this is the point in the transition band at which
        the gain first reaches -`rs`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    cheb2ord, cheb2ap

    Notes
    -----
    The Chebyshev type II filter maximizes the rate of cutoff between the
    frequency response's passband and stopband, at the expense of ripple in
    the stopband and increased ringing in the step response.

    Type II filters do not roll off as fast as Type I (`cheby1`).

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.cheby2(4, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev Type II frequency response (rs=40)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-40, color='green') # rs
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 17 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.cheby2(12, 20, 17, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 17 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.show()
    """
    return iirfilter(N, Wn, rs=rs, btype=btype, analog=analog,
                     output=output, ftype='cheby2', fs=fs)


def ellip(N, rp, rs, Wn, btype='low', analog=False, output='ba', fs=None):
    """
    Elliptic (Cauer) digital and analog filter design.

    Design an Nth-order digital or analog elliptic filter and return
    the filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    rp : float
        The maximum ripple allowed below unity gain in the passband.
        Specified in decibels, as a positive number.
    rs : float
        The minimum attenuation required in the stop band.
        Specified in decibels, as a positive number.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For elliptic filters, this is the point in the transition band at
        which the gain first drops below -`rp`.

        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)

        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter. Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    See Also
    --------
    ellipord, ellipap

    Notes
    -----
    Also known as Cauer or Zolotarev filters, the elliptical filter maximizes
    the rate of transition between the frequency response's passband and
    stopband, at the expense of ripple in both, and increased ringing in the
    step response.

    As `rp` approaches 0, the elliptical filter becomes a Chebyshev
    type II filter (`cheby2`). As `rs` approaches 0, it becomes a Chebyshev
    type I filter (`cheby1`). As both approach 0, it becomes a Butterworth
    filter (`butter`).

    The equiripple passband has N maxima or minima (for example, a
    5th-order filter has 3 maxima and 2 minima). Consequently, the DC gain is
    unity for odd-order filters, or -rp dB for even-order filters.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Design an analog filter and plot its frequency response, showing the
    critical points:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.ellip(4, 5, 40, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Elliptic filter frequency response (rp=5, rs=40)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green') # cutoff frequency
    >>> plt.axhline(-40, color='green') # rs
    >>> plt.axhline(-5, color='green') # rp
    >>> plt.show()

    Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz

    >>> t = np.linspace(0, 1, 1000, False)  # 1 second
    >>> sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    >>> fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    >>> ax1.plot(t, sig)
    >>> ax1.set_title('10 Hz and 20 Hz sinusoids')
    >>> ax1.axis([0, 1, -2, 2])

    Design a digital high-pass filter at 17 Hz to remove the 10 Hz tone, and
    apply it to the signal. (It's recommended to use second-order sections
    format when filtering, to avoid numerical error with transfer function
    (``ba``) format):

    >>> sos = signal.ellip(8, 1, 100, 17, 'hp', fs=1000, output='sos')
    >>> filtered = signal.sosfilt(sos, sig)
    >>> ax2.plot(t, filtered)
    >>> ax2.set_title('After 17 Hz high-pass filter')
    >>> ax2.axis([0, 1, -2, 2])
    >>> ax2.set_xlabel('Time [seconds]')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    return iirfilter(N, Wn, rs=rs, rp=rp, btype=btype, analog=analog,
                     output=output, ftype='elliptic', fs=fs)


def bessel(N, Wn, btype='low', analog=False, output='ba', norm='phase',
           fs=None):
    """
    Bessel/Thomson digital and analog filter design.

    Design an Nth-order digital or analog Bessel filter and return the
    filter coefficients.

    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies (defined
        by the `norm` parameter).
        For analog filters, `Wn` is an angular frequency (e.g., rad/s).

        For digital filters, `Wn` are in the same units as `fs`.  By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned. (See Notes.)
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.
    norm : {'phase', 'delay', 'mag'}, optional
        Critical frequency normalization:

        ``phase``
            The filter is normalized such that the phase response reaches its
            midpoint at angular (e.g. rad/s) frequency `Wn`. This happens for
            both low-pass and high-pass filters, so this is the
            "phase-matched" case.

            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.

            This is the default, and matches MATLAB's implementation.

        ``delay``
            The filter is normalized such that the group delay in the passband
            is 1/`Wn` (e.g., seconds). This is the "natural" type obtained by
            solving Bessel polynomials.

        ``mag``
            The filter is normalized such that the gain magnitude is -3 dB at
            angular frequency `Wn`.

        .. versionadded:: 0.18.0
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.

    Notes
    -----
    Also known as a Thomson filter, the analog Bessel filter has maximally
    flat group delay and maximally linear phase response, with very little
    ringing in the step response. [1]_

    The Bessel is inherently an analog filter. This function generates digital
    Bessel filters using the bilinear transform, which does not preserve the
    phase response of the analog filter. As such, it is only approximately
    correct at frequencies below about fs/4. To get maximally-flat group
    delay at higher frequencies, the analog Bessel filter must be transformed
    using phase-preserving techniques.

    See `besselap` for implementation details and references.

    The ``'sos'`` output parameter was added in 0.16.0.

    Examples
    --------
    Plot the phase-normalized frequency response, showing the relationship
    to the Butterworth's cutoff frequency (green):

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.butter(4, 100, 'low', analog=True)
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)), color='silver', ls='dashed')
    >>> b, a = signal.bessel(4, 100, 'low', analog=True, norm='phase')
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)))
    >>> plt.title('Bessel filter magnitude response (with Butterworth)')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(100, color='green')  # cutoff frequency
    >>> plt.show()

    and the phase midpoint:

    >>> plt.figure()
    >>> plt.semilogx(w, np.unwrap(np.angle(h)))
    >>> plt.axvline(100, color='green')  # cutoff frequency
    >>> plt.axhline(-np.pi, color='red')  # phase midpoint
    >>> plt.title('Bessel filter phase response')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Phase [radians]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.show()

    Plot the magnitude-normalized frequency response, showing the -3 dB cutoff:

    >>> b, a = signal.bessel(3, 10, 'low', analog=True, norm='mag')
    >>> w, h = signal.freqs(b, a)
    >>> plt.semilogx(w, 20 * np.log10(np.abs(h)))
    >>> plt.axhline(-3, color='red')  # -3 dB magnitude
    >>> plt.axvline(10, color='green')  # cutoff frequency
    >>> plt.title('Magnitude-normalized Bessel filter frequency response')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.show()

    Plot the delay-normalized filter, showing the maximally-flat group delay
    at 0.1 seconds:

    >>> b, a = signal.bessel(5, 1/0.1, 'low', analog=True, norm='delay')
    >>> w, h = signal.freqs(b, a)
    >>> plt.figure()
    >>> plt.semilogx(w[1:], -np.diff(np.unwrap(np.angle(h)))/np.diff(w))
    >>> plt.axhline(0.1, color='red')  # 0.1 seconds group delay
    >>> plt.title('Bessel filter group delay')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Group delay [seconds]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.show()

    References
    ----------
    .. [1] Thomson, W.E., "Delay Networks having Maximally Flat Frequency
           Characteristics", Proceedings of the Institution of Electrical
           Engineers, Part III, November 1949, Vol. 96, No. 44, pp. 487-490.

    """
    return iirfilter(N, Wn, btype=btype, analog=analog,
                     output=output, ftype='bessel_'+norm, fs=fs)


def maxflat():
    pass


def yulewalk():
    pass


def band_stop_obj(wp, ind, passb, stopb, gpass, gstop, type):
    """
    Band Stop Objective Function for order minimization.

    Returns the non-integer order for an analog band stop filter.

    Parameters
    ----------
    wp : scalar
        Edge of passband `passb`.
    ind : int, {0, 1}
        Index specifying which `passb` edge to vary (0 or 1).
    passb : ndarray
        Two element sequence of fixed passband edges.
    stopb : ndarray
        Two element sequence of fixed stopband edges.
    gstop : float
        Amount of attenuation in stopband in dB.
    gpass : float
        Amount of ripple in the passband in dB.
    type : {'butter', 'cheby', 'ellip'}
        Type of filter.

    Returns
    -------
    n : scalar
        Filter order (possibly non-integer).

    """

    _validate_gpass_gstop(gpass, gstop)

    passbC = passb.copy()
    passbC[ind] = wp
    nat = (stopb * (passbC[0] - passbC[1]) /
           (stopb ** 2 - passbC[0] * passbC[1]))
    nat = min(abs(nat))

    if type == 'butter':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = (log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat)))
    elif type == 'cheby':
        GSTOP = 10 ** (0.1 * abs(gstop))
        GPASS = 10 ** (0.1 * abs(gpass))
        n = arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) / arccosh(nat)
    elif type == 'ellip':
        GSTOP = 10 ** (0.1 * gstop)
        GPASS = 10 ** (0.1 * gpass)
        arg1 = sqrt((GPASS - 1.0) / (GSTOP - 1.0))
        arg0 = 1.0 / nat
        d0 = special.ellipk([arg0 ** 2, 1 - arg0 ** 2])
        d1 = special.ellipk([arg1 ** 2, 1 - arg1 ** 2])
        n = (d0[0] * d1[1] / (d0[1] * d1[0]))
    else:
        raise ValueError("Incorrect type: %s" % type)
    return n


def buttord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Butterworth filter order selection.

    Return the order of the lowest order digital or analog Butterworth filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Butterworth filter which meets specs.
    wn : ndarray or float
        The Butterworth natural frequency (i.e. the "3dB frequency"). Should
        be used with `butter` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `butter`.

    See Also
    --------
    butter : Filter design using order and critical points
    cheb1ord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design an analog bandpass filter with passband within 3 dB from 20 to
    50 rad/s, while rejecting at least -40 dB below 14 and above 60 rad/s.
    Plot its frequency response, showing the passband and stopband
    constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
    >>> b, a = signal.butter(N, Wn, 'band', True)
    >>> w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth bandpass filter fit to constraints')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
    >>> plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
    >>> plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
    >>> plt.axis([10, 100, -60, 3])
    >>> plt.show()

    """

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2*wp/fs
        ws = 2*ws/fs

    filter_type = 2 * (len(wp) - 1)
    filter_type += 1
    if wp[0] >= ws[0]:
        filter_type += 1

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:            # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:          # stop
        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop,
                                       'butter'),
                                 disp=0)
        passb[0] = wp0
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop,
                                       'butter'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:          # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat))))

    # Find the Butterworth natural frequency WN (or the "3dB" frequency")
    # to give exactly gpass at passb.
    try:
        W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord))
    except ZeroDivisionError:
        W0 = 1.0
        print("Warning, order is zero...check input parameters.")

    # now convert this frequency back from lowpass prototype
    # to the original analog filter

    if filter_type == 1:  # low
        WN = W0 * passb
    elif filter_type == 2:  # high
        WN = passb / W0
    elif filter_type == 3:  # stop
        WN = numpy.empty(2, float)
        discr = sqrt((passb[1] - passb[0]) ** 2 +
                     4 * W0 ** 2 * passb[0] * passb[1])
        WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
        WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
        WN = numpy.sort(abs(WN))
    elif filter_type == 4:  # pass
        W0 = numpy.array([-W0, W0], float)
        WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
              sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                   passb[0] * passb[1]))
        WN = numpy.sort(abs(WN))
    else:
        raise ValueError("Bad type: %s" % filter_type)

    if not analog:
        wn = (2.0 / pi) * arctan(WN)
    else:
        wn = WN

    if len(wn) == 1:
        wn = wn[0]

    if fs is not None:
        wn = wn*fs/2

    return ord, wn


def cheb1ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type I filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type I
    filter that loses no more than `gpass` dB in the passband and has at
    least `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type I filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby1` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby1`.

    See Also
    --------
    cheby1 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design a digital lowpass filter such that the passband is within 3 dB up
    to 0.2*(fs/2), while rejecting at least -40 dB above 0.3*(fs/2). Plot its
    frequency response, showing the passband and stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.cheb1ord(0.2, 0.3, 3, 40)
    >>> b, a = signal.cheby1(N, 3, Wn, 'low')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev I lowpass filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, 0.2, 0.2, .01], [-3, -3, -99, -99], '0.9', lw=0) # stop
    >>> plt.fill([0.3, 0.3,   2,   2], [ 9, -40, -40,  9], '0.9', lw=0) # pass
    >>> plt.axis([0.08, 1, -60, 3])
    >>> plt.show()

    """

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2*wp/fs
        ws = 2*ws/fs

    filter_type = 2 * (len(wp) - 1)
    if wp[0] < ws[0]:
        filter_type += 1
    else:
        filter_type += 2

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:           # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:     # stop
        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[0] = wp0
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:  # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) /
                   arccosh(nat)))

    # Natural frequencies are just the passband edges
    if not analog:
        wn = (2.0 / pi) * arctan(passb)
    else:
        wn = passb

    if len(wn) == 1:
        wn = wn[0]

    if fs is not None:
        wn = wn*fs/2

    return ord, wn


def cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Chebyshev type II filter order selection.

    Return the order of the lowest order digital or analog Chebyshev Type II
    filter that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for a Chebyshev type II filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `cheby2` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `cheby2`.

    See Also
    --------
    cheby2 : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design a digital bandstop filter which rejects -60 dB from 0.2*(fs/2) to
    0.5*(fs/2), while staying within 3 dB below 0.1*(fs/2) or above
    0.6*(fs/2). Plot its frequency response, showing the passband and
    stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.cheb2ord([0.1, 0.6], [0.2, 0.5], 3, 60)
    >>> b, a = signal.cheby2(N, 60, Wn, 'stop')
    >>> w, h = signal.freqz(b, a)
    >>> plt.semilogx(w / np.pi, 20 * np.log10(abs(h)))
    >>> plt.title('Chebyshev II bandstop filter fit to constraints')
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.01, .1, .1, .01], [-3,  -3, -99, -99], '0.9', lw=0) # stop
    >>> plt.fill([.2,  .2, .5,  .5], [ 9, -60, -60,   9], '0.9', lw=0) # pass
    >>> plt.fill([.6,  .6,  2,   2], [-99, -3,  -3, -99], '0.9', lw=0) # stop
    >>> plt.axis([0.06, 1, -80, 3])
    >>> plt.show()

    """

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2*wp/fs
        ws = 2*ws/fs

    filter_type = 2 * (len(wp) - 1)
    if wp[0] < ws[0]:
        filter_type += 1
    else:
        filter_type += 2

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:           # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:     # stop
        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[0] = wp0
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop, 'cheby'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:  # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))) /
                   arccosh(nat)))

    # Find frequency where analog response is -gpass dB.
    # Then convert back from low-pass prototype to the original filter.

    new_freq = cosh(1.0 / ord * arccosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0))))
    new_freq = 1.0 / new_freq

    if filter_type == 1:
        nat = passb / new_freq
    elif filter_type == 2:
        nat = passb * new_freq
    elif filter_type == 3:
        nat = numpy.empty(2, float)
        nat[0] = (new_freq / 2.0 * (passb[0] - passb[1]) +
                  sqrt(new_freq ** 2 * (passb[1] - passb[0]) ** 2 / 4.0 +
                       passb[1] * passb[0]))
        nat[1] = passb[1] * passb[0] / nat[0]
    elif filter_type == 4:
        nat = numpy.empty(2, float)
        nat[0] = (1.0 / (2.0 * new_freq) * (passb[0] - passb[1]) +
                  sqrt((passb[1] - passb[0]) ** 2 / (4.0 * new_freq ** 2) +
                       passb[1] * passb[0]))
        nat[1] = passb[0] * passb[1] / nat[0]

    if not analog:
        wn = (2.0 / pi) * arctan(nat)
    else:
        wn = nat

    if len(wn) == 1:
        wn = wn[0]

    if fs is not None:
        wn = wn*fs/2

    return ord, wn


_POW10_LOG10 = np.log(10)

def _pow10m1(x):
    """10 ** x - 1 for x near 0"""
    return np.expm1(_POW10_LOG10 * x)


def ellipord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Elliptic (Cauer) filter order selection.

    Return the order of the lowest order digital or analog elliptic filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.

    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.

        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:

            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]

        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    ord : int
        The lowest order for an Elliptic (Cauer) filter that meets specs.
    wn : ndarray or float
        The Chebyshev natural frequency (the "3dB frequency") for use with
        `ellip` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `ellip`.

    See Also
    --------
    ellip : Filter design using order and critical points
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec

    Examples
    --------
    Design an analog highpass filter such that the passband is within 3 dB
    above 30 rad/s, while rejecting -60 dB at 10 rad/s. Plot its
    frequency response, showing the passband and stopband constraints in gray.

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> N, Wn = signal.ellipord(30, 10, 3, 60, True)
    >>> b, a = signal.ellip(N, 3, 60, Wn, 'high', True)
    >>> w, h = signal.freqs(b, a, np.logspace(0, 3, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Elliptical highpass filter fit to constraints')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([.1, 10,  10,  .1], [1e4, 1e4, -60, -60], '0.9', lw=0) # stop
    >>> plt.fill([30, 30, 1e9, 1e9], [-99,  -3,  -3, -99], '0.9', lw=0) # pass
    >>> plt.axis([1, 300, -80, 3])
    >>> plt.show()

    """

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2*wp/fs
        ws = 2*ws/fs

    filter_type = 2 * (len(wp) - 1)
    filter_type += 1
    if wp[0] >= ws[0]:
        filter_type += 1

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:           # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:     # stop
        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop, 'ellip'),
                                 disp=0)
        passb[0] = wp0
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop, 'ellip'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:  # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    arg1_sq = _pow10m1(0.1 * gpass) / _pow10m1(0.1 * gstop)
    arg0 = 1.0 / nat
    d0 = special.ellipk(arg0 ** 2), special.ellipkm1(arg0 ** 2)
    d1 = special.ellipk(arg1_sq), special.ellipkm1(arg1_sq)
    ord = int(ceil(d0[0] * d1[1] / (d0[1] * d1[0])))

    if not analog:
        wn = arctan(passb) * 2.0 / pi
    else:
        wn = passb

    if len(wn) == 1:
        wn = wn[0]

    if fs is not None:
        wn = wn*fs/2

    return ord, wn


def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.

    See Also
    --------
    butter : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = numpy.array([])
    m = numpy.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -numpy.exp(1j * pi * m / (2 * N))
    k = 1
    return z, p, k


def cheb1ap(N, rp):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rp` decibels of ripple in the passband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    cheby1 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero error
        # Even order filters have DC gain of -rp dB
        return numpy.array([]), numpy.array([]), 10**(-rp/20)
    z = numpy.array([])

    # Ripple factor (epsilon)
    eps = numpy.sqrt(10 ** (0.1 * rp) - 1.0)
    mu = 1.0 / N * arcsinh(1 / eps)

    # Arrange poles in an ellipse on the left half of the S-plane
    m = numpy.arange(-N+1, N, 2)
    theta = pi * m / (2*N)
    p = -sinh(mu + 1j*theta)

    k = numpy.prod(-p, axis=0).real
    if N % 2 == 0:
        k = k / sqrt((1 + eps * eps))

    return z, p, k


def cheb2ap(N, rs):
    """
    Return (z,p,k) for Nth-order Chebyshev type I analog lowpass filter.

    The returned filter prototype has `rs` decibels of ripple in the stopband.

    The filter's angular (e.g. rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first reaches ``-rs``.

    See Also
    --------
    cheby2 : Filter design function using this prototype

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        return numpy.array([]), numpy.array([]), 1

    # Ripple factor (epsilon)
    de = 1.0 / sqrt(10 ** (0.1 * rs) - 1)
    mu = arcsinh(1.0 / de) / N

    if N % 2:
        m = numpy.concatenate((numpy.arange(-N+1, 0, 2),
                               numpy.arange(2, N, 2)))
    else:
        m = numpy.arange(-N+1, N, 2)

    z = -conjugate(1j / sin(m * pi / (2.0 * N)))

    # Poles around the unit circle like Butterworth
    p = -exp(1j * pi * numpy.arange(-N+1, N, 2) / (2 * N))
    # Warp into Chebyshev II
    p = sinh(mu) * p.real + 1j * cosh(mu) * p.imag
    p = 1.0 / p

    k = (numpy.prod(-p, axis=0) / numpy.prod(-z, axis=0)).real
    return z, p, k


EPSILON = 2e-16

# number of terms in solving degree equation
_ELLIPDEG_MMAX = 7

def _ellipdeg(n, m1):
    """Solve degree equation using nomes

    Given n, m1, solve
       n * K(m) / K'(m) = K1(m1) / K1'(m1)
    for m

    See [1], Eq. (49)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """
    K1 = special.ellipk(m1)
    K1p = special.ellipkm1(m1)

    q1 = np.exp(-np.pi * K1p / K1)
    q = q1 ** (1/n)

    mnum = np.arange(_ELLIPDEG_MMAX + 1)
    mden = np.arange(1, _ELLIPDEG_MMAX + 2)

    num = np.sum(q ** (mnum * (mnum+1)))
    den = 1 + 2 * np.sum(q ** (mden**2))

    return 16 * q * (num / den) ** 4


# Maximum number of iterations in Landen transformation recursion
# sequence.  10 is conservative; unit tests pass with 4, Orfanidis
# (see _arc_jac_cn [1]) suggests 5.
_ARC_JAC_SN_MAXITER = 10

def _arc_jac_sn(w, m):
    """Inverse Jacobian elliptic sn

    Solve for z in w = sn(z, m)

    Parameters
    ----------
    w - complex scalar
        argument

    m - scalar
        modulus; in interval [0, 1]


    See [1], Eq. (56)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

    """

    def _complement(kx):
        # (1-k**2) ** 0.5; the expression below 
        # works for small kx
        return ((1 - kx) * (1 + kx)) ** 0.5


    k = m ** 0.5

    if k > 1:
        return np.nan
    elif k == 1:
        return np.arctanh(w)

    ks = [k]
    niter = 0
    while ks[-1] != 0:
        k_ = ks[-1]
        k_p = _complement(k_)
        ks.append((1 - k_p) / (1 + k_p))
        niter += 1
        if niter > _ARC_JAC_SN_MAXITER:
            raise ValueError('Landen transformation not converging')

    K = np.product(1 + np.array(ks[1:])) * np.pi/2

    wns = [w]

    for kn, knext in zip(ks[:-1], ks[1:]):
        wn = wns[-1]
        wnext = ( 2 * wn
                  /
                 ( (1 + knext) * (1 + _complement(kn * wn)) ) )
        wns.append(wnext)

    u = 2 / np.pi * np.arcsin(wns[-1])

    z = K * u
    return z


def _arc_jac_sc1(w, m):
    """Real inverse Jacobian sc, with complementary modulus

    Solve for z in w = sc(z, 1-m)

    w - real scalar

    m - modulus

    From [1], sc(z, m) = -i * sn(i * z, 1 - m)

    References
    ----------
    .. [1] https://functions.wolfram.com/EllipticFunctions/JacobiSC/introductions/JacobiPQs/ShowAll.html, 
       "Representations through other Jacobi functions"

    """

    zcomplex = _arc_jac_sn(1j * w, m)
    if abs(zcomplex.real) > 1e-14:
        raise ValueError

    return zcomplex.imag


def ellipap(N, rp, rs):
    """Return (z,p,k) of Nth-order elliptic analog lowpass filter.

    The filter is a normalized prototype that has `rp` decibels of ripple
    in the passband and a stopband `rs` decibels down.

    The filter's angular (e.g., rad/s) cutoff frequency is normalized to 1,
    defined as the point at which the gain first drops below ``-rp``.

    See Also
    --------
    ellip : Filter design function using this prototype

    References
    ----------
    .. [1] Lutova, Tosic, and Evans, "Filter Design for Signal Processing",
           Chapters 5 and 12.

    .. [2] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    elif N == 0:
        # Avoid divide-by-zero warning
        # Even order filters have DC gain of -rp dB
        return numpy.array([]), numpy.array([]), 10**(-rp/20)
    elif N == 1:
        p = -sqrt(1.0 / _pow10m1(0.1 * rp))
        k = -p
        z = []
        return asarray(z), asarray(p), k

    eps_sq = _pow10m1(0.1 * rp)

    eps = np.sqrt(eps_sq)
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0:
        raise ValueError("Cannot design a filter with given rp and rs"
                         " specifications.")

    val = special.ellipk(ck1_sq), special.ellipkm1(ck1_sq)

    m = _ellipdeg(N, ck1_sq)

    capk = special.ellipk(m)

    j = numpy.arange(1 - N % 2, N, 2)
    jj = len(j)

    [s, c, d, phi] = special.ellipj(j * capk / N, m * numpy.ones(jj))
    snew = numpy.compress(abs(s) > EPSILON, s, axis=-1)
    z = 1.0 / (sqrt(m) * snew)
    z = 1j * z
    z = numpy.concatenate((z, conjugate(z)))

    r = _arc_jac_sc1(1. / eps, ck1_sq)
    v0 = capk * r / (N * val[0])

    [sv, cv, dv, phi] = special.ellipj(v0, 1 - m)
    p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)

    if N % 2:
        newp = numpy.compress(abs(p.imag) > EPSILON *
                              numpy.sqrt(numpy.sum(p * numpy.conjugate(p),
                                                   axis=0).real),
                              p, axis=-1)
        p = numpy.concatenate((p, conjugate(newp)))
    else:
        p = numpy.concatenate((p, conjugate(p)))

    k = (numpy.prod(-p, axis=0) / numpy.prod(-z, axis=0)).real
    if N % 2 == 0:
        k = k / numpy.sqrt((1 + eps_sq))

    return z, p, k


# TODO: Make this a real public function scipy.misc.ff
def _falling_factorial(x, n):
    r"""
    Return the factorial of `x` to the `n` falling.

    This is defined as:

    .. math::   x^\underline n = (x)_n = x (x-1) \cdots (x-n+1)

    This can more efficiently calculate ratios of factorials, since:

    n!/m! == falling_factorial(n, n-m)

    where n >= m

    skipping the factors that cancel out

    the usual factorial n! == ff(n, n)
    """
    val = 1
    for k in range(x - n + 1, x + 1):
        val *= k
    return val


def _bessel_poly(n, reverse=False):
    """
    Return the coefficients of Bessel polynomial of degree `n`

    If `reverse` is true, a reverse Bessel polynomial is output.

    Output is a list of coefficients:
    [1]                   = 1
    [1,  1]               = 1*s   +  1
    [1,  3,  3]           = 1*s^2 +  3*s   +  3
    [1,  6, 15, 15]       = 1*s^3 +  6*s^2 + 15*s   +  15
    [1, 10, 45, 105, 105] = 1*s^4 + 10*s^3 + 45*s^2 + 105*s + 105
    etc.

    Output is a Python list of arbitrary precision long ints, so n is only
    limited by your hardware's memory.

    Sequence is http://oeis.org/A001498, and output can be confirmed to
    match http://oeis.org/A001498/b001498.txt :

    >>> i = 0
    >>> for n in range(51):
    ...     for x in _bessel_poly(n, reverse=True):
    ...         print(i, x)
    ...         i += 1

    """
    if abs(int(n)) != n:
        raise ValueError("Polynomial order must be a nonnegative integer")
    else:
        n = int(n)  # np.int32 doesn't work, for instance

    out = []
    for k in range(n + 1):
        num = _falling_factorial(2*n - k, n)
        den = 2**(n - k) * math.factorial(k)
        out.append(num // den)

    if reverse:
        return out[::-1]
    else:
        return out


def _campos_zeros(n):
    """
    Return approximate zero locations of Bessel polynomials y_n(x) for order
    `n` using polynomial fit (Campos-Calderon 2011)
    """
    if n == 1:
        return asarray([-1+0j])

    s = npp_polyval(n, [0, 0, 2, 0, -3, 1])
    b3 = npp_polyval(n, [16, -8]) / s
    b2 = npp_polyval(n, [-24, -12, 12]) / s
    b1 = npp_polyval(n, [8, 24, -12, -2]) / s
    b0 = npp_polyval(n, [0, -6, 0, 5, -1]) / s

    r = npp_polyval(n, [0, 0, 2, 1])
    a1 = npp_polyval(n, [-6, -6]) / r
    a2 = 6 / r

    k = np.arange(1, n+1)
    x = npp_polyval(k, [0, a1, a2])
    y = npp_polyval(k, [b0, b1, b2, b3])

    return x + 1j*y


def _aberth(f, fp, x0, tol=1e-15, maxiter=50):
    """
    Given a function `f`, its first derivative `fp`, and a set of initial
    guesses `x0`, simultaneously find the roots of the polynomial using the
    Aberth-Ehrlich method.

    ``len(x0)`` should equal the number of roots of `f`.

    (This is not a complete implementation of Bini's algorithm.)
    """

    N = len(x0)

    x = array(x0, complex)
    beta = np.empty_like(x0)

    for iteration in range(maxiter):
        alpha = -f(x) / fp(x)  # Newton's method

        # Model "repulsion" between zeros
        for k in range(N):
            beta[k] = np.sum(1/(x[k] - x[k+1:]))
            beta[k] += np.sum(1/(x[k] - x[:k]))

        x += alpha / (1 + alpha * beta)

        if not all(np.isfinite(x)):
            raise RuntimeError('Root-finding calculation failed')

        # Mekwi: The iterative process can be stopped when |hn| has become
        # less than the largest error one is willing to permit in the root.
        if all(abs(alpha) <= tol):
            break
    else:
        raise Exception('Zeros failed to converge')

    return x


def _bessel_zeros(N):
    """
    Find zeros of ordinary Bessel polynomial of order `N`, by root-finding of
    modified Bessel function of the second kind
    """
    if N == 0:
        return asarray([])

    # Generate starting points
    x0 = _campos_zeros(N)

    # Zeros are the same for exp(1/x)*K_{N+0.5}(1/x) and Nth-order ordinary
    # Bessel polynomial y_N(x)
    def f(x):
        return special.kve(N+0.5, 1/x)

    # First derivative of above
    def fp(x):
        return (special.kve(N-0.5, 1/x)/(2*x**2) -
                special.kve(N+0.5, 1/x)/(x**2) +
                special.kve(N+1.5, 1/x)/(2*x**2))

    # Starting points converge to true zeros
    x = _aberth(f, fp, x0)

    # Improve precision using Newton's method on each
    for i in range(len(x)):
        x[i] = optimize.newton(f, x[i], fp, tol=1e-15)

    # Average complex conjugates to make them exactly symmetrical
    x = np.mean((x, x[::-1].conj()), 0)

    # Zeros should sum to -1
    if abs(np.sum(x) + 1) > 1e-15:
        raise RuntimeError('Generated zeros are inaccurate')

    return x


def _norm_factor(p, k):
    """
    Numerically find frequency shift to apply to delay-normalized filter such
    that -3 dB point is at 1 rad/sec.

    `p` is an array_like of polynomial poles
    `k` is a float gain

    First 10 values are listed in "Bessel Scale Factors" table,
    "Bessel Filters Polynomials, Poles and Circuit Elements 2003, C. Bond."
    """
    p = asarray(p, dtype=complex)

    def G(w):
        """
        Gain of filter
        """
        return abs(k / prod(1j*w - p))

    def cutoff(w):
        """
        When gain = -3 dB, return 0
        """
        return G(w) - 1/np.sqrt(2)

    return optimize.newton(cutoff, 1.5)


def besselap(N, norm='phase'):
    """
    Return (z,p,k) for analog prototype of an Nth-order Bessel filter.

    Parameters
    ----------
    N : int
        The order of the filter.
    norm : {'phase', 'delay', 'mag'}, optional
        Frequency normalization:

        ``phase``
            The filter is normalized such that the phase response reaches its
            midpoint at an angular (e.g., rad/s) cutoff frequency of 1. This
            happens for both low-pass and high-pass filters, so this is the
            "phase-matched" case. [6]_

            The magnitude response asymptotes are the same as a Butterworth
            filter of the same order with a cutoff of `Wn`.

            This is the default, and matches MATLAB's implementation.

        ``delay``
            The filter is normalized such that the group delay in the passband
            is 1 (e.g., 1 second). This is the "natural" type obtained by
            solving Bessel polynomials

        ``mag``
            The filter is normalized such that the gain magnitude is -3 dB at
            angular frequency 1. This is called "frequency normalization" by
            Bond. [1]_

        .. versionadded:: 0.18.0

    Returns
    -------
    z : ndarray
        Zeros of the transfer function. Is always an empty array.
    p : ndarray
        Poles of the transfer function.
    k : scalar
        Gain of the transfer function. For phase-normalized, this is always 1.

    See Also
    --------
    bessel : Filter design function using this prototype

    Notes
    -----
    To find the pole locations, approximate starting points are generated [2]_
    for the zeros of the ordinary Bessel polynomial [3]_, then the
    Aberth-Ehrlich method [4]_ [5]_ is used on the Kv(x) Bessel function to
    calculate more accurate zeros, and these locations are then inverted about
    the unit circle.

    References
    ----------
    .. [1] C.R. Bond, "Bessel Filter Constants",
           http://www.crbond.com/papers/bsf.pdf
    .. [2] Campos and Calderon, "Approximate closed-form formulas for the
           zeros of the Bessel Polynomials", :arXiv:`1105.0957`.
    .. [3] Thomson, W.E., "Delay Networks having Maximally Flat Frequency
           Characteristics", Proceedings of the Institution of Electrical
           Engineers, Part III, November 1949, Vol. 96, No. 44, pp. 487-490.
    .. [4] Aberth, "Iteration Methods for Finding all Zeros of a Polynomial
           Simultaneously", Mathematics of Computation, Vol. 27, No. 122,
           April 1973
    .. [5] Ehrlich, "A modified Newton method for polynomials", Communications
           of the ACM, Vol. 10, Issue 2, pp. 107-108, Feb. 1967,
           :DOI:`10.1145/363067.363115`
    .. [6] Miller and Bohn, "A Bessel Filter Crossover, and Its Relation to
           Others", RaneNote 147, 1998, https://www.ranecommercial.com/legacy/note147.html

    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    if N == 0:
        p = []
        k = 1
    else:
        # Find roots of reverse Bessel polynomial
        p = 1/_bessel_zeros(N)

        a_last = _falling_factorial(2*N, N) // 2**N

        # Shift them to a different normalization if required
        if norm in ('delay', 'mag'):
            # Normalized for group delay of 1
            k = a_last
            if norm == 'mag':
                # -3 dB magnitude point is at 1 rad/sec
                norm_factor = _norm_factor(p, k)
                p /= norm_factor
                k = norm_factor**-N * a_last
        elif norm == 'phase':
            # Phase-matched (1/2 max phase shift at 1 rad/sec)
            # Asymptotes are same as Butterworth filter
            p *= 10**(-math.log10(a_last)/N)
            k = 1
        else:
            raise ValueError('normalization not understood')

    return asarray([]), asarray(p, dtype=complex), float(k)


def iirnotch(w0, Q, fs=2.0):
    """
    Design second-order IIR notch digital filter.

    A notch filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    Parameters
    ----------
    w0 : float
        Frequency to remove from a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirpeak

    Notes
    -----
    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the 60 Hz component from a
    signal sampled at 200 Hz, using a quality factor Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 200.0  # Sample frequency (Hz)
    >>> f0 = 60.0  # Frequency to be removed from signal (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design notch filter
    >>> b, a = signal.iirnotch(f0, Q, fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 100])
    >>> ax[0].set_ylim([-25, 10])
    >>> ax[0].grid()
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 100])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid()
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "notch", fs)


def iirpeak(w0, Q, fs=2.0):
    """
    Design second-order IIR peak (resonant) digital filter.

    A peak filter is a band-pass filter with a narrow bandwidth
    (high quality factor). It rejects components outside a narrow
    frequency band.

    Parameters
    ----------
    w0 : float
        Frequency to be retained in a signal. If `fs` is specified, this is in
        the same units as `fs`. By default, it is a normalized scalar that must
        satisfy  ``0 < w0 < 1``, with ``w0 = 1`` corresponding to half of the
        sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        peak filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    See Also
    --------
    iirnotch

    Notes
    -----
    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot filter to remove the frequencies other than the 300 Hz
    component from a signal sampled at 1000 Hz, using a quality factor Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 1000.0  # Sample frequency (Hz)
    >>> f0 = 300.0  # Frequency to be retained (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design peak filter
    >>> b, a = signal.iirpeak(f0, Q, fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 500])
    >>> ax[0].set_ylim([-50, 10])
    >>> ax[0].grid()
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 500])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid()
    >>> plt.show()
    """

    return _design_notch_peak_filter(w0, Q, "peak", fs)


def _design_notch_peak_filter(w0, Q, ftype, fs=2.0):
    """
    Design notch or peak digital filter.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. If `fs` is specified,
        this is in the same units as `fs`. By default, it is a normalized
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:

            - notch filter : ``notch``
            - peak filter  : ``peak``
    fs : float, optional
        The sampling frequency of the digital system.

        .. versionadded:: 1.2.0:

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """

    # Guarantee that the inputs are floats
    w0 = float(w0)
    Q = float(Q)
    w0 = 2*w0/fs

    # Checks if w0 is within the range
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError("w0 should be such that 0 < w0 < 1")

    # Get bandwidth
    bw = w0/Q

    # Normalize inputs
    bw = bw*np.pi
    w0 = w0*np.pi

    # Compute -3dB attenuation
    gb = 1/np.sqrt(2)

    if ftype == "notch":
        # Compute beta: formula 11.3.4 (p.575) from reference [1]
        beta = (np.sqrt(1.0-gb**2.0)/gb)*np.tan(bw/2.0)
    elif ftype == "peak":
        # Compute beta: formula 11.3.19 (p.579) from reference [1]
        beta = (gb/np.sqrt(1.0-gb**2.0))*np.tan(bw/2.0)
    else:
        raise ValueError("Unknown ftype.")

    # Compute gain: formula 11.3.6 (p.575) from reference [1]
    gain = 1.0/(1.0+beta)

    # Compute numerator b and denominator a
    # formulas 11.3.7 (p.575) and 11.3.21 (p.579)
    # from reference [1]
    if ftype == "notch":
        b = gain*np.array([1.0, -2.0*np.cos(w0), 1.0])
    else:
        b = (1.0-gain)*np.array([1.0, 0.0, -1.0])
    a = np.array([1.0, -2.0*gain*np.cos(w0), (2.0*gain-1.0)])

    return b, a


def iircomb(w0, Q, ftype='notch', fs=2.0):
    """
    Design IIR notching or peaking digital comb filter.

    A notching comb filter is a band-stop filter with a narrow bandwidth
    (high quality factor). It rejects a narrow frequency band and
    leaves the rest of the spectrum little changed.

    A peaking comb filter is a band-pass filter with a narrow bandwidth
    (high quality factor). It rejects components outside a narrow
    frequency band.

    Parameters
    ----------
    w0 : float
        Frequency to attenuate (notching) or boost (peaking). If `fs` is
        specified, this is in the same units as `fs`. By default, it is
        a normalized scalar that must satisfy  ``0 < w0 < 1``, with
        ``w0 = 1`` corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : {'notch', 'peak'}
        The type of comb filter generated by the function. If 'notch', then
        it returns a filter with notches at frequencies ``0``, ``w0``,
        ``2 * w0``, etc. If 'peak', then it returns a filter with peaks at
        frequencies ``0.5 * w0``, ``1.5 * w0``, ``2.5 * w0```, etc.
        Default is 'notch'.
    fs : float, optional
        The sampling frequency of the signal. Default is 2.0.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.

    Raises
    ------
    ValueError
        If `w0` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `fs` is not divisible by `w0`, if `ftype`
        is not 'notch' or 'peak'

    See Also
    --------
    iirnotch
    iirpeak

    Notes
    -----
    For implementation details, see [1]_. The TF implementation of the
    comb filter is numerically stable even at higher orders due to the
    use of a single repeated pole, which won't suffer from precision loss.

    References
    ----------
    .. [1] Sophocles J. Orfanidis, "Introduction To Signal Processing",
           Prentice-Hall, 1996

    Examples
    --------
    Design and plot notching comb filter at 20 Hz for a
    signal sampled at 200 Hz, using quality factor Q = 30

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> fs = 200.0  # Sample frequency (Hz)
    >>> f0 = 20.0  # Frequency to be removed from signal (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design notching comb filter
    >>> b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> response = abs(h)
    >>> # To avoid divide by zero when graphing
    >>> response[response == 0] = 1e-20
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(abs(response)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 100])
    >>> ax[0].set_ylim([-30, 10])
    >>> ax[0].grid()
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 100])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid()
    >>> plt.show()

    Design and plot peaking comb filter at 250 Hz for a
    signal sampled at 1000 Hz, using quality factor Q = 30

    >>> fs = 1000.0  # Sample frequency (Hz)
    >>> f0 = 250.0  # Frequency to be retained (Hz)
    >>> Q = 30.0  # Quality factor
    >>> # Design peaking filter
    >>> b, a = signal.iircomb(f0, Q, ftype='peak', fs=fs)

    >>> # Frequency response
    >>> freq, h = signal.freqz(b, a, fs=fs)
    >>> response = abs(h)
    >>> # To avoid divide by zero when graphing
    >>> response[response == 0] = 1e-20
    >>> # Plot
    >>> fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    >>> ax[0].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), color='blue')
    >>> ax[0].set_title("Frequency Response")
    >>> ax[0].set_ylabel("Amplitude (dB)", color='blue')
    >>> ax[0].set_xlim([0, 500])
    >>> ax[0].set_ylim([-80, 10])
    >>> ax[0].grid()
    >>> ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    >>> ax[1].set_ylabel("Angle (degrees)", color='green')
    >>> ax[1].set_xlabel("Frequency (Hz)")
    >>> ax[1].set_xlim([0, 500])
    >>> ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    >>> ax[1].set_ylim([-90, 90])
    >>> ax[1].grid()
    >>> plt.show()
    """

    # Convert w0, Q, and fs to float
    w0 = float(w0)
    Q = float(Q)
    fs = float(fs)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    filter_types = ['notch', 'peak']
    if not 0 < w0 < fs / 2:
        raise ValueError("w0 must be between 0 and {}"
                         " (nyquist), but given {}.".format(fs / 2, w0))
    if np.round(fs % w0) != 0:
        raise ValueError('fs must be divisible by w0.')
    if ftype not in filter_types:
        raise ValueError('ftype must be either notch or peak.')

    # Compute the order of the filter
    N = int(fs // w0)

    # Compute frequency in radians and filter bandwith
    # Eq. 11.3.1 (p. 574) from reference [1]
    w0 = (2 * np.pi * w0) / fs
    w_delta = w0 / Q

    # Define base gain values depending on notch or peak filter
    # Compute -3dB attenuation
    # Eqs. 11.4.1 and 11.4.2 (p. 582) from reference [1]
    if ftype == 'notch':
        G0, G = [1, 0]
    elif ftype == 'peak':
        G0, G = [0, 1]
    GB = 1 / np.sqrt(2)

    # Compute beta
    # Eq. 11.5.3 (p. 591) from reference [1]
    beta = np.sqrt((GB**2 - G0**2) / (G**2 - GB**2)) * np.tan(N * w_delta / 4)

    # Compute filter coefficients
    # Eq 11.5.1 (p. 590) variables a, b, c from reference [1]
    ax = (1 - beta) / (1 + beta)
    bx = (G0 + G * beta) / (1 + beta)
    cx = (G0 - G * beta) / (1 + beta)

    # Compute numerator coefficients
    # Eq 11.5.1 (p. 590) or Eq 11.5.4 (p. 591) from reference [1]
    # b - cz^-N or b + cz^-N
    b = np.zeros(N + 1)
    b[0] = bx
    b[-1] = cx
    if ftype == 'notch':
        b[-1] = -cx

    # Compute denominator coefficients
    # Eq 11.5.1 (p. 590) or Eq 11.5.4 (p. 591) from reference [1]
    # 1 - az^-N or 1 + az^-N
    a = np.zeros(N + 1)
    a[0] = 1
    a[-1] = ax
    if ftype == 'notch':
        a[-1] = -ax

    return b, a


def _hz_to_erb(hz):
    """
    Utility for converting from frequency (Hz) to the
    Equivalent Rectangular Bandwith (ERB) scale
    ERB = frequency / EarQ + minBW
    """
    EarQ = 9.26449
    minBW = 24.7
    return hz / EarQ + minBW


def gammatone(freq, ftype, order=None, numtaps=None, fs=None):
    """
    Gammatone filter design.

    This function computes the coefficients of an FIR or IIR gammatone
    digital filter [1]_.

    Parameters
    ----------
    freq : float
        Center frequency of the filter (expressed in the same units
        as `fs`).
    ftype : {'fir', 'iir'}
        The type of filter the function generates. If 'fir', the function
        will generate an Nth order FIR gammatone filter. If 'iir', the
        function will generate an 8th order digital IIR filter, modeled as
        as 4th order gammatone filter.
    order : int, optional
        The order of the filter. Only used when ``ftype='fir'``.
        Default is 4 to model the human auditory system. Must be between
        0 and 24.
    numtaps : int, optional
        Length of the filter. Only used when ``ftype='fir'``.
        Default is ``fs*0.015`` if `fs` is greater than 1000,
        15 if `fs` is less than or equal to 1000.
    fs : float, optional
        The sampling frequency of the signal. `freq` must be between
        0 and ``fs/2``. Default is 2.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials of the filter.

    Raises
    ------
    ValueError
        If `freq` is less than or equal to 0 or greater than or equal to
        ``fs/2``, if `ftype` is not 'fir' or 'iir', if `order` is less than
        or equal to 0 or greater than 24 when ``ftype='fir'``

    See Also
    --------
    firwin
    iirfilter

    References
    ----------
    .. [1] Slaney, Malcolm, "An Efficient Implementation of the
        Patterson-Holdsworth Auditory Filter Bank", Apple Computer
        Technical Report 35, 1993, pp.3-8, 34-39.

    Examples
    --------
    16-sample 4th order FIR Gammatone filter centered at 440 Hz

    >>> from scipy import signal
    >>> signal.gammatone(440, 'fir', numtaps=16, fs=16000)
    (array([ 0.00000000e+00,  2.22196719e-07,  1.64942101e-06,  4.99298227e-06,
        1.01993969e-05,  1.63125770e-05,  2.14648940e-05,  2.29947263e-05,
        1.76776931e-05,  2.04980537e-06, -2.72062858e-05, -7.28455299e-05,
       -1.36651076e-04, -2.19066855e-04, -3.18905076e-04, -4.33156712e-04]),
       [1.0])

    IIR Gammatone filter centered at 440 Hz

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> b, a = signal.gammatone(440, 'iir', fs=16000)
    >>> w, h = signal.freqz(b, a)
    >>> plt.plot(w / ((2 * np.pi) / 16000), 20 * np.log10(abs(h)))
    >>> plt.xscale('log')
    >>> plt.title('Gammatone filter frequency response')
    >>> plt.xlabel('Frequency')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.margins(0, 0.1)
    >>> plt.grid(which='both', axis='both')
    >>> plt.axvline(440, color='green') # cutoff frequency
    >>> plt.show()
    """
    # Converts freq to float
    freq = float(freq)

    # Set sampling rate if not passed
    if fs is None:
        fs = 2
    fs = float(fs)

    # Check for invalid cutoff frequency or filter type
    ftype = ftype.lower()
    filter_types = ['fir', 'iir']
    if not 0 < freq < fs / 2:
        raise ValueError("The frequency must be between 0 and {}"
                         " (nyquist), but given {}.".format(fs / 2, freq))
    if ftype not in filter_types:
        raise ValueError('ftype must be either fir or iir.')

    # Calculate FIR gammatone filter
    if ftype == 'fir':
        # Set order and numtaps if not passed
        if order is None:
            order = 4
        order = operator.index(order)

        if numtaps is None:
            numtaps = max(int(fs * 0.015), 15)
        numtaps = operator.index(numtaps)

        # Check for invalid order
        if not 0 < order <= 24:
            raise ValueError("Invalid order: order must be > 0 and <= 24.")

        # Gammatone impulse response settings
        t = np.arange(numtaps) / fs
        bw = 1.019 * _hz_to_erb(freq)

        # Calculate the FIR gammatone filter
        b = (t ** (order - 1)) * np.exp(-2 * np.pi * bw * t)
        b *= np.cos(2 * np.pi * freq * t)

        # Scale the FIR filter so the frequency response is 1 at cutoff
        scale_factor = 2 * (2 * np.pi * bw) ** (order)
        scale_factor /= float_factorial(order - 1)
        scale_factor /= fs
        b *= scale_factor
        a = [1.0]

    # Calculate IIR gammatone filter
    elif ftype == 'iir':
        # Raise warning if order and/or numtaps is passed
        if order is not None:
            warnings.warn('order is not used for IIR gammatone filter.')
        if numtaps is not None:
            warnings.warn('numtaps is not used for IIR gammatone filter.')

        # Gammatone impulse response settings
        T = 1./fs
        bw = 2 * np.pi * 1.019 * _hz_to_erb(freq)
        fr = 2 * freq * np.pi * T
        bwT = bw * T

        # Calculate the gain to normalize the volume at the center frequency
        g1 = -2 * np.exp(2j * fr) * T
        g2 = 2 * np.exp(-(bwT) + 1j * fr) * T
        g3 = np.sqrt(3 + 2 ** (3 / 2)) * np.sin(fr)
        g4 = np.sqrt(3 - 2 ** (3 / 2)) * np.sin(fr)
        g5 = np.exp(2j * fr)

        g = g1 + g2 * (np.cos(fr) - g4)
        g *= (g1 + g2 * (np.cos(fr) + g4))
        g *= (g1 + g2 * (np.cos(fr) - g3))
        g *= (g1 + g2 * (np.cos(fr) + g3))
        g /= ((-2 / np.exp(2 * bwT) - 2 * g5 + 2 * (1 + g5) / np.exp(bwT)) ** 4)
        g = np.abs(g)

        # Create empty filter coefficient lists
        b = np.empty(5)
        a = np.empty(9)

        # Calculate the numerator coefficients
        b[0] = (T ** 4) / g
        b[1] = -4 * T ** 4 * np.cos(fr) / np.exp(bw * T) / g
        b[2] = 6 * T ** 4 * np.cos(2 * fr) / np.exp(2 * bw * T) / g
        b[3] = -4 * T ** 4 * np.cos(3 * fr) / np.exp(3 * bw * T) / g
        b[4] = T ** 4 * np.cos(4 * fr) / np.exp(4 * bw * T) / g

        # Calculate the denominator coefficients
        a[0] = 1
        a[1] = -8 * np.cos(fr) / np.exp(bw * T)
        a[2] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(2 * bw * T)
        a[3] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[3] /= np.exp(3 * bw * T)
        a[4] = 2 * (18 + 16 * np.cos(2 * fr) + np.cos(4 * fr))
        a[4] /= np.exp(4 * bw * T)
        a[5] = -8 * (6 * np.cos(fr) + np.cos(3 * fr))
        a[5] /= np.exp(5 * bw * T)
        a[6] = 4 * (4 + 3 * np.cos(2 * fr)) / np.exp(6 * bw * T)
        a[7] = -8 * np.cos(fr) / np.exp(7 * bw * T)
        a[8] = np.exp(-8 * bw * T)

    return b, a


filter_dict = {'butter': [buttap, buttord],
               'butterworth': [buttap, buttord],

               'cauer': [ellipap, ellipord],
               'elliptic': [ellipap, ellipord],
               'ellip': [ellipap, ellipord],

               'bessel': [besselap],
               'bessel_phase': [besselap],
               'bessel_delay': [besselap],
               'bessel_mag': [besselap],

               'cheby1': [cheb1ap, cheb1ord],
               'chebyshev1': [cheb1ap, cheb1ord],
               'chebyshevi': [cheb1ap, cheb1ord],

               'cheby2': [cheb2ap, cheb2ord],
               'chebyshev2': [cheb2ap, cheb2ord],
               'chebyshevii': [cheb2ap, cheb2ord],
               }

band_dict = {'band': 'bandpass',
             'bandpass': 'bandpass',
             'pass': 'bandpass',
             'bp': 'bandpass',

             'bs': 'bandstop',
             'bandstop': 'bandstop',
             'bands': 'bandstop',
             'stop': 'bandstop',

             'l': 'lowpass',
             'low': 'lowpass',
             'lowpass': 'lowpass',
             'lp': 'lowpass',

             'high': 'highpass',
             'highpass': 'highpass',
             'h': 'highpass',
             'hp': 'highpass',
             }

bessel_norms = {'bessel': 'phase',
                'bessel_phase': 'phase',
                'bessel_delay': 'delay',
                'bessel_mag': 'mag'}
