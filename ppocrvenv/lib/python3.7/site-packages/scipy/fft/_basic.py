from scipy._lib.uarray import generate_multimethod, Dispatchable
import numpy as np


def _x_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the transform input array (``x``)
    """
    if len(args) > 0:
        return (dispatchables[0],) + args[1:], kwargs
    kw = kwargs.copy()
    kw['x'] = dispatchables[0]
    return args, kw


def _dispatch(func):
    """
    Function annotation that creates a uarray multimethod from the function
    """
    return generate_multimethod(func, _x_replacer, domain="numpy.scipy.fft")


@_dispatch
def fft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
        plan=None):
    """
    Compute the 1-D discrete Fourier Transform.

    This function computes the 1-D *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [1]_.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode. Default is "backward", meaning no normalization on
        the forward transforms and scaling by ``1/n`` on the `ifft`.
        "forward" instead applies the ``1/n`` factor on the forward tranform.
        For ``norm="ortho"``, both directions are scaled by ``1/sqrt(n)``.

        .. versionadded:: 1.6.0
           ``norm={"forward", "backward"}`` options were added

    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See the notes below for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``. See below for more
        details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        if `axes` is larger than the last axis of `x`.

    See Also
    --------
    ifft : The inverse of `fft`.
    fft2 : The 2-D FFT.
    fftn : The N-D FFT.
    rfftn : The N-D FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.
    next_fast_len : Size to pad input to for most efficient transforms

    Notes
    -----

    FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform
    (DFT) can be calculated efficiently, by using symmetries in the calculated
    terms. The symmetry is highest when `n` is a power of 2, and the transform
    is therefore most efficient for these sizes. For poorly factorizable sizes,
    `scipy.fft` uses Bluestein's algorithm [2]_ and so is never worse than
    O(`n` log `n`). Further performance improvements may be seen by zero-padding
    the input using `next_fast_len`.

    If ``x`` is a 1d array, then the `fft` is equivalent to ::

        y[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(n)/n))

    The frequency term ``f=k/n`` is found at ``y[k]``. At ``y[n/2]`` we reach
    the Nyquist frequency and wrap around to the negative-frequency terms. So,
    for an 8-point transform, the frequencies of the result are
    [0, 1, 2, 3, -4, -3, -2, -1]. To rearrange the fft output so that the
    zero-frequency component is centered, like [-4, -3, -2, -1, 0, 1, 2, 3],
    use `fftshift`.

    Transforms can be done in single, double, or extended precision (long
    double) floating point. Half precision inputs will be converted to single
    precision and non-floating-point inputs will be converted to double
    precision.

    If the data type of ``x`` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time. To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum. If the data are both real and
    symmetrical, the `dct` can again double the efficiency, by generating
    half of the spectrum from half of the signal.

    When ``overwrite_x=True`` is specified, the memory referenced by ``x`` may
    be used by the implementation in any way. This may include reusing the
    memory for the result, but this is in no way guaranteed. You should not
    rely on the contents of ``x`` after the transform as this may change in
    future without warning.

    The ``workers`` argument specifies the maximum number of parallel jobs to
    split the FFT computation into. This will execute independent 1-D
    FFTs within ``x``. So, ``x`` must be at least 2-D and the
    non-transformed axes must be large enough to split into chunks. If ``x`` is
    too small, fewer jobs may be used than requested.

    References
    ----------
    .. [1] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
           machine calculation of complex Fourier series," *Math. Comput.*
           19: 297-301.
    .. [2] Bluestein, L., 1970, "A linear filtering approach to the
           computation of discrete Fourier transform". *IEEE Transactions on
           Audio and Electroacoustics.* 18 (4): 451-455.

    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])

    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part:

    >>> from scipy.fft import fft, fftfreq, fftshift
    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sp = fftshift(fft(np.sin(t)))
    >>> freq = fftshift(fftfreq(t.shape[-1]))
    >>> plt.plot(freq, sp.real, freq, sp.imag)
    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 1-D inverse discrete Fourier Transform.

    This function computes the inverse of the 1-D *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(x)) == x`` to within numerical accuracy.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``x[0]`` should contain the zero frequency term,
    * ``x[1:n//2]`` should contain the positive-frequency terms,
    * ``x[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``x[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together. See `fft` for details.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
        See notes about padding issues.
    axis : int, optional
        Axis over which to compute the inverse DFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axes` is larger than the last axis of `x`.

    See Also
    --------
    fft : The 1-D (forward) FFT, of which `ifft` is the inverse.
    ifft2 : The 2-D inverse FFT.
    ifftn : The N-D inverse FFT.

    Notes
    -----
    If the input parameter `n` is larger than the size of the input, the input
    is padded by appending zeros at the end. Even though this is the common
    approach, it might lead to surprising results. If a different padding is
    desired, it must be performed before calling `ifft`.

    If ``x`` is a 1-D array, then the `ifft` is equivalent to ::

        y[k] = np.sum(x * np.exp(2j * np.pi * k * np.arange(n)/n)) / len(x)

    As with `fft`, `ifft` has support for all floating point types and is
    optimized for real input.

    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.ifft([0, 4, 0, 0])
    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary

    Create and plot a band-limited signal with random phases:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> t = np.arange(400)
    >>> n = np.zeros((400,), dtype=complex)
    >>> n[40:60] = np.exp(1j*rng.uniform(0, 2*np.pi, (20,)))
    >>> s = scipy.fft.ifft(n)
    >>> plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
    [<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
    >>> plt.legend(('real', 'imaginary'))
    <matplotlib.legend.Legend object at ...>
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def rfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 1-D discrete Fourier Transform for real input.

    This function computes the 1-D *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    x : array_like
        Input array
    n : int, optional
        Number of points along transformation axis in the input to use.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        If `n` is even, the length of the transformed axis is ``(n/2)+1``.
        If `n` is odd, the length is ``(n+1)/2``.

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `a`.

    See Also
    --------
    irfft : The inverse of `rfft`.
    fft : The 1-D FFT of general (complex) input.
    fftn : The N-D FFT.
    rfft2 : The 2-D FFT of real input.
    rfftn : The N-D FFT of real input.

    Notes
    -----
    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric, i.e., the negative frequency terms are just the complex
    conjugates of the corresponding positive-frequency terms, and the
    negative-frequency terms are therefore redundant. This function does not
    compute the negative frequency terms, and the length of the transformed
    axis of the output is therefore ``n//2 + 1``.

    When ``X = rfft(x)`` and fs is the sampling frequency, ``X[0]`` contains
    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

    If `n` is even, ``A[-1]`` contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
    the largest positive frequency (fs/2*(n-1)/n), and is complex in the
    general case.

    If the input `a` contains an imaginary part, it is silently discarded.

    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.fft([0, 1, 0, 0])
    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary
    >>> scipy.fft.rfft([0, 1, 0, 0])
    array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary

    Notice how the final element of the `fft` output is the complex conjugate
    of the second element, for real input. For `rfft`, this symmetry is
    exploited to compute only the non-negative frequency terms.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def irfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Computes the inverse of `rfft`.

    This function computes the inverse of the 1-D *n*-point
    discrete Fourier Transform of real input computed by `rfft`.
    In other words, ``irfft(rfft(x), len(x)) == x`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by `rfft`, i.e., the
    real zero-frequency term followed by the complex positive frequency terms
    in order of increasing frequency. Since the discrete Fourier Transform of
    real input is Hermitian-symmetric, the negative frequency terms are taken
    to be the complex conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    x : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary. If the
        input is longer than this, it is cropped. If it is shorter than this,
        it is padded with zeros. If `n` is not given, it is taken to be
        ``2*(m-1)``, where ``m`` is the length of the input along the axis
        specified by `axis`.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified.

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `x`.

    See Also
    --------
    rfft : The 1-D FFT of real input, of which `irfft` is inverse.
    fft : The 1-D FFT.
    irfft2 : The inverse of the 2-D FFT of real input.
    irfftn : The inverse of the N-D FFT of real input.

    Notes
    -----
    Returns the real valued `n`-point inverse discrete Fourier transform
    of `x`, where `x` contains the non-negative frequency terms of a
    Hermitian-symmetric sequence. `n` is the length of the result, not the
    input.

    If you specify an `n` such that `a` must be zero-padded or truncated, the
    extra/removed values will be added/removed at high frequencies. One can
    thus resample a series to `m` points via Fourier interpolation by:
    ``a_resamp = irfft(rfft(a), m)``.

    The default value of `n` assumes an even output length. By the Hermitian
    symmetry, the last imaginary component must be 0 and so is ignored. To
    avoid losing information, the correct length of the real input *must* be
    given.

    Examples
    --------
    >>> import scipy.fft
    >>> scipy.fft.ifft([1, -1j, -1, 1j])
    array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary
    >>> scipy.fft.irfft([1, -1j, -1])
    array([0.,  1.,  0.,  0.])

    Notice how the last term in the input to the ordinary `ifft` is the
    complex conjugate of the second term, and the output has zero imaginary
    part everywhere. When calling `irfft`, the negative frequencies are not
    specified, and the output array is purely real.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def hfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
    spectrum.

    Parameters
    ----------
    x : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output. For `n` output
        points, ``n//2 + 1`` input points are necessary. If the input is
        longer than this, it is cropped. If it is shorter than this, it is
        padded with zeros. If `n` is not given, it is taken to be ``2*(m-1)``,
        where ``m`` is the length of the input along the axis specified by
        `axis`.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*m - 2``, where ``m`` is the length of the transformed axis of
        the input. To get an odd number of output points, `n` must be
        specified, for instance, as ``2*m - 1`` in the typical case,

    Raises
    ------
    IndexError
        If `axis` is larger than the last axis of `a`.

    See Also
    --------
    rfft : Compute the 1-D FFT for real input.
    ihfft : The inverse of `hfft`.
    hfftn : Compute the N-D FFT of a Hermitian signal.

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So, here, it's `hfft`, for
    which you must supply the length of the result if it is to be odd.
    * even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

    Examples
    --------
    >>> from scipy.fft import fft, hfft
    >>> a = 2 * np.pi * np.arange(10) / 10
    >>> signal = np.cos(a) + 3j * np.sin(3 * a)
    >>> fft(signal).round(10)
    array([ -0.+0.j,   5.+0.j,  -0.+0.j,  15.-0.j,   0.+0.j,   0.+0.j,
            -0.+0.j, -15.-0.j,   0.+0.j,   5.+0.j])
    >>> hfft(signal[:6]).round(10) # Input first half of signal
    array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
    >>> hfft(signal, 10)  # Input entire signal and truncate
    array([  0.,   5.,   0.,  15.,  -0.,   0.,   0., -15.,  -0.,   5.])
    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ihfft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    Parameters
    ----------
    x : array_like
        Input array.
    n : int, optional
        Length of the inverse FFT, the number of points along
        transformation axis in the input to use.  If `n` is smaller than
        the length of the input, the input is cropped. If it is larger,
        the input is padded with zeros. If `n` is not given, the length of
        the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is ``n//2 + 1``.

    See Also
    --------
    hfft, irfft

    Notes
    -----
    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
    opposite case: here, the signal has Hermitian symmetry in the time
    domain and is real in the frequency domain. So, here, it's `hfft`, for
    which you must supply the length of the result if it is to be odd:
    * even: ``ihfft(hfft(a, 2*len(a) - 2) == a``, within roundoff error,
    * odd: ``ihfft(hfft(a, 2*len(a) - 1) == a``, within roundoff error.

    Examples
    --------
    >>> from scipy.fft import ifft, ihfft
    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
    >>> ifft(spectrum)
    array([1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.+0.j]) # may vary
    >>> ihfft(spectrum)
    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j]) # may vary
    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def fftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the N-D discrete Fourier Transform.

    This function computes the N-D discrete Fourier Transform over
    any number of axes in an M-D array by means of the Fast Fourier
    Transform (FFT).

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    ifftn : The inverse of `fftn`, the inverse N-D FFT.
    fft : The 1-D FFT, with definitions and conventions used.
    rfftn : The N-D FFT of real input.
    fft2 : The 2-D FFT.
    fftshift : Shifts zero-frequency terms to centre of array.

    Notes
    -----
    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of all axes, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Examples
    --------
    >>> import scipy.fft
    >>> x = np.mgrid[:3, :3, :3][0]
    >>> scipy.fft.fftn(x, axes=(1, 2))
    array([[[ 0.+0.j,   0.+0.j,   0.+0.j], # may vary
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[ 9.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]],
           [[18.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j],
            [ 0.+0.j,   0.+0.j,   0.+0.j]]])
    >>> scipy.fft.fftn(x, (2, 2), axes=(0, 1))
    array([[[ 2.+0.j,  2.+0.j,  2.+0.j], # may vary
            [ 0.+0.j,  0.+0.j,  0.+0.j]],
           [[-2.+0.j, -2.+0.j, -2.+0.j],
            [ 0.+0.j,  0.+0.j,  0.+0.j]]])

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
    ...                      2 * np.pi * np.arange(200) / 34)
    >>> S = np.sin(X) + np.cos(Y) + rng.uniform(0, 1, X.shape)
    >>> FS = scipy.fft.fftn(S)
    >>> plt.imshow(np.log(np.abs(scipy.fft.fftshift(FS))**2))
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ifftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D inverse discrete Fourier Transform.

    This function computes the inverse of the N-D discrete
    Fourier Transform over any number of axes in an M-D array by
    means of the Fast Fourier Transform (FFT).  In other words,
    ``ifftn(fftn(x)) == x`` to within numerical accuracy.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fftn`, i.e., it should have the term for zero frequency
    in all axes in the low-order corner, the positive frequency terms in the
    first half of all axes, the term for the Nyquist frequency in the middle
    of all axes and the negative frequency terms in the second half of all
    axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``ifft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used. See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the IFFT.  If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    fftn : The forward N-D FFT, of which `ifftn` is the inverse.
    ifft : The 1-D inverse FFT.
    ifft2 : The 2-D inverse FFT.
    ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning
        of array.

    Notes
    -----
    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension. Although this is the common
    approach, it might lead to surprising results. If another form of zero
    padding is desired, it must be performed before `ifftn` is called.

    Examples
    --------
    >>> import scipy.fft
    >>> x = np.eye(4)
    >>> scipy.fft.ifftn(scipy.fft.fftn(x, axes=(0,)), axes=(1,))
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])


    Create and plot an image with band-limited frequency content:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> n = np.zeros((200,200), dtype=complex)
    >>> n[60:80, 20:40] = np.exp(1j*rng.uniform(0, 2*np.pi, (20, 20)))
    >>> im = scipy.fft.ifftn(n).real
    >>> plt.imshow(im)
    <matplotlib.image.AxesImage object at 0x...>
    >>> plt.show()

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    """
    Compute the 2-D discrete Fourier Transform

    This function computes the N-D discrete Fourier Transform
    over any axes in an M-D array by means of the
    Fast Fourier Transform (FFT). By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT.

    Parameters
    ----------
    x : array_like
        Input array, can be complex
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        This corresponds to ``n`` for ``fft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two axes are
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    ifft2 : The inverse 2-D FFT.
    fft : The 1-D FFT.
    fftn : The N-D FFT.
    fftshift : Shifts zero-frequency terms to the center of the array.
        For 2-D input, swaps first and third quadrants, and second
        and fourth quadrants.

    Notes
    -----
    `fft2` is just `fftn` with a different default for `axes`.

    The output, analogously to `fft`, contains the term for zero frequency in
    the low-order corner of the transformed axes, the positive frequency terms
    in the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    the axes, in order of decreasingly negative frequency.

    See `fftn` for details and a plotting example, and `fft` for
    definitions and conventions used.


    Examples
    --------
    >>> import scipy.fft
    >>> x = np.mgrid[:5, :5][0]
    >>> scipy.fft.fft2(x)
    array([[ 50.  +0.j        ,   0.  +0.j        ,   0.  +0.j        , # may vary
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5+17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 +4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5 -4.0614962j ,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ],
           [-12.5-17.20477401j,   0.  +0.j        ,   0.  +0.j        ,
              0.  +0.j        ,   0.  +0.j        ]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ifft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D inverse discrete Fourier Transform.

    This function computes the inverse of the 2-D discrete Fourier
    Transform over any number of axes in an M-D array by means of
    the Fast Fourier Transform (FFT). In other words, ``ifft2(fft2(x)) == x``
    to within numerical accuracy. By default, the inverse transform is
    computed over the last two axes of the input array.

    The input, analogously to `ifft`, should be ordered in the same way as is
    returned by `fft2`, i.e., it should have the term for zero frequency
    in the low-order corner of the two axes, the positive frequency terms in
    the first half of these axes, the term for the Nyquist frequency in the
    middle of the axes and the negative frequency terms in the second half of
    both axes, in order of decreasingly negative frequency.

    Parameters
    ----------
    x : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
        ``s[1]`` to axis 1, etc.). This corresponds to `n` for ``ifft(x, n)``.
        Along each axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.  See notes for issue on `ifft` zero padding.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last two
        axes are used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or the last two axes if `axes` is not given.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length, or `axes` not given and
        ``len(s) != 2``.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    fft2 : The forward 2-D FFT, of which `ifft2` is the inverse.
    ifftn : The inverse of the N-D FFT.
    fft : The 1-D FFT.
    ifft : The 1-D inverse FFT.

    Notes
    -----
    `ifft2` is just `ifftn` with a different default for `axes`.

    See `ifftn` for details and a plotting example, and `fft` for
    definition and conventions used.

    Zero-padding, analogously with `ifft`, is performed by appending zeros to
    the input along the specified dimension. Although this is the common
    approach, it might lead to surprising results. If another form of zero
    padding is desired, it must be performed before `ifft2` is called.

    Examples
    --------
    >>> import scipy.fft
    >>> x = 4 * np.eye(4)
    >>> scipy.fft.ifft2(x)
    array([[1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j], # may vary
           [0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
           [0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def rfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D discrete Fourier Transform for real input.

    This function computes the N-D discrete Fourier Transform over
    any number of axes in an M-D real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    irfftn : The inverse of `rfftn`, i.e., the inverse of the N-D FFT
         of real input.
    fft : The 1-D FFT, with definitions and conventions used.
    rfft : The 1-D FFT of real input.
    fftn : The N-D FFT.
    rfft2 : The 2-D FFT of real input.

    Notes
    -----
    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`. The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    See `fft` for details, definitions and conventions used.

    Examples
    --------
    >>> import scipy.fft
    >>> x = np.ones((2, 2, 2))
    >>> scipy.fft.rfftn(x)
    array([[[8.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    >>> scipy.fft.rfftn(x, axes=(2, 0))
    array([[[4.+0.j,  0.+0.j], # may vary
            [4.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def rfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D FFT of a real array.

    Parameters
    ----------
    x : array
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape of the FFT.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the real 2-D FFT.

    See Also
    --------
    irfft2 : The inverse of the 2-D FFT of real input.
    rfft : The 1-D FFT of real input.
    rfftn : Compute the N-D discrete Fourier Transform for real
            input.

    Notes
    -----
    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def irfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Computes the inverse of `rfftn`

    This function computes the inverse of the N-D discrete
    Fourier Transform for real input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In
    other words, ``irfftn(rfftn(x), x.shape) == x`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e., as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Parameters
    ----------
    x : array_like
        Input array.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped. If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to be
        ``2*(m-1)``, where ``m`` is the length of the input along that axis.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given. In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)``, where ``m`` is the
        length of the final transformed axis of the input. To get an odd
        number of output points in the final axis, `s` must be specified.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    rfftn : The forward N-D FFT of real input,
            of which `ifftn` is the inverse.
    fft : The 1-D FFT, with definitions and conventions used.
    irfft : The inverse of the 1-D FFT of real input.
    irfft2 : The inverse of the 2-D FFT of real input.

    Notes
    -----
    See `fft` for definitions and conventions used.

    See `rfft` for definitions and conventions used for real input.

    The default value of `s` assumes an even output length in the final
    transformation axis. When performing the final complex to real
    transformation, the Hermitian symmetry requires that the last imaginary
    component along that axis must be 0 and so it is ignored. To avoid losing
    information, the correct length of the real input *must* be given.

    Examples
    --------
    >>> import scipy.fft
    >>> x = np.zeros((3, 2, 2))
    >>> x[0, 0, 0] = 3 * 2 * 2
    >>> scipy.fft.irfftn(x)
    array([[[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]],
           [[1.,  1.],
            [1.,  1.]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def irfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Computes the inverse of `rfft2`

    Parameters
    ----------
    x : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real output to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    rfft2 : The 2-D FFT of real input.
    irfft : The inverse of the 1-D FFT of real input.
    irfftn : The inverse of the N-D FFT of real input.

    Notes
    -----
    This is really `irfftn` with different defaults.
    For more details see `irfftn`.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def hfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the N-D FFT of Hermitian symmetric complex input, i.e., a
    signal with a real spectrum.

    This function computes the N-D discrete Fourier Transform for a
    Hermitian symmetric complex input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In other
    words, ``ihfftn(hfftn(x, s)) == x`` to within numerical accuracy. (``s``
    here is ``x.shape`` with ``s[-1] = x.shape[-1] * 2 - 1``, this is necessary
    for the same reason ``x.shape`` would be necessary for `irfft`.)

    Parameters
    ----------
    x : array_like
        Input array.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
        number of input points used along this axis, except for the last axis,
        where ``s[-1]//2+1`` points of the input are used.
        Along any axis, if the shape indicated by `s` is smaller than that of
        the input, the input is cropped. If it is larger, the input is padded
        with zeros. If `s` is not given, the shape of the input along the axes
        specified by axes is used. Except for the last axis which is taken to be
        ``2*(m-1)`` where ``m`` is the length of the input along that axis.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. If not given, the last
        `len(s)` axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` or `x`,
        as explained in the parameters section above.
        The length of each transformed axis is as given by the corresponding
        element of `s`, or the length of the input in every axis except for the
        last one if `s` is not given.  In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
        length of the final transformed axis of the input.  To get an odd
        number of output points in the final axis, `s` must be specified.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    ihfftn : The inverse N-D FFT with real spectrum. Inverse of `hfftn`.
    fft : The 1-D FFT, with definitions and conventions used.
    rfft : Forward FFT of real input.

    Notes
    -----

    For a 1-D signal ``x`` to have a real spectrum, it must satisfy
    the Hermitian property::

        x[i] == np.conj(x[-i]) for all i

    This generalizes into higher dimensions by reflecting over each axis in
    turn::

        x[i, j, k, ...] == np.conj(x[-i, -j, -k, ...]) for all i, j, k, ...

    This should not be confused with a Hermitian matrix, for which the
    transpose is its own conjugate::

        x[i, j] == np.conj(x[j, i]) for all i, j


    The default value of `s` assumes an even output length in the final
    transformation axis. When performing the final complex to real
    transformation, the Hermitian symmetry requires that the last imaginary
    component along that axis must be 0 and so it is ignored. To avoid losing
    information, the correct length of the real input *must* be given.

    Examples
    --------
    >>> import scipy.fft
    >>> x = np.ones((3, 2, 2))
    >>> scipy.fft.hfftn(x)
    array([[[12.,  0.],
            [ 0.,  0.]],
           [[ 0.,  0.],
            [ 0.,  0.]],
           [[ 0.,  0.],
            [ 0.,  0.]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def hfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
          plan=None):
    """
    Compute the 2-D FFT of a Hermitian complex array.

    Parameters
    ----------
    x : array
        Input array, taken to be Hermitian complex.
    s : sequence of ints, optional
        Shape of the real output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See `fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The real result of the 2-D Hermitian complex real FFT.

    See Also
    --------
    hfftn : Compute the N-D discrete Fourier Transform for Hermitian
            complex input.

    Notes
    -----
    This is really just `hfftn` with different default behavior.
    For more details see `hfftn`.

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ihfftn(x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Compute the N-D inverse discrete Fourier Transform for a real
    spectrum.

    This function computes the N-D inverse discrete Fourier Transform
    over any number of axes in an M-D real array by means of the Fast
    Fourier Transform (FFT). By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining transforms
    are complex.

    Parameters
    ----------
    x : array_like
        Input array, taken to be real.
    s : sequence of ints, optional
        Shape (length along each transformed axis) to use from the input.
        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
        Along any axis, if the given shape is smaller than that of the input,
        the input is cropped. If it is larger, the input is padded with zeros.
        if `s` is not given, the shape of the input along the axes specified
        by `axes` is used.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axes
        indicated by `axes`, or by a combination of `s` and `x`,
        as explained in the parameters section above.
        The length of the last axis transformed will be ``s[-1]//2+1``,
        while the remaining transformed axes will have lengths according to
        `s`, or unchanged from the input.

    Raises
    ------
    ValueError
        If `s` and `axes` have different length.
    IndexError
        If an element of `axes` is larger than than the number of axes of `x`.

    See Also
    --------
    hfftn : The forward N-D FFT of Hermitian input.
    hfft : The 1-D FFT of Hermitian input.
    fft : The 1-D FFT, with definitions and conventions used.
    fftn : The N-D FFT.
    hfft2 : The 2-D FFT of Hermitian input.

    Notes
    -----

    The transform for real input is performed over the last transformation
    axis, as by `ihfft`, then the transform over the remaining axes is
    performed as by `ifftn`. The order of the output is the positive part of
    the Hermitian output signal, in the same format as `rfft`.

    Examples
    --------
    >>> import scipy.fft
    >>> x = np.ones((2, 2, 2))
    >>> scipy.fft.ihfftn(x)
    array([[[1.+0.j,  0.+0.j], # may vary
            [0.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])
    >>> scipy.fft.ihfftn(x, axes=(2, 0))
    array([[[1.+0.j,  0.+0.j], # may vary
            [1.+0.j,  0.+0.j]],
           [[0.+0.j,  0.+0.j],
            [0.+0.j,  0.+0.j]]])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def ihfft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False, workers=None, *,
           plan=None):
    """
    Compute the 2-D inverse FFT of a real spectrum.

    Parameters
    ----------
    x : array_like
        The input array
    s : sequence of ints, optional
        Shape of the real input to the inverse FFT.
    axes : sequence of ints, optional
        The axes over which to compute the inverse fft.
        Default is the last two axes.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `fft`). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
        See :func:`fft` for more details.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.
    plan : object, optional
        This argument is reserved for passing in a precomputed plan provided
        by downstream FFT vendors. It is currently not used in SciPy.

        .. versionadded:: 1.5.0

    Returns
    -------
    out : ndarray
        The result of the inverse real 2-D FFT.

    See Also
    --------
    ihfftn : Compute the inverse of the N-D FFT of Hermitian input.

    Notes
    -----
    This is really `ihfftn` with different defaults.
    For more details see `ihfftn`.

    """
    return (Dispatchable(x, np.ndarray),)
