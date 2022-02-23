import numpy as np
from scipy.linalg import eig
from scipy.special import comb
from scipy.signal import convolve

__all__ = ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'morlet2', 'cwt']


def daub(p):
    """
    The coefficients for the FIR low-pass filter producing Daubechies wavelets.

    p>=1 gives the order of the zero at f=1/2.
    There are 2p filter coefficients.

    Parameters
    ----------
    p : int
        Order of the zero at f=1/2, can have values from 1 to 34.

    Returns
    -------
    daub : ndarray
        Return

    """
    sqrt = np.sqrt
    if p < 1:
        raise ValueError("p must be at least 1.")
    if p == 1:
        c = 1 / sqrt(2)
        return np.array([c, c])
    elif p == 2:
        f = sqrt(2) / 8
        c = sqrt(3)
        return f * np.array([1 + c, 3 + c, 3 - c, 1 - c])
    elif p == 3:
        tmp = 12 * sqrt(10)
        z1 = 1.5 + sqrt(15 + tmp) / 6 - 1j * (sqrt(15) + sqrt(tmp - 15)) / 6
        z1c = np.conj(z1)
        f = sqrt(2) / 8
        d0 = np.real((1 - z1) * (1 - z1c))
        a0 = np.real(z1 * z1c)
        a1 = 2 * np.real(z1)
        return f / d0 * np.array([a0, 3 * a0 - a1, 3 * a0 - 3 * a1 + 1,
                                  a0 - 3 * a1 + 3, 3 - a1, 1])
    elif p < 35:
        # construct polynomial and factor it
        if p < 35:
            P = [comb(p - 1 + k, k, exact=1) for k in range(p)][::-1]
            yj = np.roots(P)
        else:  # try different polynomial --- needs work
            P = [comb(p - 1 + k, k, exact=1) / 4.0**k
                 for k in range(p)][::-1]
            yj = np.roots(P) / 4
        # for each root, compute two z roots, select the one with |z|>1
        # Build up final polynomial
        c = np.poly1d([1, 1])**p
        q = np.poly1d([1])
        for k in range(p - 1):
            yval = yj[k]
            part = 2 * sqrt(yval * (yval - 1))
            const = 1 - 2 * yval
            z1 = const + part
            if (abs(z1)) < 1:
                z1 = const - part
            q = q * [1, -z1]

        q = c * np.real(q)
        # Normalize result
        q = q / np.sum(q) * sqrt(2)
        return q.c[::-1]
    else:
        raise ValueError("Polynomial factorization does not work "
                         "well for p too large.")


def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    N = len(hk) - 1
    asgn = [{0: 1, 1: -1}[k % 2] for k in range(N + 1)]
    return hk[::-1] * np.array(asgn)


def cascade(hk, J=7):
    """
    Return (x, phi, psi) at dyadic points ``K/2**J`` from filter coefficients.

    Parameters
    ----------
    hk : array_like
        Coefficients of low-pass filter.
    J : int, optional
        Values will be computed at grid points ``K/2**J``. Default is 7.

    Returns
    -------
    x : ndarray
        The dyadic points ``K/2**J`` for ``K=0...N * (2**J)-1`` where
        ``len(hk) = len(gk) = N+1``.
    phi : ndarray
        The scaling function ``phi(x)`` at `x`:
        ``phi(x) = sum(hk * phi(2x-k))``, where k is from 0 to N.
    psi : ndarray, optional
        The wavelet function ``psi(x)`` at `x`:
        ``phi(x) = sum(gk * phi(2x-k))``, where k is from 0 to N.
        `psi` is only returned if `gk` is not None.

    Notes
    -----
    The algorithm uses the vector cascade algorithm described by Strang and
    Nguyen in "Wavelets and Filter Banks".  It builds a dictionary of values
    and slices for quick reuse.  Then inserts vectors into final vector at the
    end.

    """
    N = len(hk) - 1

    if (J > 30 - np.log2(N + 1)):
        raise ValueError("Too many levels.")
    if (J < 1):
        raise ValueError("Too few levels.")

    # construct matrices needed
    nn, kk = np.ogrid[:N, :N]
    s2 = np.sqrt(2)
    # append a zero so that take works
    thk = np.r_[hk, 0]
    gk = qmf(hk)
    tgk = np.r_[gk, 0]

    indx1 = np.clip(2 * nn - kk, -1, N + 1)
    indx2 = np.clip(2 * nn - kk + 1, -1, N + 1)
    m = np.empty((2, 2, N, N), 'd')
    m[0, 0] = np.take(thk, indx1, 0)
    m[0, 1] = np.take(thk, indx2, 0)
    m[1, 0] = np.take(tgk, indx1, 0)
    m[1, 1] = np.take(tgk, indx2, 0)
    m *= s2

    # construct the grid of points
    x = np.arange(0, N * (1 << J), dtype=float) / (1 << J)
    phi = 0 * x

    psi = 0 * x

    # find phi0, and phi1
    lam, v = eig(m[0, 0])
    ind = np.argmin(np.absolute(lam - 1))
    # a dictionary with a binary representation of the
    #   evaluation points x < 1 -- i.e. position is 0.xxxx
    v = np.real(v[:, ind])
    # need scaling function to integrate to 1 so find
    #  eigenvector normalized to sum(v,axis=0)=1
    sm = np.sum(v)
    if sm < 0:  # need scaling function to integrate to 1
        v = -v
        sm = -sm
    bitdic = {'0': v / sm}
    bitdic['1'] = np.dot(m[0, 1], bitdic['0'])
    step = 1 << J
    phi[::step] = bitdic['0']
    phi[(1 << (J - 1))::step] = bitdic['1']
    psi[::step] = np.dot(m[1, 0], bitdic['0'])
    psi[(1 << (J - 1))::step] = np.dot(m[1, 1], bitdic['0'])
    # descend down the levels inserting more and more values
    #  into bitdic -- store the values in the correct location once we
    #  have computed them -- stored in the dictionary
    #  for quicker use later.
    prevkeys = ['1']
    for level in range(2, J + 1):
        newkeys = ['%d%s' % (xx, yy) for xx in [0, 1] for yy in prevkeys]
        fac = 1 << (J - level)
        for key in newkeys:
            # convert key to number
            num = 0
            for pos in range(level):
                if key[pos] == '1':
                    num += (1 << (level - 1 - pos))
            pastphi = bitdic[key[1:]]
            ii = int(key[0])
            temp = np.dot(m[0, ii], pastphi)
            bitdic[key] = temp
            phi[num * fac::step] = temp
            psi[num * fac::step] = np.dot(m[1, ii], pastphi)
        prevkeys = newkeys

    return x, phi, psi


def morlet(M, w=5.0, s=1.0, complete=True):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    morlet2 : Implementation of Morlet wavelet, compatible with `cwt`.
    scipy.signal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    """
    x = np.linspace(-s * 2 * np.pi, s * 2 * np.pi, M)
    output = np.exp(1j * w * x)

    if complete:
        output -= np.exp(-0.5 * (w**2))

    output *= np.exp(-0.5 * (x**2)) * np.pi**(-0.25)

    return output


def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A * (1 - (x/a)**2) * exp(-0.5*(x/a)**2)``,

    where ``A = 2/(sqrt(3*a)*(pi**0.25))``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = signal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(vec2)
    >>> plt.show()

    """
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def morlet2(M, s, w=5):
    """
    Complex Morlet wavelet, designed to work with `cwt`.

    Returns the complete version of morlet wavelet, normalised
    according to `s`::

        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)

    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter of the wavelet.
    w : float, optional
        Omega0. Default is 5

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    morlet : Implementation of Morlet wavelet, incompatible with `cwt`

    Notes
    -----

    .. versionadded:: 1.4.0

    This function was designed to work with `cwt`. Because `morlet2`
    returns an array of complex numbers, the `dtype` argument of `cwt`
    should be set to `complex128` for best results.

    Note the difference in implementation with `morlet`.
    The fundamental frequency of this wavelet in Hz is given by::

        f = w*fs / (2*s*np.pi)

    where ``fs`` is the sampling rate and `s` is the wavelet width parameter.
    Similarly we can get the wavelet width parameter at ``f``::

        s = w*fs / (2*f*np.pi)

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> M = 100
    >>> s = 4.0
    >>> w = 2.0
    >>> wavelet = signal.morlet2(M, s, w)
    >>> plt.plot(abs(wavelet))
    >>> plt.show()

    This example shows basic use of `morlet2` with `cwt` in time-frequency
    analysis:

    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t, dt = np.linspace(0, 1, 200, retstep=True)
    >>> fs = 1/dt
    >>> w = 6.
    >>> sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
    >>> freq = np.linspace(1, fs/2, 100)
    >>> widths = w*fs / (2*freq*np.pi)
    >>> cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
    >>> plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
    >>> plt.show()

    """
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
    output = np.sqrt(1/s) * wavelet
    return output


def cwt(data, wavelet, widths, dtype=None, **kwargs):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter. The `wavelet` function
    is allowed to be complex.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.
    dtype : data-type, optional
        The desired data type of output. Defaults to ``float64`` if the
        output of `wavelet` is real and ``complex128`` if it is complex.

        .. versionadded:: 1.4.0

    kwargs
        Keyword arguments passed to wavelet function.

        .. versionadded:: 1.4.0

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----

    .. versionadded:: 1.4.0

    For non-symmetric, complex-valued wavelets, the input signal is convolved
    with the time-reversed complex-conjugate of the wavelet data [1].

    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = signal.convolve(data, np.conj(wavelet(length, width[ii],
                                        **kwargs))[::-1], mode='same')

    References
    ----------
    .. [1] S. Mallat, "A Wavelet Tour of Signal Processing (3rd Edition)",
        Academic Press, 2009.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    >>> widths = np.arange(1, 31)
    >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)
    >>> plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    >>> plt.show()
    """
    if wavelet == ricker:
        window_size = kwargs.pop('window_size', None)
    # Determine output type
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        # the conditional block below and the window_size
        # kwarg pop above may be removed eventually; these
        # are shims for 32-bit arch + NumPy <= 1.14.5 to
        # address gh-11095
        if wavelet == ricker and window_size is None:
            ceil = np.ceil(N)
            if ceil != N:
                N = int(N)
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode='same')
    return output
