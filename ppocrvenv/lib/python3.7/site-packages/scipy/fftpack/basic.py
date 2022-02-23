"""
Discrete Fourier Transforms - basic.py
"""
# Created by Pearu Peterson, August,September 2002
__all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
           'fft2','ifft2']

from scipy.fft import _pocketfft
from .helper import _good_shape


def fft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)``, where

    ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.

    Parameters
    ----------
    x : array_like
        Array to Fourier transform.
    n : int, optional
        Length of the Fourier transform. If ``n < x.shape[axis]``, `x` is
        truncated. If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the fft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    z : complex ndarray
        with the elements::

            [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
            [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd

        where::

            y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1

    See Also
    --------
    ifft : Inverse FFT
    rfft : FFT of a real sequence

    Notes
    -----
    The packing of the result is "standard": If ``A = fft(a, n)``, then
    ``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
    positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
    terms, in order of decreasingly negative frequency. So ,for an 8-point
    transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
    To rearrange the fft output so that the zero-frequency component is
    centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    Note that if ``x`` is real-valued, then ``A[j] == A[n-j].conjugate()``.
    If ``x`` is real-valued and ``n`` is even, then ``A[n/2]`` is real.

    If the data type of `x` is real, a "real FFT" algorithm is automatically
    used, which roughly halves the computation time. To increase efficiency
    a little further, use `rfft`, which does the same calculation, but only
    outputs half of the symmetrical spectrum. If the data is both real and
    symmetrical, the `dct` can again double the efficiency by generating
    half of the spectrum from half of the signal.

    Examples
    --------
    >>> from scipy.fftpack import fft, ifft
    >>> x = np.arange(5)
    >>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
    True

    """
    return _pocketfft.fft(x, n, axis, None, overwrite_x)


def ifft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return discrete inverse Fourier transform of real or complex sequence.

    The returned complex array contains ``y(0), y(1),..., y(n-1)``, where

    ``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,
        `x` is truncated. If ``n > x.shape[axis]``, `x` is zero-padded.
        The default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the ifft's are computed; the default is over the
        last axis (i.e., ``axis=-1``).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    ifft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    fft : Forward FFT

    Notes
    -----
    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    This function is most efficient when `n` is a power of two, and least
    efficient when `n` is prime.

    If the data type of `x` is real, a "real IFFT" algorithm is automatically
    used, which roughly halves the computation time.

    Examples
    --------
    >>> from scipy.fftpack import fft, ifft
    >>> import numpy as np
    >>> x = np.arange(5)
    >>> np.allclose(ifft(fft(x)), x, atol=1e-15)  # within numerical accuracy.
    True

    """
    return _pocketfft.ifft(x, n, axis, None, overwrite_x)


def rfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Discrete Fourier transform of a real sequence.

    Parameters
    ----------
    x : array_like, real-valued
        The data to transform.
    n : int, optional
        Defines the length of the Fourier transform. If `n` is not specified
        (the default) then ``n = x.shape[axis]``. If ``n < x.shape[axis]``,
        `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.
    axis : int, optional
        The axis along which the transform is applied. The default is the
        last axis.
    overwrite_x : bool, optional
        If set to true, the contents of `x` can be overwritten. Default is
        False.

    Returns
    -------
    z : real ndarray
        The returned real array contains::

          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even
          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd

        where::

          y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
          j = 0..n-1

    See Also
    --------
    fft, irfft, scipy.fft.rfft

    Notes
    -----
    Within numerical accuracy, ``y == rfft(irfft(y))``.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    To get an output with a complex datatype, consider using the newer
    function `scipy.fft.rfft`.

    Examples
    --------
    >>> from scipy.fftpack import fft, rfft
    >>> a = [9, -9, 1, 3]
    >>> fft(a)
    array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])
    >>> rfft(a)
    array([  4.,   8.,  12.,  16.])

    """
    return _pocketfft.rfft_fftpack(x, n, axis, None, overwrite_x)


def irfft(x, n=None, axis=-1, overwrite_x=False):
    """
    Return inverse discrete Fourier transform of real sequence x.

    The contents of `x` are interpreted as the output of the `rfft`
    function.

    Parameters
    ----------
    x : array_like
        Transformed data to invert.
    n : int, optional
        Length of the inverse Fourier transform.
        If n < x.shape[axis], x is truncated.
        If n > x.shape[axis], x is zero-padded.
        The default results in n = x.shape[axis].
    axis : int, optional
        Axis along which the ifft's are computed; the default is over
        the last axis (i.e., axis=-1).
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    irfft : ndarray of floats
        The inverse discrete Fourier transform.

    See Also
    --------
    rfft, ifft, scipy.fft.irfft

    Notes
    -----
    The returned real array contains::

        [y(0),y(1),...,y(n-1)]

    where for n is even::

        y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0] + (-1)**(j) x[n-1])

    and for n is odd::

        y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
                                     * exp(sqrt(-1)*j*k* 2*pi/n)
                    + c.c. + x[0])

    c.c. denotes complex conjugate of preceding expression.

    For details on input parameters, see `rfft`.

    To process (conjugate-symmetric) frequency-domain data with a complex
    datatype, consider using the newer function `scipy.fft.irfft`.

    Examples
    --------
    >>> from scipy.fftpack import rfft, irfft
    >>> a = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> irfft(a)
    array([ 2.6       , -3.16405192,  1.24398433, -1.14955713,  1.46962473])
    >>> irfft(rfft(a))
    array([1., 2., 3., 4., 5.])

    """
    return _pocketfft.irfft_fftpack(x, n, axis, None, overwrite_x)


def fftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return multidimensional discrete Fourier transform.

    The returned array contains::

      y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)

    where d = len(x.shape) and n = x.shape.

    Parameters
    ----------
    x : array_like
        The (N-D) array to transform.
    shape : int or array_like of ints or None, optional
        The shape of the result. If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        The axes of `x` (`y` if `shape` is not None) along which the
        transform is applied.
        The default is over all axes.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed. Default is False.

    Returns
    -------
    y : complex-valued N-D NumPy array
        The (N-D) DFT of the input array.

    See Also
    --------
    ifftn

    Notes
    -----
    If ``x`` is real-valued, then
    ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.

    Both single and double precision routines are implemented. Half precision
    inputs will be converted to single precision. Non-floating-point inputs
    will be converted to double precision. Long-double precision inputs are
    not supported.

    Examples
    --------
    >>> from scipy.fftpack import fftn, ifftn
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, fftn(ifftn(y)))
    True

    """
    shape = _good_shape(x, shape, axes)
    return _pocketfft.fftn(x, shape, axes, None, overwrite_x)


def ifftn(x, shape=None, axes=None, overwrite_x=False):
    """
    Return inverse multidimensional discrete Fourier transform.

    The sequence can be of an arbitrary type.

    The returned array contains::

      y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
         x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)

    where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.

    For description of parameters see `fftn`.

    See Also
    --------
    fftn : for detailed information.

    Examples
    --------
    >>> from scipy.fftpack import fftn, ifftn
    >>> import numpy as np
    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
    >>> np.allclose(y, ifftn(fftn(y)))
    True

    """
    shape = _good_shape(x, shape, axes)
    return _pocketfft.ifftn(x, shape, axes, None, overwrite_x)


def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    """
    2-D discrete Fourier transform.

    Return the 2-D discrete Fourier transform of the 2-D argument
    `x`.

    See Also
    --------
    fftn : for detailed information.

    Examples
    --------
    >>> from scipy.fftpack import fft2, ifft2
    >>> y = np.mgrid[:5, :5][0]
    >>> y
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> np.allclose(y, ifft2(fft2(y)))
    True
    """
    return fftn(x,shape,axes,overwrite_x)


def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    """
    2-D discrete inverse Fourier transform of real or complex sequence.

    Return inverse 2-D discrete Fourier transform of
    arbitrary type sequence x.

    See `ifft` for more information.

    See also
    --------
    fft2, ifft

    Examples
    --------
    >>> from scipy.fftpack import fft2, ifft2
    >>> y = np.mgrid[:5, :5][0]
    >>> y
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> np.allclose(y, fft2(ifft2(y)))
    True

    """
    return ifftn(x,shape,axes,overwrite_x)
