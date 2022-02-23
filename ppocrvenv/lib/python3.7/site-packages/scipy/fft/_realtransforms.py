from ._basic import _dispatch
from scipy._lib.uarray import Dispatchable
import numpy as np

__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']


@_dispatch
def dctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result. If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idctn : Inverse multidimensional DCT

    Notes
    -----
    For full details of the DCT types and normalization modes, as well as
    references, see `dct`.

    Examples
    --------
    >>> from scipy.fft import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idctn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are
        None, `s` is ``x.shape``; if `s` is None but `axes` is
        not None, then `s` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the IDCT is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dctn : multidimensional DCT

    Notes
    -----
    For full details of the IDCT types and normalization modes, as well as
    references, see `idct`.

    Examples
    --------
    >>> from scipy.fft import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `shape` is -1, the size of the corresponding dimension
        of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the DST is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idstn : Inverse multidimensional DST

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see `dst`.

    Examples
    --------
    >>> from scipy.fft import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idstn(x, type=2, s=None, axes=None, norm=None, overwrite_x=False,
          workers=None):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    s : int or array_like of ints or None, optional
        The shape of the result.  If both `s` and `axes` (see below) are None,
        `s` is ``x.shape``; if `s` is None but `axes` is not None, then `s` is
        ``numpy.take(x.shape, axes, axis=0)``.
        If ``s[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``s[i] < x.shape[i]``, the ith dimension is truncated to length
        ``s[i]``.
        If any element of `s` is -1, the size of the corresponding dimension of
        `x` is used.
    axes : int or array_like of ints or None, optional
        Axes over which the IDST is computed. If not given, the last ``len(s)``
        axes are used, or all axes if `s` is also not specified.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dstn : multidimensional DST

    Notes
    -----
    For full details of the IDST types and normalization modes, as well as
    references, see `idst`.

    Examples
    --------
    >>> from scipy.fft import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y)))
    True

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    r"""Return the Discrete Cosine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the dct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idct : Inverse DCT

    Notes
    -----
    For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
    MATLAB ``dct(x)``.

    For ``norm="backward"``, there is no scaling on `dct` and the `idct` is
    scaled by ``1/N`` where ``N`` is the "logical" size of the DCT. For
    ``norm="forward"`` the ``1/N`` normalization is applied to the forward
    `dct` instead and the `idct` is unnormalized. For ``norm='ortho'`` both
    directions are scaled by the same factor of ``1/sqrt(N)``.

    There are, theoretically, 8 types of the DCT, only the first 4 types are
    implemented in SciPy.'The' DCT generally refers to DCT type 2, and 'the'
    Inverse DCT generally refers to DCT type 3.

    **Type I**

    There are several definitions of the DCT-I; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
       \frac{\pi k n}{N-1} \right)

    If ``norm='ortho'``, ``x[0]`` and ``x[N-1]`` are multiplied by a scaling
    factor of :math:`\sqrt{2}`, and ``y[k]`` is multiplied by a scaling factor
    ``f``

    .. math::

        f = \begin{cases}
         \frac{1}{2}\sqrt{\frac{1}{N-1}} & \text{if }k=0\text{ or }N-1, \\
         \frac{1}{2}\sqrt{\frac{2}{N-1}} & \text{otherwise} \end{cases}

    .. note::
       The DCT-I is only supported for input size > 1.

    **Type II**

    There are several definitions of the DCT-II; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

    If ``norm="ortho"``, ``y[k]`` is multiplied by a scaling factor ``f``

    .. math::
       f = \begin{cases}
       \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
       \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

    which makes the corresponding matrix of coefficients orthonormal
    (``O @ O.T = np.eye(N)``).

    **Type III**

    There are several definitions, we use the following (for
    ``norm="backward"``)

    .. math::

       y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

    or, for ``norm="ortho"``

    .. math::

       y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
       \cos\left(\frac{\pi(2k+1)n}{2N}\right)

    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
    to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
    the orthonormalized DCT-II.

    **Type IV**

    There are several definitions of the DCT-IV; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)

    If ``norm="ortho"``, ``y[k]`` is multiplied by a scaling factor ``f``

    .. math::

        f = \frac{1}{\sqrt{2N}}

    References
    ----------
    .. [1] 'A Fast Cosine Transform in One and Two Dimensions', by J.
           Makhoul, `IEEE Transactions on acoustics, speech and signal
           processing` vol. 28(1), pp. 27-34,
           :doi:`10.1109/TASSP.1980.1163351` (1980).
    .. [2] Wikipedia, "Discrete cosine transform",
           https://en.wikipedia.org/wiki/Discrete_cosine_transform

    Examples
    --------
    The Type 1 DCT is equivalent to the FFT (though faster) for real,
    even-symmetrical inputs. The output is also real and even-symmetrical.
    Half of the FFT input is used to generate half of the FFT output:

    >>> from scipy.fft import fft, dct
    >>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
    array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
    >>> dct(np.array([4., 3., 5., 10.]), 1)
    array([ 30.,  -8.,   6.,  -2.])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None):
    """
    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    idct : ndarray of real
        The transformed input array.

    See Also
    --------
    dct : Forward DCT

    Notes
    -----
    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
    MATLAB ``idct(x)``.

    'The' IDCT is the IDCT-II, which is the same as the normalized DCT-III.

    The IDCT is equivalent to a normal DCT except for the normalization and
    type. DCT type 1 and 4 are their own inverse and DCTs 2 and 3 are each
    other's inverses.

    Examples
    --------
    The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
    inputs. The output is also real and even-symmetrical. Half of the IFFT
    input is used to generate half of the IFFT output:

    >>> from scipy.fft import ifft, idct
    >>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
    array([  4.,   3.,   5.,  10.,   5.,   3.])
    >>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1)
    array([  4.,   3.,   5.,  10.])

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, workers=None):
    r"""Return the Discrete Sine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the dst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    dst : ndarray of reals
        The transformed input array.

    See Also
    --------
    idst : Inverse DST

    Notes
    -----
    For a single dimension array ``x``.

    For ``norm="backward"``, there is no scaling on the `dst` and the `idst` is
    scaled by ``1/N`` where ``N`` is the "logical" size of the DST. For
    ``norm='ortho'`` both directions are scaled by the same factor
    ``1/sqrt(N)``.

    There are, theoretically, 8 types of the DST for different combinations of
    even/odd boundary conditions and boundary off sets [1]_, only the first
    4 types are implemented in SciPy.

    **Type I**

    There are several definitions of the DST-I; we use the following for
    ``norm="backward"``. DST-I assumes the input is odd around :math:`n=-1` and
    :math:`n=N`.

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)

    Note that the DST-I is only supported for input size > 1.
    The (unnormalized) DST-I is its own inverse, up to a factor :math:`2(N+1)`.
    The orthonormalized DST-I is exactly its own inverse.

    **Type II**

    There are several definitions of the DST-II; we use the following for
    ``norm="backward"``. DST-II assumes the input is odd around :math:`n=-1/2` and
    :math:`n=N-1/2`; the output is odd around :math:`k=-1` and even around :math:`k=N-1`

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(2n+1)}{2N}\right)

    if ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

    .. math::

        f = \begin{cases}
        \sqrt{\frac{1}{4N}} & \text{if }k = 0, \\
        \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

    **Type III**

    There are several definitions of the DST-III, we use the following (for
    ``norm="backward"``). DST-III assumes the input is odd around :math:`n=-1` and
    even around :math:`n=N-1`

    .. math::

        y_k = (-1)^k x_{N-1} + 2 \sum_{n=0}^{N-2} x_n \sin\left(
        \frac{\pi(2k+1)(n+1)}{2N}\right)

    The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up
    to a factor :math:`2N`. The orthonormalized DST-III is exactly the inverse of the
    orthonormalized DST-II.

    **Type IV**

    There are several definitions of the DST-IV, we use the following (for
    ``norm="backward"``). DST-IV assumes the input is odd around :math:`n=-0.5` and
    even around :math:`n=N-0.5`

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)

    The (unnormalized) DST-IV is its own inverse, up to a factor :math:`2N`. The
    orthonormalized DST-IV is exactly its own inverse.

    References
    ----------
    .. [1] Wikipedia, "Discrete sine transform",
           https://en.wikipedia.org/wiki/Discrete_sine_transform

    """
    return (Dispatchable(x, np.ndarray),)


@_dispatch
def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False,
         workers=None):
    """
    Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform. If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see Notes). Default is "backward".
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.
        See :func:`~scipy.fft.fft` for more details.

    Returns
    -------
    idst : ndarray of real
        The transformed input array.

    See Also
    --------
    dst : Forward DST

    Notes
    -----

    'The' IDST is the IDST-II, which is the same as the normalized DST-III.

    The IDST is equivalent to a normal DST except for the normalization and
    type. DST type 1 and 4 are their own inverse and DSTs 2 and 3 are each
    other's inverses.

    """
    return (Dispatchable(x, np.ndarray),)
