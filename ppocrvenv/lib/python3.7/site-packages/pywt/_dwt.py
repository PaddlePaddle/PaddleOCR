from numbers import Number

import numpy as np

from ._c99_config import _have_c99_complex
from ._extensions._pywt import Wavelet, Modes, _check_dtype, wavelist
from ._extensions._dwt import (dwt_single, dwt_axis, idwt_single, idwt_axis,
                               upcoef as _upcoef, downcoef as _downcoef,
                               dwt_max_level as _dwt_max_level,
                               dwt_coeff_len as _dwt_coeff_len)
from ._utils import string_types, _as_wavelet


__all__ = ["dwt", "idwt", "downcoef", "upcoef", "dwt_max_level",
           "dwt_coeff_len", "pad"]


def dwt_max_level(data_len, filter_len):
    r"""
    dwt_max_level(data_len, filter_len)

    Compute the maximum useful level of decomposition.

    Parameters
    ----------
    data_len : int
        Input data length.
    filter_len : int, str or Wavelet
        The wavelet filter length.  Alternatively, the name of a discrete
        wavelet or a Wavelet object can be specified.

    Returns
    -------
    max_level : int
        Maximum level.

    Notes
    -----
    The rational for the choice of levels is the maximum level where at least
    one coefficient in the output is uncorrupted by edge effects caused by
    signal extension.  Put another way, decomposition stops when the signal
    becomes shorter than the FIR filter length for a given wavelet.  This
    corresponds to:

    .. max_level = floor(log2(data_len/(filter_len - 1)))

    .. math::
        \mathtt{max\_level} = \left\lfloor\log_2\left(\mathtt{
            \frac{data\_len}{filter\_len - 1}}\right)\right\rfloor

    Examples
    --------
    >>> import pywt
    >>> w = pywt.Wavelet('sym5')
    >>> pywt.dwt_max_level(data_len=1000, filter_len=w.dec_len)
    6
    >>> pywt.dwt_max_level(1000, w)
    6
    >>> pywt.dwt_max_level(1000, 'sym5')
    6
    """
    if isinstance(filter_len, Wavelet):
        filter_len = filter_len.dec_len
    elif isinstance(filter_len, string_types):
        if filter_len in wavelist(kind='discrete'):
            filter_len = Wavelet(filter_len).dec_len
        else:
            raise ValueError(
                ("'{}', is not a recognized discrete wavelet.  A list of "
                 "supported wavelet names can be obtained via "
                 "pywt.wavelist(kind='discrete')").format(filter_len))
    elif not (isinstance(filter_len, Number) and filter_len % 1 == 0):
        raise ValueError(
            "filter_len must be an integer, discrete Wavelet object, or the "
            "name of a discrete wavelet.")

    if filter_len < 2:
        raise ValueError("invalid wavelet filter length")

    return _dwt_max_level(data_len, filter_len)


def dwt_coeff_len(data_len, filter_len, mode):
    """
    dwt_coeff_len(data_len, filter_len, mode='symmetric')

    Returns length of dwt output for given data length, filter length and mode

    Parameters
    ----------
    data_len : int
        Data length.
    filter_len : int
        Filter length.
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.

    Returns
    -------
    len : int
        Length of dwt output.

    Notes
    -----
    For all modes except periodization::

        len(cA) == len(cD) == floor((len(data) + wavelet.dec_len - 1) / 2)

    for periodization mode ("per")::

        len(cA) == len(cD) == ceil(len(data) / 2)

    """
    if isinstance(filter_len, Wavelet):
        filter_len = filter_len.dec_len

    return _dwt_coeff_len(data_len, filter_len, Modes.from_object(mode))


def dwt(data, wavelet, mode='symmetric', axis=-1):
    """
    dwt(data, wavelet, mode='symmetric', axis=-1)

    Single level Discrete Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    wavelet : Wavelet object or name
        Wavelet to use
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the
        last axis is used.

    Returns
    -------
    (cA, cD) : tuple
        Approximation and detail coefficients.

    Notes
    -----
    Length of coefficients arrays depends on the selected mode.
    For all modes except periodization:

        ``len(cA) == len(cD) == floor((len(data) + wavelet.dec_len - 1) / 2)``

    For periodization mode ("per"):

        ``len(cA) == len(cD) == ceil(len(data) / 2)``

    Examples
    --------
    >>> import pywt
    >>> (cA, cD) = pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')
    >>> cA
    array([ 2.12132034,  4.94974747,  7.77817459])
    >>> cD
    array([-0.70710678, -0.70710678, -0.70710678])

    """
    if not _have_c99_complex and np.iscomplexobj(data):
        data = np.asarray(data)
        cA_r, cD_r = dwt(data.real, wavelet, mode, axis)
        cA_i, cD_i = dwt(data.imag, wavelet, mode, axis)
        return (cA_r + 1j*cA_i, cD_r + 1j*cD_i)

    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.asarray(data, dtype=dt, order='C')
    mode = Modes.from_object(mode)
    wavelet = _as_wavelet(wavelet)

    if axis < 0:
        axis = axis + data.ndim
    if not 0 <= axis < data.ndim:
        raise np.AxisError("Axis greater than data dimensions")

    if data.ndim == 1:
        cA, cD = dwt_single(data, wavelet, mode)
        # TODO: Check whether this makes a copy
        cA, cD = np.asarray(cA, dt), np.asarray(cD, dt)
    else:
        cA, cD = dwt_axis(data, wavelet, mode, axis=axis)

    return (cA, cD)


def idwt(cA, cD, wavelet, mode='symmetric', axis=-1):
    """
    idwt(cA, cD, wavelet, mode='symmetric', axis=-1)

    Single level Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    cA : array_like or None
        Approximation coefficients.  If None, will be set to array of zeros
        with same shape as ``cD``.
    cD : array_like or None
        Detail coefficients.  If None, will be set to array of zeros
        with same shape as ``cA``.
    wavelet : Wavelet object or name
        Wavelet to use
    mode : str, optional (default: 'symmetric')
        Signal extension mode, see :ref:`Modes <ref-modes>`.
    axis: int, optional
        Axis over which to compute the inverse DWT. If not given, the
        last axis is used.

    Returns
    -------
    rec: array_like
        Single level reconstruction of signal from given coefficients.

    Examples
    --------
    >>> import pywt
    >>> (cA, cD) = pywt.dwt([1,2,3,4,5,6], 'db2', 'smooth')
    >>> pywt.idwt(cA, cD, 'db2', 'smooth')
    array([ 1.,  2.,  3.,  4.,  5.,  6.])

    One of the neat features of ``idwt`` is that one of the ``cA`` and ``cD``
    arguments can be set to None.  In that situation the reconstruction will be
    performed using only the other one.  Mathematically speaking, this is
    equivalent to passing a zero-filled array as one of the arguments.

    >>> (cA, cD) = pywt.dwt([1,2,3,4,5,6], 'db2', 'smooth')
    >>> A = pywt.idwt(cA, None, 'db2', 'smooth')
    >>> D = pywt.idwt(None, cD, 'db2', 'smooth')
    >>> A + D
    array([ 1.,  2.,  3.,  4.,  5.,  6.])

    """
    # TODO: Lots of possible allocations to eliminate (zeros_like, asarray(rec))
    # accept array_like input; make a copy to ensure a contiguous array

    if cA is None and cD is None:
        raise ValueError("At least one coefficient parameter must be "
                         "specified.")

    # for complex inputs: compute real and imaginary separately then combine
    if not _have_c99_complex and (np.iscomplexobj(cA) or np.iscomplexobj(cD)):
        if cA is None:
            cD = np.asarray(cD)
            cA = np.zeros_like(cD)
        elif cD is None:
            cA = np.asarray(cA)
            cD = np.zeros_like(cA)
        return (idwt(cA.real, cD.real, wavelet, mode, axis) +
                1j*idwt(cA.imag, cD.imag, wavelet, mode, axis))

    if cA is not None:
        dt = _check_dtype(cA)
        cA = np.asarray(cA, dtype=dt, order='C')
    if cD is not None:
        dt = _check_dtype(cD)
        cD = np.asarray(cD, dtype=dt, order='C')

    if cA is not None and cD is not None:
        if cA.dtype != cD.dtype:
            # need to upcast to common type
            if cA.dtype.kind == 'c' or cD.dtype.kind == 'c':
                dtype = np.complex128
            else:
                dtype = np.float64
            cA = cA.astype(dtype)
            cD = cD.astype(dtype)
    elif cA is None:
        cA = np.zeros_like(cD)
    elif cD is None:
        cD = np.zeros_like(cA)

    # cA and cD should be same dimension by here
    ndim = cA.ndim

    mode = Modes.from_object(mode)
    wavelet = _as_wavelet(wavelet)

    if axis < 0:
        axis = axis + ndim
    if not 0 <= axis < ndim:
        raise np.AxisError("Axis greater than coefficient dimensions")

    if ndim == 1:
        rec = idwt_single(cA, cD, wavelet, mode)
    else:
        rec = idwt_axis(cA, cD, wavelet, mode, axis=axis)

    return rec


def downcoef(part, data, wavelet, mode='symmetric', level=1):
    """
    downcoef(part, data, wavelet, mode='symmetric', level=1)

    Partial Discrete Wavelet Transform data decomposition.

    Similar to ``pywt.dwt``, but computes only one set of coefficients.
    Useful when you need only approximation or only details at the given level.

    Parameters
    ----------
    part : str
        Coefficients type:

        * 'a' - approximations reconstruction is performed
        * 'd' - details reconstruction is performed

    data : array_like
        Input signal.
    wavelet : Wavelet object or name
        Wavelet to use
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.
    level : int, optional
        Decomposition level.  Default is 1.

    Returns
    -------
    coeffs : ndarray
        1-D array of coefficients.

    See Also
    --------
    upcoef

    """
    if not _have_c99_complex and np.iscomplexobj(data):
        return (downcoef(part, data.real, wavelet, mode, level) +
                1j*downcoef(part, data.imag, wavelet, mode, level))
    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.asarray(data, dtype=dt, order='C')
    if data.ndim > 1:
        raise ValueError("downcoef only supports 1d data.")
    if part not in 'ad':
        raise ValueError("Argument 1 must be 'a' or 'd', not '%s'." % part)
    mode = Modes.from_object(mode)
    wavelet = _as_wavelet(wavelet)
    return np.asarray(_downcoef(part == 'a', data, wavelet, mode, level))


def upcoef(part, coeffs, wavelet, level=1, take=0):
    """
    upcoef(part, coeffs, wavelet, level=1, take=0)

    Direct reconstruction from coefficients.

    Parameters
    ----------
    part : str
        Coefficients type:
        * 'a' - approximations reconstruction is performed
        * 'd' - details reconstruction is performed
    coeffs : array_like
        Coefficients array to recontruct
    wavelet : Wavelet object or name
        Wavelet to use
    level : int, optional
        Multilevel reconstruction level.  Default is 1.
    take : int, optional
        Take central part of length equal to 'take' from the result.
        Default is 0.

    Returns
    -------
    rec : ndarray
        1-D array with reconstructed data from coefficients.

    See Also
    --------
    downcoef

    Examples
    --------
    >>> import pywt
    >>> data = [1,2,3,4,5,6]
    >>> (cA, cD) = pywt.dwt(data, 'db2', 'smooth')
    >>> pywt.upcoef('a', cA, 'db2') + pywt.upcoef('d', cD, 'db2')
    array([-0.25      , -0.4330127 ,  1.        ,  2.        ,  3.        ,
            4.        ,  5.        ,  6.        ,  1.78589838, -1.03108891])
    >>> n = len(data)
    >>> pywt.upcoef('a', cA, 'db2', take=n) + pywt.upcoef('d', cD, 'db2', take=n)
    array([ 1.,  2.,  3.,  4.,  5.,  6.])

    """
    if not _have_c99_complex and np.iscomplexobj(coeffs):
        return (upcoef(part, coeffs.real, wavelet, level, take) +
                1j*upcoef(part, coeffs.imag, wavelet, level, take))
    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(coeffs)
    coeffs = np.asarray(coeffs, dtype=dt, order='C')
    if coeffs.ndim > 1:
        raise ValueError("upcoef only supports 1d coeffs.")
    wavelet = _as_wavelet(wavelet)
    if part not in 'ad':
        raise ValueError("Argument 1 must be 'a' or 'd', not '%s'." % part)
    return np.asarray(_upcoef(part == 'a', coeffs, wavelet, level, take))


def pad(x, pad_widths, mode):
    """Extend a 1D signal using a given boundary mode.

    This function operates like :func:`numpy.pad` but supports all signal
    extension modes that can be used by PyWavelets discrete wavelet transforms.

    Parameters
    ----------
    x : ndarray
        The array to pad
    pad_widths : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), … (before_N, after_N))`` unique pad widths for
        each axis. ``((before, after),)`` yields same before and after pad for
        each axis. ``(pad,)`` or int is a shortcut for
        ``before = after = pad width`` for all axes.
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to array with shape increased according to
        ``pad_widths``.

    Notes
    -----
    The performance of padding in dimensions > 1 may be substantially slower
    for modes ``'smooth'`` and ``'antisymmetric'`` as these modes are not
    supported efficiently by the underlying :func:`numpy.pad` function.

    Note that the behavior of the ``'constant'`` mode here follows the
    PyWavelets convention which is different from NumPy (it is equivalent to
    ``mode='edge'`` in :func:`numpy.pad`).
    """
    x = np.asanyarray(x)

    # process pad_widths exactly as in numpy.pad
    pad_widths = np.array(pad_widths)
    pad_widths = np.round(pad_widths).astype(np.intp, copy=False)
    if pad_widths.min() < 0:
        raise ValueError("pad_widths must be > 0")
    pad_widths = np.broadcast_to(pad_widths, (x.ndim, 2)).tolist()

    if mode in ['symmetric', 'reflect']:
        xp = np.pad(x, pad_widths, mode=mode)
    elif mode in ['periodic', 'periodization']:
        if mode == 'periodization':
            # Promote odd-sized dimensions to even length by duplicating the
            # last value.
            edge_pad_widths = [(0, x.shape[ax] % 2)
                               for ax in range(x.ndim)]
            x = np.pad(x, edge_pad_widths, mode='edge')
        xp = np.pad(x, pad_widths, mode='wrap')
    elif mode == 'zero':
        xp = np.pad(x, pad_widths, mode='constant', constant_values=0)
    elif mode == 'constant':
        xp = np.pad(x, pad_widths, mode='edge')
    elif mode == 'smooth':
        def pad_smooth(vector, pad_width, iaxis, kwargs):
            # smooth extension to left
            left = vector[pad_width[0]]
            slope_left = (left - vector[pad_width[0] + 1])
            vector[:pad_width[0]] = \
                left + np.arange(pad_width[0], 0, -1) * slope_left

            # smooth extension to right
            right = vector[-pad_width[1] - 1]
            slope_right = (right - vector[-pad_width[1] - 2])
            vector[-pad_width[1]:] = \
                right + np.arange(1, pad_width[1] + 1) * slope_right
            return vector
        xp = np.pad(x, pad_widths, pad_smooth)
    elif mode == 'antisymmetric':
        def pad_antisymmetric(vector, pad_width, iaxis, kwargs):
            # smooth extension to left
            # implement by flipping portions symmetric padding
            npad_l, npad_r = pad_width
            vsize_nonpad = vector.size - npad_l - npad_r
            # Note: must modify vector in-place
            vector[:] = np.pad(vector[pad_width[0]:-pad_width[-1]],
                               pad_width, mode='symmetric')
            vp = vector
            r_edge = npad_l + vsize_nonpad - 1
            l_edge = npad_l
            # width of each reflected segment
            seg_width = vsize_nonpad
            # flip reflected segments on the right of the original signal
            n = 1
            while r_edge <= vp.size:
                segment_slice = slice(r_edge + 1,
                                      min(r_edge + 1 + seg_width, vp.size))
                if n % 2:
                    vp[segment_slice] *= -1
                r_edge += seg_width
                n += 1

            # flip reflected segments on the left of the original signal
            n = 1
            while l_edge >= 0:
                segment_slice = slice(max(0, l_edge - seg_width), l_edge)
                if n % 2:
                    vp[segment_slice] *= -1
                l_edge -= seg_width
                n += 1
            return vector
        xp = np.pad(x, pad_widths, pad_antisymmetric)
    elif mode == 'antireflect':
        xp = np.pad(x, pad_widths, mode='reflect', reflect_type='odd')
    else:
        raise ValueError(
            ("unsupported mode: {}. The supported modes are {}").format(
                mode, Modes.modes))
    return xp
