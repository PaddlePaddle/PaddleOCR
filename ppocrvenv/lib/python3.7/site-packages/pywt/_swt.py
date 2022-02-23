import warnings
from itertools import product

import numpy as np

from ._c99_config import _have_c99_complex
from ._extensions._dwt import idwt_single
from ._extensions._swt import swt_max_level, swt as _swt, swt_axis as _swt_axis
from ._extensions._pywt import Wavelet, Modes, _check_dtype
from ._multidim import idwt2, idwtn
from ._utils import _as_wavelet, _wavelets_per_axis


__all__ = ["swt", "swt_max_level", 'iswt', 'swt2', 'iswt2', 'swtn', 'iswtn']


def _rescale_wavelet_filterbank(wavelet, sf):
    wav = Wavelet(wavelet.name + 'r',
                  [np.asarray(f) * sf for f in wavelet.filter_bank])

    # copy attributes from the original wavelet
    wav.orthogonal = wavelet.orthogonal
    wav.biorthogonal = wavelet.biorthogonal
    return wav


def swt(data, wavelet, level=None, start_level=0, axis=-1,
        trim_approx=False, norm=False):
    """
    Multilevel 1D stationary wavelet transform.

    Parameters
    ----------
    data :
        Input signal
    wavelet :
        Wavelet to use (Wavelet object or name)
    level : int, optional
        The number of decomposition steps to perform.
    start_level : int, optional
        The level at which the decomposition will begin (it allows one to
        skip a given number of transform steps and compute
        coefficients starting from start_level) (default: 0)
    axis: int, optional
        Axis over which to compute the SWT. If not given, the
        last axis is used.
    trim_approx : bool, optional
        If True, approximation coefficients at the final level are retained.
    norm : bool, optional
        If True, transform is normalized so that the energy of the coefficients
        will be equal to the energy of ``data``. In other words,
        ``np.linalg.norm(data.ravel())`` will equal the norm of the
        concatenated transform coefficients when ``trim_approx`` is True.

    Returns
    -------
    coeffs : list
        List of approximation and details coefficients pairs in order
        similar to wavedec function::

            [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]

        where n equals input parameter ``level``.

        If ``start_level = m`` is given, then the beginning m steps are
        skipped::

            [(cAm+n, cDm+n), ..., (cAm+1, cDm+1), (cAm, cDm)]

        If ``trim_approx`` is ``True``, then the output list is exactly as in
        ``pywt.wavedec``, where the first coefficient in the list is the
        approximation coefficient at the final level and the rest are the
        detail coefficients::

            [cAn, cDn, ..., cD2, cD1]

    Notes
    -----
    The implementation here follows the "algorithm a-trous" and requires that
    the signal length along the transformed axis be a multiple of ``2**level``.
    If this is not the case, the user should pad up to an appropriate size
    using a function such as ``numpy.pad``.

    A primary benefit of this transform in comparison to its decimated
    counterpart (``pywt.wavedecn``), is that it is shift-invariant. This comes
    at cost of redundancy in the transform (the size of the output coefficients
    is larger than the input).

    When the following three conditions are true:

        1. The wavelet is orthogonal
        2. ``swt`` is called with ``norm=True``
        3. ``swt`` is called with ``trim_approx=True``

    the transform has the following additional properties that may be
    desirable in applications:

        1. energy is conserved
        2. variance is partitioned across scales

    When used with ``norm=True``, this transform is closely related to the
    multiple-overlap DWT (MODWT) as popularized for time-series analysis,
    although the underlying implementation is slightly different from the one
    published in [1]_. Specifically, the implementation used here requires a
    signal that is a multiple of ``2**level`` in length.

    References
    ----------
    .. [1] DB Percival and AT Walden. Wavelet Methods for Time Series Analysis.
        Cambridge University Press, 2000.
    """

    if not _have_c99_complex and np.iscomplexobj(data):
        data = np.asarray(data)
        kwargs = dict(wavelet=wavelet, level=level, start_level=start_level,
                      trim_approx=trim_approx, axis=axis, norm=norm)
        coeffs_real = swt(data.real, **kwargs)
        coeffs_imag = swt(data.imag, **kwargs)
        if not trim_approx:
            coeffs_cplx = []
            for (cA_r, cD_r), (cA_i, cD_i) in zip(coeffs_real, coeffs_imag):
                coeffs_cplx.append((cA_r + 1j*cA_i, cD_r + 1j*cD_i))
        else:
            coeffs_cplx = [cr + 1j*ci
                           for (cr, ci) in zip(coeffs_real, coeffs_imag)]
        return coeffs_cplx

    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.array(data, dtype=dt)

    wavelet = _as_wavelet(wavelet)
    if norm:
        if not wavelet.orthogonal:
            warnings.warn(
                "norm=True, but the wavelet is not orthogonal: \n"
                "\tThe conditions for energy preservation are not satisfied.")
        wavelet = _rescale_wavelet_filterbank(wavelet, 1/np.sqrt(2))

    if axis < 0:
        axis = axis + data.ndim
    if not 0 <= axis < data.ndim:
        raise np.AxisError("Axis greater than data dimensions")

    if level is None:
        level = swt_max_level(data.shape[axis])

    if data.ndim == 1:
        ret = _swt(data, wavelet, level, start_level, trim_approx)
    else:
        ret = _swt_axis(data, wavelet, level, start_level, axis, trim_approx)
    return ret


def iswt(coeffs, wavelet, norm=False, axis=-1):
    """
    Multilevel 1D inverse discrete stationary wavelet transform.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list of tuples::

            [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]

        where cA is approximation, cD is details.  Index 1 corresponds to
        ``start_level`` from ``pywt.swt``.
    wavelet : Wavelet object or name string
        Wavelet to use
    norm : bool, optional
        Controls the normalization used by the inverse transform. This must
        be set equal to the value that was used by ``pywt.swt`` to preserve the
        energy of a round-trip transform.

    Returns
    -------
    1D array of reconstructed data.

    Examples
    --------
    >>> import pywt
    >>> coeffs = pywt.swt([1,2,3,4,5,6,7,8], 'db2', level=2)
    >>> pywt.iswt(coeffs, 'db2')
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    """
    # copy to avoid modification of input data
    # If swt was called with trim_approx=False, first element is a tuple
    trim_approx = not isinstance(coeffs[0], (tuple, list))
    cA = coeffs[0] if trim_approx else coeffs[0][0]
    if cA.ndim > 1:
        # convert to swtn coefficient format and call iswtn
        if trim_approx:
            coeffs_nd = [cA] + [{'d': d} for d in coeffs[1:]]
        else:
            coeffs_nd = [{'a': a, 'd': d} for a, d in coeffs]
        return iswtn(coeffs_nd, wavelet, axes=(axis,), norm=norm)
    elif axis != 0 and axis != -1:
        raise np.AxisError("Axis greater than data dimensions")
    if not _have_c99_complex and np.iscomplexobj(cA):
        if trim_approx:
            coeffs_real = [c.real for c in coeffs]
            coeffs_imag = [c.imag for c in coeffs]
        else:
            coeffs_real = [(ca.real, cd.real) for ca, cd in coeffs]
            coeffs_imag = [(ca.imag, cd.imag) for ca, cd in coeffs]
        kwargs = dict(wavelet=wavelet, norm=norm)
        y = iswt(coeffs_real, **kwargs)
        return y + 1j * iswt(coeffs_imag, **kwargs)

    if trim_approx:
        coeffs = coeffs[1:]

    if cA.ndim != 1:
        raise ValueError("iswt only supports 1D data")

    dt = _check_dtype(cA)
    output = np.array(cA, dtype=dt, copy=True)

    # num_levels, equivalent to the decomposition level, n
    num_levels = len(coeffs)
    wavelet = _as_wavelet(wavelet)
    if norm:
        wavelet = _rescale_wavelet_filterbank(wavelet, np.sqrt(2))
    mode = Modes.from_object('periodization')
    for j in range(num_levels, 0, -1):
        step_size = int(pow(2, j-1))
        last_index = step_size
        if trim_approx:
            cD = coeffs[-j]
        else:
            _, cD = coeffs[-j]
        cD = np.asarray(cD, dtype=_check_dtype(cD))
        if cD.dtype != output.dtype:
            # upcast to a common dtype (float64 or complex128)
            if output.dtype.kind == 'c' or cD.dtype.kind == 'c':
                dtype = np.complex128
            else:
                dtype = np.float64
            output = np.asarray(output, dtype=dtype)
            cD = np.asarray(cD, dtype=dtype)
        for first in range(last_index):  # 0 to last_index - 1

            # Getting the indices that we will transform
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2]
            # select the odd indices
            odd_indices = indices[1::2]

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            # Note:  indexing with an array of ints returns a contiguous
            #        copy as required by idwt_single.
            x1 = idwt_single(output[even_indices],
                             cD[even_indices],
                             wavelet, mode)
            x2 = idwt_single(output[odd_indices],
                             cD[odd_indices],
                             wavelet, mode)

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.

    return output


def swt2(data, wavelet, level, start_level=0, axes=(-2, -1),
         trim_approx=False, norm=False):
    """
    Multilevel 2D stationary wavelet transform.

    Parameters
    ----------
    data : array_like
        2D array with input data
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple of wavelets to apply per
        axis in ``axes``.
    level : int
        The number of decomposition steps to perform.
    start_level : int, optional
        The level at which the decomposition will start (default: 0)
    axes : 2-tuple of ints, optional
        Axes over which to compute the SWT. Repeated elements are not allowed.
    trim_approx : bool, optional
        If True, approximation coefficients at the final level are retained.
    norm : bool, optional
        If True, transform is normalized so that the energy of the coefficients
        will be equal to the energy of ``data``. In other words,
        ``np.linalg.norm(data.ravel())`` will equal the norm of the
        concatenated transform coefficients when ``trim_approx`` is True.

    Returns
    -------
    coeffs : list
        Approximation and details coefficients (for ``start_level = m``).
        If ``trim_approx`` is ``False``, approximation coefficients are
        retained for all levels::

            [
                (cA_m+level,
                    (cH_m+level, cV_m+level, cD_m+level)
                ),
                ...,
                (cA_m+1,
                    (cH_m+1, cV_m+1, cD_m+1)
                ),
                (cA_m,
                    (cH_m, cV_m, cD_m)
                )
            ]

        where cA is approximation, cH is horizontal details, cV is
        vertical details, cD is diagonal details and m is ``start_level``.

        If ``trim_approx`` is ``True``, approximation coefficients are only
        retained at the final level of decomposition. This matches the format
        used by ``pywt.wavedec2``::

            [
                cA_m+level,
                (cH_m+level, cV_m+level, cD_m+level),
                ...,
                (cH_m+1, cV_m+1, cD_m+1),
                (cH_m, cV_m, cD_m),
            ]

    Notes
    -----
    The implementation here follows the "algorithm a-trous" and requires that
    the signal length along the transformed axes be a multiple of ``2**level``.
    If this is not the case, the user should pad up to an appropriate size
    using a function such as ``numpy.pad``.

    A primary benefit of this transform in comparison to its decimated
    counterpart (``pywt.wavedecn``), is that it is shift-invariant. This comes
    at cost of redundancy in the transform (the size of the output coefficients
    is larger than the input).

    When the following three conditions are true:

        1. The wavelet is orthogonal
        2. ``swt2`` is called with ``norm=True``
        3. ``swt2`` is called with ``trim_approx=True``

    the transform has the following additional properties that may be
    desirable in applications:

        1. energy is conserved
        2. variance is partitioned across scales

    """
    axes = tuple(axes)
    data = np.asarray(data)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to swt2 must be unique.")
    if data.ndim < len(np.unique(axes)):
        raise ValueError("Input array has fewer dimensions than the specified "
                         "axes")

    coefs = swtn(data, wavelet, level, start_level, axes, trim_approx, norm)
    ret = []
    if trim_approx:
        ret.append(coefs[0])
        coefs = coefs[1:]
    for c in coefs:
        if trim_approx:
            ret.append((c['da'], c['ad'], c['dd']))
        else:
            ret.append((c['aa'], (c['da'], c['ad'], c['dd'])))
    return ret


def iswt2(coeffs, wavelet, norm=False, axes=(-2, -1)):
    """
    Multilevel 2D inverse discrete stationary wavelet transform.

    Parameters
    ----------
    coeffs : list
        Approximation and details coefficients::

            [
                (cA_n,
                    (cH_n, cV_n, cD_n)
                ),
                ...,
                (cA_2,
                    (cH_2, cV_2, cD_2)
                ),
                (cA_1,
                    (cH_1, cV_1, cD_1)
                )
            ]

        where cA is approximation, cH is horizontal details, cV is
        vertical details, cD is diagonal details and n is the number of
        levels.  Index 1 corresponds to ``start_level`` from ``pywt.swt2``.
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a 2-tuple of wavelets to apply per
        axis.
    norm : bool, optional
        Controls the normalization used by the inverse transform. This must
        be set equal to the value that was used by ``pywt.swt2`` to preserve
        the energy of a round-trip transform.

    Returns
    -------
    2D array of reconstructed data.

    Examples
    --------
    >>> import pywt
    >>> coeffs = pywt.swt2([[1,2,3,4],[5,6,7,8],
    ...                     [9,10,11,12],[13,14,15,16]],
    ...                    'db1', level=2)
    >>> pywt.iswt2(coeffs, 'db1')
    array([[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.],
           [ 13.,  14.,  15.,  16.]])

    """

    # If swt was called with trim_approx=False, first element is a tuple
    trim_approx = not isinstance(coeffs[0], (tuple, list))
    cA = coeffs[0] if trim_approx else coeffs[0][0]
    if cA.ndim != 2 or axes != (-2, -1):
        # convert to swtn coefficient format and call iswtn instead
        if trim_approx:
            coeffs_nd = [cA] + [{'da': h, 'ad': v, 'dd': d}
                                for h, v, d in coeffs[1:]]
        else:
            coeffs_nd = [{'aa': a, 'da': h, 'ad': v, 'dd': d}
                         for a, (h, v, d) in coeffs]
        return iswtn(coeffs_nd, wavelet, axes=axes, norm=norm)
    if not _have_c99_complex and np.iscomplexobj(cA):
        if trim_approx:
            coeffs_real = [cA.real]
            coeffs_real += [(h.real, v.real, d.real) for h, v, d in coeffs[1:]]
            coeffs_imag = [cA.imag]
            coeffs_imag += [(h.imag, v.imag, d.imag) for h, v, d in coeffs[1:]]
        else:
            coeffs_real = [(a.real, (h.real, v.real, d.real))
                            for a, (h, v, d) in coeffs]
            coeffs_imag = [(a.imag, (h.imag, v.imag, d.imag))
                            for a, (h, v, d) in coeffs]
        kwargs = dict(wavelet=wavelet, norm=norm)
        y = iswt2(coeffs_real, **kwargs)
        return y + 1j * iswt2(coeffs_imag, **kwargs)

    if trim_approx:
        coeffs = coeffs[1:]

    # copy to avoid modification of input data
    dt = _check_dtype(cA)
    output = np.array(cA, dtype=dt, copy=True)

    if output.ndim != 2:
        raise ValueError(
            "iswt2 only supports 2D arrays.  see iswtn for a general "
            "n-dimensionsal ISWT")
    # num_levels, equivalent to the decomposition level, n
    num_levels = len(coeffs)
    wavelets = _wavelets_per_axis(wavelet, axes=(0, 1))
    if norm:
        wavelets = [_rescale_wavelet_filterbank(wav, np.sqrt(2))
                    for wav in wavelets]

    for j in range(num_levels):
        step_size = int(pow(2, num_levels-j-1))
        last_index = step_size
        if trim_approx:
            (cH, cV, cD) = coeffs[j]
        else:
            _, (cH, cV, cD) = coeffs[j]
        # We are going to assume cH, cV, and cD are of equal size
        if (cH.shape != cV.shape) or (cH.shape != cD.shape):
            raise RuntimeError(
                "Mismatch in shape of intermediate coefficient arrays")

        # make sure output shares the common dtype
        # (conversion of dtype for individual coeffs is handled within idwt2 )
        common_dtype = np.result_type(*(
            [dt, ] + [_check_dtype(c) for c in [cH, cV, cD]]))
        if output.dtype != common_dtype:
            output = output.astype(common_dtype)

        for first_h in range(last_index):  # 0 to last_index - 1
            for first_w in range(last_index):  # 0 to last_index - 1
                # Getting the indices that we will transform
                indices_h = slice(first_h, cH.shape[0], step_size)
                indices_w = slice(first_w, cH.shape[1], step_size)

                even_idx_h = slice(first_h, cH.shape[0], 2*step_size)
                even_idx_w = slice(first_w, cH.shape[1], 2*step_size)
                odd_idx_h = slice(first_h + step_size, cH.shape[0], 2*step_size)
                odd_idx_w = slice(first_w + step_size, cH.shape[1], 2*step_size)

                # perform the inverse dwt on the selected indices,
                # making sure to use periodic boundary conditions
                x1 = idwt2((output[even_idx_h, even_idx_w],
                           (cH[even_idx_h, even_idx_w],
                            cV[even_idx_h, even_idx_w],
                            cD[even_idx_h, even_idx_w])),
                           wavelets, 'periodization')
                x2 = idwt2((output[even_idx_h, odd_idx_w],
                           (cH[even_idx_h, odd_idx_w],
                            cV[even_idx_h, odd_idx_w],
                            cD[even_idx_h, odd_idx_w])),
                           wavelets, 'periodization')
                x3 = idwt2((output[odd_idx_h, even_idx_w],
                           (cH[odd_idx_h, even_idx_w],
                            cV[odd_idx_h, even_idx_w],
                            cD[odd_idx_h, even_idx_w])),
                           wavelets, 'periodization')
                x4 = idwt2((output[odd_idx_h, odd_idx_w],
                           (cH[odd_idx_h, odd_idx_w],
                            cV[odd_idx_h, odd_idx_w],
                            cD[odd_idx_h, odd_idx_w])),
                           wavelets, 'periodization')

                # perform a circular shifts
                x2 = np.roll(x2, 1, axis=1)
                x3 = np.roll(x3, 1, axis=0)
                x4 = np.roll(x4, 1, axis=0)
                x4 = np.roll(x4, 1, axis=1)
                output[indices_h, indices_w] = (x1 + x2 + x3 + x4) / 4

    return output


def swtn(data, wavelet, level, start_level=0, axes=None, trim_approx=False,
         norm=False):
    """
    n-dimensional stationary wavelet transform.

    Parameters
    ----------
    data : array_like
        n-dimensional array with input data.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple of wavelets to apply per
        axis in ``axes``.
    level : int
        The number of decomposition steps to perform.
    start_level : int, optional
        The level at which the decomposition will start (default: 0)
    axes : sequence of ints, optional
        Axes over which to compute the SWT. A value of ``None`` (the
        default) selects all axes. Axes may not be repeated.
    trim_approx : bool, optional
        If True, approximation coefficients at the final level are retained.
    norm : bool, optional
        If True, transform is normalized so that the energy of the coefficients
        will be equal to the energy of ``data``. In other words,
        ``np.linalg.norm(data.ravel())`` will equal the norm of the
        concatenated transform coefficients when ``trim_approx`` is True.

    Returns
    -------
    [{coeffs_level_n}, ..., {coeffs_level_1}]: list of dict
        Results for each level are arranged in a dictionary, where the key
        specifies the transform type on each dimension and value is a
        n-dimensional coefficients array.

        For example, for a 2D case the result at a given level will look
        something like this::

            {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
             'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
             'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
             'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
            }

        For user-specified ``axes``, the order of the characters in the
        dictionary keys map to the specified ``axes``.

        If ``trim_approx`` is ``True``, the first element of the list contains
        the array of approximation coefficients from the final level of
        decomposition, while the remaining coefficient dictionaries contain
        only detail coefficients. This matches the behavior of `pywt.wavedecn`.

    Notes
    -----
    The implementation here follows the "algorithm a-trous" and requires that
    the signal length along the transformed axes be a multiple of ``2**level``.
    If this is not the case, the user should pad up to an appropriate size
    using a function such as ``numpy.pad``.

    A primary benefit of this transform in comparison to its decimated
    counterpart (``pywt.wavedecn``), is that it is shift-invariant. This comes
    at cost of redundancy in the transform (the size of the output coefficients
    is larger than the input).

    When the following three conditions are true:

        1. The wavelet is orthogonal
        2. ``swtn`` is called with ``norm=True``
        3. ``swtn`` is called with ``trim_approx=True``

    the transform has the following additional properties that may be
    desirable in applications:

        1. energy is conserved
        2. variance is partitioned across scales

    """
    data = np.asarray(data)
    if not _have_c99_complex and np.iscomplexobj(data):
        kwargs = dict(wavelet=wavelet, level=level, start_level=start_level,
                      trim_approx=trim_approx, axes=axes, norm=norm)
        real = swtn(data.real, **kwargs)
        imag = swtn(data.imag, **kwargs)
        if trim_approx:
            cplx = [real[0] + 1j * imag[0]]
            offset = 1
        else:
            cplx = []
            offset = 0
        for rdict, idict in zip(real[offset:], imag[offset:]):
            cplx.append(
                dict((k, rdict[k] + 1j * idict[k]) for k in rdict.keys()))
        return cplx

    if data.dtype == np.dtype('object'):
        raise TypeError("Input must be a numeric array-like")
    if data.ndim < 1:
        raise ValueError("Input data must be at least 1D")

    if axes is None:
        axes = range(data.ndim)
    axes = [a + data.ndim if a < 0 else a for a in axes]
    if any(a < 0 or a >= data.ndim for a in axes):
        raise np.AxisError("Axis greater than data dimensions")
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to swtn must be unique.")
    num_axes = len(axes)

    wavelets = _wavelets_per_axis(wavelet, axes)
    if norm:
        if not np.all([wav.orthogonal for wav in wavelets]):
            warnings.warn(
                "norm=True, but the wavelets used are not orthogonal: \n"
                "\tThe conditions for energy preservation are not satisfied.")
        wavelets = [_rescale_wavelet_filterbank(wav, 1/np.sqrt(2))
                    for wav in wavelets]
    ret = []
    for i in range(start_level, start_level + level):
        coeffs = [('', data)]
        for axis, wavelet in zip(axes, wavelets):
            new_coeffs = []
            for subband, x in coeffs:
                cA, cD = _swt_axis(x, wavelet, level=1, start_level=i,
                                   axis=axis)[0]
                new_coeffs.extend([(subband + 'a', cA),
                                   (subband + 'd', cD)])
            coeffs = new_coeffs

        coeffs = dict(coeffs)
        ret.append(coeffs)

        # data for the next level is the approximation coeffs from this level
        data = coeffs['a' * num_axes]
        if trim_approx:
            coeffs.pop('a' * num_axes)
    if trim_approx:
        ret.append(data)
    ret.reverse()
    return ret


def iswtn(coeffs, wavelet, axes=None, norm=False):
    """
    Multilevel nD inverse discrete stationary wavelet transform.

    Parameters
    ----------
    coeffs : list
        [{coeffs_level_n}, ..., {coeffs_level_1}]: list of dict
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple of wavelets to apply per
        axis in ``axes``.
    axes : sequence of ints, optional
        Axes over which to compute the inverse SWT. Axes may not be repeated.
        The default is ``None``, which means transform all axes
        (``axes = range(data.ndim)``).
    norm : bool, optional
        Controls the normalization used by the inverse transform. This must
        be set equal to the value that was used by ``pywt.swtn`` to preserve
        the energy of a round-trip transform.

    Returns
    -------
    nD array of reconstructed data.

    Examples
    --------
    >>> import pywt
    >>> coeffs = pywt.swtn([[1,2,3,4],[5,6,7,8],
    ...                     [9,10,11,12],[13,14,15,16]],
    ...                    'db1', level=2)
    >>> pywt.iswtn(coeffs, 'db1')
    array([[  1.,   2.,   3.,   4.],
           [  5.,   6.,   7.,   8.],
           [  9.,  10.,  11.,  12.],
           [ 13.,  14.,  15.,  16.]])

    """

    # key length matches the number of axes transformed
    ndim_transform = max(len(key) for key in coeffs[-1].keys())
    trim_approx = not isinstance(coeffs[0], dict)
    cA = coeffs[0] if trim_approx else coeffs[0]['a'*ndim_transform]

    if not _have_c99_complex and np.iscomplexobj(cA):
        if trim_approx:
            coeffs_real = [coeffs[0].real]
            coeffs_imag = [coeffs[0].imag]
            coeffs = coeffs[1:]
        else:
            coeffs_real = []
            coeffs_imag = []
        coeffs_real += [{k: v.real for k, v in c.items()} for c in coeffs]
        coeffs_imag += [{k: v.imag for k, v in c.items()} for c in coeffs]
        kwargs = dict(wavelet=wavelet, axes=axes, norm=norm)
        y = iswtn(coeffs_real, **kwargs)
        return y + 1j * iswtn(coeffs_imag, **kwargs)

    if trim_approx:
        coeffs = coeffs[1:]

    # copy to avoid modification of input data
    dt = _check_dtype(cA)
    output = np.array(cA, dtype=dt, copy=True)
    ndim = output.ndim

    if axes is None:
        axes = range(output.ndim)
    axes = [a + ndim if a < 0 else a for a in axes]
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to swtn must be unique.")
    if ndim_transform != len(axes):
        raise ValueError("The number of axes used in iswtn must match the "
                         "number of dimensions transformed in swtn.")

    # num_levels, equivalent to the decomposition level, n
    num_levels = len(coeffs)
    wavelets = _wavelets_per_axis(wavelet, axes)
    if norm:
        wavelets = [_rescale_wavelet_filterbank(wav, np.sqrt(2))
                    for wav in wavelets]

    # initialize various slice objects used in the loops below
    # these will remain slice(None) only on axes that aren't transformed
    indices = [slice(None), ]*ndim
    even_indices = [slice(None), ]*ndim
    odd_indices = [slice(None), ]*ndim
    odd_even_slices = [slice(None), ]*ndim

    for j in range(num_levels):
        step_size = int(pow(2, num_levels-j-1))
        last_index = step_size
        if not trim_approx:
            a = coeffs[j].pop('a'*ndim_transform)  # will restore later
        details = coeffs[j]
        # make sure dtype matches the coarsest level approximation coefficients
        common_dtype = np.result_type(*(
            [dt, ] + [v.dtype for v in details.values()]))
        if output.dtype != common_dtype:
            output = output.astype(common_dtype)

        # We assume all coefficient arrays are of equal size
        shapes = [v.shape for k, v in details.items()]
        if len(set(shapes)) != 1:
            raise RuntimeError(
                "Mismatch in shape of intermediate coefficient arrays")

        # shape of a single coefficient array, excluding non-transformed axes
        coeff_trans_shape = tuple([shapes[0][ax] for ax in axes])

        # nested loop over all combinations of axis offsets at this level
        for firsts in product(*([range(last_index), ]*ndim_transform)):
            for first, sh, ax in zip(firsts, coeff_trans_shape, axes):
                indices[ax] = slice(first, sh, step_size)
                even_indices[ax] = slice(first, sh, 2*step_size)
                odd_indices[ax] = slice(first+step_size, sh, 2*step_size)

            # nested loop over all combinations of odd/even inidices
            approx = output.copy()
            output[tuple(indices)] = 0
            ntransforms = 0
            for odds in product(*([(0, 1), ]*ndim_transform)):
                for o, ax in zip(odds, axes):
                    if o:
                        odd_even_slices[ax] = odd_indices[ax]
                    else:
                        odd_even_slices[ax] = even_indices[ax]
                # extract the odd/even indices for all detail coefficients
                details_slice = {}
                for key, value in details.items():
                    details_slice[key] = value[tuple(odd_even_slices)]
                details_slice['a'*ndim_transform] = approx[
                    tuple(odd_even_slices)]

                # perform the inverse dwt on the selected indices,
                # making sure to use periodic boundary conditions
                x = idwtn(details_slice, wavelets, 'periodization', axes=axes)
                for o, ax in zip(odds, axes):
                    # circular shift along any odd indexed axis
                    if o:
                        x = np.roll(x, 1, axis=ax)
                output[tuple(indices)] += x
                ntransforms += 1
            output[tuple(indices)] /= ntransforms  # normalize
        if not trim_approx:
            coeffs[j]['a'*ndim_transform] = a  # restore approx coeffs to dict
    return output
