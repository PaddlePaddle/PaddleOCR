# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2018 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See COPYING for license details.

"""
Multilevel 1D and 2D Discrete Wavelet Transform
and Inverse Discrete Wavelet Transform.
"""

from __future__ import division, print_function, absolute_import

import numbers
import warnings
from itertools import product
from copy import copy
import numpy as np

from ._extensions._pywt import Wavelet, Modes
from ._extensions._dwt import dwt_max_level
from ._dwt import dwt, idwt, dwt_coeff_len
from ._multidim import dwt2, idwt2, dwtn, idwtn, _fix_coeffs
from ._utils import _as_wavelet, _wavelets_per_axis, _modes_per_axis

__all__ = ['wavedec', 'waverec', 'wavedec2', 'waverec2', 'wavedecn',
           'waverecn', 'coeffs_to_array', 'array_to_coeffs', 'ravel_coeffs',
           'unravel_coeffs', 'dwtn_max_level', 'wavedecn_size',
           'wavedecn_shapes', 'fswavedecn', 'fswaverecn', 'FswavedecnResult']


def _check_level(sizes, dec_lens, level):
    if np.isscalar(sizes):
        sizes = (sizes, )
    if np.isscalar(dec_lens):
        dec_lens = (dec_lens, )
    max_level = np.min([dwt_max_level(s, d) for s, d in zip(sizes, dec_lens)])
    if level is None:
        level = max_level
    elif level < 0:
        raise ValueError(
            "Level value of %d is too low . Minimum level is 0." % level)
    elif level > max_level:
        warnings.warn(
            ("Level value of {} is too high: all coefficients will experience "
             "boundary effects.").format(level))
    return level


def wavedec(data, wavelet, mode='symmetric', level=None, axis=-1):
    """
    Multilevel 1D Discrete Wavelet Transform of data.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the ``dwt_max_level`` function.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the
        last axis is used.

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where ``n`` denotes the level of decomposition. The first element
        (``cA_n``) of the result is approximation coefficients array and the
        following elements (``cD_n`` - ``cD_1``) are details coefficients
        arrays.

    Examples
    --------
    >>> from pywt import wavedec
    >>> coeffs = wavedec([1,2,3,4,5,6,7,8], 'db1', level=2)
    >>> cA2, cD2, cD1 = coeffs
    >>> cD1
    array([-0.70710678, -0.70710678, -0.70710678, -0.70710678])
    >>> cD2
    array([-2., -2.])
    >>> cA2
    array([  5.,  13.])

    """
    data = np.asarray(data)

    wavelet = _as_wavelet(wavelet)
    try:
        axes_shape = data.shape[axis]
    except IndexError:
        raise np.AxisError("Axis greater than data dimensions")
    level = _check_level(axes_shape, wavelet.dec_len, level)

    coeffs_list = []

    a = data
    for i in range(level):
        a, d = dwt(a, wavelet, mode, axis)
        coeffs_list.append(d)

    coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list


def waverec(coeffs, wavelet, mode='symmetric', axis=-1):
    """
    Multilevel 1D Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, cDn, cDn-1, ..., cD2, cD1]
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`.
    axis: int, optional
        Axis over which to compute the inverse DWT. If not given, the
        last axis is used.

    Notes
    -----
    It may sometimes be desired to run ``waverec`` with some sets of
    coefficients omitted.  This can best be done by setting the corresponding
    arrays to zero arrays of matching shape and dtype.  Explicitly removing
    list entries or setting them to None is not supported.

    Specifically, to ignore detail coefficients at level 2, one could do::

        coeffs[-2] == np.zeros_like(coeffs[-2])

    Examples
    --------
    >>> import pywt
    >>> coeffs = pywt.wavedec([1,2,3,4,5,6,7,8], 'db1', level=2)
    >>> pywt.waverec(coeffs, 'db1')
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    """

    if not isinstance(coeffs, (list, tuple)):
        raise ValueError("Expected sequence of coefficient arrays.")

    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 arrays required).")
    elif len(coeffs) == 1:
        # level 0 transform (just returns the approximation coefficients)
        return coeffs[0]

    a, ds = coeffs[0], coeffs[1:]

    for d in ds:
        if d is not None and not isinstance(d, np.ndarray):
            raise ValueError((
                "Unexpected detail coefficient type: {}. Detail coefficients "
                "must be arrays as returned by wavedec. If you are using "
                "pywt.array_to_coeffs or pywt.unravel_coeffs, please specify "
                "output_format='wavedec'").format(type(d)))
        if (a is not None) and (d is not None):
            try:
                if a.shape[axis] == d.shape[axis] + 1:
                    a = a[tuple(slice(s) for s in d.shape)]
                elif a.shape[axis] != d.shape[axis]:
                    raise ValueError("coefficient shape mismatch")
            except IndexError:
                raise np.AxisError("Axis greater than coefficient dimensions")
        a = idwt(a, d, wavelet, mode, axis)

    return a


def wavedec2(data, wavelet, mode='symmetric', level=None, axes=(-2, -1)):
    """
    Multilevel 2D Discrete Wavelet Transform.

    Parameters
    ----------
    data : ndarray
        2D input data
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or 2-tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the ``dwt_max_level`` function.
    axes : 2-tuple of ints, optional
        Axes over which to compute the DWT. Repeated elements are not allowed.

    Returns
    -------
    [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)] : list
        Coefficients list.  For user-specified ``axes``, ``cH*``
        corresponds to ``axes[0]`` while ``cV*`` corresponds to ``axes[1]``.
        The first element returned is the approximation coefficients for the
        nth level of decomposition.  Remaining elements are tuples of detail
        coefficients in descending order of decomposition level.
        (i.e. ``cH1`` are the horizontal detail coefficients at the first
        level)

    Examples
    --------
    >>> import pywt
    >>> import numpy as np
    >>> coeffs = pywt.wavedec2(np.ones((4,4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> pywt.waverec2(coeffs, 'db1')
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])
    """
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("Expected input data to have at least 2 dimensions.")

    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to wavedec2 must be unique.")
    try:
        axes_sizes = [data.shape[ax] for ax in axes]
    except IndexError:
        raise np.AxisError("Axis greater than data dimensions")

    wavelets = _wavelets_per_axis(wavelet, axes)
    dec_lengths = [w.dec_len for w in wavelets]

    level = _check_level(axes_sizes, dec_lengths, level)

    coeffs_list = []

    a = data
    for i in range(level):
        a, ds = dwt2(a, wavelet, mode, axes)
        coeffs_list.append(ds)

    coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list


def waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1)):
    """
    Multilevel 2D Inverse Discrete Wavelet Transform.

    coeffs : list or tuple
        Coefficients list [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)]
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or 2-tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    axes : 2-tuple of ints, optional
        Axes over which to compute the IDWT. Repeated elements are not allowed.

    Returns
    -------
    2D array of reconstructed data.

    Notes
    -----
    It may sometimes be desired to run ``waverec2`` with some sets of
    coefficients omitted.  This can best be done by setting the corresponding
    arrays to zero arrays of matching shape and dtype.  Explicitly removing
    list or tuple entries or setting them to None is not supported.

    Specifically, to ignore all detail coefficients at level 2, one could do::

        coeffs[-2] == tuple([np.zeros_like(v) for v in coeffs[-2]])

    Examples
    --------
    >>> import pywt
    >>> import numpy as np
    >>> coeffs = pywt.wavedec2(np.ones((4,4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> pywt.waverec2(coeffs, 'db1')
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])
    """
    if not isinstance(coeffs, (list, tuple)):
        raise ValueError("Expected sequence of coefficient arrays.")

    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to waverec2 must be unique.")

    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 array required).")
    elif len(coeffs) == 1:
        # level 0 transform (just returns the approximation coefficients)
        return coeffs[0]

    a, ds = coeffs[0], coeffs[1:]
    a = np.asarray(a)

    for d in ds:
        if not isinstance(d, (list, tuple)) or len(d) != 3:
            raise ValueError((
                "Unexpected detail coefficient type: {}. Detail coefficients "
                "must be a 3-tuple of arrays as returned by wavedec2. If you "
                "are using pywt.array_to_coeffs or pywt.unravel_coeffs, "
                "please specify output_format='wavedec2'").format(type(d)))
        d = tuple(np.asarray(coeff) if coeff is not None else None
                  for coeff in d)
        d_shapes = (coeff.shape for coeff in d if coeff is not None)
        try:
            d_shape = next(d_shapes)
        except StopIteration:
            idxs = slice(None), slice(None)
        else:
            if not all(s == d_shape for s in d_shapes):
                raise ValueError("All detail shapes must be the same length.")
            idxs = tuple(slice(None, -1 if a_len == d_len + 1 else None)
                         for a_len, d_len in zip(a.shape, d_shape))
        a = idwt2((a[idxs], d), wavelet, mode, axes)

    return a


def _prep_axes_wavedecn(shape, axes):
    if len(shape) < 1:
        raise ValueError("Expected at least 1D input data.")
    ndim = len(shape)
    if np.isscalar(axes):
        axes = (axes, )
    if axes is None:
        axes = range(ndim)
    else:
        axes = tuple(axes)
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to wavedecn must be unique.")
    try:
        axes_shapes = [shape[ax] for ax in axes]
    except IndexError:
        raise np.AxisError("Axis greater than data dimensions")
    ndim_transform = len(axes)
    return axes, axes_shapes, ndim_transform


def wavedecn(data, wavelet, mode='symmetric', level=None, axes=None):
    """
    Multilevel nD Discrete Wavelet Transform.

    Parameters
    ----------
    data : ndarray
        nD input data
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the ``dwt_max_level`` function.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Axes may not be repeated. The
        default is None, which means transform all axes
        (``axes = range(data.ndim)``).

    Returns
    -------
    [cAn, {details_level_n}, ... {details_level_1}] : list
        Coefficients list.  Coefficients are listed in descending order of
        decomposition level.  ``cAn`` are the approximation coefficients at
        level ``n``.  Each ``details_level_i`` element is a dictionary
        containing detail coefficients at level ``i`` of the decomposition. As
        a concrete example, a 3D decomposition would have the following set of
        keys in each ``details_level_i`` dictionary::

            {'aad', 'ada', 'daa', 'add', 'dad', 'dda', 'ddd'}

        where the order of the characters in each key map to the specified
        ``axes``.

    Examples
    --------
    >>> import numpy as np
    >>> from pywt import wavedecn, waverecn
    >>> coeffs = wavedecn(np.ones((4, 4, 4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> waverecn(coeffs, 'db1')  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]]])

    """
    data = np.asarray(data)
    axes, axes_shapes, ndim_transform = _prep_axes_wavedecn(data.shape, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)
    dec_lengths = [w.dec_len for w in wavelets]

    level = _check_level(axes_shapes, dec_lengths, level)

    coeffs_list = []

    a = data
    for i in range(level):
        coeffs = dwtn(a, wavelet, mode, axes)
        a = coeffs.pop('a' * ndim_transform)
        coeffs_list.append(coeffs)

    coeffs_list.append(a)
    coeffs_list.reverse()

    return coeffs_list


def _match_coeff_dims(a_coeff, d_coeff_dict):
    # For each axis, compare the approximation coeff shape to one of the
    # stored detail coeffs and truncate the last element along the axis
    # if necessary.
    if a_coeff is None:
        return None
    if not d_coeff_dict:
        return a_coeff
    d_coeff = d_coeff_dict[next(iter(d_coeff_dict))]
    size_diffs = np.subtract(a_coeff.shape, d_coeff.shape)
    if np.any((size_diffs < 0) | (size_diffs > 1)):
        print(size_diffs)
        raise ValueError("incompatible coefficient array sizes")
    return a_coeff[tuple(slice(s) for s in d_coeff.shape)]


def waverecn(coeffs, wavelet, mode='symmetric', axes=None):
    """
    Multilevel nD Inverse Discrete Wavelet Transform.

    coeffs : array_like
        Coefficients list [cAn, {details_level_n}, ... {details_level_1}]
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    axes : sequence of ints, optional
        Axes over which to compute the IDWT.  Axes may not be repeated.

    Returns
    -------
    nD array of reconstructed data.

    Notes
    -----
    It may sometimes be desired to run ``waverecn`` with some sets of
    coefficients omitted.  This can best be done by setting the corresponding
    arrays to zero arrays of matching shape and dtype.  Explicitly removing
    list or dictionary entries or setting them to None is not supported.

    Specifically, to ignore all detail coefficients at level 2, one could do::

        coeffs[-2] = {k: np.zeros_like(v) for k, v in coeffs[-2].items()}

    Examples
    --------
    >>> import numpy as np
    >>> from pywt import wavedecn, waverecn
    >>> coeffs = wavedecn(np.ones((4, 4, 4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> waverecn(coeffs, 'db1')  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],
           [[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]]])

    """
    if len(coeffs) < 1:
        raise ValueError(
            "Coefficient list too short (minimum 1 array required).")

    a, ds = coeffs[0], coeffs[1:]

    # this dictionary check must be prior to the call to _fix_coeffs
    if len(ds) > 0 and not all([isinstance(d, dict) for d in ds]):
        raise ValueError((
            "Unexpected detail coefficient type: {}. Detail coefficients "
            "must be a dicionary of arrays as returned by wavedecn. If "
            "you are using pywt.array_to_coeffs or pywt.unravel_coeffs, "
            "please specify output_format='wavedecn'").format(type(ds[0])))

    # Raise error for invalid key combinations
    ds = list(map(_fix_coeffs, ds))

    if not ds:
        # level 0 transform (just returns the approximation coefficients)
        return coeffs[0]
    if a is None and not any(ds):
        raise ValueError(
            "At least one coefficient must contain a valid value.")

    coeff_ndims = []
    if a is not None:
        a = np.asarray(a)
        coeff_ndims.append(a.ndim)
    for d in ds:
        coeff_ndims += [v.ndim for k, v in d.items()]

    # test that all coefficients have a matching number of dimensions
    unique_coeff_ndims = np.unique(coeff_ndims)
    if len(unique_coeff_ndims) == 1:
        ndim = unique_coeff_ndims[0]
    else:
        raise ValueError(
            "All coefficients must have a matching number of dimensions")

    if np.isscalar(axes):
        axes = (axes, )
    if axes is None:
        axes = range(ndim)
    else:
        axes = tuple(axes)
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to waverecn must be unique.")
    ndim_transform = len(axes)

    for idx, d in enumerate(ds):
        if a is None and not d:
            continue
        # The following if statement handles the case where the approximation
        # coefficient returned at the previous level may exceed the size of the
        # stored detail coefficients by 1 on any given axis.
        if idx > 0:
            a = _match_coeff_dims(a, d)
        d['a' * ndim_transform] = a
        a = idwtn(d, wavelet, mode, axes)

    return a


def _coeffs_wavedec_to_wavedecn(coeffs):
    """Convert wavedec coefficients to the wavedecn format."""
    if len(coeffs) == 0:
        return coeffs
    coeffs = copy(coeffs)
    for n in range(1, len(coeffs)):
        if coeffs[n] is None:
            continue
        if coeffs[n].ndim != 1:
            raise ValueError("expected a 1D coefficient array")
        coeffs[n] = dict(d=coeffs[n])
    return coeffs


def _coeffs_wavedec2_to_wavedecn(coeffs):
    """Convert wavedec2 coefficients to the wavedecn format."""
    if len(coeffs) == 0:
        return coeffs
    coeffs = copy(coeffs)
    for n in range(1, len(coeffs)):
        if not isinstance(coeffs[n], (tuple, list)) or len(coeffs[n]) != 3:
            raise ValueError("expected a 3-tuple of detail coefficients")
        (da, ad, dd) = coeffs[n]
        if da is None or ad is None or dd is None:
            raise ValueError(
                "Expected numpy arrays of detail coefficients. Setting "
                "coefficients to None is not supported.")
        coeffs[n] = dict(ad=ad, da=da, dd=dd)
    return coeffs


def _determine_coeff_array_shape(coeffs, axes):
    arr_shape = np.asarray(coeffs[0].shape)
    axes = np.asarray(axes)  # axes that were transformed
    ndim_transform = len(axes)
    ncoeffs = coeffs[0].size
    for d in coeffs[1:]:
        arr_shape[axes] += np.asarray(d['d'*ndim_transform].shape)[axes]
        for k, v in d.items():
            ncoeffs += v.size
    arr_shape = tuple(arr_shape.tolist())
    # if the total number of coefficients doesn't equal the size of the array
    # then tight packing is not possible.
    is_tight_packing = (np.prod(arr_shape) == ncoeffs)
    return arr_shape, is_tight_packing


def _prepare_coeffs_axes(coeffs, axes):
    """Helper function to check type of coeffs and axes.

    This code is used by both coeffs_to_array and ravel_coeffs.
    """
    if not isinstance(coeffs, list) or len(coeffs) == 0:
        raise ValueError("input must be a list of coefficients from wavedecn")
    if coeffs[0] is None:
        raise ValueError("coeffs_to_array does not support missing "
                         "coefficients.")
    if not isinstance(coeffs[0], np.ndarray):
        raise ValueError("first list element must be a numpy array")
    ndim = coeffs[0].ndim

    if len(coeffs) > 1:
        # convert wavedec or wavedec2 format coefficients to waverecn format
        if isinstance(coeffs[1], dict):
            pass
        elif isinstance(coeffs[1], np.ndarray):
            coeffs = _coeffs_wavedec_to_wavedecn(coeffs)
        elif isinstance(coeffs[1], (tuple, list)):
            coeffs = _coeffs_wavedec2_to_wavedecn(coeffs)
        else:
            raise ValueError("invalid coefficient list")

    if len(coeffs) == 1:
        # no detail coefficients were found
        return coeffs, axes, ndim, None

    # Determine the number of dimensions that were transformed via key length
    ndim_transform = len(list(coeffs[1].keys())[0])
    if axes is None:
        if ndim_transform < ndim:
            raise ValueError(
                "coeffs corresponds to a DWT performed over only a subset of "
                "the axes.  In this case, axes must be specified.")
        axes = np.arange(ndim)

    if len(axes) != ndim_transform:
        raise ValueError(
            "The length of axes doesn't match the number of dimensions "
            "transformed.")

    return coeffs, axes, ndim, ndim_transform


def coeffs_to_array(coeffs, padding=0, axes=None):
    """
    Arrange a wavelet coefficient list from ``wavedecn`` into a single array.

    Parameters
    ----------

    coeffs : array-like
        Dictionary of wavelet coefficients as returned by pywt.wavedecn
    padding : float or None, optional
        The value to use for the background if the coefficients cannot be
        tightly packed. If None, raise an error if the coefficients cannot be
        tightly packed.
    axes : sequence of ints, optional
        Axes over which the DWT that created ``coeffs`` was performed.  The
        default value of None corresponds to all axes.

    Returns
    -------
    coeff_arr : array-like
        Wavelet transform coefficient array.
    coeff_slices : list
        List of slices corresponding to each coefficient.  As a 2D example,
        ``coeff_arr[coeff_slices[1]['dd']]`` would extract the first level
        detail coefficients from ``coeff_arr``.

    See Also
    --------
    array_to_coeffs : the inverse of coeffs_to_array

    Notes
    -----
    Assume a 2D coefficient dictionary, c, from a two-level transform.

    Then all 2D coefficients will be stacked into a single larger 2D array
    as follows::

        +---------------+---------------+-------------------------------+
        |               |               |                               |
        |     c[0]      |  c[1]['da']   |                               |
        |               |               |                               |
        +---------------+---------------+           c[2]['da']          |
        |               |               |                               |
        | c[1]['ad']    |  c[1]['dd']   |                               |
        |               |               |                               |
        +---------------+---------------+ ------------------------------+
        |                               |                               |
        |                               |                               |
        |                               |                               |
        |          c[2]['ad']           |           c[2]['dd']          |
        |                               |                               |
        |                               |                               |
        |                               |                               |
        +-------------------------------+-------------------------------+

    If the transform was not performed with mode "periodization" or the signal
    length was not a multiple of ``2**level``, coefficients at each subsequent
    scale will not be exactly 1/2 the size of those at the previous level due
    to additional coefficients retained to handle the boundary condition. In
    these cases, the default setting of `padding=0` indicates to pad the
    individual coefficient arrays with 0 as needed so that they can be stacked
    into a single, contiguous array.

    Examples
    --------
    >>> import pywt
    >>> cam = pywt.data.camera()
    >>> coeffs = pywt.wavedecn(cam, wavelet='db2', level=3)
    >>> arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    """

    coeffs, axes, ndim, ndim_transform = _prepare_coeffs_axes(coeffs, axes)

    # initialize with the approximation coefficients.
    a_coeffs = coeffs[0]
    a_shape = a_coeffs.shape

    if len(coeffs) == 1:
        # only a single approximation coefficient array was found
        return a_coeffs, [tuple([slice(None)] * ndim)]

    # determine size of output and if tight packing is possible
    arr_shape, is_tight_packing = _determine_coeff_array_shape(coeffs, axes)

    # preallocate output array
    if padding is None:
        if not is_tight_packing:
            raise ValueError("array coefficients cannot be tightly packed")
        coeff_arr = np.empty(arr_shape, dtype=a_coeffs.dtype)
    else:
        coeff_arr = np.full(arr_shape, padding, dtype=a_coeffs.dtype)

    a_slices = tuple([slice(s) for s in a_shape])
    coeff_arr[a_slices] = a_coeffs

    # initialize list of coefficient slices
    coeff_slices = []
    coeff_slices.append(a_slices)

    # loop over the detail cofficients, adding them to coeff_arr
    ds = coeffs[1:]
    for coeff_dict in ds:
        coeff_slices.append({})  # new dictionary for detail coefficients
        if np.any([d is None for d in coeff_dict.values()]):
            raise ValueError("coeffs_to_array does not support missing "
                             "coefficients.")
        d_shape = coeff_dict['d' * ndim_transform].shape
        for key in coeff_dict.keys():
            d = coeff_dict[key]
            slice_array = [slice(None), ] * ndim
            for i, let in enumerate(key):
                ax_i = axes[i]  # axis corresponding to this transform index
                if let == 'a':
                    slice_array[ax_i] = slice(d.shape[ax_i])
                elif let == 'd':
                    slice_array[ax_i] = slice(a_shape[ax_i],
                                              a_shape[ax_i] + d.shape[ax_i])
                else:
                    raise ValueError("unexpected letter: {}".format(let))
            slice_array = tuple(slice_array)
            coeff_arr[slice_array] = d
            coeff_slices[-1][key] = slice_array
        a_shape = [a_shape[n] + d_shape[n] for n in range(ndim)]
    return coeff_arr, coeff_slices


def array_to_coeffs(arr, coeff_slices, output_format='wavedecn'):
    """
    Convert a combined array of coefficients back to a list compatible with
    ``waverecn``.

    Parameters
    ----------

    arr : array-like
        An array containing all wavelet coefficients.  This should have been
        generated via ``coeffs_to_array``.
    coeff_slices : list of tuples
        List of slices corresponding to each coefficient as obtained from
        ``array_to_coeffs``.
    output_format : {'wavedec', 'wavedec2', 'wavedecn'}
        Make the form of the coefficients compatible with this type of
        multilevel transform.

    Returns
    -------
    coeffs: array-like
        Wavelet transform coefficient array.

    See Also
    --------
    coeffs_to_array : the inverse of array_to_coeffs

    Notes
    -----
    A single large array containing all coefficients will have subsets stored,
    into a ``waverecn`` list, c, as indicated below::

        +---------------+---------------+-------------------------------+
        |               |               |                               |
        |     c[0]      |  c[1]['da']   |                               |
        |               |               |                               |
        +---------------+---------------+           c[2]['da']          |
        |               |               |                               |
        | c[1]['ad']    |  c[1]['dd']   |                               |
        |               |               |                               |
        +---------------+---------------+ ------------------------------+
        |                               |                               |
        |                               |                               |
        |                               |                               |
        |          c[2]['ad']           |           c[2]['dd']          |
        |                               |                               |
        |                               |                               |
        |                               |                               |
        +-------------------------------+-------------------------------+

    Examples
    --------
    >>> import pywt
    >>> from numpy.testing import assert_array_almost_equal
    >>> cam = pywt.data.camera()
    >>> coeffs = pywt.wavedecn(cam, wavelet='db2', level=3)
    >>> arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    >>> coeffs_from_arr = pywt.array_to_coeffs(arr, coeff_slices,
    ...                                        output_format='wavedecn')
    >>> cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db2')
    >>> assert_array_almost_equal(cam, cam_recon)

    """
    arr = np.asarray(arr)
    coeffs = []
    if len(coeff_slices) == 0:
        raise ValueError("empty list of coefficient slices")
    else:
        coeffs.append(arr[coeff_slices[0]])

    # difference coefficients at each level
    for n in range(1, len(coeff_slices)):
        if output_format == 'wavedec':
            d = arr[coeff_slices[n]['d']]
        elif output_format == 'wavedec2':
            d = (arr[coeff_slices[n]['da']],
                 arr[coeff_slices[n]['ad']],
                 arr[coeff_slices[n]['dd']])
        elif output_format == 'wavedecn':
            d = {}
            for k, v in coeff_slices[n].items():
                d[k] = arr[v]
        else:
            raise ValueError(
                "Unrecognized output format: {}".format(output_format))
        coeffs.append(d)
    return coeffs


def wavedecn_shapes(shape, wavelet, mode='symmetric', level=None, axes=None):
    """Subband shapes for a multilevel nD discrete wavelet transform.

    Parameters
    ----------
    shape : sequence of ints
        The shape of the data to be transformed.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the ``dwt_max_level`` function.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Axes may not be repeated. The
        default is None, which means transform all axes
        (``axes = range(data.ndim)``).

    Returns
    -------
    shapes : [cAn, {details_level_n}, ... {details_level_1}] : list
        Coefficients shape list.  Mirrors the output of ``wavedecn``, except
        it contains only the shapes of the coefficient arrays rather than the
        arrays themselves.

    Examples
    --------
    >>> import pywt
    >>> pywt.wavedecn_shapes((64, 32), wavelet='db2', level=3, axes=(0, ))
    [(10, 32), {'d': (10, 32)}, {'d': (18, 32)}, {'d': (33, 32)}]
    """
    axes, axes_shapes, ndim_transform = _prep_axes_wavedecn(shape, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)
    modes = _modes_per_axis(mode, axes)
    dec_lengths = [w.dec_len for w in wavelets]

    level = _check_level(min(axes_shapes), max(dec_lengths), level)

    shapes = []
    for i in range(level):
        detail_keys = [''.join(c) for c in product('ad', repeat=len(axes))]
        new_shapes = {k: list(shape) for k in detail_keys}
        for axis, wav, mode in zip(axes, wavelets, modes):
            s = dwt_coeff_len(shape[axis], filter_len=wav.dec_len, mode=mode)
            for k in detail_keys:
                new_shapes[k][axis] = s
        for k, v in new_shapes.items():
            new_shapes[k] = tuple(v)
        shapes.append(new_shapes)
        shape = new_shapes.pop('a' * ndim_transform)
    shapes.append(shape)
    shapes.reverse()
    return shapes


def wavedecn_size(shapes):
    """Compute the total number of wavedecn coefficients.

    Parameters
    ----------
    shapes : list of coefficient shapes
        A set of coefficient shapes as returned by ``wavedecn_shapes``.
        Alternatively, the user can specify a set of coefficients as returned
        by ``wavedecn``.

    Returns
    -------
    size : int
        The total number of coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> import pywt
    >>> data_shape = (64, 32)
    >>> shapes = pywt.wavedecn_shapes(data_shape, 'db2', mode='periodization')
    >>> pywt.wavedecn_size(shapes)
    2048
    >>> coeffs = pywt.wavedecn(np.ones(data_shape), 'sym4', mode='symmetric')
    >>> pywt.wavedecn_size(coeffs)
    3087
    """
    def _size(x):
        """Size corresponding to ``x`` as either a shape tuple or ndarray."""
        if isinstance(x, np.ndarray):
            return x.size
        else:
            return np.prod(x)
    ncoeffs = _size(shapes[0])
    for d in shapes[1:]:
        for k, v in d.items():
            if v is None:
                raise ValueError(
                    "Setting coefficient arrays to None is not supported.")
            ncoeffs += _size(v)
    return ncoeffs


def dwtn_max_level(shape, wavelet, axes=None):
    """Compute the maximum level of decomposition for n-dimensional data.

    This returns the maximum number of levels of decomposition suitable for use
    with ``wavedec``, ``wavedec2`` or ``wavedecn``.

    Parameters
    ----------
    shape : sequence of ints
        Input data shape.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use. This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Axes may not be repeated.

    Returns
    -------
    level : int
        Maximum level.

    Notes
    -----
    The level returned is the smallest ``dwt_max_level`` over all axes.

    Examples
    --------
    >>> import pywt
    >>> pywt.dwtn_max_level((64, 32), 'db2')
    3
    """
    # Determine the axes and shape for the transform
    axes, axes_shapes, ndim_transform = _prep_axes_wavedecn(shape, axes)

    # initialize a Wavelet object per (transformed) axis
    wavelets = _wavelets_per_axis(wavelet, axes)

    # maximum level of decomposition per axis
    max_levels = [dwt_max_level(n, wav.dec_len)
                  for n, wav in zip(axes_shapes, wavelets)]
    return min(max_levels)


def ravel_coeffs(coeffs, axes=None):
    """Ravel a set of multilevel wavelet coefficients into a single 1D array.

    Parameters
    ----------
    coeffs : array-like
        A list of multilevel wavelet coefficients as returned by
        ``wavedec``, ``wavedec2`` or ``wavedecn``. This function is also
        compatible with the output of ``swt``, ``swt2`` and ``swtn`` if those
        functions were called with ``trim_approx=True``.
    axes : sequence of ints, optional
        Axes over which the DWT that created ``coeffs`` was performed. The
        default value of None corresponds to all axes.

    Returns
    -------
    coeff_arr : array-like
        Wavelet transform coefficient array. All coefficients have been
        concatenated into a single array.
    coeff_slices : list
        List of slices corresponding to each coefficient. As a 2D example,
        ``coeff_arr[coeff_slices[1]['dd']]`` would extract the first level
        detail coefficients from ``coeff_arr``.
    coeff_shapes : list
        List of shapes corresponding to each coefficient. For example, in 2D,
        ``coeff_shapes[1]['dd']`` would contain the original shape of the first
        level detail coefficients array.

    See Also
    --------
    unravel_coeffs : the inverse of ravel_coeffs

    Examples
    --------
    >>> import pywt
    >>> cam = pywt.data.camera()
    >>> coeffs = pywt.wavedecn(cam, wavelet='db2', level=3)
    >>> arr, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)

    """
    coeffs, axes, ndim, ndim_transform = _prepare_coeffs_axes(coeffs, axes)

    # initialize with the approximation coefficients.
    a_coeffs = coeffs[0]
    a_size = a_coeffs.size

    if len(coeffs) == 1:
        # only a single approximation coefficient array was found
        return a_coeffs.ravel(), [slice(a_size), ], [a_coeffs.shape, ]

    # preallocate output array
    arr_size = wavedecn_size(coeffs)
    coeff_arr = np.empty((arr_size, ), dtype=a_coeffs.dtype)

    a_slice = slice(a_size)
    coeff_arr[a_slice] = a_coeffs.ravel()

    # initialize list of coefficient slices
    coeff_slices = []
    coeff_shapes = []
    coeff_slices.append(a_slice)
    coeff_shapes.append(coeffs[0].shape)

    # loop over the detail cofficients, embedding them in coeff_arr
    ds = coeffs[1:]
    offset = a_size
    for coeff_dict in ds:
        # new dictionaries for detail coefficient slices and shapes
        coeff_slices.append({})
        coeff_shapes.append({})
        if np.any([d is None for d in coeff_dict.values()]):
            raise ValueError("coeffs_to_array does not support missing "
                             "coefficients.")
        # sort to make sure key order is consistent across Python versions
        keys = sorted(coeff_dict.keys())
        for key in keys:
            d = coeff_dict[key]
            sl = slice(offset, offset + d.size)
            offset += d.size
            coeff_arr[sl] = d.ravel()
            coeff_slices[-1][key] = sl
            coeff_shapes[-1][key] = d.shape
    return coeff_arr, coeff_slices, coeff_shapes


def unravel_coeffs(arr, coeff_slices, coeff_shapes, output_format='wavedecn'):
    """Unravel a raveled array of multilevel wavelet coefficients.

    Parameters
    ----------
    arr : array-like
        An array containing all wavelet coefficients. This should have been
        generated by applying ``ravel_coeffs`` to the output of ``wavedec``,
        ``wavedec2`` or ``wavedecn`` (or via ``swt``, ``swt2`` or ``swtn``
        with ``trim_approx=True``).
    coeff_slices : list of tuples
        List of slices corresponding to each coefficient as obtained from
        ``ravel_coeffs``.
    coeff_shapes : list of tuples
        List of shapes corresponding to each coefficient as obtained from
        ``ravel_coeffs``.
    output_format : {'wavedec', 'wavedec2', 'wavedecn', 'swt', 'swt2', 'swtn'}, optional
        Make the form of the unraveled coefficients compatible with this type
        of multilevel transform. The default is ``'wavedecn'``.

    Returns
    -------
    coeffs: list
        List of wavelet transform coefficients. The specific format of the list
        elements is determined by ``output_format``.

    See Also
    --------
    ravel_coeffs : the inverse of unravel_coeffs

    Examples
    --------
    >>> import pywt
    >>> from numpy.testing import assert_array_almost_equal
    >>> cam = pywt.data.camera()
    >>> coeffs = pywt.wavedecn(cam, wavelet='db2', level=3)
    >>> arr, coeff_slices, coeff_shapes = pywt.ravel_coeffs(coeffs)
    >>> coeffs_from_arr = pywt.unravel_coeffs(arr, coeff_slices, coeff_shapes,
    ...                                       output_format='wavedecn')
    >>> cam_recon = pywt.waverecn(coeffs_from_arr, wavelet='db2')
    >>> assert_array_almost_equal(cam, cam_recon)

    """
    arr = np.asarray(arr)
    coeffs = []
    if len(coeff_slices) == 0:
        raise ValueError("empty list of coefficient slices")
    elif len(coeff_shapes) == 0:
        raise ValueError("empty list of coefficient shapes")
    elif len(coeff_shapes) != len(coeff_slices):
        raise ValueError("coeff_shapes and coeff_slices have unequal length")
    else:
        coeffs.append(arr[coeff_slices[0]].reshape(coeff_shapes[0]))

    # difference coefficients at each level
    for n in range(1, len(coeff_slices)):
        slice_dict = coeff_slices[n]
        shape_dict = coeff_shapes[n]
        if output_format in ['wavedec', 'swt']:
            d = arr[slice_dict['d']].reshape(shape_dict['d'])
        elif output_format in ['wavedec2', 'swt2']:
            d = (arr[slice_dict['da']].reshape(shape_dict['da']),
                 arr[slice_dict['ad']].reshape(shape_dict['ad']),
                 arr[slice_dict['dd']].reshape(shape_dict['dd']))
        elif output_format in ['wavedecn', 'swtn']:
            d = {}
            for k, v in coeff_slices[n].items():
                d[k] = arr[v].reshape(shape_dict[k])
        else:
            raise ValueError(
                "Unrecognized output format: {}".format(output_format))
        coeffs.append(d)
    return coeffs


def _check_fswavedecn_axes(data, axes):
    """Axes checks common to fswavedecn, fswaverecn."""
    if len(axes) != len(set(axes)):
        raise np.AxisError("The axes passed to fswavedecn must be unique.")
    try:
        [data.shape[ax] for ax in axes]
    except IndexError:
        raise np.AxisError("Axis greater than data dimensions")


class FswavedecnResult(object):
    """Object representing fully separable wavelet transform coefficients.

    Parameters
    ----------
    coeffs : ndarray
        The coefficient array.
    coeff_slices : list
        List of slices corresponding to each detail or approximation
        coefficient array.
    wavelets : list of pywt.DiscreteWavelet objects
        The wavelets used.  Will be a list with length equal to
        ``len(axes)``.
    mode_enums : list of int
        The border modes used.  Will be a list with length equal to
        ``len(axes)``.
    axes : tuple of int
        The set of axes over which the transform was performed.

    """
    def __init__(self, coeffs, coeff_slices, wavelets, mode_enums,
                 axes):
        self._coeffs = coeffs
        self._coeff_slices = coeff_slices
        self._axes = axes
        if not np.all(isinstance(w, Wavelet) for w in wavelets):
            raise ValueError(
                "wavelets must contain pywt.Wavelet objects")
        self._wavelets = wavelets
        if not np.all(isinstance(m, int) for m in mode_enums):
            raise ValueError(
                "mode_enums must be integers")
        self._mode_enums = mode_enums

    @property
    def coeffs(self):
        """ndarray: All coefficients stacked into a single array."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, c):
        if c.shape != self._coeffs.shape:
            raise ValueError("new coefficient array must match the existing "
                             "coefficient shape")
        self._coeffs = c

    @property
    def coeff_slices(self):
        """List: List of coefficient slices."""
        return self._coeff_slices

    @property
    def ndim(self):
        """int: Number of data dimensions."""
        return self.coeffs.ndim

    @property
    def ndim_transform(self):
        """int: Number of axes transformed."""
        return len(self.axes)

    @property
    def axes(self):
        """List of str: The axes the transform was performed along."""
        return self._axes

    @property
    def levels(self):
        """List of int: Levels of decomposition along each transformed axis."""
        return [len(s) - 1 for s in self.coeff_slices]

    @property
    def wavelets(self):
        """List of pywt.DiscreteWavelet: wavelet for each transformed axis."""
        return self._wavelets

    @property
    def wavelet_names(self):
        """List of pywt.DiscreteWavelet: wavelet for each transformed axis."""
        return [w.name for w in self._wavelets]

    @property
    def modes(self):
        """List of str: The border mode used along each transformed axis."""
        names_dict = {getattr(Modes, mode): mode
                      for mode in Modes.modes}
        return [names_dict[m] for m in self._mode_enums]

    def _get_coef_sl(self, levels):
        sl = [slice(None), ] * self.ndim
        for n, (ax, lev) in enumerate(zip(self.axes, levels)):
            sl[ax] = self.coeff_slices[n][lev]
        return tuple(sl)

    @property
    def approx(self):
        """ndarray: The approximation coefficients."""
        sl = self._get_coef_sl((0, )*self.ndim)
        return self._coeffs[sl]

    @approx.setter
    def approx(self, a):
        sl = self._get_coef_sl((0, )*self.ndim)
        if self._coeffs[sl].shape != a.shape:
            raise ValueError(
                "x does not match the shape of the requested coefficient")
        self._coeffs[sl] = a

    def _validate_index(self, levels):
        levels = tuple(levels)

        if len(levels) != len(self.axes):
            raise ValueError(
                "levels must match the number of transformed axes")

        # check that all elements are non-negative integers
        if (not np.all([isinstance(lev, numbers.Number) for lev in levels]) or
                np.any(np.asarray(levels) % 1 > 0) or
                np.any([lev < 0 for lev in levels])):
            raise ValueError("Index must be a tuple of non-negative integers")
        # convert integer-valued floats to int
        levels = tuple([int(lev) for lev in levels])

        # check for out of range levels
        if np.any([lev > maxlev for lev, maxlev in zip(levels, self.levels)]):
            raise ValueError(
                "Specified indices exceed the number of transform levels.")

    def __getitem__(self, levels):
        """Retrieve a coefficient subband.

        Parameters
        ----------
        levels : tuple of int
            The number of degrees of decomposition along each transformed
            axis.
        """
        self._validate_index(levels)
        sl = self._get_coef_sl(levels)
        return self._coeffs[sl]

    def __setitem__(self, levels, x):
        """Assign values to a coefficient subband.

        Parameters
        ----------
        levels : tuple of int
            The number of degrees of decomposition along each transformed
            axis.
        x : ndarray
            The data corresponding to assign. It must match the expected
            shape and dtype of the specified subband.
        """
        self._validate_index(levels)
        sl = self._get_coef_sl(levels)
        current_dtype = self._coeffs[sl].dtype
        if self._coeffs[sl].shape != x.shape:
            raise ValueError(
                "x does not match the shape of the requested coefficient")
        if x.dtype != current_dtype:
            warnings.warn("dtype mismatch:  converting the provided array to"
                          "dtype {}".format(current_dtype))
        self._coeffs[sl] = x

    def detail_keys(self):
        """Return a list of all detail coefficient keys.

        Returns
        -------
        keys : list of str
            List of all detail coefficient keys.
        """
        keys = list(product(*(range(l+1) for l in self.levels)))
        keys.remove((0, )*len(self.axes))
        return sorted(keys)


def fswavedecn(data, wavelet, mode='symmetric', levels=None, axes=None):
    """Fully Separable Wavelet Decomposition.

    This is a variant of the multilevel discrete wavelet transform where all
    levels of decomposition are performed along a single axis prior to moving
    onto the next axis.  Unlike in ``wavedecn``, the number of levels of
    decomposition are not required to be the same along each axis which can be
    a benefit for anisotropic data.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    levels : int or sequence of ints, optional
        Decomposition levels along each axis (must be >= 0). If an integer is
        provided, the same number of levels are used for all axes. If
        ``levels`` is None (default), ``dwt_max_level`` will be used to compute
        the maximum number of levels possible for each axis.
    axes : sequence of ints, optional
        Axes over which to compute the transform. Axes may not be repeated. The
        default is to transform along all axes.

    Returns
    -------
    fswavedecn_result : FswavedecnResult object
        Contains the wavelet coefficients, slice objects to allow obtaining
        the coefficients per detail or approximation level, and more.
        See ``FswavedecnResult`` for details.

    Examples
    --------
    >>> from pywt import fswavedecn
    >>> fs_result = fswavedecn(np.ones((32, 32)), 'sym2', levels=(1, 3))
    >>> print(fs_result.detail_keys())
    [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    >>> approx_coeffs = fs_result.approx
    >>> detail_1_2 = fs_result[(1, 2)]


    Notes
    -----
    This transformation has been variously referred to as the (fully) separable
    wavelet transform (e.g. refs [1]_, [3]_), the tensor-product wavelet
    ([2]_) or the hyperbolic wavelet transform ([4]_).  It is well suited to
    data with anisotropic smoothness.

    In [2]_ it was demonstrated that fully separable transform performs at
    least as well as the DWT for image compression.  Computation time is a
    factor 2 larger than that for the DWT.

    See Also
    --------
    fswaverecn : inverse of fswavedecn

    References
    ----------
    .. [1] PH Westerink. Subband Coding of Images. Ph.D. dissertation, Dept.
       Elect. Eng., Inf. Theory Group, Delft Univ. Technol., Delft, The
       Netherlands, 1989.  (see Section 2.3)
       http://resolver.tudelft.nl/uuid:a4d195c3-1f89-4d66-913d-db9af0969509

    .. [2] CP Rosiene and TQ Nguyen. Tensor-product wavelet vs. Mallat
       decomposition: A comparative analysis, in Proc. IEEE Int. Symp.
       Circuits and Systems, Orlando, FL, Jun. 1999, pp. 431-434.

    .. [3] V Velisavljevic, B Beferull-Lozano, M Vetterli and PL Dragotti.
       Directionlets: Anisotropic Multidirectional Representation With
       Separable Filtering. IEEE Transactions on Image Processing, Vol. 15,
       No. 7, July 2006.

    .. [4] RA DeVore, SV Konyagin and VN Temlyakov. "Hyperbolic wavelet
       approximation," Constr. Approx. 14 (1998), 1-26.
    """
    data = np.asarray(data)
    if axes is None:
        axes = tuple(np.arange(data.ndim))
    _check_fswavedecn_axes(data, axes)

    if levels is None or np.isscalar(levels):
        levels = [levels, ] * len(axes)
    if len(levels) != len(axes):
        raise ValueError("levels must match the length of the axes list")

    modes = _modes_per_axis(mode, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    coeff_slices = [slice(None), ] * len(axes)
    coeffs_arr = data
    for ax_count, (ax, lev, wav, mode) in enumerate(
            zip(axes, levels, wavelets, modes)):
        coeffs = wavedec(coeffs_arr, wav, mode=mode, level=lev, axis=ax)

        # Slice objects for accessing coefficient subsets.
        # These can be used to access specific detail coefficient arrays
        # (e.g. as needed for inverse transformation via fswaverecn).
        c_shapes = [c.shape[ax] for c in coeffs]
        c_offsets = np.cumsum([0, ] + c_shapes)
        coeff_slices[ax_count] = [
            slice(c_offsets[d], c_offsets[d+1]) for d in range(len(c_shapes))]

        # stack the coefficients from all levels into a single array
        coeffs_arr = np.concatenate(coeffs, axis=ax)

    return FswavedecnResult(coeffs_arr, coeff_slices, wavelets, modes, axes)


def fswaverecn(fswavedecn_result):
    """Fully Separable Inverse Wavelet Reconstruction.

    Parameters
    ----------
    fswavedecn_result : FswavedecnResult object
        FswavedecnResult object from ``fswavedecn``.

    Returns
    -------
    reconstructed : ndarray
        Array of reconstructed data.

    Notes
    -----
    This transformation has been variously referred to as the (fully) separable
    wavelet transform (e.g. refs [1]_, [3]_), the tensor-product wavelet
    ([2]_) or the hyperbolic wavelet transform ([4]_).  It is well suited to
    data with anisotropic smoothness.

    In [2]_ it was demonstrated that the fully separable transform performs at
    least as well as the DWT for image compression. Computation time is a
    factor 2 larger than that for the DWT.

    See Also
    --------
    fswavedecn : inverse of fswaverecn

    References
    ----------
    .. [1] PH Westerink. Subband Coding of Images. Ph.D. dissertation, Dept.
       Elect. Eng., Inf. Theory Group, Delft Univ. Technol., Delft, The
       Netherlands, 1989.  (see Section 2.3)
       http://resolver.tudelft.nl/uuid:a4d195c3-1f89-4d66-913d-db9af0969509

    .. [2] CP Rosiene and TQ Nguyen. Tensor-product wavelet vs. Mallat
       decomposition: A comparative analysis, in Proc. IEEE Int. Symp.
       Circuits and Systems, Orlando, FL, Jun. 1999, pp. 431-434.

    .. [3] V Velisavljevic, B Beferull-Lozano, M Vetterli and PL Dragotti.
       Directionlets: Anisotropic Multidirectional Representation With
       Separable Filtering. IEEE Transactions on Image Processing, Vol. 15,
       No. 7, July 2006.

    .. [4] RA DeVore, SV Konyagin and VN Temlyakov. "Hyperbolic wavelet
       approximation," Constr. Approx. 14 (1998), 1-26.
    """
    coeffs_arr = fswavedecn_result.coeffs
    coeff_slices = fswavedecn_result.coeff_slices
    axes = fswavedecn_result.axes
    modes = fswavedecn_result.modes
    wavelets = fswavedecn_result.wavelets

    _check_fswavedecn_axes(coeffs_arr, axes)
    if len(axes) != len(coeff_slices):
        raise ValueError("dimension mismatch")

    arr = coeffs_arr
    csl = [slice(None), ] * arr.ndim
    # for ax_count, (ax, wav, mode) in reversed(
    #         list(enumerate(zip(axes, wavelets, modes)))):
    for ax_count, (ax, wav, mode) in enumerate(zip(axes, wavelets, modes)):
        coeffs = []
        for sl in coeff_slices[ax_count]:
            csl[ax] = sl
            coeffs.append(arr[tuple(csl)])
        csl[ax] = slice(None)
        arr = waverec(coeffs, wav, mode=mode, axis=ax)
    return arr
