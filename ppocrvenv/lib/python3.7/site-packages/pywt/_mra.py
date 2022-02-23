from functools import partial, reduce

import numpy as np

from ._multilevel import (_prep_axes_wavedecn, wavedec, wavedec2, wavedecn,
                          waverec, waverec2, waverecn)
from ._swt import iswt, iswt2, iswtn, swt, swt2, swt_max_level, swtn
from ._utils import _modes_per_axis, _wavelets_per_axis

__all__ = ["mra", "mra2", "mran", "imra", "imra2", "imran"]


def mra(data, wavelet, level=None, axis=-1, transform='swt',
        mode='periodization'):
    """Forward 1D multiresolution analysis.

    It is a projection onto the wavelet subspaces.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the `dwt_max_level` function.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the last axis is
        used. Currently only available when ``transform='dwt'``.
    transform : {'dwt', 'swt'}
        Whether to use the DWT or SWT for the transforms.
    mode : str, optional
        Signal extension mode, see `Modes` (default: 'symmetric'). This option
        is only used when transform='dwt'.

    Returns
    -------
    [cAn, {details_level_n}, ... {details_level_1}] : list
        For more information, see the detailed description in `wavedec`

    See Also
    --------
    imra, swt

    Notes
    -----
    This is sometimes referred to as an additive decomposition because the
    inverse transform (``imra``) is just the sum of the coefficient arrays
    [1]_. The decomposition using ``transform='dwt'`` corresponds to section
    2.2 while that using an undecimated transform (``transform='swt'``) is
    described in section 3.2 and appendix A.

    This transform does not share the variance partition property of ``swt``
    with `norm=True`. It does however, result in coefficients that are
    temporally aligned regardless of the symmetry of the wavelet used.

    The redundancy of this transform is ``(level + 1)``.

    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551

    """
    if transform == 'swt':
        if mode != 'periodization':
            raise ValueError(
                "transform swt only supports mode='periodization'")
        kwargs = dict(wavelet=wavelet, axis=axis, norm=True)
        forward = partial(swt, level=level, trim_approx=True, **kwargs)
        inverse = partial(iswt, **kwargs)
        is_swt = True
    elif transform == 'dwt':
        kwargs = dict(wavelet=wavelet, mode=mode, axis=axis)
        forward = partial(wavedec, level=level, **kwargs)
        inverse = partial(waverec, **kwargs)
        is_swt = False
    else:
        raise ValueError("unrecognized transform: {}".format(transform))

    wav_coeffs = forward(data)

    mra_coeffs = []
    nc = len(wav_coeffs)

    if is_swt:
        # replicate same zeros array to save memory
        z = np.zeros_like(wav_coeffs[0])
        tmp = [z, ] * nc
    else:
        # zero arrays have variable size in DWT case
        tmp = [np.zeros_like(c) for c in wav_coeffs]

    for j in range(nc):
        # tmp has arrays of zeros except for the jth entry
        tmp[j] = wav_coeffs[j]

        # reconstruct
        rec = inverse(tmp)
        if rec.shape != data.shape:
            # trim any excess coefficients
            rec = rec[tuple([slice(sz) for sz in data.shape])]
        mra_coeffs.append(rec)

        # restore zeros
        if is_swt:
            tmp[j] = z
        else:
            tmp[j] = np.zeros_like(tmp[j])
    return mra_coeffs


def imra(mra_coeffs):
    """Inverse 1D multiresolution analysis via summation.

    Parameters
    ----------
    mra_coeffs : list of ndarray
        Multiresolution analysis coefficients as returned by `mra`.

    Returns
    -------
    rec : ndarray
        The reconstructed signal.

    See Also
    --------
    mra

    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    return reduce(lambda x, y: x + y, mra_coeffs)


def mra2(data, wavelet, level=None, axes=(-2, -1), transform='swt2',
         mode='periodization'):
    """Forward 2D multiresolution analysis.

    It is a projection onto wavelet subspaces.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in `axes`.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the `dwt_max_level` function.
    axes : 2-tuple of ints, optional
        Axes over which to compute the DWT. Repeated elements are not allowed.
        Currently only available when ``transform='dwt2'``.
    transform : {'dwt2', 'swt2'}
        Whether to use the DWT or SWT for the transforms.
    mode : str or 2-tuple of str, optional
        Signal extension mode, see `Modes` (default: 'symmetric'). This option
        is only used when transform='dwt2'.

    Returns
    -------
    coeffs : list
        For more information, see the detailed description in `wavedec2`

    Notes
    -----
    This is sometimes referred to as an additive decomposition because the
    inverse transform (``imra2``) is just the sum of the coefficient arrays
    [1]_. The decomposition using ``transform='dwt'`` corresponds to section
    2.2 while that using an undecimated transform (``transform='swt'``) is
    described in section 3.2 and appendix A.

    This transform does not share the variance partition property of ``swt2``
    with `norm=True`. It does however, result in coefficients that are
    temporally aligned regardless of the symmetry of the wavelet used.

    The redundancy of this transform is ``3 * level + 1``.

    See Also
    --------
    imra2, swt2

    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    if transform == 'swt2':
        if mode != 'periodization':
            raise ValueError(
                "transform swt only supports mode='periodization'")
        if level is None:
            level = min(swt_max_level(s) for s in data.shape)
        kwargs = dict(wavelet=wavelet, axes=axes, norm=True)
        forward = partial(swt2, level=level, trim_approx=True, **kwargs)
        inverse = partial(iswt2, **kwargs)
    elif transform == 'dwt2':
        kwargs = dict(wavelet=wavelet, mode=mode, axes=axes)
        forward = partial(wavedec2, level=level, **kwargs)
        inverse = partial(waverec2, **kwargs)
    else:
        raise ValueError("unrecognized transform: {}".format(transform))

    wav_coeffs = forward(data)

    mra_coeffs = []
    nc = len(wav_coeffs)
    z = np.zeros_like(wav_coeffs[0])
    tmp = [z]
    for j in range(1, nc):
        tmp.append([np.zeros_like(c) for c in wav_coeffs[j]])

    # tmp has arrays of zeros except for the jth entry
    tmp[0] = wav_coeffs[0]
    # reconstruct
    rec = inverse(tmp)
    if rec.shape != data.shape:
        # trim any excess coefficients
        rec = rec[tuple([slice(sz) for sz in data.shape])]
    mra_coeffs.append(rec)
    # restore zeros
    tmp[0] = z

    for j in range(1, nc):
        dcoeffs = []
        for n in range(3):
            # tmp has arrays of zeros except for the jth entry
            z = tmp[j][n]
            tmp[j][n] = wav_coeffs[j][n]
            # reconstruct
            rec = inverse(tmp)
            if rec.shape != data.shape:
                # trim any excess coefficients
                rec = rec[tuple([slice(sz) for sz in data.shape])]
            dcoeffs.append(rec)
            # restore zeros
            tmp[j][n] = z
        mra_coeffs.append(tuple(dcoeffs))
    return mra_coeffs


def imra2(mra_coeffs):
    """Inverse 2D multiresolution analysis via summation.

    Parameters
    ----------
    mra_coeffs : list
        Multiresolution analysis coefficients as returned by `mra2`.

    Returns
    -------
    rec : ndarray
        The reconstructed signal.

    See Also
    --------
    mra2

    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    rec = mra_coeffs[0]
    for j in range(1, len(mra_coeffs)):
        for n in range(3):
            rec += mra_coeffs[j][n]
    return rec


def mran(data, wavelet, level=None, axes=None, transform='swtn',
         mode='periodization'):
    """Forward nD multiresolution analysis.

    It is a projection onto the wavelet subspaces.

    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use. This can also be a tuple containing a wavelet to
        apply along each axis in `axes`.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the `dwt_max_level` function.
    axes : tuple of ints, optional
        Axes over which to compute the DWT. Repeated elements are not allowed.
    transform : {'dwtn', 'swtn'}
        Whether to use the DWT or SWT for the transforms.
    mode : str or tuple of str, optional
        Signal extension mode, see `Modes` (default: 'symmetric'). This option
        is only used when transform='dwtn'.

    Returns
    -------
    coeffs : list
        For more information, see the detailed description in `wavedecn`.

    See Also
    --------
    imran, swtn

    Notes
    -----
    This is sometimes referred to as an additive decomposition because the
    inverse transform (``imran``) is just the sum of the coefficient arrays
    [1]_. The decomposition using ``transform='dwt'`` corresponds to section
    2.2 while that using an undecimated transform (``transform='swt'``) is
    described in section 3.2 and appendix A.

    This transform does not share the variance partition property of ``swtn``
    with `norm=True`. It does however, result in coefficients that are
    temporally aligned regardless of the symmetry of the wavelet used.

    The redundancy of this transform is ``(2**n - 1) * level + 1`` where ``n``
    corresponds to the number of axes transformed.

    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    axes, axes_shapes, ndim_transform = _prep_axes_wavedecn(data.shape, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    if transform == 'swtn':
        if mode != 'periodization':
            raise ValueError(
                "transform swt only supports mode='periodization'")
        if level is None:
            level = min(swt_max_level(s) for s in data.shape)
        kwargs = dict(wavelet=wavelets, axes=axes, norm=True)
        forward = partial(swtn, level=level, trim_approx=True, **kwargs)
        inverse = partial(iswtn, **kwargs)
    elif transform == 'dwtn':
        modes = _modes_per_axis(mode, axes)
        kwargs = dict(wavelet=wavelets, mode=modes, axes=axes)
        forward = partial(wavedecn, level=level, **kwargs)
        inverse = partial(waverecn, **kwargs)
    else:
        raise ValueError("unrecognized transform: {}".format(transform))

    wav_coeffs = forward(data)

    mra_coeffs = []
    nc = len(wav_coeffs)
    z = np.zeros_like(wav_coeffs[0])
    tmp = [z]
    for j in range(1, nc):
        tmp.append({k: np.zeros_like(v) for k, v in wav_coeffs[j].items()})

    # tmp has arrays of zeros except for the jth entry
    tmp[0] = wav_coeffs[0]
    # reconstruct
    rec = inverse(tmp)
    if rec.shape != data.shape:
        # trim any excess coefficients
        rec = rec[tuple([slice(sz) for sz in data.shape])]
    mra_coeffs.append(rec)
    # restore zeros
    tmp[0] = z

    for j in range(1, nc):
        dcoeffs = {}
        dkeys = list(wav_coeffs[j].keys())
        for k in dkeys:
            # tmp has arrays of zeros except for the jth entry
            z = tmp[j][k]
            tmp[j][k] = wav_coeffs[j][k]
            # tmp[j]['a' * len(k)] = z
            # reconstruct
            rec = inverse(tmp)
            if rec.shape != data.shape:
                # trim any excess coefficients
                rec = rec[tuple([slice(sz) for sz in data.shape])]
            dcoeffs[k] = rec
            # restore zeros
            tmp[j][k] = z
            # tmp[j].pop('a' * len(k))
        mra_coeffs.append(dcoeffs)
    return mra_coeffs


def imran(mra_coeffs):
    """Inverse nD multiresolution analysis via summation.

    Parameters
    ----------
    mra_coeffs : list
        Multiresolution analysis coefficients as returned by `mra2`.

    Returns
    -------
    rec : ndarray
        The reconstructed signal.

    See Also
    --------
    mran

    References
    ----------
    .. [1] Donald B. Percival and Harold O. Mofjeld. Analysis of Subtidal
        Coastal Sea Level Fluctuations Using Wavelets. Journal of the American
        Statistical Association Vol. 92, No. 439 (Sep., 1997), pp. 868-880.
        https://doi.org/10.2307/2965551
    """
    rec = mra_coeffs[0]
    for j in range(1, len(mra_coeffs)):
        for k, v in mra_coeffs[j].items():
            rec += v
    return rec
