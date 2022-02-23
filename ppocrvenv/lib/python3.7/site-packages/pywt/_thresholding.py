# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2016 The PyWavelets Developers
#                         <https://github.com/PyWavelets/pywt>
# See COPYING for license details.

"""
The thresholding helper module implements the most popular signal thresholding
functions.
"""

from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = ['threshold', 'threshold_firm']


def soft(data, value, substitute=0):
    data = np.asarray(data)
    magnitude = np.absolute(data)

    with np.errstate(divide='ignore'):
        # divide by zero okay as np.inf values get clipped, so ignore warning.
        thresholded = (1 - value/magnitude)
        thresholded.clip(min=0, max=None, out=thresholded)
        thresholded = data * thresholded

    if substitute == 0:
        return thresholded
    else:
        cond = np.less(magnitude, value)
        return np.where(cond, substitute, thresholded)


def nn_garrote(data, value, substitute=0):
    """Non-negative Garrote."""
    data = np.asarray(data)
    magnitude = np.absolute(data)

    with np.errstate(divide='ignore'):
        # divide by zero okay as np.inf values get clipped, so ignore warning.
        thresholded = (1 - value**2/magnitude**2)
        thresholded.clip(min=0, max=None, out=thresholded)
        thresholded = data * thresholded

    if substitute == 0:
        return thresholded
    else:
        cond = np.less(magnitude, value)
        return np.where(cond, substitute, thresholded)


def hard(data, value, substitute=0):
    data = np.asarray(data)
    cond = np.less(np.absolute(data), value)
    return np.where(cond, substitute, data)


def greater(data, value, substitute=0):
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ValueError("greater thresholding only supports real data")
    return np.where(np.less(data, value), substitute, data)


def less(data, value, substitute=0):
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ValueError("less thresholding only supports real data")
    return np.where(np.greater(data, value), substitute, data)


thresholding_options = {'soft': soft,
                        'hard': hard,
                        'greater': greater,
                        'less': less,
                        'garrote': nn_garrote,
                        # misspelled garrote for backwards compatibility
                        'garotte': nn_garrote,
                        }


def threshold(data, value, mode='soft', substitute=0):
    """
    Thresholds the input data depending on the mode argument.

    In ``soft`` thresholding [1]_, data values with absolute value less than
    `param` are replaced with `substitute`. Data values with absolute value
    greater or equal to the thresholding value are shrunk toward zero
    by `value`.  In other words, the new value is
    ``data/np.abs(data) * np.maximum(np.abs(data) - value, 0)``.

    In ``hard`` thresholding, the data values where their absolute value is
    less than the value param are replaced with `substitute`. Data values with
    absolute value greater or equal to the thresholding value stay untouched.

    ``garrote`` corresponds to the Non-negative garrote threshold [2]_, [3]_.
    It is intermediate between ``hard`` and ``soft`` thresholding.  It behaves
    like soft thresholding for small data values and approaches hard
    thresholding for large data values.

    In ``greater`` thresholding, the data is replaced with `substitute` where
    data is below the thresholding value. Greater data values pass untouched.

    In ``less`` thresholding, the data is replaced with `substitute` where data
    is above the thresholding value. Lesser data values pass untouched.

    Both ``hard`` and ``soft`` thresholding also support complex-valued data.

    Parameters
    ----------
    data : array_like
        Numeric data.
    value : scalar
        Thresholding value.
    mode : {'soft', 'hard', 'garrote', 'greater', 'less'}
        Decides the type of thresholding to be applied on input data. Default
        is 'soft'.
    substitute : float, optional
        Substitute value (default: 0).

    Returns
    -------
    output : array
        Thresholded array.

    See Also
    --------
    threshold_firm

    References
    ----------
    .. [1] D.L. Donoho and I.M. Johnstone. Ideal Spatial Adaptation via
        Wavelet Shrinkage. Biometrika. Vol. 81, No. 3, pp.425-455, 1994.
        DOI:10.1093/biomet/81.3.425
    .. [2] L. Breiman. Better Subset Regression Using the Nonnegative Garrote.
        Technometrics, Vol. 37, pp. 373-384, 1995.
        DOI:10.2307/1269730
    .. [3] H-Y. Gao.  Wavelet Shrinkage Denoising Using the Non-Negative
        Garrote.  Journal of Computational and Graphical Statistics Vol. 7,
        No. 4, pp.469-488. 1998.
        DOI:10.1080/10618600.1998.10474789

    Examples
    --------
    >>> import numpy as np
    >>> import pywt
    >>> data = np.linspace(1, 4, 7)
    >>> data
    array([ 1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ])
    >>> pywt.threshold(data, 2, 'soft')
    array([ 0. ,  0. ,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> pywt.threshold(data, 2, 'hard')
    array([ 0. ,  0. ,  2. ,  2.5,  3. ,  3.5,  4. ])
    >>> pywt.threshold(data, 2, 'garrote')
    array([ 0.        ,  0.        ,  0.        ,  0.9       ,  1.66666667,
            2.35714286,  3.        ])
    >>> pywt.threshold(data, 2, 'greater')
    array([ 0. ,  0. ,  2. ,  2.5,  3. ,  3.5,  4. ])
    >>> pywt.threshold(data, 2, 'less')
    array([ 1. ,  1.5,  2. ,  0. ,  0. ,  0. ,  0. ])

    """

    try:
        return thresholding_options[mode](data, value, substitute)
    except KeyError:
        # Make sure error is always identical by sorting keys
        keys = ("'{0}'".format(key) for key in
                sorted(thresholding_options.keys()))
        raise ValueError("The mode parameter only takes values from: {0}."
                         .format(', '.join(keys)))


def threshold_firm(data, value_low, value_high):
    """Firm threshold.

    The approach is intermediate between soft and hard thresholding [1]_. It
    behaves the same as soft-thresholding for values below `value_low` and
    the same as hard-thresholding for values above `thresh_high`.  For
    intermediate values, the thresholded value is in between that corresponding
    to soft or hard thresholding.

    Parameters
    ----------
    data : array-like
        The data to threshold.  This can be either real or complex-valued.
    value_low : float
        Any values smaller then `value_low` will be set to zero.
    value_high : float
        Any values larger than `value_high` will not be modified.

    Notes
    -----
    This thresholding technique is also known as semi-soft thresholding [2]_.

    For each value, `x`, in `data`. This function computes::

        if np.abs(x) <= value_low:
            return 0
        elif np.abs(x) > value_high:
            return x
        elif value_low < np.abs(x) and np.abs(x) <= value_high:
            return x * value_high * (1 - value_low/x)/(value_high - value_low)

    ``firm`` is a continuous function (like soft thresholding), but is
    unbiased for large values (like hard thresholding).

    If ``value_high == value_low`` this function becomes hard-thresholding.
    If ``value_high`` is infinity, this function becomes soft-thresholding.

    Returns
    -------
    val_new : array-like
        The values after firm thresholding at the specified thresholds.

    See Also
    --------
    threshold

    References
    ----------
    .. [1] H.-Y. Gao and A.G. Bruce. Waveshrink with firm shrinkage.
        Statistica Sinica, Vol. 7, pp. 855-874, 1997.
    .. [2] A. Bruce and H-Y. Gao. WaveShrink: Shrinkage Functions and
        Thresholds. Proc. SPIE 2569, Wavelet Applications in Signal and
        Image Processing III, 1995.
        DOI:10.1117/12.217582
    """

    if value_low < 0:
        raise ValueError("value_low must be non-negative.")

    if value_high < value_low:
        raise ValueError(
            "value_high must be greater than or equal to value_low.")

    data = np.asarray(data)
    magnitude = np.absolute(data)
    with np.errstate(divide='ignore'):
        # divide by zero okay as np.inf values get clipped, so ignore warning.
        vdiff = value_high - value_low
        thresholded = value_high * (1 - value_low/magnitude) / vdiff
        thresholded.clip(min=0, max=None, out=thresholded)
        thresholded = data * thresholded

    # restore hard-thresholding behavior for values > value_high
    large_vals = np.where(magnitude > value_high)
    if np.any(large_vals[0]):
        thresholded[large_vals] = data[large_vals]
    return thresholded
