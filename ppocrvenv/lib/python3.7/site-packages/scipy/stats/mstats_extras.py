"""
Additional statistics functions with support for masked arrays.

"""

# Original author (2007): Pierre GF Gerard-Marchant


__all__ = ['compare_medians_ms',
           'hdquantiles', 'hdmedian', 'hdquantiles_sd',
           'idealfourths',
           'median_cihs','mjci','mquantiles_cimj',
           'rsh',
           'trimmed_mean_ci',]


import numpy as np
from numpy import float_, int_, ndarray

import numpy.ma as ma
from numpy.ma import MaskedArray

from . import mstats_basic as mstats

from scipy.stats.distributions import norm, beta, t, binom


def hdquantiles(data, prob=list([.25,.5,.75]), axis=None, var=False,):
    """
    Computes quantile estimates with the Harrell-Davis method.

    The quantile estimates are calculated as a weighted linear combination
    of order statistics.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdquantiles : MaskedArray
        A (p,) array of quantiles (if `var` is False), or a (2,p) array of
        quantiles and variances (if `var` is True), where ``p`` is the
        number of quantiles.

    See Also
    --------
    hdquantiles_sd

    """
    def _hd_1D(data,prob,var):
        "Computes the HD quantiles for a 1D array. Returns nan for invalid data."
        xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
        # Don't use length here, in case we have a numpy scalar
        n = xsorted.size

        hd = np.empty((2,len(prob)), float_)
        if n < 2:
            hd.flat = np.nan
            if var:
                return hd
            return hd[0]

        v = np.arange(n+1) / float(n)
        betacdf = beta.cdf
        for (i,p) in enumerate(prob):
            _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
            w = _w[1:] - _w[:-1]
            hd_mean = np.dot(w, xsorted)
            hd[0,i] = hd_mean
            #
            hd[1,i] = np.dot(w, (xsorted-hd_mean)**2)
            #
        hd[0, prob == 0] = xsorted[0]
        hd[0, prob == 1] = xsorted[-1]
        if var:
            hd[1, prob == 0] = hd[1, prob == 1] = np.nan
            return hd
        return hd[0]
    # Initialization & checks
    data = ma.array(data, copy=False, dtype=float_)
    p = np.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None) or (data.ndim == 1):
        result = _hd_1D(data, p, var)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hd_1D, axis, data, p, var)

    return ma.fix_invalid(result, copy=False)


def hdmedian(data, axis=-1, var=False):
    """
    Returns the Harrell-Davis estimate of the median along the given axis.

    Parameters
    ----------
    data : ndarray
        Data array.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdmedian : MaskedArray
        The median values.  If ``var=True``, the variance is returned inside
        the masked array.  E.g. for a 1-D array the shape change from (1,) to
        (2,).

    """
    result = hdquantiles(data,[0.5], axis=axis, var=var)
    return result.squeeze()


def hdquantiles_sd(data, prob=list([.25,.5,.75]), axis=None):
    """
    The standard error of the Harrell-Davis quantile estimates by jackknife.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    hdquantiles_sd : MaskedArray
        Standard error of the Harrell-Davis quantile estimates.

    See Also
    --------
    hdquantiles

    """
    def _hdsd_1D(data, prob):
        "Computes the std error for 1D arrays."
        xsorted = np.sort(data.compressed())
        n = len(xsorted)

        hdsd = np.empty(len(prob), float_)
        if n < 2:
            hdsd.flat = np.nan

        vv = np.arange(n) / float(n-1)
        betacdf = beta.cdf
        
        for (i,p) in enumerate(prob):
            _w = betacdf(vv, (n+1)*p, (n+1)*(1-p))
            w = _w[1:] - _w[:-1]
            mx_ = np.fromiter([w[:k] @ xsorted[:k] + w[k:] @ xsorted[k+1:]
                               for k in range(n)], dtype=float_)
            mx_var = np.array(mx_.var(), copy=False, ndmin=1) * n / float(n-1)
            hdsd[i] = float(n-1) * np.sqrt(np.diag(mx_var).diagonal() / float(n))
        return hdsd

    # Initialization & checks
    data = ma.array(data, copy=False, dtype=float_)
    p = np.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _hdsd_1D(data, p)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hdsd_1D, axis, data, p)

    return ma.fix_invalid(result, copy=False).ravel()


def trimmed_mean_ci(data, limits=(0.2,0.2), inclusive=(True,True),
                    alpha=0.05, axis=None):
    """
    Selected confidence interval of the trimmed mean along the given axis.

    Parameters
    ----------
    data : array_like
        Input data.
    limits : {None, tuple}, optional
        None or a two item tuple.
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1. If ``n``
        is the number of unmasked data before trimming, then
        (``n * limits[0]``)th smallest data and (``n * limits[1]``)th
        largest data are masked.  The total number of unmasked data after
        trimming is ``n * (1. - sum(limits))``.
        The value of one limit can be set to None to indicate an open interval.

        Defaults to (0.2, 0.2).
    inclusive : (2,) tuple of boolean, optional
        If relative==False, tuple indicating whether values exactly equal to
        the absolute limits are allowed.
        If relative==True, tuple indicating whether the number of data being
        masked on each side should be rounded (True) or truncated (False).

        Defaults to (True, True).
    alpha : float, optional
        Confidence level of the intervals.

        Defaults to 0.05.
    axis : int, optional
        Axis along which to cut. If None, uses a flattened version of `data`.

        Defaults to None.

    Returns
    -------
    trimmed_mean_ci : (2,) ndarray
        The lower and upper confidence intervals of the trimmed data.

    """
    data = ma.array(data, copy=False)
    trimmed = mstats.trimr(data, limits=limits, inclusive=inclusive, axis=axis)
    tmean = trimmed.mean(axis)
    tstde = mstats.trimmed_stde(data,limits=limits,inclusive=inclusive,axis=axis)
    df = trimmed.count(axis) - 1
    tppf = t.ppf(1-alpha/2.,df)
    return np.array((tmean - tppf*tstde, tmean+tppf*tstde))


def mjci(data, prob=[0.25,0.5,0.75], axis=None):
    """
    Returns the Maritz-Jarrett estimators of the standard error of selected
    experimental quantiles of the data.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    """
    def _mjci_1D(data, p):
        data = np.sort(data.compressed())
        n = data.size
        prob = (np.array(p) * n + 0.5).astype(int_)
        betacdf = beta.cdf

        mj = np.empty(len(prob), float_)
        x = np.arange(1,n+1, dtype=float_) / n
        y = x - 1./n
        for (i,m) in enumerate(prob):
            W = betacdf(x,m-1,n-m) - betacdf(y,m-1,n-m)
            C1 = np.dot(W,data)
            C2 = np.dot(W,data**2)
            mj[i] = np.sqrt(C2 - C1**2)
        return mj

    data = ma.array(data, copy=False)
    if data.ndim > 2:
        raise ValueError("Array 'data' must be at most two dimensional, "
                         "but got data.ndim = %d" % data.ndim)

    p = np.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _mjci_1D(data, p)
    else:
        return ma.apply_along_axis(_mjci_1D, axis, data, p)


def mquantiles_cimj(data, prob=[0.25,0.50,0.75], alpha=0.05, axis=None):
    """
    Computes the alpha confidence interval for the selected quantiles of the
    data, with Maritz-Jarrett estimators.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles.
        If None, use a flattened array.

    Returns
    -------
    ci_lower : ndarray
        The lower boundaries of the confidence interval.  Of the same length as
        `prob`.
    ci_upper : ndarray
        The upper boundaries of the confidence interval.  Of the same length as
        `prob`.

    """
    alpha = min(alpha, 1 - alpha)
    z = norm.ppf(1 - alpha/2.)
    xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
    smj = mjci(data, prob, axis=axis)
    return (xq - z * smj, xq + z * smj)


def median_cihs(data, alpha=0.05, axis=None):
    """
    Computes the alpha-level confidence interval for the median of the data.

    Uses the Hettmasperger-Sheather method.

    Parameters
    ----------
    data : array_like
        Input data. Masked values are discarded. The input should be 1D only,
        or `axis` should be set to None.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    median_cihs
        Alpha level confidence interval.

    """
    def _cihs_1D(data, alpha):
        data = np.sort(data.compressed())
        n = len(data)
        alpha = min(alpha, 1-alpha)
        k = int(binom._ppf(alpha/2., n, 0.5))
        gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        if gk < 1-alpha:
            k -= 1
            gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        gkk = binom.cdf(n-k-1,n,0.5) - binom.cdf(k,n,0.5)
        I = (gk - 1 + alpha)/(gk - gkk)
        lambd = (n-k) * I / float(k + (n-2*k)*I)
        lims = (lambd*data[k] + (1-lambd)*data[k-1],
                lambd*data[n-k-1] + (1-lambd)*data[n-k])
        return lims
    data = ma.array(data, copy=False)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _cihs_1D(data, alpha)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_cihs_1D, axis, data, alpha)

    return result


def compare_medians_ms(group_1, group_2, axis=None):
    """
    Compares the medians from two independent groups along the given axis.

    The comparison is performed using the McKean-Schrader estimate of the
    standard error of the medians.

    Parameters
    ----------
    group_1 : array_like
        First dataset.  Has to be of size >=7.
    group_2 : array_like
        Second dataset.  Has to be of size >=7.
    axis : int, optional
        Axis along which the medians are estimated. If None, the arrays are
        flattened.  If `axis` is not None, then `group_1` and `group_2`
        should have the same shape.

    Returns
    -------
    compare_medians_ms : {float, ndarray}
        If `axis` is None, then returns a float, otherwise returns a 1-D
        ndarray of floats with a length equal to the length of `group_1`
        along `axis`.

    """
    (med_1, med_2) = (ma.median(group_1,axis=axis), ma.median(group_2,axis=axis))
    (std_1, std_2) = (mstats.stde_median(group_1, axis=axis),
                      mstats.stde_median(group_2, axis=axis))
    W = np.abs(med_1 - med_2) / ma.sqrt(std_1**2 + std_2**2)
    return 1 - norm.cdf(W)


def idealfourths(data, axis=None):
    """
    Returns an estimate of the lower and upper quartiles.

    Uses the ideal fourths algorithm.

    Parameters
    ----------
    data : array_like
        Input array.
    axis : int, optional
        Axis along which the quartiles are estimated. If None, the arrays are
        flattened.

    Returns
    -------
    idealfourths : {list of floats, masked array}
        Returns the two internal values that divide `data` into four parts
        using the ideal fourths algorithm either along the flattened array
        (if `axis` is None) or along `axis` of `data`.

    """
    def _idf(data):
        x = data.compressed()
        n = len(x)
        if n < 3:
            return [np.nan,np.nan]
        (j,h) = divmod(n/4. + 5/12.,1)
        j = int(j)
        qlo = (1-h)*x[j-1] + h*x[j]
        k = n - j
        qup = (1-h)*x[k] + h*x[k-1]
        return [qlo, qup]
    data = ma.sort(data, axis=axis).view(MaskedArray)
    if (axis is None):
        return _idf(data)
    else:
        return ma.apply_along_axis(_idf, axis, data)


def rsh(data, points=None):
    """
    Evaluates Rosenblatt's shifted histogram estimators for each data point.

    Rosenblatt's estimator is a centered finite-difference approximation to the
    derivative of the empirical cumulative distribution function.

    Parameters
    ----------
    data : sequence
        Input data, should be 1-D. Masked values are ignored.
    points : sequence or None, optional
        Sequence of points where to evaluate Rosenblatt shifted histogram.
        If None, use the data.

    """
    data = ma.array(data, copy=False)
    if points is None:
        points = data
    else:
        points = np.array(points, copy=False, ndmin=1)

    if data.ndim != 1:
        raise AttributeError("The input array should be 1D only !")

    n = data.count()
    r = idealfourths(data, axis=None)
    h = 1.2 * (r[-1]-r[0]) / n**(1./5)
    nhi = (data[:,None] <= points[None,:] + h).sum(0)
    nlo = (data[:,None] < points[None,:] - h).sum(0)
    return (nhi-nlo) / (2.*n*h)
