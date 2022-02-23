from __future__ import annotations
import math
import warnings
from collections import namedtuple

import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, atleast_1d, sqrt, array,
                   compress, pi, exp, ravel, count_nonzero, sin, cos,
                   arctan2, hypot)

from scipy import optimize
from scipy import special
from . import statlib
from . import stats
from .stats import find_repeats, _contains_nan, _normtest_finish
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr


__all__ = ['mvsdist',
           'bayes_mvs', 'kstat', 'kstatvar', 'probplot', 'ppcc_max', 'ppcc_plot',
           'boxcox_llf', 'boxcox', 'boxcox_normmax', 'boxcox_normplot',
           'shapiro', 'anderson', 'ansari', 'bartlett', 'levene', 'binom_test',
           'fligner', 'mood', 'wilcoxon', 'median_test',
           'circmean', 'circvar', 'circstd', 'anderson_ksamp',
           'yeojohnson_llf', 'yeojohnson', 'yeojohnson_normmax',
           'yeojohnson_normplot'
           ]


Mean = namedtuple('Mean', ('statistic', 'minmax'))
Variance = namedtuple('Variance', ('statistic', 'minmax'))
Std_dev = namedtuple('Std_dev', ('statistic', 'minmax'))


def bayes_mvs(data, alpha=0.90):
    r"""
    Bayesian confidence intervals for the mean, var, and std.

    Parameters
    ----------
    data : array_like
        Input data, if multi-dimensional it is flattened to 1-D by `bayes_mvs`.
        Requires 2 or more data points.
    alpha : float, optional
        Probability that the returned confidence interval contains
        the true parameter.

    Returns
    -------
    mean_cntr, var_cntr, std_cntr : tuple
        The three results are for the mean, variance and standard deviation,
        respectively.  Each result is a tuple of the form::

            (center, (lower, upper))

        with `center` the mean of the conditional pdf of the value given the
        data, and `(lower, upper)` a confidence interval, centered on the
        median, containing the estimate to a probability ``alpha``.

    See Also
    --------
    mvsdist

    Notes
    -----
    Each tuple of mean, variance, and standard deviation estimates represent
    the (center, (lower, upper)) with center the mean of the conditional pdf
    of the value given the data and (lower, upper) is a confidence interval
    centered on the median, containing the estimate to a probability
    ``alpha``.

    Converts data to 1-D and assumes all data has the same mean and variance.
    Uses Jeffrey's prior for variance and std.

    Equivalent to ``tuple((x.mean(), x.interval(alpha)) for x in mvsdist(dat))``

    References
    ----------
    T.E. Oliphant, "A Bayesian perspective on estimating mean, variance, and
    standard-deviation from data", https://scholarsarchive.byu.edu/facpub/278,
    2006.

    Examples
    --------
    First a basic example to demonstrate the outputs:

    >>> from scipy import stats
    >>> data = [6, 9, 12, 7, 8, 8, 13]
    >>> mean, var, std = stats.bayes_mvs(data)
    >>> mean
    Mean(statistic=9.0, minmax=(7.103650222612533, 10.896349777387467))
    >>> var
    Variance(statistic=10.0, minmax=(3.176724206..., 24.45910382...))
    >>> std
    Std_dev(statistic=2.9724954732045084, minmax=(1.7823367265645143, 4.945614605014631))

    Now we generate some normally distributed random data, and get estimates of
    mean and standard deviation with 95% confidence intervals for those
    estimates:

    >>> n_samples = 100000
    >>> data = stats.norm.rvs(size=n_samples)
    >>> res_mean, res_var, res_std = stats.bayes_mvs(data, alpha=0.95)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.hist(data, bins=100, density=True, label='Histogram of data')
    >>> ax.vlines(res_mean.statistic, 0, 0.5, colors='r', label='Estimated mean')
    >>> ax.axvspan(res_mean.minmax[0],res_mean.minmax[1], facecolor='r',
    ...            alpha=0.2, label=r'Estimated mean (95% limits)')
    >>> ax.vlines(res_std.statistic, 0, 0.5, colors='g', label='Estimated scale')
    >>> ax.axvspan(res_std.minmax[0],res_std.minmax[1], facecolor='g', alpha=0.2,
    ...            label=r'Estimated scale (95% limits)')

    >>> ax.legend(fontsize=10)
    >>> ax.set_xlim([-4, 4])
    >>> ax.set_ylim([0, 0.5])
    >>> plt.show()

    """
    m, v, s = mvsdist(data)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("0 < alpha < 1 is required, but alpha=%s was given."
                         % alpha)

    m_res = Mean(m.mean(), m.interval(alpha))
    v_res = Variance(v.mean(), v.interval(alpha))
    s_res = Std_dev(s.mean(), s.interval(alpha))

    return m_res, v_res, s_res


def mvsdist(data):
    """
    'Frozen' distributions for mean, variance, and standard deviation of data.

    Parameters
    ----------
    data : array_like
        Input array. Converted to 1-D using ravel.
        Requires 2 or more data-points.

    Returns
    -------
    mdist : "frozen" distribution object
        Distribution object representing the mean of the data.
    vdist : "frozen" distribution object
        Distribution object representing the variance of the data.
    sdist : "frozen" distribution object
        Distribution object representing the standard deviation of the data.

    See Also
    --------
    bayes_mvs

    Notes
    -----
    The return values from ``bayes_mvs(data)`` is equivalent to
    ``tuple((x.mean(), x.interval(0.90)) for x in mvsdist(data))``.

    In other words, calling ``<dist>.mean()`` and ``<dist>.interval(0.90)``
    on the three distribution objects returned from this function will give
    the same results that are returned from `bayes_mvs`.

    References
    ----------
    T.E. Oliphant, "A Bayesian perspective on estimating mean, variance, and
    standard-deviation from data", https://scholarsarchive.byu.edu/facpub/278,
    2006.

    Examples
    --------
    >>> from scipy import stats
    >>> data = [6, 9, 12, 7, 8, 8, 13]
    >>> mean, var, std = stats.mvsdist(data)

    We now have frozen distribution objects "mean", "var" and "std" that we can
    examine:

    >>> mean.mean()
    9.0
    >>> mean.interval(0.95)
    (6.6120585482655692, 11.387941451734431)
    >>> mean.std()
    1.1952286093343936

    """
    x = ravel(data)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 data-points.")
    xbar = x.mean()
    C = x.var()
    if n > 1000:  # gaussian approximations for large n
        mdist = distributions.norm(loc=xbar, scale=math.sqrt(C / n))
        sdist = distributions.norm(loc=math.sqrt(C), scale=math.sqrt(C / (2. * n)))
        vdist = distributions.norm(loc=C, scale=math.sqrt(2.0 / n) * C)
    else:
        nm1 = n - 1
        fac = n * C / 2.
        val = nm1 / 2.
        mdist = distributions.t(nm1, loc=xbar, scale=math.sqrt(C / nm1))
        sdist = distributions.gengamma(val, -2, scale=math.sqrt(fac))
        vdist = distributions.invgamma(val, scale=fac)
    return mdist, vdist, sdist


def kstat(data, n=2):
    r"""
    Return the nth k-statistic (1<=n<=4 so far).

    The nth k-statistic k_n is the unique symmetric unbiased estimator of the
    nth cumulant kappa_n.

    Parameters
    ----------
    data : array_like
        Input array. Note that n-D input gets flattened.
    n : int, {1, 2, 3, 4}, optional
        Default is equal to 2.

    Returns
    -------
    kstat : float
        The nth k-statistic.

    See Also
    --------
    kstatvar: Returns an unbiased estimator of the variance of the k-statistic.
    moment: Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    For a sample size n, the first few k-statistics are given by:

    .. math::

        k_{1} = \mu
        k_{2} = \frac{n}{n-1} m_{2}
        k_{3} = \frac{ n^{2} } {(n-1) (n-2)} m_{3}
        k_{4} = \frac{ n^{2} [(n + 1)m_{4} - 3(n - 1) m^2_{2}]} {(n-1) (n-2) (n-3)}

    where :math:`\mu` is the sample mean, :math:`m_2` is the sample
    variance, and :math:`m_i` is the i-th sample central moment.

    References
    ----------
    http://mathworld.wolfram.com/k-Statistic.html

    http://mathworld.wolfram.com/Cumulant.html

    Examples
    --------
    >>> from scipy import stats
    >>> from numpy.random import default_rng
    >>> rng = default_rng()

    As sample size increases, n-th moment and n-th k-statistic converge to the
    same number (although they aren't identical). In the case of the normal
    distribution, they converge to zero.

    >>> for n in [2, 3, 4, 5, 6, 7]:
    ...     x = rng.normal(size=10**n)
    ...     m, k = stats.moment(x, 3), stats.kstat(x, 3)
    ...     print("%.3g %.3g %.3g" % (m, k, m-k))
    -0.631 -0.651 0.0194  # random
    0.0282 0.0283 -8.49e-05
    -0.0454 -0.0454 1.36e-05
    7.53e-05 7.53e-05 -2.26e-09
    0.00166 0.00166 -4.99e-09
    -2.88e-06 -2.88e-06 8.63e-13
    """
    if n > 4 or n < 1:
        raise ValueError("k-statistics only supported for 1<=n<=4")
    n = int(n)
    S = np.zeros(n + 1, np.float64)
    data = ravel(data)
    N = data.size

    # raise ValueError on empty input
    if N == 0:
        raise ValueError("Data input must not be empty")

    # on nan input, return nan without warning
    if np.isnan(np.sum(data)):
        return np.nan

    for k in range(1, n + 1):
        S[k] = np.sum(data**k, axis=0)
    if n == 1:
        return S[1] * 1.0/N
    elif n == 2:
        return (N*S[2] - S[1]**2.0) / (N*(N - 1.0))
    elif n == 3:
        return (2*S[1]**3 - 3*N*S[1]*S[2] + N*N*S[3]) / (N*(N - 1.0)*(N - 2.0))
    elif n == 4:
        return ((-6*S[1]**4 + 12*N*S[1]**2 * S[2] - 3*N*(N-1.0)*S[2]**2 -
                 4*N*(N+1)*S[1]*S[3] + N*N*(N+1)*S[4]) /
                 (N*(N-1.0)*(N-2.0)*(N-3.0)))
    else:
        raise ValueError("Should not be here.")


def kstatvar(data, n=2):
    r"""Return an unbiased estimator of the variance of the k-statistic.

    See `kstat` for more details of the k-statistic.

    Parameters
    ----------
    data : array_like
        Input array. Note that n-D input gets flattened.
    n : int, {1, 2}, optional
        Default is equal to 2.

    Returns
    -------
    kstatvar : float
        The nth k-statistic variance.

    See Also
    --------
    kstat: Returns the n-th k-statistic.
    moment: Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    The variances of the first few k-statistics are given by:

    .. math::

        var(k_{1}) = \frac{\kappa^2}{n}
        var(k_{2}) = \frac{\kappa^4}{n} + \frac{2\kappa^2_{2}}{n - 1}
        var(k_{3}) = \frac{\kappa^6}{n} + \frac{9 \kappa_2 \kappa_4}{n - 1} +
                     \frac{9 \kappa^2_{3}}{n - 1} +
                     \frac{6 n \kappa^3_{2}}{(n-1) (n-2)}
        var(k_{4}) = \frac{\kappa^8}{n} + \frac{16 \kappa_2 \kappa_6}{n - 1} +
                     \frac{48 \kappa_{3} \kappa_5}{n - 1} +
                     \frac{34 \kappa^2_{4}}{n-1} + \frac{72 n \kappa^2_{2} \kappa_4}{(n - 1) (n - 2)} +
                     \frac{144 n \kappa_{2} \kappa^2_{3}}{(n - 1) (n - 2)} +
                     \frac{24 (n + 1) n \kappa^4_{2}}{(n - 1) (n - 2) (n - 3)}
    """
    data = ravel(data)
    N = len(data)
    if n == 1:
        return kstat(data, n=2) * 1.0/N
    elif n == 2:
        k2 = kstat(data, n=2)
        k4 = kstat(data, n=4)
        return (2*N*k2**2 + (N-1)*k4) / (N*(N+1))
    else:
        raise ValueError("Only n=1 or n=2 supported.")


def _calc_uniform_order_statistic_medians(n):
    """Approximations of uniform order statistic medians.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    v : 1d float array
        Approximations of the order statistic medians.

    References
    ----------
    .. [1] James J. Filliben, "The Probability Plot Correlation Coefficient
           Test for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    Order statistics of the uniform distribution on the unit interval
    are marginally distributed according to beta distributions.
    The expectations of these order statistic are evenly spaced across
    the interval, but the distributions are skewed in a way that
    pushes the medians slightly towards the endpoints of the unit interval:

    >>> n = 4
    >>> k = np.arange(1, n+1)
    >>> from scipy.stats import beta
    >>> a = k
    >>> b = n-k+1
    >>> beta.mean(a, b)
    array([0.2, 0.4, 0.6, 0.8])
    >>> beta.median(a, b)
    array([0.15910358, 0.38572757, 0.61427243, 0.84089642])

    The Filliben approximation uses the exact medians of the smallest
    and greatest order statistics, and the remaining medians are approximated
    by points spread evenly across a sub-interval of the unit interval:

    >>> from scipy.morestats import _calc_uniform_order_statistic_medians
    >>> _calc_uniform_order_statistic_medians(n)
    array([0.15910358, 0.38545246, 0.61454754, 0.84089642])

    This plot shows the skewed distributions of the order statistics
    of a sample of size four from a uniform distribution on the unit interval:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0.0, 1.0, num=50, endpoint=True)
    >>> pdfs = [beta.pdf(x, a[i], b[i]) for i in range(n)]
    >>> plt.figure()
    >>> plt.plot(x, pdfs[0], x, pdfs[1], x, pdfs[2], x, pdfs[3])

    """
    v = np.empty(n, dtype=np.float64)
    v[-1] = 0.5**(1.0 / n)
    v[0] = 1 - v[-1]
    i = np.arange(2, n)
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    return v


def _parse_dist_kw(dist, enforce_subclass=True):
    """Parse `dist` keyword.

    Parameters
    ----------
    dist : str or stats.distributions instance.
        Several functions take `dist` as a keyword, hence this utility
        function.
    enforce_subclass : bool, optional
        If True (default), `dist` needs to be a
        `_distn_infrastructure.rv_generic` instance.
        It can sometimes be useful to set this keyword to False, if a function
        wants to accept objects that just look somewhat like such an instance
        (for example, they have a ``ppf`` method).

    """
    if isinstance(dist, rv_generic):
        pass
    elif isinstance(dist, str):
        try:
            dist = getattr(distributions, dist)
        except AttributeError as e:
            raise ValueError("%s is not a valid distribution name" % dist) from e
    elif enforce_subclass:
        msg = ("`dist` should be a stats.distributions instance or a string "
               "with the name of such a distribution.")
        raise ValueError(msg)

    return dist


def _add_axis_labels_title(plot, xlabel, ylabel, title):
    """Helper function to add axes labels and a title to stats plots."""
    try:
        if hasattr(plot, 'set_title'):
            # Matplotlib Axes instance or something that looks like it
            plot.set_title(title)
            plot.set_xlabel(xlabel)
            plot.set_ylabel(ylabel)
        else:
            # matplotlib.pyplot module
            plot.title(title)
            plot.xlabel(xlabel)
            plot.ylabel(ylabel)
    except Exception:
        # Not an MPL object or something that looks (enough) like it.
        # Don't crash on adding labels or title
        pass


def probplot(x, sparams=(), dist='norm', fit=True, plot=None, rvalue=False):
    """
    Calculate quantiles for a probability plot, and optionally show the plot.

    Generates a probability plot of sample data against the quantiles of a
    specified theoretical distribution (the normal distribution by default).
    `probplot` optionally calculates a best-fit line for the data and plots the
    results using Matplotlib or a given plot function.

    Parameters
    ----------
    x : array_like
        Sample/response data from which `probplot` creates the plot.
    sparams : tuple, optional
        Distribution-specific shape parameters (shape parameters plus location
        and scale).
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name. The default is 'norm' for a
        normal probability plot.  Objects that look enough like a
        stats.distributions instance (i.e. they have a ``ppf`` method) are also
        accepted.
    fit : bool, optional
        Fit a least-squares regression (best-fit) line to the sample data if
        True (default).
    plot : object, optional
        If given, plots the quantiles.
        If given and `fit` is True, also plots the least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.

    Returns
    -------
    (osm, osr) : tuple of ndarrays
        Tuple of theoretical quantiles (osm, or order statistic medians) and
        ordered responses (osr).  `osr` is simply sorted input `x`.
        For details on how `osm` is calculated see the Notes section.
    (slope, intercept, r) : tuple of floats, optional
        Tuple  containing the result of the least-squares fit, if that is
        performed by `probplot`. `r` is the square root of the coefficient of
        determination.  If ``fit=False`` and ``plot=None``, this tuple is not
        returned.

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by `probplot`;
    ``plt.show()`` or ``plt.savefig('figname.png')`` should be used after
    calling `probplot`.

    `probplot` generates a probability plot, which should not be confused with
    a Q-Q or a P-P plot.  Statsmodels has more extensive functionality of this
    type, see ``statsmodels.api.ProbPlot``.

    The formula used for the theoretical quantiles (horizontal axis of the
    probability plot) is Filliben's estimate::

        quantiles = dist.ppf(val), for

                0.5**(1/n),                  for i = n
          val = (i - 0.3175) / (n + 0.365),  for i = 2, ..., n-1
                1 - 0.5**(1/n),              for i = 1

    where ``i`` indicates the i-th ordered value and ``n`` is the total number
    of values.

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> nsample = 100
    >>> rng = np.random.default_rng()

    A t distribution with small degrees of freedom:

    >>> ax1 = plt.subplot(221)
    >>> x = stats.t.rvs(3, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    A t distribution with larger degrees of freedom:

    >>> ax2 = plt.subplot(222)
    >>> x = stats.t.rvs(25, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    A mixture of two normal distributions with broadcasting:

    >>> ax3 = plt.subplot(223)
    >>> x = stats.norm.rvs(loc=[0,5], scale=[1,1.5],
    ...                    size=(nsample//2,2), random_state=rng).ravel()
    >>> res = stats.probplot(x, plot=plt)

    A standard normal distribution:

    >>> ax4 = plt.subplot(224)
    >>> x = stats.norm.rvs(loc=0, scale=1, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    Produce a new figure with a loggamma distribution, using the ``dist`` and
    ``sparams`` keywords:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> x = stats.loggamma.rvs(c=2.5, size=500, random_state=rng)
    >>> res = stats.probplot(x, dist=stats.loggamma, sparams=(2.5,), plot=ax)
    >>> ax.set_title("Probplot for loggamma dist with shape parameter 2.5")

    Show the results with Matplotlib:

    >>> plt.show()

    """
    x = np.asarray(x)
    if x.size == 0:
        if fit:
            return (x, x), (np.nan, np.nan, 0.0)
        else:
            return x, x

    osm_uniform = _calc_uniform_order_statistic_medians(len(x))
    dist = _parse_dist_kw(dist, enforce_subclass=False)
    if sparams is None:
        sparams = ()
    if isscalar(sparams):
        sparams = (sparams,)
    if not isinstance(sparams, tuple):
        sparams = tuple(sparams)

    osm = dist.ppf(osm_uniform, *sparams)
    osr = sort(x)
    if fit:
        # perform a linear least squares fit.
        slope, intercept, r, prob, _ = stats.linregress(osm, osr)

    if plot is not None:
        plot.plot(osm, osr, 'bo')
        if fit:
            plot.plot(osm, slope*osm + intercept, 'r-')
        _add_axis_labels_title(plot, xlabel='Theoretical quantiles',
                               ylabel='Ordered Values',
                               title='Probability Plot')

        # Add R^2 value to the plot as text
        if rvalue:
            xmin = amin(osm)
            xmax = amax(osm)
            ymin = amin(x)
            ymax = amax(x)
            posx = xmin + 0.70 * (xmax - xmin)
            posy = ymin + 0.01 * (ymax - ymin)
            plot.text(posx, posy, "$R^2=%1.4f$" % r**2)

    if fit:
        return (osm, osr), (slope, intercept, r)
    else:
        return osm, osr


def ppcc_max(x, brack=(0.0, 1.0), dist='tukeylambda'):
    """Calculate the shape parameter that maximizes the PPCC.

    The probability plot correlation coefficient (PPCC) plot can be used
    to determine the optimal shape parameter for a one-parameter family
    of distributions. ``ppcc_max`` returns the shape parameter that would
    maximize the probability plot correlation coefficient for the given
    data to a one-parameter family of distributions.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : tuple, optional
        Triple (a,b,c) where (a<b<c). If bracket consists of two numbers (a, c)
        then they are assumed to be a starting interval for a downhill bracket
        search (see `scipy.optimize.brent`).
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.

    Returns
    -------
    shape_value : float
        The shape parameter at which the probability plot correlation
        coefficient reaches its max value.

    See Also
    --------
    ppcc_plot, probplot, boxcox

    Notes
    -----
    The brack keyword serves as a starting point which is useful in corner
    cases. One can use a plot to obtain a rough visual estimate of the location
    for the maximum to start the search near it.

    References
    ----------
    .. [1] J.J. Filliben, "The Probability Plot Correlation Coefficient Test
           for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.
    .. [2] Engineering Statistics Handbook, NIST/SEMATEC,
           https://www.itl.nist.gov/div898/handbook/eda/section3/ppccplot.htm

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5:

    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Generate the PPCC plot for this data with the Weibull distribution.

    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax)

    We calculate the value where the shape should reach its maximum and a
    red line is drawn there. The line should coincide with the highest
    point in the PPCC graph.

    >>> cmax = stats.ppcc_max(x, brack=(c/2, 2*c), dist='weibull_min')
    >>> ax.axvline(cmax, color='r')
    >>> plt.show()

    """
    dist = _parse_dist_kw(dist)
    osm_uniform = _calc_uniform_order_statistic_medians(len(x))
    osr = sort(x)

    # this function computes the x-axis values of the probability plot
    #  and computes a linear regression (including the correlation)
    #  and returns 1-r so that a minimization function maximizes the
    #  correlation
    def tempfunc(shape, mi, yvals, func):
        xvals = func(mi, shape)
        r, prob = stats.pearsonr(xvals, yvals)
        return 1 - r

    return optimize.brent(tempfunc, brack=brack,
                          args=(osm_uniform, osr, dist.ppf))


def ppcc_plot(x, a, b, dist='tukeylambda', plot=None, N=80):
    """Calculate and optionally plot probability plot correlation coefficient.

    The probability plot correlation coefficient (PPCC) plot can be used to
    determine the optimal shape parameter for a one-parameter family of
    distributions.  It cannot be used for distributions without shape
    parameters
    (like the normal distribution) or with multiple shape parameters.

    By default a Tukey-Lambda distribution (`stats.tukeylambda`) is used. A
    Tukey-Lambda PPCC plot interpolates from long-tailed to short-tailed
    distributions via an approximately normal one, and is therefore
    particularly useful in practice.

    Parameters
    ----------
    x : array_like
        Input array.
    a, b : scalar
        Lower and upper bounds of the shape parameter to use.
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.
    plot : object, optional
        If given, plots PPCC against the shape parameter.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `a` to `b`).

    Returns
    -------
    svals : ndarray
        The shape values for which `ppcc` was calculated.
    ppcc : ndarray
        The calculated probability plot correlation coefficient values.

    See Also
    --------
    ppcc_max, probplot, boxcox_normplot, tukeylambda

    References
    ----------
    J.J. Filliben, "The Probability Plot Correlation Coefficient Test for
    Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5, and plot the histogram of the data:

    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Take a look at the histogram of the data.

    >>> fig1, ax = plt.subplots(figsize=(9, 4))
    >>> ax.hist(x, bins=50)
    >>> ax.set_title('Histogram of x')
    >>> plt.show()

    Now we explore this data with a PPCC plot as well as the related
    probability plot and Box-Cox normplot.  A red line is drawn where we
    expect the PPCC value to be maximal (at the shape parameter ``c``
    used above):

    >>> fig2 = plt.figure(figsize=(12, 4))
    >>> ax1 = fig2.add_subplot(1, 3, 1)
    >>> ax2 = fig2.add_subplot(1, 3, 2)
    >>> ax3 = fig2.add_subplot(1, 3, 3)
    >>> res = stats.probplot(x, plot=ax1)
    >>> res = stats.boxcox_normplot(x, -4, 4, plot=ax2)
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax3)
    >>> ax3.axvline(c, color='r')
    >>> plt.show()

    """
    if b <= a:
        raise ValueError("`b` has to be larger than `a`.")

    svals = np.linspace(a, b, num=N)
    ppcc = np.empty_like(svals)
    for k, sval in enumerate(svals):
        _, r2 = probplot(x, sval, dist=dist, fit=True)
        ppcc[k] = r2[-1]

    if plot is not None:
        plot.plot(svals, ppcc, 'x')
        _add_axis_labels_title(plot, xlabel='Shape Values',
                               ylabel='Prob Plot Corr. Coef.',
                               title='(%s) PPCC Plot' % dist)

    return svals, ppcc


def boxcox_llf(lmb, data):
    r"""The boxcox log-likelihood function.

    Parameters
    ----------
    lmb : scalar
        Parameter for Box-Cox transformation.  See `boxcox` for details.
    data : array_like
        Data to calculate Box-Cox log-likelihood for.  If `data` is
        multi-dimensional, the log-likelihood is calculated along the first
        axis.

    Returns
    -------
    llf : float or ndarray
        Box-Cox log-likelihood of `data` given `lmb`.  A float for 1-D `data`,
        an array otherwise.

    See Also
    --------
    boxcox, probplot, boxcox_normplot, boxcox_normmax

    Notes
    -----
    The Box-Cox log-likelihood function is defined here as

    .. math::

        llf = (\lambda - 1) \sum_i(\log(x_i)) -
              N/2 \log(\sum_i (y_i - \bar{y})^2 / N),

    where ``y`` is the Box-Cox transformed input data ``x``.

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    Generate some random variates and calculate Box-Cox log-likelihood values
    for them for a range of ``lmbda`` values:

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, loc=10, size=1000, random_state=rng)
    >>> lmbdas = np.linspace(-2, 10)
    >>> llf = np.zeros(lmbdas.shape, dtype=float)
    >>> for ii, lmbda in enumerate(lmbdas):
    ...     llf[ii] = stats.boxcox_llf(lmbda, x)

    Also find the optimal lmbda value with `boxcox`:

    >>> x_most_normal, lmbda_optimal = stats.boxcox(x)

    Plot the log-likelihood as function of lmbda.  Add the optimal lmbda as a
    horizontal line to check that that's really the optimum:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(lmbdas, llf, 'b.-')
    >>> ax.axhline(stats.boxcox_llf(lmbda_optimal, x), color='r')
    >>> ax.set_xlabel('lmbda parameter')
    >>> ax.set_ylabel('Box-Cox log-likelihood')

    Now add some probability plots to show that where the log-likelihood is
    maximized the data transformed with `boxcox` looks closest to normal:

    >>> locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
    >>> for lmbda, loc in zip([-1, lmbda_optimal, 9], locs):
    ...     xt = stats.boxcox(x, lmbda=lmbda)
    ...     (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
    ...     ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
    ...     ax_inset.plot(osm, osr, 'c.', osm, slope*osm + intercept, 'k-')
    ...     ax_inset.set_xticklabels([])
    ...     ax_inset.set_yticklabels([])
    ...     ax_inset.set_title(r'$\lambda=%1.2f$' % lmbda)

    >>> plt.show()

    """
    data = np.asarray(data)
    N = data.shape[0]
    if N == 0:
        return np.nan

    logdata = np.log(data)

    # Compute the variance of the transformed data.
    if lmb == 0:
        variance = np.var(logdata, axis=0)
    else:
        # Transform without the constant offset 1/lmb.  The offset does
        # not effect the variance, and the subtraction of the offset can
        # lead to loss of precision.
        variance = np.var(data**lmb / lmb, axis=0)

    return (lmb - 1) * np.sum(logdata, axis=0) - N/2 * np.log(variance)


def _boxcox_conf_interval(x, lmax, alpha):
    # Need to find the lambda for which
    #  f(x,lmbda) >= f(x,lmax) - 0.5*chi^2_alpha;1
    fac = 0.5 * distributions.chi2.ppf(1 - alpha, 1)
    target = boxcox_llf(lmax, x) - fac

    def rootfunc(lmbda, data, target):
        return boxcox_llf(lmbda, data) - target

    # Find positive endpoint of interval in which answer is to be found
    newlm = lmax + 0.5
    N = 0
    while (rootfunc(newlm, x, target) > 0.0) and (N < 500):
        newlm += 0.1
        N += 1

    if N == 500:
        raise RuntimeError("Could not find endpoint.")

    lmplus = optimize.brentq(rootfunc, lmax, newlm, args=(x, target))

    # Now find negative interval in the same way
    newlm = lmax - 0.5
    N = 0
    while (rootfunc(newlm, x, target) > 0.0) and (N < 500):
        newlm -= 0.1
        N += 1

    if N == 500:
        raise RuntimeError("Could not find endpoint.")

    lmminus = optimize.brentq(rootfunc, newlm, lmax, args=(x, target))
    return lmminus, lmplus


def boxcox(x, lmbda=None, alpha=None, optimizer=None):
    r"""Return a dataset transformed by a Box-Cox power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Must be positive 1-dimensional.  Must not be constant.
    lmbda : {None, scalar}, optional
        If `lmbda` is not None, do the transformation for that value.
        If `lmbda` is None, find the lambda that maximizes the log-likelihood
        function and return it as the second output argument.
    alpha : {None, float}, optional
        If ``alpha`` is not None, return the ``100 * (1-alpha)%`` confidence
        interval for `lmbda` as the third output argument.
        Must be between 0.0 and 1.0.
    optimizer : callable, optional
        If `lmbda` is None, `optimizer` is the scalar optimizer used to find
        the value of `lmbda` that minimizes the negative log-likelihood
        function. `optimizer` is a callable that accepts one argument:

        fun : callable
            The objective function, which evaluates the negative
            log-likelihood function at a provided value of `lmbda`

        and returns an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal value of
        `lmbda` in an attribute `x`.

        See the example in `boxcox_normmax` or the documentation of
        `scipy.optimize.minimize_scalar` for more information.

        If `lmbda` is not None, `optimizer` is ignored.

    Returns
    -------
    boxcox : ndarray
        Box-Cox power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.
    (min_ci, max_ci) : tuple of float, optional
        If `lmbda` parameter is None and ``alpha`` is not None, this returned
        tuple of floats represents the minimum and maximum confidence limits
        given ``alpha``.

    See Also
    --------
    probplot, boxcox_normplot, boxcox_normmax, boxcox_llf

    Notes
    -----
    The Box-Cox transform is given by::

        y = (x**lmbda - 1) / lmbda,  for lmbda != 0
            log(x),                  for lmbda = 0

    `boxcox` requires the input data to be positive.  Sometimes a Box-Cox
    transformation provides a shift parameter to achieve this; `boxcox` does
    not.  Such a shift parameter is equivalent to adding a positive constant to
    `x` before calling `boxcox`.

    The confidence limits returned when ``alpha`` is provided give the interval
    where:

    .. math::

        llf(\hat{\lambda}) - llf(\lambda) < \frac{1}{2}\chi^2(1 - \alpha, 1),

    with ``llf`` the log-likelihood function and :math:`\chi^2` the chi-squared
    function.

    References
    ----------
    G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal of the
    Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `boxcox` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, _ = stats.boxcox(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Box-Cox transformation')

    >>> plt.show()

    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")

    if x.size == 0:
        return x

    if np.all(x == x[0]):
        raise ValueError("Data must not be constant.")

    if np.any(x <= 0):
        raise ValueError("Data must be positive.")

    if lmbda is not None:  # single transformation
        return special.boxcox(x, lmbda)

    # If lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = boxcox_normmax(x, method='mle', optimizer=optimizer)
    y = boxcox(x, lmax)

    if alpha is None:
        return y, lmax
    else:
        # Find confidence interval
        interval = _boxcox_conf_interval(x, lmax, alpha)
        return y, lmax, interval


def boxcox_normmax(x, brack=None, method='pearsonr', optimizer=None):
    """Compute optimal Box-Cox transform parameter for input data.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : 2-tuple, optional, default (-2.0, 2.0)
         The starting interval for a downhill bracket search for the default
         `optimize.brent` solver. Note that this is in most cases not
         critical; the final result is allowed to be outside this bracket.
         If `optimizer` is passed, `brack` must be None.
    method : str, optional
        The method to determine the optimal transform parameter (`boxcox`
        ``lmbda`` parameter). Options are:

        'pearsonr'  (default)
            Maximizes the Pearson correlation coefficient between
            ``y = boxcox(x)`` and the expected values for ``y`` if `x` would be
            normally-distributed.

        'mle'
            Minimizes the log-likelihood `boxcox_llf`.  This is the method used
            in `boxcox`.

        'all'
            Use all optimization methods available, and return all results.
            Useful to compare different methods.
    optimizer : callable, optional
        `optimizer` is a callable that accepts one argument:

        fun : callable
            The objective function to be optimized. `fun` accepts one argument,
            the Box-Cox transform parameter `lmbda`, and returns the negative
            log-likelihood function at the provided value. The job of `optimizer`
            is to find the value of `lmbda` that minimizes `fun`.

        and returns an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal value of
        `lmbda` in an attribute `x`.

        See the example below or the documentation of
        `scipy.optimize.minimize_scalar` for more information.

    Returns
    -------
    maxlog : float or ndarray
        The optimal transform parameter found.  An array instead of a scalar
        for ``method='all'``.

    See Also
    --------
    boxcox, boxcox_llf, boxcox_normplot, scipy.optimize.minimize_scalar

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We can generate some data and determine the optimal ``lmbda`` in various
    ways:

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> y, lmax_mle = stats.boxcox(x)
    >>> lmax_pearsonr = stats.boxcox_normmax(x)

    >>> lmax_mle
    1.4613865614008015
    >>> lmax_pearsonr
    1.6685004886804342
    >>> stats.boxcox_normmax(x, method='all')
    array([1.66850049, 1.46138656])

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.boxcox_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax_mle, color='r')
    >>> ax.axvline(lmax_pearsonr, color='g', ls='--')

    >>> plt.show()

    Alternatively, we can define our own `optimizer` function. Suppose we
    are only interested in values of `lmbda` on the interval [6, 7], we
    want to use `scipy.optimize.minimize_scalar` with ``method='bounded'``,
    and we want to use tighter tolerances when optimizing the log-likelihood
    function. To do this, we define a function that accepts positional argument
    `fun` and uses `scipy.optimize.minimize_scalar` to minimize `fun` subject
    to the provided bounds and tolerances:

    >>> from scipy import optimize
    >>> options = {'xatol': 1e-12}  # absolute tolerance on `x`
    >>> def optimizer(fun):
    ...     return optimize.minimize_scalar(fun, bounds=(6, 7),
    ...                                     method="bounded", options=options)
    >>> stats.boxcox_normmax(x, optimizer=optimizer)
    6.000...
    """
    # If optimizer is not given, define default 'brent' optimizer.
    if optimizer is None:

        # Set default value for `brack`.
        if brack is None:
            brack = (-2.0, 2.0)

        def _optimizer(func, args):
            return optimize.brent(func, args=args, brack=brack)

    # Otherwise check optimizer.
    else:
        if not callable(optimizer):
            raise ValueError("`optimizer` must be a callable")

        if brack is not None:
            raise ValueError("`brack` must be None if `optimizer` is given")

        # `optimizer` is expected to return a `OptimizeResult` object, we here
        # get the solution to the optimization problem.
        def _optimizer(func, args):
            def func_wrapped(x):
                return func(x, *args)
            return getattr(optimizer(func_wrapped), 'x', None)

    def _pearsonr(x):
        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            # This function computes the x-axis values of the probability plot
            # and computes a linear regression (including the correlation) and
            # returns ``1 - r`` so that a minimization function maximizes the
            # correlation.
            y = boxcox(samps, lmbda)
            yvals = np.sort(y)
            r, prob = stats.pearsonr(xvals, yvals)
            return 1 - r

        return _optimizer(_eval_pearsonr, args=(xvals, x))

    def _mle(x):
        def _eval_mle(lmb, data):
            # function to minimize
            return -boxcox_llf(lmb, data)

        return _optimizer(_eval_mle, args=(x,))

    def _all(x):
        maxlog = np.empty(2, dtype=float)
        maxlog[0] = _pearsonr(x)
        maxlog[1] = _mle(x)
        return maxlog

    methods = {'pearsonr': _pearsonr,
               'mle': _mle,
               'all': _all}
    if method not in methods.keys():
        raise ValueError("Method %s not recognized." % method)

    optimfunc = methods[method]
    res = optimfunc(x)
    if res is None:
        message = ("`optimizer` must return an object containing the optimal "
                   "`lmbda` in attribute `x`")
        raise ValueError(message)
    return res


def _normplot(method, x, la, lb, plot=None, N=80):
    """Compute parameters for a Box-Cox or Yeo-Johnson normality plot,
    optionally show it.

    See `boxcox_normplot` or `yeojohnson_normplot` for details.
    """

    if method == 'boxcox':
        title = 'Box-Cox Normality Plot'
        transform_func = boxcox
    else:
        title = 'Yeo-Johnson Normality Plot'
        transform_func = yeojohnson

    x = np.asarray(x)
    if x.size == 0:
        return x

    if lb <= la:
        raise ValueError("`lb` has to be larger than `la`.")

    lmbdas = np.linspace(la, lb, num=N)
    ppcc = lmbdas * 0.0
    for i, val in enumerate(lmbdas):
        # Determine for each lmbda the square root of correlation coefficient
        # of transformed x
        z = transform_func(x, lmbda=val)
        _, (_, _, r) = probplot(z, dist='norm', fit=True)
        ppcc[i] = r

    if plot is not None:
        plot.plot(lmbdas, ppcc, 'x')
        _add_axis_labels_title(plot, xlabel='$\\lambda$',
                               ylabel='Prob Plot Corr. Coef.',
                               title=title)

    return lmbdas, ppcc


def boxcox_normplot(x, la, lb, plot=None, N=80):
    """Compute parameters for a Box-Cox normality plot, optionally show it.

    A Box-Cox normality plot shows graphically what the best transformation
    parameter is to use in `boxcox` to obtain a distribution that is close
    to normal.

    Parameters
    ----------
    x : array_like
        Input array.
    la, lb : scalar
        The lower and upper bounds for the ``lmbda`` values to pass to `boxcox`
        for Box-Cox transformations.  These are also the limits of the
        horizontal axis of the plot if that is generated.
    plot : object, optional
        If given, plots the quantiles and least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `la` to `lb`).

    Returns
    -------
    lmbdas : ndarray
        The ``lmbda`` values for which a Box-Cox transform was done.
    ppcc : ndarray
        Probability Plot Correlelation Coefficient, as obtained from `probplot`
        when fitting the Box-Cox transformed input `x` against a normal
        distribution.

    See Also
    --------
    probplot, boxcox, boxcox_normmax, boxcox_llf, ppcc_max

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by
    `boxcox_normplot`; ``plt.show()`` or ``plt.savefig('figname.png')``
    should be used after calling `probplot`.

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some non-normally distributed data, and create a Box-Cox plot:

    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.boxcox_normplot(x, -20, 20, plot=ax)

    Determine and plot the optimal ``lmbda`` to transform ``x`` and plot it in
    the same plot:

    >>> _, maxlog = stats.boxcox(x)
    >>> ax.axvline(maxlog, color='r')

    >>> plt.show()

    """
    return _normplot('boxcox', x, la, lb, plot, N)


def yeojohnson(x, lmbda=None):
    r"""Return a dataset transformed by a Yeo-Johnson power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Should be 1-dimensional.
    lmbda : float, optional
        If ``lmbda`` is ``None``, find the lambda that maximizes the
        log-likelihood function and return it as the second output argument.
        Otherwise the transformation is done for the given value.

    Returns
    -------
    yeojohnson: ndarray
        Yeo-Johnson power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.

    See Also
    --------
    probplot, yeojohnson_normplot, yeojohnson_normmax, yeojohnson_llf, boxcox

    Notes
    -----
    The Yeo-Johnson transform is given by::

        y = ((x + 1)**lmbda - 1) / lmbda,                for x >= 0, lmbda != 0
            log(x + 1),                                  for x >= 0, lmbda = 0
            -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda),  for x < 0, lmbda != 2
            -log(-x + 1),                                for x < 0, lmbda = 2

    Unlike `boxcox`, `yeojohnson` does not require the input data to be
    positive.

    .. versionadded:: 1.2.0


    References
    ----------
    I. Yeo and R.A. Johnson, "A New Family of Power Transformations to
    Improve Normality or Symmetry", Biometrika 87.4 (2000):


    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `yeojohnson` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, lmbda = stats.yeojohnson(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Yeo-Johnson transformation')

    >>> plt.show()

    """
    x = np.asarray(x)
    if x.size == 0:
        return x

    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError('Yeo-Johnson transformation is not defined for '
                         'complex numbers.')

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64, copy=False)

    if lmbda is not None:
        return _yeojohnson_transform(x, lmbda)

    # if lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = yeojohnson_normmax(x)
    y = _yeojohnson_transform(x, lmax)

    return y, lmax


def _yeojohnson_transform(x, lmbda):
    """Returns `x` transformed by the Yeo-Johnson power transform with given
    parameter `lmbda`.
    """
    out = np.zeros_like(x)
    pos = x >= 0  # binary mask

    # when x >= 0
    if abs(lmbda) < np.spacing(1.):
        out[pos] = np.log1p(x[pos])
    else:  # lmbda != 0
        out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.):
        out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -np.log1p(-x[~pos])

    return out


def yeojohnson_llf(lmb, data):
    r"""The yeojohnson log-likelihood function.

    Parameters
    ----------
    lmb : scalar
        Parameter for Yeo-Johnson transformation. See `yeojohnson` for
        details.
    data : array_like
        Data to calculate Yeo-Johnson log-likelihood for. If `data` is
        multi-dimensional, the log-likelihood is calculated along the first
        axis.

    Returns
    -------
    llf : float
        Yeo-Johnson log-likelihood of `data` given `lmb`.

    See Also
    --------
    yeojohnson, probplot, yeojohnson_normplot, yeojohnson_normmax

    Notes
    -----
    The Yeo-Johnson log-likelihood function is defined here as

    .. math::

        llf = -N/2 \log(\hat{\sigma}^2) + (\lambda - 1)
              \sum_i \text{ sign }(x_i)\log(|x_i| + 1)

    where :math:`\hat{\sigma}^2` is estimated variance of the the Yeo-Johnson
    transformed input data ``x``.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    Generate some random variates and calculate Yeo-Johnson log-likelihood
    values for them for a range of ``lmbda`` values:

    >>> x = stats.loggamma.rvs(5, loc=10, size=1000)
    >>> lmbdas = np.linspace(-2, 10)
    >>> llf = np.zeros(lmbdas.shape, dtype=float)
    >>> for ii, lmbda in enumerate(lmbdas):
    ...     llf[ii] = stats.yeojohnson_llf(lmbda, x)

    Also find the optimal lmbda value with `yeojohnson`:

    >>> x_most_normal, lmbda_optimal = stats.yeojohnson(x)

    Plot the log-likelihood as function of lmbda.  Add the optimal lmbda as a
    horizontal line to check that that's really the optimum:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(lmbdas, llf, 'b.-')
    >>> ax.axhline(stats.yeojohnson_llf(lmbda_optimal, x), color='r')
    >>> ax.set_xlabel('lmbda parameter')
    >>> ax.set_ylabel('Yeo-Johnson log-likelihood')

    Now add some probability plots to show that where the log-likelihood is
    maximized the data transformed with `yeojohnson` looks closest to normal:

    >>> locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
    >>> for lmbda, loc in zip([-1, lmbda_optimal, 9], locs):
    ...     xt = stats.yeojohnson(x, lmbda=lmbda)
    ...     (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
    ...     ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
    ...     ax_inset.plot(osm, osr, 'c.', osm, slope*osm + intercept, 'k-')
    ...     ax_inset.set_xticklabels([])
    ...     ax_inset.set_yticklabels([])
    ...     ax_inset.set_title(r'$\lambda=%1.2f$' % lmbda)

    >>> plt.show()

    """
    data = np.asarray(data)
    n_samples = data.shape[0]

    if n_samples == 0:
        return np.nan

    trans = _yeojohnson_transform(data, lmb)

    loglike = -n_samples / 2 * np.log(trans.var(axis=0))
    loglike += (lmb - 1) * (np.sign(data) * np.log(np.abs(data) + 1)).sum(axis=0)

    return loglike


def yeojohnson_normmax(x, brack=(-2, 2)):
    """Compute optimal Yeo-Johnson transform parameter.

    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : 2-tuple, optional
        The starting interval for a downhill bracket search with
        `optimize.brent`. Note that this is in most cases not critical; the
        final result is allowed to be outside this bracket.

    Returns
    -------
    maxlog : float
        The optimal transform parameter found.

    See Also
    --------
    yeojohnson, yeojohnson_llf, yeojohnson_normplot

    Notes
    -----
    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some data and determine optimal ``lmbda``

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> lmax = stats.yeojohnson_normmax(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax, color='r')

    >>> plt.show()

    """
    def _neg_llf(lmbda, data):
        return -yeojohnson_llf(lmbda, data)

    return optimize.brent(_neg_llf, brack=brack, args=(x,))


def yeojohnson_normplot(x, la, lb, plot=None, N=80):
    """Compute parameters for a Yeo-Johnson normality plot, optionally show it.

    A Yeo-Johnson normality plot shows graphically what the best
    transformation parameter is to use in `yeojohnson` to obtain a
    distribution that is close to normal.

    Parameters
    ----------
    x : array_like
        Input array.
    la, lb : scalar
        The lower and upper bounds for the ``lmbda`` values to pass to
        `yeojohnson` for Yeo-Johnson transformations. These are also the
        limits of the horizontal axis of the plot if that is generated.
    plot : object, optional
        If given, plots the quantiles and least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `la` to `lb`).

    Returns
    -------
    lmbdas : ndarray
        The ``lmbda`` values for which a Yeo-Johnson transform was done.
    ppcc : ndarray
        Probability Plot Correlelation Coefficient, as obtained from `probplot`
        when fitting the Box-Cox transformed input `x` against a normal
        distribution.

    See Also
    --------
    probplot, yeojohnson, yeojohnson_normmax, yeojohnson_llf, ppcc_max

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by
    `boxcox_normplot`; ``plt.show()`` or ``plt.savefig('figname.png')``
    should be used after calling `probplot`.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some non-normally distributed data, and create a Yeo-Johnson plot:

    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -20, 20, plot=ax)

    Determine and plot the optimal ``lmbda`` to transform ``x`` and plot it in
    the same plot:

    >>> _, maxlog = stats.yeojohnson(x)
    >>> ax.axvline(maxlog, color='r')

    >>> plt.show()

    """
    return _normplot('yeojohnson', x, la, lb, plot, N)


ShapiroResult = namedtuple('ShapiroResult', ('statistic', 'pvalue'))


def shapiro(x):
    """Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that the
    data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.

    Returns
    -------
    statistic : float
        The test statistic.
    p-value : float
        The p-value for the hypothesis test.

    See Also
    --------
    anderson : The Anderson-Darling test for normality
    kstest : The Kolmogorov-Smirnov test for goodness of fit.

    Notes
    -----
    The algorithm used is described in [4]_ but censoring parameters as
    described are not implemented. For N > 5000 the W test statistic is accurate
    but the p-value may not be.

    The chance of rejecting the null hypothesis when it is true is close to 5%
    regardless of sample size.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for
           normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
    .. [3] Razali, N. M. & Wah, Y. B. (2011) Power comparisons of Shapiro-Wilk,
           Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests, Journal of
           Statistical Modeling and Analytics, Vol. 2, pp. 21-33.
    .. [4] ALGORITHM AS R94 APPL. STATIST. (1995) VOL. 44, NO. 4.

    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = stats.norm.rvs(loc=5, scale=3, size=100, random_state=rng)
    >>> shapiro_test = stats.shapiro(x)
    >>> shapiro_test
    ShapiroResult(statistic=0.9813305735588074, pvalue=0.16855233907699585)
    >>> shapiro_test.statistic
    0.9813305735588074
    >>> shapiro_test.pvalue
    0.16855233907699585

    """
    x = np.ravel(x)

    N = len(x)
    if N < 3:
        raise ValueError("Data must be at least length 3.")

    a = zeros(N, 'f')
    init = 0

    y = sort(x)
    a, w, pw, ifault = statlib.swilk(y, a[:N//2], init)
    if ifault not in [0, 2]:
        warnings.warn("Input data for shapiro has range zero. The results "
                      "may not be accurate.")
    if N > 5000:
        warnings.warn("p-value may not be accurate for N > 5000.")

    return ShapiroResult(w, pw)


# Values from Stephens, M A, "EDF Statistics for Goodness of Fit and
#             Some Comparisons", Journal of the American Statistical
#             Association, Vol. 69, Issue 347, Sept. 1974, pp 730-737
_Avals_norm = array([0.576, 0.656, 0.787, 0.918, 1.092])
_Avals_expon = array([0.922, 1.078, 1.341, 1.606, 1.957])
# From Stephens, M A, "Goodness of Fit for the Extreme Value Distribution",
#             Biometrika, Vol. 64, Issue 3, Dec. 1977, pp 583-588.
_Avals_gumbel = array([0.474, 0.637, 0.757, 0.877, 1.038])
# From Stephens, M A, "Tests of Fit for the Logistic Distribution Based
#             on the Empirical Distribution Function.", Biometrika,
#             Vol. 66, Issue 3, Dec. 1979, pp 591-595.
_Avals_logistic = array([0.426, 0.563, 0.660, 0.769, 0.906, 1.010])


AndersonResult = namedtuple('AndersonResult', ('statistic',
                                               'critical_values',
                                               'significance_level'))


def anderson(x, dist='norm'):
    """Anderson-Darling test for data coming from a particular distribution.

    The Anderson-Darling test tests the null hypothesis that a sample is
    drawn from a population that follows a particular distribution.
    For the Anderson-Darling test, the critical values depend on
    which distribution is being tested against.  This function works
    for normal, exponential, logistic, or Gumbel (Extreme Value
    Type I) distributions.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    dist : {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1'}, optional
        The type of distribution to test against.  The default is 'norm'.
        The names 'extreme1', 'gumbel_l' and 'gumbel' are synonyms for the
        same distribution.

    Returns
    -------
    statistic : float
        The Anderson-Darling test statistic.
    critical_values : list
        The critical values for this distribution.
    significance_level : list
        The significance levels for the corresponding critical values
        in percents.  The function returns critical values for a
        differing set of significance levels depending on the
        distribution that is being tested against.

    See Also
    --------
    kstest : The Kolmogorov-Smirnov test for goodness-of-fit.

    Notes
    -----
    Critical values provided are for the following significance levels:

    normal/exponential
        15%, 10%, 5%, 2.5%, 1%
    logistic
        25%, 10%, 5%, 2.5%, 1%, 0.5%
    Gumbel
        25%, 10%, 5%, 2.5%, 1%

    If the returned statistic is larger than these critical values then
    for the corresponding significance level, the null hypothesis that
    the data come from the chosen distribution can be rejected.
    The returned statistic is referred to as 'A2' in the references.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and
           Some Comparisons, Journal of the American Statistical Association,
           Vol. 69, pp. 730-737.
    .. [3] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
           Statistics with Unknown Parameters, Annals of Statistics, Vol. 4,
           pp. 357-369.
    .. [4] Stephens, M. A. (1977). Goodness of Fit for the Extreme Value
           Distribution, Biometrika, Vol. 64, pp. 583-588.
    .. [5] Stephens, M. A. (1977). Goodness of Fit with Special Reference
           to Tests for Exponentiality , Technical Report No. 262,
           Department of Statistics, Stanford University, Stanford, CA.
    .. [6] Stephens, M. A. (1979). Tests of Fit for the Logistic Distribution
           Based on the Empirical Distribution Function, Biometrika, Vol. 66,
           pp. 591-595.

    """
    if dist not in ['norm', 'expon', 'gumbel', 'gumbel_l',
                    'gumbel_r', 'extreme1', 'logistic']:
        raise ValueError("Invalid distribution; dist must be 'norm', "
                         "'expon', 'gumbel', 'extreme1' or 'logistic'.")
    y = sort(x)
    xbar = np.mean(x, axis=0)
    N = len(y)
    if dist == 'norm':
        s = np.std(x, ddof=1, axis=0)
        w = (y - xbar) / s
        logcdf = distributions.norm.logcdf(w)
        logsf = distributions.norm.logsf(w)
        sig = array([15, 10, 5, 2.5, 1])
        critical = around(_Avals_norm / (1.0 + 4.0/N - 25.0/N/N), 3)
    elif dist == 'expon':
        w = y / xbar
        logcdf = distributions.expon.logcdf(w)
        logsf = distributions.expon.logsf(w)
        sig = array([15, 10, 5, 2.5, 1])
        critical = around(_Avals_expon / (1.0 + 0.6/N), 3)
    elif dist == 'logistic':
        def rootfunc(ab, xj, N):
            a, b = ab
            tmp = (xj - a) / b
            tmp2 = exp(tmp)
            val = [np.sum(1.0/(1+tmp2), axis=0) - 0.5*N,
                   np.sum(tmp*(1.0-tmp2)/(1+tmp2), axis=0) + N]
            return array(val)

        sol0 = array([xbar, np.std(x, ddof=1, axis=0)])
        sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-5)
        w = (y - sol[0]) / sol[1]
        logcdf = distributions.logistic.logcdf(w)
        logsf = distributions.logistic.logsf(w)
        sig = array([25, 10, 5, 2.5, 1, 0.5])
        critical = around(_Avals_logistic / (1.0 + 0.25/N), 3)
    elif dist == 'gumbel_r':
        xbar, s = distributions.gumbel_r.fit(x)
        w = (y - xbar) / s
        logcdf = distributions.gumbel_r.logcdf(w)
        logsf = distributions.gumbel_r.logsf(w)
        sig = array([25, 10, 5, 2.5, 1])
        critical = around(_Avals_gumbel / (1.0 + 0.2/sqrt(N)), 3)
    else:  # (dist == 'gumbel') or (dist == 'gumbel_l') or (dist == 'extreme1')
        xbar, s = distributions.gumbel_l.fit(x)
        w = (y - xbar) / s
        logcdf = distributions.gumbel_l.logcdf(w)
        logsf = distributions.gumbel_l.logsf(w)
        sig = array([25, 10, 5, 2.5, 1])
        critical = around(_Avals_gumbel / (1.0 + 0.2/sqrt(N)), 3)

    i = arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0)

    return AndersonResult(A2, critical, sig)


def _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N):
    """Compute A2akN equation 7 of Scholz and Stephens.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.

    Returns
    -------
    A2aKN : float
        The A2aKN statistics of Scholz and Stephens 1987.

    """
    A2akN = 0.
    Z_ssorted_left = Z.searchsorted(Zstar, 'left')
    if N == Zstar.size:
        lj = 1.
    else:
        lj = Z.searchsorted(Zstar, 'right') - Z_ssorted_left
    Bj = Z_ssorted_left + lj / 2.
    for i in arange(0, k):
        s = np.sort(samples[i])
        s_ssorted_right = s.searchsorted(Zstar, side='right')
        Mij = s_ssorted_right.astype(float)
        fij = s_ssorted_right - s.searchsorted(Zstar, 'left')
        Mij -= fij / 2.
        inner = lj / float(N) * (N*Mij - Bj*n[i])**2 / (Bj*(N - Bj) - N*lj/4.)
        A2akN += inner.sum() / n[i]
    A2akN *= (N - 1.) / N
    return A2akN


def _anderson_ksamp_right(samples, Z, Zstar, k, n, N):
    """Compute A2akN equation 6 of Scholz & Stephens.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.

    Returns
    -------
    A2KN : float
        The A2KN statistics of Scholz and Stephens 1987.

    """
    A2kN = 0.
    lj = Z.searchsorted(Zstar[:-1], 'right') - Z.searchsorted(Zstar[:-1],
                                                              'left')
    Bj = lj.cumsum()
    for i in arange(0, k):
        s = np.sort(samples[i])
        Mij = s.searchsorted(Zstar[:-1], side='right')
        inner = lj / float(N) * (N * Mij - Bj * n[i])**2 / (Bj * (N - Bj))
        A2kN += inner.sum() / n[i]
    return A2kN


Anderson_ksampResult = namedtuple('Anderson_ksampResult',
                                  ('statistic', 'critical_values',
                                   'significance_level'))


def anderson_ksamp(samples, midrank=True):
    """The Anderson-Darling test for k-samples.

    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample data in arrays.
    midrank : bool, optional
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.

    Returns
    -------
    statistic : float
        Normalized k-sample Anderson-Darling test statistic.
    critical_values : array
        The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
        0.5%, 0.1%.
    significance_level : float
        An approximate significance level at which the null hypothesis for the
        provided samples can be rejected. The value is floored / capped at
        0.1% / 25%.

    Raises
    ------
    ValueError
        If less than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.

    See Also
    --------
    ks_2samp : 2 sample Kolmogorov-Smirnov test
    anderson : 1 sample Anderson-Darling test

    Notes
    -----
    [1]_ defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.

    The critical values corresponding to the significance levels from 0.01
    to 0.25 are taken from [1]_. p-values are floored / capped
    at 0.1% / 25%. Since the range of critical values might be extended in
    future releases, it is recommended not to test ``p == 0.25``, but rather
    ``p >= 0.25`` (analogously for the lower bound).

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.

    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()

    The null hypothesis that the two random samples come from the same
    distribution can be rejected at the 5% level because the returned
    test value is greater than the critical value for 5% (1.961) but
    not at the 2.5% level. The interpolation gives an approximate
    significance level of 3.2%:

    >>> stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(loc=0.5, size=30)])
    (1.974403288713695,
      array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546]),
      0.04991293614572478)


    The null hypothesis cannot be rejected for three samples from an
    identical distribution. The reported p-value (25%) has been capped and
    may not be very accurate (since it corresponds to the value 0.449
    whereas the statistic is -0.731):

    >>> stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(size=30), rng.normal(size=20)])
    (-0.29103725200789504,
      array([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856,
      4.07210043, 5.56419101]),
      0.25)

    """
    k = len(samples)
    if (k < 2):
        raise ValueError("anderson_ksamp needs at least two samples")

    samples = list(map(np.asarray, samples))
    Z = np.sort(np.hstack(samples))
    N = Z.size
    Zstar = np.unique(Z)
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct "
                         "observation")

    n = np.array([sample.size for sample in samples])
    if np.any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without "
                         "observations")

    if midrank:
        A2kN = _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N)
    else:
        A2kN = _anderson_ksamp_right(samples, Z, Zstar, k, n, N)

    H = (1. / n).sum()
    hs_cs = (1. / arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / arange(2, N)).sum()

    a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k**2 - 4*h*k
    sigmasq = (a*N**3 + b*N**2 + c*N + d) / ((N - 1.) * (N - 2.) * (N - 3.))
    m = k - 1
    A2 = (A2kN - m) / math.sqrt(sigmasq)

    # The b_i values are the interpolation coefficients from Table 2
    # of Scholz and Stephens 1987
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    critical = b0 + b1 / math.sqrt(m) + b2 / m

    sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])
    if A2 < critical.min():
        p = sig.max()
        warnings.warn("p-value capped: true value larger than {}".format(p),
                      stacklevel=2)
    elif A2 > critical.max():
        p = sig.min()
        warnings.warn("p-value floored: true value smaller than {}".format(p),
                      stacklevel=2)
    else:
        # interpolation of probit of significance level
        pf = np.polyfit(critical, log(sig), 2)
        p = math.exp(np.polyval(pf, A2))

    return Anderson_ksampResult(A2, critical, p)


AnsariResult = namedtuple('AnsariResult', ('statistic', 'pvalue'))


class _ABW:
    """Distribution of Ansari-Bradley W-statistic under the null hypothesis."""
    # TODO: calculate exact distribution considering ties
    # We could avoid summing over more than half the frequencies,
    # but inititally it doesn't seem worth the extra complexity

    def __init__(self):
        """Minimal initializer."""
        self.m = None
        self.n = None
        self.astart = None
        self.total = None
        self.freqs = None

    def _recalc(self, n, m):
        """When necessary, recalculate exact distribution."""
        if n != self.n or m != self.m:
            self.n, self.m = n, m
            # distribution is NOT symmetric when m + n is odd
            # n is len(x), m is len(y), and ratio of scales is defined x/y
            astart, a1, _ = statlib.gscale(n, m)
            self.astart = astart  # minimum value of statistic
            # Exact distribution of test statistic under null hypothesis
            # expressed as frequencies/counts/integers to maintain precision.
            # Stored as floats to avoid overflow of sums.
            self.freqs = a1.astype(np.float64)
            self.total = self.freqs.sum()  # could calculate from m and n
            # probability mass is self.freqs / self.total;

    def pmf(self, k, n, m):
        """Probability mass function."""
        self._recalc(n, m)
        # The convention here is that PMF at k = 12.5 is the same as at k = 12,
        # -> use `floor` in case of ties.
        ind = np.floor(k - self.astart).astype(int)
        return self.freqs[ind] / self.total

    def cdf(self, k, n, m):
        """Cumulative distribution function."""
        self._recalc(n, m)
        # Null distribution derived without considering ties is
        # approximate. Round down to avoid Type I error.
        ind = np.ceil(k - self.astart).astype(int)
        return self.freqs[:ind+1].sum() / self.total

    def sf(self, k, n, m):
        """Survival function."""
        self._recalc(n, m)
        # Null distribution derived without considering ties is
        # approximate. Round down to avoid Type I error.
        ind = np.floor(k - self.astart).astype(int)
        return self.freqs[ind:].sum() / self.total


# Maintain state for faster repeat calls to ansari w/ method='exact'
_abw_state = _ABW()


def ansari(x, y, alternative='two-sided'):
    """Perform the Ansari-Bradley test for equal scale parameters.

    The Ansari-Bradley test ([1]_, [2]_) is a non-parametric test
    for the equality of the scale parameter of the distributions
    from which two samples were drawn. The null hypothesis states that
    the ratio of the scale of the distribution underlying `x` to the scale
    of the distribution underlying `y` is 1.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the ratio of scales is not equal to 1.
        * 'less': the ratio of scales is less than 1.
        * 'greater': the ratio of scales is greater than 1.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The Ansari-Bradley test statistic.
    pvalue : float
        The p-value of the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    mood : A non-parametric test for the equality of two scale parameters

    Notes
    -----
    The p-value given is exact when the sample sizes are both less than
    55 and there are no ties, otherwise a normal approximation for the
    p-value is used.

    References
    ----------
    .. [1] Ansari, A. R. and Bradley, R. A. (1960) Rank-sum tests for
           dispersions, Annals of Mathematical Statistics, 31, 1174-1189.
    .. [2] Sprent, Peter and N.C. Smeeton.  Applied nonparametric
           statistical methods.  3rd ed. Chapman and Hall/CRC. 2001.
           Section 5.8.2.
    .. [3] Nathaniel E. Helwig "Nonparametric Dispersion and Equality
           Tests" at http://users.stat.umn.edu/~helwig/notes/npde-Notes.pdf

    Examples
    --------
    >>> from scipy.stats import ansari
    >>> rng = np.random.default_rng()

    For these examples, we'll create three random data sets.  The first
    two, with sizes 35 and 25, are drawn from a normal distribution with
    mean 0 and standard deviation 2.  The third data set has size 25 and
    is drawn from a normal distribution with standard deviation 1.25.

    >>> x1 = rng.normal(loc=0, scale=2, size=35)
    >>> x2 = rng.normal(loc=0, scale=2, size=25)
    >>> x3 = rng.normal(loc=0, scale=1.25, size=25)

    First we apply `ansari` to `x1` and `x2`.  These samples are drawn
    from the same distribution, so we expect the Ansari-Bradley test
    should not lead us to conclude that the scales of the distributions
    are different.

    >>> ansari(x1, x2)
    AnsariResult(statistic=541.0, pvalue=0.9762532927399098)

    With a p-value close to 1, we cannot conclude that there is a
    significant difference in the scales (as expected).

    Now apply the test to `x1` and `x3`:

    >>> ansari(x1, x3)
    AnsariResult(statistic=425.0, pvalue=0.0003087020407974518)

    The probability of observing such an extreme value of the statistic
    under the null hypothesis of equal scales is only 0.03087%. We take this
    as evidence against the null hypothesis in favor of the alternative:
    the scales of the distributions from which the samples were drawn
    are not equal.

    We can use the `alternative` parameter to perform a one-tailed test.
    In the above example, the scale of `x1` is greater than `x3` and so
    the ratio of scales of `x1` and `x3` is greater than 1. This means
    that the p-value when ``alternative='greater'`` should be near 0 and
    hence we should be able to reject the null hypothesis:

    >>> ansari(x1, x3, alternative='greater')
    AnsariResult(statistic=425.0, pvalue=0.0001543510203987259)

    As we can see, the p-value is indeed quite low. Use of
    ``alternative='less'`` should thus yield a large p-value:

    >>> ansari(x1, x3, alternative='less')
    AnsariResult(statistic=425.0, pvalue=0.9998643258449039)

    """
    if alternative not in {'two-sided', 'greater', 'less'}:
        raise ValueError("'alternative' must be 'two-sided',"
                         " 'greater', or 'less'.")
    x, y = asarray(x), asarray(y)
    n = len(x)
    m = len(y)
    if m < 1:
        raise ValueError("Not enough other observations.")
    if n < 1:
        raise ValueError("Not enough test observations.")

    N = m + n
    xy = r_[x, y]  # combine
    rank = stats.rankdata(xy)
    symrank = amin(array((rank, N - rank + 1)), 0)
    AB = np.sum(symrank[:n], axis=0)
    uxy = unique(xy)
    repeats = (len(uxy) != len(xy))
    exact = ((m < 55) and (n < 55) and not repeats)
    if repeats and (m < 55 or n < 55):
        warnings.warn("Ties preclude use of exact statistic.")
    if exact:
        if alternative == 'two-sided':
            pval = 2.0 * np.minimum(_abw_state.cdf(AB, n, m),
                                    _abw_state.sf(AB, n, m))
        elif alternative == 'greater':
            # AB statistic is _smaller_ when ratio of scales is larger,
            # so this is the opposite of the usual calculation
            pval = _abw_state.cdf(AB, n, m)
        else:
            pval = _abw_state.sf(AB, n, m)
        return AnsariResult(AB, min(1.0, pval))

    # otherwise compute normal approximation
    if N % 2:  # N odd
        mnAB = n * (N+1.0)**2 / 4.0 / N
        varAB = n * m * (N+1.0) * (3+N**2) / (48.0 * N**2)
    else:
        mnAB = n * (N+2.0) / 4.0
        varAB = m * n * (N+2) * (N-2.0) / 48 / (N-1.0)
    if repeats:   # adjust variance estimates
        # compute np.sum(tj * rj**2,axis=0)
        fac = np.sum(symrank**2, axis=0)
        if N % 2:  # N odd
            varAB = m * n * (16*N*fac - (N+1)**4) / (16.0 * N**2 * (N-1))
        else:  # N even
            varAB = m * n * (16*fac - N*(N+2)**2) / (16.0 * N * (N-1))

    # Small values of AB indicate larger dispersion for the x sample.
    # Large values of AB indicate larger dispersion for the y sample.
    # This is opposite to the way we define the ratio of scales. see [1]_.
    z = (mnAB - AB) / sqrt(varAB)
    z, pval = _normtest_finish(z, alternative)
    return AnsariResult(AB, pval)


BartlettResult = namedtuple('BartlettResult', ('statistic', 'pvalue'))


def bartlett(*args):
    """Perform Bartlett's test for equal variances.

    Bartlett's test tests the null hypothesis that all input samples
    are from populations with equal variances.  For samples
    from significantly non-normal populations, Levene's test
    `levene` is more robust.

    Parameters
    ----------
    sample1, sample2,... : array_like
        arrays of sample data.  Only 1d arrays are accepted, they may have
        different lengths.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    levene : A robust parametric test for equality of k variances

    Notes
    -----
    Conover et al. (1981) examine many of the existing parametric and
    nonparametric tests by extensive simulations and they conclude that the
    tests proposed by Fligner and Killeen (1976) and Levene (1960) appear to be
    superior in terms of robustness of departures from normality and power
    ([3]_).

    References
    ----------
    .. [1]  https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm

    .. [2]  Snedecor, George W. and Cochran, William G. (1989), Statistical
              Methods, Eighth Edition, Iowa State University Press.

    .. [3] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.

    .. [4] Bartlett, M. S. (1937). Properties of Sufficiency and Statistical
           Tests. Proceedings of the Royal Society of London. Series A,
           Mathematical and Physical Sciences, Vol. 160, No.901, pp. 268-282.

    Examples
    --------
    Test whether or not the lists `a`, `b` and `c` come from populations
    with equal variances.

    >>> from scipy.stats import bartlett
    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    >>> stat, p = bartlett(a, b, c)
    >>> p
    1.1254782518834628e-05

    The very small p-value suggests that the populations do not have equal
    variances.

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    """
    # Handle empty input and input that is not 1d
    for a in args:
        if np.asanyarray(a).size == 0:
            return BartlettResult(np.nan, np.nan)
        if np.asanyarray(a).ndim > 1:
            raise ValueError('Samples must be one-dimensional.')

    k = len(args)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")
    Ni = np.empty(k)
    ssq = np.empty(k, 'd')
    for j in range(k):
        Ni[j] = len(args[j])
        ssq[j] = np.var(args[j], ddof=1)
    Ntot = np.sum(Ni, axis=0)
    spsq = np.sum((Ni - 1)*ssq, axis=0) / (1.0*(Ntot - k))
    numer = (Ntot*1.0 - k) * log(spsq) - np.sum((Ni - 1.0)*log(ssq), axis=0)
    denom = 1.0 + 1.0/(3*(k - 1)) * ((np.sum(1.0/(Ni - 1.0), axis=0)) -
                                     1.0/(Ntot - k))
    T = numer / denom
    pval = distributions.chi2.sf(T, k - 1)  # 1 - cdf

    return BartlettResult(T, pval)


LeveneResult = namedtuple('LeveneResult', ('statistic', 'pvalue'))


def levene(*args, center='median', proportiontocut=0.05):
    """Perform Levene test for equal variances.

    The Levene test tests the null hypothesis that all input samples
    are from populations with equal variances.  Levene's test is an
    alternative to Bartlett's test `bartlett` in the case where
    there are significant deviations from normality.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample data, possibly with different lengths. Only one-dimensional
        samples are accepted.
    center : {'mean', 'median', 'trimmed'}, optional
        Which function of the data to use in the test.  The default
        is 'median'.
    proportiontocut : float, optional
        When `center` is 'trimmed', this gives the proportion of data points
        to cut from each end. (See `scipy.stats.trim_mean`.)
        Default is 0.05.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value for the test.

    Notes
    -----
    Three variations of Levene's test are possible.  The possibilities
    and their recommended usages are:

      * 'median' : Recommended for skewed (non-normal) distributions>
      * 'mean' : Recommended for symmetric, moderate-tailed distributions.
      * 'trimmed' : Recommended for heavy-tailed distributions.

    The test version using the mean was proposed in the original article
    of Levene ([2]_) while the median and trimmed mean have been studied by
    Brown and Forsythe ([3]_), sometimes also referred to as Brown-Forsythe
    test.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
    .. [2] Levene, H. (1960). In Contributions to Probability and Statistics:
           Essays in Honor of Harold Hotelling, I. Olkin et al. eds.,
           Stanford University Press, pp. 278-292.
    .. [3] Brown, M. B. and Forsythe, A. B. (1974), Journal of the American
           Statistical Association, 69, 364-367

    Examples
    --------
    Test whether or not the lists `a`, `b` and `c` come from populations
    with equal variances.

    >>> from scipy.stats import levene
    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    >>> stat, p = levene(a, b, c)
    >>> p
    0.002431505967249681

    The small p-value suggests that the populations do not have equal
    variances.

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    """
    if center not in ['mean', 'median', 'trimmed']:
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")

    k = len(args)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")
    # check for 1d input
    for j in range(k):
        if np.asanyarray(args[j]).ndim > 1:
            raise ValueError('Samples must be one-dimensional.')

    Ni = np.empty(k)
    Yci = np.empty(k, 'd')

    if center == 'median':
        func = lambda x: np.median(x, axis=0)
    elif center == 'mean':
        func = lambda x: np.mean(x, axis=0)
    else:  # center == 'trimmed'
        args = tuple(stats.trimboth(np.sort(arg), proportiontocut)
                     for arg in args)
        func = lambda x: np.mean(x, axis=0)

    for j in range(k):
        Ni[j] = len(args[j])
        Yci[j] = func(args[j])
    Ntot = np.sum(Ni, axis=0)

    # compute Zij's
    Zij = [None] * k
    for i in range(k):
        Zij[i] = abs(asarray(args[i]) - Yci[i])

    # compute Zbari
    Zbari = np.empty(k, 'd')
    Zbar = 0.0
    for i in range(k):
        Zbari[i] = np.mean(Zij[i], axis=0)
        Zbar += Zbari[i] * Ni[i]

    Zbar /= Ntot
    numer = (Ntot - k) * np.sum(Ni * (Zbari - Zbar)**2, axis=0)

    # compute denom_variance
    dvar = 0.0
    for i in range(k):
        dvar += np.sum((Zij[i] - Zbari[i])**2, axis=0)

    denom = (k - 1.0) * dvar

    W = numer / denom
    pval = distributions.f.sf(W, k-1, Ntot-k)  # 1 - cdf
    return LeveneResult(W, pval)


def binom_test(x, n=None, p=0.5, alternative='two-sided'):
    """Perform a test that the probability of success is p.

    Note: `binom_test` is deprecated; it is recommended that `binomtest`
    be used instead.

    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.

    Parameters
    ----------
    x : int or array_like
        The number of successes, or if x has length 2, it is the
        number of successes and the number of failures.
    n : int
        The number of trials.  This is ignored if x gives both the
        number of successes and failures.
    p : float, optional
        The hypothesized probability of success.  ``0 <= p <= 1``. The
        default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.

    Returns
    -------
    p-value : float
        The p-value of the hypothesis test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_test

    Examples
    --------
    >>> from scipy import stats

    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:

    >>> stats.binom_test(3, n=15, p=0.1, alternative='greater')
    0.18406106910639114

    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.

    """
    x = atleast_1d(x).astype(np.int_)
    if len(x) == 2:
        n = x[1] + x[0]
        x = x[0]
    elif len(x) == 1:
        x = x[0]
        if n is None or n < x:
            raise ValueError("n must be >= x")
        n = np.int_(n)
    else:
        raise ValueError("Incorrect length for x.")

    if (p > 1.0) or (p < 0.0):
        raise ValueError("p must be in range [0,1]")

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'less':
        pval = distributions.binom.cdf(x, n, p)
        return pval

    if alternative == 'greater':
        pval = distributions.binom.sf(x-1, n, p)
        return pval

    # if alternative was neither 'less' nor 'greater', then it's 'two-sided'
    d = distributions.binom.pmf(x, n, p)
    rerr = 1 + 1e-7
    if x == p * n:
        # special case as shortcut, would also be handled by `else` below
        pval = 1.
    elif x < p * n:
        i = np.arange(np.ceil(p * n), n+1)
        y = np.sum(distributions.binom.pmf(i, n, p) <= d*rerr, axis=0)
        pval = (distributions.binom.cdf(x, n, p) +
                distributions.binom.sf(n - y, n, p))
    else:
        i = np.arange(np.floor(p*n) + 1)
        y = np.sum(distributions.binom.pmf(i, n, p) <= d*rerr, axis=0)
        pval = (distributions.binom.cdf(y-1, n, p) +
                distributions.binom.sf(x-1, n, p))

    return min(1.0, pval)


def _apply_func(x, g, func):
    # g is list of indices into x
    #  separating x into different groups
    #  func should be applied over the groups
    g = unique(r_[0, g, len(x)])
    output = [func(x[g[k]:g[k+1]]) for k in range(len(g) - 1)]

    return asarray(output)


FlignerResult = namedtuple('FlignerResult', ('statistic', 'pvalue'))


def fligner(*args, center='median', proportiontocut=0.05):
    """Perform Fligner-Killeen test for equality of variance.

    Fligner's test tests the null hypothesis that all input samples
    are from populations with equal variances.  Fligner-Killeen's test is
    distribution free when populations are identical [2]_.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        Arrays of sample data.  Need not be the same length.
    center : {'mean', 'median', 'trimmed'}, optional
        Keyword argument controlling which function of the data is used in
        computing the test statistic.  The default is 'median'.
    proportiontocut : float, optional
        When `center` is 'trimmed', this gives the proportion of data points
        to cut from each end. (See `scipy.stats.trim_mean`.)
        Default is 0.05.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value for the hypothesis test.

    See Also
    --------
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A robust parametric test for equality of k variances

    Notes
    -----
    As with Levene's test there are three variants of Fligner's test that
    differ by the measure of central tendency used in the test.  See `levene`
    for more information.

    Conover et al. (1981) examine many of the existing parametric and
    nonparametric tests by extensive simulations and they conclude that the
    tests proposed by Fligner and Killeen (1976) and Levene (1960) appear to be
    superior in terms of robustness of departures from normality and power [3]_.

    References
    ----------
    .. [1] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
           https://cecas.clemson.edu/~cspark/cv/paper/qif/draftqif2.pdf

    .. [2] Fligner, M.A. and Killeen, T.J. (1976). Distribution-free two-sample
           tests for scale. 'Journal of the American Statistical Association.'
           71(353), 210-213.

    .. [3] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.

    .. [4] Conover, W. J., Johnson, M. E. and Johnson M. M. (1981). A
           comparative study of tests for homogeneity of variances, with
           applications to the outer continental shelf biding data.
           Technometrics, 23(4), 351-361.

    Examples
    --------
    Test whether or not the lists `a`, `b` and `c` come from populations
    with equal variances.

    >>> from scipy.stats import fligner
    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    >>> stat, p = fligner(a, b, c)
    >>> p
    0.00450826080004775

    The small p-value suggests that the populations do not have equal
    variances.

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    """
    if center not in ['mean', 'median', 'trimmed']:
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")

    # Handle empty input
    for a in args:
        if np.asanyarray(a).size == 0:
            return FlignerResult(np.nan, np.nan)

    k = len(args)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    if center == 'median':
        func = lambda x: np.median(x, axis=0)
    elif center == 'mean':
        func = lambda x: np.mean(x, axis=0)
    else:  # center == 'trimmed'
        args = tuple(stats.trimboth(arg, proportiontocut) for arg in args)
        func = lambda x: np.mean(x, axis=0)

    Ni = asarray([len(args[j]) for j in range(k)])
    Yci = asarray([func(args[j]) for j in range(k)])
    Ntot = np.sum(Ni, axis=0)
    # compute Zij's
    Zij = [abs(asarray(args[i]) - Yci[i]) for i in range(k)]
    allZij = []
    g = [0]
    for i in range(k):
        allZij.extend(list(Zij[i]))
        g.append(len(allZij))

    ranks = stats.rankdata(allZij)
    a = distributions.norm.ppf(ranks / (2*(Ntot + 1.0)) + 0.5)

    # compute Aibar
    Aibar = _apply_func(a, g, np.sum) / Ni
    anbar = np.mean(a, axis=0)
    varsq = np.var(a, axis=0, ddof=1)
    Xsq = np.sum(Ni * (asarray(Aibar) - anbar)**2.0, axis=0) / varsq
    pval = distributions.chi2.sf(Xsq, k - 1)  # 1 - cdf
    return FlignerResult(Xsq, pval)


def mood(x, y, axis=0, alternative="two-sided"):
    """Perform Mood's test for equal scale parameters.

    Mood's two-sample test for scale parameters is a non-parametric
    test for the null hypothesis that two samples are drawn from the
    same distribution with the same scale parameter.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data.
    axis : int, optional
        The axis along which the samples are tested.  `x` and `y` can be of
        different length along `axis`.
        If `axis` is None, `x` and `y` are flattened and the test is done on
        all values in the flattened arrays.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the scales of the distributions underlying `x` and `y`
          are different.
        * 'less': the scale of the distribution underlying `x` is less than
          the scale of the distribution underlying `y`.
        * 'greater': the scale of the distribution underlying `x` is greater
          than the scale of the distribution underlying `y`.

        .. versionadded:: 1.7.0

    Returns
    -------
    z : scalar or ndarray
        The z-score for the hypothesis test.  For 1-D inputs a scalar is
        returned.
    p-value : scalar ndarray
        The p-value for the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    ansari : A non-parametric test for the equality of 2 variances
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A parametric test for equality of k variances

    Notes
    -----
    The data are assumed to be drawn from probability distributions ``f(x)``
    and ``f(x/s) / s`` respectively, for some probability density function f.
    The null hypothesis is that ``s == 1``.

    For multi-dimensional arrays, if the inputs are of shapes
    ``(n0, n1, n2, n3)``  and ``(n0, m1, n2, n3)``, then if ``axis=1``, the
    resulting z and p values will have shape ``(n0, n2, n3)``.  Note that
    ``n1`` and ``m1`` don't have to be equal, but the other dimensions do.

    Examples
    --------
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x2 = rng.standard_normal((2, 45, 6, 7))
    >>> x1 = rng.standard_normal((2, 30, 6, 7))
    >>> z, p = stats.mood(x1, x2, axis=1)
    >>> p.shape
    (2, 6, 7)

    Find the number of points where the difference in scale is not significant:

    >>> (p > 0.1).sum()
    78

    Perform the test with different scales:

    >>> x1 = rng.standard_normal((2, 30))
    >>> x2 = rng.standard_normal((2, 35)) * 10.0
    >>> stats.mood(x1, x2, axis=1)
    (array([-5.76174136, -6.12650783]), array([8.32505043e-09, 8.98287869e-10]))

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if axis is None:
        x = x.flatten()
        y = y.flatten()
        axis = 0

    if axis < 0:
        axis = x.ndim + axis

    # Determine shape of the result arrays
    res_shape = tuple([x.shape[ax] for ax in range(len(x.shape)) if ax != axis])
    if not (res_shape == tuple([y.shape[ax] for ax in range(len(y.shape)) if
                                ax != axis])):
        raise ValueError("Dimensions of x and y on all axes except `axis` "
                         "should match")

    n = x.shape[axis]
    m = y.shape[axis]
    N = m + n
    if N < 3:
        raise ValueError("Not enough observations.")

    xy = np.concatenate((x, y), axis=axis)
    if axis != 0:
        xy = np.rollaxis(xy, axis)

    xy = xy.reshape(xy.shape[0], -1)

    # Generalized to the n-dimensional case by adding the axis argument, and
    # using for loops, since rankdata is not vectorized.  For improving
    # performance consider vectorizing rankdata function.
    all_ranks = np.empty_like(xy)
    for j in range(xy.shape[1]):
        all_ranks[:, j] = stats.rankdata(xy[:, j])

    Ri = all_ranks[:n]
    M = np.sum((Ri - (N + 1.0) / 2)**2, axis=0)
    # Approx stat.
    mnM = n * (N * N - 1.0) / 12
    varM = m * n * (N + 1.0) * (N + 2) * (N - 2) / 180
    z = (M - mnM) / sqrt(varM)
    z, pval = _normtest_finish(z, alternative)

    if res_shape == ():
        # Return scalars, not 0-D arrays
        z = z[0]
        pval = pval[0]
    else:
        z.shape = res_shape
        pval.shape = res_shape

    return z, pval


WilcoxonResult = namedtuple('WilcoxonResult', ('statistic', 'pvalue'))


def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided", mode='auto'):
    """Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        Either the first set of measurements (in which case ``y`` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case ``y`` is not to be specified.)  Must be
        one-dimensional.
    y : array_like, optional
        Either the second set of measurements (if ``x`` is the first set of
        measurements), or not specified (if ``x`` is the differences between
        two sets of measurements.)  Must be one-dimensional.
    zero_method : {"pratt", "wilcox", "zsplit"}, optional
        The following options are available (default is "wilcox"):

          * "pratt": Includes zero-differences in the ranking process,
            but drops the ranks of the zeros, see [4]_, (more conservative).
          * "wilcox": Discards all zero-differences, the default.
          * "zsplit": Includes zero-differences in the ranking process and
            split the zero rank between positive and negative ones.
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic if a normal approximation is used.  Default is False.
    alternative : {"two-sided", "greater", "less"}, optional
        The alternative hypothesis to be tested, see Notes. Default is
        "two-sided".
    mode : {"auto", "exact", "approx"}
        Method to calculate the p-value, see Notes. Default is "auto".

    Returns
    -------
    statistic : float
        If ``alternative`` is "two-sided", the sum of the ranks of the
        differences above or below zero, whichever is smaller.
        Otherwise the sum of the ranks of the differences above zero.
    pvalue : float
        The p-value for the test depending on ``alternative`` and ``mode``.

    See Also
    --------
    kruskal, mannwhitneyu

    Notes
    -----
    The test has been introduced in [4]_. Given n independent samples
    (xi, yi) from a bivariate distribution (i.e. paired samples),
    it computes the differences di = xi - yi. One assumption of the test
    is that the differences are symmetric, see [2]_.
    The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from
    zero. The one-sided test has the null hypothesis that the median is
    positive against the alternative that it is negative
    (``alternative == 'less'``), or vice versa (``alternative == 'greater.'``).

    To derive the p-value, the exact distribution (``mode == 'exact'``)
    can be used for sample sizes of up to 25. The default ``mode == 'auto'``
    uses the exact distribution if there are at most 25 observations and no
    ties, otherwise a normal approximation is used (``mode == 'approx'``).

    The treatment of ties can be controlled by the parameter `zero_method`.
    If ``zero_method == 'pratt'``, the normal approximation is adjusted as in
    [5]_. A typical rule is to require that n > 20 ([2]_, p. 383).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    .. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
       Sampling Distribution When Zero Differences are Present,
       Journal of the American Statistical Association, Vol. 62, 1967,
       pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`

    Examples
    --------
    In [4]_, the differences in height between cross- and self-fertilized
    corn plants is given as follows:

    >>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

    Cross-fertilized plants appear to be be higher. To test the null
    hypothesis that there is no height difference, we can apply the
    two-sided test:

    >>> from scipy.stats import wilcoxon
    >>> w, p = wilcoxon(d)
    >>> w, p
    (24.0, 0.041259765625)

    Hence, we would reject the null hypothesis at a confidence level of 5%,
    concluding that there is a difference in height between the groups.
    To confirm that the median of the differences can be assumed to be
    positive, we use:

    >>> w, p = wilcoxon(d, alternative='greater')
    >>> w, p
    (96.0, 0.0206298828125)

    This shows that the null hypothesis that the median is negative can be
    rejected at a confidence level of 5% in favor of the alternative that
    the median is greater than zero. The p-values above are exact. Using the
    normal approximation gives very similar values:

    >>> w, p = wilcoxon(d, mode='approx')
    >>> w, p
    (24.0, 0.04088813291185591)

    Note that the statistic changed to 96 in the one-sided case (the sum
    of ranks of positive differences) whereas it is 24 in the two-sided
    case (the minimum of sum of ranks above and below zero).

    """
    if mode not in ["auto", "approx", "exact"]:
        raise ValueError("mode must be either 'auto', 'approx' or 'exact'")

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method must be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("Alternative must be either 'two-sided', "
                         "'greater' or 'less'")

    if y is None:
        d = asarray(x)
        if d.ndim > 1:
            raise ValueError('Sample x must be one-dimensional.')
    else:
        x, y = map(asarray, (x, y))
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError('Samples x and y must be one-dimensional.')
        if len(x) != len(y):
            raise ValueError('The samples x and y must have the same length.')
        d = x - y

    if mode == "auto":
        if len(d) <= 25:
            mode = "exact"
        else:
            mode = "approx"

    n_zero = np.sum(d == 0)
    if n_zero > 0 and mode == "exact":
        mode = "approx"
        warnings.warn("Exact p-value calculation does not work if there are "
                      "ties. Switching to normal approximation.")

    if mode == "approx":
        if zero_method in ["wilcox", "pratt"]:
            if n_zero == len(d):
                raise ValueError("zero_method 'wilcox' and 'pratt' do not "
                                 "work if x - y is zero for all elements.")
        if zero_method == "wilcox":
            # Keep all non-zero differences
            d = compress(np.not_equal(d, 0), d)

    count = len(d)
    if count < 10 and mode == "approx":
        warnings.warn("Sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r)
    r_minus = np.sum((d < 0) * r)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    # return min for two-sided test, but r_plus for one-sided test
    # the literature is not consistent here
    # r_plus is more informative since r_plus + r_minus = count*(count+1)/2,
    # i.e. the sum of the ranks, so r_minus and the min can be inferred
    # (If alternative='pratt', r_plus + r_minus = count*(count+1)/2 - r_zero.)
    # [3] uses the r_plus for the one-sided test, keep min for two-sided test
    # to keep backwards compatibility
    if alternative == "two-sided":
        T = min(r_plus, r_minus)
    else:
        T = r_plus

    if mode == "approx":
        mn = count * (count + 1.) * 0.25
        se = count * (count + 1.) * (2. * count + 1.)

        if zero_method == "pratt":
            r = r[d != 0]
            # normal approximation needs to be adjusted, see Cureton (1967)
            mn -= n_zero * (n_zero + 1.) * 0.25
            se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.)

        replist, repnum = find_repeats(r)
        if repnum.size != 0:
            # Correction for repeated elements.
            se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

        se = sqrt(se / 24)

        # apply continuity correction if applicable
        d = 0
        if correction:
            if alternative == "two-sided":
                d = 0.5 * np.sign(T - mn)
            elif alternative == "less":
                d = -0.5
            else:
                d = 0.5

        # compute statistic and p-value using normal approximation
        z = (T - mn - d) / se
        if alternative == "two-sided":
            prob = 2. * distributions.norm.sf(abs(z))
        elif alternative == "greater":
            # large T = r_plus indicates x is greater than y; i.e.
            # accept alternative in that case and return small p-value (sf)
            prob = distributions.norm.sf(z)
        else:
            prob = distributions.norm.cdf(z)
    elif mode == "exact":
        # get frequencies cnt of the possible positive ranksums r_plus
        cnt = _get_wilcoxon_distr(count)
        # note: r_plus is int (ties not allowed), need int for slices below
        r_plus = int(r_plus)
        if alternative == "two-sided":
            if r_plus == (len(cnt) - 1) // 2:
                # r_plus is the center of the distribution.
                prob = 1.0
            else:
                p_less = np.sum(cnt[:r_plus + 1]) / 2**count
                p_greater = np.sum(cnt[r_plus:]) / 2**count
                prob = 2*min(p_greater, p_less)
        elif alternative == "greater":
            prob = np.sum(cnt[r_plus:]) / 2**count
        else:
            prob = np.sum(cnt[:r_plus + 1]) / 2**count

    return WilcoxonResult(T, prob)


def median_test(*args, ties='below', correction=True, lambda_=1,
                nan_policy='propagate'):
    """Perform a Mood's median test.

    Test that two or more samples come from populations with the same median.

    Let ``n = len(args)`` be the number of samples.  The "grand median" of
    all the data is computed, and a contingency table is formed by
    classifying the values in each sample as being above or below the grand
    median.  The contingency table, along with `correction` and `lambda_`,
    are passed to `scipy.stats.chi2_contingency` to compute the test statistic
    and p-value.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The set of samples.  There must be at least two samples.
        Each sample must be a one-dimensional sequence containing at least
        one value.  The samples are not required to have the same length.
    ties : str, optional
        Determines how values equal to the grand median are classified in
        the contingency table.  The string must be one of::

            "below":
                Values equal to the grand median are counted as "below".
            "above":
                Values equal to the grand median are counted as "above".
            "ignore":
                Values equal to the grand median are not counted.

        The default is "below".
    correction : bool, optional
        If True, *and* there are just two samples, apply Yates' correction
        for continuity when computing the test statistic associated with
        the contingency table.  Default is True.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family to be used instead.  See
        `power_divergence` for details.
        Default is 1 (Pearson's chi-squared statistic).
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    stat : float
        The test statistic.  The statistic that is returned is determined by
        `lambda_`.  The default is Pearson's chi-squared statistic.
    p : float
        The p-value of the test.
    m : float
        The grand median.
    table : ndarray
        The contingency table.  The shape of the table is (2, n), where
        n is the number of samples.  The first row holds the counts of the
        values above the grand median, and the second row holds the counts
        of the values below the grand median.  The table allows further
        analysis with, for example, `scipy.stats.chi2_contingency`, or with
        `scipy.stats.fisher_exact` if there are two samples, without having
        to recompute the table.  If ``nan_policy`` is "propagate" and there
        are nans in the input, the return value for ``table`` is ``None``.

    See Also
    --------
    kruskal : Compute the Kruskal-Wallis H-test for independent samples.
    mannwhitneyu : Computes the Mann-Whitney rank test on samples x and y.

    Notes
    -----
    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Mood, A. M., Introduction to the Theory of Statistics. McGraw-Hill
        (1950), pp. 394-399.
    .. [2] Zar, J. H., Biostatistical Analysis, 5th ed. Prentice Hall (2010).
        See Sections 8.12 and 10.15.

    Examples
    --------
    A biologist runs an experiment in which there are three groups of plants.
    Group 1 has 16 plants, group 2 has 15 plants, and group 3 has 17 plants.
    Each plant produces a number of seeds.  The seed counts for each group
    are::

        Group 1: 10 14 14 18 20 22 24 25 31 31 32 39 43 43 48 49
        Group 2: 28 30 31 33 34 35 36 40 44 55 57 61 91 92 99
        Group 3:  0  3  9 22 23 25 25 33 34 34 40 45 46 48 62 67 84

    The following code applies Mood's median test to these samples.

    >>> g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    >>> g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    >>> g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
    >>> from scipy.stats import median_test
    >>> stat, p, med, tbl = median_test(g1, g2, g3)

    The median is

    >>> med
    34.0

    and the contingency table is

    >>> tbl
    array([[ 5, 10,  7],
           [11,  5, 10]])

    `p` is too large to conclude that the medians are not the same:

    >>> p
    0.12609082774093244

    The "G-test" can be performed by passing ``lambda_="log-likelihood"`` to
    `median_test`.

    >>> g, p, med, tbl = median_test(g1, g2, g3, lambda_="log-likelihood")
    >>> p
    0.12224779737117837

    The median occurs several times in the data, so we'll get a different
    result if, for example, ``ties="above"`` is used:

    >>> stat, p, med, tbl = median_test(g1, g2, g3, ties="above")
    >>> p
    0.063873276069553273

    >>> tbl
    array([[ 5, 11,  9],
           [11,  4,  8]])

    This example demonstrates that if the data set is not large and there
    are values equal to the median, the p-value can be sensitive to the
    choice of `ties`.

    """
    if len(args) < 2:
        raise ValueError('median_test requires two or more samples.')

    ties_options = ['below', 'above', 'ignore']
    if ties not in ties_options:
        raise ValueError("invalid 'ties' option '%s'; 'ties' must be one "
                         "of: %s" % (ties, str(ties_options)[1:-1]))

    data = [np.asarray(arg) for arg in args]

    # Validate the sizes and shapes of the arguments.
    for k, d in enumerate(data):
        if d.size == 0:
            raise ValueError("Sample %d is empty. All samples must "
                             "contain at least one value." % (k + 1))
        if d.ndim != 1:
            raise ValueError("Sample %d has %d dimensions.  All "
                             "samples must be one-dimensional sequences." %
                             (k + 1, d.ndim))

    cdata = np.concatenate(data)
    contains_nan, nan_policy = _contains_nan(cdata, nan_policy)
    if contains_nan and nan_policy == 'propagate':
        return np.nan, np.nan, np.nan, None

    if contains_nan:
        grand_median = np.median(cdata[~np.isnan(cdata)])
    else:
        grand_median = np.median(cdata)
    # When the minimum version of numpy supported by scipy is 1.9.0,
    # the above if/else statement can be replaced by the single line:
    #     grand_median = np.nanmedian(cdata)

    # Create the contingency table.
    table = np.zeros((2, len(data)), dtype=np.int64)
    for k, sample in enumerate(data):
        sample = sample[~np.isnan(sample)]

        nabove = count_nonzero(sample > grand_median)
        nbelow = count_nonzero(sample < grand_median)
        nequal = sample.size - (nabove + nbelow)
        table[0, k] += nabove
        table[1, k] += nbelow
        if ties == "below":
            table[1, k] += nequal
        elif ties == "above":
            table[0, k] += nequal

    # Check that no row or column of the table is all zero.
    # Such a table can not be given to chi2_contingency, because it would have
    # a zero in the table of expected frequencies.
    rowsums = table.sum(axis=1)
    if rowsums[0] == 0:
        raise ValueError("All values are below the grand median (%r)." %
                         grand_median)
    if rowsums[1] == 0:
        raise ValueError("All values are above the grand median (%r)." %
                         grand_median)
    if ties == "ignore":
        # We already checked that each sample has at least one value, but it
        # is possible that all those values equal the grand median.  If `ties`
        # is "ignore", that would result in a column of zeros in `table`.  We
        # check for that case here.
        zero_cols = np.nonzero((table == 0).all(axis=0))[0]
        if len(zero_cols) > 0:
            msg = ("All values in sample %d are equal to the grand "
                   "median (%r), so they are ignored, resulting in an "
                   "empty sample." % (zero_cols[0] + 1, grand_median))
            raise ValueError(msg)

    stat, p, dof, expected = chi2_contingency(table, lambda_=lambda_,
                                              correction=correction)
    return stat, p, grand_median, table


def _circfuncs_common(samples, high, low, nan_policy='propagate'):
    # Ensure samples are array-like and size is not zero
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = sin((samples - low)*2.*pi / (high - low))
    cos_samp = cos((samples - low)*2.*pi / (high - low))

    # Apply the NaN policy
    contains_nan, nan_policy = _contains_nan(samples, nan_policy)
    if contains_nan and nan_policy == 'omit':
        mask = np.isnan(samples)
        # Set the sines and cosines that are NaN to zero
        sin_samp[mask] = 0.0
        cos_samp[mask] = 0.0
    else:
        mask = None

    return samples, sin_samp, cos_samp, mask


def circmean(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    """Compute the circular mean for samples in a range.

    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for circular mean range.  Default is ``2*pi``.
    low : float or int, optional
        Low boundary for circular mean range.  Default is 0.
    axis : int, optional
        Axis along which means are computed.  The default is to compute
        the mean of the flattened array.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    circmean : float
        Circular mean.

    Examples
    --------
    >>> from scipy.stats import circmean
    >>> circmean([0.1, 2*np.pi+0.2, 6*np.pi+0.3])
    0.2

    >>> from scipy.stats import circmean
    >>> circmean([0.2, 1.4, 2.6], high = 1, low = 0)
    0.4

    """
    samples, sin_samp, cos_samp, nmask = _circfuncs_common(samples, high, low,
                                                           nan_policy=nan_policy)
    sin_sum = sin_samp.sum(axis=axis)
    cos_sum = cos_samp.sum(axis=axis)
    res = arctan2(sin_sum, cos_sum)

    mask_nan = ~np.isnan(res)
    if mask_nan.ndim > 0:
        mask = res[mask_nan] < 0
    else:
        mask = res < 0

    if mask.ndim > 0:
        mask_nan[mask_nan] = mask
        res[mask_nan] += 2*pi
    elif mask:
        res += 2*pi

    # Set output to NaN if no samples went into the mean
    if nmask is not None:
        if nmask.all():
            res = np.full(shape=res.shape, fill_value=np.nan)
        else:
            # Find out if any of the axis that are being averaged consist
            # entirely of NaN.  If one exists, set the result (res) to NaN
            nshape = 0 if axis is None else axis
            smask = nmask.shape[nshape] == nmask.sum(axis=axis)
            if smask.any():
                res[smask] = np.nan

    return res*(high - low)/2.0/pi + low


def circvar(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    """Compute the circular variance for samples assumed to be in a range.

    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for circular variance range.  Default is ``2*pi``.
    low : float or int, optional
        Low boundary for circular variance range.  Default is 0.
    axis : int, optional
        Axis along which variances are computed.  The default is to compute
        the variance of the flattened array.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    circvar : float
        Circular variance.

    Notes
    -----
    This uses a definition of circular variance that in the limit of small
    angles returns a number close to the 'linear' variance.

    Examples
    --------
    >>> from scipy.stats import circvar
    >>> circvar([0, 2*np.pi/3, 5*np.pi/3])
    2.19722457734

    """
    samples, sin_samp, cos_samp, mask = _circfuncs_common(samples, high, low,
                                                          nan_policy=nan_policy)
    if mask is None:
        sin_mean = sin_samp.mean(axis=axis)
        cos_mean = cos_samp.mean(axis=axis)
    else:
        nsum = np.asarray(np.sum(~mask, axis=axis).astype(float))
        nsum[nsum == 0] = np.nan
        sin_mean = sin_samp.sum(axis=axis) / nsum
        cos_mean = cos_samp.sum(axis=axis) / nsum
    # hypot can go slightly above 1 due to rounding errors
    with np.errstate(invalid='ignore'):
        R = np.minimum(1, hypot(sin_mean, cos_mean))

    return ((high - low)/2.0/pi)**2 * -2 * log(R)


def circstd(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    """
    Compute the circular standard deviation for samples assumed to be in the
    range [low to high].

    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for circular standard deviation range.
        Default is ``2*pi``.
    low : float or int, optional
        Low boundary for circular standard deviation range.  Default is 0.
    axis : int, optional
        Axis along which standard deviations are computed.  The default is
        to compute the standard deviation of the flattened array.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    circstd : float
        Circular standard deviation.

    Notes
    -----
    This uses a definition of circular standard deviation that in the limit of
    small angles returns a number close to the 'linear' standard deviation.

    Examples
    --------
    >>> from scipy.stats import circstd
    >>> circstd([0, 0.1*np.pi/2, 0.001*np.pi, 0.03*np.pi/2])
    0.063564063306

    """
    samples, sin_samp, cos_samp, mask = _circfuncs_common(samples, high, low,
                                                          nan_policy=nan_policy)
    if mask is None:
        sin_mean = sin_samp.mean(axis=axis)
        cos_mean = cos_samp.mean(axis=axis)
    else:
        nsum = np.asarray(np.sum(~mask, axis=axis).astype(float))
        nsum[nsum == 0] = np.nan
        sin_mean = sin_samp.sum(axis=axis) / nsum
        cos_mean = cos_samp.sum(axis=axis) / nsum
    # hypot can go slightly above 1 due to rounding errors
    with np.errstate(invalid='ignore'):
        R = np.minimum(1, hypot(sin_mean, cos_mean))

    return ((high - low)/2.0/pi) * sqrt(-2*log(R))
