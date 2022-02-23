from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval


class BinomTestResult:
    """
    Result of `scipy.stats.binomtest`.

    Attributes
    ----------
    k : int
        The number of successes (copied from `binomtest` input).
    n : int
        The number of trials (copied from `binomtest` input).
    alternative : str
        Indicates the alternative hypothesis specified in the input
        to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
        or ``'less'``.
    pvalue : float
        The p-value of the hypothesis test.
    proportion_estimate : float
        The estimate of the proportion of successes.

    """
    def __init__(self, k, n, alternative, pvalue, proportion_estimate):
        self.k = k
        self.n = n
        self.alternative = alternative
        self.proportion_estimate = proportion_estimate
        self.pvalue = pvalue

    def __repr__(self):
        s = ("BinomTestResult("
             f"k={self.k}, "
             f"n={self.n}, "
             f"alternative={self.alternative!r}, "
             f"proportion_estimate={self.proportion_estimate}, "
             f"pvalue={self.pvalue})")
        return s

    def proportion_ci(self, confidence_level=0.95, method='exact'):
        """
        Compute the confidence interval for the estimated proportion.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is 0.95.
        method : {'exact', 'wilson', 'wilsoncc'}, optional
            Selects the method used to compute the confidence interval
            for the estimate of the proportion:

            'exact' :
                Use the Clopper-Pearson exact method [1]_.
            'wilson' :
                Wilson's method, without continuity correction ([2]_, [3]_).
            'wilsoncc' :
                Wilson's method, with continuity correction ([2]_, [3]_).

            Default is ``'exact'``.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence interval.

        References
        ----------
        .. [1] C. J. Clopper and E. S. Pearson, The use of confidence or
               fiducial limits illustrated in the case of the binomial,
               Biometrika, Vol. 26, No. 4, pp 404-413 (Dec. 1934).
        .. [2] E. B. Wilson, Probable inference, the law of succession, and
               statistical inference, J. Amer. Stat. Assoc., 22, pp 209-212
               (1927).
        .. [3] Robert G. Newcombe, Two-sided confidence intervals for the
               single proportion: comparison of seven methods, Statistics
               in Medicine, 17, pp 857-872 (1998).

        Examples
        --------
        >>> from scipy.stats import binomtest
        >>> result = binomtest(k=7, n=50, p=0.1)
        >>> result.proportion_estimate
        0.14
        >>> result.proportion_ci()
        ConfidenceInterval(low=0.05819170033997342, high=0.26739600249700846)
        """
        if method not in ('exact', 'wilson', 'wilsoncc'):
            raise ValueError("method must be one of 'exact', 'wilson' or "
                             "'wilsoncc'.")
        if not (0 <= confidence_level <= 1):
            raise ValueError('confidence_level must be in the interval '
                             '[0, 1].')
        if method == 'exact':
            low, high = _binom_exact_conf_int(self.k, self.n,
                                              confidence_level,
                                              self.alternative)
        else:
            # method is 'wilson' or 'wilsoncc'
            low, high = _binom_wilson_conf_int(self.k, self.n,
                                               confidence_level,
                                               self.alternative,
                                               correction=method == 'wilsoncc')
        return ConfidenceInterval(low=low, high=high)


def _findp(func):
    try:
        p = brentq(func, 0, 1)
    except RuntimeError:
        raise RuntimeError('numerical solver failed to converge when '
                           'computing the confidence limits') from None
    except ValueError as exc:
        raise ValueError('brentq raised a ValueError; report this to the '
                         'SciPy developers') from exc
    return p


def _binom_exact_conf_int(k, n, confidence_level, alternative):
    """
    Compute the estimate and confidence interval for the binomial test.

    Returns proportion, prop_low, prop_high
    """
    if alternative == 'two-sided':
        alpha = (1 - confidence_level) / 2
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'less':
        alpha = 1 - confidence_level
        plow = 0.0
        if k == n:
            phigh = 1.0
        else:
            phigh = _findp(lambda p: binom.cdf(k, n, p) - alpha)
    elif alternative == 'greater':
        alpha = 1 - confidence_level
        if k == 0:
            plow = 0.0
        else:
            plow = _findp(lambda p: binom.sf(k-1, n, p) - alpha)
        phigh = 1.0
    return plow, phigh


def _binom_wilson_conf_int(k, n, confidence_level, alternative, correction):
    # This function assumes that the arguments have already been validated.
    # In particular, `alternative` must be one of 'two-sided', 'less' or
    # 'greater'.
    p = k / n
    if alternative == 'two-sided':
        z = ndtri(0.5 + 0.5*confidence_level)
    else:
        z = ndtri(confidence_level)

    # For reference, the formulas implemented here are from
    # Newcombe (1998) (ref. [3] in the proportion_ci docstring).
    denom = 2*(n + z**2)
    center = (2*n*p + z**2)/denom
    q = 1 - p
    if correction:
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            dlo = (1 + z*sqrt(z**2 - 2 - 1/n + 4*p*(n*q + 1))) / denom
            lo = center - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            dhi = (1 + z*sqrt(z**2 + 2 - 1/n + 4*p*(n*q - 1))) / denom
            hi = center + dhi
    else:
        delta = z/denom * sqrt(4*n*p*q + z**2)
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = center - delta
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = center + delta

    return lo, hi


def binomtest(k, n, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    The binomial test [1]_ is a test of the null hypothesis that the
    probability of success in a Bernoulli experiment is `p`.

    Details of the test can be found in many texts on statistics, such
    as section 24.5 of [2]_.

    Parameters
    ----------
    k : int
        The number of successes.
    n : int
        The number of trials.
    p : float, optional
        The hypothesized probability of success, i.e. the expected
        proportion of successes.  The value must be in the interval
        ``0 <= p <= 1``. The default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.

    Returns
    -------
    result : `~scipy.stats._result_classes.BinomTestResult` instance
        The return value is an object with the following attributes:

        k : int
            The number of successes (copied from `binomtest` input).
        n : int
            The number of trials (copied from `binomtest` input).
        alternative : str
            Indicates the alternative hypothesis specified in the input
            to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
            or ``'less'``.
        pvalue : float
            The p-value of the hypothesis test.
        proportion_estimate : float
            The estimate of the proportion of successes.

        The object has the following methods:

        proportion_ci(confidence_level=0.95, method='exact') :
            Compute the confidence interval for ``proportion_estimate``.

    Notes
    -----
    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Binomial test, https://en.wikipedia.org/wiki/Binomial_test
    .. [2] Jerrold H. Zar, Biostatistical Analysis (fifth edition),
           Prentice Hall, Upper Saddle River, New Jersey USA (2010)

    Examples
    --------
    >>> from scipy.stats import binomtest

    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:

    >>> result = binomtest(3, n=15, p=0.1, alternative='greater')
    >>> result.pvalue
    0.18406106910639114

    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.

    The estimated proportion is simply ``3/15``:

    >>> result.proportion_estimate
    0.2

    We can use the `proportion_ci()` method of the result to compute the
    confidence interval of the estimate:

    >>> result.proportion_ci(confidence_level=0.95)
    ConfidenceInterval(low=0.05684686759024681, high=1.0)

    """
    k = _validate_int(k, 'k', minimum=0)
    n = _validate_int(n, 'n', minimum=1)
    if k > n:
        raise ValueError('k must not be greater than n.')

    if not (0 <= p <= 1):
        raise ValueError("p must be in range [0,1]")

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized; \n"
                         "must be 'two-sided', 'less' or 'greater'")
    if alternative == 'less':
        pval = binom.cdf(k, n, p)
    elif alternative == 'greater':
        pval = binom.sf(k-1, n, p)
    else:
        # alternative is 'two-sided'
        d = binom.pmf(k, n, p)
        rerr = 1 + 1e-7
        if k == p * n:
            # special case as shortcut, would also be handled by `else` below
            pval = 1.
        elif k < p * n:
            ix = _binary_search_for_binom_tst(lambda x1: -binom.pmf(x1, n, p),
                                              -d*rerr, np.ceil(p * n), n)
            # y is the number of terms between mode and n that are <= d*rerr.
            # ix gave us the first term where a(ix) <= d*rerr < a(ix-1)
            # if the first equality doesn't hold, y=n-ix. Otherwise, we
            # need to include ix as well as the equality holds. Note that
            # the equality will hold in very very rare situations due to rerr.
            y = n - ix + int(d*rerr == binom.pmf(ix, n, p))
            pval = binom.cdf(k, n, p) + binom.sf(n - y, n, p)
        else:
            ix = _binary_search_for_binom_tst(lambda x1: binom.pmf(x1, n, p),
                                              d*rerr, 0, np.floor(p * n))
            # y is the number of terms between 0 and mode that are <= d*rerr.
            # we need to add a 1 to account for the 0 index.
            # For comparing this with old behavior, see
            # tst_binary_srch_for_binom_tst method in test_morestats.
            y = ix + 1
            pval = binom.cdf(y-1, n, p) + binom.sf(k-1, n, p)

        pval = min(1.0, pval)

    result = BinomTestResult(k=k, n=n, alternative=alternative,
                             proportion_estimate=k/n, pvalue=pval)
    return result


def _binary_search_for_binom_tst(a, d, lo, hi):
    """
    Conducts an implicit binary search on a function specified by `a`.

    Meant to be used on the binomial PMF for the case of two-sided tests
    to obtain the value on the other side of the mode where the tail
    probability should be computed. The values on either side of
    the mode are always in order, meaning binary search is applicable.

    Parameters
    ----------
    a : callable
      The function over which to perform binary search. Its values
      for inputs lo and hi should be in ascending order.
    d : float
      The value to search.
    lo : int
      The lower end of range to search.
    hi : int
      The higher end of the range to search.

    Returns
    ----------
    int
      The index, i between lo and hi
      such that a(i)<=d<a(i+1)
    """
    while lo < hi:
        mid = lo + (hi-lo)//2
        midval = a(mid)
        if midval < d:
            lo = mid+1
        elif midval > d:
            hi = mid-1
        else:
            return mid
    if a(lo) <= d:
        return lo
    else:
        return lo-1
