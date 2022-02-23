import numpy as np
from dataclasses import make_dataclass
from collections import namedtuple
from scipy import special
from scipy import stats


def _broadcast_concatenate(x, y, axis):
    '''Broadcast then concatenate arrays, leaving concatenation axis last'''
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    z = np.concatenate((x, y), axis=-1)
    return x, y, z


class _MWU:
    '''Distribution of MWU statistic under the null hypothesis'''
    # Possible improvement: if m and n are small enough, use integer arithmetic

    def __init__(self):
        '''Minimal initializer'''
        self._fmnks = -np.ones((1, 1, 1))

    def pmf(self, k, m, n):
        '''Probability mass function'''
        self._resize_fmnks(m, n, np.max(k))
        # could loop over just the unique elements, but probably not worth
        # the time to find them
        for i in np.ravel(k):
            self._f(m, n, i)
        return self._fmnks[m, n, k] / special.binom(m + n, m)

    def cdf(self, k, m, n):
        '''Cumulative distribution function'''
        # We could use the fact that the distribution is symmetric to avoid
        # summing more than m*n/2 terms, but it might not be worth the
        # overhead. Let's leave that to an improvement.
        pmfs = self.pmf(np.arange(0, np.max(k) + 1), m, n)
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k, m, n):
        '''Survival function'''
        # Use the fact that the distribution is symmetric; i.e.
        # _f(m, n, m*n-k) = _f(m, n, k), and sum from the left
        k = m*n - k
        # Note that both CDF and SF include the PMF at k. The p-value is
        # calculated from the SF and should include the mass at k, so this
        # is desirable
        return self.cdf(k, m, n)

    def _resize_fmnks(self, m, n, k):
        '''If necessary, expand the array that remembers PMF values'''
        # could probably use `np.pad` but I'm not sure it would save code
        shape_old = np.array(self._fmnks.shape)
        shape_new = np.array((m+1, n+1, k+1))
        if np.any(shape_new > shape_old):
            shape = np.maximum(shape_old, shape_new)
            fmnks = -np.ones(shape)             # create the new array
            m0, n0, k0 = shape_old
            fmnks[:m0, :n0, :k0] = self._fmnks  # copy remembered values
            self._fmnks = fmnks

    def _f(self, m, n, k):
        '''Recursive implementation of function of [3] Theorem 2.5'''

        # [3] Theorem 2.5 Line 1
        if k < 0 or m < 0 or n < 0 or k > m*n:
            return 0

        # if already calculated, return the value
        if self._fmnks[m, n, k] >= 0:
            return self._fmnks[m, n, k]

        if k == 0 and m >= 0 and n >= 0:  # [3] Theorem 2.5 Line 2
            fmnk = 1
        else:   # [3] Theorem 2.5 Line 3 / Equation 3
            fmnk = self._f(m-1, n, k-n) + self._f(m, n-1, k)

        self._fmnks[m, n, k] = fmnk  # remember result

        return fmnk


# Maintain state for faster repeat calls to mannwhitneyu w/ method='exact'
_mwu_state = _MWU()


def _tie_term(ranks):
    """Tie correction term"""
    # element i of t is the number of elements sharing rank i
    _, t = np.unique(ranks, return_counts=True, axis=-1)
    return (t**3 - t).sum(axis=-1)


def _get_mwu_z(U, n1, n2, ranks, axis=0, continuity=True):
    '''Standardized MWU statistic'''
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

    # Tie correction according to [2]
    tie_term = np.apply_along_axis(_tie_term, -1, ranks)
    s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))

    # equivalent to using scipy.stats.tiecorrect
    # T = np.apply_along_axis(stats.tiecorrect, -1, ranks)
    # s = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)

    numerator = U - mu

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    if continuity:
        numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z


def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    ''' Input validation and standardization for mannwhitneyu '''
    # Would use np.asarray_chkfinite, but infs are OK
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')

    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')

    alternatives = {"two-sided", "less", "greater"}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')

    methods = {"asymptotic", "exact", "auto"}
    method = method.lower()
    if method not in methods:
        raise ValueError(f'`method` must be one of {methods}.')

    return x, y, use_continuity, alternative, axis_int, method


def _tie_check(xy):
    """Find any ties in data"""
    _, t = np.unique(xy, return_counts=True, axis=-1)
    return np.any(t != 1)


def _mwu_choose_method(n1, n2, xy, method):
    """Choose method 'asymptotic' or 'exact' depending on input size, ties"""

    # if both inputs are large, asymptotic is OK
    if n1 > 8 and n2 > 8:
        return "asymptotic"

    # if there are any ties, asymptotic is preferred
    if np.apply_along_axis(_tie_check, -1, xy).any():
        return "asymptotic"

    return "exact"


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))


def mannwhitneyu(x, y, use_continuity=True, alternative="two-sided",
                 axis=0, method="auto"):
    r'''Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    of difference in location between distributions.

    Parameters
    ----------
    x, y : array-like
        N-d arrays of samples. The arrays must be broadcastable except along
        the dimension given by `axis`.
    use_continuity : bool, optional
            Whether a continuity correction (1/2) should be applied.
            Default is True when `method` is ``'asymptotic'``; has no effect
            otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        Let *F(u)* and *G(u)* be the cumulative distribution functions of the
        distributions underlying `x` and `y`, respectively. Then the following
        alternative hypotheses are available:

        * 'two-sided': the distributions are not equal, i.e. *F(u) ≠ G(u)* for
          at least one *u*.
        * 'less': the distribution underlying `x` is stochastically less
          than the distribution underlying `y`, i.e. *F(u) > G(u)* for all *u*.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`, i.e. *F(u) < G(u)* for all *u*.

        Under a more restrictive set of assumptions, the alternative hypotheses
        can be expressed in terms of the locations of the distributions;
        see [5] section 5.1.
    axis : int, optional
        Axis along which to perform the test. Default is 0.
    method : {'auto', 'asymptotic', 'exact'}, optional
        Selects the method used to calculate the *p*-value.
        Default is 'auto'. The following options are available.

        * ``'asymptotic'``: compares the standardized test statistic
          against the normal distribution, correcting for ties.
        * ``'exact'``: computes the exact *p*-value by comparing the observed
          :math:`U` statistic against the exact distribution of the :math:`U`
          statistic under the null hypothesis. No correction is made for ties.
        * ``'auto'``: chooses ``'exact'`` when the size of one of the samples
          is less than 8 and there are no ties; chooses ``'asymptotic'``
          otherwise.

    Returns
    -------
    res : MannwhitneyuResult
        An object containing attributes:

        statistic : float
            The Mann-Whitney U statistic corresponding with sample `x`. See
            Notes for the test statistic corresponding with sample `y`.
        pvalue : float
            The associated *p*-value for the chosen `alternative`.

    Notes
    -----
    If ``U1`` is the statistic corresponding with sample `x`, then the
    statistic corresponding with sample `y` is
    `U2 = `x.shape[axis] * y.shape[axis] - U1``.

    `mannwhitneyu` is for independent samples. For related / paired samples,
    consider `scipy.stats.wilcoxon`.

    `method` ``'exact'`` is recommended when there are no ties and when either
    sample size is less than 8 [1]_. The implementation follows the recurrence
    relation originally proposed in [1]_ as it is described in [3]_.
    Note that the exact method is *not* corrected for ties, but
    `mannwhitneyu` will not raise errors or warnings if there are ties in the
    data.

    The Mann-Whitney U test is a non-parametric version of the t-test for
    independent samples. When the the means of samples from the populations
    are normally distributed, consider `scipy.stats.ttest_ind`.

    See Also
    --------
    scipy.stats.wilcoxon, scipy.stats.ranksums, scipy.stats.ttest_ind

    References
    ----------
    .. [1] H.B. Mann and D.R. Whitney, "On a test of whether one of two random
           variables is stochastically larger than the other", The Annals of
           Mathematical Statistics, Vol. 18, pp. 50-60, 1947.
    .. [2] Mann-Whitney U Test, Wikipedia,
           http://en.wikipedia.org/wiki/Mann-Whitney_U_test
    .. [3] A. Di Bucchianico, "Combinatorics, computer algebra, and the
           Wilcoxon-Mann-Whitney test", Journal of Statistical Planning and
           Inference, Vol. 79, pp. 349-364, 1999.
    .. [4] Rosie Shier, "Statistics: 2.3 The Mann-Whitney U Test", Mathematics
           Learning Support Centre, 2004.
    .. [5] Michael P. Fay and Michael A. Proschan. "Wilcoxon-Mann-Whitney
           or t-test? On assumptions for hypothesis tests and multiple \
           interpretations of decision rules." Statistics surveys, Vol. 4, pp.
           1-39, 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857732/

    Examples
    --------
    We follow the example from [4]_: nine randomly sampled young adults were
    diagnosed with type II diabetes at the ages below.

    >>> males = [19, 22, 16, 29, 24]
    >>> females = [20, 11, 17, 12]

    We use the Mann-Whitney U test to assess whether there is a statistically
    significant difference in the diagnosis age of males and females.
    The null hypothesis is that the distribution of male diagnosis ages is
    the same as the distribution of female diagnosis ages. We decide
    that a confidence level of 95% is required to reject the null hypothesis
    in favor of the alternative that that the distributions are different.
    Since the number of samples is very small and there are no ties in the
    data, we can compare the observed test statistic against the *exact*
    distribution of the test statistic under the null hypothesis.

    >>> from scipy.stats import mannwhitneyu
    >>> U1, p = mannwhitneyu(males, females, method="exact")
    >>> print(U1)
    17.0

    `mannwhitneyu` always reports the statistic associated with the first
    sample, which, in this case, is males. This agrees with :math:`U_M = 17`
    reported in [4]_. The statistic associated with the second statistic
    can be calculated:

    >>> nx, ny = len(males), len(females)
    >>> U2 = nx*ny - U1
    >>> print(U2)
    3.0

    This agrees with :math:`U_F = 3` reported in [4]_. The two-sided
    *p*-value can be calculated from either statistic, and the value produced
    by `mannwhitneyu` agrees with :math:`p = 0.11` reported in [4]_.

    >>> print(p)
    0.1111111111111111

    The exact distribution of the test statistic is asymptotically normal, so
    the example continues by comparing the exact *p*-value against the
    *p*-value produced using the normal approximation.

    >>> _, pnorm = mannwhitneyu(males, females, method="asymptotic")
    >>> print(pnorm)
    0.11134688653314041

    Here `mannwhitneyu`'s reported *p*-value appears to conflict with the
    value :math:`p = 0.09` given in [4]_. The reason is that [4]_
    does not apply the continuity correction performed by `mannwhitneyu`;
    `mannwhitneyu` reduces the distance between the test statistic and the
    mean :math:`\mu = n_x n_y / 2` by 0.5 to correct for the fact that the
    discrete statistic is being compared against a continuous distribution.
    Here, the :math:`U` statistic used is less than the mean, so we reduce
    the distance by adding 0.5 in the numerator.

    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> U = min(U1, U2)
    >>> N = nx + ny
    >>> z = (U - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (N + 1)/ 12)
    >>> p = 2 * norm.cdf(z)  # use CDF to get p-value from smaller statistic
    >>> print(p)
    0.11134688653314041

    If desired, we can disable the continuity correction to get a result
    that agrees with that reported in [4]_.

    >>> _, pnorm = mannwhitneyu(males, females, use_continuity=False,
    ...                         method="asymptotic")
    >>> print(pnorm)
    0.0864107329737

    Regardless of whether we perform an exact or asymptotic test, the
    probability of the test statistic being as extreme or more extreme by
    chance exceeds 5%, so we do not consider the results statistically
    significant.

    Suppose that, before seeing the data, we had hypothesized that females
    would tend to be diagnosed at a younger age than males.
    In that case, it would be natural to provide the female ages as the
    first input, and we would have performed a one-sided test using
    ``alternative = 'less'``: females are diagnosed at an age that is
    stochastically less than that of males.

    >>> res = mannwhitneyu(females, males, alternative="less", method="exact")
    >>> print(res)
    MannwhitneyuResult(statistic=3.0, pvalue=0.05555555555555555)

    Again, the probability of getting a sufficiently low value of the
    test statistic by chance under the null hypothesis is greater than 5%,
    so we do not reject the null hypothesis in favor of our alternative.

    If it is reasonable to assume that the means of samples from the
    populations are normally distributed, we could have used a t-test to
    perform the analysis.

    >>> from scipy.stats import ttest_ind
    >>> res = ttest_ind(females, males, alternative="less")
    >>> print(res)
    Ttest_indResult(statistic=-2.239334696520584, pvalue=0.030068441095757924)

    Under this assumption, the *p*-value would be low enough to reject the
    null hypothesis in favor of the alternative.

    '''

    x, y, use_continuity, alternative, axis_int, method = (
        _mwu_input_validation(x, y, use_continuity, alternative, axis, method))

    x, y, xy = _broadcast_concatenate(x, y, axis)

    n1, n2 = x.shape[-1], y.shape[-1]

    if method == "auto":
        method = _mwu_choose_method(n1, n2, xy, method)

    # Follows [2]
    ranks = stats.rankdata(xy, axis=-1)  # method 2, step 1
    R1 = ranks[..., :n1].sum(axis=-1)    # method 2, step 2
    U1 = R1 - n1*(n1+1)/2                # method 2, step 3
    U2 = n1 * n2 - U1                    # as U1 + U2 = n1 * n2

    if alternative == "greater":
        U, f = U1, 1  # U is the statistic to use for p-value, f is a factor
    elif alternative == "less":
        U, f = U2, 1  # Due to symmetry, use SF of U2 rather than CDF of U1
    else:
        U, f = np.maximum(U1, U2), 2  # multiply SF by two for two-sided test

    if method == "exact":
        p = _mwu_state.sf(U.astype(int), n1, n2)
    elif method == "asymptotic":
        z = _get_mwu_z(U, n1, n2, ranks, continuity=use_continuity)
        p = stats.norm.sf(z)
    p *= f

    # Ensure that test statistic is not greater than 1
    # This could happen for exact test when U = m*n/2
    p = np.clip(p, 0, 1)

    return MannwhitneyuResult(U1, p)
