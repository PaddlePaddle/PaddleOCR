from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import make_dataclass


PageTrendTestResult = make_dataclass("PageTrendTestResult",
                                     ("statistic", "pvalue", "method"))


def page_trend_test(data, ranked=False, predicted_ranks=None, method='auto'):
    r"""
    Perform Page's Test, a measure of trend in observations between treatments.

    Page's Test (also known as Page's :math:`L` test) is useful when:

    * there are :math:`n \geq 3` treatments,
    * :math:`m \geq 2` subjects are observed for each treatment, and
    * the observations are hypothesized to have a particular order.

    Specifically, the test considers the null hypothesis that

    .. math::

        m_1 = m_2 = m_3 \cdots = m_n,

    where :math:`m_j` is the mean of the observed quantity under treatment
    :math:`j`, against the alternative hypothesis that

    .. math::

        m_1 \leq m_2 \leq m_3 \leq \cdots \leq m_n,

    where at least one inequality is strict.

    As noted by [4]_, Page's :math:`L` test has greater statistical power than
    the Friedman test against the alternative that there is a difference in
    trend, as Friedman's test only considers a difference in the means of the
    observations without considering their order. Whereas Spearman :math:`\rho`
    considers the correlation between the ranked observations of two variables
    (e.g. the airspeed velocity of a swallow vs. the weight of the coconut it
    carries), Page's :math:`L` is concerned with a trend in an observation
    (e.g. the airspeed velocity of a swallow) across several distinct
    treatments (e.g. carrying each of five coconuts of different weight) even
    as the observation is repeated with multiple subjects (e.g. one European
    swallow and one African swallow).

    Parameters
    ----------
    data : array-like
        A :math:`m \times n` array; the element in row :math:`i` and
        column :math:`j` is the observation corresponding with subject
        :math:`i` and treatment :math:`j`. By default, the columns are
        assumed to be arranged in order of increasing predicted mean.

    ranked : boolean, optional
        By default, `data` is assumed to be observations rather than ranks;
        it will be ranked with `scipy.stats.rankdata` along ``axis=1``. If
        `data` is provided in the form of ranks, pass argument ``True``.

    predicted_ranks : array-like, optional
        The predicted ranks of the column means. If not specified,
        the columns are assumed to be arranged in order of increasing
        predicted mean, so the default `predicted_ranks` are
        :math:`[1, 2, \dots, n-1, n]`.

    method : {'auto', 'asymptotic', 'exact'}, optional
        Selects the method used to calculate the *p*-value. The following
        options are available.

        * 'auto': selects between 'exact' and 'asymptotic' to
          achieve reasonably accurate results in reasonable time (default)
        * 'asymptotic': compares the standardized test statistic against
          the normal distribution
        * 'exact': computes the exact *p*-value by comparing the observed
          :math:`L` statistic against those realized by all possible
          permutations of ranks (under the null hypothesis that each
          permutation is equally likely)

    Returns
    -------
    res : PageTrendTestResult
        An object containing attributes:

        statistic : float
            Page's :math:`L` test statistic.
        pvalue : float
            The associated *p*-value
        method : {'asymptotic', 'exact'}
            The method used to compute the *p*-value

    See Also
    --------
    rankdata, friedmanchisquare, spearmanr

    Notes
    -----
    As noted in [1]_, "the :math:`n` 'treatments' could just as well represent
    :math:`n` objects or events or performances or persons or trials ranked."
    Similarly, the :math:`m` 'subjects' could equally stand for :math:`m`
    "groupings by ability or some other control variable, or judges doing
    the ranking, or random replications of some other sort."

    The procedure for calculating the :math:`L` statistic, adapted from
    [1]_, is:

    1. "Predetermine with careful logic the appropriate hypotheses
       concerning the predicted ording of the experimental results.
       If no reasonable basis for ordering any treatments is known, the
       :math:`L` test is not appropriate."
    2. "As in other experiments, determine at what level of confidence
       you will reject the null hypothesis that there is no agreement of
       experimental results with the monotonic hypothesis."
    3. "Cast the experimental material into a two-way table of :math:`n`
       columns (treatments, objects ranked, conditions) and :math:`m`
       rows (subjects, replication groups, levels of control variables)."
    4. "When experimental observations are recorded, rank them across each
       row", e.g. ``ranks = scipy.stats.rankdata(data, axis=1)``.
    5. "Add the ranks in each column", e.g.
       ``colsums = np.sum(ranks, axis=0)``.
    6. "Multiply each sum of ranks by the predicted rank for that same
       column", e.g. ``products = predicted_ranks * colsums``.
    7. "Sum all such products", e.g. ``L = products.sum()``.

    [1]_ continues by suggesting use of the standardized statistic

    .. math::

        \chi_L^2 = \frac{\left[12L-3mn(n+1)^2\right]^2}{mn^2(n^2-1)(n+1)}

    "which is distributed approximately as chi-square with 1 degree of
    freedom. The ordinary use of :math:`\chi^2` tables would be
    equivalent to a two-sided test of agreement. If a one-sided test
    is desired, *as will almost always be the case*, the probability
    discovered in the chi-square table should be *halved*."

    However, this standardized statistic does not distinguish between the
    observed values being well correlated with the predicted ranks and being
    _anti_-correlated with the predicted ranks. Instead, we follow [2]_
    and calculate the standardized statistic

    .. math::

        \Lambda = \frac{L - E_0}{\sqrt{V_0}},

    where :math:`E_0 = \frac{1}{4} mn(n+1)^2` and
    :math:`V_0 = \frac{1}{144} mn^2(n+1)(n^2-1)`, "which is asymptotically
    normal under the null hypothesis".

    The *p*-value for ``method='exact'`` is generated by comparing the observed
    value of :math:`L` against the :math:`L` values generated for all
    :math:`(n!)^m` possible permutations of ranks. The calculation is performed
    using the recursive method of [5].

    The *p*-values are not adjusted for the possibility of ties. When
    ties are present, the reported  ``'exact'`` *p*-values may be somewhat
    larger (i.e. more conservative) than the true *p*-value [2]_. The
    ``'asymptotic'``` *p*-values, however, tend to be smaller (i.e. less
    conservative) than the ``'exact'`` *p*-values.

    References
    ----------
    .. [1] Ellis Batten Page, "Ordered hypotheses for multiple treatments:
       a significant test for linear ranks", *Journal of the American
       Statistical Association* 58(301), p. 216--230, 1963.

    .. [2] Markus Neuhauser, *Nonparametric Statistical Test: A computational
       approach*, CRC Press, p. 150--152, 2012.

    .. [3] Statext LLC, "Page's L Trend Test - Easy Statistics", *Statext -
       Statistics Study*, https://www.statext.com/practice/PageTrendTest03.php,
       Accessed July 12, 2020.

    .. [4] "Page's Trend Test", *Wikipedia*, WikimediaFoundation,
       https://en.wikipedia.org/wiki/Page%27s_trend_test,
       Accessed July 12, 2020.

    .. [5] Robert E. Odeh, "The exact distribution of Page's L-statistic in
       the two-way layout", *Communications in Statistics - Simulation and
       Computation*,  6(1), p. 49--61, 1977.

    Examples
    --------
    We use the example from [3]_: 10 students are asked to rate three
    teaching methods - tutorial, lecture, and seminar - on a scale of 1-5,
    with 1 being the lowest and 5 being the highest. We have decided that
    a confidence level of 99% is required to reject the null hypothesis in
    favor of our alternative: that the seminar will have the highest ratings
    and the tutorial will have the lowest. Initially, the data have been
    tabulated with each row representing an individual student's ratings of
    the three methods in the following order: tutorial, lecture, seminar.

    >>> table = [[3, 4, 3],
    ...          [2, 2, 4],
    ...          [3, 3, 5],
    ...          [1, 3, 2],
    ...          [2, 3, 2],
    ...          [2, 4, 5],
    ...          [1, 2, 4],
    ...          [3, 4, 4],
    ...          [2, 4, 5],
    ...          [1, 3, 4]]

    Because the tutorial is hypothesized to have the lowest ratings, the
    column corresponding with tutorial rankings should be first; the seminar
    is hypothesized to have the highest ratings, so its column should be last.
    Since the columns are already arranged in this order of increasing
    predicted mean, we can pass the table directly into `page_trend_test`.

    >>> from scipy.stats import page_trend_test
    >>> res = page_trend_test(table)
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    This *p*-value indicates that there is a 0.1819% chance that
    the :math:`L` statistic would reach such an extreme value under the null
    hypothesis. Because 0.1819% is less than 1%, we have evidence to reject
    the null hypothesis in favor of our alternative at a 99% confidence level.

    The value of the :math:`L` statistic is 133.5. To check this manually,
    we rank the data such that high scores correspond with high ranks, settling
    ties with an average rank:

    >>> from scipy.stats import rankdata
    >>> ranks = rankdata(table, axis=1)
    >>> ranks
    array([[1.5, 3. , 1.5],
           [1.5, 1.5, 3. ],
           [1.5, 1.5, 3. ],
           [1. , 3. , 2. ],
           [1.5, 3. , 1.5],
           [1. , 2. , 3. ],
           [1. , 2. , 3. ],
           [1. , 2.5, 2.5],
           [1. , 2. , 3. ],
           [1. , 2. , 3. ]])

    We add the ranks within each column, multiply the sums by the
    predicted ranks, and sum the products.

    >>> import numpy as np
    >>> m, n = ranks.shape
    >>> predicted_ranks = np.arange(1, n+1)
    >>> L = (predicted_ranks * np.sum(ranks, axis=0)).sum()
    >>> res.statistic == L
    True

    As presented in [3]_, the asymptotic approximation of the *p*-value is the
    survival function of the normal distribution evaluated at the standardized
    test statistic:

    >>> from scipy.stats import norm
    >>> E0 = (m*n*(n+1)**2)/4
    >>> V0 = (m*n**2*(n+1)*(n**2-1))/144
    >>> Lambda = (L-E0)/np.sqrt(V0)
    >>> p = norm.sf(Lambda)
    >>> p
    0.0012693433690751756

    This does not precisely match the *p*-value reported by `page_trend_test`
    above. The asymptotic distribution is not very accurate, nor conservative,
    for :math:`m \leq 12` and :math:`n \leq 8`, so `page_trend_test` chose to
    use ``method='exact'`` based on the dimensions of the table and the
    recommendations in Page's original paper [1]_. To override
    `page_trend_test`'s choice, provide the `method` argument.

    >>> res = page_trend_test(table, method="asymptotic")
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0012693433690751756,
                        method='asymptotic')

    If the data are already ranked, we can pass in the ``ranks`` instead of
    the ``table`` to save computation time.

    >>> res = page_trend_test(ranks,             # ranks of data
    ...                       ranked=True,       # data is already ranked
    ...                       )
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    Suppose the raw data had been tabulated in an order different from the
    order of predicted means, say lecture, seminar, tutorial.

    >>> table = np.asarray(table)[:, [1, 2, 0]]

    Since the arrangement of this table is not consistent with the assumed
    ordering, we can either rearrange the table or provide the
    `predicted_ranks`. Remembering that the lecture is predicted
    to have the middle rank, the seminar the highest, and tutorial the lowest,
    we pass:

    >>> res = page_trend_test(table,             # data as originally tabulated
    ...                       predicted_ranks=[2, 3, 1],  # our predicted order
    ...                       )
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    """

    # Possible values of the method parameter and the corresponding function
    # used to evaluate the p value
    methods = {"asymptotic": _l_p_asymptotic,
               "exact": _l_p_exact,
               "auto": None}
    if method not in methods:
        raise ValueError(f"`method` must be in {set(methods)}")

    ranks = np.array(data, copy=False)
    if ranks.ndim != 2:  # TODO: relax this to accept 3d arrays?
        raise ValueError("`data` must be a 2d array.")

    m, n = ranks.shape
    if m < 2 or n < 3:
        raise ValueError("Page's L is only appropriate for data with two "
                         "or more rows and three or more columns.")

    if np.any(np.isnan(data)):
        raise ValueError("`data` contains NaNs, which cannot be ranked "
                         "meaningfully")

    # ensure NumPy array and rank the data if it's not already ranked
    if ranked:
        # Only a basic check on whether data is ranked. Checking that the data
        # is properly ranked could take as much time as ranking it.
        if not (ranks.min() >= 1 and ranks.max() <= ranks.shape[1]):
            raise ValueError("`data` is not properly ranked. Rank the data or "
                             "pass `ranked=False`.")
    else:
        ranks = scipy.stats.rankdata(data, axis=-1)

    # generate predicted ranks if not provided, ensure valid NumPy array
    if predicted_ranks is None:
        predicted_ranks = np.arange(1, n+1)
    else:
        predicted_ranks = np.array(predicted_ranks, copy=False)
        if (predicted_ranks.ndim < 1 or
                (set(predicted_ranks) != set(range(1, n+1)) or
                 len(predicted_ranks) != n)):
            raise ValueError(f"`predicted_ranks` must include each integer "
                             f"from 1 to {n} (the number of columns in "
                             f"`data`) exactly once.")

    if type(ranked) is not bool:
        raise TypeError("`ranked` must be boolean.")

    # Calculate the L statistic
    L = _l_vectorized(ranks, predicted_ranks)

    # Calculate the p-value
    if method == "auto":
        method = _choose_method(ranks)
    p_fun = methods[method]  # get the function corresponding with the method
    p = p_fun(L, m, n)

    page_result = PageTrendTestResult(statistic=L, pvalue=p, method=method)
    return page_result


def _choose_method(ranks):
    '''Choose method for computing p-value automatically'''
    m, n = ranks.shape
    if n > 8 or (m > 12 and n > 3) or m > 20:  # as in [1], [4]
        method = "asymptotic"
    else:
        method = "exact"
    return method


def _l_vectorized(ranks, predicted_ranks):
    '''Calculate's Page's L statistic for each page of a 3d array'''
    colsums = ranks.sum(axis=-2, keepdims=True)
    products = predicted_ranks * colsums
    Ls = products.sum(axis=-1)
    Ls = Ls[0] if Ls.size == 1 else Ls.ravel()
    return Ls


def _l_p_asymptotic(L, m, n):
    '''Calculate the p-value of Page's L from the asymptotic distribution'''
    # Using [1] as a reference, the asymptotic p-value would be calculated as:
    # chi_L = (12*L - 3*m*n*(n+1)**2)**2/(m*n**2*(n**2-1)*(n+1))
    # p = chi2.sf(chi_L, df=1, loc=0, scale=1)/2
    # but this is insentive to the direction of the hypothesized ranking

    # See [2] page 151
    E0 = (m*n*(n+1)**2)/4
    V0 = (m*n**2*(n+1)*(n**2-1))/144
    Lambda = (L-E0)/np.sqrt(V0)
    # This is a one-sided "greater" test - calculate the probability that the
    # L statistic under H0 would be greater than the observed L statistic
    p = norm.sf(Lambda)
    return p


def _l_p_exact(L, m, n):
    '''Calculate the p-value of Page's L exactly'''
    # [1] uses m, n; [5] uses n, k.
    # Switch convention here because exact calculation code references [5].
    L, n, k = int(L), int(m), int(n)
    _pagel_state.set_k(k)
    return _pagel_state.sf(L, n)


class _PageL:
    '''Maintains state between `page_trend_test` executions'''

    def __init__(self):
        '''Lightweight initialization'''
        self.all_pmfs = {}

    def set_k(self, k):
        '''Calculate lower and upper limits of L for single row'''
        self.k = k
        # See [5] top of page 52
        self.a, self.b = (k*(k+1)*(k+2))//6, (k*(k+1)*(2*k+1))//6

    def sf(self, l, n):
        '''Survival function of Page's L statistic'''
        ps = [self.pmf(l, n) for l in range(l, n*self.b + 1)]
        return np.sum(ps)

    def p_l_k_1(self):
        '''Relative frequency of each L value over all possible single rows'''

        # See [5] Equation (6)
        ranks = range(1, self.k+1)
        # generate all possible rows of length k
        rank_perms = np.array(list(permutations(ranks)))
        # compute Page's L for all possible rows
        Ls = (ranks*rank_perms).sum(axis=1)
        # count occurences of each L value
        counts = np.histogram(Ls, np.arange(self.a-0.5, self.b+1.5))[0]
        # factorial(k) is number of possible permutations
        return counts/math.factorial(self.k)

    def pmf(self, l, n):
        '''Recursive function to evaluate p(l, k, n); see [5] Equation 1'''

        if n not in self.all_pmfs:
            self.all_pmfs[n] = {}
        if self.k not in self.all_pmfs[n]:
            self.all_pmfs[n][self.k] = {}

        # Cache results to avoid repeating calculation. Initially this was
        # written with lru_cache, but this seems faster? Also, we could add
        # an option to save this for future lookup.
        if l in self.all_pmfs[n][self.k]:
            return self.all_pmfs[n][self.k][l]

        if n == 1:
            ps = self.p_l_k_1()  # [5] Equation 6
            ls = range(self.a, self.b+1)
            # not fast, but we'll only be here once
            self.all_pmfs[n][self.k] = {l: p for l, p in zip(ls, ps)}
            return self.all_pmfs[n][self.k][l]

        p = 0
        low = max(l-(n-1)*self.b, self.a)  # [5] Equation 2
        high = min(l-(n-1)*self.a, self.b)

        # [5] Equation 1
        for t in range(low, high+1):
            p1 = self.pmf(l-t, n-1)
            p2 = self.pmf(t, 1)
            p += p1*p2
        self.all_pmfs[n][self.k][l] = p
        return p


# Maintain state for faster repeat calls to page_trend_test w/ method='exact'
_pagel_state = _PageL()
