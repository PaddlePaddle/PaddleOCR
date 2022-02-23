"""
Contingency table functions (:mod:`scipy.stats.contingency`)
============================================================

Functions for creating and analyzing contingency tables.

.. currentmodule:: scipy.stats.contingency

.. autosummary::
   :toctree: generated/

   chi2_contingency
   relative_risk
   crosstab
   association

   expected_freq
   margins

"""


from functools import reduce
import math
import numpy as np
from .stats import power_divergence
from ._relative_risk import relative_risk
from ._crosstab import crosstab


__all__ = ['margins', 'expected_freq', 'chi2_contingency', 'crosstab',
           'association', 'relative_risk']


def margins(a):
    """Return a list of the marginal sums of the array `a`.

    Parameters
    ----------
    a : ndarray
        The array for which to compute the marginal sums.

    Returns
    -------
    margsums : list of ndarrays
        A list of length `a.ndim`.  `margsums[k]` is the result
        of summing `a` over all axes except `k`; it has the same
        number of dimensions as `a`, but the length of each axis
        except axis `k` will be 1.

    Examples
    --------
    >>> a = np.arange(12).reshape(2, 6)
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11]])
    >>> from scipy.stats.contingency import margins
    >>> m0, m1 = margins(a)
    >>> m0
    array([[15],
           [51]])
    >>> m1
    array([[ 6,  8, 10, 12, 14, 16]])

    >>> b = np.arange(24).reshape(2,3,4)
    >>> m0, m1, m2 = margins(b)
    >>> m0
    array([[[ 66]],
           [[210]]])
    >>> m1
    array([[[ 60],
            [ 92],
            [124]]])
    >>> m2
    array([[[60, 66, 72, 78]]])
    """
    margsums = []
    ranged = list(range(a.ndim))
    for k in ranged:
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums


def expected_freq(observed):
    """
    Compute the expected frequencies from a contingency table.

    Given an n-dimensional contingency table of observed frequencies,
    compute the expected frequencies for the table based on the marginal
    sums under the assumption that the groups associated with each
    dimension are independent.

    Parameters
    ----------
    observed : array_like
        The table of observed frequencies.  (While this function can handle
        a 1-D array, that case is trivial.  Generally `observed` is at
        least 2-D.)

    Returns
    -------
    expected : ndarray of float64
        The expected frequencies, based on the marginal sums of the table.
        Same shape as `observed`.

    Examples
    --------
    >>> from scipy.stats.contingency import expected_freq
    >>> observed = np.array([[10, 10, 20],[20, 20, 20]])
    >>> expected_freq(observed)
    array([[ 12.,  12.,  16.],
           [ 18.,  18.,  24.]])

    """
    # Typically `observed` is an integer array. If `observed` has a large
    # number of dimensions or holds large values, some of the following
    # computations may overflow, so we first switch to floating point.
    observed = np.asarray(observed, dtype=np.float64)

    # Create a list of the marginal sums.
    margsums = margins(observed)

    # Create the array of expected frequencies.  The shapes of the
    # marginal sums returned by apply_over_axes() are just what we
    # need for broadcasting in the following product.
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected


def chi2_contingency(observed, correction=True, lambda_=None):
    """Chi-square test of independence of variables in a contingency table.

    This function computes the chi-square statistic and p-value for the
    hypothesis test of independence of the observed frequencies in the
    contingency table [1]_ `observed`.  The expected frequencies are computed
    based on the marginal sums under the assumption of independence; see
    `scipy.stats.contingency.expected_freq`.  The number of degrees of
    freedom is (expressed using numpy functions and attributes)::

        dof = observed.size - sum(observed.shape) + observed.ndim - 1


    Parameters
    ----------
    observed : array_like
        The contingency table. The table contains the observed frequencies
        (i.e. number of occurrences) in each category.  In the two-dimensional
        case, the table is often described as an "R x C table".
    correction : bool, optional
        If True, *and* the degrees of freedom is 1, apply Yates' correction
        for continuity.  The effect of the correction is to adjust each
        observed value by 0.5 towards the corresponding expected value.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic [2]_.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family [3]_ to be used instead.  See
        `power_divergence` for details.

    Returns
    -------
    chi2 : float
        The test statistic.
    p : float
        The p-value of the test
    dof : int
        Degrees of freedom
    expected : ndarray, same shape as `observed`
        The expected frequencies, based on the marginal sums of the table.

    See Also
    --------
    contingency.expected_freq
    fisher_exact
    chisquare
    power_divergence
    barnard_exact
    boschloo_exact

    Notes
    -----
    An often quoted guideline for the validity of this calculation is that
    the test should be used only if the observed and expected frequencies
    in each cell are at least 5.

    This is a test for the independence of different categories of a
    population. The test is only meaningful when the dimension of
    `observed` is two or more.  Applying the test to a one-dimensional
    table will always result in `expected` equal to `observed` and a
    chi-square statistic equal to 0.

    This function does not handle masked arrays, because the calculation
    does not make sense with missing values.

    Like stats.chisquare, this function computes a chi-square statistic;
    the convenience this function provides is to figure out the expected
    frequencies and degrees of freedom from the given contingency table.
    If these were already known, and if the Yates' correction was not
    required, one could use stats.chisquare.  That is, if one calls::

        chi2, p, dof, ex = chi2_contingency(obs, correction=False)

    then the following is true::

        (chi2, p) == stats.chisquare(obs.ravel(), f_exp=ex.ravel(),
                                     ddof=obs.size - 1 - dof)

    The `lambda_` argument was added in version 0.13.0 of scipy.

    References
    ----------
    .. [1] "Contingency table",
           https://en.wikipedia.org/wiki/Contingency_table
    .. [2] "Pearson's chi-squared test",
           https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.

    Examples
    --------
    A two-way example (2 x 3):

    >>> from scipy.stats import chi2_contingency
    >>> obs = np.array([[10, 10, 20], [20, 20, 20]])
    >>> chi2_contingency(obs)
    (2.7777777777777777,
     0.24935220877729619,
     2,
     array([[ 12.,  12.,  16.],
            [ 18.,  18.,  24.]]))

    Perform the test using the log-likelihood ratio (i.e. the "G-test")
    instead of Pearson's chi-squared statistic.

    >>> g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")
    >>> g, p
    (2.7688587616781319, 0.25046668010954165)

    A four-way example (2 x 2 x 2 x 2):

    >>> obs = np.array(
    ...     [[[[12, 17],
    ...        [11, 16]],
    ...       [[11, 12],
    ...        [15, 16]]],
    ...      [[[23, 15],
    ...        [30, 22]],
    ...       [[14, 17],
    ...        [15, 16]]]])
    >>> chi2_contingency(obs)
    (8.7584514426741897,
     0.64417725029295503,
     11,
     array([[[[ 14.15462386,  14.15462386],
              [ 16.49423111,  16.49423111]],
             [[ 11.2461395 ,  11.2461395 ],
              [ 13.10500554,  13.10500554]]],
            [[[ 19.5591166 ,  19.5591166 ],
              [ 22.79202844,  22.79202844]],
             [[ 15.54012004,  15.54012004],
              [ 18.10873492,  18.10873492]]]]))
    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = expected_freq(observed)
    if np.any(expected == 0):
        # Include one of the positions where expected is zero in
        # the exception message.
        zeropos = list(zip(*np.nonzero(expected == 0)))[0]
        raise ValueError("The internally computed table of expected "
                         "frequencies has a zero element at %s." % (zeropos,))

    # The degrees of freedom
    dof = expected.size - sum(expected.shape) + expected.ndim - 1

    if dof == 0:
        # Degenerate case; this occurs when `observed` is 1D (or, more
        # generally, when it has only one nontrivial dimension).  In this
        # case, we also have observed == expected, so chi2 is 0.
        chi2 = 0.0
        p = 1.0
    else:
        if dof == 1 and correction:
            # Adjust `observed` according to Yates' correction for continuity.
            # Magnitude of correction no bigger than difference; see gh-13875
            diff = expected - observed
            direction = np.sign(diff)
            magnitude = np.minimum(0.5, np.abs(diff))
            observed = observed + magnitude * direction

        chi2, p = power_divergence(observed, expected,
                                   ddof=observed.size - 1 - dof, axis=None,
                                   lambda_=lambda_)

    return chi2, p, dof, expected


def association(observed, method="cramer", correction=False, lambda_=None):
    """Calculates degree of association between two nominal variables.

    The function provides the option for computing one of three measures of
    association between two nominal variables from the data given in a 2d
    contingency table: Tschuprow's T, Pearson's Contingency Coefficient
    and Cramer's V.

    Parameters
    ----------
    observed : array-like
        The array of observed values
    method : {"cramer", "tschuprow", "pearson"} (default = "cramer")
        The association test statistic.
    correction : bool, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`
    lambda_ : float or str, optional
        Inherited from `scipy.stats.contingency.chi2_contingency()`

    Returns
    -------
    statistic : float
        Value of the test statistic

    Notes
    -----
    Cramer's V, Tschuprow's T and Pearson's Contingency Coefficient, all
    measure the degree to which two nominal or ordinal variables are related,
    or the level of their association. This differs from correlation, although
    many often mistakenly consider them equivalent. Correlation measures in
    what way two variables are related, whereas, association measures how
    related the variables are. As such, association does not subsume
    independent variables, and is rather a test of independence. A value of
    1.0 indicates perfect association, and 0.0 means the variables have no
    association.

    Both the Cramer's V and Tschuprow's T are extensions of the phi
    coefficient.  Moreover, due to the close relationship between the
    Cramer's V and Tschuprow's T the returned values can often be similar
    or even equivalent.  They are likely to diverge more as the array shape
    diverges from a 2x2.

    References
    ----------
    .. [1] "Tschuprow's T",
           https://en.wikipedia.org/wiki/Tschuprow's_T
    .. [2] Tschuprow, A. A. (1939)
           Principles of the Mathematical Theory of Correlation;
           translated by M. Kantorowitsch. W. Hodge & Co.
    .. [3] "Cramer's V", https://en.wikipedia.org/wiki/Cramer's_V
    .. [4] "Nominal Association: Phi and Cramer's V",
           http://www.people.vcu.edu/~pdattalo/702SuppRead/MeasAssoc/NominalAssoc.html
    .. [5] Gingrich, Paul, "Association Between Variables",
           http://uregina.ca/~gingrich/ch11a.pdf

    Examples
    --------
    An example with a 4x2 contingency table:

    >>> from scipy.stats.contingency import association
    >>> obs4x2 = np.array([[100, 150], [203, 322], [420, 700], [320, 210]])

    Pearson's contingency coefficient
    >>> association(obs4x2, method="pearson")
    0.18303298140595667

    Cramer's V

    >>> association(obs4x2, method="cramer")
    0.18617813077483678

    Tschuprow's T

    >>> association(obs4x2, method="tschuprow")
    0.14146478765062995
    """
    arr = np.asarray(observed)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("`observed` must be an integer array.")

    if len(arr.shape) != 2:
        raise ValueError("method only accepts 2d arrays")

    chi2_stat = chi2_contingency(arr, correction=correction,
                                 lambda_=lambda_)

    phi2 = chi2_stat[0] / arr.sum()
    n_rows, n_cols = arr.shape
    if method == "cramer":
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == 'pearson':
        value = phi2 / (1 + phi2)
    else:
        raise ValueError("Invalid argument value: 'method' argument must "
                         "be 'cramer', 'tschuprow', or 'pearson'")

    return math.sqrt(value)
