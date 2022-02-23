"""Univariate features selection."""

# Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort, E. Duchesnay.
#          L. Buitinck, A. Joly
# License: BSD 3 clause


import numpy as np
import warnings

from scipy import special, stats
from scipy.sparse import issparse

from ..base import BaseEstimator
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_sqr, safe_mask
from ..utils.extmath import safe_sparse_dot, row_norms
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin


def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    # XXX where should this function be called? fit? scoring functions
    # themselves?
    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores


######################################################################
# Scoring functions


# The following function is a rewriting of scipy.stats.f_oneway
# Contrary to the scipy.stats.f_oneway implementation it does not
# copy the data while keeping the inputs unchanged.
def f_oneway(*args):
    """Performs a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that 2 or more groups have
    the same population mean. The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    *args : {array-like, sparse matrix}
        sample1, sample2... The sample measurements should be given as
        arguments.

    Returns
    -------
    f_statistic : float
        The computed F-value of the test.
    p_value : float
        The associated p-value from the F-distribution.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent
    2. Each sample is from a normally distributed population
    3. The population standard deviations of the groups are all equal. This
       property is known as homoscedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`_) although
    with some loss of power.

    The algorithm is from Heiman[2], pp.394-7.

    See ``scipy.stats.f_oneway`` that should give the same results while
    being less efficient.

    References
    ----------

    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 14.
           http://faculty.vassar.edu/lowry/ch14pt1.html

    .. [2] Heiman, G.W.  Research Methods in Statistics. 2002.

    """
    n_classes = len(args)
    args = [as_float_array(a) for a in args]
    n_samples_per_class = np.array([a.shape[0] for a in args])
    n_samples = np.sum(n_samples_per_class)
    ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args)
    sums_args = [np.asarray(a.sum(axis=0)) for a in args]
    square_of_sums_alldata = sum(sums_args) ** 2
    square_of_sums_args = [s ** 2 for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.0
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    constant_features_idx = np.where(msw == 0.0)[0]
    if np.nonzero(msb)[0].size != msb.size and constant_features_idx.size:
        warnings.warn("Features %s are constant." % constant_features_idx, UserWarning)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob


def f_classif(X, y):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially.

    y : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.

    See Also
    --------
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    """
    X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    return f_oneway(*args)


def _chisquare(f_obs, f_exp):
    """Fast replacement for scipy.stats.chisquare.

    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= f_exp
    chisq = chisq.sum(axis=0)
    return chisq, special.chdtrc(k - 1, chisq)


def chi2(X, y):
    """Compute chi-squared stats between each non-negative feature and class.

    This score can be used to select the n_features features with the
    highest values for the test chi-squared statistic from X, which must
    contain only non-negative features such as booleans or frequencies
    (e.g., term counts in document classification), relative to the classes.

    Recall that the chi-square test measures dependence between stochastic
    variables, so using this function "weeds out" the features that are the
    most likely to be independent of class and therefore irrelevant for
    classification.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample vectors.

    y : array-like of shape (n_samples,)
        Target vector (class labels).

    Returns
    -------
    chi2 : ndarray of shape (n_features,)
        Chi2 statistics for each feature.

    p_values : ndarray of shape (n_features,)
        P-values for each feature.

    Notes
    -----
    Complexity of this algorithm is O(n_classes * n_features).

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    """

    # XXX: we might want to do some of the following in logspace instead for
    # numerical stability.
    X = check_array(X, accept_sparse="csr")
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features

    feature_count = X.sum(axis=0).reshape(1, -1)
    class_prob = Y.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)

    return _chisquare(observed, expected)


def r_regression(X, y, *, center=True):
    """Compute Pearson's r for each features and the target.

    Pearson's r is also known as the Pearson correlation coefficient.

    .. versionadded:: 1.0

    Linear model for testing the individual effect of each of many regressors.
    This is a scoring function to be used in a feature selection procedure, not
    a free standing feature selection procedure.

    The cross correlation between each regressor and the target is computed
    as ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) * std(y)).

    For more on usage see the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.

    y : array-like of shape (n_samples,)
        The target vector.

    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.

    Returns
    -------
    correlation_coefficient : ndarray of shape (n_features,)
        Pearson's R correlation coefficients of features.

    See Also
    --------
    f_regression: Univariate linear regression tests returning f-statistic
        and p-values
    mutual_info_regression: Mutual information for a continuous target.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    """
    X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"], dtype=np.float64)
    n_samples = X.shape[0]

    # Compute centered values
    # Note that E[(x - mean(x))*(y - mean(y))] = E[x*(y - mean(y))], so we
    # need not center X
    if center:
        y = y - np.mean(y)
        if issparse(X):
            X_means = X.mean(axis=0).getA1()
        else:
            X_means = X.mean(axis=0)
        # Compute the scaled standard deviations via moments
        X_norms = np.sqrt(row_norms(X.T, squared=True) - n_samples * X_means ** 2)
    else:
        X_norms = row_norms(X.T)

    correlation_coefficient = safe_sparse_dot(y, X)
    correlation_coefficient /= X_norms
    correlation_coefficient /= np.linalg.norm(y)
    return correlation_coefficient


def f_regression(X, y, *, center=True):
    """Univariate linear regression tests returning F-statistic and p-values.

    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors.

    This is done in 2 steps:

    1. The cross correlation between each regressor and the target is computed,
       that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) *
       std(y)) using r_regression function.
    2. It is converted to an F score and then to a p-value.

    :func:`f_regression` is derived from :func:`r_regression` and will rank
    features in the same order if all the features are positively correlated
    with the target.

    Note however that contrary to :func:`f_regression`, :func:`r_regression`
    values lie in [-1, 1] and can thus be negative. :func:`f_regression` is
    therefore recommended as a feature selection criterion to identify
    potentially predictive feature for a downstream classifier, irrespective of
    the sign of the association with the target variable.

    Furthermore :func:`f_regression` returns p-values while
    :func:`r_regression` does not.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data matrix.

    y : array-like of shape (n_samples,)
        The target vector.

    center : bool, default=True
        Whether or not to center the data matrix `X` and the target vector `y`.
        By default, `X` and `y` will be centered.

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.

    See Also
    --------
    r_regression: Pearson's R between label/feature for regression tasks.
    f_classif: ANOVA F-value between label/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    SelectPercentile: Select features based on percentile of the highest
        scores.
    """
    correlation_coefficient = r_regression(X, y, center=center)
    deg_of_freedom = y.size - (2 if center else 1)

    corr_coef_squared = correlation_coefficient ** 2
    f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
    p_values = stats.f.sf(f_statistic, 1, deg_of_freedom)
    return f_statistic, p_values


######################################################################
# Base classes


class _BaseFilter(SelectorMixin, BaseEstimator):
    """Initialize the univariate feature selection.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
    """

    def __init__(self, score_func):
        self.score_func = score_func

    def fit(self, X, y):
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc"], multi_output=True
        )

        if not callable(self.score_func):
            raise TypeError(
                "The score function should be a callable, %s (%s) was passed."
                % (self.score_func, type(self.score_func))
            )

        self._check_params(X, y)
        score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        self.scores_ = np.asarray(self.scores_)

        return self

    def _check_params(self, X, y):
        pass

    def _more_tags(self):
        return {"requires_y": True}


######################################################################
# Specific filters
######################################################################
class SelectPercentile(_BaseFilter):
    """Select features according to a percentile of the highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

        .. versionadded:: 0.18

    percentile : int, default=10
        Percent of features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectPercentile, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
    >>> X_new.shape
    (1797, 7)
    """

    def __init__(self, score_func=f_classif, *, percentile=10):
        super().__init__(score_func=score_func)
        self.percentile = percentile

    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError(
                "percentile should be >=0, <=100; got %r" % self.percentile
            )

    def _get_support_mask(self):
        check_is_fitted(self)

        # Cater for NaNs
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=bool)
        elif self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=bool)

        scores = _clean_nans(self.scores_)
        threshold = np.percentile(scores, 100 - self.percentile)
        mask = scores > threshold
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[: max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask


class SelectKBest(_BaseFilter):
    """Select features according to the k highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

        .. versionadded:: 0.18

    k : int or "all", default=10
        Number of top features to select.
        The "all" option bypasses selection, for use in a parameter search.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned only scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif: ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    mutual_info_regression: Mutual information for a continuous target.
    SelectPercentile: Select features based on percentile of the highest
        scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> X, y = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
    >>> X_new.shape
    (1797, 20)
    """

    def __init__(self, score_func=f_classif, *, k=10):
        super().__init__(score_func=score_func)
        self.k = k

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be >=0, <= n_features = %d; got %r. "
                "Use k='all' to return all features." % (X.shape[1], self.k)
            )

    def _get_support_mask(self):
        check_is_fitted(self)

        if self.k == "all":
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per
            # megafeature on x86-64).
            mask[np.argsort(scores, kind="mergesort")[-self.k :]] = 1
            return mask


class SelectFpr(_BaseFilter):
    """Filter: Select the pvalues below alpha based on a FPR test.

    FPR test stands for False Positive Rate test. It controls the total
    amount of false detections.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

    alpha : float, default=5e-2
        Features with p-values less than `alpha` are selected.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    mutual_info_classif: Mutual information for a discrete target.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFpr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)
    """

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.pvalues_ < self.alpha


class SelectFdr(_BaseFilter):
    """Filter: Select the p-values for an estimated false discovery rate.

    This uses the Benjamini-Hochberg procedure. ``alpha`` is an upper bound
    on the expected false discovery rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

    alpha : float, default=5e-2
        The highest uncorrected p-value for features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a contnuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFwe : Select features based on family-wise error rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    References
    ----------
    https://en.wikipedia.org/wiki/False_discovery_rate

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFdr, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 16)
    """

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        n_features = len(self.pvalues_)
        sv = np.sort(self.pvalues_)
        selected = sv[
            sv <= float(self.alpha) / n_features * np.arange(1, n_features + 1)
        ]
        if selected.size == 0:
            return np.zeros_like(self.pvalues_, dtype=bool)
        return self.pvalues_ <= selected.max()


class SelectFwe(_BaseFilter):
    """Filter: Select the p-values corresponding to Family-wise error rate.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).
        Default is f_classif (see below "See Also"). The default function only
        works with classification tasks.

    alpha : float, default=5e-2
        The highest uncorrected p-value for features to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    GenericUnivariateSelect : Univariate feature selector with configurable
        mode.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import SelectFwe, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> X_new = SelectFwe(chi2, alpha=0.01).fit_transform(X, y)
    >>> X_new.shape
    (569, 15)
    """

    def __init__(self, score_func=f_classif, *, alpha=5e-2):
        super().__init__(score_func=score_func)
        self.alpha = alpha

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.pvalues_ < self.alpha / len(self.pvalues_)


######################################################################
# Generic filter
######################################################################

# TODO this class should fit on either p-values or scores,
# depending on the mode.
class GenericUnivariateSelect(_BaseFilter):
    """Univariate feature selector with configurable strategy.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues). For modes 'percentile' or 'kbest' it can return
        a single array scores.

    mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}, default='percentile'
        Feature selection mode.

    param : float or int depending on the feature selection mode, default=1e-5
        Parameter of the corresponding mode.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned scores only.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)
    >>> X_new = transformer.fit_transform(X, y)
    >>> X_new.shape
    (569, 20)
    """

    _selection_modes: dict = {
        "percentile": SelectPercentile,
        "k_best": SelectKBest,
        "fpr": SelectFpr,
        "fdr": SelectFdr,
        "fwe": SelectFwe,
    }

    def __init__(self, score_func=f_classif, *, mode="percentile", param=1e-5):
        super().__init__(score_func=score_func)
        self.mode = mode
        self.param = param

    def _make_selector(self):
        selector = self._selection_modes[self.mode](score_func=self.score_func)

        # Now perform some acrobatics to set the right named parameter in
        # the selector
        possible_params = selector._get_param_names()
        possible_params.remove("score_func")
        selector.set_params(**{possible_params[0]: self.param})

        return selector

    def _check_params(self, X, y):
        if self.mode not in self._selection_modes:
            raise ValueError(
                "The mode passed should be one of %s, %r, (type %s) was passed."
                % (self._selection_modes.keys(), self.mode, type(self.mode))
            )

        self._make_selector()._check_params(X, y)

    def _get_support_mask(self):
        check_is_fitted(self)

        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        return selector._get_support_mask()
