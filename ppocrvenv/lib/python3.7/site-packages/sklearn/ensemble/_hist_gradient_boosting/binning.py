"""
This module contains the BinMapper class.

BinMapper is used for mapping a real-valued dataset into integer-valued bins.
Bin thresholds are computed with the quantiles so that each bin contains
approximately the same number of samples.
"""
# Author: Nicolas Hug

import numpy as np

from ...utils import check_random_state, check_array
from ...base import BaseEstimator, TransformerMixin
from ...utils.validation import check_is_fitted
from ...utils.fixes import percentile
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ._binning import _map_to_bins
from .common import X_DTYPE, X_BINNED_DTYPE, ALMOST_INF, X_BITSET_INNER_DTYPE
from ._bitset import set_bitset_memoryview


def _find_binning_thresholds(col_data, max_bins):
    """Extract quantiles from a continuous feature.

    Missing values are ignored for finding the thresholds.

    Parameters
    ----------
    col_data : array-like, shape (n_samples,)
        The continuous feature to bin.
    max_bins: int
        The maximum number of bins to use for non-missing values. If for a
        given feature the number of unique values is less than ``max_bins``,
        then those unique values will be used to compute the bin thresholds,
        instead of the quantiles

    Return
    ------
    binning_thresholds : ndarray of shape(min(max_bins, n_unique_values) - 1,)
        The increasing numeric values that can be used to separate the bins.
        A given value x will be mapped into bin value i iff
        bining_thresholds[i - 1] < x <= binning_thresholds[i]
    """
    # ignore missing values when computing bin thresholds
    missing_mask = np.isnan(col_data)
    if missing_mask.any():
        col_data = col_data[~missing_mask]
    col_data = np.ascontiguousarray(col_data, dtype=X_DTYPE)
    distinct_values = np.unique(col_data)
    if len(distinct_values) <= max_bins:
        midpoints = distinct_values[:-1] + distinct_values[1:]
        midpoints *= 0.5
    else:
        # We sort again the data in this case. We could compute
        # approximate midpoint percentiles using the output of
        # np.unique(col_data, return_counts) instead but this is more
        # work and the performance benefit will be limited because we
        # work on a fixed-size subsample of the full data.
        percentiles = np.linspace(0, 100, num=max_bins + 1)
        percentiles = percentiles[1:-1]
        midpoints = percentile(col_data, percentiles, method="midpoint").astype(X_DTYPE)
        assert midpoints.shape[0] == max_bins - 1

    # We avoid having +inf thresholds: +inf thresholds are only allowed in
    # a "split on nan" situation.
    np.clip(midpoints, a_min=None, a_max=ALMOST_INF, out=midpoints)
    return midpoints


class _BinMapper(TransformerMixin, BaseEstimator):
    """Transformer that maps a dataset into integer-valued bins.

    For continuous features, the bins are created in a feature-wise fashion,
    using quantiles so that each bins contains approximately the same number
    of samples. For large datasets, quantiles are computed on a subset of the
    data to speed-up the binning, but the quantiles should remain stable.

    For categorical features, the raw categorical values are expected to be
    in [0, 254] (this is not validated here though) and each category
    corresponds to a bin. All categorical values must be known at
    initialization: transform() doesn't know how to bin unknown categorical
    values. Note that transform() is only used on non-training data in the
    case of early stopping.

    Features with a small number of values may be binned into less than
    ``n_bins`` bins. The last bin (at index ``n_bins - 1``) is always reserved
    for missing values.

    Parameters
    ----------
    n_bins : int, default=256
        The maximum number of bins to use (including the bin for missing
        values). Should be in [3, 256]. Non-missing values are binned on
        ``max_bins = n_bins - 1`` bins. The last bin is always reserved for
        missing values. If for a given feature the number of unique values is
        less than ``max_bins``, then those unique values will be used to
        compute the bin thresholds, instead of the quantiles. For categorical
        features indicated by ``is_categorical``, the docstring for
        ``is_categorical`` details on this procedure.
    subsample : int or None, default=2e5
        If ``n_samples > subsample``, then ``sub_samples`` samples will be
        randomly chosen to compute the quantiles. If ``None``, the whole data
        is used.
    is_categorical : ndarray of bool of shape (n_features,), default=None
        Indicates categorical features. By default, all features are
        considered continuous.
    known_categories : list of {ndarray, None} of shape (n_features,), \
            default=none
        For each categorical feature, the array indicates the set of unique
        categorical values. These should be the possible values over all the
        data, not just the training data. For continuous features, the
        corresponding entry should be None.
    random_state: int, RandomState instance or None, default=None
        Pseudo-random number generator to control the random sub-sampling.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.
    n_threads : int, default=None
        Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
        to determine the effective number of threads use, which takes cgroups CPU
        quotes into account. See the docstring of `_openmp_effective_n_threads`
        for details.

    Attributes
    ----------
    bin_thresholds_ : list of ndarray
        For each feature, each array indicates how to map a feature into a
        binned feature. The semantic and size depends on the nature of the
        feature:
        - for real-valued features, the array corresponds to the real-valued
          bin thresholds (the upper bound of each bin). There are ``max_bins
          - 1`` thresholds, where ``max_bins = n_bins - 1`` is the number of
          bins used for non-missing values.
        - for categorical features, the array is a map from a binned category
          value to the raw category value. The size of the array is equal to
          ``min(max_bins, category_cardinality)`` where we ignore missing
          values in the cardinality.
    n_bins_non_missing_ : ndarray, dtype=np.uint32
        For each feature, gives the number of bins actually used for
        non-missing values. For features with a lot of unique values, this is
        equal to ``n_bins - 1``.
    is_categorical_ : ndarray of shape (n_features,), dtype=np.uint8
        Indicator for categorical features.
    missing_values_bin_idx_ : np.uint8
        The index of the bin where missing values are mapped. This is a
        constant across all features. This corresponds to the last bin, and
        it is always equal to ``n_bins - 1``. Note that if ``n_bins_missing_``
        is less than ``n_bins - 1`` for a given feature, then there are
        empty (and unused) bins.
    """

    def __init__(
        self,
        n_bins=256,
        subsample=int(2e5),
        is_categorical=None,
        known_categories=None,
        random_state=None,
        n_threads=None,
    ):
        self.n_bins = n_bins
        self.subsample = subsample
        self.is_categorical = is_categorical
        self.known_categories = known_categories
        self.random_state = random_state
        self.n_threads = n_threads

    def fit(self, X, y=None):
        """Fit data X by computing the binning thresholds.

        The last bin is reserved for missing values, whether missing values
        are present in the data or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to bin.
        y: None
            Ignored.

        Returns
        -------
        self : object
        """
        if not (3 <= self.n_bins <= 256):
            # min is 3: at least 2 distinct bins and a missing values bin
            raise ValueError(
                "n_bins={} should be no smaller than 3 and no larger than 256.".format(
                    self.n_bins
                )
            )

        X = check_array(X, dtype=[X_DTYPE], force_all_finite=False)
        max_bins = self.n_bins - 1

        rng = check_random_state(self.random_state)
        if self.subsample is not None and X.shape[0] > self.subsample:
            subset = rng.choice(X.shape[0], self.subsample, replace=False)
            X = X.take(subset, axis=0)

        if self.is_categorical is None:
            self.is_categorical_ = np.zeros(X.shape[1], dtype=np.uint8)
        else:
            self.is_categorical_ = np.asarray(self.is_categorical, dtype=np.uint8)

        n_features = X.shape[1]
        known_categories = self.known_categories
        if known_categories is None:
            known_categories = [None] * n_features

        # validate is_categorical and known_categories parameters
        for f_idx in range(n_features):
            is_categorical = self.is_categorical_[f_idx]
            known_cats = known_categories[f_idx]
            if is_categorical and known_cats is None:
                raise ValueError(
                    f"Known categories for feature {f_idx} must be provided."
                )
            if not is_categorical and known_cats is not None:
                raise ValueError(
                    f"Feature {f_idx} isn't marked as a categorical feature, "
                    "but categories were passed."
                )

        self.missing_values_bin_idx_ = self.n_bins - 1

        self.bin_thresholds_ = []
        n_bins_non_missing = []

        for f_idx in range(n_features):
            if not self.is_categorical_[f_idx]:
                thresholds = _find_binning_thresholds(X[:, f_idx], max_bins)
                n_bins_non_missing.append(thresholds.shape[0] + 1)
            else:
                # Since categories are assumed to be encoded in
                # [0, n_cats] and since n_cats <= max_bins,
                # the thresholds *are* the unique categorical values. This will
                # lead to the correct mapping in transform()
                thresholds = known_categories[f_idx]
                n_bins_non_missing.append(thresholds.shape[0])

            self.bin_thresholds_.append(thresholds)

        self.n_bins_non_missing_ = np.array(n_bins_non_missing, dtype=np.uint32)
        return self

    def transform(self, X):
        """Bin data X.

        Missing values will be mapped to the last bin.

        For categorical features, the mapping will be incorrect for unknown
        categories. Since the BinMapper is given known_categories of the
        entire training data (i.e. before the call to train_test_split() in
        case of early-stopping), this never happens.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to bin.

        Returns
        -------
        X_binned : array-like of shape (n_samples, n_features)
            The binned data (fortran-aligned).
        """
        X = check_array(X, dtype=[X_DTYPE], force_all_finite=False)
        check_is_fitted(self)
        if X.shape[1] != self.n_bins_non_missing_.shape[0]:
            raise ValueError(
                "This estimator was fitted with {} features but {} got passed "
                "to transform()".format(self.n_bins_non_missing_.shape[0], X.shape[1])
            )

        n_threads = _openmp_effective_n_threads(self.n_threads)
        binned = np.zeros_like(X, dtype=X_BINNED_DTYPE, order="F")
        _map_to_bins(
            X, self.bin_thresholds_, self.missing_values_bin_idx_, n_threads, binned
        )
        return binned

    def make_known_categories_bitsets(self):
        """Create bitsets of known categories.

        Returns
        -------
        - known_cat_bitsets : ndarray of shape (n_categorical_features, 8)
            Array of bitsets of known categories, for each categorical feature.
        - f_idx_map : ndarray of shape (n_features,)
            Map from original feature index to the corresponding index in the
            known_cat_bitsets array.
        """

        categorical_features_indices = np.flatnonzero(self.is_categorical_)

        n_features = self.is_categorical_.size
        n_categorical_features = categorical_features_indices.size

        f_idx_map = np.zeros(n_features, dtype=np.uint32)
        f_idx_map[categorical_features_indices] = np.arange(
            n_categorical_features, dtype=np.uint32
        )

        known_categories = self.bin_thresholds_

        known_cat_bitsets = np.zeros(
            (n_categorical_features, 8), dtype=X_BITSET_INNER_DTYPE
        )

        # TODO: complexity is O(n_categorical_features * 255). Maybe this is
        # worth cythonizing
        for mapped_f_idx, f_idx in enumerate(categorical_features_indices):
            for raw_cat_val in known_categories[f_idx]:
                set_bitset_memoryview(known_cat_bitsets[mapped_f_idx], raw_cat_val)

        return known_cat_bitsets, f_idx_map
