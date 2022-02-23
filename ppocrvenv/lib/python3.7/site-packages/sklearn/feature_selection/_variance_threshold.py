# Author: Lars Buitinck
# License: 3-clause BSD

import numpy as np
from ..base import BaseEstimator
from ._base import SelectorMixin
from ..utils.sparsefuncs import mean_variance_axis, min_max_axis
from ..utils.validation import check_is_fitted


class VarianceThreshold(SelectorMixin, BaseEstimator):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, default=0
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SelectFromModel: Meta-transformer for selecting features based on
        importance weights.
    SelectPercentile : Select features according to a percentile of the highest
        scores.
    SequentialFeatureSelector : Transformer that performs Sequential Feature
        Selection.

    Notes
    -----
    Allows NaN in the input.
    Raises ValueError if no feature in X meets the variance threshold.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> from sklearn.feature_selection import VarianceThreshold
        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            force_all_finite="allow-nan",
        )

        if hasattr(X, "toarray"):  # sparse matrix
            _, self.variances_ = mean_variance_axis(X, axis=0)
            if self.threshold == 0:
                mins, maxes = min_max_axis(X, axis=0)
                peak_to_peaks = maxes - mins
        else:
            self.variances_ = np.nanvar(X, axis=0)
            if self.threshold == 0:
                peak_to_peaks = np.ptp(X, axis=0)

        if self.threshold == 0:
            # Use peak-to-peak to avoid numeric precision issues
            # for constant features
            compare_arr = np.array([self.variances_, peak_to_peaks])
            self.variances_ = np.nanmin(compare_arr, axis=0)
        elif self.threshold < 0.0:
            raise ValueError(f"Threshold must be non-negative. Got: {self.threshold}")

        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.variances_ > self.threshold

    def _more_tags(self):
        return {"allow_nan": True}
