# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Eric Martin <eric@ericmart.in>
#          Giorgio Patrini <giorgio.patrini@anu.edu.au>
#          Eric Chang <ericchang2017@u.northwestern.edu>
# License: BSD 3 clause


import warnings

import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize
from scipy.special import boxcox

from ..base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
from ..utils import check_array
from ..utils.deprecation import deprecated
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs_fast import (
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)
from ..utils.sparsefuncs import (
    inplace_column_scale,
    mean_variance_axis,
    incr_mean_variance_axis,
    min_max_axis,
)
from ..utils.validation import (
    check_is_fitted,
    check_random_state,
    _check_sample_weight,
    FLOAT_DTYPES,
)

from ._encoders import OneHotEncoder


BOUNDS_THRESHOLD = 1e-7

__all__ = [
    "Binarizer",
    "KernelCenterer",
    "MinMaxScaler",
    "MaxAbsScaler",
    "Normalizer",
    "OneHotEncoder",
    "RobustScaler",
    "StandardScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "add_dummy_feature",
    "binarize",
    "normalize",
    "scale",
    "robust_scale",
    "maxabs_scale",
    "minmax_scale",
    "quantile_transform",
    "power_transform",
]


def _is_constant_feature(var, mean, n_samples):
    """Detect if a feature is indistinguishable from a constant feature.

    The detection is based on its computed variance and on the theoretical
    error bounds of the '2 pass algorithm' for variance computation.

    See "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    """
    # In scikit-learn, variance is always computed using float64 accumulators.
    eps = np.finfo(np.float64).eps

    upper_bound = n_samples * eps * var + (n_samples * mean * eps) ** 2
    return var <= upper_bound


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale


def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : int, default=0
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSC matrix and if axis is 1).

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the CSC matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSC matrix.

    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.StandardScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(StandardScaler(), LogisticRegression())`.

    See Also
    --------
    StandardScaler : Performs scaling to unit variance using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    """  # noqa
    X = check_array(
        X,
        accept_sparse="csc",
        copy=copy,
        ensure_2d=False,
        estimator="the scale function",
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )
    if sparse.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives."
            )
        if axis != 0:
            raise ValueError(
                "Can only scale sparse matrix on axis=0,  got axis=%d" % axis
            )
        if with_std:
            _, var = mean_variance_axis(X, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        if with_mean:
            mean_ = np.nanmean(X, axis)
        if with_std:
            scale_ = np.nanstd(X, axis)
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = np.nanmean(Xr, axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1, 0):
                warnings.warn(
                    "Numerical issues were encountered "
                    "when centering the data "
                    "and might not be solved. Dataset may "
                    "contain too large values. You may need "
                    "to prescale your features."
                )
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = np.nanmean(Xr, axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
                if not np.allclose(mean_2, 0):
                    warnings.warn(
                        "Numerical issues were encountered "
                        "when scaling the data "
                        "and might not be solved. The standard "
                        "deviation of the data is probably "
                        "very close to 0. "
                    )
                    Xr -= mean_2
    return X


class MinMaxScaler(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.

        .. versionadded:: 0.24

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``

    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

        .. versionadded:: 0.17
           *scale_* attribute.

    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data

        .. versionadded:: 0.17
           *data_min_*

    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data

        .. versionadded:: 0.17
           *data_max_*

    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data

        .. versionadded:: 0.17
           *data_range_*

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    minmax_scale : Equivalent function without the estimator API.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]
    """

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        if sparse.issparse(X):
            raise TypeError(
                "MinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        first_pass = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            reset=first_pass,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X,
            copy=self.copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
            reset=False,
        )

        X *= self.scale_
        X += self.min_
        if self.clip:
            np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(
            X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {"allow_nan": True}


def minmax_scale(X, feature_range=(0, 1), *, axis=0, copy=True):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by (when ``axis=0``)::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as (when ``axis=0``)::

       X_scaled = scale * X + min - X.min(axis=0) * scale
       where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    .. versionadded:: 0.17
       *minmax_scale* function interface
       to :class:`~sklearn.preprocessing.MinMaxScaler`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    axis : int, default=0
        Axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    Returns
    -------
    X_tr : ndarray of shape (n_samples, n_features)
        The transformed data.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.minmax_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.MinMaxScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(MinMaxScaler(), LogisticRegression())`.

    See Also
    --------
    MinMaxScaler : Performs scaling to a given range using the Transformer
        API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.
    X = check_array(
        X, copy=False, ensure_2d=False, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
    )
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MinMaxScaler(feature_range=feature_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X


class StandardScaler(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

        .. versionadded:: 0.17
           *scale_*

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    See Also
    --------
    scale : Equivalent function without the estimator API.

    :class:`~sklearn.decomposition.PCA` : Further removes the linear
        correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    def partial_fit(self, X, y=None, sample_weight=None):
        """Online computation of mean and std on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        first_call = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
            reset=first_call,
        )
        n_features = X.shape[1]

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        # if n_samples_seen_ is an integer (i.e. no missing values), we need to
        # transform it to a NumPy array of shape (n_features,) required by
        # incr_mean_variance_axis and _incremental_variance_axis
        dtype = np.int64 if sample_weight is None else X.dtype
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = np.zeros(n_features, dtype=dtype)
        elif np.size(self.n_samples_seen_) == 1:
            self.n_samples_seen_ = np.repeat(self.n_samples_seen_, X.shape[1])
            self.n_samples_seen_ = self.n_samples_seen_.astype(dtype, copy=False)

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives."
                )
            sparse_constructor = (
                sparse.csr_matrix if X.format == "csr" else sparse.csc_matrix
            )

            if self.with_std:
                # First pass
                if not hasattr(self, "scale_"):
                    self.mean_, self.var_, self.n_samples_seen_ = mean_variance_axis(
                        X, axis=0, weights=sample_weight, return_sum_weights=True
                    )
                # Next passes
                else:
                    (
                        self.mean_,
                        self.var_,
                        self.n_samples_seen_,
                    ) = incr_mean_variance_axis(
                        X,
                        axis=0,
                        last_mean=self.mean_,
                        last_var=self.var_,
                        last_n=self.n_samples_seen_,
                        weights=sample_weight,
                    )
                # We force the mean and variance to float64 for large arrays
                # See https://github.com/scikit-learn/scikit-learn/pull/12338
                self.mean_ = self.mean_.astype(np.float64, copy=False)
                self.var_ = self.var_.astype(np.float64, copy=False)
            else:
                self.mean_ = None  # as with_mean must be False for sparse
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (np.isnan(X.data), X.indices, X.indptr), shape=X.shape
                )
                self.n_samples_seen_ += (np.sum(weights) - sum_weights_nan).astype(
                    dtype
                )
        else:
            # First pass
            if not hasattr(self, "scale_"):
                self.mean_ = 0.0
                if self.with_std:
                    self.var_ = 0.0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - np.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = _incremental_mean_and_var(
                    X,
                    self.mean_,
                    self.var_,
                    self.n_samples_seen_,
                    sample_weight=sample_weight,
                )

        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        if np.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_std:
            # Extract the list of near constant features on the raw variances,
            # before taking the square root.
            constant_mask = _is_constant_feature(
                self.var_, self.mean_, self.n_samples_seen_
            )
            self.scale_ = _handle_zeros_in_scale(
                np.sqrt(self.var_), copy=False, constant_mask=constant_mask
            )
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = self._validate_data(
            X,
            reset=False,
            accept_sparse="csr",
            copy=copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = check_array(
            X,
            accept_sparse="csr",
            copy=copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X

    def _more_tags(self):
        return {"allow_nan": True, "preserves_dtype": [np.float64, np.float32]}


class MaxAbsScaler(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data, and
    thus does not destroy any sparsity.

    This scaler can also be applied to sparse CSR or CSC matrices.

    .. versionadded:: 0.17

    Parameters
    ----------
    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.

        .. versionadded:: 0.17
           *scale_* attribute.

    max_abs_ : ndarray of shape (n_features,)
        Per feature maximum absolute value.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    See Also
    --------
    maxabs_scale : Equivalent function without the estimator API.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> from sklearn.preprocessing import MaxAbsScaler
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = MaxAbsScaler().fit(X)
    >>> transformer
    MaxAbsScaler()
    >>> transformer.transform(X)
    array([[ 0.5, -1. ,  1. ],
           [ 1. ,  0. ,  0. ],
           [ 0. ,  1. , -0.5]])
    """

    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.max_abs_

    def fit(self, X, y=None):
        """Compute the maximum absolute value to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of max absolute value of X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        first_pass = not hasattr(self, "n_samples_seen_")
        X = self._validate_data(
            X,
            reset=first_pass,
            accept_sparse=("csr", "csc"),
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
            max_abs = np.maximum(np.abs(mins), np.abs(maxs))
        else:
            max_abs = np.nanmax(np.abs(X), axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            max_abs = np.maximum(self.max_abs_, max_abs)
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs
        self.scale_ = _handle_zeros_in_scale(max_abs, copy=True)
        return self

    def transform(self, X):
        """Scale the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data that should be scaled.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            reset=False,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            inplace_column_scale(X, 1.0 / self.scale_)
        else:
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data that should be transformed back.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            inplace_column_scale(X, self.scale_)
        else:
            X *= self.scale_
        return X

    def _more_tags(self):
        return {"allow_nan": True}


def maxabs_scale(X, *, axis=0, copy=True):
    """Scale each feature to the [-1, 1] range without breaking the sparsity.

    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data.

    axis : int, default=0
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.maxabs_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.MaxAbsScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(MaxAbsScaler(), LogisticRegression())`.

    See Also
    --------
    MaxAbsScaler : Performs scaling to the [-1, 1] range using
        the Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    # Unlike the scaler object, this function allows 1d input.

    # If copy is required, it will be done inside the scaler object.
    X = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=False,
        ensure_2d=False,
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MaxAbsScaler(copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X


class RobustScaler(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).

    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the :meth:`transform` method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators. Typically this is done by removing the mean
    and scaling to unit variance. However, outliers can often influence the
    sample mean / variance in a negative way. In such cases, the median and
    the interquartile range often give better results.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    with_centering : bool, default=True
        If `True`, center the data before scaling.
        This will cause :meth:`transform` to raise an exception when attempted
        on sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_scaling : bool, default=True
        If `True`, scale the data to interquartile range.

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, \
        default=(25.0, 75.0)
        Quantile range used to calculate `scale_`. By default this is equal to
        the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
        quantile.

        .. versionadded:: 0.18

    copy : bool, default=True
        If `False`, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    unit_variance : bool, default=False
        If `True`, scale data so that normally distributed features have a
        variance of 1. In general, if the difference between the x-values of
        `q_max` and `q_min` for a standard normal distribution is greater
        than 1, the dataset will be scaled down. If less than 1, the dataset
        will be scaled up.

        .. versionadded:: 0.24

    Attributes
    ----------
    center_ : array of floats
        The median value for each feature in the training set.

    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.

        .. versionadded:: 0.17
           *scale_* attribute.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    robust_scale : Equivalent function without the estimator API.
    sklearn.decomposition.PCA : Further removes the linear correlation across
        features with 'whiten=True'.

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    https://en.wikipedia.org/wiki/Median
    https://en.wikipedia.org/wiki/Interquartile_range

    Examples
    --------
    >>> from sklearn.preprocessing import RobustScaler
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> transformer = RobustScaler().fit(X)
    >>> transformer
    RobustScaler()
    >>> transformer.transform(X)
    array([[ 0. , -2. ,  0. ],
           [-1. ,  0. ,  0.4],
           [ 1. ,  0. , -1.6]])
    """

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the median and quantiles to be used for scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the median and quantiles
            used for later scaling along the features axis.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        # at fit, convert sparse matrices to csc for optimized computation of
        # the quantiles
        X = self._validate_data(
            X,
            accept_sparse="csc",
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" % str(self.quantile_range))

        if self.with_centering:
            if sparse.issparse(X):
                raise ValueError(
                    "Cannot center sparse matrices: use `with_centering=False`"
                    " instead. See docstring for motivation and alternatives."
                )
            self.center_ = np.nanmedian(X, axis=0)
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = []
            for feature_idx in range(X.shape[1]):
                if sparse.issparse(X):
                    column_nnz_data = X.data[
                        X.indptr[feature_idx] : X.indptr[feature_idx + 1]
                    ]
                    column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
                    column_data[: len(column_nnz_data)] = column_nnz_data
                else:
                    column_data = X[:, feature_idx]

                quantiles.append(np.nanpercentile(column_data, self.quantile_range))

            quantiles = np.transpose(quantiles)

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
            if self.unit_variance:
                adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
                self.scale_ = self.scale_ / adjust
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """Center and scale the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            reset=False,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, 1.0 / self.scale_)
        else:
            if self.with_centering:
                X -= self.center_
            if self.with_scaling:
                X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The rescaled data to be transformed back.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_scaling:
                X *= self.scale_
            if self.with_centering:
                X += self.center_
        return X

    def _more_tags(self):
        return {"allow_nan": True}


def robust_scale(
    X,
    *,
    axis=0,
    with_centering=True,
    with_scaling=True,
    quantile_range=(25.0, 75.0),
    copy=True,
    unit_variance=False,
):
    """Standardize a dataset along any axis.

    Center to the median and component wise scale
    according to the interquartile range.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_sample, n_features)
        The data to center and scale.

    axis : int, default=0
        Axis used to compute the medians and IQR along. If 0,
        independently scale each feature, otherwise (if 1) scale
        each sample.

    with_centering : bool, default=True
        If `True`, center the data before scaling.

    with_scaling : bool, default=True
        If `True`, scale the data to unit variance (or equivalently,
        unit standard deviation).

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0,\
        default=(25.0, 75.0)
        Quantile range used to calculate `scale_`. By default this is equal to
        the IQR, i.e., `q_min` is the first quantile and `q_max` is the third
        quantile.

        .. versionadded:: 0.18

    copy : bool, default=True
        Set to `False` to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    unit_variance : bool, default=False
        If `True`, scale data so that normally distributed features have a
        variance of 1. In general, if the difference between the x-values of
        `q_max` and `q_min` for a standard normal distribution is greater
        than 1, the dataset will be scaled down. If less than 1, the dataset
        will be scaled up.

        .. versionadded:: 0.24

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    Notes
    -----
    This implementation will refuse to center scipy.sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_centering=False` (in that case, only variance scaling will be
    performed on the features of the CSR matrix) or to call `X.toarray()`
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSR matrix.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.robust_scale` unless you know
        what you are doing. A common mistake is to apply it to the entire data
        *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.RobustScaler` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking: `pipe = make_pipeline(RobustScaler(), LogisticRegression())`.

    See Also
    --------
    RobustScaler : Performs centering and scaling using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).
    """
    X = check_array(
        X,
        accept_sparse=("csr", "csc"),
        copy=False,
        ensure_2d=False,
        dtype=FLOAT_DTYPES,
        force_all_finite="allow-nan",
    )
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = RobustScaler(
        with_centering=with_centering,
        with_scaling=with_scaling,
        quantile_range=quantile_range,
        unit_variance=unit_variance,
        copy=copy,
    )
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X


def normalize(X, norm="l2", *, axis=1, copy=True, return_norm=False):
    """Scale input vectors individually to unit norm (vector length).

    Read more in the :ref:`User Guide <preprocessing_normalization>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : {0, 1}, default=1
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : bool, default=True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    return_norm : bool, default=False
        whether to return the computed norms

    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        Normalized input X.

    norms : ndarray of shape (n_samples, ) if axis=1 else (n_features, )
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    See Also
    --------
    Normalizer : Performs normalization using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    if norm not in ("l1", "l2", "max"):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = "csc"
    elif axis == 1:
        sparse_format = "csr"
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_array(
        X,
        accept_sparse=sparse_format,
        copy=copy,
        estimator="the normalize function",
        dtype=FLOAT_DTYPES,
    )
    if axis == 0:
        X = X.T

    if sparse.issparse(X):
        if return_norm and norm in ("l1", "l2"):
            raise NotImplementedError(
                "return_norm=True is not implemented "
                "for sparse matrices with norm 'l1' "
                "or norm 'l2'"
            )
        if norm == "l1":
            inplace_csr_row_normalize_l1(X)
        elif norm == "l2":
            inplace_csr_row_normalize_l2(X)
        elif norm == "max":
            mins, maxes = min_max_axis(X, 1)
            norms = np.maximum(abs(mins), maxes)
            norms_elementwise = norms.repeat(np.diff(X.indptr))
            mask = norms_elementwise != 0
            X.data[mask] /= norms_elementwise[mask]
    else:
        if norm == "l1":
            norms = np.abs(X).sum(axis=1)
        elif norm == "l2":
            norms = row_norms(X)
        elif norm == "max":
            norms = np.max(abs(X), axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    if return_norm:
        return X, norms
    else:
        return X


class Normalizer(TransformerMixin, BaseEstimator):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1, l2 or inf) equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden of
    a copy / conversion).

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Read more in the :ref:`User Guide <preprocessing_normalization>`.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    normalize : Equivalent function without the estimator API.

    Notes
    -----
    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> from sklearn.preprocessing import Normalizer
    >>> X = [[4, 1, 2, 2],
    ...      [1, 3, 9, 3],
    ...      [5, 7, 5, 1]]
    >>> transformer = Normalizer().fit(X)  # fit does nothing.
    >>> transformer
    Normalizer()
    >>> transformer.transform(X)
    array([[0.8, 0.2, 0.4, 0.4],
           [0.1, 0.3, 0.9, 0.3],
           [0.5, 0.7, 0.5, 0.1]])
    """

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to estimate the normalization parameters.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        """Scale each non zero row of X to unit norm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.

        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, accept_sparse="csr", reset=False)
        return normalize(X, norm=self.norm, axis=1, copy=copy)

    def _more_tags(self):
        return {"stateless": True}


def binarize(X, *, threshold=0.0, copy=True):
    """Boolean thresholding of array-like or scipy.sparse matrix.

    Read more in the :ref:`User Guide <preprocessing_binarization>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to binarize, element by element.
        scipy.sparse matrices should be in CSR or CSC format to avoid an
        un-necessary copy.

    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : bool, default=True
        set to False to perform inplace binarization and avoid a copy
        (if the input is already a numpy array or a scipy.sparse CSR / CSC
        matrix and if axis is 1).

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    Binarizer : Performs binarization using the Transformer API
        (e.g. as part of a preprocessing :class:`~sklearn.pipeline.Pipeline`).
    """
    X = check_array(X, accept_sparse=["csr", "csc"], copy=copy)
    if sparse.issparse(X):
        if threshold < 0:
            raise ValueError("Cannot binarize a sparse matrix with threshold < 0")
        cond = X.data > threshold
        not_cond = np.logical_not(cond)
        X.data[cond] = 1
        X.data[not_cond] = 0
        X.eliminate_zeros()
    else:
        cond = X > threshold
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0
    return X


class Binarizer(TransformerMixin, BaseEstimator):
    """Binarize data (set feature values to 0 or 1) according to a threshold.

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    Binarization is a common operation on text count data where the
    analyst can decide to only consider the presence or absence of a
    feature rather than a quantified number of occurrences for instance.

    It can also be used as a pre-processing step for estimators that
    consider boolean random variables (e.g. modelled using the Bernoulli
    distribution in a Bayesian setting).

    Read more in the :ref:`User Guide <preprocessing_binarization>`.

    Parameters
    ----------
    threshold : float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : bool, default=True
        Set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    binarize : Equivalent function without the estimator API.
    KBinsDiscretizer : Bin continuous data into intervals.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.

    Notes
    -----
    If the input is a sparse matrix, only the non-zero values are subject
    to update by the Binarizer class.

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    Examples
    --------
    >>> from sklearn.preprocessing import Binarizer
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = Binarizer().fit(X)  # fit does nothing.
    >>> transformer
    Binarizer()
    >>> transformer.transform(X)
    array([[1., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])
    """

    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._validate_data(X, accept_sparse="csr")
        return self

    def transform(self, X, copy=None):
        """Binarize each element of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.

        copy : bool
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        copy = copy if copy is not None else self.copy
        # TODO: This should be refactored because binarize also calls
        # check_array
        X = self._validate_data(X, accept_sparse=["csr", "csc"], copy=copy, reset=False)
        return binarize(X, threshold=self.threshold, copy=False)

    def _more_tags(self):
        return {"stateless": True}


class KernelCenterer(TransformerMixin, BaseEstimator):
    r"""Center an arbitrary kernel matrix :math:`K`.

    Let define a kernel :math:`K` such that:

    .. math::
        K(X, Y) = \phi(X) . \phi(Y)^{T}

    :math:`\phi(X)` is a function mapping of rows of :math:`X` to a
    Hilbert space and :math:`K` is of shape `(n_samples, n_samples)`.

    This class allows to compute :math:`\tilde{K}(X, Y)` such that:

    .. math::
        \tilde{K(X, Y)} = \tilde{\phi}(X) . \tilde{\phi}(Y)^{T}

    :math:`\tilde{\phi}(X)` is the centered mapped data in the Hilbert
    space.

    `KernelCenterer` centers the features without explicitly computing the
    mapping :math:`\phi(\cdot)`. Working with centered kernels is sometime
    expected when dealing with algebra computation such as eigendecomposition
    for :class:`~sklearn.decomposition.KernelPCA` for instance.

    Read more in the :ref:`User Guide <kernel_centering>`.

    Attributes
    ----------
    K_fit_rows_ : ndarray of shape (n_samples,)
        Average of each column of kernel matrix.

    K_fit_all_ : float
        Average of kernel matrix.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.kernel_approximation.Nystroem : Approximate a kernel map
        using a subset of the training data.

    References
    ----------
    .. [1] `Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller.
       "Nonlinear component analysis as a kernel eigenvalue problem."
       Neural computation 10.5 (1998): 1299-1319.
       <https://www.mlpack.org/papers/kpca.pdf>`_

    Examples
    --------
    >>> from sklearn.preprocessing import KernelCenterer
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> K = pairwise_kernels(X, metric='linear')
    >>> K
    array([[  9.,   2.,  -2.],
           [  2.,  14., -13.],
           [ -2., -13.,  21.]])
    >>> transformer = KernelCenterer().fit(K)
    >>> transformer
    KernelCenterer()
    >>> transformer.transform(K)
    array([[  5.,   0.,  -5.],
           [  0.,  14., -14.],
           [ -5., -14.,  19.]])
    """

    def __init__(self):
        # Needed for backported inspect.signature compatibility with PyPy
        pass

    def fit(self, K, y=None):
        """Fit KernelCenterer.

        Parameters
        ----------
        K : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        K = self._validate_data(K, dtype=FLOAT_DTYPES)

        if K.shape[0] != K.shape[1]:
            raise ValueError(
                "Kernel matrix must be a square matrix."
                " Input is a {}x{} matrix.".format(K.shape[0], K.shape[1])
            )

        n_samples = K.shape[0]
        self.K_fit_rows_ = np.sum(K, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n_samples1, n_samples2)
            Kernel matrix.

        copy : bool, default=True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : ndarray of shape (n_samples1, n_samples2)
            Returns the instance itself.
        """
        check_is_fitted(self)

        K = self._validate_data(K, copy=copy, dtype=FLOAT_DTYPES, reset=False)

        K_pred_cols = (np.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, np.newaxis]

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K

    def _more_tags(self):
        return {"pairwise": True}

    # TODO: Remove in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `_pairwise` was deprecated in "
        "version 0.24 and will be removed in 1.1."
    )
    @property
    def _pairwise(self):
        return True


def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features + 1)
        Same data with dummy feature added as first column.

    Examples
    --------
    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[1., 0., 1.],
           [1., 1., 0.]])
    """
    X = check_array(X, accept_sparse=["csc", "csr", "coo"], dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    if sparse.issparse(X):
        if sparse.isspmatrix_coo(X):
            # Shift columns to the right.
            col = X.col + 1
            # Column indices of dummy feature are 0 everywhere.
            col = np.concatenate((np.zeros(n_samples), col))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            row = np.concatenate((np.arange(n_samples), X.row))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.full(n_samples, value), X.data))
            return sparse.coo_matrix((data, (row, col)), shape)
        elif sparse.isspmatrix_csc(X):
            # Shift index pointers since we need to add n_samples elements.
            indptr = X.indptr + n_samples
            # indptr[0] must be 0.
            indptr = np.concatenate((np.array([0]), indptr))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            indices = np.concatenate((np.arange(n_samples), X.indices))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.full(n_samples, value), X.data))
            return sparse.csc_matrix((data, indices, indptr), shape)
        else:
            klass = X.__class__
            return klass(add_dummy_feature(X.tocoo(), value))
    else:
        return np.hstack((np.full((n_samples, 1), value), X))


class QuantileTransformer(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.19

    Parameters
    ----------
    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, default=1e5
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    Attributes
    ----------
    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    quantiles_ : ndarray of shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.

    references_ : ndarray of shape (n_quantiles, )
        Quantiles of references.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    quantile_transform : Equivalent function without the estimator API.
    PowerTransformer : Perform mapping to a normal distribution using a power
        transform.
    StandardScaler : Perform standardization that is faster, but less robust
        to outliers.
    RobustScaler : Perform robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
    >>> qt.fit_transform(X)
    array([...])
    """

    def __init__(
        self,
        *,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=int(1e5),
        random_state=None,
        copy=True,
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn(
                "'ignore_implicit_zeros' takes effect only with"
                " sparse matrix. This parameter has no effect."
            )

        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for col in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample, replace=False
                )
                col = col.take(subsample_idx, mode="clip")
            self.quantiles_.append(np.nanpercentile(col, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # Due to floating-point precision error in `np.nanpercentile`,
        # make sure that quantiles are monotonically increasing.
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative. If a sparse matrix is provided,
            it will be converted into a sparse ``csc_matrix``.
        """
        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for feature_idx in range(n_features):
            column_nnz_data = X.data[X.indptr[feature_idx] : X.indptr[feature_idx + 1]]
            if len(column_nnz_data) > self.subsample:
                column_subsample = self.subsample * len(column_nnz_data) // n_samples
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=column_subsample, dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=self.subsample, dtype=X.dtype)
                column_data[:column_subsample] = random_state.choice(
                    column_nnz_data, size=column_subsample, replace=False
                )
            else:
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=len(column_nnz_data), dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=n_samples, dtype=X.dtype)
                column_data[: len(column_nnz_data)] = column_nnz_data

            if not column_data.size:
                # if no nnz, an error will be raised for computing the
                # quantiles. Force the quantiles to be zeros.
                self.quantiles_.append([0] * len(references))
            else:
                self.quantiles_.append(np.nanpercentile(column_data, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # due to floating-point precision error in `np.nanpercentile`,
        # make sure the quantiles are monotonically increasing
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        y : None
            Ignored.

        Returns
        -------
        self : object
           Fitted transformer.
        """
        if self.n_quantiles <= 0:
            raise ValueError(
                "Invalid value for 'n_quantiles': %d. "
                "The number of quantiles must be at least one."
                % self.n_quantiles
            )

        if self.subsample <= 0:
            raise ValueError(
                "Invalid value for 'subsample': %d. "
                "The number of subsamples must be at least one."
                % self.subsample
            )

        if self.n_quantiles > self.subsample:
            raise ValueError(
                "The number of quantiles cannot be greater than"
                " the number of samples used. Got {} quantiles"
                " and {} samples.".format(self.n_quantiles, self.subsample)
            )

        X = self._check_inputs(X, in_fit=True, copy=False)
        n_samples = X.shape[0]

        if self.n_quantiles > n_samples:
            warnings.warn(
                "n_quantiles (%s) is greater than the total number "
                "of samples (%s). n_quantiles is set to "
                "n_samples." % (self.n_quantiles, n_samples)
            )
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)

        return self

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature."""

        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = stats.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = 0.5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1], -self.references_[::-1])
            )
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite, self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse="csc",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if (
                not accept_sparse_negative
                and not self.ignore_implicit_zeros
                and (sparse.issparse(X) and np.any(X.data < 0))
            ):
                raise ValueError(
                    "QuantileTransformer only accepts non-negative sparse matrices."
                )

        # check the output distribution
        if self.output_distribution not in ("normal", "uniform"):
            raise ValueError(
                "'output_distribution' has to be either 'normal'"
                " or 'uniform'. Got '{}' instead.".format(self.output_distribution)
            )

        return X

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                column_slice = slice(X.indptr[feature_idx], X.indptr[feature_idx + 1])
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice], self.quantiles_[:, feature_idx], inverse
                )
        else:
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
                )

        return X

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        return self._transform(X, inverse=False)

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(
            X, in_fit=False, accept_sparse_negative=True, copy=self.copy
        )

        return self._transform(X, inverse=True)

    def _more_tags(self):
        return {"allow_nan": True}


def quantile_transform(
    X,
    *,
    axis=0,
    n_quantiles=1000,
    output_distribution="uniform",
    ignore_implicit_zeros=False,
    subsample=int(1e5),
    random_state=None,
    copy=True,
):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to transform.

    axis : int, default=0
        Axis used to compute the means and standard deviations along. If 0,
        transform each feature, otherwise (if 1) transform each sample.

    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, default=1e5
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array). If True, a copy of `X` is transformed,
        leaving the original `X` unchanged

        ..versionchanged:: 0.23
            The default value of `copy` changed from False to True in 0.23.

    Returns
    -------
    Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import quantile_transform
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
    array([...])

    See Also
    --------
    QuantileTransformer : Performs quantile-based scaling using the
        Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).
    power_transform : Maps data to a normal distribution using a
        power transformation.
    scale : Performs standardization that is faster, but less robust
        to outliers.
    robust_scale : Performs robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.quantile_transform` unless
        you know what you are doing. A common mistake is to apply it
        to the entire data *before* splitting into training and
        test sets. This will bias the model evaluation because
        information would have leaked from the test set to the
        training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.QuantileTransformer` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking:`pipe = make_pipeline(QuantileTransformer(),
        LogisticRegression())`.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    n = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=subsample,
        ignore_implicit_zeros=ignore_implicit_zeros,
        random_state=random_state,
        copy=copy,
    )
    if axis == 0:
        return n.fit_transform(X)
    elif axis == 1:
        return n.fit_transform(X.T).T
    else:
        raise ValueError(
            "axis should be either equal to 0 or 1. Got axis={}".format(axis)
        )


class PowerTransformer(_OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        Set to False to perform inplace computation during transformation.

    Attributes
    ----------
    lambdas_ : ndarray of float of shape (n_features,)
        The parameters of the power transformation for the selected features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    power_transform : Equivalent function without the estimator API.

    QuantileTransformer : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer()
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]
    """

    def __init__(self, method="yeo-johnson", *, standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        """Estimate the optimal parameter lambda for each feature.

        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._fit(X, y=y, force_transform=False)
        return self

    def fit_transform(self, X, y=None):
        """Fit `PowerTransformer` to `X`, then transform `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters
            and to be transformed using a power transformation.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self._fit(X, y, force_transform=True)

    def _fit(self, X, y=None, force_transform=False):
        X = self._check_input(X, in_fit=True, check_positive=True, check_method=True)

        if not self.copy and not force_transform:  # if call from fit()
            X = X.copy()  # force copy so that fit does not change X inplace

        optim_function = {
            "box-cox": self._box_cox_optimize,
            "yeo-johnson": self._yeo_johnson_optimize,
        }[self.method]
        with np.errstate(invalid="ignore"):  # hide NaN warnings
            self.lambdas_ = np.array([optim_function(col) for col in X.T])

        if self.standardize or force_transform:
            transform_function = {
                "box-cox": boxcox,
                "yeo-johnson": self._yeo_johnson_transform,
            }[self.method]
            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            self._scaler = StandardScaler(copy=False)
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)

        return X

    def transform(self, X):
        """Apply the power transform to each feature using the fitted lambdas.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed using a power transformation.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_positive=True, check_shape=True)

        transform_function = {
            "box-cox": boxcox,
            "yeo-johnson": self._yeo_johnson_transform,
        }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid="ignore"):  # hide NaN warnings
                X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            X = self._scaler.transform(X)

        return X

    def inverse_transform(self, X):
        """Apply the inverse power transformation using the fitted lambdas.

        The inverse of the Box-Cox transformation is given by::

            if lambda_ == 0:
                X = exp(X_trans)
            else:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_)

        The inverse of the Yeo-Johnson transformation is given by::

            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The original data.
        """
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        if self.standardize:
            X = self._scaler.inverse_transform(X)

        inv_fun = {
            "box-cox": self._box_cox_inverse_tranform,
            "yeo-johnson": self._yeo_johnson_inverse_transform,
        }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid="ignore"):  # hide NaN warnings
                X[:, i] = inv_fun(X[:, i], lmbda)

        return X

    def _box_cox_inverse_tranform(self, x, lmbda):
        """Return inverse-transformed input x following Box-Cox inverse
        transform with parameter lambda.
        """
        if lmbda == 0:
            x_inv = np.exp(x)
        else:
            x_inv = (x * lmbda + 1) ** (1 / lmbda)

        return x_inv

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        """
        x_inv = np.zeros_like(x)
        pos = x >= 0

        # when x >= 0
        if abs(lmbda) < np.spacing(1.0):
            x_inv[pos] = np.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.0):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-x[~pos])

        return x_inv

    def _yeo_johnson_transform(self, x, lmbda):
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """

        out = np.zeros_like(x)
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < np.spacing(1.0):
            out[pos] = np.log1p(x[pos])
        else:  # lmbda != 0
            out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.0):
            out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            out[~pos] = -np.log1p(-x[~pos])

        return out

    def _box_cox_optimize(self, x):
        """Find and return optimal lambda parameter of the Box-Cox transform by
        MLE, for observed data x.

        We here use scipy builtins which uses the brent optimizer.
        """
        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)

        return lmbda

    def _yeo_johnson_optimize(self, x):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed data x.

        Like for Box-Cox, MLE is done via the brent optimizer.
        """

        def _neg_log_likelihood(lmbda):
            """Return the negative log likelihood of the observed data x as a
            function of lambda."""
            x_trans = self._yeo_johnson_transform(x, lmbda)
            n_samples = x.shape[0]

            loglike = -n_samples / 2 * np.log(x_trans.var())
            loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()

            return -loglike

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        # choosing bracket -2, 2 like for boxcox
        return optimize.brent(_neg_log_likelihood, brack=(-2, 2))

    def _check_input(
        self, X, in_fit, check_positive=False, check_shape=False, check_method=False
    ):
        """Validate the input before fit and transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        in_fit : bool
            Whether or not `_check_input` is called from `fit` or other
            methods, e.g. `predict`, `transform`, etc.

        check_positive : bool, default=False
            If True, check that all data is positive and non-zero (only if
            ``self.method=='box-cox'``).

        check_shape : bool, default=False
            If True, check that n_features matches the length of self.lambdas_

        check_method : bool, default=False
            If True, check that the transformation method is valid.
        """
        X = self._validate_data(
            X,
            ensure_2d=True,
            dtype=FLOAT_DTYPES,
            copy=self.copy,
            force_all_finite="allow-nan",
            reset=in_fit,
        )

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            if check_positive and self.method == "box-cox" and np.nanmin(X) <= 0:
                raise ValueError(
                    "The Box-Cox transformation can only be "
                    "applied to strictly positive data"
                )

        if check_shape and not X.shape[1] == len(self.lambdas_):
            raise ValueError(
                "Input data has a different number of features "
                "than fitting data. Should have {n}, data has {m}".format(
                    n=len(self.lambdas_), m=X.shape[1]
                )
            )

        valid_methods = ("box-cox", "yeo-johnson")
        if check_method and self.method not in valid_methods:
            raise ValueError(
                "'method' must be one of {}, got {} instead.".format(
                    valid_methods, self.method
                )
            )

        return X

    def _more_tags(self):
        return {"allow_nan": True}


def power_transform(X, method="yeo-johnson", *, standardize=True, copy=True):
    """
    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, power_transform supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.


    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to be transformed using a power transformation.

    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

        .. versionchanged:: 0.23
            The default value of the `method` parameter changed from
            'box-cox' to 'yeo-johnson' in 0.23.

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        Set to False to perform inplace computation during transformation.

    Returns
    -------
    X_trans : ndarray of shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import power_transform
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(power_transform(data, method='box-cox'))
    [[-1.332... -0.707...]
     [ 0.256... -0.707...]
     [ 1.076...  1.414...]]

    .. warning:: Risk of data leak.
        Do not use :func:`~sklearn.preprocessing.power_transform` unless you
        know what you are doing. A common mistake is to apply it to the entire
        data *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.PowerTransformer` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking, e.g.: `pipe = make_pipeline(PowerTransformer(),
        LogisticRegression())`.

    See Also
    --------
    PowerTransformer : Equivalent transformation with the
        Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    quantile_transform : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    """
    pt = PowerTransformer(method=method, standardize=standardize, copy=copy)
    return pt.fit_transform(X)
