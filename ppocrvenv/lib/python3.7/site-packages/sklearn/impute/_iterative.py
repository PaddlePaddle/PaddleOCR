from time import time
from collections import namedtuple
import warnings

from scipy import stats
import numpy as np

from ..base import clone
from ..exceptions import ConvergenceWarning
from ..preprocessing import normalize
from ..utils import check_array, check_random_state, _safe_indexing, is_scalar_nan
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
from ..utils._mask import _get_mask

from ._base import _BaseImputer
from ._base import SimpleImputer
from ._base import _check_inputs_dtype


_ImputerTriplet = namedtuple(
    "_ImputerTriplet", ["feat_idx", "neighbor_feat_idx", "estimator"]
)


class IterativeImputer(_BaseImputer):
    """Multivariate imputer that estimates each feature from all the others.

    A strategy for imputing missing values by modeling each feature with
    missing values as a function of other features in a round-robin fashion.

    Read more in the :ref:`User Guide <iterative_imputer>`.

    .. versionadded:: 0.21

    .. note::

      This estimator is still **experimental** for now: the predictions
      and the API might change without any deprecation cycle. To use it,
      you need to explicitly import `enable_iterative_imputer`::

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_iterative_imputer  # noqa
        >>> # now you can import normally from sklearn.impute
        >>> from sklearn.impute import IterativeImputer

    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If `sample_posterior=True`, the estimator must support
        `return_std` in its `predict` method.

    missing_values : int or np.nan, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    sample_posterior : bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        `return_std` in its `predict` method if set to `True`. Set to
        `True` if using `IterativeImputer` for multiple imputations.

    max_iter : int, default=10
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each feature with missing values. The stopping criterion
        is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
        where `X_t` is `X` at iteration `t`. Note that early stopping is only
        applied if `sample_posterior=False`.

    tol : float, default=1e-3
        Tolerance of the stopping condition.

    n_nearest_features : int, default=None
        Number of other features to use to estimate the missing values of
        each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of features throughout the
        imputation process, the neighbor features are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of features is huge. If `None`, all features will be used.

    initial_strategy : {'mean', 'median', 'most_frequent', 'constant'}, \
            default='mean'
        Which strategy to use to initialize the missing values. Same as the
        `strategy` parameter in :class:`~sklearn.impute.SimpleImputer`.

    imputation_order : {'ascending', 'descending', 'roman', 'arabic', \
            'random'}, default='ascending'
        The order in which the features will be imputed. Possible values:

        - `'ascending'`: From features with fewest missing values to most.
        - `'descending'`: From features with most missing values to fewest.
        - `'roman'`: Left to right.
        - `'arabic'`: Right to left.
        - `'random'`: A random order for each round.

    skip_complete : bool, default=False
        If `True` then features with missing values during :meth:`transform`
        which did not have any missing values during :meth:`fit` will be
        imputed with the initial imputation method only. Set to `True` if you
        have many features with no missing values at both :meth:`fit` and
        :meth:`transform` time to save compute.

    min_value : float or array-like of shape (n_features,), default=-np.inf
        Minimum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one min value for
        each feature. The default is `-np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    max_value : float or array-like of shape (n_features,), default=np.inf
        Maximum possible imputed value. Broadcast to shape `(n_features,)` if
        scalar. If array-like, expects shape `(n_features,)`, one max value for
        each feature. The default is `np.inf`.

        .. versionchanged:: 0.23
           Added support for array-like.

    verbose : int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator features if `n_nearest_features` is not `None`,
        the `imputation_order` if `random`, and the sampling from posterior if
        `sample_posterior=True`. Use an integer for determinism.
        See :term:`the Glossary <random_state>`.

    add_indicator : bool, default=False
        If `True`, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    initial_imputer_ : object of type :class:`~sklearn.impute.SimpleImputer`
        Imputer used to initialize the missing values.

    imputation_sequence_ : list of tuples
        Each tuple has `(feat_idx, neighbor_feat_idx, estimator)`, where
        `feat_idx` is the current feature to be imputed,
        `neighbor_feat_idx` is the array of other features used to impute the
        current feature, and `estimator` is the trained estimator used for
        the imputation. Length is `self.n_features_with_missing_ *
        self.n_iter_`.

    n_iter_ : int
        Number of iteration rounds that occurred. Will be less than
        `self.max_iter` if early stopping criterion was reached.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_features_with_missing_ : int
        Number of features with missing values.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generator or by `np.random`.

    See Also
    --------
    SimpleImputer : Univariate imputation of missing values.

    Notes
    -----
    To support imputation in inductive mode we store each feature's estimator
    during the :meth:`fit` phase, and predict without refitting (in order)
    during the :meth:`transform` phase.

    Features which contain all missing values at :meth:`fit` are discarded upon
    :meth:`transform`.

    References
    ----------
    .. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
        Multivariate Imputation by Chained Equations in R". Journal of
        Statistical Software 45: 1-67.
        <https://www.jstatsoft.org/article/view/v045i03>`_

    .. [2] `S. F. Buck, (1960). "A Method of Estimation of Missing Values in
        Multivariate Data Suitable for use with an Electronic Computer".
        Journal of the Royal Statistical Society 22(2): 302-306.
        <https://www.jstor.org/stable/2984099>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.experimental import enable_iterative_imputer
    >>> from sklearn.impute import IterativeImputer
    >>> imp_mean = IterativeImputer(random_state=0)
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    IterativeImputer(random_state=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> imp_mean.transform(X)
    array([[ 6.9584...,  2.       ,  3.        ],
           [ 4.       ,  2.6000...,  6.        ],
           [10.       ,  4.9999...,  9.        ]])
    """

    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
    ):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)

        self.estimator = estimator
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True,
    ):
        """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using
        the current estimates of all the other features. The `estimator` must
        support `return_std=True` in its `predict` method for this function
        to work.

        Parameters
        ----------
        X_filled : ndarray
            Input data with the most recent imputations.

        mask_missing_values : ndarray
            Input data's missing indicator matrix.

        feat_idx : int
            Index of the feature currently being imputed.

        neighbor_feat_idx : ndarray
            Indices of the features to be used in imputing `feat_idx`.

        estimator : object
            The estimator to use at this step of the round-robin imputation.
            If `sample_posterior=True`, the estimator must support
            `return_std` in its `predict` method.
            If None, it will be cloned from self._estimator.

        fit_mode : boolean, default=True
            Whether to fit and predict with the estimator or just predict.

        Returns
        -------
        X_filled : ndarray
            Input data with `X_filled[missing_row_mask, feat_idx]` updated.

        estimator : estimator with sklearn API
            The fitted estimator used to impute
            `X_filled[missing_row_mask, feat_idx]`.
        """
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            X_train = _safe_indexing(X_filled[:, neighbor_feat_idx], ~missing_row_mask)
            y_train = _safe_indexing(X_filled[:, feat_idx], ~missing_row_mask)
            estimator.fit(X_train, y_train)

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        # get posterior samples if there is at least one missing value
        X_test = _safe_indexing(X_filled[:, neighbor_feat_idx], missing_row_mask)
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # two types of problems: (1) non-positive sigmas
            # (2) mus outside legal range of min_value and max_value
            # (results in inf sample)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value[feat_idx] - mus) / sigmas
            b = (self._max_value[feat_idx] - mus) / sigmas

            truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
            imputed_values[inrange_mask] = truncated_normal.rvs(
                random_state=self.random_state_
            )
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(
                imputed_values, self._min_value[feat_idx], self._max_value[feat_idx]
            )

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, estimator

    def _get_neighbor_feat_idx(self, n_features, feat_idx, abs_corr_mat):
        """Get a list of other features to predict `feat_idx`.

        If `self.n_nearest_features` is less than or equal to the total
        number of features, then use a probability proportional to the absolute
        correlation between `feat_idx` and each other feature to randomly
        choose a subsample of the other features (without replacement).

        Parameters
        ----------
        n_features : int
            Number of features in `X`.

        feat_idx : int
            Index of the feature currently being imputed.

        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X`. The diagonal has been zeroed
            out and each feature has been normalized to sum to 1. Can be None.

        Returns
        -------
        neighbor_feat_idx : array-like
            The features to use to impute `feat_idx`.
        """
        if self.n_nearest_features is not None and self.n_nearest_features < n_features:
            p = abs_corr_mat[:, feat_idx]
            neighbor_feat_idx = self.random_state_.choice(
                np.arange(n_features), self.n_nearest_features, replace=False, p=p
            )
        else:
            inds_left = np.arange(feat_idx)
            inds_right = np.arange(feat_idx + 1, n_features)
            neighbor_feat_idx = np.concatenate((inds_left, inds_right))
        return neighbor_feat_idx

    def _get_ordered_idx(self, mask_missing_values):
        """Decide in what order we will update the features.

        As a homage to the MICE R package, we will have 4 main options of
        how to order the updates, and use a random order if anything else
        is specified.

        Also, this function skips features which have no missing values.

        Parameters
        ----------
        mask_missing_values : array-like, shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features.

        Returns
        -------
        ordered_idx : ndarray, shape (n_features,)
            The order in which to impute the features.
        """
        frac_of_missing_values = mask_missing_values.mean(axis=0)
        if self.skip_complete:
            missing_values_idx = np.flatnonzero(frac_of_missing_values)
        else:
            missing_values_idx = np.arange(np.shape(frac_of_missing_values)[0])
        if self.imputation_order == "roman":
            ordered_idx = missing_values_idx
        elif self.imputation_order == "arabic":
            ordered_idx = missing_values_idx[::-1]
        elif self.imputation_order == "ascending":
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
        elif self.imputation_order == "descending":
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
        elif self.imputation_order == "random":
            ordered_idx = missing_values_idx
            self.random_state_.shuffle(ordered_idx)
        else:
            raise ValueError(
                "Got an invalid imputation order: '{0}'. It must "
                "be one of the following: 'roman', 'arabic', "
                "'ascending', 'descending', or "
                "'random'.".format(self.imputation_order)
            )
        return ordered_idx

    def _get_abs_corr_mat(self, X_filled, tolerance=1e-6):
        """Get absolute correlation matrix between features.

        Parameters
        ----------
        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        tolerance : float, default=1e-6
            `abs_corr_mat` can have nans, which will be replaced
            with `tolerance`.

        Returns
        -------
        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X` at the beginning of the
            current round. The diagonal has been zeroed out and each feature's
            absolute correlations with all others have been normalized to sum
            to 1.
        """
        n_features = X_filled.shape[1]
        if self.n_nearest_features is None or self.n_nearest_features >= n_features:
            return None
        with np.errstate(invalid="ignore"):
            # if a feature in the neighborhood has only a single value
            # (e.g., categorical feature), the std. dev. will be null and
            # np.corrcoef will raise a warning due to a division by zero
            abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
        # np.corrcoef is not defined for features with zero std
        abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
        # ensures exploration, i.e. at least some probability of sampling
        np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
        # features are not their own neighbors
        np.fill_diagonal(abs_corr_mat, 0)
        # needs to sum to 1 for np.random.choice sampling
        abs_corr_mat = normalize(abs_corr_mat, norm="l1", axis=0, copy=False)
        return abs_corr_mat

    def _initial_imputation(self, X, in_fit=False):
        """Perform initial imputation for input `X`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        in_fit : bool, default=False
            Whether function is called in :meth:`fit`.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        mask_missing_values : ndarray, shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features.

        X_missing_mask : ndarray, shape (n_samples, n_features)
            Input data's mask matrix indicating missing datapoints, where
            `n_samples` is the number of samples and `n_features` is the
            number of features.
        """
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            force_all_finite=force_all_finite,
        )
        _check_inputs_dtype(X, self.missing_values)

        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()
        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                missing_values=self.missing_values, strategy=self.initial_strategy
            )
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            X_filled = self.initial_imputer_.transform(X)

        valid_mask = np.flatnonzero(
            np.logical_not(np.isnan(self.initial_imputer_.statistics_))
        )
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]

        return Xt, X_filled, mask_missing_values, X_missing_mask

    @staticmethod
    def _validate_limit(limit, limit_type, n_features):
        """Validate the limits (min/max) of the feature values.

        Converts scalar min/max limits to vectors of shape `(n_features,)`.

        Parameters
        ----------
        limit: scalar or array-like
            The user-specified limit (i.e, min_value or max_value).
        limit_type: {'max', 'min'}
            Type of limit to validate.
        n_features: int
            Number of features in the dataset.

        Returns
        -------
        limit: ndarray, shape(n_features,)
            Array of limits, one for each feature.
        """
        limit_bound = np.inf if limit_type == "max" else -np.inf
        limit = limit_bound if limit is None else limit
        if np.isscalar(limit):
            limit = np.full(n_features, limit)
        limit = check_array(limit, force_all_finite=False, copy=False, ensure_2d=False)
        if not limit.shape[0] == n_features:
            raise ValueError(
                f"'{limit_type}_value' should be of "
                f"shape ({n_features},) when an array-like "
                f"is provided. Got {limit.shape}, instead."
            )
        return limit

    def fit_transform(self, X, y=None):
        """Fit the imputer on `X` and return the transformed `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        self.random_state_ = getattr(
            self, "random_state_", check_random_state(self.random_state)
        )

        if self.max_iter < 0:
            raise ValueError(
                "'max_iter' should be a positive integer. Got {} instead.".format(
                    self.max_iter
                )
            )

        if self.tol < 0:
            raise ValueError(
                "'tol' should be a non-negative float. Got {} instead.".format(self.tol)
            )

        if self.estimator is None:
            from ..linear_model import BayesianRidge

            self._estimator = BayesianRidge()
        else:
            self._estimator = clone(self.estimator)

        self.imputation_sequence_ = []

        self.initial_imputer_ = None

        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
            X, in_fit=True
        )

        super()._fit_indicator(complete_mask)
        X_indicator = super()._transform_indicator(complete_mask)

        if self.max_iter == 0 or np.all(mask_missing_values):
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        # Edge case: a single feature. We return the initial ...
        if Xt.shape[1] == 1:
            self.n_iter_ = 0
            return super()._concatenate_indicator(Xt, X_indicator)

        self._min_value = self._validate_limit(self.min_value, "min", X.shape[1])
        self._max_value = self._validate_limit(self.max_value, "max", X.shape[1])

        if not np.all(np.greater(self._max_value, self._min_value)):
            raise ValueError("One (or more) features have min_value >= max_value.")

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)
        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.imputation_order == "random":
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(
                    n_features, feat_idx, abs_corr_mat
                )
                Xt, estimator = self._impute_one_feature(
                    Xt,
                    mask_missing_values,
                    feat_idx,
                    neighbor_feat_idx,
                    estimator=None,
                    fit_mode=True,
                )
                estimator_triplet = _ImputerTriplet(
                    feat_idx, neighbor_feat_idx, estimator
                )
                self.imputation_sequence_.append(estimator_triplet)

            if self.verbose > 1:
                print(
                    "[IterativeImputer] Ending imputation round "
                    "%d/%d, elapsed time %0.2f"
                    % (self.n_iter_, self.max_iter, time() - start_t)
                )

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf, axis=None)
                if self.verbose > 0:
                    print(
                        "[IterativeImputer] Change: {}, scaled tolerance: {} ".format(
                            inf_norm, normalized_tol
                        )
                    )
                if inf_norm < normalized_tol:
                    if self.verbose > 0:
                        print("[IterativeImputer] Early stopping criterion reached.")
                    break
                Xt_previous = Xt.copy()
        else:
            if not self.sample_posterior:
                warnings.warn(
                    "[IterativeImputer] Early stopping criterion not reached.",
                    ConvergenceWarning,
                )
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return super()._concatenate_indicator(Xt, X_indicator)

    def transform(self, X):
        """Impute all missing values in `X`.

        Note that this is stochastic, and that if `random_state` is not fixed,
        repeated calls, or permuted input, results will differ.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        check_is_fitted(self)

        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(X)

        X_indicator = super()._transform_indicator(complete_mask)

        if self.n_iter_ == 0 or np.all(mask_missing_values):
            return super()._concatenate_indicator(Xt, X_indicator)

        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        i_rnd = 0
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s" % (X.shape,))
        start_t = time()
        for it, estimator_triplet in enumerate(self.imputation_sequence_):
            Xt, _ = self._impute_one_feature(
                Xt,
                mask_missing_values,
                estimator_triplet.feat_idx,
                estimator_triplet.neighbor_feat_idx,
                estimator=estimator_triplet.estimator,
                fit_mode=False,
            )
            if not (it + 1) % imputations_per_round:
                if self.verbose > 1:
                    print(
                        "[IterativeImputer] Ending imputation round "
                        "%d/%d, elapsed time %0.2f"
                        % (i_rnd + 1, self.n_iter_, time() - start_t)
                    )
                i_rnd += 1

        Xt[~mask_missing_values] = X[~mask_missing_values]

        return super()._concatenate_indicator(Xt, X_indicator)

    def fit(self, X, y=None):
        """Fit the imputer on `X` and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self
