"""Fast Gradient Boosting decision trees for classification and regression."""
# Author: Nicolas Hug

from abc import ABC, abstractmethod
from functools import partial
import warnings

import numpy as np
from timeit import default_timer as time
from ...base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier
from ...utils import check_random_state, resample
from ...utils.validation import (
    check_is_fitted,
    check_consistent_length,
    _check_sample_weight,
)
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils.multiclass import check_classification_targets
from ...metrics import check_scoring
from ...model_selection import train_test_split
from ...preprocessing import LabelEncoder
from ._gradient_boosting import _update_raw_predictions
from .common import Y_DTYPE, X_DTYPE, X_BINNED_DTYPE

from .binning import _BinMapper
from .grower import TreeGrower
from .loss import _LOSSES
from .loss import BaseLoss


class BaseHistGradientBoosting(BaseEstimator, ABC):
    """Base class for histogram-based gradient boosting estimators."""

    @abstractmethod
    def __init__(
        self,
        loss,
        *,
        learning_rate,
        max_iter,
        max_leaf_nodes,
        max_depth,
        min_samples_leaf,
        l2_regularization,
        max_bins,
        categorical_features,
        monotonic_cst,
        warm_start,
        early_stopping,
        scoring,
        validation_fraction,
        n_iter_no_change,
        tol,
        verbose,
        random_state,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.monotonic_cst = monotonic_cst
        self.categorical_features = categorical_features
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _validate_parameters(self):
        """Validate parameters passed to __init__.

        The parameters that are directly passed to the grower are checked in
        TreeGrower."""

        if self.loss not in self._VALID_LOSSES and not isinstance(self.loss, BaseLoss):
            raise ValueError(
                "Loss {} is not supported for {}. Accepted losses: {}.".format(
                    self.loss, self.__class__.__name__, ", ".join(self._VALID_LOSSES)
                )
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate={} must be strictly positive".format(self.learning_rate)
            )
        if self.max_iter < 1:
            raise ValueError(
                "max_iter={} must not be smaller than 1.".format(self.max_iter)
            )
        if self.n_iter_no_change < 0:
            raise ValueError(
                "n_iter_no_change={} must be positive.".format(self.n_iter_no_change)
            )
        if self.validation_fraction is not None and self.validation_fraction <= 0:
            raise ValueError(
                "validation_fraction={} must be strictly positive, or None.".format(
                    self.validation_fraction
                )
            )
        if self.tol < 0:
            raise ValueError("tol={} must not be smaller than 0.".format(self.tol))

        if not (2 <= self.max_bins <= 255):
            raise ValueError(
                "max_bins={} should be no smaller than 2 "
                "and no larger than 255.".format(self.max_bins)
            )

        if self.monotonic_cst is not None and self.n_trees_per_iteration_ != 1:
            raise ValueError(
                "monotonic constraints are not supported for multiclass classification."
            )

    def _check_categories(self, X):
        """Check and validate categorical features in X

        Return
        ------
        is_categorical : ndarray of shape (n_features,) or None, dtype=bool
            Indicates whether a feature is categorical. If no feature is
            categorical, this is None.
        known_categories : list of size n_features or None
            The list contains, for each feature:
                - an array of shape (n_categories,) with the unique cat values
                - None if the feature is not categorical
            None if no feature is categorical.
        """
        if self.categorical_features is None:
            return None, None

        categorical_features = np.asarray(self.categorical_features)

        if categorical_features.size == 0:
            return None, None

        if categorical_features.dtype.kind not in ("i", "b"):
            raise ValueError(
                "categorical_features must be an array-like of "
                "bools or array-like of ints."
            )

        n_features = X.shape[1]

        # check for categorical features as indices
        if categorical_features.dtype.kind == "i":
            if (
                np.max(categorical_features) >= n_features
                or np.min(categorical_features) < 0
            ):
                raise ValueError(
                    "categorical_features set as integer "
                    "indices must be in [0, n_features - 1]"
                )
            is_categorical = np.zeros(n_features, dtype=bool)
            is_categorical[categorical_features] = True
        else:
            if categorical_features.shape[0] != n_features:
                raise ValueError(
                    "categorical_features set as a boolean mask "
                    "must have shape (n_features,), got: "
                    f"{categorical_features.shape}"
                )
            is_categorical = categorical_features

        if not np.any(is_categorical):
            return None, None

        # compute the known categories in the training data. We need to do
        # that here instead of in the BinMapper because in case of early
        # stopping, the mapper only gets a fraction of the training data.
        known_categories = []

        for f_idx in range(n_features):
            if is_categorical[f_idx]:
                categories = np.unique(X[:, f_idx])
                missing = np.isnan(categories)
                if missing.any():
                    categories = categories[~missing]

                if categories.size > self.max_bins:
                    raise ValueError(
                        f"Categorical feature at index {f_idx} is "
                        "expected to have a "
                        f"cardinality <= {self.max_bins}"
                    )

                if (categories >= self.max_bins).any():
                    raise ValueError(
                        f"Categorical feature at index {f_idx} is "
                        "expected to be encoded with "
                        f"values < {self.max_bins}"
                    )
            else:
                categories = None
            known_categories.append(categories)

        return is_categorical, known_categories

    def fit(self, X, y, sample_weight=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) default=None
            Weights of training data.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Fitted estimator.
        """
        fit_start_time = time()
        acc_find_split_time = 0.0  # time spent finding the best splits
        acc_apply_split_time = 0.0  # time spent splitting nodes
        acc_compute_hist_time = 0.0  # time spent computing histograms
        # time spent predicting X for gradient and hessians update
        acc_prediction_time = 0.0
        X, y = self._validate_data(X, y, dtype=[X_DTYPE], force_all_finite=False)
        y = self._encode_y(y)
        check_consistent_length(X, y)
        # Do not create unit sample weights by default to later skip some
        # computation
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
            # TODO: remove when PDP supports sample weights
            self._fitted_with_sw = True

        rng = check_random_state(self.random_state)

        # When warm starting, we want to re-use the same seed that was used
        # the first time fit was called (e.g. for subsampling or for the
        # train/val split).
        if not (self.warm_start and self._is_fitted()):
            self._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")

        self._validate_parameters()

        # used for validation in predict
        n_samples, self._n_features = X.shape

        self.is_categorical_, known_categories = self._check_categories(X)

        # we need this stateful variable to tell raw_predict() that it was
        # called from fit() (this current method), and that the data it has
        # received is pre-binned.
        # predicting is faster on pre-binned data, so we want early stopping
        # predictions to be made on pre-binned data. Unfortunately the _scorer
        # can only call predict() or predict_proba(), not raw_predict(), and
        # there's no way to tell the scorer that it needs to predict binned
        # data.
        self._in_fit = True

        # `_openmp_effective_n_threads` is used to take cgroups CPU quotes
        # into account when determine the maximum number of threads to use.
        n_threads = _openmp_effective_n_threads()

        if isinstance(self.loss, str):
            self._loss = self._get_loss(
                sample_weight=sample_weight, n_threads=n_threads
            )
        elif isinstance(self.loss, BaseLoss):
            self._loss = self.loss

        if self.early_stopping == "auto":
            self.do_early_stopping_ = n_samples > 10000
        else:
            self.do_early_stopping_ = self.early_stopping

        # create validation data if needed
        self._use_validation_data = self.validation_fraction is not None
        if self.do_early_stopping_ and self._use_validation_data:
            # stratify for classification
            stratify = y if hasattr(self._loss, "predict_proba") else None

            # Save the state of the RNG for the training and validation split.
            # This is needed in order to have the same split when using
            # warm starting.

            if sample_weight is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                    random_state=self._random_seed,
                )
                sample_weight_train = sample_weight_val = None
            else:
                # TODO: incorporate sample_weight in sampling here, as well as
                # stratify
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    sample_weight_train,
                    sample_weight_val,
                ) = train_test_split(
                    X,
                    y,
                    sample_weight,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                    random_state=self._random_seed,
                )
        else:
            X_train, y_train, sample_weight_train = X, y, sample_weight
            X_val = y_val = sample_weight_val = None

        # Bin the data
        # For ease of use of the API, the user-facing GBDT classes accept the
        # parameter max_bins, which doesn't take into account the bin for
        # missing values (which is always allocated). However, since max_bins
        # isn't the true maximal number of bins, all other private classes
        # (binmapper, histbuilder...) accept n_bins instead, which is the
        # actual total number of bins. Everywhere in the code, the
        # convention is that n_bins == max_bins + 1
        n_bins = self.max_bins + 1  # + 1 for missing values
        self._bin_mapper = _BinMapper(
            n_bins=n_bins,
            is_categorical=self.is_categorical_,
            known_categories=known_categories,
            random_state=self._random_seed,
            n_threads=n_threads,
        )
        X_binned_train = self._bin_data(X_train, is_training_data=True)
        if X_val is not None:
            X_binned_val = self._bin_data(X_val, is_training_data=False)
        else:
            X_binned_val = None

        # Uses binned data to check for missing values
        has_missing_values = (
            (X_binned_train == self._bin_mapper.missing_values_bin_idx_)
            .any(axis=0)
            .astype(np.uint8)
        )

        if self.verbose:
            print("Fitting gradient boosted rounds:")

        n_samples = X_binned_train.shape[0]

        # First time calling fit, or no warm start
        if not (self._is_fitted() and self.warm_start):
            # Clear random state and score attributes
            self._clear_state()

            # initialize raw_predictions: those are the accumulated values
            # predicted by the trees for the training data. raw_predictions has
            # shape (n_trees_per_iteration, n_samples) where
            # n_trees_per_iterations is n_classes in multiclass classification,
            # else 1.
            self._baseline_prediction = self._loss.get_baseline_prediction(
                y_train, sample_weight_train, self.n_trees_per_iteration_
            )
            raw_predictions = np.zeros(
                shape=(self.n_trees_per_iteration_, n_samples),
                dtype=self._baseline_prediction.dtype,
            )
            raw_predictions += self._baseline_prediction

            # predictors is a matrix (list of lists) of TreePredictor objects
            # with shape (n_iter_, n_trees_per_iteration)
            self._predictors = predictors = []

            # Initialize structures and attributes related to early stopping
            self._scorer = None  # set if scoring != loss
            raw_predictions_val = None  # set if scoring == loss and use val
            self.train_score_ = []
            self.validation_score_ = []

            if self.do_early_stopping_:
                # populate train_score and validation_score with the
                # predictions of the initial model (before the first tree)

                if self.scoring == "loss":
                    # we're going to compute scoring w.r.t the loss. As losses
                    # take raw predictions as input (unlike the scorers), we
                    # can optimize a bit and avoid repeating computing the
                    # predictions of the previous trees. We'll re-use
                    # raw_predictions (as it's needed for training anyway) for
                    # evaluating the training loss, and create
                    # raw_predictions_val for storing the raw predictions of
                    # the validation data.

                    if self._use_validation_data:
                        raw_predictions_val = np.zeros(
                            shape=(self.n_trees_per_iteration_, X_binned_val.shape[0]),
                            dtype=self._baseline_prediction.dtype,
                        )

                        raw_predictions_val += self._baseline_prediction

                    self._check_early_stopping_loss(
                        raw_predictions,
                        y_train,
                        sample_weight_train,
                        raw_predictions_val,
                        y_val,
                        sample_weight_val,
                    )
                else:
                    self._scorer = check_scoring(self, self.scoring)
                    # _scorer is a callable with signature (est, X, y) and
                    # calls est.predict() or est.predict_proba() depending on
                    # its nature.
                    # Unfortunately, each call to _scorer() will compute
                    # the predictions of all the trees. So we use a subset of
                    # the training set to compute train scores.

                    # Compute the subsample set
                    (
                        X_binned_small_train,
                        y_small_train,
                        sample_weight_small_train,
                    ) = self._get_small_trainset(
                        X_binned_train, y_train, sample_weight_train, self._random_seed
                    )

                    self._check_early_stopping_scorer(
                        X_binned_small_train,
                        y_small_train,
                        sample_weight_small_train,
                        X_binned_val,
                        y_val,
                        sample_weight_val,
                    )
            begin_at_stage = 0

        # warm start: this is not the first time fit was called
        else:
            # Check that the maximum number of iterations is not smaller
            # than the number of iterations from the previous fit
            if self.max_iter < self.n_iter_:
                raise ValueError(
                    "max_iter=%d must be larger than or equal to "
                    "n_iter_=%d when warm_start==True" % (self.max_iter, self.n_iter_)
                )

            # Convert array attributes to lists
            self.train_score_ = self.train_score_.tolist()
            self.validation_score_ = self.validation_score_.tolist()

            # Compute raw predictions
            raw_predictions = self._raw_predict(X_binned_train, n_threads=n_threads)
            if self.do_early_stopping_ and self._use_validation_data:
                raw_predictions_val = self._raw_predict(
                    X_binned_val, n_threads=n_threads
                )
            else:
                raw_predictions_val = None

            if self.do_early_stopping_ and self.scoring != "loss":
                # Compute the subsample set
                (
                    X_binned_small_train,
                    y_small_train,
                    sample_weight_small_train,
                ) = self._get_small_trainset(
                    X_binned_train, y_train, sample_weight_train, self._random_seed
                )

            # Get the predictors from the previous fit
            predictors = self._predictors

            begin_at_stage = self.n_iter_

        # initialize gradients and hessians (empty arrays).
        # shape = (n_trees_per_iteration, n_samples).
        gradients, hessians = self._loss.init_gradients_and_hessians(
            n_samples=n_samples,
            prediction_dim=self.n_trees_per_iteration_,
            sample_weight=sample_weight_train,
        )

        for iteration in range(begin_at_stage, self.max_iter):

            if self.verbose:
                iteration_start_time = time()
                print(
                    "[{}/{}] ".format(iteration + 1, self.max_iter), end="", flush=True
                )

            # Update gradients and hessians, inplace
            self._loss.update_gradients_and_hessians(
                gradients, hessians, y_train, raw_predictions, sample_weight_train
            )

            # Append a list since there may be more than 1 predictor per iter
            predictors.append([])

            # Build `n_trees_per_iteration` trees.
            for k in range(self.n_trees_per_iteration_):
                grower = TreeGrower(
                    X_binned_train,
                    gradients[k, :],
                    hessians[k, :],
                    n_bins=n_bins,
                    n_bins_non_missing=self._bin_mapper.n_bins_non_missing_,
                    has_missing_values=has_missing_values,
                    is_categorical=self.is_categorical_,
                    monotonic_cst=self.monotonic_cst,
                    max_leaf_nodes=self.max_leaf_nodes,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    l2_regularization=self.l2_regularization,
                    shrinkage=self.learning_rate,
                    n_threads=n_threads,
                )
                grower.grow()

                acc_apply_split_time += grower.total_apply_split_time
                acc_find_split_time += grower.total_find_split_time
                acc_compute_hist_time += grower.total_compute_hist_time

                if self._loss.need_update_leaves_values:
                    self._loss.update_leaves_values(
                        grower, y_train, raw_predictions[k, :], sample_weight_train
                    )

                predictor = grower.make_predictor(
                    binning_thresholds=self._bin_mapper.bin_thresholds_
                )
                predictors[-1].append(predictor)

                # Update raw_predictions with the predictions of the newly
                # created tree.
                tic_pred = time()
                _update_raw_predictions(raw_predictions[k, :], grower, n_threads)
                toc_pred = time()
                acc_prediction_time += toc_pred - tic_pred

            should_early_stop = False
            if self.do_early_stopping_:
                if self.scoring == "loss":
                    # Update raw_predictions_val with the newest tree(s)
                    if self._use_validation_data:
                        for k, pred in enumerate(self._predictors[-1]):
                            raw_predictions_val[k, :] += pred.predict_binned(
                                X_binned_val,
                                self._bin_mapper.missing_values_bin_idx_,
                                n_threads,
                            )

                    should_early_stop = self._check_early_stopping_loss(
                        raw_predictions,
                        y_train,
                        sample_weight_train,
                        raw_predictions_val,
                        y_val,
                        sample_weight_val,
                    )

                else:
                    should_early_stop = self._check_early_stopping_scorer(
                        X_binned_small_train,
                        y_small_train,
                        sample_weight_small_train,
                        X_binned_val,
                        y_val,
                        sample_weight_val,
                    )

            if self.verbose:
                self._print_iteration_stats(iteration_start_time)

            # maybe we could also early stop if all the trees are stumps?
            if should_early_stop:
                break

        if self.verbose:
            duration = time() - fit_start_time
            n_total_leaves = sum(
                predictor.get_n_leaf_nodes()
                for predictors_at_ith_iteration in self._predictors
                for predictor in predictors_at_ith_iteration
            )
            n_predictors = sum(
                len(predictors_at_ith_iteration)
                for predictors_at_ith_iteration in self._predictors
            )
            print(
                "Fit {} trees in {:.3f} s, ({} total leaves)".format(
                    n_predictors, duration, n_total_leaves
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent computing histograms:", acc_compute_hist_time
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent finding best splits:", acc_find_split_time
                )
            )
            print(
                "{:<32} {:.3f}s".format(
                    "Time spent applying splits:", acc_apply_split_time
                )
            )
            print(
                "{:<32} {:.3f}s".format("Time spent predicting:", acc_prediction_time)
            )

        self.train_score_ = np.asarray(self.train_score_)
        self.validation_score_ = np.asarray(self.validation_score_)
        del self._in_fit  # hard delete so we're sure it can't be used anymore
        return self

    def _is_fitted(self):
        return len(getattr(self, "_predictors", [])) > 0

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        for var in ("train_score_", "validation_score_"):
            if hasattr(self, var):
                delattr(self, var)

    def _get_small_trainset(self, X_binned_train, y_train, sample_weight_train, seed):
        """Compute the indices of the subsample set and return this set.

        For efficiency, we need to subsample the training set to compute scores
        with scorers.
        """
        # TODO: incorporate sample_weights here in `resample`
        subsample_size = 10000
        if X_binned_train.shape[0] > subsample_size:
            indices = np.arange(X_binned_train.shape[0])
            stratify = y_train if is_classifier(self) else None
            indices = resample(
                indices,
                n_samples=subsample_size,
                replace=False,
                random_state=seed,
                stratify=stratify,
            )
            X_binned_small_train = X_binned_train[indices]
            y_small_train = y_train[indices]
            if sample_weight_train is not None:
                sample_weight_small_train = sample_weight_train[indices]
            else:
                sample_weight_small_train = None
            X_binned_small_train = np.ascontiguousarray(X_binned_small_train)
            return (X_binned_small_train, y_small_train, sample_weight_small_train)
        else:
            return X_binned_train, y_train, sample_weight_train

    def _check_early_stopping_scorer(
        self,
        X_binned_small_train,
        y_small_train,
        sample_weight_small_train,
        X_binned_val,
        y_val,
        sample_weight_val,
    ):
        """Check if fitting should be early-stopped based on scorer.

        Scores are computed on validation data or on training data.
        """
        if is_classifier(self):
            y_small_train = self.classes_[y_small_train.astype(int)]

        if sample_weight_small_train is None:
            self.train_score_.append(
                self._scorer(self, X_binned_small_train, y_small_train)
            )
        else:
            self.train_score_.append(
                self._scorer(
                    self,
                    X_binned_small_train,
                    y_small_train,
                    sample_weight=sample_weight_small_train,
                )
            )

        if self._use_validation_data:
            if is_classifier(self):
                y_val = self.classes_[y_val.astype(int)]
            if sample_weight_val is None:
                self.validation_score_.append(self._scorer(self, X_binned_val, y_val))
            else:
                self.validation_score_.append(
                    self._scorer(
                        self, X_binned_val, y_val, sample_weight=sample_weight_val
                    )
                )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)

    def _check_early_stopping_loss(
        self,
        raw_predictions,
        y_train,
        sample_weight_train,
        raw_predictions_val,
        y_val,
        sample_weight_val,
    ):
        """Check if fitting should be early-stopped based on loss.

        Scores are computed on validation data or on training data.
        """

        self.train_score_.append(
            -self._loss(y_train, raw_predictions, sample_weight_train)
        )

        if self._use_validation_data:
            self.validation_score_.append(
                -self._loss(y_val, raw_predictions_val, sample_weight_val)
            )
            return self._should_stop(self.validation_score_)
        else:
            return self._should_stop(self.train_score_)

    def _should_stop(self, scores):
        """
        Return True (do early stopping) if the last n scores aren't better
        than the (n-1)th-to-last score, up to some tolerance.
        """
        reference_position = self.n_iter_no_change + 1
        if len(scores) < reference_position:
            return False

        # A higher score is always better. Higher tol means that it will be
        # harder for subsequent iteration to be considered an improvement upon
        # the reference score, and therefore it is more likely to early stop
        # because of the lack of significant improvement.
        reference_score = scores[-reference_position] + self.tol
        recent_scores = scores[-reference_position + 1 :]
        recent_improvements = [score > reference_score for score in recent_scores]
        return not any(recent_improvements)

    def _bin_data(self, X, is_training_data):
        """Bin data X.

        If is_training_data, then fit the _bin_mapper attribute.
        Else, the binned data is converted to a C-contiguous array.
        """

        description = "training" if is_training_data else "validation"
        if self.verbose:
            print(
                "Binning {:.3f} GB of {} data: ".format(X.nbytes / 1e9, description),
                end="",
                flush=True,
            )
        tic = time()
        if is_training_data:
            X_binned = self._bin_mapper.fit_transform(X)  # F-aligned array
        else:
            X_binned = self._bin_mapper.transform(X)  # F-aligned array
            # We convert the array to C-contiguous since predicting is faster
            # with this layout (training is faster on F-arrays though)
            X_binned = np.ascontiguousarray(X_binned)
        toc = time()
        if self.verbose:
            duration = toc - tic
            print("{:.3f} s".format(duration))

        return X_binned

    def _print_iteration_stats(self, iteration_start_time):
        """Print info about the current fitting iteration."""
        log_msg = ""

        predictors_of_ith_iteration = [
            predictors_list
            for predictors_list in self._predictors[-1]
            if predictors_list
        ]
        n_trees = len(predictors_of_ith_iteration)
        max_depth = max(
            predictor.get_max_depth() for predictor in predictors_of_ith_iteration
        )
        n_leaves = sum(
            predictor.get_n_leaf_nodes() for predictor in predictors_of_ith_iteration
        )

        if n_trees == 1:
            log_msg += "{} tree, {} leaves, ".format(n_trees, n_leaves)
        else:
            log_msg += "{} trees, {} leaves ".format(n_trees, n_leaves)
            log_msg += "({} on avg), ".format(int(n_leaves / n_trees))

        log_msg += "max depth = {}, ".format(max_depth)

        if self.do_early_stopping_:
            if self.scoring == "loss":
                factor = -1  # score_ arrays contain the negative loss
                name = "loss"
            else:
                factor = 1
                name = "score"
            log_msg += "train {}: {:.5f}, ".format(name, factor * self.train_score_[-1])
            if self._use_validation_data:
                log_msg += "val {}: {:.5f}, ".format(
                    name, factor * self.validation_score_[-1]
                )

        iteration_time = time() - iteration_start_time
        log_msg += "in {:0.3f}s".format(iteration_time)

        print(log_msg)

    def _raw_predict(self, X, n_threads=None):
        """Return the sum of the leaves values over all predictors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        n_threads : int, default=None
            Number of OpenMP threads to use. `_openmp_effective_n_threads` is called
            to determine the effective number of threads use, which takes cgroups CPU
            quotes into account. See the docstring of `_openmp_effective_n_threads`
            for details.

        Returns
        -------
        raw_predictions : array, shape (n_trees_per_iteration, n_samples)
            The raw predicted values.
        """
        is_binned = getattr(self, "_in_fit", False)
        dtype = X_BINNED_DTYPE if is_binned else X_DTYPE
        X = self._validate_data(X, dtype=dtype, force_all_finite=False, reset=False)
        check_is_fitted(self)
        if X.shape[1] != self._n_features:
            raise ValueError(
                "X has {} features but this estimator was trained with "
                "{} features.".format(X.shape[1], self._n_features)
            )
        n_samples = X.shape[0]
        raw_predictions = np.zeros(
            shape=(self.n_trees_per_iteration_, n_samples),
            dtype=self._baseline_prediction.dtype,
        )
        raw_predictions += self._baseline_prediction

        # We intentionally decouple the number of threads used at prediction
        # time from the number of threads used at fit time because the model
        # can be deployed on a different machine for prediction purposes.
        n_threads = _openmp_effective_n_threads(n_threads)
        self._predict_iterations(
            X, self._predictors, raw_predictions, is_binned, n_threads
        )
        return raw_predictions

    def _predict_iterations(self, X, predictors, raw_predictions, is_binned, n_threads):
        """Add the predictions of the predictors to raw_predictions."""
        if not is_binned:
            (
                known_cat_bitsets,
                f_idx_map,
            ) = self._bin_mapper.make_known_categories_bitsets()

        for predictors_of_ith_iteration in predictors:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                if is_binned:
                    predict = partial(
                        predictor.predict_binned,
                        missing_values_bin_idx=self._bin_mapper.missing_values_bin_idx_,
                        n_threads=n_threads,
                    )
                else:
                    predict = partial(
                        predictor.predict,
                        known_cat_bitsets=known_cat_bitsets,
                        f_idx_map=f_idx_map,
                        n_threads=n_threads,
                    )
                raw_predictions[k, :] += predict(X)

    def _staged_raw_predict(self, X):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        -------
        raw_predictions : generator of ndarray of shape \
            (n_trees_per_iteration, n_samples)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        X = self._validate_data(X, dtype=X_DTYPE, force_all_finite=False, reset=False)
        check_is_fitted(self)
        if X.shape[1] != self._n_features:
            raise ValueError(
                "X has {} features but this estimator was trained with "
                "{} features.".format(X.shape[1], self._n_features)
            )
        n_samples = X.shape[0]
        raw_predictions = np.zeros(
            shape=(self.n_trees_per_iteration_, n_samples),
            dtype=self._baseline_prediction.dtype,
        )
        raw_predictions += self._baseline_prediction

        # We intentionally decouple the number of threads used at prediction
        # time from the number of threads used at fit time because the model
        # can be deployed on a different machine for prediction purposes.
        n_threads = _openmp_effective_n_threads()
        for iteration in range(len(self._predictors)):
            self._predict_iterations(
                X,
                self._predictors[iteration : iteration + 1],
                raw_predictions,
                is_binned=False,
                n_threads=n_threads,
            )
            yield raw_predictions.copy()

    def _compute_partial_dependence_recursion(self, grid, target_features):
        """Fast partial dependence computation.

        Parameters
        ----------
        grid : ndarray, shape (n_samples, n_target_features)
            The grid points on which the partial dependence should be
            evaluated.
        target_features : ndarray, shape (n_target_features)
            The set of target features for which the partial dependence
            should be evaluated.

        Returns
        -------
        averaged_predictions : ndarray, shape \
                (n_trees_per_iteration, n_samples)
            The value of the partial dependence function on each grid point.
        """

        if getattr(self, "_fitted_with_sw", False):
            raise NotImplementedError(
                "{} does not support partial dependence "
                "plots with the 'recursion' method when "
                "sample weights were given during fit "
                "time.".format(self.__class__.__name__)
            )

        grid = np.asarray(grid, dtype=X_DTYPE, order="C")
        averaged_predictions = np.zeros(
            (self.n_trees_per_iteration_, grid.shape[0]), dtype=Y_DTYPE
        )

        for predictors_of_ith_iteration in self._predictors:
            for k, predictor in enumerate(predictors_of_ith_iteration):
                predictor.compute_partial_dependence(
                    grid, target_features, averaged_predictions[k]
                )
        # Note that the learning rate is already accounted for in the leaves
        # values.

        return averaged_predictions

    def _more_tags(self):
        return {"allow_nan": True}

    @abstractmethod
    def _get_loss(self, sample_weight, n_threads):
        pass

    @abstractmethod
    def _encode_y(self, y=None):
        pass

    @property
    def n_iter_(self):
        """Number of iterations of the boosting process."""
        check_is_fitted(self)
        return len(self._predictors)


class HistGradientBoostingRegressor(RegressorMixin, BaseHistGradientBoosting):
    """Histogram-based Gradient Boosting Regression Tree.

    This estimator is much faster than
    :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
    for big datasets (n_samples >= 10 000).

    This estimator has native support for missing values (NaNs). During
    training, the tree grower learns at each split point whether samples
    with missing values should go to the left or right child, based on the
    potential gain. When predicting, samples with missing values are
    assigned to the left or right child consequently. If no missing values
    were encountered for a given feature during training, then samples with
    missing values are mapped to whichever child has the most samples.

    This implementation is inspired by
    `LightGBM <https://github.com/Microsoft/LightGBM>`_.

    Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    loss : {'squared_error', 'absolute_error', 'poisson'}, \
            default='squared_error'
        The loss function to use in the boosting process. Note that the
        "squared error" and "poisson" losses actually implement
        "half least squares loss" and "half poisson deviance" to simplify the
        computation of the gradient. Furthermore, "poisson" loss internally
        uses a log-link and requires ``y >= 0``.

        .. versionchanged:: 0.23
           Added option 'poisson'.

        .. deprecated:: 1.0
            The loss 'least_squares' was deprecated in v1.0 and will be removed
            in version 1.2. Use `loss='squared_error'` which is equivalent.

        .. deprecated:: 1.0
            The loss 'least_absolute_deviation' was deprecated in v1.0 and will
            be removed in version 1.2. Use `loss='absolute_error'` which is
            equivalent.

    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees.
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    l2_regularization : float, default=0
        The L2 regularization parameter. Use ``0`` for no regularization
        (default).
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    categorical_features : array-like of {bool, int} of shape (n_features) \
            or shape (n_categorical_features,), default=None
        Indicates the categorical features.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

        Read more in the :ref:`User Guide <categorical_support_gbdt>`.

        .. versionadded:: 0.24

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonic constraint to enforce on each feature. -1, 1
        and 0 respectively correspond to a negative constraint, positive
        constraint and no constraint. Read more in the :ref:`User Guide
        <monotonic_cst_gbdt>`.

        .. versionadded:: 0.23

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.
    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.

        .. versionadded:: 0.23

    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used. If
        ``scoring='loss'``, early stopping is checked w.r.t the loss value.
        Only used if early stopping is performed.
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if early stopping is performed.
    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. Only used if early stopping is performed.
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores during early
        stopping. The higher the tolerance, the more likely we are to early
        stop: higher tolerance means that it will be harder for subsequent
        iterations to be considered an improvement upon the reference score.
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    do_early_stopping_ : bool
        Indicates whether early stopping is used during training.
    n_iter_ : int
        The number of iterations as selected by early stopping, depending on
        the `early_stopping` parameter. Otherwise it corresponds to max_iter.
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. For regressors,
        this is always 1.
    train_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    validation_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the held-out validation data. The
        first entry is the score of the ensemble before the first iteration.
        Scores are computed according to the ``scoring`` parameter. Empty if
        no early stopping or if ``validation_fraction`` is None.
    is_categorical_ : ndarray, shape (n_features, ) or None
        Boolean mask for the categorical features. ``None`` if there are no
        categorical features.
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GradientBoostingRegressor : Exact gradient boosting method that does not
        scale as good on datasets with a large number of samples.
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    RandomForestRegressor : A meta-estimator that fits a number of decision
        tree regressors on various sub-samples of the dataset and uses
        averaging to improve the statistical performance and control
        over-fitting.
    AdaBoostRegressor : A meta-estimator that begins by fitting a regressor
        on the original dataset and then fits additional copies of the
        regressor on the same dataset but where the weights of instances are
        adjusted according to the error of the current prediction. As such,
        subsequent regressors focus more on difficult cases.

    Examples
    --------
    >>> from sklearn.ensemble import HistGradientBoostingRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> est = HistGradientBoostingRegressor().fit(X, y)
    >>> est.score(X, y)
    0.92...
    """

    _VALID_LOSSES = (
        "squared_error",
        "least_squares",
        "absolute_error",
        "least_absolute_deviation",
        "poisson",
    )

    def __init__(
        self,
        loss="squared_error",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
    ):
        super(HistGradientBoostingRegressor, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            monotonic_cst=monotonic_cst,
            categorical_features=categorical_features,
            early_stopping=early_stopping,
            warm_start=warm_start,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

    def predict(self, X):
        """Predict values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        # Return inverse link of raw predictions after converting
        # shape (n_samples, 1) to (n_samples,)
        return self._loss.inverse_link_function(self._raw_predict(X).ravel())

    def staged_predict(self, X):
        """Predict regression target for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted values of the input samples, for each iteration.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield self._loss.inverse_link_function(raw_predictions.ravel())

    def _encode_y(self, y):
        # Just convert y to the expected dtype
        self.n_trees_per_iteration_ = 1
        y = y.astype(Y_DTYPE, copy=False)
        if self.loss == "poisson":
            # Ensure y >= 0 and sum(y) > 0
            if not (np.all(y >= 0) and np.sum(y) > 0):
                raise ValueError(
                    "loss='poisson' requires non-negative y and sum(y) > 0."
                )
        return y

    def _get_loss(self, sample_weight, n_threads):
        # TODO: Remove in v1.2
        if self.loss == "least_squares":
            warnings.warn(
                "The loss 'least_squares' was deprecated in v1.0 and will be "
                "removed in version 1.2. Use 'squared_error' which is "
                "equivalent.",
                FutureWarning,
            )
            return _LOSSES["squared_error"](
                sample_weight=sample_weight, n_threads=n_threads
            )
        elif self.loss == "least_absolute_deviation":
            warnings.warn(
                "The loss 'least_absolute_deviation' was deprecated in v1.0 "
                " and will be removed in version 1.2. Use 'absolute_error' "
                "which is equivalent.",
                FutureWarning,
            )
            return _LOSSES["absolute_error"](
                sample_weight=sample_weight, n_threads=n_threads
            )

        return _LOSSES[self.loss](sample_weight=sample_weight, n_threads=n_threads)


class HistGradientBoostingClassifier(ClassifierMixin, BaseHistGradientBoosting):
    """Histogram-based Gradient Boosting Classification Tree.

    This estimator is much faster than
    :class:`GradientBoostingClassifier<sklearn.ensemble.GradientBoostingClassifier>`
    for big datasets (n_samples >= 10 000).

    This estimator has native support for missing values (NaNs). During
    training, the tree grower learns at each split point whether samples
    with missing values should go to the left or right child, based on the
    potential gain. When predicting, samples with missing values are
    assigned to the left or right child consequently. If no missing values
    were encountered for a given feature during training, then samples with
    missing values are mapped to whichever child has the most samples.

    This implementation is inspired by
    `LightGBM <https://github.com/Microsoft/LightGBM>`_.

    Read more in the :ref:`User Guide <histogram_based_gradient_boosting>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    loss : {'auto', 'binary_crossentropy', 'categorical_crossentropy'}, \
            default='auto'
        The loss function to use in the boosting process. 'binary_crossentropy'
        (also known as logistic loss) is used for binary classification and
        generalizes to 'categorical_crossentropy' for multiclass
        classification. 'auto' will automatically choose either loss depending
        on the nature of the problem.
    learning_rate : float, default=0.1
        The learning rate, also known as *shrinkage*. This is used as a
        multiplicative factor for the leaves values. Use ``1`` for no
        shrinkage.
    max_iter : int, default=100
        The maximum number of iterations of the boosting process, i.e. the
        maximum number of trees for binary classification. For multiclass
        classification, `n_classes` trees per iteration are built.
    max_leaf_nodes : int or None, default=31
        The maximum number of leaves for each tree. Must be strictly greater
        than 1. If None, there is no maximum limit.
    max_depth : int or None, default=None
        The maximum depth of each tree. The depth of a tree is the number of
        edges to go from the root to the deepest leaf.
        Depth isn't constrained by default.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf. For small datasets with less
        than a few hundred samples, it is recommended to lower this value
        since only very shallow trees would be built.
    l2_regularization : float, default=0
        The L2 regularization parameter. Use 0 for no regularization.
    max_bins : int, default=255
        The maximum number of bins to use for non-missing values. Before
        training, each feature of the input array `X` is binned into
        integer-valued bins, which allows for a much faster training stage.
        Features with a small number of unique values may use less than
        ``max_bins`` bins. In addition to the ``max_bins`` bins, one more bin
        is always reserved for missing values. Must be no larger than 255.
    categorical_features : array-like of {bool, int} of shape (n_features) \
            or shape (n_categorical_features,), default=None
        Indicates the categorical features.

        - None : no feature will be considered categorical.
        - boolean array-like : boolean mask indicating categorical features.
        - integer array-like : integer indices indicating categorical
          features.

        For each categorical feature, there must be at most `max_bins` unique
        categories, and each categorical value must be in [0, max_bins -1].

        Read more in the :ref:`User Guide <categorical_support_gbdt>`.

        .. versionadded:: 0.24

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonic constraint to enforce on each feature. -1, 1
        and 0 respectively correspond to a negative constraint, positive
        constraint and no constraint. Read more in the :ref:`User Guide
        <monotonic_cst_gbdt>`.

        .. versionadded:: 0.23

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble. For results to be valid, the
        estimator should be re-trained on the same data only.
        See :term:`the Glossary <warm_start>`.
    early_stopping : 'auto' or bool, default='auto'
        If 'auto', early stopping is enabled if the sample size is larger than
        10000. If True, early stopping is enabled, otherwise early stopping is
        disabled.

        .. versionadded:: 0.23

    scoring : str or callable or None, default='loss'
        Scoring parameter to use for early stopping. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer
        is used. If ``scoring='loss'``, early stopping is checked
        w.r.t the loss value. Only used if early stopping is performed.
    validation_fraction : int or float or None, default=0.1
        Proportion (or absolute size) of training data to set aside as
        validation data for early stopping. If None, early stopping is done on
        the training data. Only used if early stopping is performed.
    n_iter_no_change : int, default=10
        Used to determine when to "early stop". The fitting process is
        stopped when none of the last ``n_iter_no_change`` scores are better
        than the ``n_iter_no_change - 1`` -th-to-last one, up to some
        tolerance. Only used if early stopping is performed.
    tol : float, default=1e-7
        The absolute tolerance to use when comparing scores. The higher the
        tolerance, the more likely we are to early stop: higher tolerance
        means that it will be harder for subsequent iterations to be
        considered an improvement upon the reference score.
    verbose : int, default=0
        The verbosity level. If not zero, print some information about the
        fitting process.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the subsampling in the
        binning process, and the train/validation data split if early stopping
        is enabled.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        Class labels.
    do_early_stopping_ : bool
        Indicates whether early stopping is used during training.
    n_iter_ : int
        The number of iterations as selected by early stopping, depending on
        the `early_stopping` parameter. Otherwise it corresponds to max_iter.
    n_trees_per_iteration_ : int
        The number of tree that are built at each iteration. This is equal to 1
        for binary classification, and to ``n_classes`` for multiclass
        classification.
    train_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the training data. The first entry
        is the score of the ensemble before the first iteration. Scores are
        computed according to the ``scoring`` parameter. If ``scoring`` is
        not 'loss', scores are computed on a subset of at most 10 000
        samples. Empty if no early stopping.
    validation_score_ : ndarray, shape (n_iter_+1,)
        The scores at each iteration on the held-out validation data. The
        first entry is the score of the ensemble before the first iteration.
        Scores are computed according to the ``scoring`` parameter. Empty if
        no early stopping or if ``validation_fraction`` is None.
    is_categorical_ : ndarray, shape (n_features, ) or None
        Boolean mask for the categorical features. ``None`` if there are no
        categorical features.
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GradientBoostingClassifier : Exact gradient boosting method that does not
        scale as good on datasets with a large number of samples.
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    RandomForestClassifier : A meta-estimator that fits a number of decision
        tree classifiers on various sub-samples of the dataset and uses
        averaging to improve the predictive accuracy and control over-fitting.
    AdaBoostClassifier : A meta-estimator that begins by fitting a classifier
        on the original dataset and then fits additional copies of the
        classifier on the same dataset where the weights of incorrectly
        classified instances are adjusted such that subsequent classifiers
        focus more on difficult cases.

    Examples
    --------
    >>> from sklearn.ensemble import HistGradientBoostingClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = HistGradientBoostingClassifier().fit(X, y)
    >>> clf.score(X, y)
    1.0
    """

    _VALID_LOSSES = ("binary_crossentropy", "categorical_crossentropy", "auto")

    def __init__(
        self,
        loss="auto",
        *,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_bins=255,
        categorical_features=None,
        monotonic_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        random_state=None,
    ):
        super(HistGradientBoostingClassifier, self).__init__(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

    def predict(self, X):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        # TODO: This could be done in parallel
        encoded_classes = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[encoded_classes]

    def staged_predict(self, X):
        """Predict classes at each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted classes of the input samples, for each iteration.
        """
        for proba in self.staged_predict_proba(X):
            encoded_classes = np.argmax(proba, axis=1)
            yield self.classes_.take(encoded_classes, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray, shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        raw_predictions = self._raw_predict(X)
        return self._loss.predict_proba(raw_predictions)

    def staged_predict_proba(self, X):
        """Predict class probabilities at each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted class probabilities of the input samples,
            for each iteration.
        """
        for raw_predictions in self._staged_raw_predict(X):
            yield self._loss.predict_proba(raw_predictions)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        decision : ndarray, shape (n_samples,) or \
                (n_samples, n_trees_per_iteration)
            The raw predicted values (i.e. the sum of the trees leaves) for
            each sample. n_trees_per_iteration is equal to the number of
            classes in multiclass classification.
        """
        decision = self._raw_predict(X)
        if decision.shape[0] == 1:
            decision = decision.ravel()
        return decision.T

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        -------
        decision : generator of ndarray of shape (n_samples,) or \
                (n_samples, n_trees_per_iteration)
            The decision function of the input samples, which corresponds to
            the raw values predicted from the trees of the ensemble . The
            classes corresponds to that in the attribute :term:`classes_`.
        """
        for staged_decision in self._staged_raw_predict(X):
            if staged_decision.shape[0] == 1:
                staged_decision = staged_decision.ravel()
            yield staged_decision.T

    def _encode_y(self, y):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y.astype(Y_DTYPE, copy=False)
        return encoded_y

    def _get_loss(self, sample_weight, n_threads):
        if self.loss == "categorical_crossentropy" and self.n_trees_per_iteration_ == 1:
            raise ValueError(
                "'categorical_crossentropy' is not suitable for "
                "a binary classification problem. Please use "
                "'auto' or 'binary_crossentropy' instead."
            )

        if self.loss == "auto":
            if self.n_trees_per_iteration_ == 1:
                return _LOSSES["binary_crossentropy"](
                    sample_weight=sample_weight, n_threads=n_threads
                )
            else:
                return _LOSSES["categorical_crossentropy"](
                    sample_weight=sample_weight, n_threads=n_threads
                )

        return _LOSSES[self.loss](sample_weight=sample_weight, n_threads=n_threads)
