# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com> (main author)
#          Mathieu Blondel (partial_fit support)
#
# License: BSD 3 clause
"""Classification, regression and One-Class SVM using Stochastic Gradient
Descent (SGD).
"""

import numpy as np
import warnings

from abc import ABCMeta, abstractmethod

from joblib import Parallel

from ..base import clone, is_classifier
from ._base import LinearClassifierMixin, SparseCoefMixin
from ._base import make_dataset
from ..base import BaseEstimator, RegressorMixin, OutlierMixin
from ..utils import check_random_state
from ..utils.metaestimators import available_if
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils.fixes import delayed
from ..exceptions import ConvergenceWarning
from ..model_selection import StratifiedShuffleSplit, ShuffleSplit

from ._sgd_fast import _plain_sgd
from ..utils import compute_class_weight
from ._sgd_fast import Hinge
from ._sgd_fast import SquaredHinge
from ._sgd_fast import Log
from ._sgd_fast import ModifiedHuber
from ._sgd_fast import SquaredLoss
from ._sgd_fast import Huber
from ._sgd_fast import EpsilonInsensitive
from ._sgd_fast import SquaredEpsilonInsensitive
from ..utils.fixes import _joblib_parallel_args

LEARNING_RATE_TYPES = {
    "constant": 1,
    "optimal": 2,
    "invscaling": 3,
    "adaptive": 4,
    "pa1": 5,
    "pa2": 6,
}

PENALTY_TYPES = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}

DEFAULT_EPSILON = 0.1
# Default value of ``epsilon`` parameter.

MAX_INT = np.iinfo(np.int32).max


class _ValidationScoreCallback:
    """Callback for early stopping based on validation score"""

    def __init__(self, estimator, X_val, y_val, sample_weight_val, classes=None):
        self.estimator = clone(estimator)
        self.estimator.t_ = 1  # to pass check_is_fitted
        if classes is not None:
            self.estimator.classes_ = classes
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weight_val = sample_weight_val

    def __call__(self, coef, intercept):
        est = self.estimator
        est.coef_ = coef.reshape(1, -1)
        est.intercept_ = np.atleast_1d(intercept)
        return est.score(self.X_val, self.y_val, self.sample_weight_val)


class BaseSGD(SparseCoefMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for SGD classification and regression."""

    def __init__(
        self,
        loss,
        *,
        penalty="l2",
        alpha=0.0001,
        C=1.0,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=0.1,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        self.loss = loss
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.alpha = alpha
        self.C = C
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    def _validate_params(self, for_partial_fit=False):
        """Validate input params."""
        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False")
        if self.early_stopping and for_partial_fit:
            raise ValueError("early_stopping should be False with partial_fit")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be > zero. Got %f" % self.max_iter)
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if not isinstance(self, SGDOneClassSVM) and self.alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if self.n_iter_no_change < 1:
            raise ValueError("n_iter_no_change must be >= 1")
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in range (0, 1)")
        if self.learning_rate in ("constant", "invscaling", "adaptive"):
            if self.eta0 <= 0.0:
                raise ValueError("eta0 must be > 0")
        if self.learning_rate == "optimal" and self.alpha == 0:
            raise ValueError(
                "alpha must be > 0 since "
                "learning_rate is 'optimal'. alpha is used "
                "to compute the optimal learning rate."
            )

        # raises ValueError if not registered
        self._get_penalty_type(self.penalty)
        self._get_learning_rate_type(self.learning_rate)

        if self.loss not in self.loss_functions:
            raise ValueError("The loss %s is not supported. " % self.loss)

        if self.loss == "squared_loss":
            warnings.warn(
                "The loss 'squared_loss' was deprecated in v1.0 and will be "
                "removed in version 1.2. Use `loss='squared_error'` which is "
                "equivalent.",
                FutureWarning,
            )

    def _get_loss_function(self, loss):
        """Get concrete ``LossFunction`` object for str ``loss``."""
        try:
            loss_ = self.loss_functions[loss]
            loss_class, args = loss_[0], loss_[1:]
            if loss in ("huber", "epsilon_insensitive", "squared_epsilon_insensitive"):
                args = (self.epsilon,)
            return loss_class(*args)
        except KeyError as e:
            raise ValueError("The loss %s is not supported. " % loss) from e

    def _get_learning_rate_type(self, learning_rate):
        try:
            return LEARNING_RATE_TYPES[learning_rate]
        except KeyError as e:
            raise ValueError(
                "learning rate %s is not supported. " % learning_rate
            ) from e

    def _get_penalty_type(self, penalty):
        penalty = str(penalty).lower()
        try:
            return PENALTY_TYPES[penalty]
        except KeyError as e:
            raise ValueError("Penalty %s is not supported. " % penalty) from e

    def _allocate_parameter_mem(
        self, n_classes, n_features, coef_init=None, intercept_init=None, one_class=0
    ):
        """Allocate mem for parameters; initialize if provided."""
        if n_classes > 2:
            # allocate coef_ for multi-class
            if coef_init is not None:
                coef_init = np.asarray(coef_init, order="C")
                if coef_init.shape != (n_classes, n_features):
                    raise ValueError("Provided ``coef_`` does not match dataset. ")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros(
                    (n_classes, n_features), dtype=np.float64, order="C"
                )

            # allocate intercept_ for multi-class
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, order="C")
                if intercept_init.shape != (n_classes,):
                    raise ValueError("Provided intercept_init does not match dataset.")
                self.intercept_ = intercept_init
            else:
                self.intercept_ = np.zeros(n_classes, dtype=np.float64, order="C")
        else:
            # allocate coef_
            if coef_init is not None:
                coef_init = np.asarray(coef_init, dtype=np.float64, order="C")
                coef_init = coef_init.ravel()
                if coef_init.shape != (n_features,):
                    raise ValueError("Provided coef_init does not match dataset.")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros(n_features, dtype=np.float64, order="C")

            # allocate intercept_
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, dtype=np.float64)
                if intercept_init.shape != (1,) and intercept_init.shape != ():
                    raise ValueError("Provided intercept_init does not match dataset.")
                if one_class:
                    self.offset_ = intercept_init.reshape(
                        1,
                    )
                else:
                    self.intercept_ = intercept_init.reshape(
                        1,
                    )
            else:
                if one_class:
                    self.offset_ = np.zeros(1, dtype=np.float64, order="C")
                else:
                    self.intercept_ = np.zeros(1, dtype=np.float64, order="C")

        # initialize average parameters
        if self.average > 0:
            self._standard_coef = self.coef_
            self._average_coef = np.zeros(self.coef_.shape, dtype=np.float64, order="C")
            if one_class:
                self._standard_intercept = 1 - self.offset_
            else:
                self._standard_intercept = self.intercept_

            self._average_intercept = np.zeros(
                self._standard_intercept.shape, dtype=np.float64, order="C"
            )

    def _make_validation_split(self, y):
        """Split the dataset between training set and validation set.

        Parameters
        ----------
        y : ndarray of shape (n_samples, )
            Target values.

        Returns
        -------
        validation_mask : ndarray of shape (n_samples, )
            Equal to 1 on the validation set, 0 on the training set.
        """
        n_samples = y.shape[0]
        validation_mask = np.zeros(n_samples, dtype=np.uint8)
        if not self.early_stopping:
            # use the full set for training, with an empty validation set
            return validation_mask

        if is_classifier(self):
            splitter_type = StratifiedShuffleSplit
        else:
            splitter_type = ShuffleSplit
        cv = splitter_type(
            test_size=self.validation_fraction, random_state=self.random_state
        )
        idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))
        if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
            raise ValueError(
                "Splitting %d samples into a train set and a validation set "
                "with validation_fraction=%r led to an empty set (%d and %d "
                "samples). Please either change validation_fraction, increase "
                "number of samples, or disable early_stopping."
                % (
                    n_samples,
                    self.validation_fraction,
                    idx_train.shape[0],
                    idx_val.shape[0],
                )
            )

        validation_mask[idx_val] = 1
        return validation_mask

    def _make_validation_score_cb(
        self, validation_mask, X, y, sample_weight, classes=None
    ):
        if not self.early_stopping:
            return None

        return _ValidationScoreCallback(
            self,
            X[validation_mask],
            y[validation_mask],
            sample_weight[validation_mask],
            classes=classes,
        )


def _prepare_fit_binary(est, y, i):
    """Initialization for fit_binary.

    Returns y, coef, intercept, average_coef, average_intercept.
    """
    y_i = np.ones(y.shape, dtype=np.float64, order="C")
    y_i[y != est.classes_[i]] = -1.0
    average_intercept = 0
    average_coef = None

    if len(est.classes_) == 2:
        if not est.average:
            coef = est.coef_.ravel()
            intercept = est.intercept_[0]
        else:
            coef = est._standard_coef.ravel()
            intercept = est._standard_intercept[0]
            average_coef = est._average_coef.ravel()
            average_intercept = est._average_intercept[0]
    else:
        if not est.average:
            coef = est.coef_[i]
            intercept = est.intercept_[i]
        else:
            coef = est._standard_coef[i]
            intercept = est._standard_intercept[i]
            average_coef = est._average_coef[i]
            average_intercept = est._average_intercept[i]

    return y_i, coef, intercept, average_coef, average_intercept


def fit_binary(
    est,
    i,
    X,
    y,
    alpha,
    C,
    learning_rate,
    max_iter,
    pos_weight,
    neg_weight,
    sample_weight,
    validation_mask=None,
    random_state=None,
):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.

    Parameters
    ----------
    est : Estimator object
        The estimator to fit

    i : int
        Index of the positive class

    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, ]
        Target values

    alpha : float
        The regularization parameter

    C : float
        Maximum step size for passive aggressive

    learning_rate : str
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.

    max_iter : int
        The maximum number of iterations (epochs)

    pos_weight : float
        The weight of the positive class

    neg_weight : float
        The weight of the negative class

    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample

    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.

    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    # if average is not true, average_coef, and average_intercept will be
    # unused
    y_i, coef, intercept, average_coef, average_intercept = _prepare_fit_binary(
        est, y, i
    )
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]

    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset(
        X, y_i, sample_weight, random_state=random_state
    )

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    validation_score_cb = est._make_validation_score_cb(
        validation_mask, X, y_i, sample_weight, classes=classes
    )

    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(MAX_INT)

    tol = est.tol if est.tol is not None else -np.inf

    coef, intercept, average_coef, average_intercept, n_iter_ = _plain_sgd(
        coef,
        intercept,
        average_coef,
        average_intercept,
        est.loss_function_,
        penalty_type,
        alpha,
        C,
        est.l1_ratio,
        dataset,
        validation_mask,
        est.early_stopping,
        validation_score_cb,
        int(est.n_iter_no_change),
        max_iter,
        tol,
        int(est.fit_intercept),
        int(est.verbose),
        int(est.shuffle),
        seed,
        pos_weight,
        neg_weight,
        learning_rate_type,
        est.eta0,
        est.power_t,
        0,
        est.t_,
        intercept_decay,
        est.average,
    )

    if est.average:
        if len(est.classes_) == 2:
            est._average_intercept[0] = average_intercept
        else:
            est._average_intercept[i] = average_intercept

    return coef, intercept, n_iter_


class BaseSGDClassifier(LinearClassifierMixin, BaseSGD, metaclass=ABCMeta):

    # TODO: Remove squared_loss in v1.2
    loss_functions = {
        "hinge": (Hinge, 1.0),
        "squared_hinge": (SquaredHinge, 1.0),
        "perceptron": (Hinge, 0.0),
        "log": (Log,),
        "modified_huber": (ModifiedHuber,),
        "squared_error": (SquaredLoss,),
        "squared_loss": (SquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive, DEFAULT_EPSILON),
    }

    @abstractmethod
    def __init__(
        self,
        loss="hinge",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):

        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _partial_fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        max_iter,
        classes,
        sample_weight,
        coef_init,
        intercept_init,
    ):
        first_call = not hasattr(self, "classes_")
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            reset=first_call,
        )

        n_samples, n_features = X.shape

        _check_partial_fit_first_call(self, classes)

        n_classes = self.classes_.shape[0]

        # Allocate datastructures from input arguments
        self._expanded_class_weight = compute_class_weight(
            self.class_weight, classes=self.classes_, y=y
        )
        sample_weight = _check_sample_weight(sample_weight, X)

        if getattr(self, "coef_", None) is None or coef_init is not None:
            self._allocate_parameter_mem(
                n_classes, n_features, coef_init, intercept_init
            )
        elif n_features != self.coef_.shape[-1]:
            raise ValueError(
                "Number of features %d does not match previous data %d."
                % (n_features, self.coef_.shape[-1])
            )

        self.loss_function_ = self._get_loss_function(loss)
        if not hasattr(self, "t_"):
            self.t_ = 1.0

        # delegate to concrete training procedure
        if n_classes > 2:
            self._fit_multiclass(
                X,
                y,
                alpha=alpha,
                C=C,
                learning_rate=learning_rate,
                sample_weight=sample_weight,
                max_iter=max_iter,
            )
        elif n_classes == 2:
            self._fit_binary(
                X,
                y,
                alpha=alpha,
                C=C,
                learning_rate=learning_rate,
                sample_weight=sample_weight,
                max_iter=max_iter,
            )
        else:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % n_classes
            )

        return self

    def _fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        coef_init=None,
        intercept_init=None,
        sample_weight=None,
    ):
        self._validate_params()
        if hasattr(self, "classes_"):
            # delete the attribute otherwise _partial_fit thinks it's not the first call
            delattr(self, "classes_")

        # labels can be encoded as float, int, or string literals
        # np.unique sorts in asc order; largest class id is positive class
        y = self._validate_data(y=y)
        classes = np.unique(y)

        if self.warm_start and hasattr(self, "coef_"):
            if coef_init is None:
                coef_init = self.coef_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None

        if self.average > 0:
            self._standard_coef = self.coef_
            self._standard_intercept = self.intercept_
            self._average_coef = None
            self._average_intercept = None

        # Clear iteration count for multiple call to fit.
        self.t_ = 1.0

        self._partial_fit(
            X,
            y,
            alpha,
            C,
            loss,
            learning_rate,
            self.max_iter,
            classes,
            sample_weight,
            coef_init,
            intercept_init,
        )

        if (
            self.tol is not None
            and self.tol > -np.inf
            and self.n_iter_ == self.max_iter
        ):
            warnings.warn(
                "Maximum number of iteration reached before "
                "convergence. Consider increasing max_iter to "
                "improve the fit.",
                ConvergenceWarning,
            )
        return self

    def _fit_binary(self, X, y, alpha, C, sample_weight, learning_rate, max_iter):
        """Fit a binary classifier on X and y."""
        coef, intercept, n_iter_ = fit_binary(
            self,
            1,
            X,
            y,
            alpha,
            C,
            learning_rate,
            max_iter,
            self._expanded_class_weight[1],
            self._expanded_class_weight[0],
            sample_weight,
            random_state=self.random_state,
        )

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        # need to be 2d
        if self.average > 0:
            if self.average <= self.t_ - 1:
                self.coef_ = self._average_coef.reshape(1, -1)
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef.reshape(1, -1)
                self._standard_intercept = np.atleast_1d(intercept)
                self.intercept_ = self._standard_intercept
        else:
            self.coef_ = coef.reshape(1, -1)
            # intercept is a float, need to convert it to an array of length 1
            self.intercept_ = np.atleast_1d(intercept)

    def _fit_multiclass(self, X, y, alpha, C, learning_rate, sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
        # Precompute the validation split using the multiclass labels
        # to ensure proper balancing of the classes.
        validation_mask = self._make_validation_split(y)

        # Use joblib to fit OvA in parallel.
        # Pick the random seed for each job outside of fit_binary to avoid
        # sharing the estimator random state between threads which could lead
        # to non-deterministic behavior
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        result = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(require="sharedmem"),
        )(
            delayed(fit_binary)(
                self,
                i,
                X,
                y,
                alpha,
                C,
                learning_rate,
                max_iter,
                self._expanded_class_weight[i],
                1.0,
                sample_weight,
                validation_mask=validation_mask,
                random_state=seed,
            )
            for i, seed in enumerate(seeds)
        )

        # take the maximum of n_iter_ over every binary fit
        n_iter_ = 0.0
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self._average_coef
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef
                self._standard_intercept = np.atleast_1d(self.intercept_)
                self.intercept_ = self._standard_intercept

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Perform one epoch of stochastic gradient descent on given samples.

        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence, early stopping, and
        learning rate adjustments should be handled by the user.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.

        y : ndarray of shape (n_samples,)
            Subset of the target values.

        classes : ndarray of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self._validate_params(for_partial_fit=True)
        if self.class_weight in ["balanced"]:
            raise ValueError(
                "class_weight '{0}' is not supported for "
                "partial_fit. In order to use 'balanced' weights,"
                " use compute_class_weight('{0}', "
                "classes=classes, y=y). "
                "In place of y you can us a large enough sample "
                "of the full training set target to properly "
                "estimate the class frequency distributions. "
                "Pass the resulting weights as the class_weight "
                "parameter.".format(self.class_weight)
            )
        return self._partial_fit(
            X,
            y,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            classes=classes,
            sample_weight=sample_weight,
            coef_init=None,
            intercept_init=None,
        )

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_classes, n_features), default=None
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (n_classes,), default=None
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        return self._fit(
            X,
            y,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            coef_init=coef_init,
            intercept_init=intercept_init,
            sample_weight=sample_weight,
        )


class SGDClassifier(BaseSGDClassifier):
    """Linear classifiers (SVM, logistic regression, etc.) with SGD training.

    This estimator implements regularized linear models with stochastic
    gradient descent (SGD) learning: the gradient of the loss is estimated
    each sample at a time and the model is updated along the way with a
    decreasing strength schedule (aka learning rate). SGD allows minibatch
    (online/out-of-core) learning via the `partial_fit` method.
    For best results using the default learning rate schedule, the data should
    have zero mean and unit variance.

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. The model it fits can be
    controlled with the loss parameter; by default, it fits a linear support
    vector machine (SVM).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : str, default='hinge'
        The loss function to be used. Defaults to 'hinge', which gives a
        linear SVM.

        The possible options are 'hinge', 'log', 'modified_huber',
        'squared_hinge', 'perceptron', or a regression loss: 'squared_error',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.

        The 'log' loss gives logistic regression, a probabilistic classifier.
        'modified_huber' is another smooth loss that brings tolerance to
        outliers as well as probability estimates.
        'squared_hinge' is like hinge but is quadratically penalized.
        'perceptron' is the linear loss used by the perceptron algorithm.
        The other losses are designed for regression but can be useful in
        classification as well; see
        :class:`~sklearn.linear_model.SGDRegressor` for a description.

        More details about the losses formulas can be found in the
        :ref:`User Guide <sgd_mathematical_formulation>`.

        .. deprecated:: 1.0
            The loss 'squared_loss' was deprecated in v1.0 and will be removed
            in version 1.2. Use `loss='squared_error'` which is equivalent.

    penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization.
        Also used to compute the learning rate when set to `learning_rate` is
        set to 'optimal'.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Only used if `penalty` is 'elasticnet'.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float, default=1e-3
        The stopping criterion. If it is not None, training will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.

    n_jobs : int, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    learning_rate : str, default='optimal'
        The learning rate schedule:

        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where t0 is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': eta = eta0, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          early_stopping is True, the current learning rate is divided by 5.

            .. versionadded:: 0.20
                Added 'adaptive' option

    eta0 : float, default=0.0
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.0 as eta0 is not used by
        the default schedule 'optimal'.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate [default 0.5].

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
        training when validation score returned by the `score` method is not
        improving by at least tol for n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20
            Added 'early_stopping' option

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `early_stopping` is True.

        .. versionadded:: 0.20
            Added 'validation_fraction' option

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        fitting.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.

        .. versionadded:: 0.20
            Added 'n_iter_no_change' option

    class_weight : dict, {class_label: weight} or "balanced", default=None
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit`` will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights across all
        updates and stores the result in the ``coef_`` attribute. If set to
        an int greater than 1, averaging will begin once the total number of
        samples seen reaches `average`. So ``average=10`` will begin
        averaging after seeing 10 samples.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else \
            (n_classes, n_features)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    loss_function_ : concrete ``LossFunction``

    classes_ : array of shape (n_classes,)

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.svm.LinearSVC : Linear support vector classification.
    LogisticRegression : Logistic regression.
    Perceptron : Inherits from SGDClassifier. ``Perceptron()`` is equivalent to
        ``SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant",
        penalty=None)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> # Always scale the input. The most convenient way is to use a pipeline.
    >>> clf = make_pipeline(StandardScaler(),
    ...                     SGDClassifier(max_iter=1000, tol=1e-3))
    >>> clf.fit(X, Y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('sgdclassifier', SGDClassifier())])
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(
        self,
        loss="hinge",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

    def _check_proba(self):
        if self.loss not in ("log", "modified_huber"):
            raise AttributeError(
                "probability estimates are not available for loss=%r" % self.loss
            )
        return True

    @available_if(_check_proba)
    def predict_proba(self, X):
        """Probability estimates.

        This method is only available for log loss and modified Huber loss.

        Multiclass probability estimates are derived from binary (one-vs.-rest)
        estimates by simple normalization, as recommended by Zadrozny and
        Elkan.

        Binary probability estimates for loss="modified_huber" are given by
        (clip(decision_function(X), -1, 1) + 1) / 2. For other loss functions
        it is necessary to perform proper probability calibration by wrapping
        the classifier with
        :class:`~sklearn.calibration.CalibratedClassifierCV` instead.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.

        References
        ----------
        Zadrozny and Elkan, "Transforming classifier scores into multiclass
        probability estimates", SIGKDD'02,
        https://dl.acm.org/doi/pdf/10.1145/775047.775151

        The justification for the formula in the loss="modified_huber"
        case is in the appendix B in:
        http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
        """
        check_is_fitted(self)

        if self.loss == "log":
            return self._predict_proba_lr(X)

        elif self.loss == "modified_huber":
            binary = len(self.classes_) == 2
            scores = self.decision_function(X)

            if binary:
                prob2 = np.ones((scores.shape[0], 2))
                prob = prob2[:, 1]
            else:
                prob = scores

            np.clip(scores, -1, 1, prob)
            prob += 1.0
            prob /= 2.0

            if binary:
                prob2[:, 0] -= prob
                prob = prob2
            else:
                # the above might assign zero to all classes, which doesn't
                # normalize neatly; work around this to produce uniform
                # probabilities
                prob_sum = prob.sum(axis=1)
                all_zero = prob_sum == 0
                if np.any(all_zero):
                    prob[all_zero, :] = 1
                    prob_sum[all_zero] = len(self.classes_)

                # normalize
                prob /= prob_sum.reshape((prob.shape[0], -1))

            return prob

        else:
            raise NotImplementedError(
                "predict_(log_)proba only supported when"
                " loss='log' or loss='modified_huber' "
                "(%r given)"
                % self.loss
            )

    @available_if(_check_proba)
    def predict_log_proba(self, X):
        """Log of probability estimates.

        This method is only available for log loss and modified Huber loss.

        When loss="modified_huber", probability estimates may be hard zeros
        and ones, so taking the logarithm is not possible.

        See ``predict_proba`` for details.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        T : array-like, shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in
            `self.classes_`.
        """
        return np.log(self.predict_proba(X))

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }


class BaseSGDRegressor(RegressorMixin, BaseSGD):

    # TODO: Remove squared_loss in v1.2
    loss_functions = {
        "squared_error": (SquaredLoss,),
        "squared_loss": (SquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive, DEFAULT_EPSILON),
    }

    @abstractmethod
    def __init__(
        self,
        loss="squared_error",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )

    def _partial_fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        max_iter,
        sample_weight,
        coef_init,
        intercept_init,
    ):
        first_call = getattr(self, "coef_", None) is None
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            copy=False,
            order="C",
            dtype=np.float64,
            accept_large_sparse=False,
            reset=first_call,
        )
        y = y.astype(np.float64, copy=False)

        n_samples, n_features = X.shape

        sample_weight = _check_sample_weight(sample_weight, X)

        # Allocate datastructures from input arguments
        if first_call:
            self._allocate_parameter_mem(1, n_features, coef_init, intercept_init)
        if self.average > 0 and getattr(self, "_average_coef", None) is None:
            self._average_coef = np.zeros(n_features, dtype=np.float64, order="C")
            self._average_intercept = np.zeros(1, dtype=np.float64, order="C")

        self._fit_regressor(
            X, y, alpha, C, loss, learning_rate, sample_weight, max_iter
        )

        return self

    def partial_fit(self, X, y, sample_weight=None):
        """Perform one epoch of stochastic gradient descent on given samples.

        Internally, this method uses ``max_iter = 1``. Therefore, it is not
        guaranteed that a minimum of the cost function is reached after calling
        it once. Matters such as objective convergence and early stopping
        should be handled by the user.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data.

        y : numpy array of shape (n_samples,)
            Subset of target values.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self._validate_params(for_partial_fit=True)
        return self._partial_fit(
            X,
            y,
            self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            sample_weight=sample_weight,
            coef_init=None,
            intercept_init=None,
        )

    def _fit(
        self,
        X,
        y,
        alpha,
        C,
        loss,
        learning_rate,
        coef_init=None,
        intercept_init=None,
        sample_weight=None,
    ):
        self._validate_params()
        if self.warm_start and getattr(self, "coef_", None) is not None:
            if coef_init is None:
                coef_init = self.coef_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None

        # Clear iteration count for multiple call to fit.
        self.t_ = 1.0

        self._partial_fit(
            X,
            y,
            alpha,
            C,
            loss,
            learning_rate,
            self.max_iter,
            sample_weight,
            coef_init,
            intercept_init,
        )

        if (
            self.tol is not None
            and self.tol > -np.inf
            and self.n_iter_ == self.max_iter
        ):
            warnings.warn(
                "Maximum number of iteration reached before "
                "convergence. Consider increasing max_iter to "
                "improve the fit.",
                ConvergenceWarning,
            )

        return self

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_features,), default=None
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (1,), default=None
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Fitted `SGDRegressor` estimator.
        """
        return self._fit(
            X,
            y,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            coef_init=coef_init,
            intercept_init=intercept_init,
            sample_weight=sample_weight,
        )

    def _decision_function(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)

        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return scores.ravel()

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        return self._decision_function(X)

    def _fit_regressor(
        self, X, y, alpha, C, loss, learning_rate, sample_weight, max_iter
    ):
        dataset, intercept_decay = make_dataset(X, y, sample_weight)

        loss_function = self._get_loss_function(loss)
        penalty_type = self._get_penalty_type(self.penalty)
        learning_rate_type = self._get_learning_rate_type(learning_rate)

        if not hasattr(self, "t_"):
            self.t_ = 1.0

        validation_mask = self._make_validation_split(y)
        validation_score_cb = self._make_validation_score_cb(
            validation_mask, X, y, sample_weight
        )

        random_state = check_random_state(self.random_state)
        # numpy mtrand expects a C long which is a signed 32 bit integer under
        # Windows
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        tol = self.tol if self.tol is not None else -np.inf

        if self.average:
            coef = self._standard_coef
            intercept = self._standard_intercept
            average_coef = self._average_coef
            average_intercept = self._average_intercept
        else:
            coef = self.coef_
            intercept = self.intercept_
            average_coef = None  # Not used
            average_intercept = [0]  # Not used

        coef, intercept, average_coef, average_intercept, self.n_iter_ = _plain_sgd(
            coef,
            intercept[0],
            average_coef,
            average_intercept[0],
            loss_function,
            penalty_type,
            alpha,
            C,
            self.l1_ratio,
            dataset,
            validation_mask,
            self.early_stopping,
            validation_score_cb,
            int(self.n_iter_no_change),
            max_iter,
            tol,
            int(self.fit_intercept),
            int(self.verbose),
            int(self.shuffle),
            seed,
            1.0,
            1.0,
            learning_rate_type,
            self.eta0,
            self.power_t,
            0,
            self.t_,
            intercept_decay,
            self.average,
        )

        self.t_ += self.n_iter_ * X.shape[0]

        if self.average > 0:
            self._average_intercept = np.atleast_1d(average_intercept)
            self._standard_intercept = np.atleast_1d(intercept)

            if self.average <= self.t_ - 1.0:
                # made enough updates for averaging to be taken into account
                self.coef_ = average_coef
                self.intercept_ = np.atleast_1d(average_intercept)
            else:
                self.coef_ = coef
                self.intercept_ = np.atleast_1d(intercept)

        else:
            self.intercept_ = np.atleast_1d(intercept)


class SGDRegressor(BaseSGDRegressor):
    """Linear model fitted by minimizing a regularized empirical loss with SGD.

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : str, default='squared_error'
        The loss function to be used. The possible values are 'squared_error',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'

        The 'squared_error' refers to the ordinary least squares fit.
        'huber' modifies 'squared_error' to focus less on getting outliers
        correct by switching from squared to linear loss past a distance of
        epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
        linear past that; this is the loss function used in SVR.
        'squared_epsilon_insensitive' is the same but becomes squared loss past
        a tolerance of epsilon.

        More details about the losses formulas can be found in the
        :ref:`User Guide <sgd_mathematical_formulation>`.

        .. deprecated:: 1.0
            The loss 'squared_loss' was deprecated in v1.0 and will be removed
            in version 1.2. Use `loss='squared_error'` which is equivalent.

    penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. The higher the
        value, the stronger the regularization.
        Also used to compute the learning rate when set to `learning_rate` is
        set to 'optimal'.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Only used if `penalty` is 'elasticnet'.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float, default=1e-3
        The stopping criterion. If it is not None, training will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    learning_rate : str, default='invscaling'
        The learning rate schedule:

        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where t0 is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': eta = eta0, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          early_stopping is True, the current learning rate is divided by 5.

            .. versionadded:: 0.20
                Added 'adaptive' option

    eta0 : float, default=0.01
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.01.

    power_t : float, default=0.25
        The exponent for inverse scaling learning rate.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score returned by the `score` method is not
        improving by at least `tol` for `n_iter_no_change` consecutive
        epochs.

        .. versionadded:: 0.20
            Added 'early_stopping' option

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if `early_stopping` is True.

        .. versionadded:: 0.20
            Added 'validation_fraction' option

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        fitting.
        Convergence is checked against the training loss or the
        validation loss depending on the `early_stopping` parameter.

        .. versionadded:: 0.20
            Added 'n_iter_no_change' option

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights across all
        updates and stores the result in the ``coef_`` attribute. If set to
        an int greater than 1, averaging will begin once the total number of
        samples seen reaches `average`. So ``average=10`` will begin
        averaging after seeing 10 samples.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,)
        The intercept term.

    n_iter_ : int
        The actual number of iterations before reaching the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    HuberRegressor : Linear regression model that is robust to outliers.
    Lars : Least Angle Regression model.
    Lasso : Linear Model trained with L1 prior as regularizer.
    RANSACRegressor : RANSAC (RANdom SAmple Consensus) algorithm.
    Ridge : Linear least squares with l2 regularization.
    sklearn.svm.SVR : Epsilon-Support Vector Regression.
    TheilSenRegressor : Theil-Sen Estimator robust multivariate regression model.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import SGDRegressor
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> # Always scale the input. The most convenient way is to use a pipeline.
    >>> reg = make_pipeline(StandardScaler(),
    ...                     SGDRegressor(max_iter=1000, tol=1e-3))
    >>> reg.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('sgdregressor', SGDRegressor())])
    """

    def __init__(
        self,
        loss="squared_error",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        random_state=None,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        warm_start=False,
        average=False,
    ):
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            warm_start=warm_start,
            average=average,
        )

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }


class SGDOneClassSVM(BaseSGD, OutlierMixin):
    """Solves linear One-Class SVM using Stochastic Gradient Descent.

    This implementation is meant to be used with a kernel approximation
    technique (e.g. `sklearn.kernel_approximation.Nystroem`) to obtain results
    similar to `sklearn.svm.OneClassSVM` which uses a Gaussian kernel by
    default.

    Read more in the :ref:`User Guide <sgd_online_one_class_svm>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    nu : float, default=0.5
        The nu parameter of the One Class SVM: an upper bound on the
        fraction of training errors and a lower bound of the fraction of
        support vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        `partial_fit`. Defaults to 1000.

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol). Defaults to 1e-3.

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.
        Defaults to True.

    verbose : int, default=0
        The verbosity level.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    learning_rate : {'constant', 'optimal', 'invscaling', 'adaptive'}, default='optimal'
        The learning rate schedule to use with `fit`. (If using `partial_fit`,
        learning rate must be controlled directly).

        - 'constant': `eta = eta0`
        - 'optimal': `eta = 1.0 / (alpha * (t + t0))`
          where t0 is chosen by a heuristic proposed by Leon Bottou.
        - 'invscaling': `eta = eta0 / pow(t, power_t)`
        - 'adaptive': eta = eta0, as long as the training keeps decreasing.
          Each time n_iter_no_change consecutive epochs fail to decrease the
          training loss by tol or fail to increase validation score by tol if
          early_stopping is True, the current learning rate is divided by 5.

    eta0 : float, default=0.0
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.0 as eta0 is not used by
        the default schedule 'optimal'.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate [default 0.5].

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So ``average=10`` will begin averaging after seeing 10
        samples.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weights assigned to the features.

    offset_ : ndarray of shape (1,)
        Offset used to define the decision function from the raw scores.
        We have the relation: decision_function = score_samples - offset.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    loss_function_ : concrete ``LossFunction``

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.

    Notes
    -----
    This estimator has a linear complexity in the number of training samples
    and is thus better suited than the `sklearn.svm.OneClassSVM`
    implementation for datasets with a large number of training samples (say
    > 10,000).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> clf = linear_model.SGDOneClassSVM(random_state=42)
    >>> clf.fit(X)
    SGDOneClassSVM(random_state=42)

    >>> print(clf.predict([[4, 4]]))
    [1]
    """

    loss_functions = {"hinge": (Hinge, 1.0)}

    def __init__(
        self,
        nu=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        warm_start=False,
        average=False,
    ):

        alpha = nu / 2
        self.nu = nu
        super(SGDOneClassSVM, self).__init__(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            C=1.0,
            l1_ratio=0,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=DEFAULT_EPSILON,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            warm_start=warm_start,
            average=average,
        )

    def _validate_params(self, for_partial_fit=False):
        """Validate input params."""
        if not (0 < self.nu <= 1):
            raise ValueError("nu must be in (0, 1], got nu=%f" % self.nu)

        super(SGDOneClassSVM, self)._validate_params(for_partial_fit=for_partial_fit)

    def _fit_one_class(self, X, alpha, C, sample_weight, learning_rate, max_iter):
        """Uses SGD implementation with X and y=np.ones(n_samples)."""

        # The One-Class SVM uses the SGD implementation with
        # y=np.ones(n_samples).
        n_samples = X.shape[0]
        y = np.ones(n_samples, dtype=np.float64, order="C")

        dataset, offset_decay = make_dataset(X, y, sample_weight)

        penalty_type = self._get_penalty_type(self.penalty)
        learning_rate_type = self._get_learning_rate_type(learning_rate)

        # early stopping is set to False for the One-Class SVM. thus
        # validation_mask and validation_score_cb will be set to values
        # associated to early_stopping=False in _make_validation_split and
        # _make_validation_score_cb respectively.
        validation_mask = self._make_validation_split(y)
        validation_score_cb = self._make_validation_score_cb(
            validation_mask, X, y, sample_weight
        )

        random_state = check_random_state(self.random_state)
        # numpy mtrand expects a C long which is a signed 32 bit integer under
        # Windows
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        tol = self.tol if self.tol is not None else -np.inf

        one_class = 1
        # There are no class weights for the One-Class SVM and they are
        # therefore set to 1.
        pos_weight = 1
        neg_weight = 1

        if self.average:
            coef = self._standard_coef
            intercept = self._standard_intercept
            average_coef = self._average_coef
            average_intercept = self._average_intercept
        else:
            coef = self.coef_
            intercept = 1 - self.offset_
            average_coef = None  # Not used
            average_intercept = [0]  # Not used

        coef, intercept, average_coef, average_intercept, self.n_iter_ = _plain_sgd(
            coef,
            intercept[0],
            average_coef,
            average_intercept[0],
            self.loss_function_,
            penalty_type,
            alpha,
            C,
            self.l1_ratio,
            dataset,
            validation_mask,
            self.early_stopping,
            validation_score_cb,
            int(self.n_iter_no_change),
            max_iter,
            tol,
            int(self.fit_intercept),
            int(self.verbose),
            int(self.shuffle),
            seed,
            neg_weight,
            pos_weight,
            learning_rate_type,
            self.eta0,
            self.power_t,
            one_class,
            self.t_,
            offset_decay,
            self.average,
        )

        self.t_ += self.n_iter_ * n_samples

        if self.average > 0:

            self._average_intercept = np.atleast_1d(average_intercept)
            self._standard_intercept = np.atleast_1d(intercept)

            if self.average <= self.t_ - 1.0:
                # made enough updates for averaging to be taken into account
                self.coef_ = average_coef
                self.offset_ = 1 - np.atleast_1d(average_intercept)
            else:
                self.coef_ = coef
                self.offset_ = 1 - np.atleast_1d(intercept)

        else:
            self.offset_ = 1 - np.atleast_1d(intercept)

    def _partial_fit(
        self,
        X,
        alpha,
        C,
        loss,
        learning_rate,
        max_iter,
        sample_weight,
        coef_init,
        offset_init,
    ):
        first_call = getattr(self, "coef_", None) is None
        X = self._validate_data(
            X,
            None,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
            reset=first_call,
        )

        n_features = X.shape[1]

        # Allocate datastructures from input arguments
        sample_weight = _check_sample_weight(sample_weight, X)

        # We use intercept = 1 - offset where intercept is the intercept of
        # the SGD implementation and offset is the offset of the One-Class SVM
        # optimization problem.
        if getattr(self, "coef_", None) is None or coef_init is not None:
            self._allocate_parameter_mem(1, n_features, coef_init, offset_init, 1)
        elif n_features != self.coef_.shape[-1]:
            raise ValueError(
                "Number of features %d does not match previous data %d."
                % (n_features, self.coef_.shape[-1])
            )

        if self.average and getattr(self, "_average_coef", None) is None:
            self._average_coef = np.zeros(n_features, dtype=np.float64, order="C")
            self._average_intercept = np.zeros(1, dtype=np.float64, order="C")

        self.loss_function_ = self._get_loss_function(loss)
        if not hasattr(self, "t_"):
            self.t_ = 1.0

        # delegate to concrete training procedure
        self._fit_one_class(
            X,
            alpha=alpha,
            C=C,
            learning_rate=learning_rate,
            sample_weight=sample_weight,
            max_iter=max_iter,
        )

        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        """Fit linear One-Class SVM with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of the training data.
        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """

        alpha = self.nu / 2
        self._validate_params(for_partial_fit=True)

        return self._partial_fit(
            X,
            alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            sample_weight=sample_weight,
            coef_init=None,
            offset_init=None,
        )

    def _fit(
        self,
        X,
        alpha,
        C,
        loss,
        learning_rate,
        coef_init=None,
        offset_init=None,
        sample_weight=None,
    ):
        self._validate_params()

        if self.warm_start and hasattr(self, "coef_"):
            if coef_init is None:
                coef_init = self.coef_
            if offset_init is None:
                offset_init = self.offset_
        else:
            self.coef_ = None
            self.offset_ = None

        # Clear iteration count for multiple call to fit.
        self.t_ = 1.0

        self._partial_fit(
            X,
            alpha,
            C,
            loss,
            learning_rate,
            self.max_iter,
            sample_weight,
            coef_init,
            offset_init,
        )

        if (
            self.tol is not None
            and self.tol > -np.inf
            and self.n_iter_ == self.max_iter
        ):
            warnings.warn(
                "Maximum number of iteration reached before "
                "convergence. Consider increasing max_iter to "
                "improve the fit.",
                ConvergenceWarning,
            )

        return self

    def fit(self, X, y=None, coef_init=None, offset_init=None, sample_weight=None):
        """Fit linear One-Class SVM with Stochastic Gradient Descent.

        This solves an equivalent optimization problem of the
        One-Class SVM primal optimization problem and returns a weight vector
        w and an offset rho such that the decision function is given by
        <w, x> - rho.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        coef_init : array, shape (n_classes, n_features)
            The initial coefficients to warm-start the optimization.

        offset_init : array, shape (n_classes,)
            The initial offset to warm-start the optimization.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed. These weights will
            be multiplied with class_weight (passed through the
            constructor) if class_weight is specified.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """

        alpha = self.nu / 2
        self._fit(
            X,
            alpha=alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            coef_init=coef_init,
            offset_init=offset_init,
            sample_weight=sample_weight,
        )

        return self

    def decision_function(self, X):
        """Signed distance to the separating hyperplane.

        Signed distance is positive for an inlier and negative for an
        outlier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        dec : array-like, shape (n_samples,)
            Decision function values of the samples.
        """

        check_is_fitted(self, "coef_")

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        decisions = safe_sparse_dot(X, self.coef_.T, dense_output=True) - self.offset_

        return decisions.ravel()

    def score_samples(self, X):
        """Raw scoring function of the samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        score_samples : array-like, shape (n_samples,)
            Unshiffted scoring function values of the samples.
        """
        score_samples = self.decision_function(X) + self.offset_
        return score_samples

    def predict(self, X):
        """Return labels (1 inlier, -1 outlier) of the samples.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Testing data.

        Returns
        -------
        y : array, shape (n_samples,)
            Labels of the samples.
        """
        y = (self.decision_function(X) >= 0).astype(np.int32)
        y[y == 0] = -1  # for consistency with outlier detectors
        return y

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                )
            }
        }
