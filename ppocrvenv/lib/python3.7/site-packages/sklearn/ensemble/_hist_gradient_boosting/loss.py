"""
This module contains the loss classes.

Specific losses are used for regression, binary classification or multiclass
classification.
"""
# Author: Nicolas Hug

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import expit, logsumexp, xlogy

from .common import Y_DTYPE
from .common import G_H_DTYPE
from ._loss import _update_gradients_least_squares
from ._loss import _update_gradients_hessians_least_squares
from ._loss import _update_gradients_least_absolute_deviation
from ._loss import _update_gradients_hessians_least_absolute_deviation
from ._loss import _update_gradients_hessians_binary_crossentropy
from ._loss import _update_gradients_hessians_categorical_crossentropy
from ._loss import _update_gradients_hessians_poisson
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils.stats import _weighted_percentile


class BaseLoss(ABC):
    """Base class for a loss."""

    def __init__(self, hessians_are_constant, n_threads=None):
        self.hessians_are_constant = hessians_are_constant
        self.n_threads = _openmp_effective_n_threads(n_threads)

    def __call__(self, y_true, raw_predictions, sample_weight):
        """Return the weighted average loss"""
        return np.average(
            self.pointwise_loss(y_true, raw_predictions), weights=sample_weight
        )

    @abstractmethod
    def pointwise_loss(self, y_true, raw_predictions):
        """Return loss value for each input"""

    # This variable indicates whether the loss requires the leaves values to
    # be updated once the tree has been trained. The trees are trained to
    # predict a Newton-Raphson step (see grower._finalize_leaf()). But for
    # some losses (e.g. least absolute deviation) we need to adjust the tree
    # values to account for the "line search" of the gradient descent
    # procedure. See the original paper Greedy Function Approximation: A
    # Gradient Boosting Machine by Friedman
    # (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) for the theory.
    need_update_leaves_values = False

    def init_gradients_and_hessians(self, n_samples, prediction_dim, sample_weight):
        """Return initial gradients and hessians.

        Unless hessians are constant, arrays are initialized with undefined
        values.

        Parameters
        ----------
        n_samples : int
            The number of samples passed to `fit()`.

        prediction_dim : int
            The dimension of a raw prediction, i.e. the number of trees
            built at each iteration. Equals 1 for regression and binary
            classification, or K where K is the number of classes for
            multiclass classification.

        sample_weight : array-like of shape(n_samples,) default=None
            Weights of training data.

        Returns
        -------
        gradients : ndarray, shape (prediction_dim, n_samples)
            The initial gradients. The array is not initialized.
        hessians : ndarray, shape (prediction_dim, n_samples)
            If hessians are constant (e.g. for `LeastSquares` loss, the
            array is initialized to ``1``. Otherwise, the array is allocated
            without being initialized.
        """
        shape = (prediction_dim, n_samples)
        gradients = np.empty(shape=shape, dtype=G_H_DTYPE)

        if self.hessians_are_constant:
            # If the hessians are constant, we consider they are equal to 1.
            # - This is correct for the half LS loss
            # - For LAD loss, hessians are actually 0, but they are always
            #   ignored anyway.
            hessians = np.ones(shape=(1, 1), dtype=G_H_DTYPE)
        else:
            hessians = np.empty(shape=shape, dtype=G_H_DTYPE)

        return gradients, hessians

    @abstractmethod
    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        """Return initial predictions (before the first iteration).

        Parameters
        ----------
        y_train : ndarray, shape (n_samples,)
            The target training values.

        sample_weight : array-like of shape(n_samples,) default=None
            Weights of training data.

        prediction_dim : int
            The dimension of one prediction: 1 for binary classification and
            regression, n_classes for multiclass classification.

        Returns
        -------
        baseline_prediction : float or ndarray, shape (1, prediction_dim)
            The baseline prediction.
        """

    @abstractmethod
    def update_gradients_and_hessians(
        self, gradients, hessians, y_true, raw_predictions, sample_weight
    ):
        """Update gradients and hessians arrays, inplace.

        The gradients (resp. hessians) are the first (resp. second) order
        derivatives of the loss for each sample with respect to the
        predictions of model, evaluated at iteration ``i - 1``.

        Parameters
        ----------
        gradients : ndarray, shape (prediction_dim, n_samples)
            The gradients (treated as OUT array).

        hessians : ndarray, shape (prediction_dim, n_samples) or \
            (1,)
            The hessians (treated as OUT array).

        y_true : ndarray, shape (n_samples,)
            The true target values or each training sample.

        raw_predictions : ndarray, shape (prediction_dim, n_samples)
            The raw_predictions (i.e. values from the trees) of the tree
            ensemble at iteration ``i - 1``.

        sample_weight : array-like of shape(n_samples,) default=None
            Weights of training data.
        """


class LeastSquares(BaseLoss):
    """Least squares loss, for regression.

    For a given sample x_i, least squares loss is defined as::

        loss(x_i) = 0.5 * (y_true_i - raw_pred_i)**2

    This actually computes the half least squares loss to simplify
    the computation of the gradients and get a unit hessian (and be consistent
    with what is done in LightGBM).
    """

    def __init__(self, sample_weight, n_threads=None):
        # If sample weights are provided, the hessians and gradients
        # are multiplied by sample_weight, which means the hessians are
        # equal to sample weights.
        super().__init__(
            hessians_are_constant=sample_weight is None, n_threads=n_threads
        )

    def pointwise_loss(self, y_true, raw_predictions):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        loss = 0.5 * np.power(y_true - raw_predictions, 2)
        return loss

    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        return np.average(y_train, weights=sample_weight)

    @staticmethod
    def inverse_link_function(raw_predictions):
        return raw_predictions

    def update_gradients_and_hessians(
        self, gradients, hessians, y_true, raw_predictions, sample_weight
    ):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        gradients = gradients.reshape(-1)
        if sample_weight is None:
            _update_gradients_least_squares(
                gradients, y_true, raw_predictions, self.n_threads
            )
        else:
            hessians = hessians.reshape(-1)
            _update_gradients_hessians_least_squares(
                gradients,
                hessians,
                y_true,
                raw_predictions,
                sample_weight,
                self.n_threads,
            )


class LeastAbsoluteDeviation(BaseLoss):
    """Least absolute deviation, for regression.

    For a given sample x_i, the loss is defined as::

        loss(x_i) = |y_true_i - raw_pred_i|
    """

    def __init__(self, sample_weight, n_threads=None):
        # If sample weights are provided, the hessians and gradients
        # are multiplied by sample_weight, which means the hessians are
        # equal to sample weights.
        super().__init__(
            hessians_are_constant=sample_weight is None, n_threads=n_threads
        )

    # This variable indicates whether the loss requires the leaves values to
    # be updated once the tree has been trained. The trees are trained to
    # predict a Newton-Raphson step (see grower._finalize_leaf()). But for
    # some losses (e.g. least absolute deviation) we need to adjust the tree
    # values to account for the "line search" of the gradient descent
    # procedure. See the original paper Greedy Function Approximation: A
    # Gradient Boosting Machine by Friedman
    # (https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) for the theory.
    need_update_leaves_values = True

    def pointwise_loss(self, y_true, raw_predictions):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        loss = np.abs(y_true - raw_predictions)
        return loss

    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        if sample_weight is None:
            return np.median(y_train)
        else:
            return _weighted_percentile(y_train, sample_weight, 50)

    @staticmethod
    def inverse_link_function(raw_predictions):
        return raw_predictions

    def update_gradients_and_hessians(
        self, gradients, hessians, y_true, raw_predictions, sample_weight
    ):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        gradients = gradients.reshape(-1)
        if sample_weight is None:
            _update_gradients_least_absolute_deviation(
                gradients,
                y_true,
                raw_predictions,
                self.n_threads,
            )
        else:
            hessians = hessians.reshape(-1)
            _update_gradients_hessians_least_absolute_deviation(
                gradients,
                hessians,
                y_true,
                raw_predictions,
                sample_weight,
                self.n_threads,
            )

    def update_leaves_values(self, grower, y_true, raw_predictions, sample_weight):
        # Update the values predicted by the tree with
        # median(y_true - raw_predictions).
        # See note about need_update_leaves_values in BaseLoss.

        # TODO: ideally this should be computed in parallel over the leaves
        # using something similar to _update_raw_predictions(), but this
        # requires a cython version of median()
        for leaf in grower.finalized_leaves:
            indices = leaf.sample_indices
            if sample_weight is None:
                median_res = np.median(y_true[indices] - raw_predictions[indices])
            else:
                median_res = _weighted_percentile(
                    y_true[indices] - raw_predictions[indices],
                    sample_weight=sample_weight[indices],
                    percentile=50,
                )
            leaf.value = grower.shrinkage * median_res
            # Note that the regularization is ignored here


class Poisson(BaseLoss):
    """Poisson deviance loss with log-link, for regression.

    For a given sample x_i, Poisson deviance loss is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_pred_i))
                    - y_true_i + exp(raw_pred_i))

    This actually computes half the Poisson deviance to simplify
    the computation of the gradients.
    """

    def __init__(self, sample_weight, n_threads=None):
        super().__init__(hessians_are_constant=False, n_threads=n_threads)

    inverse_link_function = staticmethod(np.exp)

    def pointwise_loss(self, y_true, raw_predictions):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        # TODO: For speed, we could remove the constant xlogy(y_true, y_true)
        # Advantage of this form: minimum of zero at raw_predictions = y_true.
        loss = (
            xlogy(y_true, y_true)
            - y_true * (raw_predictions + 1)
            + np.exp(raw_predictions)
        )
        return loss

    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        y_pred = np.average(y_train, weights=sample_weight)
        eps = np.finfo(y_train.dtype).eps
        y_pred = np.clip(y_pred, eps, None)
        return np.log(y_pred)

    def update_gradients_and_hessians(
        self, gradients, hessians, y_true, raw_predictions, sample_weight
    ):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        gradients = gradients.reshape(-1)
        hessians = hessians.reshape(-1)
        _update_gradients_hessians_poisson(
            gradients,
            hessians,
            y_true,
            raw_predictions,
            sample_weight,
            self.n_threads,
        )


class BinaryCrossEntropy(BaseLoss):
    """Binary cross-entropy loss, for binary classification.

    For a given sample x_i, the binary cross-entropy loss is defined as the
    negative log-likelihood of the model which can be expressed as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).
    """

    def __init__(self, sample_weight, n_threads=None):
        super().__init__(hessians_are_constant=False, n_threads=n_threads)

    inverse_link_function = staticmethod(expit)

    def pointwise_loss(self, y_true, raw_predictions):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        # logaddexp(0, x) = log(1 + exp(x))
        loss = np.logaddexp(0, raw_predictions) - y_true * raw_predictions
        return loss

    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        if prediction_dim > 2:
            raise ValueError(
                "loss='binary_crossentropy' is not defined for multiclass"
                " classification with n_classes=%d, use"
                " loss='categorical_crossentropy' instead" % prediction_dim
            )
        proba_positive_class = np.average(y_train, weights=sample_weight)
        eps = np.finfo(y_train.dtype).eps
        proba_positive_class = np.clip(proba_positive_class, eps, 1 - eps)
        # log(x / 1 - x) is the anti function of sigmoid, or the link function
        # of the Binomial model.
        return np.log(proba_positive_class / (1 - proba_positive_class))

    def update_gradients_and_hessians(
        self, gradients, hessians, y_true, raw_predictions, sample_weight
    ):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        gradients = gradients.reshape(-1)
        hessians = hessians.reshape(-1)
        _update_gradients_hessians_binary_crossentropy(
            gradients, hessians, y_true, raw_predictions, sample_weight, self.n_threads
        )

    def predict_proba(self, raw_predictions):
        # shape (1, n_samples) --> (n_samples,). reshape(-1) is more likely to
        # return a view.
        raw_predictions = raw_predictions.reshape(-1)
        proba = np.empty((raw_predictions.shape[0], 2), dtype=Y_DTYPE)
        proba[:, 1] = expit(raw_predictions)
        proba[:, 0] = 1 - proba[:, 1]
        return proba


class CategoricalCrossEntropy(BaseLoss):
    """Categorical cross-entropy loss, for multiclass classification.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the model and generalizes the binary
    cross-entropy to more than 2 classes.
    """

    def __init__(self, sample_weight, n_threads=None):
        super().__init__(hessians_are_constant=False, n_threads=n_threads)

    def pointwise_loss(self, y_true, raw_predictions):
        one_hot_true = np.zeros_like(raw_predictions)
        prediction_dim = raw_predictions.shape[0]
        for k in range(prediction_dim):
            one_hot_true[k, :] = y_true == k

        loss = logsumexp(raw_predictions, axis=0) - (
            one_hot_true * raw_predictions
        ).sum(axis=0)
        return loss

    def get_baseline_prediction(self, y_train, sample_weight, prediction_dim):
        init_value = np.zeros(shape=(prediction_dim, 1), dtype=Y_DTYPE)
        eps = np.finfo(y_train.dtype).eps
        for k in range(prediction_dim):
            proba_kth_class = np.average(y_train == k, weights=sample_weight)
            proba_kth_class = np.clip(proba_kth_class, eps, 1 - eps)
            init_value[k, :] += np.log(proba_kth_class)

        return init_value

    def update_gradients_and_hessians(
        self, gradients, hessians, y_true, raw_predictions, sample_weight
    ):
        _update_gradients_hessians_categorical_crossentropy(
            gradients, hessians, y_true, raw_predictions, sample_weight, self.n_threads
        )

    def predict_proba(self, raw_predictions):
        # TODO: This could be done in parallel
        # compute softmax (using exp(log(softmax)))
        proba = np.exp(
            raw_predictions - logsumexp(raw_predictions, axis=0)[np.newaxis, :]
        )
        return proba.T


_LOSSES = {
    "squared_error": LeastSquares,
    "absolute_error": LeastAbsoluteDeviation,
    "binary_crossentropy": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "poisson": Poisson,
}
