# Author: Mathieu Blondel <mathieu@mblondel.org>
#         Arnaud Joly <a.joly@ulg.ac.be>
#         Maheshakya Wijewardena <maheshakya.10@cse.mrt.ac.lk>
# License: BSD 3 clause

import warnings
import numpy as np
import scipy.sparse as sp

from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .base import MultiOutputMixin
from .utils import check_random_state
from .utils import deprecated
from .utils.validation import _num_samples
from .utils.validation import check_array
from .utils.validation import check_consistent_length
from .utils.validation import check_is_fitted, _check_sample_weight
from .utils.random import _random_choice_csc
from .utils.stats import _weighted_percentile
from .utils.multiclass import class_distribution


class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    """DummyClassifier makes predictions that ignore the input features.

    This classifier serves as a simple baseline to compare against other more
    complex classifiers.

    The specific behavior of the baseline is selected with the `strategy`
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the `X` argument to `fit` and `predict`. The predictions, however,
    typically depend on values observed in the `y` parameter passed to `fit`.

    Note that the "stratified" and "uniform" strategies lead to
    non-deterministic predictions that can be rendered deterministic by setting
    the `random_state` parameter if needed. The other strategies are naturally
    deterministic and, once fit, always return a the same constant prediction
    for any value of `X`.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    strategy : {"most_frequent", "prior", "stratified", "uniform", \
            "constant"}, default="prior"
        Strategy to use to generate predictions.

        * "most_frequent": the `predict` method always returns the most
          frequent class label in the observed `y` argument passed to `fit`.
          The `predict_proba` method returns the matching one-hot encoded
          vector.
        * "prior": the `predict` method always returns the most frequent
          class label in the observed `y` argument passed to `fit` (like
          "most_frequent"). ``predict_proba`` always returns the empirical
          class distribution of `y` also known as the empirical class prior
          distribution.
        * "stratified": the `predict_proba` method randomly samples one-hot
          vectors from a multinomial distribution parametrized by the empirical
          class prior probabilities.
          The `predict` method returns the class label which got probability
          one in the one-hot vector of `predict_proba`.
          Each sampled row of both methods is therefore independent and
          identically distributed.
        * "uniform": generates predictions uniformly at random from the list
          of unique classes observed in `y`, i.e. each class has equal
          probability.
        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class.

          .. versionchanged:: 0.24
             The default value of `strategy` has changed to "prior" in version
             0.24.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness to generate the predictions when
        ``strategy='stratified'`` or ``strategy='uniform'``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    constant : int or str or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of such arrays
        Unique class labels observed in `y`. For multi-output classification
        problems, this attribute is a list of arrays as each output has an
        independent set of possible classes.

    n_classes_ : int or list of int
        Number of label for each output.

    class_prior_ : ndarray of shape (n_classes,) or list of such arrays
        Frequency of each class observed in `y`. For multioutput classification
        problems, this is computed independently for each output.

    n_outputs_ : int
        Number of outputs.

    n_features_in_ : `None`
        Always set to `None`.

        .. versionadded:: 0.24
        .. deprecated:: 1.0
            Will be removed in 1.0

    sparse_output_ : bool
        True if the array returned from predict is to be in sparse CSC format.
        Is automatically set to True if the input `y` is passed in sparse
        format.

    See Also
    --------
    DummyRegressor : Regressor that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyClassifier
    >>> X = np.array([-1, 1, 1, 1])
    >>> y = np.array([0, 1, 1, 1])
    >>> dummy_clf = DummyClassifier(strategy="most_frequent")
    >>> dummy_clf.fit(X, y)
    DummyClassifier(strategy='most_frequent')
    >>> dummy_clf.predict(X)
    array([1, 1, 1, 1])
    >>> dummy_clf.score(X, y)
    0.75
    """

    def __init__(self, *, strategy="prior", random_state=None, constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        allowed_strategies = (
            "most_frequent",
            "stratified",
            "uniform",
            "constant",
            "prior",
        )

        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Unknown strategy type: %s, expected one of %s."
                % (self.strategy, allowed_strategies)
            )

        self._strategy = self.strategy

        if self._strategy == "uniform" and sp.issparse(y):
            y = y.toarray()
            warnings.warn(
                "A local copy of the target data has been converted "
                "to a numpy array. Predicting on sparse target data "
                "with the uniform strategy would not save memory "
                "and would be slower.",
                UserWarning,
            )

        self.sparse_output_ = sp.issparse(y)

        if not self.sparse_output_:
            y = np.asarray(y)
            y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        check_consistent_length(X, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if self._strategy == "constant":
            if self.constant is None:
                raise ValueError(
                    "Constant target value has to be specified "
                    "when the constant strategy is used."
                )
            else:
                constant = np.reshape(np.atleast_1d(self.constant), (-1, 1))
                if constant.shape[0] != self.n_outputs_:
                    raise ValueError(
                        "Constant target value should have shape (%d, 1)."
                        % self.n_outputs_
                    )

        (self.classes_, self.n_classes_, self.class_prior_) = class_distribution(
            y, sample_weight
        )

        if self._strategy == "constant":
            for k in range(self.n_outputs_):
                if not any(constant[k][0] == c for c in self.classes_[k]):
                    # Checking in case of constant strategy if the constant
                    # provided by the user is in y.
                    err_msg = (
                        "The constant target value must be present in "
                        "the training data. You provided constant={}. "
                        "Possible values are: {}.".format(
                            self.constant, list(self.classes_[k])
                        )
                    )
                    raise ValueError(err_msg)

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            self.class_prior_ = self.class_prior_[0]

        return self

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.
        """
        check_is_fitted(self)

        # numpy random_state expects Python int and not long as size argument
        # under Windows
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1:
            # Get same type even for self.n_outputs_ == 1
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]
        # Compute probability only once
        if self._strategy == "stratified":
            proba = self.predict_proba(X)
            if self.n_outputs_ == 1:
                proba = [proba]

        if self.sparse_output_:
            class_prob = None
            if self._strategy in ("most_frequent", "prior"):
                classes_ = [np.array([cp.argmax()]) for cp in class_prior_]

            elif self._strategy == "stratified":
                class_prob = class_prior_

            elif self._strategy == "uniform":
                raise ValueError(
                    "Sparse target prediction is not "
                    "supported with the uniform strategy"
                )

            elif self._strategy == "constant":
                classes_ = [np.array([c]) for c in constant]

            y = _random_choice_csc(n_samples, classes_, class_prob, self.random_state)
        else:
            if self._strategy in ("most_frequent", "prior"):
                y = np.tile(
                    [
                        classes_[k][class_prior_[k].argmax()]
                        for k in range(self.n_outputs_)
                    ],
                    [n_samples, 1],
                )

            elif self._strategy == "stratified":
                y = np.vstack(
                    [
                        classes_[k][proba[k].argmax(axis=1)]
                        for k in range(self.n_outputs_)
                    ]
                ).T

            elif self._strategy == "uniform":
                ret = [
                    classes_[k][rs.randint(n_classes_[k], size=n_samples)]
                    for k in range(self.n_outputs_)
                ]
                y = np.vstack(ret).T

            elif self._strategy == "constant":
                y = np.tile(self.constant, (n_samples, 1))

            if self.n_outputs_ == 1:
                y = np.ravel(y)

        return y

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        check_is_fitted(self)

        # numpy random_state expects Python int and not long as size argument
        # under Windows
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1:
            # Get same type even for self.n_outputs_ == 1
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]

        P = []
        for k in range(self.n_outputs_):
            if self._strategy == "most_frequent":
                ind = class_prior_[k].argmax()
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0
            elif self._strategy == "prior":
                out = np.ones((n_samples, 1)) * class_prior_[k]

            elif self._strategy == "stratified":
                out = rs.multinomial(1, class_prior_[k], size=n_samples)
                out = out.astype(np.float64)

            elif self._strategy == "uniform":
                out = np.ones((n_samples, n_classes_[k]), dtype=np.float64)
                out /= n_classes_[k]

            elif self._strategy == "constant":
                ind = np.where(classes_[k] == constant[k])
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0

            P.append(out)

        if self.n_outputs_ == 1:
            P = P[0]

        return P

    def predict_log_proba(self, X):
        """
        Return log probability estimates for the test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data.

        Returns
        -------
        P : ndarray of shape (n_samples, n_classes) or list of such arrays
            Returns the log probability of the sample for each class in
            the model, where classes are ordered arithmetically for each
            output.
        """
        proba = self.predict_proba(X)
        if self.n_outputs_ == 1:
            return np.log(proba)
        else:
            return [np.log(p) for p in proba]

    def _more_tags(self):
        return {
            "poor_score": True,
            "no_validation": True,
            "_xfail_checks": {
                "check_methods_subset_invariance": "fails for the predict method",
                "check_methods_sample_order_invariance": "fails for the predict method",
            },
        }

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since DummyClassifier
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "`n_features_in_` is deprecated in 1.0 and will be removed in 1.2."
    )
    @property
    def n_features_in_(self):
        check_is_fitted(self)
        return None


class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Regressor that makes predictions using simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_features_in_ : `None`
        Always set to `None`.

        .. versionadded:: 0.24
        .. deprecated:: 1.0
            Will be removed in 1.0

    n_outputs_ : int
        Number of outputs.

    See Also
    --------
    DummyClassifier: Classifier that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyRegressor
    >>> X = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([2.0, 3.0, 5.0, 10.0])
    >>> dummy_regr = DummyRegressor(strategy="mean")
    >>> dummy_regr.fit(X, y)
    DummyRegressor()
    >>> dummy_regr.predict(X)
    array([5., 5., 5., 5.])
    >>> dummy_regr.score(X, y)
    0.0
    """

    def __init__(self, *, strategy="mean", constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    def fit(self, X, y, sample_weight=None):
        """Fit the random regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        allowed_strategies = ("mean", "median", "quantile", "constant")
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Unknown strategy type: %s, expected one of %s."
                % (self.strategy, allowed_strategies)
            )

        y = check_array(y, ensure_2d=False)
        if len(y) == 0:
            raise ValueError("y must not be empty.")

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        check_consistent_length(X, y, sample_weight)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if self.strategy == "mean":
            self.constant_ = np.average(y, axis=0, weights=sample_weight)

        elif self.strategy == "median":
            if sample_weight is None:
                self.constant_ = np.median(y, axis=0)
            else:
                self.constant_ = [
                    _weighted_percentile(y[:, k], sample_weight, percentile=50.0)
                    for k in range(self.n_outputs_)
                ]

        elif self.strategy == "quantile":
            if self.quantile is None or not np.isscalar(self.quantile):
                raise ValueError(
                    "Quantile must be a scalar in the range [0.0, 1.0], but got %s."
                    % self.quantile
                )

            percentile = self.quantile * 100.0
            if sample_weight is None:
                self.constant_ = np.percentile(y, axis=0, q=percentile)
            else:
                self.constant_ = [
                    _weighted_percentile(y[:, k], sample_weight, percentile=percentile)
                    for k in range(self.n_outputs_)
                ]

        elif self.strategy == "constant":
            if self.constant is None:
                raise TypeError(
                    "Constant target value has to be specified "
                    "when the constant strategy is used."
                )

            self.constant = check_array(
                self.constant,
                accept_sparse=["csr", "csc", "coo"],
                ensure_2d=False,
                ensure_min_samples=0,
            )

            if self.n_outputs_ != 1 and self.constant.shape[0] != y.shape[1]:
                raise ValueError(
                    "Constant target value should have shape (%d, 1)." % y.shape[1]
                )

            self.constant_ = self.constant

        self.constant_ = np.reshape(self.constant_, (1, -1))
        return self

    def predict(self, X, return_std=False):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.

            .. versionadded:: 0.20

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.

        y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Standard deviation of predictive distribution of query points.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)

        y = np.full(
            (n_samples, self.n_outputs_),
            self.constant_,
            dtype=np.array(self.constant_).dtype,
        )
        y_std = np.zeros((n_samples, self.n_outputs_))

        if self.n_outputs_ == 1:
            y = np.ravel(y)
            y_std = np.ravel(y_std)

        return (y, y_std) if return_std else y

    def _more_tags(self):
        return {"poor_score": True, "no_validation": True}

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as `(1 - u/v)`, where `u` is the
        residual sum of squares `((y_true - y_pred) ** 2).sum()` and `v` is the
        total sum of squares `((y_true - y_true.mean()) ** 2).sum()`. The best
        possible score is 1.0 and it can be negative (because the model can be
        arbitrarily worse). A constant model that always predicts the expected
        value of y, disregarding the input features, would get a R^2 score of
        0.0.

        Parameters
        ----------
        X : None or array-like of shape (n_samples, n_features)
            Test samples. Passing None as test samples gives the same result
            as passing real test samples, since `DummyRegressor`
            operates independently of the sampled observations.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of `self.predict(X)` wrt. y.
        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)

    # TODO: Remove in 1.2
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "`n_features_in_` is deprecated in 1.0 and will be removed in 1.2."
    )
    @property
    def n_features_in_(self):
        check_is_fitted(self)
        return None
