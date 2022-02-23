import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from .validation import _num_samples, check_array, check_is_fitted


class ArraySlicingWrapper:
    """
    Parameters
    ----------
    array
    """

    def __init__(self, array):
        self.array = array

    def __getitem__(self, aslice):
        return MockDataFrame(self.array[aslice])


class MockDataFrame:
    """
    Parameters
    ----------
    array
    """

    # have shape and length but don't support indexing.

    def __init__(self, array):
        self.array = array
        self.values = array
        self.shape = array.shape
        self.ndim = array.ndim
        # ugly hack to make iloc work.
        self.iloc = ArraySlicingWrapper(array)

    def __len__(self):
        return len(self.array)

    def __array__(self, dtype=None):
        # Pandas data frames also are array-like: we want to make sure that
        # input validation in cross-validation does not try to call that
        # method.
        return self.array

    def __eq__(self, other):
        return MockDataFrame(self.array == other.array)

    def __ne__(self, other):
        return not self == other

    def take(self, indices, axis=0):
        return MockDataFrame(self.array.take(indices, axis=axis))


class CheckingClassifier(ClassifierMixin, BaseEstimator):
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of `X` and `y`in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.

    Can also be used to check if `fit_params` are passed correctly, and
    to force a certain score to be returned.

    Parameters
    ----------
    check_y, check_X : callable, default=None
        The callable used to validate `X` and `y`. These callable should return
        a bool where `False` will trigger an `AssertionError`.

    check_y_params, check_X_params : dict, default=None
        The optional parameters to pass to `check_X` and `check_y`.

    methods_to_check : "all" or list of str, default="all"
        The methods in which the checks should be applied. By default,
        all checks will be done on all methods (`fit`, `predict`,
        `predict_proba`, `decision_function` and `score`).

    foo_param : int, default=0
        A `foo` param. When `foo > 1`, the output of :meth:`score` will be 1
        otherwise it is 0.

    expected_fit_params : list of str, default=None
        A list of the expected parameters given when calling `fit`.

    Attributes
    ----------
    classes_ : int
        The classes seen during `fit`.

    n_features_in_ : int
        The number of features seen during `fit`.

    Examples
    --------
    >>> from sklearn.utils._mocking import CheckingClassifier

    This helper allow to assert to specificities regarding `X` or `y`. In this
    case we expect `check_X` or `check_y` to return a boolean.

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = CheckingClassifier(check_X=lambda x: x.shape == (150, 4))
    >>> clf.fit(X, y)
    CheckingClassifier(...)

    We can also provide a check which might raise an error. In this case, we
    expect `check_X` to return `X` and `check_y` to return `y`.

    >>> from sklearn.utils import check_array
    >>> clf = CheckingClassifier(check_X=check_array)
    >>> clf.fit(X, y)
    CheckingClassifier(...)
    """

    def __init__(
        self,
        *,
        check_y=None,
        check_y_params=None,
        check_X=None,
        check_X_params=None,
        methods_to_check="all",
        foo_param=0,
        expected_fit_params=None,
    ):
        self.check_y = check_y
        self.check_y_params = check_y_params
        self.check_X = check_X
        self.check_X_params = check_X_params
        self.methods_to_check = methods_to_check
        self.foo_param = foo_param
        self.expected_fit_params = expected_fit_params

    def _check_X_y(self, X, y=None, should_be_fitted=True):
        """Validate X and y and make extra check.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data set.
        y : array-like of shape (n_samples), default=None
            The corresponding target, by default None.
        should_be_fitted : bool, default=True
            Whether or not the classifier should be already fitted.
            By default True.

        Returns
        -------
        X, y
        """
        if should_be_fitted:
            check_is_fitted(self)
        if self.check_X is not None:
            params = {} if self.check_X_params is None else self.check_X_params
            checked_X = self.check_X(X, **params)
            if isinstance(checked_X, (bool, np.bool_)):
                assert checked_X
            else:
                X = checked_X
        if y is not None and self.check_y is not None:
            params = {} if self.check_y_params is None else self.check_y_params
            checked_y = self.check_y(y, **params)
            if isinstance(checked_y, (bool, np.bool_)):
                assert checked_y
            else:
                y = checked_y
        return X, y

    def fit(self, X, y, **fit_params):
        """Fit classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_outputs) or (n_samples,), \
                default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        Returns
        -------
        self
        """
        assert _num_samples(X) == _num_samples(y)
        if self.methods_to_check == "all" or "fit" in self.methods_to_check:
            X, y = self._check_X_y(X, y, should_be_fitted=False)
        self.n_features_in_ = np.shape(X)[1]
        self.classes_ = np.unique(check_array(y, ensure_2d=False, allow_nd=True))
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            if missing:
                raise AssertionError(
                    f"Expected fit parameter(s) {list(missing)} not seen."
                )
            for key, value in fit_params.items():
                if _num_samples(value) != _num_samples(X):
                    raise AssertionError(
                        f"Fit parameter {key} has length {_num_samples(value)}"
                        f"; expected {_num_samples(X)}."
                    )

        return self

    def predict(self, X):
        """Predict the first class seen in `classes_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        preds : ndarray of shape (n_samples,)
            Predictions of the first class seens in `classes_`.
        """
        if self.methods_to_check == "all" or "predict" in self.methods_to_check:
            X, y = self._check_X_y(X)
        return self.classes_[np.zeros(_num_samples(X), dtype=int)]

    def predict_proba(self, X):
        """Predict probabilities for each class.

        Here, the dummy classifier will provide a probability of 1 for the
        first class of `classes_` and 0 otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The probabilities for each sample and class.
        """
        if self.methods_to_check == "all" or "predict_proba" in self.methods_to_check:
            X, y = self._check_X_y(X)
        proba = np.zeros((_num_samples(X), len(self.classes_)))
        proba[:, 0] = 1
        return proba

    def decision_function(self, X):
        """Confidence score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        decision : ndarray of shape (n_samples,) if n_classes == 2\
                else (n_samples, n_classes)
            Confidence score.
        """
        if (
            self.methods_to_check == "all"
            or "decision_function" in self.methods_to_check
        ):
            X, y = self._check_X_y(X)
        if len(self.classes_) == 2:
            # for binary classifier, the confidence score is related to
            # classes_[1] and therefore should be null.
            return np.zeros(_num_samples(X))
        else:
            decision = np.zeros((_num_samples(X), len(self.classes_)))
            decision[:, 0] = 1
            return decision

    def score(self, X=None, Y=None):
        """Fake score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Y : array-like of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            Either 0 or 1 depending of `foo_param` (i.e. `foo_param > 1 =>
            score=1` otherwise `score=0`).
        """
        if self.methods_to_check == "all" or "score" in self.methods_to_check:
            self._check_X_y(X, Y)
        if self.foo_param > 1:
            score = 1.0
        else:
            score = 0.0
        return score

    def _more_tags(self):
        return {"_skip_test": True, "X_types": ["1dlabel"]}


class NoSampleWeightWrapper(BaseEstimator):
    """Wrap estimator which will not expose `sample_weight`.

    Parameters
    ----------
    est : estimator, default=None
        The estimator to wrap.
    """

    def __init__(self, est=None):
        self.est = est

    def fit(self, X, y):
        return self.est.fit(X, y)

    def predict(self, X):
        return self.est.predict(X)

    def predict_proba(self, X):
        return self.est.predict_proba(X)

    def _more_tags(self):
        return {"_skip_test": True}
