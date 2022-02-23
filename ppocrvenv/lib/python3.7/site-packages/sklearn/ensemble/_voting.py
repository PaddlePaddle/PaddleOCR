"""
Soft Voting/Majority Rule classifier and Voting regressor.

This module contains:
 - A Soft Voting/Majority Rule classifier for classification estimators.
 - A Voting regressor for regression estimators.
"""

# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#          Gilles Louppe <g.louppe@gmail.com>,
#          Ramil Nugmanov <stsouko@live.ru>
#          Mohamed Ali Jamaoui <m.ali.jamaoui@gmail.com>
#
# License: BSD 3 clause

from abc import abstractmethod

import numpy as np

from joblib import Parallel

from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..base import TransformerMixin
from ..base import clone
from ._base import _fit_single_estimator
from ._base import _BaseHeterogeneousEnsemble
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..utils.validation import column_or_1d
from ..exceptions import NotFittedError
from ..utils._estimator_html_repr import _VisualBlock
from ..utils.fixes import delayed


class _BaseVoting(TransformerMixin, _BaseHeterogeneousEnsemble):
    """Base class for voting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return "(%d of %d) Processing %s" % (idx, total, name)

    @property
    def _weights_not_none(self):
        """Get the weights of not `None` estimators."""
        if self.weights is None:
            return None
        return [w for est, w in zip(self.estimators, self.weights) if est[1] != "drop"]

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict(X) for est in self.estimators_]).T

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal"
                "; got %d weights, %d estimators"
                % (len(self.weights), len(self.estimators))
            )

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf),
                X,
                y,
                sample_weight=sample_weight,
                message_clsname="Voting",
                message=self._log_message(names[idx], idx + 1, len(clfs)),
            )
            for idx, clf in enumerate(clfs)
            if clf != "drop"
        )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

            if hasattr(current_est, "feature_names_in_"):
                self.feature_names_in_ = current_est.feature_names_in_

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Return class labels or probabilities for each estimator.

        Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            Input samples.

        y : ndarray of shape (n_samples,), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        return super().fit_transform(X, y, **fit_params)

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.estimators_[0].n_features_in_

    def _sk_visual_block_(self):
        names, estimators = zip(*self.estimators)
        return _VisualBlock("parallel", estimators, names=names)

    def _more_tags(self):
        return {"preserves_dtype": []}


class VotingClassifier(ClassifierMixin, _BaseVoting):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    Read more in the :ref:`User Guide <voting_classifier>`.

    .. versionadded:: 0.17

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'``
        using ``set_params``.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    le_ : :class:`~sklearn.preprocessing.LabelEncoder`
        Transformer used to encode the labels during fit and decode during
        prediction.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0

    See Also
    --------
    VotingRegressor : Prediction voting regressor.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)
    """

    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
    ):
        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                "Multilabel and multi-output classification is not supported."
            )

        if self.voting not in ("soft", "hard"):
            raise ValueError(
                "Voting must be 'soft' or 'hard'; got (voting=%r)" % self.voting
            )

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        return super().fit(X, transformed_y, sample_weight)

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )

        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _check_voting(self):
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={repr(self.voting)}"
            )
        return True

    @available_if(_check_voting)
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        avg : array-like of shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        check_is_fitted(self)
        avg = np.average(
            self._collect_probas(X), axis=0, weights=self._weights_not_none
        )
        return avg

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns ndarray of shape (n_classifiers, n_samples *
                n_classes), being class probabilities calculated by each
                classifier.
            If `voting='soft' and `flatten_transform=False`:
                ndarray of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                ndarray of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        check_is_fitted(self)

        if self.voting == "soft":
            probas = self._collect_probas(X)
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)

        else:
            return self._predict(X)


class VotingRegressor(RegressorMixin, _BaseVoting):
    """Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base
    regressors, each on the whole dataset. Then it averages the individual
    predictions to form a final prediction.

    Read more in the :ref:`User Guide <voting_regressor>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        ``set_params``.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying regressor exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0

    See Also
    --------
    VotingClassifier : Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2)])
    >>> print(er.fit(X, y).predict(X))
    [ 3.3  5.7 11.8 19.7 28.  40.3]
    """

    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        super().__init__(estimators=estimators)
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        y = column_or_1d(y, warn=True)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1, weights=self._weights_not_none)

    def transform(self, X):
        """Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_classifiers)
            Values predicted by each regressor.
        """
        check_is_fitted(self)
        return self._predict(X)
