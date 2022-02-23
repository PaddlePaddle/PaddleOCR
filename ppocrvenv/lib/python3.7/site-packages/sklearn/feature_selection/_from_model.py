# Authors: Gilles Louppe, Mathieu Blondel, Maheshakya Wijewardena
# License: BSD 3 clause

import numpy as np
import numbers

from ._base import SelectorMixin
from ._base import _get_feature_importances
from ..base import BaseEstimator, clone, MetaEstimatorMixin
from ..utils._tags import _safe_tags
from ..utils.validation import check_is_fitted

from ..exceptions import NotFittedError
from ..utils.metaestimators import if_delegate_has_method


def _calculate_threshold(estimator, importances, threshold):
    """Interpret the threshold value"""

    if threshold is None:
        # determine default from estimator
        est_name = estimator.__class__.__name__
        if (
            hasattr(estimator, "penalty") and estimator.penalty == "l1"
        ) or "Lasso" in est_name:
            # the natural default threshold is 0 when l1 penalty was used
            threshold = 1e-5
        else:
            threshold = "mean"

    if isinstance(threshold, str):
        if "*" in threshold:
            scale, reference = threshold.split("*")
            scale = float(scale.strip())
            reference = reference.strip()

            if reference == "median":
                reference = np.median(importances)
            elif reference == "mean":
                reference = np.mean(importances)
            else:
                raise ValueError("Unknown reference: " + reference)

            threshold = scale * reference

        elif threshold == "median":
            threshold = np.median(importances)

        elif threshold == "mean":
            threshold = np.mean(importances)

        else:
            raise ValueError(
                "Expected threshold='mean' or threshold='median' got %s" % threshold
            )

    else:
        threshold = float(threshold)

    return threshold


class SelectFromModel(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """Meta-transformer for selecting features based on importance weights.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <select_from_model>`.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``threshold`` value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``transform`` must be called directly
        and SelectFromModel cannot be used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator.
        Otherwise train the model using ``fit`` and then ``transform`` to do
        feature selection.

    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int, default=None
        The maximum number of features to select.
        To only select based on ``max_features``, set ``threshold=-np.inf``.

        .. versionadded:: 0.20

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a ``coef_``
        attribute or ``feature_importances_`` attribute of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    estimator_ : an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``SelectFromModel``, i.e when prefit is False.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    threshold_ : float
        The threshold value used for feature selection.

    See Also
    --------
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    Examples
    --------
    >>> from sklearn.feature_selection import SelectFromModel
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y = [0, 1, 0, 1]
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.55245...
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform(X)
    array([[-1.34],
           [-0.02],
           [-0.48],
           [ 1.48]])
    """

    def __init__(
        self,
        estimator,
        *,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.importance_getter = importance_getter
        self.norm_order = norm_order
        self.max_features = max_features

    def _get_support_mask(self):
        # SelectFromModel can directly call on transform.
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, "estimator_"):
            estimator = self.estimator_
        else:
            raise ValueError(
                "Either fit the model before transform or set"
                ' "prefit=True" while passing the fitted'
                " estimator to the constructor."
            )
        scores = _get_feature_importances(
            estimator=estimator,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        threshold = _calculate_threshold(estimator, scores, self.threshold)
        if self.max_features is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind="mergesort")[
                : self.max_features
            ]
            mask[candidate_indices] = True
        else:
            mask = np.ones_like(scores, dtype=bool)
        mask[scores < threshold] = False
        return mask

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : dict
            Other estimator specific parameters.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError(
                    "'max_features' should be an integer between"
                    " 0 and {} features. Got {!r} instead.".format(
                        X.shape[1], self.max_features
                    )
                )
            elif self.max_features < 0 or self.max_features > X.shape[1]:
                raise ValueError(
                    "'max_features' should be 0 and {} features.Got {} instead.".format(
                        X.shape[1], self.max_features
                    )
                )

        if self.prefit:
            raise NotFittedError("Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)

        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        else:
            self._check_feature_names(X, reset=True)

        return self

    @property
    def threshold_(self):
        """Threshold value used for feature selection."""
        scores = _get_feature_importances(
            estimator=self.estimator_,
            getter=self.importance_getter,
            transform_func="norm",
            norm_order=self.norm_order,
        )
        return _calculate_threshold(self.estimator, scores, self.threshold)

    @if_delegate_has_method("estimator")
    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : dict
            Other estimator specific parameters.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.prefit:
            raise NotFittedError("Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self

    @property
    def n_features_in_(self):
        """Number of features seen during `fit`."""
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

        return self.estimator_.n_features_in_

    def _more_tags(self):
        return {"allow_nan": _safe_tags(self.estimator, key="allow_nan")}
