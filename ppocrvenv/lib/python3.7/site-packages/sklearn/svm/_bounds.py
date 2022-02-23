"""Determination of parameter bounds"""
# Author: Paolo Losi
# License: BSD 3 clause

import numpy as np

from ..preprocessing import LabelBinarizer
from ..utils.validation import check_consistent_length, check_array
from ..utils.extmath import safe_sparse_dot


def l1_min_c(X, y, *, loss="squared_hinge", fit_intercept=True, intercept_scaling=1.0):
    """
    Return the lowest bound for C such that for C in (l1_min_C, infinity)
    the model is guaranteed not to be empty. This applies to l1 penalized
    classifiers, such as LinearSVC with penalty='l1' and
    linear_model.LogisticRegression with penalty='l1'.

    This value is valid if class_weight parameter in fit() is not set.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X.

    loss : {'squared_hinge', 'log'}, default='squared_hinge'
        Specifies the loss function.
        With 'squared_hinge' it is the squared hinge loss (a.k.a. L2 loss).
        With 'log' it is the loss of logistic regression models.

    fit_intercept : bool, default=True
        Specifies if the intercept should be fitted by the model.
        It must match the fit() method parameter.

    intercept_scaling : float, default=1.0
        when fit_intercept is True, instance vector x becomes
        [x, intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        It must match the fit() method parameter.

    Returns
    -------
    l1_min_c : float
        minimum value for C
    """
    if loss not in ("squared_hinge", "log"):
        raise ValueError('loss type not in ("squared_hinge", "log")')

    X = check_array(X, accept_sparse="csc")
    check_consistent_length(X, y)

    Y = LabelBinarizer(neg_label=-1).fit_transform(y).T
    # maximum absolute value over classes and features
    den = np.max(np.abs(safe_sparse_dot(Y, X)))
    if fit_intercept:
        bias = np.full(
            (np.size(y), 1), intercept_scaling, dtype=np.array(intercept_scaling).dtype
        )
        den = max(den, abs(np.dot(Y, bias)).max())

    if den == 0.0:
        raise ValueError(
            "Ill-posed l1_min_c calculation: l1 will always "
            "select zero coefficients for this data"
        )
    if loss == "squared_hinge":
        return 0.5 / den
    else:  # loss == 'log':
        return 2.0 / den
