"""
Logistic Regression
"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Fabian Pedregosa <f@bianp.net>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#         Lars Buitinck
#         Simon Wu <s8wu@uwaterloo.ca>
#         Arthur Mensch <arthur.mensch@m4x.org

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit, logsumexp
from joblib import Parallel, effective_n_jobs

from ._base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from ._sag import sag_solver
from ..preprocessing import LabelEncoder, LabelBinarizer
from ..svm._base import _fit_liblinear
from ..utils import check_array, check_consistent_length, compute_class_weight
from ..utils import check_random_state
from ..utils.extmath import log_logistic, safe_sparse_dot, softmax, squared_norm
from ..utils.extmath import row_norms
from ..utils.optimize import _newton_cg, _check_optimize_result
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils.multiclass import check_classification_targets
from ..utils.fixes import _joblib_parallel_args
from ..utils.fixes import delayed
from ..model_selection import check_cv
from ..metrics import get_scorer


_LOGISTIC_SOLVER_CONVERGENCE_MSG = (
    "Please also refer to the documentation for alternative solver options:\n"
    "    https://scikit-learn.org/stable/modules/linear_model.html"
    "#logistic-regression"
)


# .. some helper functions for logistic_regression_path ..
def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).

    It takes into consideration if the intercept should be fit or not.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    Returns
    -------
    w : ndarray of shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.

    c : float
        The intercept.

    yz : float
        y * np.dot(X, w).
    """
    c = 0.0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    yz = y * z
    return w, c, yz


def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss and gradient.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    out : float
        Logistic loss.

    grad : ndarray of shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + 0.5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()
    return out, grad


def _logistic_loss(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,) default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    out : float
        Logistic loss.
    """
    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + 0.5 * alpha * np.dot(w, w)
    return out


def _logistic_grad_hess(w, X, y, alpha, sample_weight=None):
    """Computes the gradient and the Hessian, in the case of a logistic loss.

    Parameters
    ----------
    w : ndarray of shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Array of labels.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,) default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    grad : ndarray of shape (n_features,) or (n_features + 1,)
        Logistic gradient.

    Hs : callable
        Function that takes the gradient as a parameter and returns the
        matrix product of the Hessian and gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)
    fit_intercept = grad.shape[0] > n_features

    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = safe_sparse_dot(X.T, z0) + alpha * w

    # Case where we fit the intercept.
    if fit_intercept:
        grad[-1] = z0.sum()

    # The mat-vec product of the Hessian
    d = sample_weight * z * (1 - z)
    if sparse.issparse(X):
        dX = safe_sparse_dot(sparse.dia_matrix((d, 0), shape=(n_samples, n_samples)), X)
    else:
        # Precompute as much as possible
        dX = d[:, np.newaxis] * X

    if fit_intercept:
        # Calculate the double derivative with respect to intercept
        # In the case of sparse matrices this returns a matrix object.
        dd_intercept = np.squeeze(np.array(dX.sum(axis=0)))

    def Hs(s):
        ret = np.empty_like(s)
        if sparse.issparse(X):
            ret[:n_features] = X.T.dot(dX.dot(s[:n_features]))
        else:
            ret[:n_features] = np.linalg.multi_dot([X.T, dX, s[:n_features]])
        ret[:n_features] += alpha * s[:n_features]

        # For the fit intercept case.
        if fit_intercept:
            ret[:n_features] += s[-1] * dd_intercept
            ret[-1] = dd_intercept.dot(s[:n_features])
            ret[-1] += d.sum() * s[-1]
        return ret

    return grad, Hs


def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray of shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w


def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray of shape (n_classes * n_features,) or \
            (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == n_classes * (n_features + 1)
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)), dtype=X.dtype)
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def _multinomial_grad_hess(w, X, Y, alpha, sample_weight):
    """
    Computes the gradient and the Hessian, in the case of a multinomial loss.

    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.

    Returns
    -------
    grad : ndarray of shape (n_classes * n_features,) or \
            (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    hessp : callable
        Function that takes in a vector input of shape (n_classes * n_features)
        or (n_classes * (n_features + 1)) and returns matrix-vector product
        with hessian.

    References
    ----------
    Barak A. Pearlmutter (1993). Fast Exact Multiplication by the Hessian.
        http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf
    """
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))

    # `loss` is unused. Refactoring to avoid computing it does not
    # significantly speed up the computation and decreases readability
    loss, grad, p = _multinomial_loss_grad(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]

    # Hessian-vector product derived by applying the R-operator on the gradient
    # of the multinomial loss function.
    def hessp(v):
        v = v.reshape(n_classes, -1)
        if fit_intercept:
            inter_terms = v[:, -1]
            v = v[:, :-1]
        else:
            inter_terms = 0
        # r_yhat holds the result of applying the R-operator on the multinomial
        # estimator.
        r_yhat = safe_sparse_dot(X, v.T)
        r_yhat += inter_terms
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        r_yhat *= sample_weight
        hessProd = np.zeros((n_classes, n_features + bool(fit_intercept)))
        hessProd[:, :n_features] = safe_sparse_dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        if fit_intercept:
            hessProd[:, -1] = r_yhat.sum(axis=0)
        return hessProd.ravel()

    return grad, hessp


def _check_solver(solver, penalty, dual):
    all_solvers = ["liblinear", "newton-cg", "lbfgs", "sag", "saga"]
    if solver not in all_solvers:
        raise ValueError(
            "Logistic Regression supports only solvers in %s, got %s."
            % (all_solvers, solver)
        )

    all_penalties = ["l1", "l2", "elasticnet", "none"]
    if penalty not in all_penalties:
        raise ValueError(
            "Logistic Regression supports only penalties in %s, got %s."
            % (all_penalties, penalty)
        )

    if solver not in ["liblinear", "saga"] and penalty not in ("l2", "none"):
        raise ValueError(
            "Solver %s supports only 'l2' or 'none' penalties, got %s penalty."
            % (solver, penalty)
        )
    if solver != "liblinear" and dual:
        raise ValueError(
            "Solver %s supports only dual=False, got dual=%s" % (solver, dual)
        )

    if penalty == "elasticnet" and solver != "saga":
        raise ValueError(
            "Only 'saga' solver supports elasticnet penalty, got solver={}.".format(
                solver
            )
        )

    if solver == "liblinear" and penalty == "none":
        raise ValueError("penalty='none' is not supported for the liblinear solver")

    return solver


def _check_multi_class(multi_class, solver, n_classes):
    if multi_class == "auto":
        if solver == "liblinear":
            multi_class = "ovr"
        elif n_classes > 2:
            multi_class = "multinomial"
        else:
            multi_class = "ovr"
    if multi_class not in ("multinomial", "ovr"):
        raise ValueError(
            "multi_class should be 'multinomial', 'ovr' or 'auto'. Got %s."
            % multi_class
        )
    if multi_class == "multinomial" and solver == "liblinear":
        raise ValueError("Solver %s does not support a multinomial backend." % solver)
    return multi_class


def _logistic_regression_path(
    X,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="lbfgs",
    coef=None,
    class_weight=None,
    dual=False,
    penalty="l2",
    intercept_scaling=1.0,
    multi_class="auto",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.

    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or array-like of shape (n_cs,), default=10
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    fit_intercept : bool, default=True
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int, default=100
        Maximum number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Numerical solver to use.

    coef : array-like of shape (n_features,), default=None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    intercept_scaling : float, default=1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'ovr', 'multinomial', 'auto'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array of shape (n_cs,)
        Actual number of iteration for each Cs.

    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.

    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != "multinomial":
        if classes.size > 2:
            raise ValueError("To fit OvR, use the pos_class argument")
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == "multinomial":
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == "ovr":
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = y == pos_class
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.0
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(
                class_weight, classes=mask_classes, y=y_bin
            )
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver not in ["sag", "saga"]:
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        w0 = np.zeros(
            (classes.size, n_features + int(fit_intercept)), order="F", dtype=X.dtype
        )

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == "ovr":
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    "Initialization coef is of shape %d, expected shape %d or %d"
                    % (coef.size, n_features, w0.size)
                )
            w0[: coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if coef.shape[0] != n_classes or coef.shape[1] not in (
                n_features,
                n_features + 1,
            ):
                raise ValueError(
                    "Initialization coef is of shape (%d, %d), expected "
                    "shape (%d, %d) or (%d, %d)"
                    % (
                        coef.shape[0],
                        coef.shape[1],
                        classes.size,
                        n_features,
                        classes.size,
                        n_features + 1,
                    )
                )

            if n_classes == 1:
                w0[0, : coef.shape[1]] = -coef
                w0[1, : coef.shape[1]] = coef
            else:
                w0[:, : coef.shape[1]] = coef

    if multi_class == "multinomial":
        # scipy.optimize.minimize and newton-cg accepts only
        # ravelled parameters.
        if solver in ["lbfgs", "newton-cg"]:
            w0 = w0.ravel()
        target = Y_multi
        if solver == "lbfgs":

            def func(x, *args):
                return _multinomial_loss_grad(x, *args)[0:2]

        elif solver == "newton-cg":

            def func(x, *args):
                return _multinomial_loss(x, *args)[0]

            def grad(x, *args):
                return _multinomial_loss_grad(x, *args)[1]

            hess = _multinomial_grad_hess
        warm_start_sag = {"coef": w0.T}
    else:
        target = y_bin
        if solver == "lbfgs":
            func = _logistic_loss_and_grad
        elif solver == "newton-cg":
            func = _logistic_loss

            def grad(x, *args):
                return _logistic_loss_and_grad(x, *args)[1]

            hess = _logistic_grad_hess
        warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == "lbfgs":
            iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)
            ]
            opt_res = optimize.minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=(X, target, 1.0 / C, sample_weight),
                options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
            )
            n_iter_i = _check_optimize_result(
                solver,
                opt_res,
                max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
            w0, loss = opt_res.x, opt_res.fun
        elif solver == "newton-cg":
            args = (X, target, 1.0 / C, sample_weight)
            w0, n_iter_i = _newton_cg(
                hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol
            )
        elif solver == "liblinear":
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X,
                target,
                C,
                fit_intercept,
                intercept_scaling,
                None,
                penalty,
                dual,
                verbose,
                max_iter,
                tol,
                random_state,
                sample_weight=sample_weight,
            )
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ["sag", "saga"]:
            if multi_class == "multinomial":
                target = target.astype(X.dtype, copy=False)
                loss = "multinomial"
            else:
                loss = "log"
            # alpha is for L2-norm, beta is for L1-norm
            if penalty == "l1":
                alpha = 0.0
                beta = 1.0 / C
            elif penalty == "l2":
                alpha = 1.0 / C
                beta = 0.0
            else:  # Elastic-Net penalty
                alpha = (1.0 / C) * (1 - l1_ratio)
                beta = (1.0 / C) * l1_ratio

            w0, n_iter_i, warm_start_sag = sag_solver(
                X,
                target,
                sample_weight,
                loss,
                alpha,
                beta,
                max_iter,
                tol,
                verbose,
                random_state,
                False,
                max_squared_sum,
                warm_start_sag,
                is_saga=(solver == "saga"),
            )

        else:
            raise ValueError(
                "solver must be one of {'liblinear', 'lbfgs', "
                "'newton-cg', 'sag'}, got '%s' instead" % solver
            )

        if multi_class == "multinomial":
            n_classes = max(2, classes.size)
            multi_w0 = np.reshape(w0, (n_classes, -1))
            if n_classes == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0.copy())
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter


# helper function for LogisticCV
def _log_reg_scoring_path(
    X,
    y,
    train,
    test,
    pos_class=None,
    Cs=10,
    scoring=None,
    fit_intercept=False,
    max_iter=100,
    tol=1e-4,
    class_weight=None,
    verbose=0,
    solver="lbfgs",
    penalty="l2",
    dual=False,
    intercept_scaling=1.0,
    multi_class="auto",
    random_state=None,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
):
    """Computes scores across logistic_regression_path

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target labels.

    train : list of indices
        The indices of the train set.

    test : list of indices
        The indices of the test set.

    pos_class : int, default=None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    Cs : int or list of floats, default=10
        Each of the values in Cs describes the inverse of
        regularization strength. If Cs is as an int, then a grid of Cs
        values are chosen in a logarithmic scale between 1e-4 and 1e4.
        If not provided, then a fixed set of values for Cs are used.

    scoring : callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`. The
        default scoring option used is accuracy_score.

    fit_intercept : bool, default=False
        If False, then the bias term is set to zero. Else the last
        term of each coef_ gives us the intercept.

    max_iter : int, default=100
        Maximum number of iterations for the solver.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    solver : {'lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'
        Decides which solver to use.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
        only supported by the 'saga' solver.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    intercept_scaling : float, default=1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. Used only in SAG solver.
        If None, it will be computed, going through all the samples.
        The value should be precomputed to speed up cross validation.

    sample_weight : array-like of shape(n_samples,), default=None
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Returns
    -------
    coefs : ndarray of shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Cs : ndarray
        Grid of Cs used for cross-validation.

    scores : ndarray of shape (n_cs,)
        Scores obtained for each Cs.

    n_iter : ndarray of shape(n_cs,)
        Actual number of iteration for each Cs.
    """
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        sample_weight = sample_weight[train]

    coefs, Cs, n_iter = _logistic_regression_path(
        X_train,
        y_train,
        Cs=Cs,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        pos_class=pos_class,
        multi_class=multi_class,
        tol=tol,
        verbose=verbose,
        dual=dual,
        penalty=penalty,
        intercept_scaling=intercept_scaling,
        random_state=random_state,
        check_input=False,
        max_squared_sum=max_squared_sum,
        sample_weight=sample_weight,
    )

    log_reg = LogisticRegression(solver=solver, multi_class=multi_class)

    # The score method of Logistic Regression has a classes_ attribute.
    if multi_class == "ovr":
        log_reg.classes_ = np.array([-1, 1])
    elif multi_class == "multinomial":
        log_reg.classes_ = np.unique(y_train)
    else:
        raise ValueError(
            "multi_class should be either multinomial or ovr, got %d" % multi_class
        )

    if pos_class is not None:
        mask = y_test == pos_class
        y_test = np.ones(y_test.shape, dtype=np.float64)
        y_test[~mask] = -1.0

    scores = list()

    scoring = get_scorer(scoring)
    for w in coefs:
        if multi_class == "ovr":
            w = w[np.newaxis, :]
        if fit_intercept:
            log_reg.coef_ = w[:, :-1]
            log_reg.intercept_ = w[:, -1]
        else:
            log_reg.coef_ = w
            log_reg.intercept_ = 0.0

        if scoring is None:
            scores.append(log_reg.score(X_test, y_test))
        else:
            scores.append(scoring(log_reg, X_test, y_test))

    return coefs, Cs, np.array(scores), n_iter


class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag', 'saga' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
    that regularization is applied by default**. It can handle both dense
    and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted
    (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation, or no regularization. The 'liblinear' solver
    supports both L1 and L2 regularization, with a dual formulation only for
    the L2 penalty. The Elastic-Net regularization is only supported by the
    'saga' solver.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
        Specify the norm of the penalty:

        - `'none'`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.

        .. warning::
           The choice of the algorithm depends on the penalty chosen:
           Supported penalties by solver:

           - 'newton-cg'   -   ['l2', 'none']
           - 'lbfgs'       -   ['l2', 'none']
           - 'liblinear'   -   ['l1', 'l2']
           - 'sag'         -   ['l2', 'none']
           - 'saga'        -   ['elasticnet', 'l1', 'l2', 'none']

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on
           features with approximately the same scale. You can
           preprocess the data with a scaler from :mod:`sklearn.preprocessing`.

        .. seealso::
           Refer to the User Guide for more information regarding
           :class:`LogisticRegression` and more specifically the
           `Table <https://scikit-learn.org/dev/modules/linear_model.html#logistic-regression>`_
           summarazing solver/penalty supports.
           <!--
           # noqa: E501
           -->

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
            The default solver changed from 'liblinear' to 'lbfgs' in 0.22.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html

    LIBLINEAR -- A Library for Large Linear Classification
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
        SAGA: A Fast Incremental Gradient Method With Support
        for Non-Strongly Convex Composite Objectives
        https://arxiv.org/abs/1407.0202

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegression(random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...
    """

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        The SAGA solver supports both float64 and float32 bit arrays.
        """
        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        if self.penalty == "elasticnet":
            if (
                not isinstance(self.l1_ratio, numbers.Number)
                or self.l1_ratio < 0
                or self.l1_ratio > 1
            ):
                raise ValueError(
                    "l1_ratio must be between 0 and 1; got (l1_ratio=%r)"
                    % self.l1_ratio
                )
        elif self.l1_ratio is not None:
            warnings.warn(
                "l1_ratio parameter is only used when penalty is "
                "'elasticnet'. Got "
                "(penalty={})".format(self.penalty)
            )
        if self.penalty == "none":
            if self.C != 1.0:  # default values
                warnings.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = np.inf
            penalty = "l2"
        else:
            C_ = self.C
            penalty = self.penalty
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )

        if solver == "lbfgs":
            _dtype = np.float64
        else:
            _dtype = [np.float64, np.float32]

        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        multi_class = _check_multi_class(self.multi_class, solver, len(self.classes_))

        if solver == "liblinear":
            if effective_n_jobs(self.n_jobs) != 1:
                warnings.warn(
                    "'n_jobs' > 1 does not have any effect when"
                    " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                    " = {}.".format(effective_n_jobs(self.n_jobs))
                )
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X,
                y,
                self.C,
                self.fit_intercept,
                self.intercept_scaling,
                self.class_weight,
                self.penalty,
                self.dual,
                self.verbose,
                self.max_iter,
                self.tol,
                self.random_state,
                sample_weight=sample_weight,
            )
            self.n_iter_ = np.array([n_iter_])
            return self

        if solver in ["sag", "saga"]:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(
                warm_start_coef, self.intercept_[:, np.newaxis], axis=1
            )

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == "multinomial":
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if solver in ["sag", "saga"]:
            prefer = "threads"
        else:
            prefer = "processes"
        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X,
                y,
                pos_class=class_,
                Cs=[C_],
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                verbose=self.verbose,
                solver=solver,
                multi_class=multi_class,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                check_input=False,
                random_state=self.random_state,
                coef=warm_start_coef_,
                penalty=penalty,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
            )
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef)
        )

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        n_features = X.shape[1]
        if multi_class == "multinomial":
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(
                n_classes, n_features + int(self.fit_intercept)
            )

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)

        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)

        ovr = self.multi_class in ["ovr", "warn"] or (
            self.multi_class == "auto"
            and (self.classes_.size <= 2 or self.solver == "liblinear")
        )
        if ovr:
            return super()._predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


class LogisticRegressionCV(LogisticRegression, LinearClassifierMixin, BaseEstimator):
    """Logistic Regression CV (aka logit, MaxEnt) classifier.

    See glossary entry for :term:`cross-validation estimator`.

    This class implements logistic regression using liblinear, newton-cg, sag
    of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2
    regularization with primal formulation. The liblinear solver supports both
    L1 and L2 regularization, with a dual formulation only for the L2 penalty.
    Elastic-Net penalty is only supported by the saga solver.

    For the grid of `Cs` values and `l1_ratios` values, the best hyperparameter
    is selected by the cross-validator
    :class:`~sklearn.model_selection.StratifiedKFold`, but it can be changed
    using the :term:`cv` parameter. The 'newton-cg', 'sag', 'saga' and 'lbfgs'
    solvers can warm-start the coefficients (see :term:`Glossary<warm_start>`).

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    Cs : int or list of floats, default=10
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    cv : int or cross-validation generator, default=None
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, then it is the number of folds used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : {'l1', 'l2', 'elasticnet'}, default='l2'
        Specify the norm of the penalty:

        - `'l2'`: add a L2 penalty term (used by default);
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`. The
        default scoring option used is 'accuracy'.

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
            default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' might be slower in :class:`LogisticRegressionCV`
              because it does not handle warm-starting. 'liblinear' is
              limited to one-versus-rest schemes.

        .. warning::
           The choice of the algorithm depends on the penalty chosen:

           - 'newton-cg'   -   ['l2']
           - 'lbfgs'       -   ['l2']
           - 'liblinear'   -   ['l1', 'l2']
           - 'sag'         -   ['l2']
           - 'saga'        -   ['elasticnet', 'l1', 'l2']

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on features
           with approximately the same scale. You can preprocess the data with
           a scaler from :mod:`sklearn.preprocessing`.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    max_iter : int, default=100
        Maximum number of iterations of the optimization algorithm.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           class_weight == 'balanced'

    n_jobs : int, default=None
        Number of CPU cores used during the cross-validation loop.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any
        positive number for verbosity.

    refit : bool, default=True
        If set to True, the scores are averaged across all folds, and the
        coefs and the C that corresponds to the best score is taken, and a
        final refit is done using these parameters.
        Otherwise the coefs, intercepts and C that correspond to the
        best scores across folds are averaged.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : {'auto, 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    random_state : int, RandomState instance, default=None
        Used when `solver='sag'`, 'saga' or 'liblinear' to shuffle the data.
        Note that this only applies to the solver and not the cross-validation
        generator. See :term:`Glossary <random_state>` for details.

    l1_ratios : list of float, default=None
        The list of Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``.
        Only used if ``penalty='elasticnet'``. A value of 0 is equivalent to
        using ``penalty='l2'``, while 1 is equivalent to using
        ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a combination
        of L1 and L2.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem
        is binary.

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape(1,) when the problem is binary.

    Cs_ : ndarray of shape (n_cs)
        Array of C i.e. inverse of regularization parameter values used
        for cross-validation.

    l1_ratios_ : ndarray of shape (n_l1_ratios)
        Array of l1_ratios used for cross-validation. If no l1_ratio is used
        (i.e. penalty is not 'elasticnet'), this is set to ``[None]``

    coefs_paths_ : ndarray of shape (n_folds, n_cs, n_features) or \
                   (n_folds, n_cs, n_features + 1)
        dict with classes as the keys, and the path of coefficients obtained
        during cross-validating across each fold and then across each Cs
        after doing an OvR for the corresponding class as values.
        If the 'multi_class' option is set to 'multinomial', then
        the coefs_paths are the coefficients corresponding to each class.
        Each dict value has shape ``(n_folds, n_cs, n_features)`` or
        ``(n_folds, n_cs, n_features + 1)`` depending on whether the
        intercept is fit or not. If ``penalty='elasticnet'``, the shape is
        ``(n_folds, n_cs, n_l1_ratios_, n_features)`` or
        ``(n_folds, n_cs, n_l1_ratios_, n_features + 1)``.

    scores_ : dict
        dict with classes as the keys, and the values as the
        grid of scores obtained during cross-validating each fold, after doing
        an OvR for the corresponding class. If the 'multi_class' option
        given is 'multinomial' then the same scores are repeated across
        all classes, since this is the multinomial class. Each dict value
        has shape ``(n_folds, n_cs`` or ``(n_folds, n_cs, n_l1_ratios)`` if
        ``penalty='elasticnet'``.

    C_ : ndarray of shape (n_classes,) or (n_classes - 1,)
        Array of C that maps to the best scores across every class. If refit is
        set to False, then for each class, the best C is the average of the
        C's that correspond to the best scores for each fold.
        `C_` is of shape(n_classes,) when the problem is binary.

    l1_ratio_ : ndarray of shape (n_classes,) or (n_classes - 1,)
        Array of l1_ratio that maps to the best scores across every class. If
        refit is set to False, then for each class, the best l1_ratio is the
        average of the l1_ratio's that correspond to the best scores for each
        fold.  `l1_ratio_` is of shape(n_classes,) when the problem is binary.

    n_iter_ : ndarray of shape (n_classes, n_folds, n_cs) or (1, n_folds, n_cs)
        Actual number of iterations for all classes, folds and Cs.
        In the binary or multinomial cases, the first dimension is equal to 1.
        If ``penalty='elasticnet'``, the shape is ``(n_classes, n_folds,
        n_cs, n_l1_ratios)`` or ``(1, n_folds, n_cs, n_l1_ratios)``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    LogisticRegression : Logistic regression without tuning the
        hyperparameter `C`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegressionCV
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :]).shape
    (2, 3)
    >>> clf.score(X, y)
    0.98...
    """

    def __init__(
        self,
        *,
        Cs=10,
        fit_intercept=True,
        cv=None,
        dual=False,
        penalty="l2",
        scoring=None,
        solver="lbfgs",
        tol=1e-4,
        max_iter=100,
        class_weight=None,
        n_jobs=None,
        verbose=0,
        refit=True,
        intercept_scaling=1.0,
        multi_class="auto",
        random_state=None,
        l1_ratios=None,
    ):
        self.Cs = Cs
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.solver = solver
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        self.random_state = random_state
        self.l1_ratios = l1_ratios

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Fitted LogisticRegressionCV estimator.
        """
        solver = _check_solver(self.solver, self.penalty, self.dual)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )
        if self.penalty == "elasticnet":
            if (
                self.l1_ratios is None
                or len(self.l1_ratios) == 0
                or any(
                    (
                        not isinstance(l1_ratio, numbers.Number)
                        or l1_ratio < 0
                        or l1_ratio > 1
                    )
                    for l1_ratio in self.l1_ratios
                )
            ):
                raise ValueError(
                    "l1_ratios must be a list of numbers between "
                    "0 and 1; got (l1_ratios=%r)"
                    % self.l1_ratios
                )
            l1_ratios_ = self.l1_ratios
        else:
            if self.l1_ratios is not None:
                warnings.warn(
                    "l1_ratios parameter is only used when penalty "
                    "is 'elasticnet'. Got (penalty={})".format(self.penalty)
                )

            l1_ratios_ = [None]

        if self.penalty == "none":
            raise ValueError(
                "penalty='none' is not useful and not supported by "
                "LogisticRegressionCV."
            )

        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)

        class_weight = self.class_weight

        # Encode for string labels
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        if isinstance(class_weight, dict):
            class_weight = {
                label_encoder.transform([cls])[0]: v for cls, v in class_weight.items()
            }

        # The original class labels
        classes = self.classes_ = label_encoder.classes_
        encoded_labels = label_encoder.transform(label_encoder.classes_)

        multi_class = _check_multi_class(self.multi_class, solver, len(classes))

        if solver in ["sag", "saga"]:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        # init cross-validation generator
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y))

        # Use the label encoded classes
        n_classes = len(encoded_labels)

        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes[0]
            )

        if n_classes == 2:
            # OvR in case of binary problems is as good as fitting
            # the higher label
            n_classes = 1
            encoded_labels = encoded_labels[1:]
            classes = classes[1:]

        # We need this hack to iterate only once over labels, in the case of
        # multi_class = multinomial, without changing the value of the labels.
        if multi_class == "multinomial":
            iter_encoded_labels = iter_classes = [None]
        else:
            iter_encoded_labels = encoded_labels
            iter_classes = classes

        # compute the class weights for the entire dataset y
        if class_weight == "balanced":
            class_weight = compute_class_weight(
                class_weight, classes=np.arange(len(self.classes_)), y=y
            )
            class_weight = dict(enumerate(class_weight))

        path_func = delayed(_log_reg_scoring_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if self.solver in ["sag", "saga"]:
            prefer = "threads"
        else:
            prefer = "processes"

        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X,
                y,
                train,
                test,
                pos_class=label,
                Cs=self.Cs,
                fit_intercept=self.fit_intercept,
                penalty=self.penalty,
                dual=self.dual,
                solver=solver,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose,
                class_weight=class_weight,
                scoring=self.scoring,
                multi_class=multi_class,
                intercept_scaling=self.intercept_scaling,
                random_state=self.random_state,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
                l1_ratio=l1_ratio,
            )
            for label in iter_encoded_labels
            for train, test in folds
            for l1_ratio in l1_ratios_
        )

        # _log_reg_scoring_path will output different shapes depending on the
        # multi_class param, so we need to reshape the outputs accordingly.
        # Cs is of shape (n_classes . n_folds . n_l1_ratios, n_Cs) and all the
        # rows are equal, so we just take the first one.
        # After reshaping,
        # - scores is of shape (n_classes, n_folds, n_Cs . n_l1_ratios)
        # - coefs_paths is of shape
        #  (n_classes, n_folds, n_Cs . n_l1_ratios, n_features)
        # - n_iter is of shape
        #  (n_classes, n_folds, n_Cs . n_l1_ratios) or
        #  (1, n_folds, n_Cs . n_l1_ratios)
        coefs_paths, Cs, scores, n_iter_ = zip(*fold_coefs_)
        self.Cs_ = Cs[0]
        if multi_class == "multinomial":
            coefs_paths = np.reshape(
                coefs_paths,
                (len(folds), len(l1_ratios_) * len(self.Cs_), n_classes, -1),
            )
            # equiv to coefs_paths = np.moveaxis(coefs_paths, (0, 1, 2, 3),
            #                                                 (1, 2, 0, 3))
            coefs_paths = np.swapaxes(coefs_paths, 0, 1)
            coefs_paths = np.swapaxes(coefs_paths, 0, 2)
            self.n_iter_ = np.reshape(
                n_iter_, (1, len(folds), len(self.Cs_) * len(l1_ratios_))
            )
            # repeat same scores across all classes
            scores = np.tile(scores, (n_classes, 1, 1))
        else:
            coefs_paths = np.reshape(
                coefs_paths,
                (n_classes, len(folds), len(self.Cs_) * len(l1_ratios_), -1),
            )
            self.n_iter_ = np.reshape(
                n_iter_, (n_classes, len(folds), len(self.Cs_) * len(l1_ratios_))
            )
        scores = np.reshape(scores, (n_classes, len(folds), -1))
        self.scores_ = dict(zip(classes, scores))
        self.coefs_paths_ = dict(zip(classes, coefs_paths))

        self.C_ = list()
        self.l1_ratio_ = list()
        self.coef_ = np.empty((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        for index, (cls, encoded_label) in enumerate(
            zip(iter_classes, iter_encoded_labels)
        ):

            if multi_class == "ovr":
                scores = self.scores_[cls]
                coefs_paths = self.coefs_paths_[cls]
            else:
                # For multinomial, all scores are the same across classes
                scores = scores[0]
                # coefs_paths will keep its original shape because
                # logistic_regression_path expects it this way

            if self.refit:
                # best_index is between 0 and (n_Cs . n_l1_ratios - 1)
                # for example, with n_cs=2 and n_l1_ratios=3
                # the layout of scores is
                # [c1, c2, c1, c2, c1, c2]
                #   l1_1 ,  l1_2 ,  l1_3
                best_index = scores.sum(axis=0).argmax()

                best_index_C = best_index % len(self.Cs_)
                C_ = self.Cs_[best_index_C]
                self.C_.append(C_)

                best_index_l1 = best_index // len(self.Cs_)
                l1_ratio_ = l1_ratios_[best_index_l1]
                self.l1_ratio_.append(l1_ratio_)

                if multi_class == "multinomial":
                    coef_init = np.mean(coefs_paths[:, :, best_index, :], axis=1)
                else:
                    coef_init = np.mean(coefs_paths[:, best_index, :], axis=0)

                # Note that y is label encoded and hence pos_class must be
                # the encoded label / None (for 'multinomial')
                w, _, _ = _logistic_regression_path(
                    X,
                    y,
                    pos_class=encoded_label,
                    Cs=[C_],
                    solver=solver,
                    fit_intercept=self.fit_intercept,
                    coef=coef_init,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    penalty=self.penalty,
                    class_weight=class_weight,
                    multi_class=multi_class,
                    verbose=max(0, self.verbose - 1),
                    random_state=self.random_state,
                    check_input=False,
                    max_squared_sum=max_squared_sum,
                    sample_weight=sample_weight,
                    l1_ratio=l1_ratio_,
                )
                w = w[0]

            else:
                # Take the best scores across every fold and the average of
                # all coefficients corresponding to the best scores.
                best_indices = np.argmax(scores, axis=1)
                if multi_class == "ovr":
                    w = np.mean(
                        [coefs_paths[i, best_indices[i], :] for i in range(len(folds))],
                        axis=0,
                    )
                else:
                    w = np.mean(
                        [
                            coefs_paths[:, i, best_indices[i], :]
                            for i in range(len(folds))
                        ],
                        axis=0,
                    )

                best_indices_C = best_indices % len(self.Cs_)
                self.C_.append(np.mean(self.Cs_[best_indices_C]))

                if self.penalty == "elasticnet":
                    best_indices_l1 = best_indices // len(self.Cs_)
                    self.l1_ratio_.append(np.mean(l1_ratios_[best_indices_l1]))
                else:
                    self.l1_ratio_.append(None)

            if multi_class == "multinomial":
                self.C_ = np.tile(self.C_, n_classes)
                self.l1_ratio_ = np.tile(self.l1_ratio_, n_classes)
                self.coef_ = w[:, : X.shape[1]]
                if self.fit_intercept:
                    self.intercept_ = w[:, -1]
            else:
                self.coef_[index] = w[: X.shape[1]]
                if self.fit_intercept:
                    self.intercept_[index] = w[-1]

        self.C_ = np.asarray(self.C_)
        self.l1_ratio_ = np.asarray(self.l1_ratio_)
        self.l1_ratios_ = np.asarray(l1_ratios_)
        # if elasticnet was used, add the l1_ratios dimension to some
        # attributes
        if self.l1_ratios is not None:
            # with n_cs=2 and n_l1_ratios=3
            # the layout of scores is
            # [c1, c2, c1, c2, c1, c2]
            #   l1_1 ,  l1_2 ,  l1_3
            # To get a 2d array with the following layout
            #      l1_1, l1_2, l1_3
            # c1 [[ .  ,  .  ,  .  ],
            # c2  [ .  ,  .  ,  .  ]]
            # We need to first reshape and then transpose.
            # The same goes for the other arrays
            for cls, coefs_path in self.coefs_paths_.items():
                self.coefs_paths_[cls] = coefs_path.reshape(
                    (len(folds), self.l1_ratios_.size, self.Cs_.size, -1)
                )
                self.coefs_paths_[cls] = np.transpose(
                    self.coefs_paths_[cls], (0, 2, 1, 3)
                )
            for cls, score in self.scores_.items():
                self.scores_[cls] = score.reshape(
                    (len(folds), self.l1_ratios_.size, self.Cs_.size)
                )
                self.scores_[cls] = np.transpose(self.scores_[cls], (0, 2, 1))

            self.n_iter_ = self.n_iter_.reshape(
                (-1, len(folds), self.l1_ratios_.size, self.Cs_.size)
            )
            self.n_iter_ = np.transpose(self.n_iter_, (0, 1, 3, 2))

        return self

    def score(self, X, y, sample_weight=None):
        """Score using the `scoring` option on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        scoring = self.scoring or "accuracy"
        scoring = get_scorer(scoring)

        return scoring(self, X, y, sample_weight=sample_weight)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }
