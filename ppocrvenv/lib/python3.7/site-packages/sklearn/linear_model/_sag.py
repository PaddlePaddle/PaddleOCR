"""Solvers for Ridge and LogisticRegression using SAG algorithm"""

# Authors: Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
#
# License: BSD 3 clause

import warnings

import numpy as np

from ._base import make_dataset
from ._sag_fast import sag32, sag64
from ..exceptions import ConvergenceWarning
from ..utils import check_array
from ..utils.validation import _check_sample_weight
from ..utils.extmath import row_norms


def get_auto_step_size(
    max_squared_sum, alpha_scaled, loss, fit_intercept, n_samples=None, is_saga=False
):
    """Compute automatic step size for SAG solver.

    The step size is set to 1 / (alpha_scaled + L + fit_intercept) where L is
    the max sum of squares for over all samples.

    Parameters
    ----------
    max_squared_sum : float
        Maximum squared sum of X over samples.

    alpha_scaled : float
        Constant that multiplies the regularization term, scaled by
        1. / n_samples, the number of samples.

    loss : {'log', 'squared', 'multinomial'}
        The loss function used in SAG solver.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) will be
        added to the decision function.

    n_samples : int, default=None
        Number of rows in X. Useful if is_saga=True.

    is_saga : bool, default=False
        Whether to return step size for the SAGA algorithm or the SAG
        algorithm.

    Returns
    -------
    step_size : float
        Step size used in SAG solver.

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202
    """
    if loss in ("log", "multinomial"):
        L = 0.25 * (max_squared_sum + int(fit_intercept)) + alpha_scaled
    elif loss == "squared":
        # inverse Lipschitz constant for squared loss
        L = max_squared_sum + int(fit_intercept) + alpha_scaled
    else:
        raise ValueError(
            "Unknown loss function for SAG solver, got %s instead of 'log' or 'squared'"
            % loss
        )
    if is_saga:
        # SAGA theoretical step size is 1/3L or 1 / (2 * (L + mu n))
        # See Defazio et al. 2014
        mun = min(2 * n_samples * alpha_scaled, L)
        step = 1.0 / (2 * L + mun)
    else:
        # SAG theoretical step size is 1/16L but it is recommended to use 1 / L
        # see http://www.birs.ca//workshops//2014/14w5003/files/schmidt.pdf,
        # slide 65
        step = 1.0 / L
    return step


def sag_solver(
    X,
    y,
    sample_weight=None,
    loss="log",
    alpha=1.0,
    beta=0.0,
    max_iter=1000,
    tol=0.001,
    verbose=0,
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    warm_start_mem=None,
    is_saga=False,
):
    """SAG solver for Ridge and LogisticRegression.

    SAG stands for Stochastic Average Gradient: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a constant learning rate.

    IMPORTANT NOTE: 'sag' solver converges faster on columns that are on the
    same scale. You can normalize the data by using
    sklearn.preprocessing.StandardScaler on your data before passing it to the
    fit method.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values for the features. It will
    fit the data according to squared loss or log loss.

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using the squared euclidean norm L2.

    .. versionadded:: 0.17

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Target values. With loss='multinomial', y must be label encoded
        (see preprocessing.LabelEncoder).

    sample_weight : array-like of shape (n_samples,), default=None
        Weights applied to individual samples (1. for unweighted).

    loss : {'log', 'squared', 'multinomial'}, default='log'
        Loss function that will be optimized:
        -'log' is the binary logistic loss, as used in LogisticRegression.
        -'squared' is the squared loss, as used in Ridge.
        -'multinomial' is the multinomial logistic loss, as used in
         LogisticRegression.

        .. versionadded:: 0.18
           *loss='multinomial'*

    alpha : float, default=1.
        L2 regularization term in the objective function
        ``(0.5 * alpha * || W ||_F^2)``.

    beta : float, default=0.
        L1 regularization term in the objective function
        ``(beta * || W ||_1)``. Only applied if ``is_saga`` is set to True.

    max_iter : int, default=1000
        The max number of passes over the training data if the stopping
        criteria is not reached.

    tol : float, default=0.001
        The stopping criteria for the weights. The iterations will stop when
        max(change in weights) / max(weights) < tol.

    verbose : int, default=0
        The verbosity level.

    random_state : int, RandomState instance or None, default=None
        Used when shuffling the data. Pass an int for reproducible output
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    check_input : bool, default=True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default=None
        Maximum squared sum of X over samples. If None, it will be computed,
        going through all the samples. The value should be precomputed
        to speed up cross validation.

    warm_start_mem : dict, default=None
        The initialization parameters used for warm starting. Warm starting is
        currently used in LogisticRegression but not in Ridge.
        It contains:
            - 'coef': the weight vector, with the intercept in last line
                if the intercept is fitted.
            - 'gradient_memory': the scalar gradient for all seen samples.
            - 'sum_gradient': the sum of gradient over all seen samples,
                for each feature.
            - 'intercept_sum_gradient': the sum of gradient over all seen
                samples, for the intercept.
            - 'seen': array of boolean describing the seen samples.
            - 'num_seen': the number of seen samples.

    is_saga : bool, default=False
        Whether to use the SAGA algorithm or the SAG algorithm. SAGA behaves
        better in the first epochs, and allow for l1 regularisation.

    Returns
    -------
    coef_ : ndarray of shape (n_features,)
        Weight vector.

    n_iter_ : int
        The number of full pass on all samples.

    warm_start_mem : dict
        Contains a 'coef' key with the fitted result, and possibly the
        fitted intercept at the end of the array. Contains also other keys
        used for warm starting.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(n_samples, n_features)
    >>> y = rng.randn(n_samples)
    >>> clf = linear_model.Ridge(solver='sag')
    >>> clf.fit(X, y)
    Ridge(solver='sag')

    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.LogisticRegression(
    ...     solver='sag', multi_class='multinomial')
    >>> clf.fit(X, y)
    LogisticRegression(multi_class='multinomial', solver='sag')

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202

    See Also
    --------
    Ridge, SGDRegressor, ElasticNet, Lasso, SVR,
    LogisticRegression, SGDClassifier, LinearSVC, Perceptron
    """
    if warm_start_mem is None:
        warm_start_mem = {}
    # Ridge default max_iter is None
    if max_iter is None:
        max_iter = 1000

    if check_input:
        _dtype = [np.float64, np.float32]
        X = check_array(X, dtype=_dtype, accept_sparse="csr", order="C")
        y = check_array(y, dtype=_dtype, ensure_2d=False, order="C")

    n_samples, n_features = X.shape[0], X.shape[1]
    # As in SGD, the alpha is scaled by n_samples.
    alpha_scaled = float(alpha) / n_samples
    beta_scaled = float(beta) / n_samples

    # if loss == 'multinomial', y should be label encoded.
    n_classes = int(y.max()) + 1 if loss == "multinomial" else 1

    # initialization
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    if "coef" in warm_start_mem.keys():
        coef_init = warm_start_mem["coef"]
    else:
        # assume fit_intercept is False
        coef_init = np.zeros((n_features, n_classes), dtype=X.dtype, order="C")

    # coef_init contains possibly the intercept_init at the end.
    # Note that Ridge centers the data before fitting, so fit_intercept=False.
    fit_intercept = coef_init.shape[0] == (n_features + 1)
    if fit_intercept:
        intercept_init = coef_init[-1, :]
        coef_init = coef_init[:-1, :]
    else:
        intercept_init = np.zeros(n_classes, dtype=X.dtype)

    if "intercept_sum_gradient" in warm_start_mem.keys():
        intercept_sum_gradient = warm_start_mem["intercept_sum_gradient"]
    else:
        intercept_sum_gradient = np.zeros(n_classes, dtype=X.dtype)

    if "gradient_memory" in warm_start_mem.keys():
        gradient_memory_init = warm_start_mem["gradient_memory"]
    else:
        gradient_memory_init = np.zeros(
            (n_samples, n_classes), dtype=X.dtype, order="C"
        )
    if "sum_gradient" in warm_start_mem.keys():
        sum_gradient_init = warm_start_mem["sum_gradient"]
    else:
        sum_gradient_init = np.zeros((n_features, n_classes), dtype=X.dtype, order="C")

    if "seen" in warm_start_mem.keys():
        seen_init = warm_start_mem["seen"]
    else:
        seen_init = np.zeros(n_samples, dtype=np.int32, order="C")

    if "num_seen" in warm_start_mem.keys():
        num_seen_init = warm_start_mem["num_seen"]
    else:
        num_seen_init = 0

    dataset, intercept_decay = make_dataset(X, y, sample_weight, random_state)

    if max_squared_sum is None:
        max_squared_sum = row_norms(X, squared=True).max()
    step_size = get_auto_step_size(
        max_squared_sum,
        alpha_scaled,
        loss,
        fit_intercept,
        n_samples=n_samples,
        is_saga=is_saga,
    )
    if step_size * alpha_scaled == 1:
        raise ZeroDivisionError(
            "Current sag implementation does not handle "
            "the case step_size * alpha_scaled == 1"
        )

    sag = sag64 if X.dtype == np.float64 else sag32
    num_seen, n_iter_ = sag(
        dataset,
        coef_init,
        intercept_init,
        n_samples,
        n_features,
        n_classes,
        tol,
        max_iter,
        loss,
        step_size,
        alpha_scaled,
        beta_scaled,
        sum_gradient_init,
        gradient_memory_init,
        seen_init,
        num_seen_init,
        fit_intercept,
        intercept_sum_gradient,
        intercept_decay,
        is_saga,
        verbose,
    )

    if n_iter_ == max_iter:
        warnings.warn(
            "The max_iter was reached which means the coef_ did not converge",
            ConvergenceWarning,
        )

    if fit_intercept:
        coef_init = np.vstack((coef_init, intercept_init))

    warm_start_mem = {
        "coef": coef_init,
        "sum_gradient": sum_gradient_init,
        "intercept_sum_gradient": intercept_sum_gradient,
        "gradient_memory": gradient_memory_init,
        "seen": seen_init,
        "num_seen": num_seen,
    }

    if loss == "multinomial":
        coef_ = coef_init.T
    else:
        coef_ = coef_init[:, 0]

    return coef_, n_iter_, warm_start_mem
