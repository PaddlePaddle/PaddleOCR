# Authors: Manoj Kumar mks542@nyu.edu
# License: BSD 3 clause

import numpy as np

from scipy import optimize

from ..base import BaseEstimator, RegressorMixin
from ._base import LinearModel
from ..utils import axis0_safe_slice
from ..utils.validation import _check_sample_weight
from ..utils.extmath import safe_sparse_dot
from ..utils.optimize import _check_optimize_result


def _huber_loss_and_gradient(w, X, y, epsilon, alpha, sample_weight=None):
    """Returns the Huber loss and the gradient.

    Parameters
    ----------
    w : ndarray, shape (n_features + 1,) or (n_features + 2,)
        Feature vector.
        w[:n_features] gives the coefficients
        w[-1] gives the scale factor and if the intercept is fit w[-2]
        gives the intercept factor.

    X : ndarray of shape (n_samples, n_features)
        Input data.

    y : ndarray of shape (n_samples,)
        Target vector.

    epsilon : float
        Robustness of the Huber estimator.

    alpha : float
        Regularization parameter.

    sample_weight : ndarray of shape (n_samples,), default=None
        Weight assigned to each sample.

    Returns
    -------
    loss : float
        Huber loss.

    gradient : ndarray, shape (len(w))
        Returns the derivative of the Huber loss with respect to each
        coefficient, intercept and the scale as a vector.
    """
    _, n_features = X.shape
    fit_intercept = n_features + 2 == w.shape[0]
    if fit_intercept:
        intercept = w[-2]
    sigma = w[-1]
    w = w[:n_features]
    n_samples = np.sum(sample_weight)

    # Calculate the values where |y - X'w -c / sigma| > epsilon
    # The values above this threshold are outliers.
    linear_loss = y - safe_sparse_dot(X, w)
    if fit_intercept:
        linear_loss -= intercept
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > epsilon * sigma

    # Calculate the linear loss due to the outliers.
    # This is equal to (2 * M * |y - X'w -c / sigma| - M**2) * sigma
    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = X.shape[0] - num_outliers

    # n_sq_outliers includes the weight give to the outliers while
    # num_outliers is just the number of outliers.
    outliers_sw = sample_weight[outliers_mask]
    n_sw_outliers = np.sum(outliers_sw)
    outlier_loss = (
        2.0 * epsilon * np.sum(outliers_sw * outliers)
        - sigma * n_sw_outliers * epsilon ** 2
    )

    # Calculate the quadratic loss due to the non-outliers.-
    # This is equal to |(y - X'w - c)**2 / sigma**2| * sigma
    non_outliers = linear_loss[~outliers_mask]
    weighted_non_outliers = sample_weight[~outliers_mask] * non_outliers
    weighted_loss = np.dot(weighted_non_outliers.T, non_outliers)
    squared_loss = weighted_loss / sigma

    if fit_intercept:
        grad = np.zeros(n_features + 2)
    else:
        grad = np.zeros(n_features + 1)

    # Gradient due to the squared loss.
    X_non_outliers = -axis0_safe_slice(X, ~outliers_mask, n_non_outliers)
    grad[:n_features] = (
        2.0 / sigma * safe_sparse_dot(weighted_non_outliers, X_non_outliers)
    )

    # Gradient due to the linear loss.
    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0
    X_outliers = axis0_safe_slice(X, outliers_mask, num_outliers)
    sw_outliers = sample_weight[outliers_mask] * signed_outliers
    grad[:n_features] -= 2.0 * epsilon * (safe_sparse_dot(sw_outliers, X_outliers))

    # Gradient due to the penalty.
    grad[:n_features] += alpha * 2.0 * w

    # Gradient due to sigma.
    grad[-1] = n_samples
    grad[-1] -= n_sw_outliers * epsilon ** 2
    grad[-1] -= squared_loss / sigma

    # Gradient due to the intercept.
    if fit_intercept:
        grad[-2] = -2.0 * np.sum(weighted_non_outliers) / sigma
        grad[-2] -= 2.0 * epsilon * np.sum(sw_outliers)

    loss = n_samples * sigma + squared_loss + outlier_loss
    loss += alpha * np.dot(w, w)
    return loss, grad


class HuberRegressor(LinearModel, RegressorMixin, BaseEstimator):
    """Linear regression model that is robust to outliers.

    The Huber Regressor optimizes the squared loss for the samples where
    ``|(y - X'w) / sigma| < epsilon`` and the absolute loss for the samples
    where ``|(y - X'w) / sigma| > epsilon``, where w and sigma are parameters
    to be optimized. The parameter sigma makes sure that if y is scaled up
    or down by a certain factor, one does not need to rescale epsilon to
    achieve the same robustness. Note that this does not take into account
    the fact that the different features of X may be of different scales.

    This makes sure that the loss function is not heavily influenced by the
    outliers while not completely ignoring their effect.

    Read more in the :ref:`User Guide <huber_regression>`

    .. versionadded:: 0.18

    Parameters
    ----------
    epsilon : float, greater than 1.0, default=1.35
        The parameter epsilon controls the number of samples that should be
        classified as outliers. The smaller the epsilon, the more robust it is
        to outliers.

    max_iter : int, default=100
        Maximum number of iterations that
        ``scipy.optimize.minimize(method="L-BFGS-B")`` should run for.

    alpha : float, default=0.0001
        Regularization parameter.

    warm_start : bool, default=False
        This is useful if the stored attributes of a previously used model
        has to be reused. If set to False, then the coefficients will
        be rewritten for every call to fit.
        See :term:`the Glossary <warm_start>`.

    fit_intercept : bool, default=True
        Whether or not to fit the intercept. This can be set to False
        if the data is already centered around the origin.

    tol : float, default=1e-05
        The iteration will stop when
        ``max{|proj g_i | i = 1, ..., n}`` <= ``tol``
        where pg_i is the i-th component of the projected gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Features got by optimizing the Huber loss.

    intercept_ : float
        Bias.

    scale_ : float
        The value by which ``|y - X'w - c|`` is scaled down.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations that
        ``scipy.optimize.minimize(method="L-BFGS-B")`` has run for.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    outliers_ : array, shape (n_samples,)
        A boolean mask which is set to True where the samples are identified
        as outliers.

    See Also
    --------
    RANSACRegressor : RANSAC (RANdom SAmple Consensus) algorithm.
    TheilSenRegressor : Theil-Sen Estimator robust multivariate regression model.
    SGDRegressor : Fitted by minimizing a regularized empirical loss with SGD.

    References
    ----------
    .. [1] Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics
           Concomitant scale estimates, pg 172
    .. [2] Art B. Owen (2006), A robust hybrid of lasso and ridge regression.
           https://statweb.stanford.edu/~owen/reports/hhu.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import HuberRegressor, LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> rng = np.random.RandomState(0)
    >>> X, y, coef = make_regression(
    ...     n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0)
    >>> X[:4] = rng.uniform(10, 20, (4, 2))
    >>> y[:4] = rng.uniform(10, 20, 4)
    >>> huber = HuberRegressor().fit(X, y)
    >>> huber.score(X, y)
    -7.284...
    >>> huber.predict(X[:1,])
    array([806.7200...])
    >>> linear = LinearRegression().fit(X, y)
    >>> print("True coefficients:", coef)
    True coefficients: [20.4923...  34.1698...]
    >>> print("Huber coefficients:", huber.coef_)
    Huber coefficients: [17.7906... 31.0106...]
    >>> print("Linear Regression coefficients:", linear.coef_)
    Linear Regression coefficients: [-1.9221...  7.0226...]
    """

    def __init__(
        self,
        *,
        epsilon=1.35,
        max_iter=100,
        alpha=0.0001,
        warm_start=False,
        fit_intercept=True,
        tol=1e-05,
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,)
            Weight given to each sample.

        Returns
        -------
        self : object
            Fitted `HuberRegressor` estimator.
        """
        X, y = self._validate_data(
            X,
            y,
            copy=False,
            accept_sparse=["csr"],
            y_numeric=True,
            dtype=[np.float64, np.float32],
        )

        sample_weight = _check_sample_weight(sample_weight, X)

        if self.epsilon < 1.0:
            raise ValueError(
                "epsilon should be greater than or equal to 1.0, got %f" % self.epsilon
            )

        if self.warm_start and hasattr(self, "coef_"):
            parameters = np.concatenate((self.coef_, [self.intercept_, self.scale_]))
        else:
            if self.fit_intercept:
                parameters = np.zeros(X.shape[1] + 2)
            else:
                parameters = np.zeros(X.shape[1] + 1)
            # Make sure to initialize the scale parameter to a strictly
            # positive value:
            parameters[-1] = 1

        # Sigma or the scale factor should be non-negative.
        # Setting it to be zero might cause undefined bounds hence we set it
        # to a value close to zero.
        bounds = np.tile([-np.inf, np.inf], (parameters.shape[0], 1))
        bounds[-1][0] = np.finfo(np.float64).eps * 10

        opt_res = optimize.minimize(
            _huber_loss_and_gradient,
            parameters,
            method="L-BFGS-B",
            jac=True,
            args=(X, y, self.epsilon, self.alpha, sample_weight),
            options={"maxiter": self.max_iter, "gtol": self.tol, "iprint": -1},
            bounds=bounds,
        )

        parameters = opt_res.x

        if opt_res.status == 2:
            raise ValueError(
                "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
                % opt_res.message
            )
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.scale_ = parameters[-1]
        if self.fit_intercept:
            self.intercept_ = parameters[-2]
        else:
            self.intercept_ = 0.0
        self.coef_ = parameters[: X.shape[1]]

        residual = np.abs(y - safe_sparse_dot(X, self.coef_) - self.intercept_)
        self.outliers_ = residual > self.scale_ * self.epsilon
        return self
