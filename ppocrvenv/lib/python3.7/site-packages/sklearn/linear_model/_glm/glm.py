"""
Generalized Linear Models with Exponential Dispersion Family
"""

# Author: Christian Lorentzen <lorentzen.ch@googlemail.com>
# some parts and tricks stolen from other sklearn files.
# License: BSD 3 clause

import numbers

import numpy as np
import scipy.optimize

from ...base import BaseEstimator, RegressorMixin
from ...utils.optimize import _check_optimize_result
from ...utils.validation import check_is_fitted, _check_sample_weight
from ..._loss.glm_distribution import (
    ExponentialDispersionModel,
    TweedieDistribution,
    EDM_DISTRIBUTIONS,
)
from .link import (
    BaseLink,
    IdentityLink,
    LogLink,
)


def _safe_lin_pred(X, coef):
    """Compute the linear predictor taking care if intercept is present."""
    if coef.size == X.shape[1] + 1:
        return X @ coef[1:] + coef[0]
    else:
        return X @ coef


def _y_pred_deviance_derivative(coef, X, y, weights, family, link):
    """Compute y_pred and the derivative of the deviance w.r.t coef."""
    lin_pred = _safe_lin_pred(X, coef)
    y_pred = link.inverse(lin_pred)
    d1 = link.inverse_derivative(lin_pred)
    temp = d1 * family.deviance_derivative(y, y_pred, weights)
    if coef.size == X.shape[1] + 1:
        devp = np.concatenate(([temp.sum()], temp @ X))
    else:
        devp = temp @ X  # same as X.T @ temp
    return y_pred, devp


class GeneralizedLinearRegressor(RegressorMixin, BaseEstimator):
    """Regression via a penalized Generalized Linear Model (GLM).

    GLMs based on a reproductive Exponential Dispersion Model (EDM) aim at
    fitting and predicting the mean of the target y as y_pred=h(X*w).
    Therefore, the fit minimizes the following objective function with L2
    priors as regularizer::

            1/(2*sum(s)) * deviance(y, h(X*w); s)
            + 1/2 * alpha * |w|_2

    with inverse link function h and s=sample_weight.
    The parameter ``alpha`` corresponds to the lambda parameter in glmnet.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    family : {'normal', 'poisson', 'gamma', 'inverse-gaussian'} \
            or an ExponentialDispersionModel instance, default='normal'
        The distributional assumption of the GLM, i.e. which distribution from
        the EDM, specifies the loss function to be minimized.

    link : {'auto', 'identity', 'log'} or an instance of class BaseLink, \
            default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:

        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions

    solver : 'lbfgs', default='lbfgs'
        Algorithm to use in the optimization problem:

        'lbfgs'
            Calls scipy's L-BFGS-B optimizer.

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_``.

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        family="normal",
        link="auto",
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.family = family
        self.link = link
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a Generalized Linear Model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted model.
        """
        if isinstance(self.family, ExponentialDispersionModel):
            self._family_instance = self.family
        elif self.family in EDM_DISTRIBUTIONS:
            self._family_instance = EDM_DISTRIBUTIONS[self.family]()
        else:
            raise ValueError(
                "The family must be an instance of class"
                " ExponentialDispersionModel or an element of"
                " ['normal', 'poisson', 'gamma', 'inverse-gaussian']"
                "; got (family={0})".format(self.family)
            )

        # Guarantee that self._link_instance is set to an instance of
        # class BaseLink
        if isinstance(self.link, BaseLink):
            self._link_instance = self.link
        else:
            if self.link == "auto":
                if isinstance(self._family_instance, TweedieDistribution):
                    if self._family_instance.power <= 0:
                        self._link_instance = IdentityLink()
                    if self._family_instance.power >= 1:
                        self._link_instance = LogLink()
                else:
                    raise ValueError(
                        "No default link known for the "
                        "specified distribution family. Please "
                        "set link manually, i.e. not to 'auto'; "
                        "got (link='auto', family={})".format(self.family)
                    )
            elif self.link == "identity":
                self._link_instance = IdentityLink()
            elif self.link == "log":
                self._link_instance = LogLink()
            else:
                raise ValueError(
                    "The link must be an instance of class Link or "
                    "an element of ['auto', 'identity', 'log']; "
                    "got (link={0})".format(self.link)
                )

        if not isinstance(self.alpha, numbers.Number) or self.alpha < 0:
            raise ValueError(
                "Penalty term must be a non-negative number; got (alpha={0})".format(
                    self.alpha
                )
            )
        if not isinstance(self.fit_intercept, bool):
            raise ValueError(
                "The argument fit_intercept must be bool; got {0}".format(
                    self.fit_intercept
                )
            )
        if self.solver not in ["lbfgs"]:
            raise ValueError(
                "GeneralizedLinearRegressor supports only solvers"
                "'lbfgs'; got {0}".format(self.solver)
            )
        solver = self.solver
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter <= 0:
            raise ValueError(
                "Maximum number of iteration must be a positive "
                "integer;"
                " got (max_iter={0!r})".format(self.max_iter)
            )
        if not isinstance(self.tol, numbers.Number) or self.tol <= 0:
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "positive; got (tol={0!r})".format(self.tol)
            )
        if not isinstance(self.warm_start, bool):
            raise ValueError(
                "The argument warm_start must be bool; got {0}".format(self.warm_start)
            )

        family = self._family_instance
        link = self._link_instance

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csc", "csr"],
            dtype=[np.float64, np.float32],
            y_numeric=True,
            multi_output=False,
        )

        weights = _check_sample_weight(sample_weight, X)

        _, n_features = X.shape

        if not np.all(family.in_y_range(y)):
            raise ValueError(
                "Some value(s) of y are out of the valid range for family {0}".format(
                    family.__class__.__name__
                )
            )
        # TODO: if alpha=0 check that X is not rank deficient

        # rescaling of sample_weight
        #
        # IMPORTANT NOTE: Since we want to minimize
        # 1/(2*sum(sample_weight)) * deviance + L2,
        # deviance = sum(sample_weight * unit_deviance),
        # we rescale weights such that sum(weights) = 1 and this becomes
        # 1/2*deviance + L2 with deviance=sum(weights * unit_deviance)
        weights = weights / weights.sum()

        if self.warm_start and hasattr(self, "coef_"):
            if self.fit_intercept:
                coef = np.concatenate((np.array([self.intercept_]), self.coef_))
            else:
                coef = self.coef_
        else:
            if self.fit_intercept:
                coef = np.zeros(n_features + 1)
                coef[0] = link(np.average(y, weights=weights))
            else:
                coef = np.zeros(n_features)

        # algorithms for optimization

        if solver == "lbfgs":

            def func(coef, X, y, weights, alpha, family, link):
                y_pred, devp = _y_pred_deviance_derivative(
                    coef, X, y, weights, family, link
                )
                dev = family.deviance(y, y_pred, weights)
                # offset if coef[0] is intercept
                offset = 1 if self.fit_intercept else 0
                coef_scaled = alpha * coef[offset:]
                obj = 0.5 * dev + 0.5 * (coef[offset:] @ coef_scaled)
                objp = 0.5 * devp
                objp[offset:] += coef_scaled
                return obj, objp

            args = (X, y, weights, self.alpha, family, link)

            opt_res = scipy.optimize.minimize(
                func,
                coef,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": self.max_iter,
                    "iprint": (self.verbose > 0) - 1,
                    "gtol": self.tol,
                    "ftol": 1e3 * np.finfo(float).eps,
                },
                args=args,
            )
            self.n_iter_ = _check_optimize_result("lbfgs", opt_res)
            coef = opt_res.x

        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            # set intercept to zero as the other linear models do
            self.intercept_ = 0.0
            self.coef_ = coef

        return self

    def _linear_predictor(self, X):
        """Compute the linear_predictor = `X @ coef_ + intercept_`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values of linear predictor.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            allow_nd=False,
            reset=False,
        )
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """Predict using GLM with feature matrix X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values.
        """
        # check_array is done in _linear_predictor
        eta = self._linear_predictor(X)
        y_pred = self._link_instance.inverse(eta)
        return y_pred

    def score(self, X, y, sample_weight=None):
        """Compute D^2, the percentage of deviance explained.

        D^2 is a generalization of the coefficient of determination R^2.
        R^2 uses squared error and D^2 deviance. Note that those two are equal
        for ``family='normal'``.

        D^2 is defined as
        :math:`D^2 = 1-\\frac{D(y_{true},y_{pred})}{D_{null}}`,
        :math:`D_{null}` is the null deviance, i.e. the deviance of a model
        with intercept alone, which corresponds to :math:`y_{pred} = \\bar{y}`.
        The mean :math:`\\bar{y}` is averaged by sample_weight.
        Best possible score is 1.0 and it can be negative (because the model
        can be arbitrarily worse).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values of target.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            D^2 of self.predict(X) w.r.t. y.
        """
        # Note, default score defined in RegressorMixin is R^2 score.
        # TODO: make D^2 a score function in module metrics (and thereby get
        #       input validation and so on)
        weights = _check_sample_weight(sample_weight, X)
        y_pred = self.predict(X)
        dev = self._family_instance.deviance(y, y_pred, weights=weights)
        y_mean = np.average(y, weights=weights)
        dev_null = self._family_instance.deviance(y, y_mean, weights=weights)
        return 1 - dev / dev_null

    def _more_tags(self):
        # create the _family_instance if fit wasn't called yet.
        if hasattr(self, "_family_instance"):
            _family_instance = self._family_instance
        elif isinstance(self.family, ExponentialDispersionModel):
            _family_instance = self.family
        elif self.family in EDM_DISTRIBUTIONS:
            _family_instance = EDM_DISTRIBUTIONS[self.family]()
        else:
            raise ValueError
        return {"requires_positive_y": not _family_instance.in_y_range(-1.0)}


class PoissonRegressor(GeneralizedLinearRegressor):
    """Generalized Linear Model with a Poisson distribution.

    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Actual number of iterations used in the solver.

    Examples
    ----------
    >>> from sklearn import linear_model
    >>> clf = linear_model.PoissonRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [12, 17, 22, 21]
    >>> clf.fit(X, y)
    PoissonRegressor()
    >>> clf.score(X, y)
    0.990...
    >>> clf.coef_
    array([0.121..., 0.158...])
    >>> clf.intercept_
    2.088...
    >>> clf.predict([[1, 1], [3, 4]])
    array([10.676..., 21.875...])

    See Also
    ----------
    GeneralizedLinearRegressor : Generalized Linear Model with a Poisson
        distribution.
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):

        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            family="poisson",
            link="log",
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    @property
    def family(self):
        """Return the string `'poisson'`."""
        # Make this attribute read-only to avoid mis-uses e.g. in GridSearch.
        return "poisson"

    @family.setter
    def family(self, value):
        if value != "poisson":
            raise ValueError("PoissonRegressor.family must be 'poisson'!")


class GammaRegressor(GeneralizedLinearRegressor):
    """Generalized Linear Model with a Gamma distribution.

    This regressor uses the 'log' link function.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X * coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_iter_ : int
        Actual number of iterations used in the solver.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PoissonRegressor : Generalized Linear Model with a Poisson distribution.
    TweedieRegressor : Generalized Linear Model with a Tweedie distribution.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.GammaRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [19, 26, 33, 30]
    >>> clf.fit(X, y)
    GammaRegressor()
    >>> clf.score(X, y)
    0.773...
    >>> clf.coef_
    array([0.072..., 0.066...])
    >>> clf.intercept_
    2.896...
    >>> clf.predict([[1, 0], [2, 8]])
    array([19.483..., 35.795...])
    """

    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):

        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            family="gamma",
            link="log",
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    @property
    def family(self):
        """Return the family of the regressor."""
        # Make this attribute read-only to avoid mis-uses e.g. in GridSearch.
        return "gamma"

    @family.setter
    def family(self, value):
        if value != "gamma":
            raise ValueError("GammaRegressor.family must be 'gamma'!")


class TweedieRegressor(GeneralizedLinearRegressor):
    """Generalized Linear Model with a Tweedie distribution.

    This estimator can be used to model different GLMs depending on the
    ``power`` parameter, which determines the underlying distribution.

    Read more in the :ref:`User Guide <Generalized_linear_regression>`.

    .. versionadded:: 0.23

    Parameters
    ----------
    power : float, default=0
            The power determines the underlying target distribution according
            to the following table:

            +-------+------------------------+
            | Power | Distribution           |
            +=======+========================+
            | 0     | Normal                 |
            +-------+------------------------+
            | 1     | Poisson                |
            +-------+------------------------+
            | (1,2) | Compound Poisson Gamma |
            +-------+------------------------+
            | 2     | Gamma                  |
            +-------+------------------------+
            | 3     | Inverse Gaussian       |
            +-------+------------------------+

            For ``0 < power < 1``, no distribution exists.

    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the linear predictor (X @ coef + intercept).

    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:

        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions

    max_iter : int, default=100
        The maximal number of iterations for the solver.

    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.

    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .

    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        Estimated coefficients for the linear predictor (`X @ coef_ +
        intercept_`) in the GLM.

    intercept_ : float
        Intercept (a.k.a. bias) added to linear predictor.

    n_iter_ : int
        Actual number of iterations used in the solver.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PoissonRegressor : Generalized Linear Model with a Poisson distribution.
    GammaRegressor : Generalized Linear Model with a Gamma distribution.

    Examples
    ----------
    >>> from sklearn import linear_model
    >>> clf = linear_model.TweedieRegressor()
    >>> X = [[1, 2], [2, 3], [3, 4], [4, 3]]
    >>> y = [2, 3.5, 5, 5.5]
    >>> clf.fit(X, y)
    TweedieRegressor()
    >>> clf.score(X, y)
    0.839...
    >>> clf.coef_
    array([0.599..., 0.299...])
    >>> clf.intercept_
    1.600...
    >>> clf.predict([[1, 1], [3, 4]])
    array([2.500..., 4.599...])
    """

    def __init__(
        self,
        *,
        power=0.0,
        alpha=1.0,
        fit_intercept=True,
        link="auto",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):

        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            family=TweedieDistribution(power=power),
            link=link,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    @property
    def family(self):
        """Return the family of the regressor."""
        # We use a property with a setter to make sure that the family is
        # always a Tweedie distribution, and that self.power and
        # self.family.power are identical by construction.
        dist = TweedieDistribution(power=self.power)
        # TODO: make the returned object immutable
        return dist

    @family.setter
    def family(self, value):
        if isinstance(value, TweedieDistribution):
            self.power = value.power
        else:
            raise TypeError(
                "TweedieRegressor.family must be of type TweedieDistribution!"
            )
