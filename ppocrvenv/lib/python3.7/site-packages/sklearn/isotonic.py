# Authors: Fabian Pedregosa <fabian@fseoane.net>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
import warnings
import math

from .base import BaseEstimator, TransformerMixin, RegressorMixin
from .utils import check_array, check_consistent_length
from .utils.validation import _check_sample_weight
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique


__all__ = ["check_increasing", "isotonic_regression", "IsotonicRegression"]


def check_increasing(x, y):
    """Determine whether y is monotonically correlated with x.

    y is found increasing or decreasing with respect to x based on a Spearman
    correlation test.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
            Training data.

    y : array-like of shape (n_samples,)
        Training target.

    Returns
    -------
    increasing_bool : boolean
        Whether the relationship is increasing or decreasing.

    Notes
    -----
    The Spearman correlation coefficient is estimated from the data, and the
    sign of the resulting estimate is used as the result.

    In the event that the 95% confidence interval based on Fisher transform
    spans zero, a warning is raised.

    References
    ----------
    Fisher transformation. Wikipedia.
    https://en.wikipedia.org/wiki/Fisher_transformation
    """

    # Calculate Spearman rho estimate and set return accordingly.
    rho, _ = spearmanr(x, y)
    increasing_bool = rho >= 0

    # Run Fisher transform to get the rho CI, but handle rho=+/-1
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        F_se = 1 / math.sqrt(len(x) - 3)

        # Use a 95% CI, i.e., +/-1.96 S.E.
        # https://en.wikipedia.org/wiki/Fisher_transformation
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)

        # Warn if the CI spans zero.
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn(
                "Confidence interval of the Spearman "
                "correlation coefficient spans zero. "
                "Determination of ``increasing`` may be "
                "suspect."
            )

    return increasing_bool


def isotonic_regression(
    y, *, sample_weight=None, y_min=None, y_max=None, increasing=True
):
    """Solve the isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The data.

    sample_weight : array-like of shape (n_samples,), default=None
        Weights on each point of the regression.
        If None, weight is set to 1 (equal weights).

    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool, default=True
        Whether to compute ``y_`` is increasing (if set to True) or decreasing
        (if set to False)

    Returns
    -------
    y_ : list of floats
        Isotonic fit of y.

    References
    ----------
    "Active set algorithms for isotonic regression; A unifying framework"
    by Michael J. Best and Nilotpal Chakravarti, section 3.
    """
    order = np.s_[:] if increasing else np.s_[::-1]
    y = check_array(y, ensure_2d=False, dtype=[np.float64, np.float32])
    y = np.array(y[order], dtype=y.dtype)
    sample_weight = _check_sample_weight(sample_weight, y, dtype=y.dtype, copy=True)
    sample_weight = np.ascontiguousarray(sample_weight[order])

    _inplace_contiguous_isotonic_regression(y, sample_weight)
    if y_min is not None or y_max is not None:
        # Older versions of np.clip don't accept None as a bound, so use np.inf
        if y_min is None:
            y_min = -np.inf
        if y_max is None:
            y_max = np.inf
        np.clip(y, y_min, y_max, y)
    return y[order]


class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    """Isotonic regression model.

    Read more in the :ref:`User Guide <isotonic>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    y_min : float, default=None
        Lower bound on the lowest predicted value (the minimum value may
        still be higher). If not set, defaults to -inf.

    y_max : float, default=None
        Upper bound on the highest predicted value (the maximum may still be
        lower). If not set, defaults to +inf.

    increasing : bool or 'auto', default=True
        Determines whether the predictions should be constrained to increase
        or decrease with `X`. 'auto' will decide based on the Spearman
        correlation estimate's sign.

    out_of_bounds : {'nan', 'clip', 'raise'}, default='nan'
        Handles how `X` values outside of the training domain are handled
        during prediction.

        - 'nan', predictions will be NaN.
        - 'clip', predictions will be set to the value corresponding to
          the nearest train interval endpoint.
        - 'raise', a `ValueError` is raised.

    Attributes
    ----------
    X_min_ : float
        Minimum value of input array `X_` for left bound.

    X_max_ : float
        Maximum value of input array `X_` for right bound.

    X_thresholds_ : ndarray of shape (n_thresholds,)
        Unique ascending `X` values used to interpolate
        the y = f(X) monotonic function.

        .. versionadded:: 0.24

    y_thresholds_ : ndarray of shape (n_thresholds,)
        De-duplicated `y` values suitable to interpolate the y = f(X)
        monotonic function.

        .. versionadded:: 0.24

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    increasing_ : bool
        Inferred value for ``increasing``.

    See Also
    --------
    sklearn.linear_model.LinearRegression : Ordinary least squares Linear
        Regression.
    sklearn.ensemble.HistGradientBoostingRegressor : Gradient boosting that
        is a non-parametric model accepting monotonicity constraints.
    isotonic_regression : Function to solve the isotonic regression model.

    Notes
    -----
    Ties are broken using the secondary method from de Leeuw, 1977.

    References
    ----------
    Isotonic Median Regression: A Linear Programming Approach
    Nilotpal Chakravarti
    Mathematics of Operations Research
    Vol. 14, No. 2 (May, 1989), pp. 303-308

    Isotone Optimization in R : Pool-Adjacent-Violators
    Algorithm (PAVA) and Active Set Methods
    de Leeuw, Hornik, Mair
    Journal of Statistical Software 2009

    Correctness of Kruskal's algorithms for monotone regression with ties
    de Leeuw, Psychometrica, 1977

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.isotonic import IsotonicRegression
    >>> X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    >>> iso_reg = IsotonicRegression().fit(X, y)
    >>> iso_reg.predict([.1, .2])
    array([1.8628..., 3.7256...])
    """

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan"):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _check_input_data_shape(self, X):
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = (
                "Isotonic regression input X should be a 1d array or "
                "2d array with 1 feature"
            )
            raise ValueError(msg)

    def _build_f(self, X, y):
        """Build the f_ interp1d function."""

        # Handle the out_of_bounds argument by setting bounds_error
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError(
                "The argument ``out_of_bounds`` must be in "
                "'nan', 'clip', 'raise'; got {0}".format(self.out_of_bounds)
            )

        bounds_error = self.out_of_bounds == "raise"
        if len(y) == 1:
            # single y, constant prediction
            self.f_ = lambda x: y.repeat(x.shape)
        else:
            self.f_ = interpolate.interp1d(
                X, y, kind="linear", bounds_error=bounds_error
            )

    def _build_y(self, X, y, sample_weight, trim_duplicates=True):
        """Build the y_ IsotonicRegression."""
        self._check_input_data_shape(X)
        X = X.reshape(-1)  # use 1d view

        # Determine increasing if auto-determination requested
        if self.increasing == "auto":
            self.increasing_ = check_increasing(X, y)
        else:
            self.increasing_ = self.increasing

        # If sample_weights is passed, removed zero-weight values and clean
        # order
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        mask = sample_weight > 0
        X, y, sample_weight = X[mask], y[mask], sample_weight[mask]

        order = np.lexsort((y, X))
        X, y, sample_weight = [array[order] for array in [X, y, sample_weight]]
        unique_X, unique_y, unique_sample_weight = _make_unique(X, y, sample_weight)

        X = unique_X
        y = isotonic_regression(
            unique_y,
            sample_weight=unique_sample_weight,
            y_min=self.y_min,
            y_max=self.y_max,
            increasing=self.increasing_,
        )

        # Handle the left and right bounds on X
        self.X_min_, self.X_max_ = np.min(X), np.max(X)

        if trim_duplicates:
            # Remove unnecessary points for faster prediction
            keep_data = np.ones((len(y),), dtype=bool)
            # Aside from the 1st and last point, remove points whose y values
            # are equal to both the point before and the point after it.
            keep_data[1:-1] = np.logical_or(
                np.not_equal(y[1:-1], y[:-2]), np.not_equal(y[1:-1], y[2:])
            )
            return X[keep_data], y[keep_data]
        else:
            # The ability to turn off trim_duplicates is only used to it make
            # easier to unit test that removing duplicates in y does not have
            # any impact the resulting interpolation function (besides
            # prediction speed).
            return X, y

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False)
        X = check_array(X, dtype=[np.float64, np.float32], **check_params)
        y = check_array(y, dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)

        # Transform y by running the isotonic regression algorithm and
        # transform X accordingly.
        X, y = self._build_y(X, y, sample_weight)

        # It is necessary to store the non-redundant part of the training set
        # on the model to make it possible to support model persistence via
        # the pickle module as the object built by scipy.interp1d is not
        # picklable directly.
        self.X_thresholds_, self.y_thresholds_ = X, y

        # Build the interpolation function
        self._build_f(X, y)
        return self

    def transform(self, T):
        """Transform new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

            .. versionchanged:: 0.24
               Also accepts 2d array with 1 feature.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The transformed data.
        """

        if hasattr(self, "X_thresholds_"):
            dtype = self.X_thresholds_.dtype
        else:
            dtype = np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False)

        self._check_input_data_shape(T)
        T = T.reshape(-1)  # use 1d view

        # Handle the out_of_bounds argument by clipping if needed
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError(
                "The argument ``out_of_bounds`` must be in "
                "'nan', 'clip', 'raise'; got {0}".format(self.out_of_bounds)
            )

        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        # on scipy 0.17, interp1d up-casts to float64, so we cast back
        res = res.astype(T.dtype)

        return res

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,) or (n_samples, 1)
            Data to transform.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self.transform(T)

    def __getstate__(self):
        """Pickle-protocol - return state of the estimator."""
        state = super().__getstate__()
        # remove interpolation method
        state.pop("f_", None)
        return state

    def __setstate__(self, state):
        """Pickle-protocol - set state of the estimator.

        We need to rebuild the interpolation function.
        """
        super().__setstate__(state)
        if hasattr(self, "X_thresholds_") and hasattr(self, "y_thresholds_"):
            self._build_f(self.X_thresholds_, self.y_thresholds_)

    def _more_tags(self):
        return {"X_types": ["1darray"]}
