"""
The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""

# Author: Edouard Duchesnay <edouard.duchesnay@cea.fr>
# License: BSD 3 clause

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import svd

from ..base import BaseEstimator, RegressorMixin, TransformerMixin
from ..base import MultiOutputMixin
from ..utils import check_array, check_consistent_length
from ..utils.fixes import sp_version
from ..utils.fixes import parse_version
from ..utils.extmath import svd_flip
from ..utils.validation import check_is_fitted, FLOAT_DTYPES
from ..exceptions import ConvergenceWarning
from ..utils.deprecation import deprecated

__all__ = ["PLSCanonical", "PLSRegression", "PLSSVD"]


if sp_version >= parse_version("1.7"):
    # Starting in scipy 1.7 pinv2 was deprecated in favor of pinv.
    # pinv now uses the svd to compute the pseudo-inverse.
    from scipy.linalg import pinv as pinv2
else:
    from scipy.linalg import pinv2


def _pinv2_old(a):
    # Used previous scipy pinv2 that was updated in:
    # https://github.com/scipy/scipy/pull/10067
    # We can not set `cond` or `rcond` for pinv2 in scipy >= 1.3 to keep the
    # same behavior of pinv2 for scipy < 1.3, because the condition used to
    # determine the rank is dependent on the output of svd.
    u, s, vh = svd(a, full_matrices=False, check_finite=False)

    t = u.dtype.char.lower()
    factor = {"f": 1e3, "d": 1e6}
    cond = np.max(s) * factor[t] * np.finfo(t).eps
    rank = np.sum(s > cond)

    u = u[:, :rank]
    u /= s[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))


def _get_first_singular_vectors_power_method(
    X, Y, mode="A", max_iter=500, tol=1e-06, norm_y_weights=False
):
    """Return the first left and right singular vectors of X'Y.

    Provides an alternative to the svd(X'Y) and uses the power method instead.
    With norm_y_weights to True and in mode A, this corresponds to the
    algorithm section 11.3 of the Wegelin's review, except this starts at the
    "update saliences" part.
    """

    eps = np.finfo(X.dtype).eps
    try:
        y_score = next(col for col in Y.T if np.any(np.abs(col) > eps))
    except StopIteration as e:
        raise StopIteration("Y residual is constant") from e

    x_weights_old = 100  # init to big value for first convergence check

    if mode == "B":
        # Precompute pseudo inverse matrices
        # Basically: X_pinv = (X.T X)^-1 X.T
        # Which requires inverting a (n_features, n_features) matrix.
        # As a result, and as detailed in the Wegelin's review, CCA (i.e. mode
        # B) will be unstable if n_features > n_samples or n_targets >
        # n_samples
        X_pinv, Y_pinv = _pinv2_old(X), _pinv2_old(Y)

    for i in range(max_iter):
        if mode == "B":
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)

        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        x_score = np.dot(X, x_weights)

        if mode == "B":
            y_weights = np.dot(Y_pinv, x_score)
        else:
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)

        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps

        y_score = np.dot(Y, y_weights) / (np.dot(y_weights, y_weights) + eps)

        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights

    n_iter = i + 1
    if n_iter == max_iter:
        warnings.warn("Maximum number of iterations reached", ConvergenceWarning)

    return x_weights, y_weights, n_iter


def _get_first_singular_vectors_svd(X, Y):
    """Return the first left and right singular vectors of X'Y.

    Here the whole SVD is computed.
    """
    C = np.dot(X.T, Y)
    U, _, Vt = svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


def _center_scale_xy(X, Y, scale=True):
    """Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


def _svd_flip_1d(u, v):
    """Same as svd_flip but works on 1d arrays, and is inplace"""
    # svd_flip would force us to convert to 2d array and would also return 2d
    # arrays. We don't want that.
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


class _PLS(
    TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta
):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://www.stat.washington.edu/research/reports/2000/tr371.pdf
    """

    @abstractmethod
    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        deflation_mode="regression",
        mode="A",
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.
        """

        check_consistent_length(X, Y)
        X = self._validate_data(
            X, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        n_components = self.n_components
        if self.deflation_mode == "regression":
            # With PLSRegression n_components is bounded by the rank of (X.T X)
            # see Wegelin page 25
            rank_upper_bound = p
            if not 1 <= n_components <= rank_upper_bound:
                # TODO: raise an error in 1.1
                warnings.warn(
                    f"As of version 0.24, n_components({n_components}) should "
                    "be in [1, n_features]."
                    f"n_components={rank_upper_bound} will be used instead. "
                    "In version 1.1 (renaming of 0.26), an error will be "
                    "raised.",
                    FutureWarning,
                )
                n_components = rank_upper_bound
        else:
            # With CCA and PLSCanonical, n_components is bounded by the rank of
            # X and the rank of Y: see Wegelin page 12
            rank_upper_bound = min(n, p, q)
            if not 1 <= self.n_components <= rank_upper_bound:
                # TODO: raise an error in 1.1
                warnings.warn(
                    f"As of version 0.24, n_components({n_components}) should "
                    "be in [1, min(n_features, n_samples, n_targets)] = "
                    f"[1, {rank_upper_bound}]. "
                    f"n_components={rank_upper_bound} will be used instead. "
                    "In version 1.1 (renaming of 0.26), an error will be "
                    "raised.",
                    FutureWarning,
                )
                n_components = rank_upper_bound

        if self.algorithm not in ("svd", "nipals"):
            raise ValueError(
                f"algorithm should be 'svd' or 'nipals', got {self.algorithm}."
            )

        self._norm_y_weights = self.deflation_mode == "canonical"  # 1.1
        norm_y_weights = self._norm_y_weights

        # Scale (in place)
        Xk, Yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, Y, self.scale
        )

        self.x_weights_ = np.zeros((p, n_components))  # U
        self.y_weights_ = np.zeros((q, n_components))  # V
        self._x_scores = np.zeros((n, n_components))  # Xi
        self._y_scores = np.zeros((n, n_components))  # Omega
        self.x_loadings_ = np.zeros((p, n_components))  # Gamma
        self.y_loadings_ = np.zeros((q, n_components))  # Delta
        self.n_iter_ = []

        # This whole thing corresponds to the algorithm in section 4.1 of the
        # review from Wegelin. See above for a notation mapping from code to
        # paper.
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(n_components):
            # Find first left and right singular vectors of the X.T.dot(Y)
            # cross-covariance matrix.
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0

                try:
                    (
                        x_weights,
                        y_weights,
                        n_iter_,
                    ) = _get_first_singular_vectors_power_method(
                        Xk,
                        Yk,
                        mode=self.mode,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        norm_y_weights=norm_y_weights,
                    )
                except StopIteration as e:
                    if str(e) != "Y residual is constant":
                        raise
                    warnings.warn(f"Y residual is constant at iteration {k}")
                    break

                self.n_iter_.append(n_iter_)

            elif self.algorithm == "svd":
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)

            # inplace sign flip for consistency across solvers and archs
            _svd_flip_1d(x_weights, y_weights)

            # compute scores, i.e. the projections of X and Y
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss

            # Deflation: subtract rank-one approx to obtain Xk+1 and Yk+1
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)

            if self.deflation_mode == "canonical":
                # regress Yk on y_score
                y_loadings = np.dot(y_scores, Yk) / np.dot(y_scores, y_scores)
                Yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == "regression":
                # regress Yk on x_score
                y_loadings = np.dot(x_scores, Yk) / np.dot(x_scores, x_scores)
                Yk -= np.outer(x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings

        # X was approximated as Xi . Gamma.T + X_(R+1)
        # Xi . Gamma.T is a sum of n_components rank-1 matrices. X_(R+1) is
        # whatever is left to fully reconstruct X, and can be 0 if X is of rank
        # n_components.
        # Similarly, Y was approximated as Omega . Delta.T + Y_(R+1)

        # Compute transformation matrices (rotations_). See User Guide.
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False),
        )
        self.y_rotations_ = np.dot(
            self.y_weights_,
            pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False),
        )

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = self.coef_ * self._y_std
        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self._y_mean
            Y /= self._y_std
            y_scores = np.dot(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Returns
        -------
        self : ndarray of shape (n_samples, n_features)
            Return the reconstructed array.

        Notes
        -----
        This transformation will only be exact if `n_components=n_features`.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self._x_std
        X_reconstructed += self._x_mean
        return X_reconstructed

    def predict(self, X, copy=True):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        Ypred = np.dot(X, self.coef_)
        return Ypred + self._y_mean

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        return self.fit(X, y).transform(X, y)

    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `norm_y_weights` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def norm_y_weights(self):
        return self._norm_y_weights

    @deprecated(  # type: ignore
        "Attribute `x_mean_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def x_mean_(self):
        return self._x_mean

    @deprecated(  # type: ignore
        "Attribute `y_mean_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def y_mean_(self):
        return self._y_mean

    @deprecated(  # type: ignore
        "Attribute `x_std_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def x_std_(self):
        return self._x_std

    @deprecated(  # type: ignore
        "Attribute `y_std_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def y_std_(self):
        return self._y_std

    @property
    def x_scores_(self):
        """Attribute `x_scores_` was deprecated in version 0.24."""
        # TODO: raise error in 1.1 instead
        if not isinstance(self, PLSRegression):
            pass
            warnings.warn(
                "Attribute `x_scores_` was deprecated in version 0.24 and "
                "will be removed in 1.1 (renaming of 0.26). Use "
                "est.transform(X) on the training data instead.",
                FutureWarning,
            )
        return self._x_scores

    @property
    def y_scores_(self):
        """Attribute `y_scores_` was deprecated in version 0.24."""
        # TODO: raise error in 1.1 instead
        if not isinstance(self, PLSRegression):
            warnings.warn(
                "Attribute `y_scores_` was deprecated in version 0.24 and "
                "will be removed in 1.1 (renaming of 0.26). Use "
                "est.transform(X) on the training data instead.",
                FutureWarning,
            )
        return self._y_scores

    def _more_tags(self):
        return {"poor_score": True, "requires_y": False}


class PLSRegression(_PLS):
    """PLS regression.

    PLSRegression is also known as PLS2 or PLS1, depending on the number of
    targets.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in :term:`fit` before applying centering,
        and potentially scaling. If `False`, these operations will be done
        inplace, modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_features, n_targets)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_`.

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, Y)
    PLSRegression()
    >>> Y_pred = pls2.predict(X)
    """

    # This implementation provides the same results that 3 PLS packages
    # provided in the R language (R-project):
    #     - "mixOmics" with function pls(X, Y, mode = "regression")
    #     - "plspm " with function plsreg2(X, Y)
    #     - "pls" with function oscorespls.fit(X, Y)

    def __init__(
        self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True
    ):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="regression",
            mode="A",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class PLSCanonical(_PLS):
    """Partial Least Squares transformer and regressor.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

        .. deprecated:: 0.24
           `x_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

        .. deprecated:: 0.24
           `y_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_features, n_targets)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_`.

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CCA : Canonical Correlation Analysis.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, Y)
    PLSCanonical()
    >>> X_c, Y_c = plsca.transform(X, Y)
    """

    # This implementation provides the same results that the "plspm" package
    # provided in the R language (R-project), using the function plsca(X, Y).
    # Results are equal or collinear with the function
    # ``pls(..., mode = "canonical")`` of the "mixOmics" package. The
    # difference relies in the fact that mixOmics implementation does not
    # exactly implement the Wold algorithm since it does not normalize
    # y_weights to one.

    def __init__(
        self,
        n_components=2,
        *,
        scale=True,
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        copy=True,
    ):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            mode="A",
            algorithm=algorithm,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class CCA(_PLS):
    """Canonical Correlation Analysis, also known as "Mode B" PLS.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    max_iter : int, default=500
        The maximum number of iterations of the power method.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

        .. deprecated:: 0.24
           `x_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

        .. deprecated:: 0.24
           `y_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_features, n_targets)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_`.

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from sklearn.cross_decomposition import CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> cca = CCA(n_components=1)
    >>> cca.fit(X, Y)
    CCA(n_components=1)
    >>> X_c, Y_c = cca.transform(X, Y)
    """

    def __init__(
        self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True
    ):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode="canonical",
            mode="B",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class PLSSVD(TransformerMixin, BaseEstimator):
    """Partial Least Square SVD.

    This transformer simply performs a SVD on the cross-covariance matrix
    `X'Y`. It is able to project both the training data `X` and the targets
    `Y`. The training data `X` is projected on the left singular vectors, while
    the targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If `False`, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

        .. deprecated:: 0.24
           `x_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

        .. deprecated:: 0.24
           `y_scores_` is deprecated in 0.24 and will be removed in 1.1
           (renaming of 0.26). You can just call `transform` on the training
           data instead.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    CCA : Canonical Correlation Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.cross_decomposition import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, Y)
    >>> X_c, Y_c = pls.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))
    """

    def __init__(self, n_components=2, *, scale=True, copy=True):
        self.n_components = n_components
        self.scale = scale
        self.copy = copy

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        check_consistent_length(X, Y)
        X = self._validate_data(
            X, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # we'll compute the SVD of the cross-covariance matrix = X.T.dot(Y)
        # This matrix rank is at most min(n_samples, n_features, n_targets) so
        # n_components cannot be bigger than that.
        n_components = self.n_components
        rank_upper_bound = min(X.shape[0], X.shape[1], Y.shape[1])
        if not 1 <= n_components <= rank_upper_bound:
            # TODO: raise an error in 1.1
            warnings.warn(
                f"As of version 0.24, n_components({n_components}) should be "
                "in [1, min(n_features, n_samples, n_targets)] = "
                f"[1, {rank_upper_bound}]. "
                f"n_components={rank_upper_bound} will be used instead. "
                "In version 1.1 (renaming of 0.26), an error will be raised.",
                FutureWarning,
            )
            n_components = rank_upper_bound

        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, Y, self.scale
        )

        # Compute SVD of cross-covariance matrix
        C = np.dot(X.T, Y)
        U, s, Vt = svd(C, full_matrices=False)
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        U, Vt = svd_flip(U, Vt)
        V = Vt.T

        self._x_scores = np.dot(X, U)  # TODO: remove in 1.1
        self._y_scores = np.dot(Y, V)  # TODO: remove in 1.1
        self.x_weights_ = U
        self.y_weights_ = V
        return self

    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `x_scores_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26). Use est.transform(X) on "
        "the training data instead."
    )
    @property
    def x_scores_(self):
        return self._x_scores

    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `y_scores_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26). Use est.transform(X, Y) "
        "on the training data instead."
    )
    @property
    def y_scores_(self):
        return self._y_scores

    @deprecated(  # type: ignore
        "Attribute `x_mean_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def x_mean_(self):
        return self._x_mean

    @deprecated(  # type: ignore
        "Attribute `y_mean_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def y_mean_(self):
        return self._y_mean

    @deprecated(  # type: ignore
        "Attribute `x_std_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def x_std_(self):
        return self._x_std

    @deprecated(  # type: ignore
        "Attribute `y_std_` was deprecated in version 0.24 and "
        "will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def y_std_(self):
        return self._y_std

    def transform(self, X, Y=None):
        """
        Apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to be transformed.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        x_scores : array-like or tuple of array-like
            The transformed data `X_tranformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        check_is_fitted(self)
        X = self._validate_data(X, dtype=np.float64, reset=False)
        Xr = (X - self._x_mean) / self._x_std
        x_scores = np.dot(Xr, self.x_weights_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Yr = (Y - self._y_mean) / self._y_std
            y_scores = np.dot(Yr, self.y_weights_)
            return x_scores, y_scores
        return x_scores

    def fit_transform(self, X, y=None):
        """Learn and apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_tranformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        return self.fit(X, y).transform(X, y)
