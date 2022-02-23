"""Truncated SVD for sparse matrices, aka latent semantic analysis (LSA).
"""

# Author: Lars Buitinck
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Michael Becker <mike@beckerfuffle.com>
# License: 3-clause BSD.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import check_is_fitted


__all__ = ["TruncatedSVD"]


class TruncatedSVD(TransformerMixin, BaseEstimator):
    """Dimensionality reduction using truncated SVD (aka LSA).

    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with sparse matrices
    efficiently.

    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in :mod:`sklearn.feature_extraction.text`. In
    that context, it is known as latent semantic analysis (LSA).

    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on `X * X.T` or
    `X.T * X`, whichever is more efficient.

    Read more in the :ref:`User Guide <LSA>`.

    Parameters
    ----------
    n_components : int, default=2
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.

    algorithm : {'arpack', 'randomized'}, default='randomized'
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    n_iter : int, default=5
        Number of iterations for randomized SVD solver. Not used by ARPACK. The
        default is larger than the default in
        :func:`~sklearn.utils.extmath.randomized_svd` to handle sparse
        matrices that may have large slowly decaying spectrum.

    random_state : int, RandomState instance or None, default=None
        Used during randomized svd. Pass an int for reproducible results across
        multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors of the input data.

    explained_variance_ : ndarray of shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.

    singular_values_ : ndarray od shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    FactorAnalysis : A simple linear generative model with
        Gaussian latent variables.
    IncrementalPCA : Incremental principal components analysis.
    KernelPCA : Kernel Principal component analysis.
    NMF : Non-Negative Matrix Factorization.
    PCA : Principal component analysis.

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.

    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X_dense = np.random.rand(100, 100)
    >>> X_dense[:, 2 * np.arange(50)] = 0
    >>> X = csr_matrix(X_dense)
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)
    TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> print(svd.explained_variance_ratio_)
    [0.0157... 0.0512... 0.0499... 0.0479... 0.0453...]
    >>> print(svd.explained_variance_ratio_.sum())
    0.2102...
    >>> print(svd.singular_values_)
    [35.2410...  4.5981...   4.5420...  4.4486...  4.3288...]
    """

    def __init__(
        self,
        n_components=2,
        *,
        algorithm="randomized",
        n_iter=5,
        random_state=None,
        tol=0.0,
    ):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y=None):
        """Fit model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        X = self._validate_data(X, accept_sparse=["csr", "csc"], ensure_min_features=2)
        random_state = check_random_state(self.random_state)

        if self.algorithm == "arpack":
            v0 = _init_arpack_v0(min(X.shape), random_state)
            U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            Sigma = Sigma[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])

        elif self.algorithm == "randomized":
            k = self.n_components
            n_features = X.shape[1]
            if k >= n_features:
                raise ValueError(
                    "n_components must be < n_features; got %d >= %d" % (k, n_features)
                )
            U, Sigma, VT = randomized_svd(
                X, self.n_components, n_iter=self.n_iter, random_state=random_state
            )
        else:
            raise ValueError("unknown algorithm %r" % self.algorithm)

        self.components_ = VT

        # As a result of the SVD approximation error on X ~ U @ Sigma @ V.T,
        # X @ V is not the same as U @ Sigma
        if self.algorithm == "randomized" or (
            self.algorithm == "arpack" and self.tol > 0
        ):
            X_transformed = safe_sparse_dot(X, self.components_.T)
        else:
            X_transformed = U * Sigma

        # Calculate explained variance & explained variance ratio
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sp.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)
        return safe_sparse_dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        X = check_array(X)
        return np.dot(X, self.components_)

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}
