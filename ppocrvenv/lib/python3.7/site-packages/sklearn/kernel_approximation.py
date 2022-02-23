"""
The :mod:`sklearn.kernel_approximation` module implements several
approximate kernel feature maps based on Fourier transforms and Count Sketches.
"""

# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Daniel Lopez-Sanchez (TensorSketch) <lope@usal.es>

# License: BSD 3 clause

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

try:
    from scipy.fft import fft, ifft
except ImportError:  # scipy < 1.4
    from scipy.fftpack import fft, ifft

from .base import BaseEstimator
from .base import TransformerMixin
from .utils import check_random_state
from .utils.extmath import safe_sparse_dot
from .utils.validation import check_is_fitted
from .metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from .utils.validation import check_non_negative


class PolynomialCountSketch(BaseEstimator, TransformerMixin):
    """Polynomial kernel approximation via Tensor Sketch.

    Implements Tensor Sketch, which approximates the feature map
    of the polynomial kernel::

        K(X, Y) = (gamma * <X, Y> + coef0)^degree

    by efficiently computing a Count Sketch of the outer product of a
    vector with itself using Fast Fourier Transforms (FFT). Read more in the
    :ref:`User Guide <polynomial_kernel_approx>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter of the polynomial kernel whose feature map
        will be approximated.

    degree : int, default=2
        Degree of the polynomial kernel whose feature map
        will be approximated.

    coef0 : int, default=0
        Constant term of the polynomial kernel whose feature map
        will be approximated.

    n_components : int, default=100
        Dimensionality of the output feature space. Usually, `n_components`
        should be greater than the number of features in input samples in
        order to achieve good performance. The optimal score / run time
        balance is typically achieved around `n_components` = 10 * `n_features`,
        but this depends on the specific dataset being used.

    random_state : int, RandomState instance, default=None
        Determines random number generation for indexHash and bitHash
        initialization. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

    Attributes
    ----------
    indexHash_ : ndarray of shape (degree, n_features), dtype=int64
        Array of indexes in range [0, n_components) used to represent
        the 2-wise independent hash functions for Count Sketch computation.

    bitHash_ : ndarray of shape (degree, n_features), dtype=float32
        Array with random entries in {+1, -1}, used to represent
        the 2-wise independent hash functions for Count Sketch computation.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    Examples
    --------
    >>> from sklearn.kernel_approximation import PolynomialCountSketch
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> ps = PolynomialCountSketch(degree=3, random_state=1)
    >>> X_features = ps.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=10, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=10)
    >>> clf.score(X_features, y)
    1.0
    """

    def __init__(
        self, *, gamma=1.0, degree=2, coef0=0, n_components=100, random_state=None
    ):
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Initializes the internal variables. The method needs no information
        about the distribution of data, so we only care about n_features in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not self.degree >= 1:
            raise ValueError(f"degree={self.degree} should be >=1.")

        X = self._validate_data(X, accept_sparse="csc")
        random_state = check_random_state(self.random_state)

        n_features = X.shape[1]
        if self.coef0 != 0:
            n_features += 1

        self.indexHash_ = random_state.randint(
            0, high=self.n_components, size=(self.degree, n_features)
        )

        self.bitHash_ = random_state.choice(a=[-1, 1], size=(self.degree, n_features))
        return self

    def transform(self, X):
        """Generate the feature map approximation for X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """

        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csc", reset=False)

        X_gamma = np.sqrt(self.gamma) * X

        if sp.issparse(X_gamma) and self.coef0 != 0:
            X_gamma = sp.hstack(
                [X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))],
                format="csc",
            )

        elif not sp.issparse(X_gamma) and self.coef0 != 0:
            X_gamma = np.hstack(
                [X_gamma, np.sqrt(self.coef0) * np.ones((X_gamma.shape[0], 1))]
            )

        if X_gamma.shape[1] != self.indexHash_.shape[1]:
            raise ValueError(
                "Number of features of test samples does not"
                " match that of training samples."
            )

        count_sketches = np.zeros((X_gamma.shape[0], self.degree, self.n_components))

        if sp.issparse(X_gamma):
            for j in range(X_gamma.shape[1]):
                for d in range(self.degree):
                    iHashIndex = self.indexHash_[d, j]
                    iHashBit = self.bitHash_[d, j]
                    count_sketches[:, d, iHashIndex] += (
                        (iHashBit * X_gamma[:, j]).toarray().ravel()
                    )

        else:
            for j in range(X_gamma.shape[1]):
                for d in range(self.degree):
                    iHashIndex = self.indexHash_[d, j]
                    iHashBit = self.bitHash_[d, j]
                    count_sketches[:, d, iHashIndex] += iHashBit * X_gamma[:, j]

        # For each same, compute a count sketch of phi(x) using the polynomial
        # multiplication (via FFT) of p count sketches of x.
        count_sketches_fft = fft(count_sketches, axis=2, overwrite_x=True)
        count_sketches_fft_prod = np.prod(count_sketches_fft, axis=1)
        data_sketch = np.real(ifft(count_sketches_fft_prod, overwrite_x=True))

        return data_sketch


class RBFSampler(TransformerMixin, BaseEstimator):
    """Approximate a RBF kernel feature map using random Fourier features.

    It implements a variant of Random Kitchen Sinks.[1]

    Read more in the :ref:`User Guide <rbf_kernel_approx>`.

    Parameters
    ----------
    gamma : float, default=1.0
        Parameter of RBF kernel: exp(-gamma * x^2).

    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype=float64
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.

    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype=float64
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    PolynomialCountSketch : Polynomial kernel approximation via Tensor Sketch.
    SkewedChi2Sampler : Approximate feature map for
        "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.

    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)

    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0
    """

    def __init__(self, *, gamma=1.0, n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = self._validate_data(X, accept_sparse="csr")
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        self.random_weights_ = np.sqrt(2 * self.gamma) * random_state.normal(
            size=(n_features, self.n_components)
        )

        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.0) / np.sqrt(self.n_components)
        return projection


class SkewedChi2Sampler(TransformerMixin, BaseEstimator):
    """Approximate feature map for "skewed chi-squared" kernel.

    Read more in the :ref:`User Guide <skewed_chi_kernel_approx>`.

    Parameters
    ----------
    skewedness : float, default=1.0
        "skewedness" parameter of the kernel. Needs to be cross-validated.

    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    random_weights_ : ndarray of shape (n_features, n_components)
        Weight array, sampled from a secant hyperbolic distribution, which will
        be used to linearly transform the log of the data.

    random_offset_ : ndarray of shape (n_features, n_components)
        Bias term, which will be added to the data. It is uniformly distributed
        between 0 and 2*pi.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    Nystroem : Approximate a kernel map using a subset of the training data.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    References
    ----------
    See "Random Fourier Approximations for Skewed Multiplicative Histogram
    Kernels" by Fuxin Li, Catalin Ionescu and Cristian Sminchisescu.

    Examples
    --------
    >>> from sklearn.kernel_approximation import SkewedChi2Sampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> chi2_feature = SkewedChi2Sampler(skewedness=.01,
    ...                                  n_components=10,
    ...                                  random_state=0)
    >>> X_features = chi2_feature.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=10, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=10)
    >>> clf.score(X_features, y)
    1.0
    """

    def __init__(self, *, skewedness=1.0, n_components=100, random_state=None):
        self.skewedness = skewedness
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Samples random projection according to n_features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = self._validate_data(X)
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        uniform = random_state.uniform(size=(n_features, self.n_components))
        # transform by inverse CDF of sech
        self.random_weights_ = 1.0 / np.pi * np.log(np.tan(np.pi / 2.0 * uniform))
        self.random_offset_ = random_state.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features. All values of X must be
            strictly greater than "-skewedness".

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, copy=True, dtype=[np.float64, np.float32], reset=False
        )
        if (X <= -self.skewedness).any():
            raise ValueError("X may not contain entries smaller than -skewedness.")

        X += self.skewedness
        np.log(X, X)
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.0) / np.sqrt(self.n_components)
        return projection


class AdditiveChi2Sampler(TransformerMixin, BaseEstimator):
    """Approximate feature map for additive chi2 kernel.

    Uses sampling the fourier transform of the kernel characteristic
    at regular intervals.

    Since the kernel that is to be approximated is additive, the components of
    the input vectors can be treated separately.  Each entry in the original
    space is transformed into 2*sample_steps+1 features, where sample_steps is
    a parameter of the method. Typical values of sample_steps include 1, 2 and
    3.

    Optimal choices for the sampling interval for certain data ranges can be
    computed (see the reference). The default values should be reasonable.

    Read more in the :ref:`User Guide <additive_chi_kernel_approx>`.

    Parameters
    ----------
    sample_steps : int, default=2
        Gives the number of (complex) sampling points.

    sample_interval : float, default=None
        Sampling interval. Must be specified when sample_steps not in {1,2,3}.

    Attributes
    ----------
    sample_interval_ : float
        Stored sampling interval. Specified as a parameter if `sample_steps`
        not in {1,2,3}.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SkewedChi2Sampler : A Fourier-approximation to a non-additive variant of
        the chi squared kernel.

    sklearn.metrics.pairwise.chi2_kernel : The exact chi squared kernel.

    sklearn.metrics.pairwise.additive_chi2_kernel : The exact additive chi
        squared kernel.

    Notes
    -----
    This estimator approximates a slightly different version of the additive
    chi squared kernel then ``metric.additive_chi2`` computes.

    References
    ----------
    See `"Efficient additive kernels via explicit feature maps"
    <http://www.robots.ox.ac.uk/~vedaldi/assets/pubs/vedaldi11efficient.pdf>`_
    A. Vedaldi and A. Zisserman, Pattern Analysis and Machine Intelligence,
    2011

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.kernel_approximation import AdditiveChi2Sampler
    >>> X, y = load_digits(return_X_y=True)
    >>> chi2sampler = AdditiveChi2Sampler(sample_steps=2)
    >>> X_transformed = chi2sampler.fit_transform(X, y)
    >>> clf = SGDClassifier(max_iter=5, random_state=0, tol=1e-3)
    >>> clf.fit(X_transformed, y)
    SGDClassifier(max_iter=5, random_state=0)
    >>> clf.score(X_transformed, y)
    0.9499...
    """

    def __init__(self, *, sample_steps=2, sample_interval=None):
        self.sample_steps = sample_steps
        self.sample_interval = sample_interval

    def fit(self, X, y=None):
        """Set the parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = self._validate_data(X, accept_sparse="csr")
        check_non_negative(X, "X in AdditiveChi2Sampler.fit")

        if self.sample_interval is None:
            # See reference, figure 2 c)
            if self.sample_steps == 1:
                self.sample_interval_ = 0.8
            elif self.sample_steps == 2:
                self.sample_interval_ = 0.5
            elif self.sample_steps == 3:
                self.sample_interval_ = 0.4
            else:
                raise ValueError(
                    "If sample_steps is not in [1, 2, 3],"
                    " you need to provide sample_interval"
                )
        else:
            self.sample_interval_ = self.sample_interval
        return self

    def transform(self, X):
        """Apply approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, \
               shape = (n_samples, n_features * (2*sample_steps + 1))
            Whether the return value is an array or sparse matrix depends on
            the type of the input X.
        """
        msg = (
            "%(name)s is not fitted. Call fit to set the parameters before"
            " calling transform"
        )
        check_is_fitted(self, msg=msg)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        check_non_negative(X, "X in AdditiveChi2Sampler.transform")
        sparse = sp.issparse(X)

        # zeroth component
        # 1/cosh = sech
        # cosh(0) = 1.0

        transf = self._transform_sparse if sparse else self._transform_dense
        return transf(X)

    def _transform_dense(self, X):
        non_zero = X != 0.0
        X_nz = X[non_zero]

        X_step = np.zeros_like(X)
        X_step[non_zero] = np.sqrt(X_nz * self.sample_interval_)

        X_new = [X_step]

        log_step_nz = self.sample_interval_ * np.log(X_nz)
        step_nz = 2 * X_nz * self.sample_interval_

        for j in range(1, self.sample_steps):
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * self.sample_interval_))

            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.cos(j * log_step_nz)
            X_new.append(X_step)

            X_step = np.zeros_like(X)
            X_step[non_zero] = factor_nz * np.sin(j * log_step_nz)
            X_new.append(X_step)

        return np.hstack(X_new)

    def _transform_sparse(self, X):
        indices = X.indices.copy()
        indptr = X.indptr.copy()

        data_step = np.sqrt(X.data * self.sample_interval_)
        X_step = sp.csr_matrix(
            (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
        )
        X_new = [X_step]

        log_step_nz = self.sample_interval_ * np.log(X.data)
        step_nz = 2 * X.data * self.sample_interval_

        for j in range(1, self.sample_steps):
            factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * self.sample_interval_))

            data_step = factor_nz * np.cos(j * log_step_nz)
            X_step = sp.csr_matrix(
                (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
            )
            X_new.append(X_step)

            data_step = factor_nz * np.sin(j * log_step_nz)
            X_step = sp.csr_matrix(
                (data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False
            )
            X_new.append(X_step)

        return sp.hstack(X_new)

    def _more_tags(self):
        return {"stateless": True, "requires_positive_X": True}


class Nystroem(TransformerMixin, BaseEstimator):
    """Approximate a kernel map using a subset of the training data.

    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.

    Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as `kernel_params`, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    coef0 : float, default=None
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    degree : float, default=None
        Degree of the polynomial kernel. Ignored by other kernels.

    kernel_params : dict, default=None
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    n_components : int, default=100
        Number of features to construct.
        How many data points will be used to construct the mapping.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the uniform sampling without
        replacement of `n_components` of the training data to construct the
        basis kernel.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the kernel matrix into `n_jobs` even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.24

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Subset of training points used to construct the feature map.

    component_indices_ : ndarray of shape (n_components)
        Indices of ``components_`` in the training set.

    normalization_ : ndarray of shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdditiveChi2Sampler : Approximate feature map for additive chi2 kernel.
    PolynomialCountSketch : Polynomial kernel approximation via Tensor Sketch.
    RBFSampler : Approximate a RBF kernel feature map using random Fourier
        features.
    SkewedChi2Sampler : Approximate feature map for "skewed chi-squared" kernel.
    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.

    References
    ----------
    * Williams, C.K.I. and Seeger, M.
      "Using the Nystroem method to speed up kernel machines",
      Advances in neural information processing systems 2001

    * T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
      "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
      Comparison",
      Advances in Neural Information Processing Systems 2012

    Examples
    --------
    >>> from sklearn import datasets, svm
    >>> from sklearn.kernel_approximation import Nystroem
    >>> X, y = datasets.load_digits(n_class=9, return_X_y=True)
    >>> data = X / 16.
    >>> clf = svm.LinearSVC()
    >>> feature_map_nystroem = Nystroem(gamma=.2,
    ...                                 random_state=1,
    ...                                 n_components=300)
    >>> data_transformed = feature_map_nystroem.fit_transform(data)
    >>> clf.fit(data_transformed, y)
    LinearSVC()
    >>> clf.score(data_transformed, y)
    0.9987...
    """

    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=None,
        coef0=None,
        degree=None,
        kernel_params=None,
        n_components=100,
        random_state=None,
        n_jobs=None,
    ):

        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse="csr")
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn(
                "n_components > n_samples. This is not possible.\n"
                "n_components was set to n_samples, which results"
                " in inefficient evaluation of the full kernel."
            )

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        basis_kernel = pairwise_kernels(
            basis,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            **self._get_kernel_params(),
        )

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = basis_inds
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(
            X,
            self.components_,
            metric=self.kernel,
            filter_params=True,
            n_jobs=self.n_jobs,
            **kernel_params,
        )
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel) and self.kernel != "precomputed":
            for param in KERNEL_PARAMS[self.kernel]:
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        else:
            if (
                self.gamma is not None
                or self.coef0 is not None
                or self.degree is not None
            ):
                raise ValueError(
                    "Don't pass gamma, coef0 or degree to "
                    "Nystroem if using a callable "
                    "or precomputed kernel"
                )

        return params

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_transformer_preserve_dtypes": (
                    "dtypes are preserved but not at a close enough precision"
                )
            },
            "preserves_dtype": [np.float64, np.float32],
        }
