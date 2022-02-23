"""
Kernel Density Estimation
-------------------------
"""
# Author: Jake Vanderplas <jakevdp@cs.washington.edu>

import numpy as np
from scipy.special import gammainc
from ..base import BaseEstimator
from ..utils import check_random_state
from ..utils.validation import _check_sample_weight, check_is_fitted

from ..utils.extmath import row_norms
from ._ball_tree import BallTree, DTYPE
from ._kd_tree import KDTree


VALID_KERNELS = [
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
]
TREE_DICT = {"ball_tree": BallTree, "kd_tree": KDTree}


# TODO: implement a brute force version for testing purposes
# TODO: bandwidth estimation
# TODO: create a density estimation base class?
class KernelDensity(BaseEstimator):
    """Kernel Density Estimation.

    Read more in the :ref:`User Guide <kernel_density>`.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth of the kernel.

    algorithm : {'kd_tree', 'ball_tree', 'auto'}, default='auto'
        The tree algorithm to use.

    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', \
                 'cosine'}, default='gaussian'
        The kernel to use.

    metric : str, default='euclidean'
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms.  Refer to the documentation of
        :class:`BallTree` and :class:`KDTree` for a description of
        available algorithms.  Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.

    atol : float, default=0
        The desired absolute tolerance of the result.  A larger tolerance will
        generally lead to faster execution.

    rtol : float, default=0
        The desired relative tolerance of the result.  A larger tolerance will
        generally lead to faster execution.

    breadth_first : bool, default=True
        If true (default), use a breadth-first approach to the problem.
        Otherwise use a depth-first approach.

    leaf_size : int, default=40
        Specify the leaf size of the underlying tree.  See :class:`BallTree`
        or :class:`KDTree` for details.

    metric_params : dict, default=None
        Additional parameters to be passed to the tree for use with the
        metric.  For more information, see the documentation of
        :class:`BallTree` or :class:`KDTree`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    tree_ : ``BinaryTree`` instance
        The tree algorithm for fast generalized N-point problems.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.neighbors.KDTree : K-dimensional tree for fast generalized N-point
        problems.
    sklearn.neighbors.BallTree : Ball tree for fast generalized N-point
        problems.

    Examples
    --------
    Compute a gaussian kernel density estimate with a fixed bandwidth.

    >>> from sklearn.neighbors import KernelDensity
    >>> import numpy as np
    >>> rng = np.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:3])
    >>> log_density
    array([-1.52955942, -1.51462041, -1.60244657])
    """

    def __init__(
        self,
        *,
        bandwidth=1.0,
        algorithm="auto",
        kernel="gaussian",
        metric="euclidean",
        atol=0,
        rtol=0,
        breadth_first=True,
        leaf_size=40,
        metric_params=None,
    ):
        self.algorithm = algorithm
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

        # run the choose algorithm code so that exceptions will happen here
        # we're using clone() in the GenerativeBayes classifier,
        # so we can't do this kind of logic in __init__
        self._choose_algorithm(self.algorithm, self.metric)

        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if kernel not in VALID_KERNELS:
            raise ValueError("invalid kernel: '{0}'".format(kernel))

    def _choose_algorithm(self, algorithm, metric):
        # given the algorithm string + metric string, choose the optimal
        # algorithm to compute the result.
        if algorithm == "auto":
            # use KD Tree if possible
            if metric in KDTree.valid_metrics:
                return "kd_tree"
            elif metric in BallTree.valid_metrics:
                return "ball_tree"
            else:
                raise ValueError("invalid metric: '{0}'".format(metric))
        elif algorithm in TREE_DICT:
            if metric not in TREE_DICT[algorithm].valid_metrics:
                raise ValueError(
                    "invalid metric for {0}: '{1}'".format(TREE_DICT[algorithm], metric)
                )
            return algorithm
        else:
            raise ValueError("invalid algorithm: '{0}'".format(algorithm))

    def fit(self, X, y=None, sample_weight=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        sample_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        algorithm = self._choose_algorithm(self.algorithm, self.metric)
        X = self._validate_data(X, order="C", dtype=DTYPE)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DTYPE)
            if sample_weight.min() <= 0:
                raise ValueError("sample_weight must have positive values")

        kwargs = self.metric_params
        if kwargs is None:
            kwargs = {}
        self.tree_ = TREE_DICT[algorithm](
            X,
            metric=self.metric,
            leaf_size=self.leaf_size,
            sample_weight=sample_weight,
            **kwargs,
        )
        return self

    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        check_is_fitted(self)
        # The returned density is normalized to the number of points.
        # For it to be a probability, we must scale it.  For this reason
        # we'll also scale atol.
        X = self._validate_data(X, order="C", dtype=DTYPE, reset=False)
        if self.tree_.sample_weight is None:
            N = self.tree_.data.shape[0]
        else:
            N = self.tree_.sum_weight
        atol_N = self.atol * N
        log_density = self.tree_.kernel_density(
            X,
            h=self.bandwidth,
            kernel=self.kernel,
            atol=atol_N,
            rtol=self.rtol,
            breadth_first=self.breadth_first,
            return_log=True,
        )
        log_density -= np.log(N)
        return log_density

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        check_is_fitted(self)
        # TODO: implement sampling for other valid kernel shapes
        if self.kernel not in ["gaussian", "tophat"]:
            raise NotImplementedError()

        data = np.asarray(self.tree_.data)

        rng = check_random_state(random_state)
        u = rng.uniform(0, 1, size=n_samples)
        if self.tree_.sample_weight is None:
            i = (u * data.shape[0]).astype(np.int64)
        else:
            cumsum_weight = np.cumsum(np.asarray(self.tree_.sample_weight))
            sum_weight = cumsum_weight[-1]
            i = np.searchsorted(cumsum_weight, u * sum_weight)
        if self.kernel == "gaussian":
            return np.atleast_2d(rng.normal(data[i], self.bandwidth))

        elif self.kernel == "tophat":
            # we first draw points from a d-dimensional normal distribution,
            # then use an incomplete gamma function to map them to a uniform
            # d-dimensional tophat distribution.
            dim = data.shape[1]
            X = rng.normal(size=(n_samples, dim))
            s_sq = row_norms(X, squared=True)
            correction = (
                gammainc(0.5 * dim, 0.5 * s_sq) ** (1.0 / dim)
                * self.bandwidth
                / np.sqrt(s_sq)
            )
            return data[i] + X * correction[:, np.newaxis]

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "sample_weight must have positive values"
                ),
            }
        }
