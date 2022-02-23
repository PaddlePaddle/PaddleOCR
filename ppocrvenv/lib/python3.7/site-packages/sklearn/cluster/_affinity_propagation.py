"""Affinity Propagation clustering algorithm."""

# Author: Alexandre Gramfort alexandre.gramfort@inria.fr
#        Gael Varoquaux gael.varoquaux@normalesup.org

# License: BSD 3 clause

import numbers
import warnings

import numpy as np

from ..exceptions import ConvergenceWarning
from ..base import BaseEstimator, ClusterMixin
from ..utils import as_float_array, check_random_state
from ..utils import check_scalar
from ..utils.deprecation import deprecated
from ..utils.validation import check_is_fitted
from ..metrics import euclidean_distances
from ..metrics import pairwise_distances_argmin
from .._config import config_context


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()


def affinity_propagation(
    S,
    *,
    preference=None,
    convergence_iter=15,
    max_iter=200,
    damping=0.5,
    copy=True,
    verbose=False,
    return_n_iter=False,
    random_state=None,
):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------

    S : array-like of shape (n_samples, n_samples)
        Matrix of similarities between points.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, default=200
        Maximum number of iterations.

    damping : float, default=0.5
        Damping factor between 0.5 and 1.

    copy : bool, default=True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency.

    verbose : bool, default=False
        The verbosity level.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Returns
    -------

    cluster_centers_indices : ndarray of shape (n_clusters,)
        Index of clusters centers.

    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it returns an empty array as
    ``cluster_center_indices`` and ``-1`` as label for each training sample.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)

    preference = np.array(preference)

    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn(
            "All samples have mutually equal similarities. "
            "Returning arbitrary cluster center(s)."
        )
        if preference.flat[0] >= S.flat[n_samples - 1]:
            return (
                (np.arange(n_samples), np.arange(n_samples), 0)
                if return_n_iter
                else (np.arange(n_samples), np.arange(n_samples))
            )
        else:
            return (
                (np.array([0]), np.array([0] * n_samples), 0)
                if return_n_iter
                else (np.array([0]), np.array([0] * n_samples))
            )

    random_state = check_random_state(random_state)

    # Place preference on the diagonal of S
    S.flat[:: (n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += (
        np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100
    ) * random_state.randn(n_samples, n_samples)

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[:: n_samples + 1] = R.flat[:: n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_samples
            if (not unconverged and (K > 0)) or (it == max_iter):
                never_converged = False
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        never_converged = True
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars

    if K > 0 and not never_converged:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn(
            "Affinity propagation did not converge, this model "
            "will not have any cluster centers.",
            ConvergenceWarning,
        )
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels


###############################################################################


class AffinityPropagation(ClusterMixin, BaseEstimator):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor in the range `[0.5, 1.0)` is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, default=200
        Maximum number of iterations.

    convergence_iter : int, default=15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : bool, default=True
        Make a copy of input data.

    preference : array-like of shape (n_samples,) or float, default=None
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : {'euclidean', 'precomputed'}, default='euclidean'
        Which affinity to use. At the moment 'precomputed' and
        ``euclidean`` are supported. 'euclidean' uses the
        negative squared euclidean distance between points.

    verbose : bool, default=False
        Whether to be verbose.

    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the starting state.
        Use an int for reproducible results across function calls.
        See the :term:`Glossary <random_state>`.

        .. versionadded:: 0.23
            this parameter was previously hardcoded as 0.

    Attributes
    ----------
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of cluster centers.

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AgglomerativeClustering : Recursively merges the pair of
        clusters that minimally increases a given linkage distance.
    FeatureAgglomeration : Similar to AgglomerativeClustering,
        but recursively merges features instead of samples.
    KMeans : K-Means clustering.
    MiniBatchKMeans : Mini-Batch K-Means clustering.
    MeanShift : Mean shift clustering using a flat kernel.
    SpectralClustering : Apply clustering to a projection
        of the normalized Laplacian.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
    array and all training samples will be labelled as ``-1``. In addition,
    ``predict`` will then label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation(random_state=5).fit(X)
    >>> clustering
    AffinityPropagation(random_state=5)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])
    """

    def __init__(
        self,
        *,
        damping=0.5,
        max_iter=200,
        convergence_iter=15,
        copy=True,
        preference=None,
        affinity="euclidean",
        verbose=False,
        random_state=None,
    ):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state

    # TODO: Remove in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `_pairwise` was deprecated in "
        "version 0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def _more_tags(self):
        return {"pairwise": self.affinity == "precomputed"}

    def fit(self, X, y=None):
        """Fit the clustering from features, or affinity matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Returns the instance itself.
        """
        if self.affinity == "precomputed":
            accept_sparse = False
        else:
            accept_sparse = "csr"
        X = self._validate_data(X, accept_sparse=accept_sparse)
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError(
                "Affinity must be 'precomputed' or 'euclidean'. Got %s instead"
                % str(self.affinity)
            )

        check_scalar(
            self.damping,
            "damping",
            target_type=numbers.Real,
            min_val=0.5,
            max_val=1,
            include_boundaries="left",
        )
        check_scalar(self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1)
        check_scalar(
            self.convergence_iter,
            "convergence_iter",
            target_type=numbers.Integral,
            min_val=1,
        )

        (
            self.cluster_centers_indices_,
            self.labels_,
            self.n_iter_,
        ) = affinity_propagation(
            self.affinity_matrix_,
            preference=self.preference,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            damping=self.damping,
            copy=self.copy,
            verbose=self.verbose,
            return_n_iter=True,
            random_state=self.random_state,
        )

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, accept_sparse="csr")
        if not hasattr(self, "cluster_centers_"):
            raise ValueError(
                "Predict method is not supported when affinity='precomputed'."
            )

        if self.cluster_centers_.shape[0] > 0:
            with config_context(assume_finite=True):
                return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn(
                "This model does not have any cluster centers "
                "because affinity propagation did not converge. "
                "Labeling every sample as '-1'.",
                ConvergenceWarning,
            )
            return np.array([-1] * X.shape[0])

    def fit_predict(self, X, y=None):
        """Fit clustering from features/affinity matrix; return cluster labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
                array-like of shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
