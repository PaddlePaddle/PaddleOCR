# -*- coding: utf-8 -*-
"""Ordering Points To Identify the Clustering Structure (OPTICS)

These routines execute the OPTICS algorithm, and implement various
cluster extraction methods of the ordered list.

Authors: Shane Grigsby <refuge@rocktalus.com>
         Adrin Jalali <adrinjalali@gmail.com>
         Erich Schubert <erich@debian.org>
         Hanmin Qin <qinhanmin2005@sina.com>
License: BSD 3 clause
"""

import warnings
import numpy as np

from ..exceptions import DataConversionWarning
from ..metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS
from ..utils import gen_batches, get_chunk_n_rows
from ..utils.validation import check_memory
from ..neighbors import NearestNeighbors
from ..base import BaseEstimator, ClusterMixin
from ..metrics import pairwise_distances


class OPTICS(ClusterMixin, BaseEstimator):
    """Estimate clustering structure from vector array.

    OPTICS (Ordering Points To Identify the Clustering Structure), closely
    related to DBSCAN, finds core sample of high density and expands clusters
    from them [1]_. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large datasets than the
    current sklearn implementation of DBSCAN.

    Clusters are then extracted using a DBSCAN-like method
    (cluster_method = 'dbscan') or an automatic
    technique proposed in [1]_ (cluster_method = 'xi').

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes, then
    computing only the distances to unprocessed points when constructing the
    cluster order. Note that we do not employ a heap to manage the expansion
    candidates, so the time complexity will be O(n^2).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int > 1 or float between 0 and 1, default=5
        The number of samples in a neighborhood for a point to be considered as
        a core point. Also, up and down steep regions can't have more than
        ``min_samples`` consecutive non-steep points. Expressed as an absolute
        number or a fraction of the number of samples (rounded to be at least
        2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", X is assumed to be a distance matrix and must be square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : int, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    cluster_method : str, default='xi'
        The extraction method used to extract clusters using the calculated
        reachability and ordering. Possible values are "xi" and "dbscan".

    eps : float, default=None
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. By default it assumes the same value
        as ``max_eps``.
        Used only when ``cluster_method='dbscan'``.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.
        Used only when ``cluster_method='xi'``.

    predecessor_correction : bool, default=True
        Correct clusters according to the predecessors calculated by OPTICS
        [2]_. This parameter has minimal effect on most datasets.
        Used only when ``cluster_method='xi'``.

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.
        Used only when ``cluster_method='xi'``.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method. (default)

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples and points which are not included in a leaf cluster
        of ``cluster_hierarchy_`` are labeled as -1.

    reachability_ : ndarray of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    ordering_ : ndarray of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    predecessor_ : ndarray of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    cluster_hierarchy_ : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to
        ``(end, -start)`` (ascending) so that larger clusters encompassing
        smaller clusters come after those smaller ones. Since ``labels_`` does
        not reflect the hierarchy, usually
        ``len(cluster_hierarchy_) > np.unique(optics.labels_)``. Please also
        note that these indices are of the ``ordering_``, i.e.
        ``X[ordering_][start:end + 1]`` form a cluster.
        Only available when ``cluster_method='xi'``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DBSCAN : A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.

    .. [2] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.

    Examples
    --------
    >>> from sklearn.cluster import OPTICS
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 5], [3, 6],
    ...               [8, 7], [8, 8], [7, 3]])
    >>> clustering = OPTICS(min_samples=2).fit(X)
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    """

    def __init__(
        self,
        *,
        min_samples=5,
        max_eps=np.inf,
        metric="minkowski",
        p=2,
        metric_params=None,
        cluster_method="xi",
        eps=None,
        xi=0.05,
        predecessor_correction=True,
        min_cluster_size=None,
        algorithm="auto",
        leaf_size=30,
        memory=None,
        n_jobs=None,
    ):
        self.max_eps = max_eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.memory = memory
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Perform OPTICS clustering.

        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using ``max_eps`` distance specified at
        OPTICS object instantiation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), or \
                (n_samples, n_samples) if metric=’precomputed’
            A feature array, or array of distances between samples if
            metric='precomputed'.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        dtype = bool if self.metric in PAIRWISE_BOOLEAN_FUNCTIONS else float
        if dtype == bool and X.dtype != bool:
            msg = (
                "Data will be converted to boolean for"
                f" metric {self.metric}, to avoid this warning,"
                " you may convert the data prior to calling fit."
            )
            warnings.warn(msg, DataConversionWarning)

        X = self._validate_data(X, dtype=dtype)
        memory = check_memory(self.memory)

        if self.cluster_method not in ["dbscan", "xi"]:
            raise ValueError(
                "cluster_method should be one of 'dbscan' or 'xi' but is %s"
                % self.cluster_method
            )

        (
            self.ordering_,
            self.core_distances_,
            self.reachability_,
            self.predecessor_,
        ) = memory.cache(compute_optics_graph)(
            X=X,
            min_samples=self.min_samples,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            p=self.p,
            n_jobs=self.n_jobs,
            max_eps=self.max_eps,
        )

        # Extract clusters from the calculated orders and reachability
        if self.cluster_method == "xi":
            labels_, clusters_ = cluster_optics_xi(
                reachability=self.reachability_,
                predecessor=self.predecessor_,
                ordering=self.ordering_,
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                xi=self.xi,
                predecessor_correction=self.predecessor_correction,
            )
            self.cluster_hierarchy_ = clusters_
        elif self.cluster_method == "dbscan":
            if self.eps is None:
                eps = self.max_eps
            else:
                eps = self.eps

            if eps > self.max_eps:
                raise ValueError(
                    "Specify an epsilon smaller than %s. Got %s." % (self.max_eps, eps)
                )

            labels_ = cluster_optics_dbscan(
                reachability=self.reachability_,
                core_distances=self.core_distances_,
                ordering=self.ordering_,
                eps=eps,
            )

        self.labels_ = labels_
        return self


def _validate_size(size, n_samples, param_name):
    if size <= 0 or (size != int(size) and size > 1):
        raise ValueError(
            "%s must be a positive integer or a float between 0 and 1. Got %r"
            % (param_name, size)
        )
    elif size > n_samples:
        raise ValueError(
            "%s must be no greater than the number of samples (%d). Got %d"
            % (param_name, n_samples, size)
        )


# OPTICS helper functions
def _compute_core_distances_(X, neighbors, min_samples, working_memory):
    """Compute the k-th nearest neighbor of each sample.

    Equivalent to neighbors.kneighbors(X, self.min_samples)[0][:, -1]
    but with more memory efficiency.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.
    neighbors : NearestNeighbors instance
        The fitted nearest neighbors estimator.
    working_memory : int, default=None
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    Returns
    -------
    core_distances : ndarray of shape (n_samples,)
        Distance at which each sample becomes a core point.
        Points which will never be core have a distance of inf.
    """
    n_samples = X.shape[0]
    core_distances = np.empty(n_samples)
    core_distances.fill(np.nan)

    chunk_n_rows = get_chunk_n_rows(
        row_bytes=16 * min_samples, max_n_rows=n_samples, working_memory=working_memory
    )
    slices = gen_batches(n_samples, chunk_n_rows)
    for sl in slices:
        core_distances[sl] = neighbors.kneighbors(X[sl], min_samples)[0][:, -1]
    return core_distances


def compute_optics_graph(
    X, *, min_samples, max_eps, metric, p, metric_params, algorithm, leaf_size, n_jobs
):
    """Compute the OPTICS reachability graph.

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), or \
            (n_samples, n_samples) if metric=’precomputed’.
        A feature array, or array of distances between samples if
        metric='precomputed'

    min_samples : int > 1 or float between 0 and 1
        The number of samples in a neighborhood for a point to be considered
        as a core point. Expressed as an absolute number or a fraction of the
        number of samples (rounded to be at least 2).

    max_eps : float, default=np.inf
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. Default value of ``np.inf`` will
        identify clusters across all scales; reducing ``max_eps`` will result
        in shorter run times.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string. If metric is
        "precomputed", X is assumed to be a distance matrix and must be square.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : int, default=2
        Parameter for the Minkowski metric from
        :class:`~sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method. (default)

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    ordering_ : array of shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : array of shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    reachability_ : array of shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    predecessor_ : array of shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    References
    ----------
    .. [1] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel,
       and Jörg Sander. "OPTICS: ordering points to identify the clustering
       structure." ACM SIGMOD Record 28, no. 2 (1999): 49-60.
    """
    n_samples = X.shape[0]
    _validate_size(min_samples, n_samples, "min_samples")
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))

    # Start all points as 'unprocessed' ##
    reachability_ = np.empty(n_samples)
    reachability_.fill(np.inf)
    predecessor_ = np.empty(n_samples, dtype=int)
    predecessor_.fill(-1)

    nbrs = NearestNeighbors(
        n_neighbors=min_samples,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        metric_params=metric_params,
        p=p,
        n_jobs=n_jobs,
    )

    nbrs.fit(X)
    # Here we first do a kNN query for each point, this differs from
    # the original OPTICS that only used epsilon range queries.
    # TODO: handle working_memory somehow?
    core_distances_ = _compute_core_distances_(
        X=X, neighbors=nbrs, min_samples=min_samples, working_memory=None
    )
    # OPTICS puts an upper limit on these, use inf for undefined.
    core_distances_[core_distances_ > max_eps] = np.inf
    np.around(
        core_distances_,
        decimals=np.finfo(core_distances_.dtype).precision,
        out=core_distances_,
    )

    # Main OPTICS loop. Not parallelizable. The order that entries are
    # written to the 'ordering_' list is important!
    # Note that this implementation is O(n^2) theoretically, but
    # supposedly with very low constant factors.
    processed = np.zeros(X.shape[0], dtype=bool)
    ordering = np.zeros(X.shape[0], dtype=int)
    for ordering_idx in range(X.shape[0]):
        # Choose next based on smallest reachability distance
        # (And prefer smaller ids on ties, possibly np.inf!)
        index = np.where(processed == 0)[0]
        point = index[np.argmin(reachability_[index])]

        processed[point] = True
        ordering[ordering_idx] = point
        if core_distances_[point] != np.inf:
            _set_reach_dist(
                core_distances_=core_distances_,
                reachability_=reachability_,
                predecessor_=predecessor_,
                point_index=point,
                processed=processed,
                X=X,
                nbrs=nbrs,
                metric=metric,
                metric_params=metric_params,
                p=p,
                max_eps=max_eps,
            )
    if np.all(np.isinf(reachability_)):
        warnings.warn(
            "All reachability values are inf. Set a larger"
            " max_eps or all data will be considered outliers.",
            UserWarning,
        )
    return ordering, core_distances_, reachability_, predecessor_


def _set_reach_dist(
    core_distances_,
    reachability_,
    predecessor_,
    point_index,
    processed,
    X,
    nbrs,
    metric,
    metric_params,
    p,
    max_eps,
):
    P = X[point_index : point_index + 1]
    # Assume that radius_neighbors is faster without distances
    # and we don't need all distances, nevertheless, this means
    # we may be doing some work twice.
    indices = nbrs.radius_neighbors(P, radius=max_eps, return_distance=False)[0]

    # Getting indices of neighbors that have not been processed
    unproc = np.compress(~np.take(processed, indices), indices)
    # Neighbors of current point are already processed.
    if not unproc.size:
        return

    # Only compute distances to unprocessed neighbors:
    if metric == "precomputed":
        dists = X[point_index, unproc]
    else:
        _params = dict() if metric_params is None else metric_params.copy()
        if metric == "minkowski" and "p" not in _params:
            # the same logic as neighbors, p is ignored if explicitly set
            # in the dict params
            _params["p"] = p
        dists = pairwise_distances(
            P, np.take(X, unproc, axis=0), metric=metric, n_jobs=None, **_params
        ).ravel()

    rdists = np.maximum(dists, core_distances_[point_index])
    np.around(rdists, decimals=np.finfo(rdists.dtype).precision, out=rdists)
    improved = np.where(rdists < np.take(reachability_, unproc))
    reachability_[unproc[improved]] = rdists[improved]
    predecessor_[unproc[improved]] = point_index


def cluster_optics_dbscan(*, reachability, core_distances, ordering, eps):
    """Perform DBSCAN extraction for an arbitrary epsilon.

    Extracting the clusters runs in linear time. Note that this results in
    ``labels_`` which are close to a :class:`~sklearn.cluster.DBSCAN` with
    similar settings and ``eps``, only if ``eps`` is close to ``max_eps``.

    Parameters
    ----------
    reachability : array of shape (n_samples,)
        Reachability distances calculated by OPTICS (``reachability_``).

    core_distances : array of shape (n_samples,)
        Distances at which points become core (``core_distances_``).

    ordering : array of shape (n_samples,)
        OPTICS ordered point indices (``ordering_``).

    eps : float
        DBSCAN ``eps`` parameter. Must be set to < ``max_eps``. Results
        will be close to DBSCAN algorithm if ``eps`` and ``max_eps`` are close
        to one another.

    Returns
    -------
    labels_ : array of shape (n_samples,)
        The estimated labels.
    """
    n_samples = len(core_distances)
    labels = np.zeros(n_samples, dtype=int)

    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    return labels


def cluster_optics_xi(
    *,
    reachability,
    predecessor,
    ordering,
    min_samples,
    min_cluster_size=None,
    xi=0.05,
    predecessor_correction=True,
):
    """Automatically extract clusters according to the Xi-steep method.

    Parameters
    ----------
    reachability : ndarray of shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`)

    predecessor : ndarray of shape (n_samples,)
        Predecessors calculated by OPTICS.

    ordering : ndarray of shape (n_samples,)
        OPTICS ordered point indices (`ordering_`)

    min_samples : int > 1 or float between 0 and 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.
        Expressed as an absolute number or a fraction of the number of samples
        (rounded to be at least 2).

    min_cluster_size : int > 1 or float between 0 and 1, default=None
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded to be
        at least 2). If ``None``, the value of ``min_samples`` is used instead.

    xi : float between 0 and 1, default=0.05
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    predecessor_correction : bool, default=True
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The labels assigned to samples. Points which are not included
        in any cluster are labeled as -1.

    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of ``[start, end]`` in each row, with
        all indices inclusive. The clusters are ordered according to ``(end,
        -start)`` (ascending) so that larger clusters encompassing smaller
        clusters come after such nested smaller clusters. Since ``labels`` does
        not reflect the hierarchy, usually ``len(clusters) >
        np.unique(labels)``.
    """
    n_samples = len(reachability)
    _validate_size(min_samples, n_samples, "min_samples")
    if min_samples <= 1:
        min_samples = max(2, int(min_samples * n_samples))
    if min_cluster_size is None:
        min_cluster_size = min_samples
    _validate_size(min_cluster_size, n_samples, "min_cluster_size")
    if min_cluster_size <= 1:
        min_cluster_size = max(2, int(min_cluster_size * n_samples))

    clusters = _xi_cluster(
        reachability[ordering],
        predecessor[ordering],
        ordering,
        xi,
        min_samples,
        min_cluster_size,
        predecessor_correction,
    )
    labels = _extract_xi_labels(ordering, clusters)
    return labels, clusters


def _extend_region(steep_point, xward_point, start, min_samples):
    """Extend the area until it's maximal.

    It's the same function for both upward and downward reagions, depending on
    the given input parameters. Assuming:

        - steep_{upward/downward}: bool array indicating whether a point is a
          steep {upward/downward};
        - upward/downward: bool array indicating whether a point is
          upward/downward;

    To extend an upward reagion, ``steep_point=steep_upward`` and
    ``xward_point=downward`` are expected, and to extend a downward region,
    ``steep_point=steep_downward`` and ``xward_point=upward``.

    Parameters
    ----------
    steep_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is steep downward (upward).

    xward_point : ndarray of shape (n_samples,), dtype=bool
        True if the point is an upward (respectively downward) point.

    start : int
        The start of the xward region.

    min_samples : int
       The same as the min_samples given to OPTICS. Up and down steep
       regions can't have more then ``min_samples`` consecutive non-steep
       points.

    Returns
    -------
    index : int
        The current index iterating over all the samples, i.e. where we are up
        to in our search.

    end : int
        The end of the region, which can be behind the index. The region
        includes the ``end`` index.
    """
    n_samples = len(steep_point)
    non_xward_points = 0
    index = start
    end = start
    # find a maximal area
    while index < n_samples:
        if steep_point[index]:
            non_xward_points = 0
            end = index
        elif not xward_point[index]:
            # it's not a steep point, but still goes up.
            non_xward_points += 1
            # region should include no more than min_samples consecutive
            # non steep xward points.
            if non_xward_points > min_samples:
                break
        else:
            return end
        index += 1
    return end


def _update_filter_sdas(sdas, mib, xi_complement, reachability_plot):
    """Update steep down areas (SDAs) using the new maximum in between (mib)
    value, and the given complement of xi, i.e. ``1 - xi``.
    """
    if np.isinf(mib):
        return []
    res = [
        sda for sda in sdas if mib <= reachability_plot[sda["start"]] * xi_complement
    ]
    for sda in res:
        sda["mib"] = max(sda["mib"], mib)
    return res


def _correct_predecessor(reachability_plot, predecessor_plot, ordering, s, e):
    """Correct for predecessors.

    Applies Algorithm 2 of [1]_.

    Input parameters are ordered by the computer OPTICS ordering.

    .. [1] Schubert, Erich, Michael Gertz.
       "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
       the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.
    """
    while s < e:
        if reachability_plot[s] > reachability_plot[e]:
            return s, e
        p_e = ordering[predecessor_plot[e]]
        for i in range(s, e):
            if p_e == ordering[i]:
                return s, e
        e -= 1
    return None, None


def _xi_cluster(
    reachability_plot,
    predecessor_plot,
    ordering,
    xi,
    min_samples,
    min_cluster_size,
    predecessor_correction,
):
    """Automatically extract clusters according to the Xi-steep method.

    This is rouphly an implementation of Figure 19 of the OPTICS paper.

    Parameters
    ----------
    reachability_plot : array-like of shape (n_samples,)
        The reachability plot, i.e. reachability ordered according to
        the calculated ordering, all computed by OPTICS.

    predecessor_plot : array-like of shape (n_samples,)
        Predecessors ordered according to the calculated ordering.

    xi : float, between 0 and 1
        Determines the minimum steepness on the reachability plot that
        constitutes a cluster boundary. For example, an upwards point in the
        reachability plot is defined by the ratio from one point to its
        successor being at most 1-xi.

    min_samples : int > 1
        The same as the min_samples given to OPTICS. Up and down steep regions
        can't have more then ``min_samples`` consecutive non-steep points.

    min_cluster_size : int > 1
        Minimum number of samples in an OPTICS cluster.

    predecessor_correction : bool
        Correct clusters based on the calculated predecessors.

    Returns
    -------
    clusters : ndarray of shape (n_clusters, 2)
        The list of clusters in the form of [start, end] in each row, with all
        indices inclusive. The clusters are ordered in a way that larger
        clusters encompassing smaller clusters come after those smaller
        clusters.
    """

    # Our implementation adds an inf to the end of reachability plot
    # this helps to find potential clusters at the end of the
    # reachability plot even if there's no upward region at the end of it.
    reachability_plot = np.hstack((reachability_plot, np.inf))

    xi_complement = 1 - xi
    sdas = []  # steep down areas, introduced in section 4.3.2 of the paper
    clusters = []
    index = 0
    mib = 0.0  # maximum in between, section 4.3.2

    # Our implementation corrects a mistake in the original
    # paper, i.e., in Definition 9 steep downward point,
    # r(p) * (1 - x1) <= r(p + 1) should be
    # r(p) * (1 - x1) >= r(p + 1)
    with np.errstate(invalid="ignore"):
        ratio = reachability_plot[:-1] / reachability_plot[1:]
        steep_upward = ratio <= xi_complement
        steep_downward = ratio >= 1 / xi_complement
        downward = ratio > 1
        upward = ratio < 1

    # the following loop is is almost exactly as Figure 19 of the paper.
    # it jumps over the areas which are not either steep down or up areas
    for steep_index in iter(np.flatnonzero(steep_upward | steep_downward)):
        # just continue if steep_index has been a part of a discovered xward
        # area.
        if steep_index < index:
            continue

        mib = max(mib, np.max(reachability_plot[index : steep_index + 1]))

        # steep downward areas
        if steep_downward[steep_index]:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            D_start = steep_index
            D_end = _extend_region(steep_downward, upward, D_start, min_samples)
            D = {"start": D_start, "end": D_end, "mib": 0.0}
            sdas.append(D)
            index = D_end + 1
            mib = reachability_plot[index]

        # steep upward areas
        else:
            sdas = _update_filter_sdas(sdas, mib, xi_complement, reachability_plot)
            U_start = steep_index
            U_end = _extend_region(steep_upward, downward, U_start, min_samples)
            index = U_end + 1
            mib = reachability_plot[index]

            U_clusters = []
            for D in sdas:
                c_start = D["start"]
                c_end = U_end

                # line (**), sc2*
                if reachability_plot[c_end + 1] * xi_complement < D["mib"]:
                    continue

                # Definition 11: criterion 4
                D_max = reachability_plot[D["start"]]
                if D_max * xi_complement >= reachability_plot[c_end + 1]:
                    # Find the first index from the left side which is almost
                    # at the same level as the end of the detected cluster.
                    while (
                        reachability_plot[c_start + 1] > reachability_plot[c_end + 1]
                        and c_start < D["end"]
                    ):
                        c_start += 1
                elif reachability_plot[c_end + 1] * xi_complement >= D_max:
                    # Find the first index from the right side which is almost
                    # at the same level as the beginning of the detected
                    # cluster.
                    # Our implementation corrects a mistake in the original
                    # paper, i.e., in Definition 11 4c, r(x) < r(sD) should be
                    # r(x) > r(sD).
                    while reachability_plot[c_end - 1] > D_max and c_end > U_start:
                        c_end -= 1

                # predecessor correction
                if predecessor_correction:
                    c_start, c_end = _correct_predecessor(
                        reachability_plot, predecessor_plot, ordering, c_start, c_end
                    )
                if c_start is None:
                    continue

                # Definition 11: criterion 3.a
                if c_end - c_start + 1 < min_cluster_size:
                    continue

                # Definition 11: criterion 1
                if c_start > D["end"]:
                    continue

                # Definition 11: criterion 2
                if c_end < U_start:
                    continue

                U_clusters.append((c_start, c_end))

            # add smaller clusters first.
            U_clusters.reverse()
            clusters.extend(U_clusters)

    return np.array(clusters)


def _extract_xi_labels(ordering, clusters):
    """Extracts the labels from the clusters returned by `_xi_cluster`.
    We rely on the fact that clusters are stored
    with the smaller clusters coming before the larger ones.

    Parameters
    ----------
    ordering : array-like of shape (n_samples,)
        The ordering of points calculated by OPTICS

    clusters : array-like of shape (n_clusters, 2)
        List of clusters i.e. (start, end) tuples,
        as returned by `_xi_cluster`.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """

    labels = np.full(len(ordering), -1, dtype=int)
    label = 0
    for c in clusters:
        if not np.any(labels[c[0] : (c[1] + 1)] != -1):
            labels[c[0] : (c[1] + 1)] = label
            label += 1
    labels[ordering] = labels.copy()
    return labels
