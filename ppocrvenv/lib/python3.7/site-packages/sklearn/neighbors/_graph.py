"""Nearest Neighbors graph functions"""

# Author: Jake Vanderplas <vanderplas@astro.washington.edu>
#         Tom Dupre la Tour
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam
from ._base import KNeighborsMixin, RadiusNeighborsMixin
from ._base import NeighborsBase
from ._unsupervised import NearestNeighbors
from ..base import TransformerMixin
from ..utils.validation import check_is_fitted


def _check_params(X, metric, p, metric_params):
    """Check the validity of the input parameters"""
    params = zip(["metric", "p", "metric_params"], [metric, p, metric_params])
    est_params = X.get_params()
    for param_name, func_param in params:
        if func_param != est_params[param_name]:
            raise ValueError(
                "Got %s for %s, while the estimator has %s for the same parameter."
                % (func_param, param_name, est_params[param_name])
            )


def _query_include_self(X, include_self, mode):
    """Return the query based on include_self param"""
    if include_self == "auto":
        include_self = mode == "connectivity"

    # it does not include each sample as its own neighbors
    if not include_self:
        X = None

    return X


def kneighbors_graph(
    X,
    n_neighbors,
    *,
    mode="connectivity",
    metric="minkowski",
    p=2,
    metric_params=None,
    include_self=False,
    n_jobs=None,
):
    """Computes the (weighted) graph of k-Neighbors for points in X

    Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or BallTree
        Sample data, in the form of a numpy array or a precomputed
        :class:`BallTree`.

    n_neighbors : int
        Number of neighbors for each sample.

    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    metric : str, default='minkowski'
        The distance metric to use for the tree. The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric.
        For a list of available metrics, see the documentation of
        :class:`~sklearn.metrics.DistanceMetric`.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        additional keyword arguments for the metric function.

    include_self : bool or 'auto', default=False
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If 'auto', then True is used for mode='connectivity' and False
        for mode='distance'.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    A : sparse matrix of shape (n_samples, n_samples)
        Graph where A[i, j] is assigned the weight of edge that
        connects i to j. The matrix is of CSR format.

    Examples
    --------
    >>> X = [[0], [3], [1]]
    >>> from sklearn.neighbors import kneighbors_graph
    >>> A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
    >>> A.toarray()
    array([[1., 0., 1.],
           [0., 1., 1.],
           [1., 0., 1.]])

    See Also
    --------
    radius_neighbors_graph
    """
    if not isinstance(X, KNeighborsMixin):
        X = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        ).fit(X)
    else:
        _check_params(X, metric, p, metric_params)

    query = _query_include_self(X._fit_X, include_self, mode)
    return X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)


def radius_neighbors_graph(
    X,
    radius,
    *,
    mode="connectivity",
    metric="minkowski",
    p=2,
    metric_params=None,
    include_self=False,
    n_jobs=None,
):
    """Computes the (weighted) graph of Neighbors for points in X

    Neighborhoods are restricted the points at a distance lower than
    radius.

    Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or BallTree
        Sample data, in the form of a numpy array or a precomputed
        :class:`BallTree`.

    radius : float
        Radius of neighborhoods.

    mode : {'connectivity', 'distance'}, default='connectivity'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    metric : str, default='minkowski'
        The distance metric to use for the tree. The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric.
        For a list of available metrics, see the documentation of
        :class:`~sklearn.metrics.DistanceMetric`.

    p : int, default=2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        additional keyword arguments for the metric function.

    include_self : bool or 'auto', default=False
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If 'auto', then True is used for mode='connectivity' and False
        for mode='distance'.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    A : sparse matrix of shape (n_samples, n_samples)
        Graph where A[i, j] is assigned the weight of edge that connects
        i to j. The matrix is of CSR format.

    Examples
    --------
    >>> X = [[0], [3], [1]]
    >>> from sklearn.neighbors import radius_neighbors_graph
    >>> A = radius_neighbors_graph(X, 1.5, mode='connectivity',
    ...                            include_self=True)
    >>> A.toarray()
    array([[1., 0., 1.],
           [0., 1., 0.],
           [1., 0., 1.]])

    See Also
    --------
    kneighbors_graph
    """
    if not isinstance(X, RadiusNeighborsMixin):
        X = NearestNeighbors(
            radius=radius,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        ).fit(X)
    else:
        _check_params(X, metric, p, metric_params)

    query = _query_include_self(X._fit_X, include_self, mode)
    return X.radius_neighbors_graph(query, radius, mode)


class KNeighborsTransformer(KNeighborsMixin, TransformerMixin, NeighborsBase):
    """Transform X into a (weighted) graph of k nearest neighbors.

    The transformed data is a sparse graph as returned by kneighbors_graph.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    n_neighbors : int, default=5
        Number of neighbors for each sample in the transformed sparse graph.
        For compatibility reasons, as each sample is considered as its own
        neighbor, one extra neighbor will be computed when mode == 'distance'.
        In this case, the sparse graph contains (n_neighbors + 1) neighbors.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

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
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=1
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    kneighbors_graph : Compute the weighted graph of k-neighbors for
        points in X.
    RadiusNeighborsTransformer : Transform X into a weighted graph of
        neighbors nearer than a radius.

    Examples
    --------
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.neighbors import KNeighborsTransformer
    >>> X, _ = load_wine(return_X_y=True)
    >>> X.shape
    (178, 13)
    >>> transformer = KNeighborsTransformer(n_neighbors=5, mode='distance')
    >>> X_dist_graph = transformer.fit_transform(X)
    >>> X_dist_graph.shape
    (178, 178)
    """

    def __init__(
        self,
        *,
        mode="distance",
        n_neighbors=5,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=1,
    ):
        super(KNeighborsTransformer, self).__init__(
            n_neighbors=n_neighbors,
            radius=None,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.mode = mode

    def fit(self, X, y=None):
        """Fit the k-nearest neighbors transformer from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : KNeighborsTransformer
            The fitted k-nearest neighbors transformer.
        """
        return self._fit(X)

    def transform(self, X):
        """Compute the (weighted) graph of Neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_samples_transform, n_features)
            Sample data.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples_transform, n_samples_fit)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        check_is_fitted(self)
        add_one = self.mode == "distance"
        return self.kneighbors_graph(
            X, mode=self.mode, n_neighbors=self.n_neighbors + add_one
        )

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        return self.fit(X).transform(X)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "check is not applicable."
            }
        }


class RadiusNeighborsTransformer(RadiusNeighborsMixin, TransformerMixin, NeighborsBase):
    """Transform X into a (weighted) graph of neighbors nearer than a radius.

    The transformed data is a sparse graph as returned by
    `radius_neighbors_graph`.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    radius : float, default=1.0
        Radius of neighborhood in the transformed sparse graph.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

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
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=1
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    kneighbors_graph : Compute the weighted graph of k-neighbors for
        points in X.
    KNeighborsTransformer : Transform X into a weighted graph of k
        nearest neighbors.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.cluster import DBSCAN
    >>> from sklearn.neighbors import RadiusNeighborsTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> X, _ = load_wine(return_X_y=True)
    >>> estimator = make_pipeline(
    ...     RadiusNeighborsTransformer(radius=42.0, mode='distance'),
    ...     DBSCAN(eps=25.0, metric='precomputed'))
    >>> X_clustered = estimator.fit_predict(X)
    >>> clusters, counts = np.unique(X_clustered, return_counts=True)
    >>> print(counts)
    [ 29  15 111  11  12]
    """

    def __init__(
        self,
        *,
        mode="distance",
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=1,
    ):
        super(RadiusNeighborsTransformer, self).__init__(
            n_neighbors=None,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.mode = mode

    def fit(self, X, y=None):
        """Fit the radius neighbors transformer from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : RadiusNeighborsTransformer
            The fitted radius neighbors transformer.
        """
        return self._fit(X)

    def transform(self, X):
        """Compute the (weighted) graph of Neighbors for points in X.

        Parameters
        ----------
        X : array-like of shape (n_samples_transform, n_features)
            Sample data.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples_transform, n_samples_fit)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        check_is_fitted(self)
        return self.radius_neighbors_graph(X, mode=self.mode, sort_results=True)

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : sparse matrix of shape (n_samples, n_samples)
            Xt[i, j] is assigned the weight of edge that connects i to j.
            Only the neighbors have an explicit value.
            The diagonal is always explicit.
            The matrix is of CSR format.
        """
        return self.fit(X).transform(X)

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_methods_sample_order_invariance": "check is not applicable."
            }
        }
