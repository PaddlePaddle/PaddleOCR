"""Unsupervised nearest neighbors learner"""
from ._base import NeighborsBase
from ._base import KNeighborsMixin
from ._base import RadiusNeighborsMixin


class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
    """Unsupervised learner for implementing neighbor searches.

    Read more in the :ref:`User Guide <unsupervised_neighbors>`.

    .. versionadded:: 0.9

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, default=1.0
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

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
        The distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. For a list of available metrics, see the documentation of
        :class:`~sklearn.metrics.DistanceMetric`.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.

    p : int, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    effective_metric_ : str
        Metric used to compute distances to neighbors.

    effective_metric_params_ : dict
        Parameters for the metric used to compute distances to neighbors.

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
    KNeighborsClassifier : Classifier implementing the k-nearest neighbors
        vote.
    RadiusNeighborsClassifier : Classifier implementing a vote among neighbors
        within a given radius.
    KNeighborsRegressor : Regression based on k-nearest neighbors.
    RadiusNeighborsRegressor : Regression based on neighbors within a fixed
        radius.
    BallTree : Space partitioning data structure for organizing points in a
        multi-dimensional space, used for nearest neighbor search.

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.neighbors import NearestNeighbors
    >>> samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]

    >>> neigh = NearestNeighbors(n_neighbors=2, radius=0.4)
    >>> neigh.fit(samples)
    NearestNeighbors(...)

    >>> neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
    array([[2, 0]]...)

    >>> nbrs = neigh.radius_neighbors(
    ...    [[0, 0, 1.3]], 0.4, return_distance=False
    ... )
    >>> np.asarray(nbrs[0][0])
    array(2)
    """

    def __init__(
        self,
        *,
        n_neighbors=5,
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y=None):
        """Fit the nearest neighbors estimator from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : NearestNeighbors
            The fitted nearest neighbors estimator.
        """
        return self._fit(X)
