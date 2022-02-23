# Copyright Anne M. Archibald 2008
# Released under the scipy license
import numpy as np
import warnings
from .ckdtree import cKDTree, cKDTreeNode

__all__ = ['minkowski_distance_p', 'minkowski_distance',
           'distance_matrix',
           'Rectangle', 'KDTree']


def minkowski_distance_p(x, y, p=2):
    """Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """Compute the L**p distance between two arrays.

    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


class Rectangle:
    """Hyperrectangle class.

    Represents a Cartesian product of intervals.
    """
    def __init__(self, maxes, mins):
        """Construct a hyperrectangle."""
        self.maxes = np.maximum(maxes,mins).astype(float)
        self.mins = np.minimum(maxes,mins).astype(float)
        self.m, = self.maxes.shape

    def __repr__(self):
        return "<Rectangle %s>" % list(zip(self.mins, self.maxes))

    def volume(self):
        """Total volume."""
        return np.prod(self.maxes-self.mins)

    def split(self, d, split):
        """Produce two hyperrectangles by splitting.

        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.

        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.

        """
        mid = np.copy(self.maxes)
        mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = Rectangle(mid, self.maxes)
        return less, greater

    def min_distance_point(self, x, p=2.):
        """
        Return the minimum distance between input and points in the
        hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(
            0, np.maximum(0, np.maximum(self.mins-x, x-self.maxes)),
            p
        )

    def max_distance_point(self, x, p=2.):
        """
        Return the maximum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(self.maxes-x, x-self.mins), p)

    def min_distance_rectangle(self, other, p=2.):
        """
        Compute the minimum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.

        """
        return minkowski_distance(
            0,
            np.maximum(0, np.maximum(self.mins-other.maxes,
                                     other.mins-self.maxes)),
            p
        )

    def max_distance_rectangle(self, other, p=2.):
        """
        Compute the maximum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(
            0, np.maximum(self.maxes-other.mins, other.maxes-self.mins), p)


class KDTree(cKDTree):
    """kd-tree for quick nearest-neighbor lookup.

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Default: 10.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.

    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with `copy_data=True`.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    size : int
        The number of nodes in the tree.

    """

    class node:
        @staticmethod
        def _create(ckdtree_node=None):
            """Create either an inner or leaf node, wrapping a cKDTreeNode instance"""
            if ckdtree_node is None:
                return KDTree.node(ckdtree_node)
            elif ckdtree_node.split_dim == -1:
                return KDTree.leafnode(ckdtree_node)
            else:
                return KDTree.innernode(ckdtree_node)

        def __init__(self, ckdtree_node=None):
            if ckdtree_node is None:
                ckdtree_node = cKDTreeNode()
            self._node = ckdtree_node

        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        @property
        def idx(self):
            return self._node.indices

        @property
        def children(self):
            return self._node.children

    class innernode(node):
        def __init__(self, ckdtreenode):
            assert isinstance(ckdtreenode, cKDTreeNode)
            super().__init__(ckdtreenode)
            self.less = KDTree.node._create(ckdtreenode.lesser)
            self.greater = KDTree.node._create(ckdtreenode.greater)

        @property
        def split_dim(self):
            return self._node.split_dim

        @property
        def split(self):
            return self._node.split

        @property
        def children(self):
            return self._node.children

    @property
    def tree(self):
        if not hasattr(self, "_tree"):
            self._tree = KDTree.node._create(super().tree)

        return self._tree

    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False,
                 balanced_tree=True, boxsize=None):
        data = np.asarray(data)
        if data.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        # Note KDTree has different default leafsize from cKDTree
        super().__init__(data, leafsize, compact_nodes, copy_data,
                         balanced_tree, boxsize)

    def query(
            self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1):
        r"""Query the kd-tree for nearest neighbors.

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : int or Sequence[int], optional
            Either the number of nearest neighbors to return, or a list of the
            k-th nearest neighbors to return, starting from 1.
        eps : nonnegative float, optional
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values distance ("Manhattan" distance).
            2 is the usual Euclidean distance.
            infinity is the maximum-coordinate-difference distance.
            A large, finite p may cause a ValueError if overflow can occur.
        distance_upper_bound : nonnegative float, optional
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is given
            all CPU threads are used. Default: 1.

            .. versionadded:: 1.6.0

        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape
            ``tuple+(k,)``.
            When k == 1, the last dimension of the output is squeezed.
            Missing neighbors are indicated with infinite distances.
            Hits are sorted by distance (nearest first).

            .. deprecated:: 1.6.0
               If ``k=None``, then ``d`` is an object array of shape ``tuple``,
               containing lists of distances. This behavior is deprecated and
               will be removed in SciPy 1.8.0, use ``query_ball_point``
               instead.

        i : integer or array of integers
            The index of each neighbor in ``self.data``.
            ``i`` is the same shape as d.
            Missing neighbors are indicated with ``self.n``.

        Examples
        --------

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> x, y = np.mgrid[0:5, 2:8]
        >>> tree = KDTree(np.c_[x.ravel(), y.ravel()])

        To query the nearest neighbours and return squeezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
        >>> print(dd, ii, sep='\n')
        [2.         0.2236068]
        [ 0 13]

        To query the nearest neighbours and return unsqueezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])
        >>> print(dd, ii, sep='\n')
        [[2.        ]
         [0.2236068]]
        [[ 0]
         [13]]

        To query the second nearest neighbours and return unsqueezed result,
        use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])
        >>> print(dd, ii, sep='\n')
        [[2.23606798]
         [0.80622577]]
        [[ 6]
         [19]]

        To query the first and second nearest neighbours, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        or, be more specific

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        """
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        if k is None:
            # k=None, return all neighbors
            warnings.warn(
                "KDTree.query with k=None is deprecated and will be removed "
                "in SciPy 1.8.0. Use KDTree.query_ball_point instead.",
                DeprecationWarning)

            # Convert index query to a lists of distance and index,
            # sorted by distance
            def inds_to_hits(point, neighbors):
                dist = minkowski_distance(point, self.data[neighbors], p)
                hits = sorted([(d, i) for d, i in zip(dist, neighbors)])
                return [d for d, i in hits], [i for d, i in hits]

            x = np.asarray(x, dtype=np.float64)
            inds = super().query_ball_point(
                x, distance_upper_bound, p, eps, workers)

            if isinstance(inds, list):
                return inds_to_hits(x, inds)

            dists = np.empty_like(inds)
            for idx in np.ndindex(inds.shape):
                dists[idx], inds[idx] = inds_to_hits(x[idx], inds[idx])

            return dists, inds

        d, i = super().query(x, k, eps, p, distance_upper_bound, workers)
        if isinstance(i, int):
            i = np.intp(i)
        return d, i

    def query_ball_point(self, x, r, p=2., eps=0, workers=1,
                         return_sorted=None, return_length=False):
        """Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : array_like, float
            The radius of points to return, must broadcast to the length of x.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
            A finite large p may cause a ValueError if overflow can occur.
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        workers : int, optional
            Number of jobs to schedule for parallel processing. If -1 is given
            all processors are used. Default: 1.

            .. versionadded:: 1.6.0
        return_sorted : bool, optional
            Sorts returned indicies if True and does not sort them if False. If
            None, does not sort single point queries, but does sort
            multi-point queries which was the behavior before this option
            was added.

            .. versionadded:: 1.6.0
        return_length: bool, optional
            Return the number of points inside the radius instead of a list
            of the indices.

            .. versionadded:: 1.6.0

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.

        Examples
        --------
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:5, 0:5]
        >>> points = np.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> sorted(tree.query_ball_point([2, 0], 1))
        [5, 10, 11, 15]

        Query multiple points and plot the results:

        >>> import matplotlib.pyplot as plt
        >>> points = np.asarray(points)
        >>> plt.plot(points[:,0], points[:,1], '.')
        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
        ...     nearby_points = points[results]
        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
        >>> plt.margins(0.1, 0.1)
        >>> plt.show()

        """
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")
        return super().query_ball_point(
            x, r, p, eps, workers, return_sorted, return_length)

    def query_ball_tree(self, other, r, p=2., eps=0):
        """
        Find all pairs of points between `self` and `other` whose distance is
        at most r.

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        Examples
        --------
        You can search all pairs of points between two kd-trees within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((15, 2))
        >>> points2 = rng.random((15, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> for i in range(len(indexes)):
        ...     for j in indexes[i]:
        ...         plt.plot([points1[i, 0], points2[j, 0]],
        ...             [points1[i, 1], points2[j, 1]], "-r")
        >>> plt.show()

        """
        return super().query_ball_tree(other, r, p, eps)

    def query_pairs(self, r, p=2., eps=0, output_type='set'):
        """Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'ndarray'. Default: 'set'

            .. versionadded:: 1.6.0

        Returns
        -------
        results : set or ndarray
            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close. If output_type is 'ndarray', an ndarry is
            returned instead of a set.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points = rng.random((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = KDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """
        return super().query_pairs(r, p, eps, output_type)

    def count_neighbors(self, other, r, p=2., weights=None, cumulative=True):
        """Count how many nearby pairs can be formed.

        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn
        from ``self`` and ``x2`` drawn from ``other``, and where
        ``distance(x1, x2, p) <= r``.

        Data points on ``self`` and ``other`` are optionally weighted by the
        ``weights`` argument. (See below)

        This is adapted from the "two-point correlation" algorithm described by
        Gray and Moore [1]_.  See notes for further discussion.

        Parameters
        ----------
        other : KDTree
            The other tree to draw points from, can be the same tree as self.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
            If the count is non-cumulative(``cumulative=False``), ``r`` defines
            the edges of the bins, and must be non-decreasing.
        p : float, optional
            1<=p<=infinity.
            Which Minkowski p-norm to use.
            Default 2.0.
            A finite large p may cause a ValueError if overflow can occur.
        weights : tuple, array_like, or None, optional
            If None, the pair-counting is unweighted.
            If given as a tuple, weights[0] is the weights of points in
            ``self``, and weights[1] is the weights of points in ``other``;
            either can be None to indicate the points are unweighted.
            If given as an array_like, weights is the weights of points in
            ``self`` and ``other``. For this to make sense, ``self`` and
            ``other`` must be the same tree. If ``self`` and ``other`` are two
            different trees, a ``ValueError`` is raised.
            Default: None

            .. versionadded:: 1.6.0
        cumulative : bool, optional
            Whether the returned counts are cumulative. When cumulative is set
            to ``False`` the algorithm is optimized to work with a large number
            of bins (>10) specified by ``r``. When ``cumulative`` is set to
            True, the algorithm is optimized to work with a small number of
            ``r``. Default: True

            .. versionadded:: 1.6.0

        Returns
        -------
        result : scalar or 1-D array
            The number of pairs. For unweighted counts, the result is integer.
            For weighted counts, the result is float.
            If cumulative is False, ``result[i]`` contains the counts with
            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

        Notes
        -----
        Pair-counting is the basic operation used to calculate the two point
        correlation functions from a data set composed of position of objects.

        Two point correlation function measures the clustering of objects and
        is widely used in cosmology to quantify the large scale structure
        in our Universe, but it may be useful for data analysis in other fields
        where self-similar assembly of objects also occur.

        The Landy-Szalay estimator for the two point correlation function of
        ``D`` measures the clustering signal in ``D``. [2]_

        For example, given the position of two sets of objects,

        - objects ``D`` (data) contains the clustering signal, and

        - objects ``R`` (random) that contains no signal,

        .. math::

             \\xi(r) = \\frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},

        where the brackets represents counting pairs between two data sets
        in a finite bin around ``r`` (distance), corresponding to setting
        `cumulative=False`, and ``f = float(len(D)) / float(len(R))`` is the
        ratio between number of objects from data and random.

        The algorithm implemented here is loosely based on the dual-tree
        algorithm described in [1]_. We switch between two different
        pair-cumulation scheme depending on the setting of ``cumulative``.
        The computing time of the method we use when for
        ``cumulative == False`` does not scale with the total number of bins.
        The algorithm for ``cumulative == True`` scales linearly with the
        number of bins, though it is slightly faster when only
        1 or 2 bins are used. [5]_.

        As an extension to the naive pair-counting,
        weighted pair-counting counts the product of weights instead
        of number of pairs.
        Weighted pair-counting is used to estimate marked correlation functions
        ([3]_, section 2.2),
        or to properly calculate the average of data per distance bin
        (e.g. [4]_, section 2.1 on redshift).

        .. [1] Gray and Moore,
               "N-body problems in statistical learning",
               Mining the sky, 2000,
               https://arxiv.org/abs/astro-ph/0012333

        .. [2] Landy and Szalay,
               "Bias and variance of angular correlation functions",
               The Astrophysical Journal, 1993,
               http://adsabs.harvard.edu/abs/1993ApJ...412...64L

        .. [3] Sheth, Connolly and Skibba,
               "Marked correlations in galaxy formation models",
               Arxiv e-print, 2005,
               https://arxiv.org/abs/astro-ph/0511773

        .. [4] Hawkins, et al.,
               "The 2dF Galaxy Redshift Survey: correlation functions,
               peculiar velocities and the matter density of the Universe",
               Monthly Notices of the Royal Astronomical Society, 2002,
               http://adsabs.harvard.edu/abs/2003MNRAS.346...78H

        .. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926

        Examples
        --------
        You can count neighbors number between two kd-trees within a distance:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> kd_tree1.count_neighbors(kd_tree2, 0.2)
        1

        This number is same as the total pair number calculated by
        `query_ball_tree`:

        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> sum([len(i) for i in indexes])
        1

        """
        return super().count_neighbors(other, r, p, weights, cumulative)

    def sparse_distance_matrix(
            self, other, max_distance, p=2., output_type='dok_matrix'):
        """Compute a sparse distance matrix.

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : KDTree

        max_distance : positive float

        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            A finite large p may cause a ValueError if overflow can occur.

        output_type : string, optional
            Which container to use for output data. Options: 'dok_matrix',
            'coo_matrix', 'dict', or 'ndarray'. Default: 'dok_matrix'.

            .. versionadded:: 1.6.0

        Returns
        -------
        result : dok_matrix, coo_matrix, dict or ndarray
            Sparse matrix representing the results in "dictionary of keys"
            format. If a dict is returned the keys are (i,j) tuples of indices.
            If output_type is 'ndarray' a record array with fields 'i', 'j',
            and 'v' is returned,

        Examples
        --------
        You can compute a sparse distance matrix between two kd-trees:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        You can check distances above the `max_distance` are zeros:

        >>> from scipy.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])

        """
        return super().sparse_distance_matrix(
            other, max_distance, p, output_type)


def distance_matrix(x, y, p=2, threshold=1000000):
    """Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.

    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.

    Examples
    --------
    >>> from scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    """

    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))

    if m*n*k <= threshold:
        return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
    else:
        result = np.empty((m,n),dtype=float)  # FIXME: figure out the best dtype
        if m < n:
            for i in range(m):
                result[i,:] = minkowski_distance(x[i],y,p)
        else:
            for j in range(n):
                result[:,j] = minkowski_distance(x,y[j],p)
        return result
