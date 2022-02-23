"""
Graph utilities and algorithms

Graphs are represented with their adjacency matrices, preferably using
sparse matrices.
"""

# Authors: Aric Hagberg <hagberg@lanl.gov>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD 3 clause

import numpy as np
from scipy import sparse

from .deprecation import deprecated
from ..metrics.pairwise import pairwise_distances


###############################################################################
# Path and connected component analysis.
# Code adapted from networkx
def single_source_shortest_path_length(graph, source, *, cutoff=None):
    """Return the shortest path length from source to all reachable nodes.

    Returns a dictionary of shortest path lengths keyed by target.

    Parameters
    ----------
    graph : {sparse matrix, ndarray} of shape (n, n)
        Adjacency matrix of the graph. Sparse matrix of format LIL is
        preferred.

    source : int
       Starting node for path.

    cutoff : int, default=None
        Depth to stop the search - only paths of length <= cutoff are returned.

    Examples
    --------
    >>> from sklearn.utils.graph import single_source_shortest_path_length
    >>> import numpy as np
    >>> graph = np.array([[ 0, 1, 0, 0],
    ...                   [ 1, 0, 1, 0],
    ...                   [ 0, 1, 0, 1],
    ...                   [ 0, 0, 1, 0]])
    >>> list(sorted(single_source_shortest_path_length(graph, 0).items()))
    [(0, 0), (1, 1), (2, 2), (3, 3)]
    >>> graph = np.ones((6, 6))
    >>> list(sorted(single_source_shortest_path_length(graph, 2).items()))
    [(0, 1), (1, 1), (2, 0), (3, 1), (4, 1), (5, 1)]
    """
    if sparse.isspmatrix(graph):
        graph = graph.tolil()
    else:
        graph = sparse.lil_matrix(graph)
    seen = {}  # level (number of hops) when seen in BFS
    level = 0  # the current level
    next_level = [source]  # dict of nodes to check at next level
    while next_level:
        this_level = next_level  # advance to next level
        next_level = set()  # and start a new list (fringe)
        for v in this_level:
            if v not in seen:
                seen[v] = level  # set the level of vertex v
                next_level.update(graph.rows[v])
        if cutoff is not None and cutoff <= level:
            break
        level += 1
    return seen  # return all path lengths as dictionary


@deprecated(
    "`graph_shortest_path` is deprecated in 1.0 (renaming of 0.25) and will "
    "be removed in 1.2. Use `scipy.sparse.csgraph.shortest_path` instead."
)
def graph_shortest_path(dist_matrix, directed=True, method="auto"):
    """Shortest-path graph search on a positive directed or undirected graph.

    Parameters
    ----------
    dist_matrix : arraylike or sparse matrix, shape = (N,N)
        Array of positive distances.
        If vertex i is connected to vertex j, then dist_matrix[i,j] gives
        the distance between the vertices.
        If vertex i is not connected to vertex j, then dist_matrix[i,j] = 0

    directed : boolean
        if True, then find the shortest path on a directed graph: only
        progress from a point to its neighbors, not the other way around.
        if False, then find the shortest path on an undirected graph: the
        algorithm can progress from a point to its neighbors and vice versa.

    method : {'auto', 'FW', 'D'}, default='auto'
        method to use.  Options are
        'auto' : attempt to choose the best method for the current problem
        'FW' : Floyd-Warshall algorithm.  O[N^3]
        'D' : Dijkstra's algorithm with Fibonacci stacks.  O[(k+log(N))N^2]

    Returns
    -------
    G : np.ndarray, float, shape = [N,N]
        G[i,j] gives the shortest distance from point i to point j
        along the graph.

    Notes
    -----
    As currently implemented, Dijkstra's algorithm does not work for
    graphs with direction-dependent distances when directed == False.
    i.e., if dist_matrix[i,j] and dist_matrix[j,i] are not equal and
    both are nonzero, method='D' will not necessarily yield the correct
    result.
    Also, these routines have not been tested for graphs with negative
    distances.  Negative distances can lead to infinite cycles that must
    be handled by specialized algorithms.
    """
    return sparse.csgraph.shortest_path(dist_matrix, method=method, directed=directed)


def _fix_connected_components(
    X,
    graph,
    n_connected_components,
    component_labels,
    mode="distance",
    metric="euclidean",
    **kwargs,
):
    """Add connections to sparse graph to connect unconnected components.

    For each pair of unconnected components, compute all pairwise distances
    from one component to the other, and add a connection on the closest pair
    of samples. This is a hacky way to get a graph with a single connected
    component, which is necessary for example to compute a shortest path
    between all pairs of samples in the graph.

    Parameters
    ----------
    X : array of shape (n_samples, n_features) or (n_samples, n_samples)
        Features to compute the pairwise distances. If `metric =
        "precomputed"`, X is the matrix of pairwise distances.

    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples.

    n_connected_components : int
        Number of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.

    component_labels : array of shape (n_samples)
        Labels of connected components, as computed by
        `scipy.sparse.csgraph.connected_components`.

    mode : {'connectivity', 'distance'}, default='distance'
        Type of graph matrix: 'connectivity' corresponds to the connectivity
        matrix with ones and zeros, and 'distance' corresponds to the distances
        between neighbors according to the given metric.

    metric : str
        Metric used in `sklearn.metrics.pairwise.pairwise_distances`.

    kwargs : kwargs
        Keyword arguments passed to
        `sklearn.metrics.pairwise.pairwise_distances`.

    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Graph of connection between samples, with a single connected component.
    """
    if metric == "precomputed" and sparse.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(component_labels == i)
        Xi = X[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(component_labels == j)
            Xj = X[idx_j]

            if metric == "precomputed":
                D = X[np.ix_(idx_i, idx_j)]
            else:
                D = pairwise_distances(Xi, Xj, metric=metric, **kwargs)

            ii, jj = np.unravel_index(D.argmin(axis=None), D.shape)
            if mode == "connectivity":
                graph[idx_i[ii], idx_j[jj]] = 1
                graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                graph[idx_i[ii], idx_j[jj]] = D[ii, jj]
                graph[idx_j[jj], idx_i[ii]] = D[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return graph
