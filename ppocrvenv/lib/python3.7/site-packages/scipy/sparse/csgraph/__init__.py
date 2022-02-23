r"""
Compressed sparse graph routines (:mod:`scipy.sparse.csgraph`)
==============================================================

.. currentmodule:: scipy.sparse.csgraph

Fast graph algorithms based on sparse matrix representations.

Contents
--------

.. autosummary::
   :toctree: generated/

   connected_components -- determine connected components of a graph
   laplacian -- compute the laplacian of a graph
   shortest_path -- compute the shortest path between points on a positive graph
   dijkstra -- use Dijkstra's algorithm for shortest path
   floyd_warshall -- use the Floyd-Warshall algorithm for shortest path
   bellman_ford -- use the Bellman-Ford algorithm for shortest path
   johnson -- use Johnson's algorithm for shortest path
   breadth_first_order -- compute a breadth-first order of nodes
   depth_first_order -- compute a depth-first order of nodes
   breadth_first_tree -- construct the breadth-first tree from a given node
   depth_first_tree -- construct a depth-first tree from a given node
   minimum_spanning_tree -- construct the minimum spanning tree of a graph
   reverse_cuthill_mckee -- compute permutation for reverse Cuthill-McKee ordering
   maximum_flow -- solve the maximum flow problem for a graph
   maximum_bipartite_matching -- compute a maximum matching of a bipartite graph
   min_weight_full_bipartite_matching - compute a minimum weight full matching of a bipartite graph
   structural_rank -- compute the structural rank of a graph
   NegativeCycleError

.. autosummary::
   :toctree: generated/

   construct_dist_matrix
   csgraph_from_dense
   csgraph_from_masked
   csgraph_masked_from_dense
   csgraph_to_dense
   csgraph_to_masked
   reconstruct_path

Graph Representations
---------------------
This module uses graphs which are stored in a matrix format. A
graph with N nodes can be represented by an (N x N) adjacency matrix G.
If there is a connection from node i to node j, then G[i, j] = w, where
w is the weight of the connection. For nodes i and j which are
not connected, the value depends on the representation:

- for dense array representations, non-edges are represented by
  G[i, j] = 0, infinity, or NaN.

- for dense masked representations (of type np.ma.MaskedArray), non-edges
  are represented by masked values. This can be useful when graphs with
  zero-weight edges are desired.

- for sparse array representations, non-edges are represented by
  non-entries in the matrix. This sort of sparse representation also
  allows for edges with zero weights.

As a concrete example, imagine that you would like to represent the following
undirected graph::

              G

             (0)
            /   \
           1     2
          /       \
        (2)       (1)

This graph has three nodes, where node 0 and 1 are connected by an edge of
weight 2, and nodes 0 and 2 are connected by an edge of weight 1.
We can construct the dense, masked, and sparse representations as follows,
keeping in mind that an undirected graph is represented by a symmetric matrix::

    >>> G_dense = np.array([[0, 2, 1],
    ...                     [2, 0, 0],
    ...                     [1, 0, 0]])
    >>> G_masked = np.ma.masked_values(G_dense, 0)
    >>> from scipy.sparse import csr_matrix
    >>> G_sparse = csr_matrix(G_dense)

This becomes more difficult when zero edges are significant. For example,
consider the situation when we slightly modify the above graph::

             G2

             (0)
            /   \
           0     2
          /       \
        (2)       (1)

This is identical to the previous graph, except nodes 0 and 2 are connected
by an edge of zero weight. In this case, the dense representation above
leads to ambiguities: how can non-edges be represented if zero is a meaningful
value? In this case, either a masked or sparse representation must be used
to eliminate the ambiguity::

    >>> G2_data = np.array([[np.inf, 2,      0     ],
    ...                     [2,      np.inf, np.inf],
    ...                     [0,      np.inf, np.inf]])
    >>> G2_masked = np.ma.masked_invalid(G2_data)
    >>> from scipy.sparse.csgraph import csgraph_from_dense
    >>> # G2_sparse = csr_matrix(G2_data) would give the wrong result
    >>> G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
    >>> G2_sparse.data
    array([ 2.,  0.,  2.,  0.])

Here we have used a utility routine from the csgraph submodule in order to
convert the dense representation to a sparse representation which can be
understood by the algorithms in submodule. By viewing the data array, we
can see that the zero values are explicitly encoded in the graph.

Directed vs. undirected
^^^^^^^^^^^^^^^^^^^^^^^
Matrices may represent either directed or undirected graphs. This is
specified throughout the csgraph module by a boolean keyword. Graphs are
assumed to be directed by default. In a directed graph, traversal from node
i to node j can be accomplished over the edge G[i, j], but not the edge
G[j, i].  Consider the following dense graph::

    >>> G_dense = np.array([[0, 1, 0],
    ...                     [2, 0, 3],
    ...                     [0, 4, 0]])

When ``directed=True`` we get the graph::

      ---1--> ---3-->
    (0)     (1)     (2)
      <--2--- <--4---

In a non-directed graph, traversal from node i to node j can be
accomplished over either G[i, j] or G[j, i].  If both edges are not null,
and the two have unequal weights, then the smaller of the two is used.

So for the same graph, when ``directed=False`` we get the graph::

    (0)--1--(1)--2--(2)

Note that a symmetric matrix will represent an undirected graph, regardless
of whether the 'directed' keyword is set to True or False. In this case,
using ``directed=True`` generally leads to more efficient computation.

The routines in this module accept as input either scipy.sparse representations
(csr, csc, or lil format), masked representations, or dense representations
with non-edges indicated by zeros, infinities, and NaN entries.
"""

__docformat__ = "restructuredtext en"

__all__ = ['connected_components',
           'laplacian',
           'shortest_path',
           'floyd_warshall',
           'dijkstra',
           'bellman_ford',
           'johnson',
           'breadth_first_order',
           'depth_first_order',
           'breadth_first_tree',
           'depth_first_tree',
           'minimum_spanning_tree',
           'reverse_cuthill_mckee',
           'maximum_flow',
           'maximum_bipartite_matching',
           'min_weight_full_bipartite_matching',
           'structural_rank',
           'construct_dist_matrix',
           'reconstruct_path',
           'csgraph_masked_from_dense',
           'csgraph_from_dense',
           'csgraph_from_masked',
           'csgraph_to_dense',
           'csgraph_to_masked',
           'NegativeCycleError']

from ._laplacian import laplacian
from ._shortest_path import (
    shortest_path, floyd_warshall, dijkstra, bellman_ford, johnson,
    NegativeCycleError
)
from ._traversal import (
    breadth_first_order, depth_first_order, breadth_first_tree,
    depth_first_tree, connected_components
)
from ._min_spanning_tree import minimum_spanning_tree
from ._flow import maximum_flow
from ._matching import (
    maximum_bipartite_matching, min_weight_full_bipartite_matching
)
from ._reordering import reverse_cuthill_mckee, structural_rank
from ._tools import (
    construct_dist_matrix, reconstruct_path, csgraph_from_dense,
    csgraph_to_dense, csgraph_masked_from_dense, csgraph_from_masked,
    csgraph_to_masked
)

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
