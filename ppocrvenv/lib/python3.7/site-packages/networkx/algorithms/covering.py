""" Functions related to graph covers."""

import networkx as nx
from networkx.utils import not_implemented_for, arbitrary_element
from functools import partial
from itertools import chain


__all__ = ["min_edge_cover", "is_edge_cover"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def min_edge_cover(G, matching_algorithm=None):
    """Returns a set of edges which constitutes
    the minimum edge cover of the graph.

    A smallest edge cover can be found in polynomial time by finding
    a maximum matching and extending it greedily so that all nodes
    are covered.

    Parameters
    ----------
    G : NetworkX graph
        An undirected bipartite graph.

    matching_algorithm : function
        A function that returns a maximum cardinality matching in a
        given bipartite graph. The function must take one input, the
        graph ``G``, and return a dictionary mapping each node to its
        mate. If not specified,
        :func:`~networkx.algorithms.bipartite.matching.hopcroft_karp_matching`
        will be used. Other possibilities include
        :func:`~networkx.algorithms.bipartite.matching.eppstein_matching`,
        or matching algorithms in the
        :mod:`networkx.algorithms.matching` module.

    Returns
    -------
    min_cover : set

        It contains all the edges of minimum edge cover
        in form of tuples. It contains both the edges `(u, v)` and `(v, u)`
        for given nodes `u` and `v` among the edges of minimum edge cover.

    Notes
    -----
    An edge cover of a graph is a set of edges such that every node of
    the graph is incident to at least one edge of the set.
    The minimum edge cover is an edge covering of smallest cardinality.

    Due to its implementation, the worst-case running time of this algorithm
    is bounded by the worst-case running time of the function
    ``matching_algorithm``.

    Minimum edge cover for bipartite graph can also be found using the
    function present in :mod:`networkx.algorithms.bipartite.covering`
    """
    if nx.number_of_isolates(G) > 0:
        # ``min_cover`` does not exist as there is an isolated node
        raise nx.NetworkXException(
            "Graph has a node with no edge incident on it, " "so no edge cover exists."
        )
    if matching_algorithm is None:
        matching_algorithm = partial(nx.max_weight_matching, maxcardinality=True)
    maximum_matching = matching_algorithm(G)
    # ``min_cover`` is superset of ``maximum_matching``
    try:
        min_cover = set(
            maximum_matching.items()
        )  # bipartite matching case returns dict
    except AttributeError:
        min_cover = maximum_matching
    # iterate for uncovered nodes
    uncovered_nodes = set(G) - {v for u, v in min_cover} - {u for u, v in min_cover}
    for v in uncovered_nodes:
        # Since `v` is uncovered, each edge incident to `v` will join it
        # with a covered node (otherwise, if there were an edge joining
        # uncovered nodes `u` and `v`, the maximum matching algorithm
        # would have found it), so we can choose an arbitrary edge
        # incident to `v`. (This applies only in a simple graph, not a
        # multigraph.)
        u = arbitrary_element(G[v])
        min_cover.add((u, v))
        min_cover.add((v, u))
    return min_cover


@not_implemented_for("directed")
def is_edge_cover(G, cover):
    """Decides whether a set of edges is a valid edge cover of the graph.

    Given a set of edges, whether it is an edge covering can
    be decided if we just check whether all nodes of the graph
    has an edge from the set, incident on it.

    Parameters
    ----------
    G : NetworkX graph
        An undirected bipartite graph.

    cover : set
        Set of edges to be checked.

    Returns
    -------
    bool
        Whether the set of edges is a valid edge cover of the graph.

    Notes
    -----
    An edge cover of a graph is a set of edges such that every node of
    the graph is incident to at least one edge of the set.
    """
    return set(G) <= set(chain.from_iterable(cover))
