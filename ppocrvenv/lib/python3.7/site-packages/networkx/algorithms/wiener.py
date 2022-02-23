"""Functions related to the Wiener index of a graph."""

from itertools import chain

from .components import is_connected
from .components import is_strongly_connected
from .shortest_paths import shortest_path_length as spl

__all__ = ["wiener_index"]

#: Rename the :func:`chain.from_iterable` function for the sake of
#: brevity.
chaini = chain.from_iterable


def wiener_index(G, weight=None):
    """Returns the Wiener index of the given graph.

    The *Wiener index* of a graph is the sum of the shortest-path
    distances between each pair of reachable nodes. For pairs of nodes
    in undirected graphs, only one orientation of the pair is counted.

    Parameters
    ----------
    G : NetworkX graph

    weight : object
        The edge attribute to use as distance when computing
        shortest-path distances. This is passed directly to the
        :func:`networkx.shortest_path_length` function.

    Returns
    -------
    float
        The Wiener index of the graph `G`.

    Raises
    ------
    NetworkXError
        If the graph `G` is not connected.

    Notes
    -----
    If a pair of nodes is not reachable, the distance is assumed to be
    infinity. This means that for graphs that are not
    strongly-connected, this function returns ``inf``.

    The Wiener index is not usually defined for directed graphs, however
    this function uses the natural generalization of the Wiener index to
    directed graphs.

    Examples
    --------
    The Wiener index of the (unweighted) complete graph on *n* nodes
    equals the number of pairs of the *n* nodes, since each pair of
    nodes is at distance one::

        >>> n = 10
        >>> G = nx.complete_graph(n)
        >>> nx.wiener_index(G) == n * (n - 1) / 2
        True

    Graphs that are not strongly-connected have infinite Wiener index::

        >>> G = nx.empty_graph(2)
        >>> nx.wiener_index(G)
        inf

    """
    is_directed = G.is_directed()
    if (is_directed and not is_strongly_connected(G)) or (
        not is_directed and not is_connected(G)
    ):
        return float("inf")
    total = sum(chaini(p.values() for v, p in spl(G, weight=weight)))
    # Need to account for double counting pairs of nodes in undirected graphs.
    return total if is_directed else total / 2
