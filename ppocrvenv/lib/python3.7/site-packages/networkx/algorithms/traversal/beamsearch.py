"""Basic algorithms for breadth-first searching the nodes of a graph."""

from .breadth_first_search import generic_bfs_edges

__all__ = ["bfs_beam_edges"]


def bfs_beam_edges(G, source, value, width=None):
    """Iterates over edges in a beam search.

    The beam search is a generalized breadth-first search in which only
    the "best" *w* neighbors of the current node are enqueued, where *w*
    is the beam width and "best" is an application-specific
    heuristic. In general, a beam search with a small beam width might
    not visit each node in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node
        Starting node for the breadth-first search; this function
        iterates over only those edges in the component reachable from
        this node.

    value : function
        A function that takes a node of the graph as input and returns a
        real number indicating how "good" it is. A higher value means it
        is more likely to be visited sooner during the search. When
        visiting a new node, only the `width` neighbors with the highest
        `value` are enqueued (in decreasing order of `value`).

    width : int (default = None)
        The beam width for the search. This is the number of neighbors
        (ordered by `value`) to enqueue when visiting each new node.

    Yields
    ------
    edge
        Edges in the beam search starting from `source`, given as a pair
        of nodes.

    Examples
    --------
    To give nodes with, for example, a higher centrality precedence
    during the search, set the `value` function to return the centrality
    value of the node:

    >>> G = nx.karate_club_graph()
    >>> centrality = nx.eigenvector_centrality(G)
    >>> source = 0
    >>> width = 5
    >>> for u, v in nx.bfs_beam_edges(G, source, centrality.get, width):
    ...     print((u, v))
    ...
    (0, 2)
    (0, 1)
    (0, 8)
    (0, 13)
    (0, 3)
    (2, 32)
    (1, 30)
    (8, 33)
    (3, 7)
    (32, 31)
    (31, 28)
    (31, 25)
    (25, 23)
    (25, 24)
    (23, 29)
    (23, 27)
    (29, 26)
    """

    if width is None:
        width = len(G)

    def successors(v):
        """Returns a list of the best neighbors of a node.

        `v` is a node in the graph `G`.

        The "best" neighbors are chosen according to the `value`
        function (higher is better). Only the `width` best neighbors of
        `v` are returned.

        The list returned by this function is in decreasing value as
        measured by the `value` function.

        """
        # TODO The Python documentation states that for small values, it
        # is better to use `heapq.nlargest`. We should determine the
        # threshold at which its better to use `heapq.nlargest()`
        # instead of `sorted()[:]` and apply that optimization here.
        #
        # If `width` is greater than the number of neighbors of `v`, all
        # neighbors are returned by the semantics of slicing in
        # Python. This occurs in the special case that the user did not
        # specify a `width`: in this case all neighbors are always
        # returned, so this is just a (slower) implementation of
        # `bfs_edges(G, source)` but with a sorted enqueue step.
        return iter(sorted(G.neighbors(v), key=value, reverse=True)[:width])

    yield from generic_bfs_edges(G, source, successors)
