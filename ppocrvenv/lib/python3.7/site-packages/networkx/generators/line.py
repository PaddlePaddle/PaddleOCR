"""Functions for generating line graphs."""
from itertools import combinations
from collections import defaultdict

import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for

__all__ = ["line_graph", "inverse_line_graph"]


def line_graph(G, create_using=None):
    r"""Returns the line graph of the graph or digraph `G`.

    The line graph of a graph `G` has a node for each edge in `G` and an
    edge joining those nodes if the two edges in `G` share a common node. For
    directed graphs, nodes are adjacent exactly when the edges they represent
    form a directed path of length two.

    The nodes of the line graph are 2-tuples of nodes in the original graph (or
    3-tuples for multigraphs, with the key of the edge as the third element).

    For information about self-loops and more discussion, see the **Notes**
    section below.

    Parameters
    ----------
    G : graph
        A NetworkX Graph, DiGraph, MultiGraph, or MultiDigraph.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    L : graph
        The line graph of G.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> L = nx.line_graph(G)
    >>> print(sorted(map(sorted, L.edges())))  # makes a 3-clique, K3
    [[(0, 1), (0, 2)], [(0, 1), (0, 3)], [(0, 2), (0, 3)]]

    Notes
    -----
    Graph, node, and edge data are not propagated to the new graph. For
    undirected graphs, the nodes in G must be sortable, otherwise the
    constructed line graph may not be correct.

    *Self-loops in undirected graphs*

    For an undirected graph `G` without multiple edges, each edge can be
    written as a set `\{u, v\}`.  Its line graph `L` has the edges of `G` as
    its nodes. If `x` and `y` are two nodes in `L`, then `\{x, y\}` is an edge
    in `L` if and only if the intersection of `x` and `y` is nonempty. Thus,
    the set of all edges is determined by the set of all pairwise intersections
    of edges in `G`.

    Trivially, every edge in G would have a nonzero intersection with itself,
    and so every node in `L` should have a self-loop. This is not so
    interesting, and the original context of line graphs was with simple
    graphs, which had no self-loops or multiple edges. The line graph was also
    meant to be a simple graph and thus, self-loops in `L` are not part of the
    standard definition of a line graph. In a pairwise intersection matrix,
    this is analogous to excluding the diagonal entries from the line graph
    definition.

    Self-loops and multiple edges in `G` add nodes to `L` in a natural way, and
    do not require any fundamental changes to the definition. It might be
    argued that the self-loops we excluded before should now be included.
    However, the self-loops are still "trivial" in some sense and thus, are
    usually excluded.

    *Self-loops in directed graphs*

    For a directed graph `G` without multiple edges, each edge can be written
    as a tuple `(u, v)`. Its line graph `L` has the edges of `G` as its
    nodes. If `x` and `y` are two nodes in `L`, then `(x, y)` is an edge in `L`
    if and only if the tail of `x` matches the head of `y`, for example, if `x
    = (a, b)` and `y = (b, c)` for some vertices `a`, `b`, and `c` in `G`.

    Due to the directed nature of the edges, it is no longer the case that
    every edge in `G` should have a self-loop in `L`. Now, the only time
    self-loops arise is if a node in `G` itself has a self-loop.  So such
    self-loops are no longer "trivial" but instead, represent essential
    features of the topology of `G`. For this reason, the historical
    development of line digraphs is such that self-loops are included. When the
    graph `G` has multiple edges, once again only superficial changes are
    required to the definition.

    References
    ----------
    * Harary, Frank, and Norman, Robert Z., "Some properties of line digraphs",
      Rend. Circ. Mat. Palermo, II. Ser. 9 (1960), 161--168.
    * Hemminger, R. L.; Beineke, L. W. (1978), "Line graphs and line digraphs",
      in Beineke, L. W.; Wilson, R. J., Selected Topics in Graph Theory,
      Academic Press Inc., pp. 271--305.

    """
    if G.is_directed():
        L = _lg_directed(G, create_using=create_using)
    else:
        L = _lg_undirected(G, selfloops=False, create_using=create_using)
    return L


def _node_func(G):
    """Returns a function which returns a sorted node for line graphs.

    When constructing a line graph for undirected graphs, we must normalize
    the ordering of nodes as they appear in the edge.

    """
    if G.is_multigraph():

        def sorted_node(u, v, key):
            return (u, v, key) if u <= v else (v, u, key)

    else:

        def sorted_node(u, v):
            return (u, v) if u <= v else (v, u)

    return sorted_node


def _edge_func(G):
    """Returns the edges from G, handling keys for multigraphs as necessary."""
    if G.is_multigraph():

        def get_edges(nbunch=None):
            return G.edges(nbunch, keys=True)

    else:

        def get_edges(nbunch=None):
            return G.edges(nbunch)

    return get_edges


def _sorted_edge(u, v):
    """Returns a sorted edge.

    During the construction of a line graph for undirected graphs, the data
    structure can be a multigraph even though the line graph will never have
    multiple edges between its nodes.  For this reason, we must make sure not
    to add any edge more than once.  This requires that we build up a list of
    edges to add and then remove all duplicates.  And so, we must normalize
    the representation of the edges.

    """
    return (u, v) if u <= v else (v, u)


def _lg_directed(G, create_using=None):
    """Returns the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : NetworkX graph constructor, optional
       Graph type to create. If graph instance, then cleared before populated.
       Default is to use the same graph class as `G`.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Create a graph specific edge function.
    get_edges = _edge_func(G)

    for from_node in get_edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in get_edges(from_node[1]):
            L.add_edge(from_node, to_node)

    return L


def _lg_undirected(G, selfloops=False, create_using=None):
    """Returns the line graph L of the (multi)graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Graph specific functions for edges and sorted nodes.
    get_edges = _edge_func(G)
    sorted_node = _node_func(G)

    # Determine if we include self-loops or not.
    shift = 0 if selfloops else 1

    edges = set()
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        nodes = [sorted_node(*x) for x in get_edges(u)]

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            L.add_node(nodes[0])

        # Add a clique of `nodes` to graph. To prevent double adding edges,
        # especially important for multigraphs, we store the edges in
        # canonical form in a set.
        for i, a in enumerate(nodes):
            edges.update([_sorted_edge(a, b) for b in nodes[i + shift :]])

    L.add_edges_from(edges)
    return L


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def inverse_line_graph(G):
    """Returns the inverse line graph of graph G.

    If H is a graph, and G is the line graph of H, such that G = L(H).
    Then H is the inverse line graph of G.

    Not all graphs are line graphs and these do not have an inverse line graph.
    In these cases this function raises a NetworkXError.

    Parameters
    ----------
    G : graph
        A NetworkX Graph

    Returns
    -------
    H : graph
        The inverse line graph of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed or a multigraph

    NetworkXError
        If G is not a line graph

    Notes
    -----
    This is an implementation of the Roussopoulos algorithm.

    If G consists of multiple components, then the algorithm doesn't work.
    You should invert every component seperately:

    >>> K5 = nx.complete_graph(5)
    >>> P4 = nx.Graph([("a", "b"), ("b", "c"), ("c", "d")])
    >>> G = nx.union(K5, P4)
    >>> root_graphs = []
    >>> for comp in nx.connected_components(G):
    ...     root_graphs.append(nx.inverse_line_graph(G.subgraph(comp)))
    >>> len(root_graphs)
    2

    References
    ----------
    * Roussopolous, N, "A max {m, n} algorithm for determining the graph H from
      its line graph G", Information Processing Letters 2, (1973), 108--112.

    """
    if G.number_of_nodes() == 0:
        return nx.empty_graph(1)
    elif G.number_of_nodes() == 1:
        v = arbitrary_element(G)
        a = (v, 0)
        b = (v, 1)
        H = nx.Graph([(a, b)])
        return H
    elif G.number_of_nodes() > 1 and G.number_of_edges() == 0:
        msg = (
            "inverse_line_graph() doesn't work on an edgeless graph. "
            "Please use this function on each component seperately."
        )
        raise nx.NetworkXError(msg)

    starting_cell = _select_starting_cell(G)
    P = _find_partition(G, starting_cell)
    # count how many times each vertex appears in the partition set
    P_count = {u: 0 for u in G.nodes}
    for p in P:
        for u in p:
            P_count[u] += 1

    if max(P_count.values()) > 2:
        msg = "G is not a line graph (vertex found in more than two partition cells)"
        raise nx.NetworkXError(msg)
    W = tuple((u,) for u in P_count if P_count[u] == 1)
    H = nx.Graph()
    H.add_nodes_from(P)
    H.add_nodes_from(W)
    for a, b in combinations(H.nodes, 2):
        if any(a_bit in b for a_bit in a):
            H.add_edge(a, b)
    return H


def _triangles(G, e):
    """Return list of all triangles containing edge e"""
    u, v = e
    if u not in G:
        raise nx.NetworkXError(f"Vertex {u} not in graph")
    if v not in G[u]:
        raise nx.NetworkXError(f"Edge ({u}, {v}) not in graph")
    triangle_list = []
    for x in G[u]:
        if x in G[v]:
            triangle_list.append((u, v, x))
    return triangle_list


def _odd_triangle(G, T):
    """Test whether T is an odd triangle in G

    Parameters
    ----------
    G : NetworkX Graph
    T : 3-tuple of vertices forming triangle in G

    Returns
    -------
    True is T is an odd triangle
    False otherwise

    Raises
    ------
    NetworkXError
        T is not a triangle in G

    Notes
    -----
    An odd triangle is one in which there exists another vertex in G which is
    adjacent to either exactly one or exactly all three of the vertices in the
    triangle.

    """
    for u in T:
        if u not in G.nodes():
            raise nx.NetworkXError(f"Vertex {u} not in graph")
    for e in list(combinations(T, 2)):
        if e[0] not in G[e[1]]:
            raise nx.NetworkXError(f"Edge ({e[0]}, {e[1]}) not in graph")

    T_neighbors = defaultdict(int)
    for t in T:
        for v in G[t]:
            if v not in T:
                T_neighbors[v] += 1
    for v in T_neighbors:
        if T_neighbors[v] in [1, 3]:
            return True
    return False


def _find_partition(G, starting_cell):
    """Find a partition of the vertices of G into cells of complete graphs

    Parameters
    ----------
    G : NetworkX Graph
    starting_cell : tuple of vertices in G which form a cell

    Returns
    -------
    List of tuples of vertices of G

    Raises
    ------
    NetworkXError
        If a cell is not a complete subgraph then G is not a line graph
    """
    G_partition = G.copy()
    P = [starting_cell]  # partition set
    G_partition.remove_edges_from(list(combinations(starting_cell, 2)))
    # keep list of partitioned nodes which might have an edge in G_partition
    partitioned_vertices = list(starting_cell)
    while G_partition.number_of_edges() > 0:
        # there are still edges left and so more cells to be made
        u = partitioned_vertices[-1]
        deg_u = len(G_partition[u])
        if deg_u == 0:
            # if u has no edges left in G_partition then we have found
            # all of its cells so we do not need to keep looking
            partitioned_vertices.pop()
        else:
            # if u still has edges then we need to find its other cell
            # this other cell must be a complete subgraph or else G is
            # not a line graph
            new_cell = [u] + list(G_partition[u])
            for u in new_cell:
                for v in new_cell:
                    if (u != v) and (v not in G_partition[u]):
                        msg = (
                            "G is not a line graph"
                            "(partition cell not a complete subgraph)"
                        )
                        raise nx.NetworkXError(msg)
            P.append(tuple(new_cell))
            G_partition.remove_edges_from(list(combinations(new_cell, 2)))
            partitioned_vertices += new_cell
    return P


def _select_starting_cell(G, starting_edge=None):
    """Select a cell to initiate _find_partition

    Parameters
    ----------
    G : NetworkX Graph
    starting_edge: an edge to build the starting cell from

    Returns
    -------
    Tuple of vertices in G

    Raises
    ------
    NetworkXError
        If it is determined that G is not a line graph

    Notes
    -----
    If starting edge not specified then pick an arbitrary edge - doesn't
    matter which. However, this function may call itself requiring a
    specific starting edge. Note that the r, s notation for counting
    triangles is the same as in the Roussopoulos paper cited above.
    """
    if starting_edge is None:
        e = arbitrary_element(G.edges())
    else:
        e = starting_edge
        if e[1] not in G[e[0]]:
            msg = f"starting_edge ({e[0]}, {e[1]}) is not in the Graph"
            raise nx.NetworkXError(msg)
    e_triangles = _triangles(G, e)
    r = len(e_triangles)
    if r == 0:
        # there are no triangles containing e, so the starting cell is just e
        starting_cell = e
    elif r == 1:
        # there is exactly one triangle, T, containing e. If other 2 edges
        # of T belong only to this triangle then T is starting cell
        T = e_triangles[0]
        a, b, c = T
        # ab was original edge so check the other 2 edges
        ac_edges = [x for x in _triangles(G, (a, c))]
        bc_edges = [x for x in _triangles(G, (b, c))]
        if len(ac_edges) == 1:
            if len(bc_edges) == 1:
                starting_cell = T
            else:
                return _select_starting_cell(G, starting_edge=(b, c))
        else:
            return _select_starting_cell(G, starting_edge=(a, c))
    else:
        # r >= 2 so we need to count the number of odd triangles, s
        s = 0
        odd_triangles = []
        for T in e_triangles:
            if _odd_triangle(G, T):
                s += 1
                odd_triangles.append(T)
        if r == 2 and s == 0:
            # in this case either triangle works, so just use T
            starting_cell = T
        elif r - 1 <= s <= r:
            # check if odd triangles containing e form complete subgraph
            # there must be exactly s+2 of them
            # and they must all be connected
            triangle_nodes = set()
            for T in odd_triangles:
                for x in T:
                    triangle_nodes.add(x)
            if len(triangle_nodes) == s + 2:
                for u in triangle_nodes:
                    for v in triangle_nodes:
                        if u != v and (v not in G[u]):
                            msg = (
                                "G is not a line graph (odd triangles "
                                "do not form complete subgraph)"
                            )
                            raise nx.NetworkXError(msg)
                # otherwise then we can use this as the starting cell
                starting_cell = tuple(triangle_nodes)
            else:
                msg = (
                    "G is not a line graph (odd triangles "
                    "do not form complete subgraph)"
                )
                raise nx.NetworkXError(msg)
        else:
            msg = (
                "G is not a line graph (incorrect number of "
                "odd triangles around starting edge)"
            )
            raise nx.NetworkXError(msg)
    return starting_cell
