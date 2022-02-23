"""Provides explicit constructions of expander graphs.

"""
import itertools
import networkx as nx

__all__ = ["margulis_gabber_galil_graph", "chordal_cycle_graph", "paley_graph"]


# Other discrete torus expanders can be constructed by using the following edge
# sets. For more information, see Chapter 4, "Expander Graphs", in
# "Pseudorandomness", by Salil Vadhan.
#
# For a directed expander, add edges from (x, y) to:
#
#     (x, y),
#     ((x + 1) % n, y),
#     (x, (y + 1) % n),
#     (x, (x + y) % n),
#     (-y % n, x)
#
# For an undirected expander, add the reverse edges.
#
# Also appearing in the paper of Gabber and Galil:
#
#     (x, y),
#     (x, (x + y) % n),
#     (x, (x + y + 1) % n),
#     ((x + y) % n, y),
#     ((x + y + 1) % n, y)
#
# and:
#
#     (x, y),
#     ((x + 2*y) % n, y),
#     ((x + (2*y + 1)) % n, y),
#     ((x + (2*y + 2)) % n, y),
#     (x, (y + 2*x) % n),
#     (x, (y + (2*x + 1)) % n),
#     (x, (y + (2*x + 2)) % n),
#
def margulis_gabber_galil_graph(n, create_using=None):
    r"""Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.

    The undirected MultiGraph is regular with degree `8`. Nodes are integer
    pairs. The second-largest eigenvalue of the adjacency matrix of the graph
    is at most `5 \sqrt{2}`, regardless of `n`.

    Parameters
    ----------
    n : int
        Determines the number of nodes in the graph: `n^2`.
    create_using : NetworkX graph constructor, optional (default MultiGraph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError
        If the graph is directed or not a multigraph.

    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed() or not G.is_multigraph():
        msg = "`create_using` must be an undirected multigraph."
        raise nx.NetworkXError(msg)

    for (x, y) in itertools.product(range(n), repeat=2):
        for (u, v) in (
            ((x + 2 * y) % n, y),
            ((x + (2 * y + 1)) % n, y),
            (x, (y + 2 * x) % n),
            (x, (y + (2 * x + 1)) % n),
        ):
            G.add_edge((x, y), (u, v))
    G.graph["name"] = f"margulis_gabber_galil_graph({n})"
    return G


def chordal_cycle_graph(p, create_using=None):
    """Returns the chordal cycle graph on `p` nodes.

    The returned graph is a cycle graph on `p` nodes with chords joining each
    vertex `x` to its inverse modulo `p`. This graph is a (mildly explicit)
    3-regular expander [1]_.

    `p` *must* be a prime number.

    Parameters
    ----------
    p : a prime number

        The number of vertices in the graph. This also indicates where the
        chordal edges in the cycle will be created.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed undirected multigraph.

    Raises
    ------
    NetworkXError

        If `create_using` indicates directed or not a multigraph.

    References
    ----------

    .. [1] Theorem 4.4.2 in A. Lubotzky. "Discrete groups, expanding graphs and
           invariant measures", volume 125 of Progress in Mathematics.
           Birkhäuser Verlag, Basel, 1994.

    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed() or not G.is_multigraph():
        msg = "`create_using` must be an undirected multigraph."
        raise nx.NetworkXError(msg)

    for x in range(p):
        left = (x - 1) % p
        right = (x + 1) % p
        # Here we apply Fermat's Little Theorem to compute the multiplicative
        # inverse of x in Z/pZ. By Fermat's Little Theorem,
        #
        #     x^p = x (mod p)
        #
        # Therefore,
        #
        #     x * x^(p - 2) = 1 (mod p)
        #
        # The number 0 is a special case: we just let its inverse be itself.
        chord = pow(x, p - 2, p) if x > 0 else 0
        for y in (left, right, chord):
            G.add_edge(x, y)
    G.graph["name"] = f"chordal_cycle_graph({p})"
    return G


def paley_graph(p, create_using=None):
    """Returns the Paley (p-1)/2-regular graph on p nodes.

    The returned graph is a graph on Z/pZ with edges between x and y
    if and only if x-y is a nonzero square in Z/pZ.

    If p = 1 mod 4, -1 is a square in Z/pZ and therefore x-y is a square if and
    only if y-x is also a square, i.e the edges in the Paley graph are symmetric.

    If p = 3 mod 4, -1 is not a square in Z/pZ and therefore either x-y or y-x
    is a square in Z/pZ but not both.

    Note that a more general definition of Paley graphs extends this construction
    to graphs over q=p^n vertices, by using the finite field F_q instead of Z/pZ.
    This construction requires to compute squares in general finite fields and is
    not what is implemented here (i.e paley_graph(25) does not return the true
    Paley graph associated with 5^2).

    Parameters
    ----------
    p : int, an odd prime number.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : graph
        The constructed directed graph.

    Raises
    ------
    NetworkXError
        If the graph is a multigraph.

    References
    ----------
    Chapter 13 in B. Bollobas, Random Graphs. Second edition.
    Cambridge Studies in Advanced Mathematics, 73.
    Cambridge University Press, Cambridge (2001).
    """
    G = nx.empty_graph(0, create_using, default=nx.DiGraph)
    if G.is_multigraph():
        msg = "`create_using` cannot be a multigraph."
        raise nx.NetworkXError(msg)

    # Compute the squares in Z/pZ.
    # Make it a set to uniquify (there are exactly (p-1)/2 squares in Z/pZ
    # when is prime).
    square_set = {(x ** 2) % p for x in range(1, p) if (x ** 2) % p != 0}

    for x in range(p):
        for x2 in square_set:
            G.add_edge(x, (x + x2) % p)
    G.graph["name"] = f"paley({p})"
    return G
