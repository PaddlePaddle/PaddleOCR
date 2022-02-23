"""Generators of  x-y pairs of node data."""
__all__ = ["node_attribute_xy", "node_degree_xy"]


def node_attribute_xy(G, attribute, nodes=None):
    """Returns iterator of node-attribute pairs for all edges in G.

    Parameters
    ----------
    G: NetworkX graph

    attribute: key
       The node attribute key.

    nodes: list or iterable (optional)
        Use only edges that are adjacency to specified nodes.
        The default is all nodes.

    Returns
    -------
    (x,y): 2-tuple
        Generates 2-tuple of (attribute,attribute) values.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_node(1, color="red")
    >>> G.add_node(2, color="blue")
    >>> G.add_edge(1, 2)
    >>> list(nx.node_attribute_xy(G, "color"))
    [('red', 'blue')]

    Notes
    -----
    For undirected graphs each edge is produced twice, once for each edge
    representation (u,v) and (v,u), with the exception of self-loop edges
    which only appear once.
    """
    if nodes is None:
        nodes = set(G)
    else:
        nodes = set(nodes)
    Gnodes = G.nodes
    for u, nbrsdict in G.adjacency():
        if u not in nodes:
            continue
        uattr = Gnodes[u].get(attribute, None)
        if G.is_multigraph():
            for v, keys in nbrsdict.items():
                vattr = Gnodes[v].get(attribute, None)
                for k, d in keys.items():
                    yield (uattr, vattr)
        else:
            for v, eattr in nbrsdict.items():
                vattr = Gnodes[v].get(attribute, None)
                yield (uattr, vattr)


def node_degree_xy(G, x="out", y="in", weight=None, nodes=None):
    """Generate node degree-degree pairs for edges in G.

    Parameters
    ----------
    G: NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Use only edges that are adjacency to specified nodes.
        The default is all nodes.

    Returns
    -------
    (x,y): 2-tuple
        Generates 2-tuple of (degree,degree) values.


    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> list(nx.node_degree_xy(G, x="out", y="in"))
    [(1, 1)]
    >>> list(nx.node_degree_xy(G, x="in", y="out"))
    [(0, 0)]

    Notes
    -----
    For undirected graphs each edge is produced twice, once for each edge
    representation (u,v) and (v,u), with the exception of self-loop edges
    which only appear once.
    """
    if nodes is None:
        nodes = set(G)
    else:
        nodes = set(nodes)
    xdeg = G.degree
    ydeg = G.degree
    if G.is_directed():
        direction = {"out": G.out_degree, "in": G.in_degree}
        xdeg = direction[x]
        ydeg = direction[y]

    for u, degu in xdeg(nodes, weight=weight):
        neighbors = (nbr for _, nbr in G.edges(u) if nbr in nodes)
        for v, degv in ydeg(neighbors, weight=weight):
            yield degu, degv
