__all__ = ["average_neighbor_degree"]


def average_neighbor_degree(G, source="out", target="out", nodes=None, weight=None):
    r"""Returns the average degree of the neighborhood of each node.

    The average neighborhood degree of a node `i` is

    .. math::

        k_{nn,i} = \frac{1}{|N(i)|} \sum_{j \in N(i)} k_j

    where `N(i)` are the neighbors of node `i` and `k_j` is
    the degree of node `j` which belongs to `N(i)`. For weighted
    graphs, an analogous measure can be defined [1]_,

    .. math::

        k_{nn,i}^{w} = \frac{1}{s_i} \sum_{j \in N(i)} w_{ij} k_j

    where `s_i` is the weighted degree of node `i`, `w_{ij}`
    is the weight of the edge that links `i` and `j` and
    `N(i)` are the neighbors of node `i`.


    Parameters
    ----------
    G : NetworkX graph

    source : string ("in"|"out"|"in+out")
       Directed graphs only.
       Use "in"- or "out"-degree for source node.

    target : string ("in"|"out"|"in+out")
       Directed graphs only.
       Use "in"- or "out"-degree for target node.

    nodes : list or iterable, optional
        Compute neighbor degree for specified nodes.  The default is
        all nodes in the graph.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    d: dict
       A dictionary keyed by node with average neighbors degree value.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.edges[0, 1]["weight"] = 5
    >>> G.edges[2, 3]["weight"] = 3

    >>> nx.average_neighbor_degree(G)
    {0: 2.0, 1: 1.5, 2: 1.5, 3: 2.0}
    >>> nx.average_neighbor_degree(G, weight="weight")
    {0: 2.0, 1: 1.1666666666666667, 2: 1.25, 3: 2.0}

    >>> G = nx.DiGraph()
    >>> nx.add_path(G, [0, 1, 2, 3])
    >>> nx.average_neighbor_degree(G, source="in", target="in")
    {0: 0.0, 1: 1.0, 2: 1.0, 3: 0.0}

    >>> nx.average_neighbor_degree(G, source="out", target="out")
    {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}

    Notes
    -----
    For directed graphs you can also specify in-degree or out-degree
    by passing keyword arguments.

    See Also
    --------
    average_degree_connectivity

    References
    ----------
    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,
       "The architecture of complex weighted networks".
       PNAS 101 (11): 3747–3752 (2004).
    """
    source_degree = G.degree
    target_degree = G.degree
    if G.is_directed():
        if source == "in":
            source_degree = G.in_degree
        elif source == "out":
            source_degree = G.out_degree
        elif source != "in+out":
            raise nx.NetworkXError(
                f"source argument {source} must be 'in', 'out' or 'in+out'"
            )

        if target == "in":
            target_degree = G.in_degree
        elif target == "out":
            target_degree = G.out_degree
        elif target != "in+out":
            raise nx.NetworkXError(
                f"target argument {target} must be 'in', 'out' or 'in+out'"
            )

    # precompute target degrees -- should *not* be weighted degree
    tgt_deg = dict(target_degree())
    # average degree of neighbors
    avg = {}
    for n, deg in source_degree(nodes, weight=weight):
        # normalize but not by zero degree
        if deg == 0:
            avg[n] = 0.0
            continue
        G_n = G[n]
        if weight is None:
            avg[n] = sum(tgt_deg[nbr] for nbr in G_n) / deg
        else:
            avg[n] = sum(G_n[nbr].get(weight, 1) * tgt_deg[nbr] for nbr in G_n) / deg
    return avg
