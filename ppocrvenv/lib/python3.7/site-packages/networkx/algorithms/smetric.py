import networkx as nx

__all__ = ["s_metric"]


def s_metric(G, normalized=True):
    """Returns the s-metric of graph.

    The s-metric is defined as the sum of the products deg(u)*deg(v)
    for every edge (u,v) in G. If norm is provided construct the
    s-max graph and compute it's s_metric, and return the normalized
    s value

    Parameters
    ----------
    G    : graph
           The graph used to compute the s-metric.
    normalized : bool (optional)
           Normalize the value.

    Returns
    -------
    s : float
        The s-metric of the graph.

    References
    ----------
    .. [1] Lun Li, David Alderson, John C. Doyle, and Walter Willinger,
           Towards a Theory of Scale-Free Graphs:
           Definition, Properties, and  Implications (Extended Version), 2005.
           https://arxiv.org/abs/cond-mat/0501169
    """
    if normalized:
        raise nx.NetworkXError("Normalization not implemented")
    #        Gmax = li_smax_graph(list(G.degree().values()))
    #        return s_metric(G,normalized=False)/s_metric(Gmax,normalized=False)
    #    else:
    return float(sum([G.degree(u) * G.degree(v) for (u, v) in G.edges()]))
