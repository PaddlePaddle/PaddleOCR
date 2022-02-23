"""Node assortativity coefficients and correlation measures.
"""
from networkx.algorithms.assortativity.mixing import (
    degree_mixing_matrix,
    attribute_mixing_matrix,
    numeric_mixing_matrix,
)
from networkx.algorithms.assortativity.pairs import node_degree_xy

__all__ = [
    "degree_pearson_correlation_coefficient",
    "degree_assortativity_coefficient",
    "attribute_assortativity_coefficient",
    "numeric_assortativity_coefficient",
]


def degree_assortativity_coefficient(G, x="out", y="in", weight=None, nodes=None):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute degree assortativity only for nodes in container.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> r = nx.degree_assortativity_coefficient(G)
    >>> print(f"{r:3.1f}")
    -0.5

    See Also
    --------
    attribute_assortativity_coefficient
    numeric_assortativity_coefficient
    degree_mixing_dict
    degree_mixing_matrix

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , where e is the joint
    probability distribution (mixing matrix) of the degrees.  If G is
    directed than the matrix e is the joint probability of the
    user-specified degree type for the source and target.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
    if nodes is None:
        nodes = G.nodes
    degrees = set([d for n, d in G.degree(nodes, weight=weight)])
    mapping = {d: i for i, d, in enumerate(degrees)}
    M = degree_mixing_matrix(G, x=x, y=y, nodes=nodes, weight=weight, mapping=mapping)
    return numeric_ac(M, mapping=mapping)


def degree_pearson_correlation_coefficient(G, x="out", y="in", weight=None, nodes=None):
    """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    This is the same as degree_assortativity_coefficient but uses the
    potentially faster scipy.stats.pearsonr function.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute pearson correlation of degrees only for specified nodes.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> r = nx.degree_pearson_correlation_coefficient(G)
    >>> print(f"{r:3.1f}")
    -0.5

    Notes
    -----
    This calls scipy.stats.pearsonr.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks
           Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
    import scipy as sp
    import scipy.stats  # call as sp.stats

    xy = node_degree_xy(G, x=x, y=y, nodes=nodes, weight=weight)
    x, y = zip(*xy)
    return sp.stats.pearsonr(x, y)[0]


def attribute_assortativity_coefficient(G, attribute, nodes=None):
    """Compute assortativity for node attributes.

    Assortativity measures the similarity of connections
    in the graph with respect to the given attribute.

    Parameters
    ----------
    G : NetworkX graph

    attribute : string
        Node attribute key

    nodes: list or iterable (optional)
        Compute attribute assortativity for nodes in container.
        The default is all nodes.

    Returns
    -------
    r: float
       Assortativity of graph for given attribute

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], color="red")
    >>> G.add_nodes_from([2, 3], color="blue")
    >>> G.add_edges_from([(0, 1), (2, 3)])
    >>> print(nx.attribute_assortativity_coefficient(G, "color"))
    1.0

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(M)-sum(M^2))/(1-sum(M^2)),
    where M is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    M = attribute_mixing_matrix(G, attribute, nodes)
    return attribute_ac(M)


def numeric_assortativity_coefficient(G, attribute, nodes=None):
    """Compute assortativity for numerical node attributes.

    Assortativity measures the similarity of connections
    in the graph with respect to the given numeric attribute.

    Parameters
    ----------
    G : NetworkX graph

    attribute : string
        Node attribute key.

    nodes: list or iterable (optional)
        Compute numeric assortativity only for attributes of nodes in
        container. The default is all nodes.

    Returns
    -------
    r: float
       Assortativity of graph for given attribute

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([0, 1], size=2)
    >>> G.add_nodes_from([2, 3], size=3)
    >>> G.add_edges_from([(0, 1), (2, 3)])
    >>> print(nx.numeric_assortativity_coefficient(G, "size"))
    1.0

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , for the mixing matrix
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks
           Physical Review E, 67 026126, 2003
    """
    if nodes is None:
        nodes = G.nodes
    vals = set(G.nodes[n][attribute] for n in nodes)
    mapping = {d: i for i, d, in enumerate(vals)}
    M = attribute_mixing_matrix(G, attribute, nodes, mapping)
    return numeric_ac(M, mapping)


def attribute_ac(M):
    """Compute assortativity for attribute matrix M.

    Parameters
    ----------
    M : numpy.ndarray
        2D ndarray representing the attribute mixing matrix.

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(e)-sum(e^2))/(1-sum(e^2)),
    where e is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    if M.sum() != 1.0:
        M = M / M.sum()
    s = (M @ M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return r


def numeric_ac(M, mapping):
    # M is a numpy matrix or array
    # numeric assortativity coefficient, pearsonr
    import numpy as np

    if M.sum() != 1.0:
        M = M / float(M.sum())
    nx, ny = M.shape  # nx=ny
    x = np.array(list(mapping.keys()))
    y = x  # x and y have the same support
    idx = list(mapping.values())
    a = M.sum(axis=0)
    b = M.sum(axis=1)
    vara = (a[idx] * x ** 2).sum() - ((a[idx] * x).sum()) ** 2
    varb = (b[idx] * y ** 2).sum() - ((b[idx] * y).sum()) ** 2
    xy = np.outer(x, y)
    ab = np.outer(a[idx], b[idx])
    return (xy * (M - ab)).sum() / np.sqrt(vara * varb)
