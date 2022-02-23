"""
Functions for hashing graphs to strings.
Isomorphic graphs should be assigned identical hashes.
For now, only Weisfeiler-Lehman hashing is implemented.
"""

from collections import Counter
from hashlib import blake2b

__all__ = ["weisfeiler_lehman_graph_hash"]


def weisfeiler_lehman_graph_hash(
    G, edge_attr=None, node_attr=None, iterations=3, digest_size=16
):
    """Return Weisfeiler Lehman (WL) graph hash.

    The function iteratively aggregates and hashes neighbourhoods of each node.
    After each node's neighbors are hashed to obtain updated node labels,
    a hashed histogram of resulting labels is returned as the final hash.

    Hashes are identical for isomorphic graphs and strong guarantees that
    non-isomorphic graphs will get different hashes. See [1] for details.

    Note: Similarity between hashes does not imply similarity between graphs.

    If no node or edge attributes are provided, the degree of each node
    is used as its initial label.
    Otherwise, node and/or edge labels are used to compute the hash.

    Parameters
    ----------
    G: graph
        The graph to be hashed.
        Can have node and/or edge attributes. Can also have no attributes.
    edge_attr: string
        The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    node_attr: string
        The key in node attribute dictionary to be used for hashing.
        If None, and no edge_attr given, use
        degree of node as label.
    iterations: int
        Number of neighbor aggregations to perform.
        Should be larger for larger graphs.
    digest_size: int
        Size of blake2b hash digest to use for hashing node labels.

    Returns
    -------
    h : string
        Hexadecimal string corresponding to hash of the input graph.

    Examples
    --------
    Two graphs with edge attributes that are isomorphic, except for
    differences in the edge labels.

    >>> G1 = nx.Graph()
    >>> G1.add_edges_from(
    ...     [
    ...         (1, 2, {"label": "A"}),
    ...         (2, 3, {"label": "A"}),
    ...         (3, 1, {"label": "A"}),
    ...         (1, 4, {"label": "B"}),
    ...     ]
    ... )
    >>> G2 = nx.Graph()
    >>> G2.add_edges_from(
    ...     [
    ...         (5, 6, {"label": "B"}),
    ...         (6, 7, {"label": "A"}),
    ...         (7, 5, {"label": "A"}),
    ...         (7, 8, {"label": "A"}),
    ...     ]
    ... )

    Omitting the `edge_attr` option, results in identical hashes.

    >>> nx.weisfeiler_lehman_graph_hash(G1)
    '0db442538bb6dc81d675bd94e6ebb7ca'
    >>> nx.weisfeiler_lehman_graph_hash(G2)
    '0db442538bb6dc81d675bd94e6ebb7ca'

    With edge labels, the graphs are no longer assigned
    the same hash digest.

    >>> nx.weisfeiler_lehman_graph_hash(G1, edge_attr="label")
    '408c18537e67d3e56eb7dc92c72cb79e'
    >>> nx.weisfeiler_lehman_graph_hash(G2, edge_attr="label")
    'f9e9cb01c6d2f3b17f83ffeaa24e5986'

    References
    ----------
    .. [1] Shervashidze, Nino, Pascal Schweitzer, Erik Jan Van Leeuwen,
       Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler Lehman
       Graph Kernels. Journal of Machine Learning Research. 2011.
       http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
    """

    def neighborhood_aggregate(G, node, node_labels, edge_attr=None):
        """
        Compute new labels for given node by aggregating
        the labels of each node's neighbors.
        """
        label_list = [node_labels[node]]
        for nei in G.neighbors(node):
            prefix = "" if not edge_attr else G[node][nei][edge_attr]
            label_list.append(prefix + node_labels[nei])
        return "".join(sorted(label_list))

    def weisfeiler_lehman_step(G, labels, edge_attr=None, node_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = dict()
        for node in G.nodes():
            new_labels[node] = neighborhood_aggregate(
                G, node, labels, edge_attr=edge_attr
            )
        return new_labels

    items = []
    node_labels = dict()
    # set initial node labels
    for node in G.nodes():
        if (not node_attr) and (not edge_attr):
            node_labels[node] = str(G.degree(node))
        elif node_attr:
            node_labels[node] = str(G.nodes[node][node_attr])
        else:
            node_labels[node] = ""

    for k in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)
        counter = Counter()
        # count node labels
        for node, d in node_labels.items():
            h = blake2b(digest_size=digest_size)
            h.update(d.encode("ascii"))
            counter.update([h.hexdigest()])
        # sort the counter, extend total counts
        items.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    h = blake2b(digest_size=digest_size)
    h.update(str(tuple(items)).encode("ascii"))
    h = h.hexdigest()
    return h
