from itertools import chain
import networkx as nx

__all__ = ["tree_data", "tree_graph"]


# NOTE: Remove attrs from signature in 3.0
def tree_data(G, root, attrs=None, ident="id", children="children"):
    """Returns data in tree format that is suitable for JSON serialization
    and use in Javascript documents.

    Parameters
    ----------
    G : NetworkX graph
       G must be an oriented tree

    root : node
       The root of the tree

    attrs : dict
        A dictionary that contains two keys 'id' and 'children'. The
        corresponding values provide the attribute names for storing
        NetworkX-internal graph data. The values should be unique. Default
        value: :samp:`dict(id='id', children='children')`.

        If some user-defined graph data use these attribute names as data keys,
        they may be silently dropped.

        .. deprecated:: 2.6

           The `attrs` keyword argument is replaced by `ident` and `children`
           and will be removed in networkx 3.0

    ident : string
        Attribute name for storing NetworkX-internal graph data. `ident` must
        have a different value than `children`. The default is 'id'.

    children : string
        Attribute name for storing NetworkX-internal graph data. `children`
        must have a different value than `ident`. The default is 'children'.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If `children` and `ident` attributes are identical.

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1, 2)])
    >>> data = json_graph.tree_data(G, root=1)

    To serialize with json

    >>> import json
    >>> s = json.dumps(data)

    Notes
    -----
    Node attributes are stored in this format but keys
    for attributes must be strings if you want to serialize with JSON.

    Graph and edge attributes are not stored.

    See Also
    --------
    tree_graph, node_link_data, adjacency_data
    """
    if G.number_of_nodes() != G.number_of_edges() + 1:
        raise TypeError("G is not a tree.")
    if not G.is_directed():
        raise TypeError("G is not directed.")

    # NOTE: to be removed in 3.0
    if attrs is not None:
        import warnings

        msg = (
            "\nThe `attrs` keyword argument of tree_data is deprecated\n"
            "and will be removed in networkx 3.0.\n"
            "It is replaced with explicit `ident` and `children` "
            "keyword arguments.\n"
            "To make this warning go away and ensure usage is forward\n"
            "compatible, replace `attrs` with `ident` and `children,\n"
            "for example:\n\n"
            "    >>> tree_data(G, root, attrs={'id': 'foo', 'children': 'bar'})\n\n"
            "should instead be written as\n\n"
            "    >>> tree_data(G, root, ident='foo', children='bar')\n\n"
            "The default values of 'id' and 'children' will not change."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        ident = attrs["id"]
        children = attrs["children"]

    if ident == children:
        raise nx.NetworkXError("The values for `id` and `children` must be different.")

    def add_children(n, G):
        nbrs = G[n]
        if len(nbrs) == 0:
            return []
        children_ = []
        for child in nbrs:
            d = dict(chain(G.nodes[child].items(), [(ident, child)]))
            c = add_children(child, G)
            if c:
                d[children] = c
            children_.append(d)
        return children_

    data = dict(chain(G.nodes[root].items(), [(ident, root)]))
    data[children] = add_children(root, G)
    return data


def tree_graph(data, attrs=None, ident="id", children="children"):
    """Returns graph from tree data format.

    Parameters
    ----------
    data : dict
        Tree formatted graph data
    attrs : dict
        A dictionary that contains two keys 'id' and 'children'. The
        corresponding values provide the attribute names for storing
        NetworkX-internal graph data. The values should be unique. Default
        value: :samp:`dict(id='id', children='children')`.

        .. deprecated:: 2.6

           The `attrs` keyword argument is replaced by `ident` and `children`
           and will be removed in networkx 3.0

    ident : string
        Attribute name for storing NetworkX-internal graph data. `ident` must
        have a different value than `children`. The default is 'id'.

    children : string
        Attribute name for storing NetworkX-internal graph data. `children`
        must have a different value than `ident`. The default is 'children'.

    Returns
    -------
    G : NetworkX DiGraph

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1, 2)])
    >>> data = json_graph.tree_data(G, root=1)
    >>> H = json_graph.tree_graph(data)

    See Also
    --------
    tree_data, node_link_data, adjacency_data
    """
    graph = nx.DiGraph()
    if attrs is not None:
        import warnings

        msg = (
            "\nThe `attrs` keyword argument of tree_graph is deprecated\n"
            "and will be removed in networkx 3.0.\n"
            "It is replaced with explicit `ident` and `children` "
            "keyword arguments.\n"
            "To make this warning go away and ensure usage is\n"
            "forward compatible, replace `attrs` with `ident` and `children,\n"
            "for example:\n\n"
            "    >>> tree_graph(data, attrs={'id': 'foo', 'children': 'bar'})\n\n"
            "should instead be written as\n\n"
            "    >>> tree_graph(data, ident='foo', children='bar')\n\n"
            "The default values of 'id' and 'children' will not change."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        ident = attrs["id"]
        children = attrs["children"]

    def add_children(parent, children_):
        for data in children_:
            child = data[ident]
            graph.add_edge(parent, child)
            grandchildren = data.get(children, [])
            if grandchildren:
                add_children(child, grandchildren)
            nodedata = {
                str(k): v for k, v in data.items() if k != ident and k != children
            }
            graph.add_node(child, **nodedata)

    root = data[ident]
    children_ = data.get(children, [])
    nodedata = {str(k): v for k, v in data.items() if k != ident and k != children}
    graph.add_node(root, **nodedata)
    add_children(root, children_)
    return graph
