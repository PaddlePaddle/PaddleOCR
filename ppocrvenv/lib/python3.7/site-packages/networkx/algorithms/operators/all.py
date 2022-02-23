"""Operations on many graphs.
"""
from itertools import zip_longest
import networkx as nx

__all__ = ["union_all", "compose_all", "disjoint_union_all", "intersection_all"]


def union_all(graphs, rename=(None,)):
    """Returns the union of all graphs.

    The graphs must be disjoint, otherwise an exception is raised.

    Parameters
    ----------
    graphs : list of graphs
       List of NetworkX graphs

    rename : bool , default=(None, None)
       Node names of G and H can be changed by specifying the tuple
       rename=('G-','H-') (for example).  Node "u" in G is then renamed
       "G-u" and "v" in H is renamed "H-v".

    Returns
    -------
    U : a graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    Notes
    -----
    To force a disjoint union with node relabeling, use
    disjoint_union_all(G,H) or convert_node_labels_to integers().

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.

    See Also
    --------
    union
    disjoint_union_all
    """
    if not graphs:
        raise ValueError("cannot apply union_all to an empty list")
    graphs_names = zip_longest(graphs, rename)
    U, gname = next(graphs_names)
    for H, hname in graphs_names:
        U = nx.union(U, H, (gname, hname))
        gname = None
    return U


def disjoint_union_all(graphs):
    """Returns the disjoint union of all graphs.

    This operation forces distinct integer node labels starting with 0
    for the first graph in the list and numbering consecutively.

    Parameters
    ----------
    graphs : list
       List of NetworkX graphs

    Returns
    -------
    U : A graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    Notes
    -----
    It is recommended that the graphs be either all directed or all undirected.

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.
    """
    if not graphs:
        raise ValueError("cannot apply disjoint_union_all to an empty list")
    graphs = iter(graphs)
    U = next(graphs)
    for H in graphs:
        U = nx.disjoint_union(U, H)
    return U


def compose_all(graphs):
    """Returns the composition of all graphs.

    Composition is the simple union of the node sets and edge sets.
    The node sets of the supplied graphs need not be disjoint.

    Parameters
    ----------
    graphs : list
       List of NetworkX graphs

    Returns
    -------
    C : A graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    Notes
    -----
    It is recommended that the supplied graphs be either all directed or all
    undirected.

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.
    """
    if not graphs:
        raise ValueError("cannot apply compose_all to an empty list")
    graphs = iter(graphs)
    C = next(graphs)
    for H in graphs:
        C = nx.compose(C, H)
    return C


def intersection_all(graphs):
    """Returns a new graph that contains only the nodes and the edges that exist in
    all graphs.

    Parameters
    ----------
    graphs : list
       List of NetworkX graphs

    Returns
    -------
    R : A new graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    Notes
    -----
    Attributes from the graph, nodes, and edges are not copied to the new
    graph.
    """
    if not graphs:
        raise ValueError("cannot apply intersection_all to an empty list")
    graphs = iter(graphs)
    R = next(graphs)
    for H in graphs:
        R = nx.intersection(R, H)
    return R
