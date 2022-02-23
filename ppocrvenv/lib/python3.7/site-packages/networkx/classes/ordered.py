"""

.. deprecated:: 2.6

   The ordered variants of graph classes in this module are deprecated and
   will be removed in version 3.0.

Consistently ordered variants of the default base classes.
Note that if you are using Python 3.6+, you shouldn't need these classes
because the dicts in Python 3.6+ are ordered.
Note also that there are many differing expectations for the word "ordered"
and that these classes may not provide the order you expect.
The intent here is to give a consistent order not a particular order.

The Ordered (Di/Multi/MultiDi) Graphs give a consistent order for reporting of
nodes and edges.  The order of node reporting agrees with node adding, but for
edges, the order is not necessarily the order that the edges were added.

In general, you should use the default (i.e., unordered) graph classes.
However, there are times (e.g., when testing) when you may need the
order preserved.

Special care is required when using subgraphs of the Ordered classes.
The order of nodes in the subclass is not necessarily the same order
as the original class.  In general it is probably better to avoid using
subgraphs and replace with code similar to:

.. code-block:: python

    # instead of SG = G.subgraph(ordered_nodes)
    SG = nx.OrderedGraph()
    SG.add_nodes_from(ordered_nodes)
    SG.add_edges_from((u, v) for (u, v) in G.edges() if u in SG if v in SG)

"""
from collections import OrderedDict
import warnings

from .graph import Graph
from .multigraph import MultiGraph
from .digraph import DiGraph
from .multidigraph import MultiDiGraph

__all__ = []

__all__.extend(
    ["OrderedGraph", "OrderedDiGraph", "OrderedMultiGraph", "OrderedMultiDiGraph"]
)


class OrderedGraph(Graph):
    """Consistently ordered variant of :class:`~networkx.Graph`.

    .. deprecated:: 2.6

       OrderedGraph is deprecated and will be removed in version 3.0.
       Use `Graph` instead, which guarantees order is preserved for
       Python >= 3.7
    """

    node_dict_factory = OrderedDict
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict

    def __init__(self, incoming_graph_data=None, **attr):
        warnings.warn(
            (
                "OrderedGraph is deprecated and will be removed in version 3.0.\n"
                "Use `Graph` instead, which guarantees order is preserved for\n"
                "Python >= 3.7\n"
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super(OrderedGraph, self).__init__(incoming_graph_data, **attr)


class OrderedDiGraph(DiGraph):
    """Consistently ordered variant of :class:`~networkx.DiGraph`.

    .. deprecated:: 2.6

       OrderedDiGraph is deprecated and will be removed in version 3.0.
       Use `DiGraph` instead, which guarantees order is preserved for
       Python >= 3.7
    """

    node_dict_factory = OrderedDict
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict

    def __init__(self, incoming_graph_data=None, **attr):
        warnings.warn(
            (
                "OrderedDiGraph is deprecated and will be removed in version 3.0.\n"
                "Use `DiGraph` instead, which guarantees order is preserved for\n"
                "Python >= 3.7\n"
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super(OrderedDiGraph, self).__init__(incoming_graph_data, **attr)


class OrderedMultiGraph(MultiGraph):
    """Consistently ordered variant of :class:`~networkx.MultiGraph`.

    .. deprecated:: 2.6

       OrderedMultiGraph is deprecated and will be removed in version 3.0.
       Use `MultiGraph` instead, which guarantees order is preserved for
       Python >= 3.7
    """

    node_dict_factory = OrderedDict
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    edge_key_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict

    def __init__(self, incoming_graph_data=None, **attr):
        warnings.warn(
            (
                "OrderedMultiGraph is deprecated and will be removed in version 3.0.\n"
                "Use `MultiGraph` instead, which guarantees order is preserved for\n"
                "Python >= 3.7\n"
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super(OrderedMultiGraph, self).__init__(incoming_graph_data, **attr)


class OrderedMultiDiGraph(MultiDiGraph):
    """Consistently ordered variant of :class:`~networkx.MultiDiGraph`.

    .. deprecated:: 2.6

       OrderedMultiDiGraph is deprecated and will be removed in version 3.0.
       Use `MultiDiGraph` instead, which guarantees order is preserved for
       Python >= 3.7
    """

    node_dict_factory = OrderedDict
    adjlist_outer_dict_factory = OrderedDict
    adjlist_inner_dict_factory = OrderedDict
    edge_key_dict_factory = OrderedDict
    edge_attr_dict_factory = OrderedDict

    def __init__(self, incoming_graph_data=None, **attr):
        warnings.warn(
            (
                "OrderedMultiDiGraph is deprecated and will be removed in version 3.0.\n"
                "Use `MultiDiGraph` instead, which guarantees order is preserved for\n"
                "Python >= 3.7\n"
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super(OrderedMultiDiGraph, self).__init__(incoming_graph_data, **attr)
