"""
Read and write NetworkX graphs as JavaScript InfoVis Toolkit (JIT) format JSON.

See the `JIT documentation`_ for more examples.

Format
------
var json = [
  {
    "id": "aUniqueIdentifier",
    "name": "usually a nodes name",
    "data": {
      "some key": "some value",
      "some other key": "some other value"
     },
    "adjacencies": [
    {
      nodeTo:"aNodeId",
      data: {} //put whatever you want here
    },
    'other adjacencies go here...'
  },

  'other nodes go here...'
];
.. _JIT documentation: http://thejit.org
"""

import json
import warnings
import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["jit_graph", "jit_data"]


def jit_graph(data, create_using=None):
    """Read a graph from JIT JSON.

    Parameters
    ----------
    data : JSON Graph Object

    create_using : Networkx Graph, optional (default: Graph())
        Return graph of this type. The provided instance will be cleared.

    Returns
    -------
    G : NetworkX Graph built from create_using if provided.

    .. deprecated:: 2.6
    """
    warnings.warn(
        ("jit_graph is deprecated and will be removed in NetworkX 3.0."),
        DeprecationWarning,
    )

    if create_using is None:
        G = nx.Graph()
    else:
        G = create_using
        G.clear()

    if isinstance(data, str):
        data = json.loads(data)

    for node in data:
        G.add_node(node["id"], **node["data"])
        if node.get("adjacencies") is not None:
            for adj in node["adjacencies"]:
                G.add_edge(node["id"], adj["nodeTo"], **adj["data"])
    return G


@not_implemented_for("multigraph")
def jit_data(G, indent=None, default=None):
    """Returns data in JIT JSON format.

    Parameters
    ----------
    G : NetworkX Graph

    indent: optional, default=None
        If indent is a non-negative integer, then JSON array elements and
        object members will be pretty-printed with that indent level.
        An indent level of 0, or negative, will only insert newlines.
        None (the default) selects the most compact representation.

    default: optional, default=None
         It will pass the value to the json.dumps function in order to
         be able to serialize custom objects used as nodes.

    Returns
    -------
    data: JIT JSON string

    .. deprecated:: 2.6
    """
    warnings.warn(
        ("jit_data is deprecated and will be removed in NetworkX 3.0."),
        DeprecationWarning,
    )
    json_graph = []
    for node in G.nodes():
        json_node = {"id": node, "name": node}
        # node data
        json_node["data"] = G.nodes[node]
        # adjacencies
        if G[node]:
            json_node["adjacencies"] = []
            for neighbour in G[node]:
                adjacency = {"nodeTo": neighbour}
                # adjacency data
                adjacency["data"] = G.edges[node, neighbour]
                json_node["adjacencies"].append(adjacency)
        json_graph.append(json_node)
    return json.dumps(json_graph, indent=indent, default=default)
