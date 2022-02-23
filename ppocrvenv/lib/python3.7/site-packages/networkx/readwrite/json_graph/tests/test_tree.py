import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import tree_data, tree_graph


def test_graph():
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3], color="red")
    G.add_edge(1, 2, foo=7)
    G.add_edge(1, 3, foo=10)
    G.add_edge(3, 4, foo=10)
    H = tree_graph(tree_data(G, 1))
    nx.is_isomorphic(G, H)


def test_graph_attributes():
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3], color="red")
    G.add_edge(1, 2, foo=7)
    G.add_edge(1, 3, foo=10)
    G.add_edge(3, 4, foo=10)
    H = tree_graph(tree_data(G, 1))
    assert H.nodes[1]["color"] == "red"

    d = json.dumps(tree_data(G, 1))
    H = tree_graph(json.loads(d))
    assert H.nodes[1]["color"] == "red"


def test_exceptions():
    with pytest.raises(TypeError, match="is not a tree."):
        G = nx.complete_graph(3)
        tree_data(G, 0)
    with pytest.raises(TypeError, match="is not directed."):
        G = nx.path_graph(3)
        tree_data(G, 0)
    with pytest.raises(nx.NetworkXError, match="must be different."):
        G = nx.MultiDiGraph()
        G.add_node(0)
        tree_data(G, 0, ident="node", children="node")


# NOTE: To be removed when deprecation expires in 3.0
def test_attrs_deprecation():
    G = nx.path_graph(3, create_using=nx.DiGraph)
    # No warnings when `attrs` kwarg not used
    with pytest.warns(None) as record:
        data = tree_data(G, 0)
        H = tree_graph(data)
    assert len(record) == 0
    # DeprecationWarning issued when `attrs` is used
    attrs = {"id": "foo", "children": "bar"}
    with pytest.warns(DeprecationWarning):
        data = tree_data(G, 0, attrs=attrs)
    with pytest.warns(DeprecationWarning):
        H = tree_graph(data, attrs=attrs)
