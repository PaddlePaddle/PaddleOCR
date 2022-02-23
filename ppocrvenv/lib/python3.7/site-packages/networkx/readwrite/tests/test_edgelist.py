"""
    Unit tests for edgelists.
"""
import pytest
import io
import tempfile
import os
import textwrap

import networkx as nx
from networkx.utils import nodes_equal, edges_equal, graphs_equal


edges_no_data = textwrap.dedent(
    """
    # comment line
    1 2
    # comment line
    2 3
    """
)


edges_with_values = textwrap.dedent(
    """
    # comment line
    1 2 2.0
    # comment line
    2 3 3.0
    """
)


edges_with_weight = textwrap.dedent(
    """
    # comment line
    1 2 {'weight':2.0}
    # comment line
    2 3 {'weight':3.0}
    """
)


edges_with_multiple_attrs = textwrap.dedent(
    """
    # comment line
    1 2 {'weight':2.0, 'color':'green'}
    # comment line
    2 3 {'weight':3.0, 'color':'red'}
    """
)


edges_with_multiple_attrs_csv = textwrap.dedent(
    """
    # comment line
    1, 2, {'weight':2.0, 'color':'green'}
    # comment line
    2, 3, {'weight':3.0, 'color':'red'}
    """
)


_expected_edges_weights = [(1, 2, {"weight": 2.0}), (2, 3, {"weight": 3.0})]
_expected_edges_multiattr = [
    (1, 2, {"weight": 2.0, "color": "green"}),
    (2, 3, {"weight": 3.0, "color": "red"}),
]


@pytest.mark.parametrize(
    ("data", "extra_kwargs"),
    (
        (edges_no_data, {}),
        (edges_with_values, {}),
        (edges_with_weight, {}),
        (edges_with_multiple_attrs, {}),
        (edges_with_multiple_attrs_csv, {"delimiter": ","}),
    ),
)
def test_read_edgelist_no_data(data, extra_kwargs):
    bytesIO = io.BytesIO(data.encode("utf-8"))
    G = nx.read_edgelist(bytesIO, nodetype=int, data=False, **extra_kwargs)
    assert edges_equal(G.edges(), [(1, 2), (2, 3)])


def test_read_weighted_edgelist():
    bytesIO = io.BytesIO(edges_with_values.encode("utf-8"))
    G = nx.read_weighted_edgelist(bytesIO, nodetype=int)
    assert edges_equal(G.edges(data=True), _expected_edges_weights)


@pytest.mark.parametrize(
    ("data", "extra_kwargs", "expected"),
    (
        (edges_with_weight, {}, _expected_edges_weights),
        (edges_with_multiple_attrs, {}, _expected_edges_multiattr),
        (edges_with_multiple_attrs_csv, {"delimiter": ","}, _expected_edges_multiattr),
    ),
)
def test_read_edgelist_with_data(data, extra_kwargs, expected):
    bytesIO = io.BytesIO(data.encode("utf-8"))
    G = nx.read_edgelist(bytesIO, nodetype=int, **extra_kwargs)
    assert edges_equal(G.edges(data=True), expected)


@pytest.fixture
def example_graph():
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 3.0), (2, 3, 27.0), (3, 4, 3.0)])
    return G


def test_parse_edgelist_no_data(example_graph):
    G = example_graph
    H = nx.parse_edgelist(["1 2", "2 3", "3 4"], nodetype=int)
    assert nodes_equal(G.nodes, H.nodes)
    assert edges_equal(G.edges, H.edges)


def test_parse_edgelist_with_data_dict(example_graph):
    G = example_graph
    H = nx.parse_edgelist(
        ["1 2 {'weight': 3}", "2 3 {'weight': 27}", "3 4 {'weight': 3.0}"],
        nodetype=int,
    )
    assert nodes_equal(G.nodes, H.nodes)
    assert edges_equal(G.edges(data=True), H.edges(data=True))


def test_parse_edgelist_with_data_list(example_graph):
    G = example_graph
    H = nx.parse_edgelist(
        ["1 2 3", "2 3 27", "3 4 3.0"], nodetype=int, data=(("weight", float),)
    )
    assert nodes_equal(G.nodes, H.nodes)
    assert edges_equal(G.edges(data=True), H.edges(data=True))


def test_parse_edgelist():
    # ignore lines with less than 2 nodes
    lines = ["1;2", "2 3", "3 4"]
    G = nx.parse_edgelist(lines, nodetype=int)
    assert list(G.edges()) == [(2, 3), (3, 4)]
    # unknown nodetype
    with pytest.raises(TypeError, match="Failed to convert nodes"):
        lines = ["1 2", "2 3", "3 4"]
        nx.parse_edgelist(lines, nodetype="nope")
    # lines have invalid edge format
    with pytest.raises(TypeError, match="Failed to convert edge data"):
        lines = ["1 2 3", "2 3", "3 4"]
        nx.parse_edgelist(lines, nodetype=int)
    # edge data and data_keys not the same length
    with pytest.raises(IndexError, match="not the same length"):
        lines = ["1 2 3", "2 3 27", "3 4 3.0"]
        nx.parse_edgelist(
            lines, nodetype=int, data=(("weight", float), ("capacity", int))
        )
    # edge data can't be converted to edge type
    with pytest.raises(TypeError, match="Failed to convert"):
        lines = ["1 2 't1'", "2 3 't3'", "3 4 't3'"]
        nx.parse_edgelist(lines, nodetype=int, data=(("weight", float),))


class TestEdgelist:
    @classmethod
    def setup_class(cls):
        cls.G = nx.Graph(name="test")
        e = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("a", "f")]
        cls.G.add_edges_from(e)
        cls.G.add_node("g")
        cls.DG = nx.DiGraph(cls.G)
        cls.XG = nx.MultiGraph()
        cls.XG.add_weighted_edges_from([(1, 2, 5), (1, 2, 5), (1, 2, 1), (3, 3, 42)])
        cls.XDG = nx.MultiDiGraph(cls.XG)

    def test_write_edgelist_1(self):
        fh = io.BytesIO()
        G = nx.OrderedGraph()
        G.add_edges_from([(1, 2), (2, 3)])
        nx.write_edgelist(G, fh, data=False)
        fh.seek(0)
        assert fh.read() == b"1 2\n2 3\n"

    def test_write_edgelist_2(self):
        fh = io.BytesIO()
        G = nx.OrderedGraph()
        G.add_edges_from([(1, 2), (2, 3)])
        nx.write_edgelist(G, fh, data=True)
        fh.seek(0)
        assert fh.read() == b"1 2 {}\n2 3 {}\n"

    def test_write_edgelist_3(self):
        fh = io.BytesIO()
        G = nx.OrderedGraph()
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        nx.write_edgelist(G, fh, data=True)
        fh.seek(0)
        assert fh.read() == b"1 2 {'weight': 2.0}\n2 3 {'weight': 3.0}\n"

    def test_write_edgelist_4(self):
        fh = io.BytesIO()
        G = nx.OrderedGraph()
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        nx.write_edgelist(G, fh, data=[("weight")])
        fh.seek(0)
        assert fh.read() == b"1 2 2.0\n2 3 3.0\n"

    def test_unicode(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, "Radiohead", **{name2: 3})
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname)
        assert graphs_equal(G, H)
        os.close(fd)
        os.unlink(fname)

    def test_latin1_issue(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, "Radiohead", **{name2: 3})
        fd, fname = tempfile.mkstemp()
        pytest.raises(
            UnicodeEncodeError, nx.write_edgelist, G, fname, encoding="latin-1"
        )
        os.close(fd)
        os.unlink(fname)

    def test_latin1(self):
        G = nx.Graph()
        name1 = "Bj" + chr(246) + "rk"
        name2 = chr(220) + "ber"
        G.add_edge(name1, "Radiohead", **{name2: 3})
        fd, fname = tempfile.mkstemp()
        nx.write_edgelist(G, fname, encoding="latin-1")
        H = nx.read_edgelist(fname, encoding="latin-1")
        assert graphs_equal(G, H)
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_graph(self):
        G = self.G
        (fd, fname) = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname)
        H2 = nx.read_edgelist(fname)
        assert H is not H2  # they should be different graphs
        G.remove_node("g")  # isolated nodes are not written in edgelist
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_digraph(self):
        G = self.DG
        (fd, fname) = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, create_using=nx.DiGraph())
        H2 = nx.read_edgelist(fname, create_using=nx.DiGraph())
        assert H is not H2  # they should be different graphs
        G.remove_node("g")  # isolated nodes are not written in edgelist
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_integers(self):
        G = nx.convert_node_labels_to_integers(self.G)
        (fd, fname) = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, nodetype=int)
        # isolated nodes are not written in edgelist
        G.remove_nodes_from(list(nx.isolates(G)))
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_multigraph(self):
        G = self.XG
        (fd, fname) = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiGraph())
        H2 = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiGraph())
        assert H is not H2  # they should be different graphs
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_multidigraph(self):
        G = self.XDG
        (fd, fname) = tempfile.mkstemp()
        nx.write_edgelist(G, fname)
        H = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiDiGraph())
        H2 = nx.read_edgelist(fname, nodetype=int, create_using=nx.MultiDiGraph())
        assert H is not H2  # they should be different graphs
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)
