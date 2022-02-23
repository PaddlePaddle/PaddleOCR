"""
    Unit tests for bipartite edgelists.
"""
import pytest
import io
import tempfile
import os

import networkx as nx
from networkx.utils import nodes_equal, edges_equal, graphs_equal
from networkx.algorithms import bipartite


class TestEdgelist:
    @classmethod
    def setup_class(cls):
        cls.G = nx.Graph(name="test")
        e = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("a", "f")]
        cls.G.add_edges_from(e)
        cls.G.add_nodes_from(["a", "c", "e"], bipartite=0)
        cls.G.add_nodes_from(["b", "d", "f"], bipartite=1)
        cls.G.add_node("g", bipartite=0)
        cls.DG = nx.DiGraph(cls.G)
        cls.MG = nx.MultiGraph()
        cls.MG.add_edges_from([(1, 2), (1, 2), (1, 2)])
        cls.MG.add_node(1, bipartite=0)
        cls.MG.add_node(2, bipartite=1)

    def test_read_edgelist_1(self):
        s = b"""\
# comment line
1 2
# comment line
2 3
"""
        bytesIO = io.BytesIO(s)
        G = bipartite.read_edgelist(bytesIO, nodetype=int)
        assert edges_equal(G.edges(), [(1, 2), (2, 3)])

    def test_read_edgelist_3(self):
        s = b"""\
# comment line
1 2 {'weight':2.0}
# comment line
2 3 {'weight':3.0}
"""
        bytesIO = io.BytesIO(s)
        G = bipartite.read_edgelist(bytesIO, nodetype=int, data=False)
        assert edges_equal(G.edges(), [(1, 2), (2, 3)])

        bytesIO = io.BytesIO(s)
        G = bipartite.read_edgelist(bytesIO, nodetype=int, data=True)
        assert edges_equal(
            G.edges(data=True), [(1, 2, {"weight": 2.0}), (2, 3, {"weight": 3.0})]
        )

    def test_write_edgelist_1(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        G.add_node(1, bipartite=0)
        G.add_node(2, bipartite=1)
        G.add_node(3, bipartite=0)
        bipartite.write_edgelist(G, fh, data=False)
        fh.seek(0)
        assert fh.read() == b"1 2\n3 2\n"

    def test_write_edgelist_2(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        G.add_node(1, bipartite=0)
        G.add_node(2, bipartite=1)
        G.add_node(3, bipartite=0)
        bipartite.write_edgelist(G, fh, data=True)
        fh.seek(0)
        assert fh.read() == b"1 2 {}\n3 2 {}\n"

    def test_write_edgelist_3(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        G.add_node(1, bipartite=0)
        G.add_node(2, bipartite=1)
        G.add_node(3, bipartite=0)
        bipartite.write_edgelist(G, fh, data=True)
        fh.seek(0)
        assert fh.read() == b"1 2 {'weight': 2.0}\n3 2 {'weight': 3.0}\n"

    def test_write_edgelist_4(self):
        fh = io.BytesIO()
        G = nx.Graph()
        G.add_edge(1, 2, weight=2.0)
        G.add_edge(2, 3, weight=3.0)
        G.add_node(1, bipartite=0)
        G.add_node(2, bipartite=1)
        G.add_node(3, bipartite=0)
        bipartite.write_edgelist(G, fh, data=[("weight")])
        fh.seek(0)
        assert fh.read() == b"1 2 2.0\n3 2 3.0\n"

    def test_unicode(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, "Radiohead", **{name2: 3})
        G.add_node(name1, bipartite=0)
        G.add_node("Radiohead", bipartite=1)
        fd, fname = tempfile.mkstemp()
        bipartite.write_edgelist(G, fname)
        H = bipartite.read_edgelist(fname)
        assert graphs_equal(G, H)
        os.close(fd)
        os.unlink(fname)

    def test_latin1_issue(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        G.add_edge(name1, "Radiohead", **{name2: 3})
        G.add_node(name1, bipartite=0)
        G.add_node("Radiohead", bipartite=1)
        fd, fname = tempfile.mkstemp()
        pytest.raises(
            UnicodeEncodeError, bipartite.write_edgelist, G, fname, encoding="latin-1"
        )
        os.close(fd)
        os.unlink(fname)

    def test_latin1(self):
        G = nx.Graph()
        name1 = "Bj" + chr(246) + "rk"
        name2 = chr(220) + "ber"
        G.add_edge(name1, "Radiohead", **{name2: 3})
        G.add_node(name1, bipartite=0)
        G.add_node("Radiohead", bipartite=1)
        fd, fname = tempfile.mkstemp()
        bipartite.write_edgelist(G, fname, encoding="latin-1")
        H = bipartite.read_edgelist(fname, encoding="latin-1")
        assert graphs_equal(G, H)
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_graph(self):
        G = self.G
        (fd, fname) = tempfile.mkstemp()
        bipartite.write_edgelist(G, fname)
        H = bipartite.read_edgelist(fname)
        H2 = bipartite.read_edgelist(fname)
        assert H is not H2  # they should be different graphs
        G.remove_node("g")  # isolated nodes are not written in edgelist
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_integers(self):
        G = nx.convert_node_labels_to_integers(self.G)
        (fd, fname) = tempfile.mkstemp()
        bipartite.write_edgelist(G, fname)
        H = bipartite.read_edgelist(fname, nodetype=int)
        # isolated nodes are not written in edgelist
        G.remove_nodes_from(list(nx.isolates(G)))
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_edgelist_multigraph(self):
        G = self.MG
        (fd, fname) = tempfile.mkstemp()
        bipartite.write_edgelist(G, fname)
        H = bipartite.read_edgelist(fname, nodetype=int, create_using=nx.MultiGraph())
        H2 = bipartite.read_edgelist(fname, nodetype=int, create_using=nx.MultiGraph())
        assert H is not H2  # they should be different graphs
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_empty_digraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            bytesIO = io.BytesIO()
            bipartite.write_edgelist(nx.DiGraph(), bytesIO)

    def test_raise_attribute(self):
        with pytest.raises(AttributeError):
            G = nx.path_graph(4)
            bytesIO = io.BytesIO()
            bipartite.write_edgelist(G, bytesIO)
