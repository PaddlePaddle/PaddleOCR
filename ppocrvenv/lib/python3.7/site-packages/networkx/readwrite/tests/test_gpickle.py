import os
import tempfile

import networkx as nx
from networkx.utils import nodes_equal, edges_equal, graphs_equal


class TestGpickle:
    @classmethod
    def setup_class(cls):
        G = nx.Graph(name="test")
        e = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("a", "f")]
        G.add_edges_from(e, width=10)
        G.add_node("g", color="green")
        G.graph["number"] = 1
        DG = nx.DiGraph(G)
        MG = nx.MultiGraph(G)
        MG.add_edge("a", "a")
        MDG = nx.MultiDiGraph(G)
        MDG.add_edge("a", "a")
        fG = G.copy()
        fDG = DG.copy()
        fMG = MG.copy()
        fMDG = MDG.copy()
        nx.freeze(fG)
        nx.freeze(fDG)
        nx.freeze(fMG)
        nx.freeze(fMDG)
        cls.G = G
        cls.DG = DG
        cls.MG = MG
        cls.MDG = MDG
        cls.fG = fG
        cls.fDG = fDG
        cls.fMG = fMG
        cls.fMDG = fMDG

    def test_gpickle(self):
        for G in [
            self.G,
            self.DG,
            self.MG,
            self.MDG,
            self.fG,
            self.fDG,
            self.fMG,
            self.fMDG,
        ]:
            (fd, fname) = tempfile.mkstemp()
            nx.write_gpickle(G, fname)
            Gin = nx.read_gpickle(fname)
            assert nodes_equal(list(G.nodes(data=True)), list(Gin.nodes(data=True)))
            assert edges_equal(list(G.edges(data=True)), list(Gin.edges(data=True)))
            assert graphs_equal(G, Gin)
            os.close(fd)
            os.unlink(fname)

    def test_protocol(self):
        for G in [
            self.G,
            self.DG,
            self.MG,
            self.MDG,
            self.fG,
            self.fDG,
            self.fMG,
            self.fMDG,
        ]:
            with tempfile.TemporaryFile() as f:
                nx.write_gpickle(G, f, 0)
                f.seek(0)
                Gin = nx.read_gpickle(f)
                assert nodes_equal(list(G.nodes(data=True)), list(Gin.nodes(data=True)))
                assert edges_equal(list(G.edges(data=True)), list(Gin.edges(data=True)))
                assert graphs_equal(G, Gin)
