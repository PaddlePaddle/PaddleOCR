import networkx as nx


def small_ego_G():
    """The sample network from https://arxiv.org/pdf/1310.6753v1.pdf"""
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("b", "d"),
        ("b", "e"),
        ("b", "f"),
        ("c", "d"),
        ("c", "f"),
        ("c", "h"),
        ("d", "f"),
        ("e", "f"),
        ("f", "h"),
        ("h", "j"),
        ("h", "k"),
        ("i", "j"),
        ("i", "k"),
        ("j", "k"),
        ("u", "a"),
        ("u", "b"),
        ("u", "c"),
        ("u", "d"),
        ("u", "e"),
        ("u", "f"),
        ("u", "g"),
        ("u", "h"),
        ("u", "i"),
        ("u", "j"),
        ("u", "k"),
    ]
    G = nx.Graph()
    G.add_edges_from(edges)

    return G


class TestDispersion:
    def test_article(self):
        """our algorithm matches article's"""
        G = small_ego_G()
        disp_uh = nx.dispersion(G, "u", "h", normalized=False)
        disp_ub = nx.dispersion(G, "u", "b", normalized=False)
        assert disp_uh == 4
        assert disp_ub == 1

    def test_results_length(self):
        """there is a result for every node"""
        G = small_ego_G()
        disp = nx.dispersion(G)
        disp_Gu = nx.dispersion(G, "u")
        disp_uv = nx.dispersion(G, "u", "h")
        assert len(disp) == len(G)
        assert len(disp_Gu) == len(G) - 1
        assert type(disp_uv) is float

    def test_impossible_things(self):
        G = nx.karate_club_graph()
        disp = nx.dispersion(G)
        for u in disp:
            for v in disp[u]:
                assert disp[u][v] >= 0
