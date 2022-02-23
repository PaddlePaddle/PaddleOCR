import networkx as nx


def test_reversed():
    G = nx.DiGraph()
    G.add_edge("A", "B")

    # no exception
    with nx.utils.reversed(G):
        pass
    assert "B" in G["A"]

    # exception
    try:
        with nx.utils.reversed(G):
            raise Exception
    except:
        assert "B" in G["A"]
