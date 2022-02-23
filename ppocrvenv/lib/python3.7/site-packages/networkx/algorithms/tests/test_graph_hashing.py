import networkx as nx


def test_empty_graph_hash():
    G1 = nx.empty_graph()
    G2 = nx.empty_graph()

    h1 = nx.weisfeiler_lehman_graph_hash(G1)
    h2 = nx.weisfeiler_lehman_graph_hash(G2)

    assert h1 == h2


def test_relabel():
    G1 = nx.Graph()
    G1.add_edges_from(
        [
            (1, 2, {"label": "A"}),
            (2, 3, {"label": "A"}),
            (3, 1, {"label": "A"}),
            (1, 4, {"label": "B"}),
        ]
    )
    h_before = nx.weisfeiler_lehman_graph_hash(G1, edge_attr="label")

    G2 = nx.relabel_nodes(G1, {u: -1 * u for u in G1.nodes()})

    h_after = nx.weisfeiler_lehman_graph_hash(G2, edge_attr="label")

    assert h_after == h_before


def test_directed():
    G1 = nx.DiGraph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 5)])

    h_directed = nx.weisfeiler_lehman_graph_hash(G1)

    G2 = G1.to_undirected()
    h_undirected = nx.weisfeiler_lehman_graph_hash(G2)

    assert h_directed != h_undirected
