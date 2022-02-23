"""Unit tests for the :mod:`networkx.algorithms.triads` module."""

import networkx as nx
from collections import defaultdict
from random import sample


def test_triadic_census():
    """Tests the triadic_census function."""
    G = nx.DiGraph()
    G.add_edges_from(["01", "02", "03", "04", "05", "12", "16", "51", "56", "65"])
    expected = {
        "030T": 2,
        "120C": 1,
        "210": 0,
        "120U": 0,
        "012": 9,
        "102": 3,
        "021U": 0,
        "111U": 0,
        "003": 8,
        "030C": 0,
        "021D": 9,
        "201": 0,
        "111D": 1,
        "300": 0,
        "120D": 0,
        "021C": 2,
    }
    actual = nx.triadic_census(G)
    assert expected == actual


def test_is_triad():
    """Tests the is_triad function"""
    G = nx.karate_club_graph()
    G = G.to_directed()
    for i in range(100):
        nodes = sample(sorted(G.nodes()), 3)
        G2 = G.subgraph(nodes)
        assert nx.is_triad(G2)


def test_all_triplets():
    """Tests the all_triplets function."""
    G = nx.DiGraph()
    G.add_edges_from(["01", "02", "03", "04", "05", "12", "16", "51", "56", "65"])
    expected = [
        f"{i},{j},{k}"
        for i in range(7)
        for j in range(i + 1, 7)
        for k in range(j + 1, 7)
    ]
    expected = [set(x.split(",")) for x in expected]
    actual = list(set(x) for x in nx.all_triplets(G))
    assert all([any([s1 == s2 for s1 in expected]) for s2 in actual])


def test_all_triads():
    """Tests the all_triplets function."""
    G = nx.DiGraph()
    G.add_edges_from(["01", "02", "03", "04", "05", "12", "16", "51", "56", "65"])
    expected = [
        f"{i},{j},{k}"
        for i in range(7)
        for j in range(i + 1, 7)
        for k in range(j + 1, 7)
    ]
    expected = [G.subgraph(x.split(",")) for x in expected]
    actual = list(nx.all_triads(G))
    assert all(any([nx.is_isomorphic(G1, G2) for G1 in expected]) for G2 in actual)


def test_triad_type():
    """Tests the triad_type function."""
    # 0 edges (1 type)
    G = nx.DiGraph({0: [], 1: [], 2: []})
    assert nx.triad_type(G) == "003"
    # 1 edge (1 type)
    G = nx.DiGraph({0: [1], 1: [], 2: []})
    assert nx.triad_type(G) == "012"
    # 2 edges (4 types)
    G = nx.DiGraph([(0, 1), (0, 2)])
    assert nx.triad_type(G) == "021D"
    G = nx.DiGraph({0: [1], 1: [0], 2: []})
    assert nx.triad_type(G) == "102"
    G = nx.DiGraph([(0, 1), (2, 1)])
    assert nx.triad_type(G) == "021U"
    G = nx.DiGraph([(0, 1), (1, 2)])
    assert nx.triad_type(G) == "021C"
    # 3 edges (4 types)
    G = nx.DiGraph([(0, 1), (1, 0), (2, 1)])
    assert nx.triad_type(G) == "111D"
    G = nx.DiGraph([(0, 1), (1, 0), (1, 2)])
    assert nx.triad_type(G) == "111U"
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    assert nx.triad_type(G) == "030T"
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    assert nx.triad_type(G) == "030C"
    # 4 edges (4 types)
    G = nx.DiGraph([(0, 1), (1, 0), (2, 0), (0, 2)])
    assert nx.triad_type(G) == "201"
    G = nx.DiGraph([(0, 1), (1, 0), (2, 0), (2, 1)])
    assert nx.triad_type(G) == "120D"
    G = nx.DiGraph([(0, 1), (1, 0), (0, 2), (1, 2)])
    assert nx.triad_type(G) == "120U"
    G = nx.DiGraph([(0, 1), (1, 0), (0, 2), (2, 1)])
    assert nx.triad_type(G) == "120C"
    # 5 edges (1 type)
    G = nx.DiGraph([(0, 1), (1, 0), (2, 1), (1, 2), (0, 2)])
    assert nx.triad_type(G) == "210"
    # 6 edges (1 type)
    G = nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
    assert nx.triad_type(G) == "300"


def test_triads_by_type():
    """Tests the all_triplets function."""
    G = nx.DiGraph()
    G.add_edges_from(["01", "02", "03", "04", "05", "12", "16", "51", "56", "65"])
    all_triads = nx.all_triads(G)
    expected = defaultdict(list)
    for triad in all_triads:
        name = nx.triad_type(triad)
        expected[name].append(triad)
    actual = nx.triads_by_type(G)
    assert set(actual.keys()) == set(expected.keys())
    for tri_type, actual_Gs in actual.items():
        expected_Gs = expected[tri_type]
        for a in actual_Gs:
            assert any(nx.is_isomorphic(a, e) for e in expected_Gs)


def test_random_triad():
    """Tests the random_triad function"""
    G = nx.karate_club_graph()
    G = G.to_directed()
    for i in range(100):
        assert nx.is_triad(nx.random_triad(G))


def test_triadic_census_nodelist():
    """Tests the triadic_census function."""
    G = nx.DiGraph()
    G.add_edges_from(["01", "02", "03", "04", "05", "12", "16", "51", "56", "65"])
    expected = {
        "030T": 2,
        "120C": 1,
        "210": 0,
        "120U": 0,
        "012": 9,
        "102": 3,
        "021U": 0,
        "111U": 0,
        "003": 8,
        "030C": 0,
        "021D": 9,
        "201": 0,
        "111D": 1,
        "300": 0,
        "120D": 0,
        "021C": 2,
    }
    actual = {k: 0 for k in expected}
    for node in G.nodes():
        node_triad_census = nx.triadic_census(G, nodelist=[node])
        for triad_key in expected:
            actual[triad_key] += node_triad_census[triad_key]
    # Divide the total count of 003 triads by 3, since we are counting them thrice
    actual["003"] //= 3
    assert expected == actual
