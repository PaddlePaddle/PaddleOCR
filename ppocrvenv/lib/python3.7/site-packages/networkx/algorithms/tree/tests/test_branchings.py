import pytest

np = pytest.importorskip("numpy")

import networkx as nx


from networkx.algorithms.tree import branchings
from networkx.algorithms.tree import recognition

#
# Explicitly discussed examples from Edmonds paper.
#

# Used in Figures A-F.
#
# fmt: off
G_array = np.array([
    # 0   1   2   3   4   5   6   7   8
    [0,  0, 12,  0, 12,  0,  0,  0,  0],  # 0
    [4,  0,  0,  0,  0, 13,  0,  0,  0],  # 1
    [0, 17,  0, 21,  0, 12,  0,  0,  0],  # 2
    [5,  0,  0,  0, 17,  0, 18,  0,  0],  # 3
    [0,  0,  0,  0,  0,  0,  0, 12,  0],  # 4
    [0,  0,  0,  0,  0,  0, 14,  0, 12],  # 5
    [0,  0, 21,  0,  0,  0,  0,  0, 15],  # 6
    [0,  0,  0, 19,  0,  0, 15,  0,  0],  # 7
    [0,  0,  0,  0,  0,  0,  0, 18,  0],  # 8
], dtype=int)
# fmt: on


def G1():
    G = nx.from_numpy_array(G_array, create_using=nx.MultiDiGraph)
    return G


def G2():
    # Now we shift all the weights by -10.
    # Should not affect optimal arborescence, but does affect optimal branching.
    Garr = G_array.copy()
    Garr[np.nonzero(Garr)] -= 10
    G = nx.from_numpy_array(Garr, create_using=nx.MultiDiGraph)
    return G


# An optimal branching for G1 that is also a spanning arborescence. So it is
# also an optimal spanning arborescence.
#
optimal_arborescence_1 = [
    (0, 2, 12),
    (2, 1, 17),
    (2, 3, 21),
    (1, 5, 13),
    (3, 4, 17),
    (3, 6, 18),
    (6, 8, 15),
    (8, 7, 18),
]

# For G2, the optimal branching of G1 (with shifted weights) is no longer
# an optimal branching, but it is still an optimal spanning arborescence
# (just with shifted weights). An optimal branching for G2 is similar to what
# appears in figure G (this is greedy_subopt_branching_1a below), but with the
# edge (3, 0, 5), which is now (3, 0, -5), removed. Thus, the optimal branching
# is not a spanning arborescence. The code finds optimal_branching_2a.
# An alternative and equivalent branching is optimal_branching_2b. We would
# need to modify the code to iterate through all equivalent optimal branchings.
#
# These are maximal branchings or arborescences.
optimal_branching_2a = [
    (5, 6, 4),
    (6, 2, 11),
    (6, 8, 5),
    (8, 7, 8),
    (2, 1, 7),
    (2, 3, 11),
    (3, 4, 7),
]
optimal_branching_2b = [
    (8, 7, 8),
    (7, 3, 9),
    (3, 4, 7),
    (3, 6, 8),
    (6, 2, 11),
    (2, 1, 7),
    (1, 5, 3),
]
optimal_arborescence_2 = [
    (0, 2, 2),
    (2, 1, 7),
    (2, 3, 11),
    (1, 5, 3),
    (3, 4, 7),
    (3, 6, 8),
    (6, 8, 5),
    (8, 7, 8),
]

# Two suboptimal maximal branchings on G1 obtained from a greedy algorithm.
# 1a matches what is shown in Figure G in Edmonds's paper.
greedy_subopt_branching_1a = [
    (5, 6, 14),
    (6, 2, 21),
    (6, 8, 15),
    (8, 7, 18),
    (2, 1, 17),
    (2, 3, 21),
    (3, 0, 5),
    (3, 4, 17),
]
greedy_subopt_branching_1b = [
    (8, 7, 18),
    (7, 6, 15),
    (6, 2, 21),
    (2, 1, 17),
    (2, 3, 21),
    (1, 5, 13),
    (3, 0, 5),
    (3, 4, 17),
]


def build_branching(edges):
    G = nx.DiGraph()
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)
    return G


def sorted_edges(G, attr="weight", default=1):
    edges = [(u, v, data.get(attr, default)) for (u, v, data) in G.edges(data=True)]
    edges = sorted(edges, key=lambda x: (x[2], x[1], x[0]))
    return edges


def assert_equal_branchings(G1, G2, attr="weight", default=1):
    edges1 = list(G1.edges(data=True))
    edges2 = list(G2.edges(data=True))
    assert len(edges1) == len(edges2)

    # Grab the weights only.
    e1 = sorted_edges(G1, attr, default)
    e2 = sorted_edges(G2, attr, default)

    # If we have an exception, let's see the edges.
    print(e1)
    print(e2)
    print

    for a, b in zip(e1, e2):
        assert a[:2] == b[:2]
        np.testing.assert_almost_equal(a[2], b[2])


################


def test_optimal_branching1():
    G = build_branching(optimal_arborescence_1)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 131


def test_optimal_branching2a():
    G = build_branching(optimal_branching_2a)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 53


def test_optimal_branching2b():
    G = build_branching(optimal_branching_2b)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 53


def test_optimal_arborescence2():
    G = build_branching(optimal_arborescence_2)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 51


def test_greedy_suboptimal_branching1a():
    G = build_branching(greedy_subopt_branching_1a)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 128


def test_greedy_suboptimal_branching1b():
    G = build_branching(greedy_subopt_branching_1b)
    assert recognition.is_arborescence(G), True
    assert branchings.branching_weight(G) == 127


def test_greedy_max1():
    # Standard test.
    #
    G = G1()
    B = branchings.greedy_branching(G)
    # There are only two possible greedy branchings. The sorting is such
    # that it should equal the second suboptimal branching: 1b.
    B_ = build_branching(greedy_subopt_branching_1b)
    assert_equal_branchings(B, B_)


def test_greedy_max2():
    # Different default weight.
    #
    G = G1()
    del G[1][0][0]["weight"]
    B = branchings.greedy_branching(G, default=6)
    # Chosen so that edge (3,0,5) is not selected and (1,0,6) is instead.

    edges = [
        (1, 0, 6),
        (1, 5, 13),
        (7, 6, 15),
        (2, 1, 17),
        (3, 4, 17),
        (8, 7, 18),
        (2, 3, 21),
        (6, 2, 21),
    ]
    B_ = build_branching(edges)
    assert_equal_branchings(B, B_)


def test_greedy_max3():
    # All equal weights.
    #
    G = G1()
    B = branchings.greedy_branching(G, attr=None)

    # This is mostly arbitrary...the output was generated by running the algo.
    edges = [
        (2, 1, 1),
        (3, 0, 1),
        (3, 4, 1),
        (5, 8, 1),
        (6, 2, 1),
        (7, 3, 1),
        (7, 6, 1),
        (8, 7, 1),
    ]
    B_ = build_branching(edges)
    assert_equal_branchings(B, B_, default=1)


def test_greedy_min():
    G = G1()
    B = branchings.greedy_branching(G, kind="min")

    edges = [
        (1, 0, 4),
        (0, 2, 12),
        (0, 4, 12),
        (2, 5, 12),
        (4, 7, 12),
        (5, 8, 12),
        (5, 6, 14),
        (7, 3, 19),
    ]
    B_ = build_branching(edges)
    assert_equal_branchings(B, B_)


def test_edmonds1_maxbranch():
    G = G1()
    x = branchings.maximum_branching(G)
    x_ = build_branching(optimal_arborescence_1)
    assert_equal_branchings(x, x_)


def test_edmonds1_maxarbor():
    G = G1()
    x = branchings.maximum_spanning_arborescence(G)
    x_ = build_branching(optimal_arborescence_1)
    assert_equal_branchings(x, x_)


def test_edmonds2_maxbranch():
    G = G2()
    x = branchings.maximum_branching(G)
    x_ = build_branching(optimal_branching_2a)
    assert_equal_branchings(x, x_)


def test_edmonds2_maxarbor():
    G = G2()
    x = branchings.maximum_spanning_arborescence(G)
    x_ = build_branching(optimal_arborescence_2)
    assert_equal_branchings(x, x_)


def test_edmonds2_minarbor():
    G = G1()
    x = branchings.minimum_spanning_arborescence(G)
    # This was obtained from algorithm. Need to verify it independently.
    # Branch weight is: 96
    edges = [
        (3, 0, 5),
        (0, 2, 12),
        (0, 4, 12),
        (2, 5, 12),
        (4, 7, 12),
        (5, 8, 12),
        (5, 6, 14),
        (2, 1, 17),
    ]
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)


def test_edmonds3_minbranch1():
    G = G1()
    x = branchings.minimum_branching(G)
    edges = []
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)


def test_edmonds3_minbranch2():
    G = G1()
    G.add_edge(8, 9, weight=-10)
    x = branchings.minimum_branching(G)
    edges = [(8, 9, -10)]
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)


# Need more tests


def test_mst():
    # Make sure we get the same results for undirected graphs.
    # Example from: https://en.wikipedia.org/wiki/Kruskal's_algorithm
    G = nx.Graph()
    edgelist = [
        (0, 3, [("weight", 5)]),
        (0, 1, [("weight", 7)]),
        (1, 3, [("weight", 9)]),
        (1, 2, [("weight", 8)]),
        (1, 4, [("weight", 7)]),
        (3, 4, [("weight", 15)]),
        (3, 5, [("weight", 6)]),
        (2, 4, [("weight", 5)]),
        (4, 5, [("weight", 8)]),
        (4, 6, [("weight", 9)]),
        (5, 6, [("weight", 11)]),
    ]
    G.add_edges_from(edgelist)
    G = G.to_directed()
    x = branchings.minimum_spanning_arborescence(G)

    edges = [
        ({0, 1}, 7),
        ({0, 3}, 5),
        ({3, 5}, 6),
        ({1, 4}, 7),
        ({4, 2}, 5),
        ({4, 6}, 9),
    ]

    assert x.number_of_edges() == len(edges)
    for u, v, d in x.edges(data=True):
        assert ({u, v}, d["weight"]) in edges


def test_mixed_nodetypes():
    # Smoke test to make sure no TypeError is raised for mixed node types.
    G = nx.Graph()
    edgelist = [(0, 3, [("weight", 5)]), (0, "1", [("weight", 5)])]
    G.add_edges_from(edgelist)
    G = G.to_directed()
    x = branchings.minimum_spanning_arborescence(G)


def test_edmonds1_minbranch():
    # Using -G_array and min should give the same as optimal_arborescence_1,
    # but with all edges negative.
    edges = [(u, v, -w) for (u, v, w) in optimal_arborescence_1]

    G = nx.from_numpy_array(-G_array, create_using=nx.DiGraph)

    # Quickly make sure max branching is empty.
    x = branchings.maximum_branching(G)
    x_ = build_branching([])
    assert_equal_branchings(x, x_)

    # Now test the min branching.
    x = branchings.minimum_branching(G)
    x_ = build_branching(edges)
    assert_equal_branchings(x, x_)


def test_edge_attribute_preservation_normal_graph():
    # Test that edge attributes are preserved when finding an optimum graph
    # using the Edmonds class for normal graphs.
    G = nx.Graph()

    edgelist = [
        (0, 1, [("weight", 5), ("otherattr", 1), ("otherattr2", 3)]),
        (0, 2, [("weight", 5), ("otherattr", 2), ("otherattr2", 2)]),
        (1, 2, [("weight", 6), ("otherattr", 3), ("otherattr2", 1)]),
    ]
    G.add_edges_from(edgelist)

    ed = branchings.Edmonds(G)
    B = ed.find_optimum("weight", preserve_attrs=True, seed=1)

    assert B[0][1]["otherattr"] == 1
    assert B[0][1]["otherattr2"] == 3


def test_edge_attribute_preservation_multigraph():

    # Test that edge attributes are preserved when finding an optimum graph
    # using the Edmonds class for multigraphs.
    G = nx.MultiGraph()

    edgelist = [
        (0, 1, [("weight", 5), ("otherattr", 1), ("otherattr2", 3)]),
        (0, 2, [("weight", 5), ("otherattr", 2), ("otherattr2", 2)]),
        (1, 2, [("weight", 6), ("otherattr", 3), ("otherattr2", 1)]),
    ]
    G.add_edges_from(edgelist * 2)  # Make sure we have duplicate edge paths

    ed = branchings.Edmonds(G)
    B = ed.find_optimum("weight", preserve_attrs=True)

    assert B[0][1][0]["otherattr"] == 1
    assert B[0][1][0]["otherattr2"] == 3


def test_edge_attribute_discard():
    # Test that edge attributes are discarded if we do not specify to keep them
    G = nx.Graph()

    edgelist = [
        (0, 1, [("weight", 5), ("otherattr", 1), ("otherattr2", 3)]),
        (0, 2, [("weight", 5), ("otherattr", 2), ("otherattr2", 2)]),
        (1, 2, [("weight", 6), ("otherattr", 3), ("otherattr2", 1)]),
    ]
    G.add_edges_from(edgelist)

    ed = branchings.Edmonds(G)
    B = ed.find_optimum("weight", preserve_attrs=False)

    edge_dict = B[0][1]
    with pytest.raises(KeyError):
        _ = edge_dict["otherattr"]
