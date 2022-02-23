from itertools import combinations
import pytest
import networkx as nx


def path_graph():
    """Return a path graph of length three."""
    G = nx.path_graph(3, create_using=nx.DiGraph)
    G.graph["name"] = "path"
    nx.freeze(G)
    return G


def fork_graph():
    """Return a three node fork graph."""
    G = nx.DiGraph(name="fork")
    G.add_edges_from([(0, 1), (0, 2)])
    nx.freeze(G)
    return G


def collider_graph():
    """Return a collider/v-structure graph with three nodes."""
    G = nx.DiGraph(name="collider")
    G.add_edges_from([(0, 2), (1, 2)])
    nx.freeze(G)
    return G


def naive_bayes_graph():
    """Return a simply Naive Bayes PGM graph."""
    G = nx.DiGraph(name="naive_bayes")
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    nx.freeze(G)
    return G


def asia_graph():
    """Return the 'Asia' PGM graph."""
    G = nx.DiGraph(name="asia")
    G.add_edges_from(
        [
            ("asia", "tuberculosis"),
            ("smoking", "cancer"),
            ("smoking", "bronchitis"),
            ("tuberculosis", "either"),
            ("cancer", "either"),
            ("either", "xray"),
            ("either", "dyspnea"),
            ("bronchitis", "dyspnea"),
        ]
    )
    nx.freeze(G)
    return G


@pytest.fixture(name="path_graph")
def path_graph_fixture():
    return path_graph()


@pytest.fixture(name="fork_graph")
def fork_graph_fixture():
    return fork_graph()


@pytest.fixture(name="collider_graph")
def collider_graph_fixture():
    return collider_graph()


@pytest.fixture(name="naive_bayes_graph")
def naive_bayes_graph_fixture():
    return naive_bayes_graph()


@pytest.fixture(name="asia_graph")
def asia_graph_fixture():
    return asia_graph()


@pytest.mark.parametrize(
    "graph",
    [path_graph(), fork_graph(), collider_graph(), naive_bayes_graph(), asia_graph()],
)
def test_markov_condition(graph):
    """Test that the Markov condition holds for each PGM graph."""
    for node in graph.nodes:
        parents = set(graph.predecessors(node))
        non_descendants = graph.nodes - nx.descendants(graph, node) - {node} - parents
        assert nx.d_separated(graph, {node}, non_descendants, parents)


def test_path_graph_dsep(path_graph):
    """Example-based test of d-separation for path_graph."""
    assert nx.d_separated(path_graph, {0}, {2}, {1})
    assert not nx.d_separated(path_graph, {0}, {2}, {})


def test_fork_graph_dsep(fork_graph):
    """Example-based test of d-separation for fork_graph."""
    assert nx.d_separated(fork_graph, {1}, {2}, {0})
    assert not nx.d_separated(fork_graph, {1}, {2}, {})


def test_collider_graph_dsep(collider_graph):
    """Example-based test of d-separation for collider_graph."""
    assert nx.d_separated(collider_graph, {0}, {1}, {})
    assert not nx.d_separated(collider_graph, {0}, {1}, {2})


def test_naive_bayes_dsep(naive_bayes_graph):
    """Example-based test of d-separation for naive_bayes_graph."""
    for u, v in combinations(range(1, 5), 2):
        assert nx.d_separated(naive_bayes_graph, {u}, {v}, {0})
        assert not nx.d_separated(naive_bayes_graph, {u}, {v}, {})


def test_asia_graph_dsep(asia_graph):
    """Example-based test of d-separation for asia_graph."""
    assert nx.d_separated(
        asia_graph, {"asia", "smoking"}, {"dyspnea", "xray"}, {"bronchitis", "either"}
    )
    assert nx.d_separated(
        asia_graph, {"tuberculosis", "cancer"}, {"bronchitis"}, {"smoking", "xray"}
    )


def test_undirected_graphs_are_not_supported():
    """
    Test that undirected graphs are not supported.

    d-separation does not apply in the case of undirected graphs.
    """
    with pytest.raises(nx.NetworkXNotImplemented):
        g = nx.path_graph(3, nx.Graph)
        nx.d_separated(g, {0}, {1}, {2})


def test_cyclic_graphs_raise_error():
    """
    Test that cycle graphs should cause erroring.

    This is because PGMs assume a directed acyclic graph.
    """
    with pytest.raises(nx.NetworkXError):
        g = nx.cycle_graph(3, nx.DiGraph)
        nx.d_separated(g, {0}, {1}, {2})


def test_invalid_nodes_raise_error(asia_graph):
    """
    Test that graphs that have invalid nodes passed in raise errors.
    """
    with pytest.raises(nx.NodeNotFound):
        nx.d_separated(asia_graph, {0}, {1}, {2})
