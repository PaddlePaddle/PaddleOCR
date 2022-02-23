"""Unit tests for the :mod:`networkx.generators.duplication` module.

"""
import pytest

from networkx.exception import NetworkXError
from networkx.generators.duplication import duplication_divergence_graph
from networkx.generators.duplication import partial_duplication_graph


class TestDuplicationDivergenceGraph:
    """Unit tests for the
    :func:`networkx.generators.duplication.duplication_divergence_graph`
    function.

    """

    def test_final_size(self):
        G = duplication_divergence_graph(3, 1)
        assert len(G) == 3
        G = duplication_divergence_graph(3, 1, seed=42)
        assert len(G) == 3

    def test_probability_too_large(self):
        with pytest.raises(NetworkXError):
            duplication_divergence_graph(3, 2)

    def test_probability_too_small(self):
        with pytest.raises(NetworkXError):
            duplication_divergence_graph(3, -1)


class TestPartialDuplicationGraph:
    """Unit tests for the
    :func:`networkx.generators.duplication.partial_duplication_graph`
    function.

    """

    def test_final_size(self):
        N = 10
        n = 5
        p = 0.5
        q = 0.5
        G = partial_duplication_graph(N, n, p, q)
        assert len(G) == N
        G = partial_duplication_graph(N, n, p, q, seed=42)
        assert len(G) == N

    def test_initial_clique_size(self):
        N = 10
        n = 10
        p = 0.5
        q = 0.5
        G = partial_duplication_graph(N, n, p, q)
        assert len(G) == n

    def test_invalid_initial_size(self):
        with pytest.raises(NetworkXError):
            N = 5
            n = 10
            p = 0.5
            q = 0.5
            G = partial_duplication_graph(N, n, p, q)

    def test_invalid_probabilities(self):
        N = 1
        n = 1
        for p, q in [(0.5, 2), (0.5, -1), (2, 0.5), (-1, 0.5)]:
            args = (N, n, p, q)
            pytest.raises(NetworkXError, partial_duplication_graph, *args)
