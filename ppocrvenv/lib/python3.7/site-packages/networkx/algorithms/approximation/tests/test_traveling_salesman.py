"""Unit tests for the traveling_salesman module."""
import pytest
import random
import networkx as nx
import networkx.algorithms.approximation as nx_app

pairwise = nx.utils.pairwise


def test_christofides_hamiltonian():
    random.seed(42)
    G = nx.complete_graph(20)
    for (u, v) in G.edges():
        G[u][v]["weight"] = random.randint(0, 10)

    H = nx.Graph()
    H.add_edges_from(pairwise(nx_app.christofides(G)))
    H.remove_edges_from(nx.find_cycle(H))
    assert len(H.edges) == 0

    tree = nx.minimum_spanning_tree(G, weight="weight")
    H = nx.Graph()
    H.add_edges_from(pairwise(nx_app.christofides(G, tree)))
    H.remove_edges_from(nx.find_cycle(H))
    assert len(H.edges) == 0


def test_christofides_incomplete_graph():
    G = nx.complete_graph(10)
    G.remove_edge(0, 1)
    pytest.raises(nx.NetworkXError, nx_app.christofides, G)


def test_christofides_ignore_selfloops():
    G = nx.complete_graph(5)
    G.add_edge(3, 3)
    cycle = nx_app.christofides(G)
    assert len(cycle) - 1 == len(G) == len(set(cycle))


# set up graphs for other tests
@classmethod
def _setup_class(cls):
    cls.DG = nx.DiGraph()
    cls.DG.add_weighted_edges_from(
        {
            ("A", "B", 3),
            ("A", "C", 17),
            ("A", "D", 14),
            ("B", "A", 3),
            ("B", "C", 12),
            ("B", "D", 16),
            ("C", "A", 13),
            ("C", "B", 12),
            ("C", "D", 4),
            ("D", "A", 14),
            ("D", "B", 15),
            ("D", "C", 2),
        }
    )
    cls.DG_cycle = ["D", "C", "B", "A", "D"]
    cls.DG_cost = 31.0

    cls.DG2 = nx.DiGraph()
    cls.DG2.add_weighted_edges_from(
        {
            ("A", "B", 3),
            ("A", "C", 17),
            ("A", "D", 14),
            ("B", "A", 30),
            ("B", "C", 2),
            ("B", "D", 16),
            ("C", "A", 33),
            ("C", "B", 32),
            ("C", "D", 34),
            ("D", "A", 14),
            ("D", "B", 15),
            ("D", "C", 2),
        }
    )
    cls.DG2_cycle = ["D", "A", "B", "C", "D"]
    cls.DG2_cost = 53.0

    cls.unweightedUG = nx.complete_graph(5, nx.Graph())
    cls.unweightedDG = nx.complete_graph(5, nx.DiGraph())

    cls.incompleteUG = nx.Graph()
    cls.incompleteUG.add_weighted_edges_from({(0, 1, 1), (1, 2, 3)})
    cls.incompleteDG = nx.DiGraph()
    cls.incompleteDG.add_weighted_edges_from({(0, 1, 1), (1, 2, 3)})

    cls.UG = nx.Graph()
    cls.UG.add_weighted_edges_from(
        {
            ("A", "B", 3),
            ("A", "C", 17),
            ("A", "D", 14),
            ("B", "C", 12),
            ("B", "D", 16),
            ("C", "D", 4),
        }
    )
    cls.UG_cycle = ["D", "C", "B", "A", "D"]
    cls.UG_cost = 33.0

    cls.UG2 = nx.Graph()
    cls.UG2.add_weighted_edges_from(
        {
            ("A", "B", 1),
            ("A", "C", 15),
            ("A", "D", 5),
            ("B", "C", 16),
            ("B", "D", 8),
            ("C", "D", 3),
        }
    )
    cls.UG2_cycle = ["D", "C", "B", "A", "D"]
    cls.UG2_cost = 25.0


def validate_solution(soln, cost, exp_soln, exp_cost):
    assert soln == exp_soln
    assert cost == exp_cost


def validate_symmetric_solution(soln, cost, exp_soln, exp_cost):
    assert soln == exp_soln or soln == exp_soln[::-1]
    assert cost == exp_cost


class TestGreedyTSP:
    setup_class = _setup_class

    def test_greedy(self):
        cycle = nx_app.greedy_tsp(self.DG, source="D")
        cost = sum(self.DG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, ["D", "C", "B", "A", "D"], 31.0)

        cycle = nx_app.greedy_tsp(self.DG2, source="D")
        cost = sum(self.DG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, ["D", "C", "B", "A", "D"], 78.0)

        cycle = nx_app.greedy_tsp(self.UG, source="D")
        cost = sum(self.UG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, ["D", "C", "B", "A", "D"], 33.0)

        cycle = nx_app.greedy_tsp(self.UG2, source="D")
        cost = sum(self.UG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, ["D", "C", "A", "B", "D"], 27.0)

    def test_not_complete_graph(self):
        pytest.raises(nx.NetworkXError, nx_app.greedy_tsp, self.incompleteUG)
        pytest.raises(nx.NetworkXError, nx_app.greedy_tsp, self.incompleteDG)

    def test_not_weighted_graph(self):
        nx_app.greedy_tsp(self.unweightedUG)
        nx_app.greedy_tsp(self.unweightedDG)

    def test_two_nodes(self):
        G = nx.Graph()
        G.add_weighted_edges_from({(1, 2, 1)})
        cycle = nx_app.greedy_tsp(G)
        cost = sum(G[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, [1, 2, 1], 2)

    def test_ignore_selfloops(self):
        G = nx.complete_graph(5)
        G.add_edge(3, 3)
        cycle = nx_app.greedy_tsp(G)
        assert len(cycle) - 1 == len(G) == len(set(cycle))


class TestSimulatedAnnealingTSP:
    setup_class = _setup_class
    tsp = staticmethod(nx_app.simulated_annealing_tsp)

    def test_simulated_annealing_directed(self):
        cycle = self.tsp(self.DG, "greedy", source="D", seed=42)
        cost = sum(self.DG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, self.DG_cycle, self.DG_cost)

        initial_sol = ["D", "B", "A", "C", "D"]
        cycle = self.tsp(self.DG, initial_sol, source="D", seed=42)
        cost = sum(self.DG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, self.DG_cycle, self.DG_cost)

        initial_sol = ["D", "A", "C", "B", "D"]
        cycle = self.tsp(self.DG, initial_sol, move="1-0", source="D", seed=42)
        cost = sum(self.DG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, self.DG_cycle, self.DG_cost)

        cycle = self.tsp(self.DG2, "greedy", source="D", seed=42)
        cost = sum(self.DG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, self.DG2_cycle, self.DG2_cost)

        cycle = self.tsp(self.DG2, "greedy", move="1-0", source="D", seed=42)
        cost = sum(self.DG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, self.DG2_cycle, self.DG2_cost)

    def test_simulated_annealing_undirected(self):
        cycle = self.tsp(self.UG, "greedy", source="D", seed=42)
        cost = sum(self.UG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, self.UG_cycle, self.UG_cost)

        cycle = self.tsp(self.UG2, "greedy", source="D", seed=42)
        cost = sum(self.UG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_symmetric_solution(cycle, cost, self.UG2_cycle, self.UG2_cost)

        cycle = self.tsp(self.UG2, "greedy", move="1-0", source="D", seed=42)
        cost = sum(self.UG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_symmetric_solution(cycle, cost, self.UG2_cycle, self.UG2_cost)

    def test_error_on_input_order_mistake(self):
        # see issue #4846 https://github.com/networkx/networkx/issues/4846
        pytest.raises(TypeError, self.tsp, self.UG, weight="weight")
        pytest.raises(nx.NetworkXError, self.tsp, self.UG, "weight")

    def test_not_complete_graph(self):
        pytest.raises(
            nx.NetworkXError,
            self.tsp,
            self.incompleteUG,
            "greedy",
            source=0,
        )
        pytest.raises(
            nx.NetworkXError,
            self.tsp,
            self.incompleteDG,
            "greedy",
            source=0,
        )

    def test_ignore_selfloops(self):
        G = nx.complete_graph(5)
        G.add_edge(3, 3)
        cycle = self.tsp(G, "greedy")
        assert len(cycle) - 1 == len(G) == len(set(cycle))

    def test_not_weighted_graph(self):
        self.tsp(self.unweightedUG, "greedy")
        self.tsp(self.unweightedDG, "greedy")

    def test_two_nodes(self):
        G = nx.Graph()
        G.add_weighted_edges_from({(1, 2, 1)})

        cycle = self.tsp(G, "greedy", source=1, seed=42)
        cost = sum(G[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, [1, 2, 1], 2)

        cycle = self.tsp(G, [1, 2, 1], source=1, seed=42)
        cost = sum(G[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        validate_solution(cycle, cost, [1, 2, 1], 2)

    def test_failure_of_costs_too_high_when_iterations_low(self):
        # Simulated Annealing Version:
        # set number of moves low and alpha high
        cycle = self.tsp(
            self.DG2, "greedy", source="D", move="1-0", alpha=1, N_inner=1, seed=42
        )
        cost = sum(self.DG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        print(cycle, cost)
        assert cost > self.DG2_cost

        # Try with an incorrect initial guess
        initial_sol = ["D", "A", "B", "C", "D"]
        cycle = self.tsp(
            self.DG,
            initial_sol,
            source="D",
            move="1-0",
            alpha=0.1,
            N_inner=1,
            max_iterations=1,
            seed=42,
        )
        cost = sum(self.DG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        print(cycle, cost)
        assert cost > self.DG_cost


class TestThresholdAcceptingTSP(TestSimulatedAnnealingTSP):
    tsp = staticmethod(nx_app.threshold_accepting_tsp)

    def test_failure_of_costs_too_high_when_iterations_low(self):
        # Threshold Version:
        # set number of moves low and number of iterations low
        cycle = self.tsp(
            self.DG2,
            "greedy",
            source="D",
            move="1-0",
            N_inner=1,
            max_iterations=1,
            seed=4,
        )
        cost = sum(self.DG2[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        assert cost > self.DG2_cost

        # set threshold too low
        initial_sol = ["D", "A", "B", "C", "D"]
        cycle = self.tsp(
            self.DG,
            initial_sol,
            source="D",
            move="1-0",
            threshold=-3,
            seed=42,
        )
        cost = sum(self.DG[n][nbr]["weight"] for n, nbr in pairwise(cycle))
        assert cost > self.DG_cost


# Tests for function traveling_salesman_problem
def test_TSP_method():
    G = nx.cycle_graph(9)
    G[4][5]["weight"] = 10

    def my_tsp_method(G, weight):
        return nx_app.simulated_annealing_tsp(G, "greedy", weight, source=4, seed=1)

    path = nx_app.traveling_salesman_problem(G, method=my_tsp_method, cycle=False)
    print(path)
    assert path == [4, 3, 2, 1, 0, 8, 7, 6, 5]


def test_TSP_unweighted():
    G = nx.cycle_graph(9)
    path = nx_app.traveling_salesman_problem(G, nodes=[3, 6], cycle=False)
    assert path in ([3, 4, 5, 6], [6, 5, 4, 3])

    cycle = nx_app.traveling_salesman_problem(G, nodes=[3, 6])
    assert cycle in ([3, 4, 5, 6, 5, 4, 3], [6, 5, 4, 3, 4, 5, 6])


def test_TSP_weighted():
    G = nx.cycle_graph(9)
    G[0][1]["weight"] = 2
    G[1][2]["weight"] = 2
    G[2][3]["weight"] = 2
    G[3][4]["weight"] = 4
    G[4][5]["weight"] = 5
    G[5][6]["weight"] = 4
    G[6][7]["weight"] = 2
    G[7][8]["weight"] = 2
    G[8][0]["weight"] = 2
    tsp = nx_app.traveling_salesman_problem

    # path between 3 and 6
    expected_paths = ([3, 2, 1, 0, 8, 7, 6], [6, 7, 8, 0, 1, 2, 3])
    # cycle between 3 and 6
    expected_cycles = (
        [3, 2, 1, 0, 8, 7, 6, 7, 8, 0, 1, 2, 3],
        [6, 7, 8, 0, 1, 2, 3, 2, 1, 0, 8, 7, 6],
    )
    # path through all nodes
    expected_tourpaths = (
        [5, 6, 7, 8, 0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0, 8, 7, 6, 5],
    )

    # Check default method
    cycle = tsp(G, nodes=[3, 6], weight="weight")
    assert cycle in expected_cycles

    path = tsp(G, nodes=[3, 6], weight="weight", cycle=False)
    assert path in expected_paths

    tourpath = tsp(G, weight="weight", cycle=False)
    assert tourpath in expected_tourpaths

    # Check all methods
    methods = [
        nx_app.christofides,
        nx_app.greedy_tsp,
        lambda G, wt: nx_app.simulated_annealing_tsp(G, "greedy", weight=wt),
        lambda G, wt: nx_app.threshold_accepting_tsp(G, "greedy", weight=wt),
    ]
    for method in methods:
        cycle = tsp(G, nodes=[3, 6], weight="weight", method=method)
        assert cycle in expected_cycles

        path = tsp(G, nodes=[3, 6], weight="weight", method=method, cycle=False)
        assert path in expected_paths

        tourpath = tsp(G, weight="weight", method=method, cycle=False)
        assert tourpath in expected_tourpaths


def test_TSP_incomplete_graph_short_path():
    G = nx.cycle_graph(9)
    G.add_edges_from([(4, 9), (9, 10), (10, 11), (11, 0)])
    G[4][5]["weight"] = 5

    cycle = nx_app.traveling_salesman_problem(G)
    print(cycle)
    assert len(cycle) == 17 and len(set(cycle)) == 12

    # make sure that cutting one edge out of complete graph formulation
    # cuts out many edges out of the path of the TSP
    path = nx_app.traveling_salesman_problem(G, cycle=False)
    print(path)
    assert len(path) == 13 and len(set(path)) == 12
