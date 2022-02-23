from itertools import product

import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest

from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
    maximum_bipartite_matching, min_weight_full_bipartite_matching
)


def test_maximum_bipartite_matching_raises_on_dense_input():
    with pytest.raises(TypeError):
        graph = np.array([[0, 1], [0, 0]])
        maximum_bipartite_matching(graph)


def test_maximum_bipartite_matching_empty_graph():
    graph = csr_matrix((0, 0))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    expected_matching = np.array([])
    assert_array_equal(expected_matching, x)
    assert_array_equal(expected_matching, y)


def test_maximum_bipartite_matching_empty_left_partition():
    graph = csr_matrix((2, 0))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    assert_array_equal(np.array([]), x)
    assert_array_equal(np.array([-1, -1]), y)


def test_maximum_bipartite_matching_empty_right_partition():
    graph = csr_matrix((0, 3))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    assert_array_equal(np.array([-1, -1, -1]), x)
    assert_array_equal(np.array([]), y)


def test_maximum_bipartite_matching_graph_with_no_edges():
    graph = csr_matrix((2, 2))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    assert_array_equal(np.array([-1, -1]), x)
    assert_array_equal(np.array([-1, -1]), y)


def test_maximum_bipartite_matching_graph_that_causes_augmentation():
    # In this graph, column 1 is initially assigned to row 1, but it should be
    # reassigned to make room for row 2.
    graph = csr_matrix([[1, 1], [1, 0]])
    x = maximum_bipartite_matching(graph, perm_type='column')
    y = maximum_bipartite_matching(graph, perm_type='row')
    expected_matching = np.array([1, 0])
    assert_array_equal(expected_matching, x)
    assert_array_equal(expected_matching, y)


def test_maximum_bipartite_matching_graph_with_more_rows_than_columns():
    graph = csr_matrix([[1, 1], [1, 0], [0, 1]])
    x = maximum_bipartite_matching(graph, perm_type='column')
    y = maximum_bipartite_matching(graph, perm_type='row')
    assert_array_equal(np.array([0, -1, 1]), x)
    assert_array_equal(np.array([0, 2]), y)


def test_maximum_bipartite_matching_graph_with_more_columns_than_rows():
    graph = csr_matrix([[1, 1, 0], [0, 0, 1]])
    x = maximum_bipartite_matching(graph, perm_type='column')
    y = maximum_bipartite_matching(graph, perm_type='row')
    assert_array_equal(np.array([0, 2]), x)
    assert_array_equal(np.array([0, -1, 1]), y)


def test_maximum_bipartite_matching_explicit_zeros_count_as_edges():
    data = [0, 0]
    indices = [1, 0]
    indptr = [0, 1, 2]
    graph = csr_matrix((data, indices, indptr), shape=(2, 2))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    expected_matching = np.array([1, 0])
    assert_array_equal(expected_matching, x)
    assert_array_equal(expected_matching, y)


def test_maximum_bipartite_matching_feasibility_of_result():
    # This is a regression test for GitHub issue #11458
    data = np.ones(50, dtype=int)
    indices = [11, 12, 19, 22, 23, 5, 22, 3, 8, 10, 5, 6, 11, 12, 13, 5, 13,
               14, 20, 22, 3, 15, 3, 13, 14, 11, 12, 19, 22, 23, 5, 22, 3, 8,
               10, 5, 6, 11, 12, 13, 5, 13, 14, 20, 22, 3, 15, 3, 13, 14]
    indptr = [0, 5, 7, 10, 10, 15, 20, 22, 22, 23, 25, 30, 32, 35, 35, 40, 45,
              47, 47, 48, 50]
    graph = csr_matrix((data, indices, indptr), shape=(20, 25))
    x = maximum_bipartite_matching(graph, perm_type='row')
    y = maximum_bipartite_matching(graph, perm_type='column')
    assert (x != -1).sum() == 13
    assert (y != -1).sum() == 13
    # Ensure that each element of the matching is in fact an edge in the graph.
    for u, v in zip(range(graph.shape[0]), y):
        if v != -1:
            assert graph[u, v]
    for u, v in zip(x, range(graph.shape[1])):
        if u != -1:
            assert graph[u, v]


def test_matching_large_random_graph_with_one_edge_incident_to_each_vertex():
    np.random.seed(42)
    A = diags(np.ones(25), offsets=0, format='csr')
    rand_perm = np.random.permutation(25)
    rand_perm2 = np.random.permutation(25)

    Rrow = np.arange(25)
    Rcol = rand_perm
    Rdata = np.ones(25, dtype=int)
    Rmat = coo_matrix((Rdata, (Rrow, Rcol))).tocsr()

    Crow = rand_perm2
    Ccol = np.arange(25)
    Cdata = np.ones(25, dtype=int)
    Cmat = coo_matrix((Cdata, (Crow, Ccol))).tocsr()
    # Randomly permute identity matrix
    B = Rmat * A * Cmat

    # Row permute
    perm = maximum_bipartite_matching(B, perm_type='row')
    Rrow = np.arange(25)
    Rcol = perm
    Rdata = np.ones(25, dtype=int)
    Rmat = coo_matrix((Rdata, (Rrow, Rcol))).tocsr()
    C1 = Rmat * B

    # Column permute
    perm2 = maximum_bipartite_matching(B, perm_type='column')
    Crow = perm2
    Ccol = np.arange(25)
    Cdata = np.ones(25, dtype=int)
    Cmat = coo_matrix((Cdata, (Crow, Ccol))).tocsr()
    C2 = B * Cmat

    # Should get identity matrix back
    assert_equal(any(C1.diagonal() == 0), False)
    assert_equal(any(C2.diagonal() == 0), False)


@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_min_weight_full_matching_trivial_graph(num_rows, num_cols):
    biadjacency_matrix = csr_matrix((num_cols, num_rows))
    row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency_matrix)
    assert len(row_ind) == 0
    assert len(col_ind) == 0


@pytest.mark.parametrize('biadjacency_matrix',
                         [
                            [[1, 1, 1], [1, 0, 0], [1, 0, 0]],
                            [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
                            [[1, 0, 0], [2, 0, 0]],
                            [[0, 1, 0], [0, 2, 0]],
                            [[1, 0], [2, 0], [5, 0]]
                         ])
def test_min_weight_full_matching_infeasible_problems(biadjacency_matrix):
    with pytest.raises(ValueError):
        min_weight_full_bipartite_matching(csr_matrix(biadjacency_matrix))


def test_explicit_zero_causes_warning():
    with pytest.warns(UserWarning):
        biadjacency_matrix = csr_matrix(((2, 0, 3), (0, 1, 1), (0, 2, 3)))
        min_weight_full_bipartite_matching(biadjacency_matrix)


# General test for linear sum assignment solvers to make it possible to rely
# on the same tests for scipy.optimize.linear_sum_assignment.
def linear_sum_assignment_assertions(
    solver, array_type, sign, test_case
):
    cost_matrix, expected_cost = test_case
    maximize = sign == -1
    cost_matrix = sign * array_type(cost_matrix)
    expected_cost = sign * np.array(expected_cost)

    row_ind, col_ind = solver(cost_matrix, maximize=maximize)
    assert_array_equal(row_ind, np.sort(row_ind))
    assert_array_equal(expected_cost,
                       np.array(cost_matrix[row_ind, col_ind]).flatten())

    cost_matrix = cost_matrix.T
    row_ind, col_ind = solver(cost_matrix, maximize=maximize)
    assert_array_equal(row_ind, np.sort(row_ind))
    assert_array_equal(np.sort(expected_cost),
                       np.sort(np.array(
                           cost_matrix[row_ind, col_ind])).flatten())


linear_sum_assignment_test_cases = product(
    [-1, 1],
    [
        # Square
        ([[400, 150, 400],
          [400, 450, 600],
          [300, 225, 300]],
         [150, 400, 300]),

        # Rectangular variant
        ([[400, 150, 400, 1],
          [400, 450, 600, 2],
          [300, 225, 300, 3]],
         [150, 2, 300]),

        ([[10, 10, 8],
          [9, 8, 1],
          [9, 7, 4]],
         [10, 1, 7]),

        # Square
        ([[10, 10, 8, 11],
          [9, 8, 1, 1],
          [9, 7, 4, 10]],
         [10, 1, 4]),

        # Rectangular variant
        ([[10, float("inf"), float("inf")],
          [float("inf"), float("inf"), 1],
          [float("inf"), 7, float("inf")]],
         [10, 1, 7])
    ])


@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_min_weight_full_matching_small_inputs(sign, test_case):
    linear_sum_assignment_assertions(
        min_weight_full_bipartite_matching, csr_matrix, sign, test_case)
