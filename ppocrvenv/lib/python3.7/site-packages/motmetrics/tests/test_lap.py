# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tests linear assignment problem solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import pytest

from motmetrics import lap

DESIRED_SOLVERS = ['lap', 'lapsolver', 'munkres', 'ortools', 'scipy']
SOLVERS = lap.available_solvers


@pytest.mark.parametrize('solver', DESIRED_SOLVERS)
def test_solver_is_available(solver):
    if solver not in lap.available_solvers:
        warnings.warn('solver not available: ' + solver)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_easy(solver):
    """Problem that could be solved by a greedy algorithm."""
    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full(solver):
    """Problem that would be incorrect using a greedy algorithm."""
    costs = np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 6 + 2 + 2.
    expected = np.asfarray([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full_negative(solver):
    costs = -7 + np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 5 + 1 + 1.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_empty(solver):
    costs = np.asfarray([[]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    np.testing.assert_equal(np.size(result), 0)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_infeasible(solver):
    """Tests that minimum-cost solution with most edges is found."""
    costs = np.asfarray([[np.nan, np.nan, 2],
                         [np.nan, np.nan, 1],
                         [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (1, 2), (2, 1).
    expected = np.array([[1, 2], [2, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_disallowed(solver):
    costs = np.asfarray([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_non_integer(solver):
    costs = (1. / 9) * np.asfarray([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_attractive_disallowed(solver):
    """Graph contains an attractive edge that cannot be used."""
    costs = np.asfarray([[-10000, -1], [-1, np.nan]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # The optimal solution is (0, 1), (1, 0) for a cost of -2.
    # Ensure that the algorithm does not choose the (0, 0) edge.
    # This would not be a perfect matching.
    expected = np.array([[0, 1], [1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_attractive_broken_ring(solver):
    """Graph contains cheap broken ring and expensive unbroken ring."""
    costs = np.asfarray([[np.nan, 1000, np.nan], [np.nan, 1, 1000], [1000, np.nan, 1]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal solution is (0, 1), (1, 2), (2, 0) with cost 1000 + 1000 + 1000.
    # Solver might choose (0, 0), (1, 1), (2, 2) with cost inf + 1 + 1.
    expected = np.array([[0, 1, 2], [1, 2, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_wide(solver):
    costs = np.asfarray([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [1, 2]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_tall(solver):
    costs = np.asfarray([[6, 10], [4, 8], [1, 2]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[1, 2], [0, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_disallowed_wide(solver):
    costs = np.asfarray([[np.nan, 11, 8], [8, np.nan, 7]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [2, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_disallowed_tall(solver):
    costs = np.asfarray([[np.nan, 9], [11, np.nan], [8, 7]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 2], [1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_infeasible(solver):
    """Tests that minimum-cost solution with most edges is found."""
    costs = np.asfarray([[np.nan, np.nan, 2],
                         [np.nan, np.nan, 1],
                         [np.nan, np.nan, 3],
                         [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (1, 2), (3, 1).
    expected = np.array([[1, 3], [2, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


def test_change_solver():
    """Tests effect of lap.set_default_solver."""

    def mysolver(_):
        mysolver.called += 1
        return np.array([]), np.array([])
    mysolver.called = 0

    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])

    with lap.set_default_solver(mysolver):
        lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
