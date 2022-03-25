# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Tools for solving linear assignment problems."""

# pylint: disable=import-outside-toplevel

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import warnings

import numpy as np


def _module_is_available_py2(name):
    try:
        imp.find_module(name)
        return True
    except ImportError:
        return False


def _module_is_available_py3(name):
    return importlib.util.find_spec(name) is not None


try:
    import importlib.util
except ImportError:
    import imp
    _module_is_available = _module_is_available_py2
else:
    _module_is_available = _module_is_available_py3


def linear_sum_assignment(costs, solver=None):
    """Solve a linear sum assignment problem (LSA).

    For large datasets solving the minimum cost assignment becomes the dominant runtime part.
    We therefore support various solvers out of the box (currently lapsolver, scipy, ortools, munkres)

    Params
    ------
    costs : np.array
        numpy matrix containing costs. Use NaN/Inf values for unassignable
        row/column pairs.

    Kwargs
    ------
    solver : callable or str, optional
        When str: name of solver to use.
        When callable: function to invoke
        When None: uses first available solver
    """
    costs = np.asarray(costs)
    if not costs.size:
        return np.array([], dtype=int), np.array([], dtype=int)

    solver = solver or default_solver

    if isinstance(solver, str):
        # Try resolve from string
        solver = solver_map.get(solver, None)

    assert callable(solver), 'Invalid LAP solver.'
    rids, cids = solver(costs)
    rids = np.asarray(rids).astype(int)
    cids = np.asarray(cids).astype(int)
    return rids, cids


def add_expensive_edges(costs):
    """Replaces non-edge costs (nan, inf) with large number.

    If the optimal solution includes one of these edges,
    then the original problem was infeasible.

    Parameters
    ----------
    costs : np.ndarray
    """
    # The graph is probably already dense if we are doing this.
    assert isinstance(costs, np.ndarray)
    # The linear_sum_assignment function in scipy does not support missing edges.
    # Replace nan with a large constant that ensures it is not chosen.
    # If it is chosen, that means the problem was infeasible.
    valid = np.isfinite(costs)
    if valid.all():
        return costs.copy()
    if not valid.any():
        return np.zeros_like(costs)
    r = min(costs.shape)
    # Assume all edges costs are within [-c, c], c >= 0.
    # The cost of an invalid edge must be such that...
    # choosing this edge once and the best-possible edge (r - 1) times
    # is worse than choosing the worst-possible edge r times.
    # l + (r - 1) (-c) > r c
    # l > r c + (r - 1) c
    # l > (2 r - 1) c
    # Choose l = 2 r c + 1 > (2 r - 1) c.
    c = np.abs(costs[valid]).max() + 1  # Doesn't hurt to add 1 here.
    large_constant = 2 * r * c + 1
    return np.where(valid, costs, large_constant)


def _exclude_missing_edges(costs, rids, cids):
    subset = [
        index for index, (i, j) in enumerate(zip(rids, cids))
        if np.isfinite(costs[i, j])
    ]
    return rids[subset], cids[subset]


def lsa_solve_scipy(costs):
    """Solves the LSA problem using the scipy library."""

    from scipy.optimize import linear_sum_assignment as scipy_solve

    # scipy (1.3.3) does not support nan or inf values
    finite_costs = add_expensive_edges(costs)
    rids, cids = scipy_solve(finite_costs)
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def lsa_solve_lapsolver(costs):
    """Solves the LSA problem using the lapsolver library."""
    from lapsolver import solve_dense

    # Note that lapsolver will add expensive finite edges internally.
    # However, older versions did not add a large enough edge.
    finite_costs = add_expensive_edges(costs)
    rids, cids = solve_dense(finite_costs)
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def lsa_solve_munkres(costs):
    """Solves the LSA problem using the Munkres library."""
    from munkres import Munkres

    m = Munkres()
    # The munkres package may hang if the problem is not feasible.
    # Therefore, add expensive edges instead of using munkres.DISALLOWED.
    finite_costs = add_expensive_edges(costs)
    # Ensure that matrix is square.
    finite_costs = _zero_pad_to_square(finite_costs)
    indices = np.array(m.compute(finite_costs), dtype=int)
    # Exclude extra matches from extension to square matrix.
    indices = indices[(indices[:, 0] < costs.shape[0])
                      & (indices[:, 1] < costs.shape[1])]
    rids, cids = indices[:, 0], indices[:, 1]
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def _zero_pad_to_square(costs):
    num_rows, num_cols = costs.shape
    if num_rows == num_cols:
        return costs
    n = max(num_rows, num_cols)
    padded = np.zeros((n, n), dtype=costs.dtype)
    padded[:num_rows, :num_cols] = costs
    return padded


def lsa_solve_ortools(costs):
    """Solves the LSA problem using Google's optimization tools. """
    from ortools.graph import pywrapgraph

    if costs.shape[0] != costs.shape[1]:
        # ortools assumes that the problem is square.
        # Non-square problem will be infeasible.
        # Default to scipy solver rather than add extra zeros.
        # (This maintains the same behaviour as previous versions.)
        return linear_sum_assignment(costs, solver='scipy')

    rs, cs = np.isfinite(costs).nonzero()  # pylint: disable=unbalanced-tuple-unpacking
    finite_costs = costs[rs, cs]
    scale = find_scale_for_integer_approximation(finite_costs)
    if scale != 1:
        warnings.warn('costs are not integers; using approximation')
    int_costs = np.round(scale * finite_costs).astype(int)

    assignment = pywrapgraph.LinearSumAssignment()
    # OR-Tools does not like to receive indices of type np.int64.
    rs = rs.tolist()  # pylint: disable=no-member
    cs = cs.tolist()
    int_costs = int_costs.tolist()
    for r, c, int_cost in zip(rs, cs, int_costs):
        assignment.AddArcWithCost(r, c, int_cost)

    status = assignment.Solve()
    try:
        _ortools_assert_is_optimal(pywrapgraph, status)
    except AssertionError:
        # Default to scipy solver rather than add finite edges.
        # (This maintains the same behaviour as previous versions.)
        return linear_sum_assignment(costs, solver='scipy')

    return _ortools_extract_solution(assignment)


def find_scale_for_integer_approximation(costs, base=10, log_max_scale=8, log_safety=2):
    """Returns a multiplicative factor to use before rounding to integers.

    Tries to find scale = base ** j (for j integer) such that:
        abs(diff(unique(costs))) <= 1 / (scale * safety)
    where safety = base ** log_safety.

    Logs a warning if the desired resolution could not be achieved.
    """
    costs = np.asarray(costs)
    costs = costs[np.isfinite(costs)]  # Exclude non-edges (nan, inf) and -inf.
    if np.size(costs) == 0:
        # No edges with numeric value. Scale does not matter.
        return 1
    unique = np.unique(costs)
    if np.size(unique) == 1:
        # All costs have equal values. Scale does not matter.
        return 1
    try:
        _assert_integer(costs)
    except AssertionError:
        pass
    else:
        # The costs are already integers.
        return 1

    # Find scale = base ** e such that:
    # 1 / scale <= tol, or
    # e = log(scale) >= -log(tol)
    # where tol = min(diff(unique(costs)))
    min_diff = np.diff(unique).min()
    e = np.ceil(np.log(min_diff) / np.log(base)).astype(int).item()
    # Add optional non-negative safety factor to reduce quantization noise.
    e += max(log_safety, 0)
    # Ensure that we do not reduce the magnitude of the costs.
    e = max(e, 0)
    # Ensure that the scale is not too large.
    if e > log_max_scale:
        warnings.warn('could not achieve desired resolution for approximation: '
                      'want exponent %d but max is %d', e, log_max_scale)
        e = log_max_scale
    scale = base ** e
    return scale


def _assert_integer(costs):
    # Check that costs are not changed by rounding.
    # Note: Elements of cost matrix may be nan, inf, -inf.
    np.testing.assert_equal(np.round(costs), costs)


def _ortools_assert_is_optimal(pywrapgraph, status):
    if status == pywrapgraph.LinearSumAssignment.OPTIMAL:
        pass
    elif status == pywrapgraph.LinearSumAssignment.INFEASIBLE:
        raise AssertionError('ortools: infeasible assignment problem')
    elif status == pywrapgraph.LinearSumAssignment.POSSIBLE_OVERFLOW:
        raise AssertionError('ortools: possible overflow in assignment problem')
    else:
        raise AssertionError('ortools: unknown status')


def _ortools_extract_solution(assignment):
    if assignment.NumNodes() == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    pairings = []
    for i in range(assignment.NumNodes()):
        pairings.append([i, assignment.RightMate(i)])

    indices = np.array(pairings, dtype=int)
    return indices[:, 0], indices[:, 1]


def lsa_solve_lapjv(costs):
    """Solves the LSA problem using lap.lapjv()."""

    from lap import lapjv

    # The lap.lapjv function supports +inf edges but there are some issues.
    # https://github.com/gatagat/lap/issues/20
    # Therefore, replace nans with large finite cost.
    finite_costs = add_expensive_edges(costs)
    row_to_col, _ = lapjv(finite_costs, return_cost=False, extend_cost=True)
    indices = np.array([np.arange(costs.shape[0]), row_to_col], dtype=int).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]  # pylint: disable=unsubscriptable-object
    rids, cids = indices[:, 0], indices[:, 1]
    # Ensure that no missing edges were chosen.
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


available_solvers = None
default_solver = None
solver_map = None


def _init_standard_solvers():
    global available_solvers, default_solver, solver_map  # pylint: disable=global-statement

    solvers = [
        ('lapsolver', lsa_solve_lapsolver),
        ('lap', lsa_solve_lapjv),
        ('scipy', lsa_solve_scipy),
        ('munkres', lsa_solve_munkres),
        ('ortools', lsa_solve_ortools),
    ]

    solver_map = dict(solvers)

    available_solvers = [s[0] for s in solvers if _module_is_available(s[0])]
    if len(available_solvers) == 0:
        default_solver = None
        warnings.warn('No standard LAP solvers found. Consider `pip install lapsolver` or `pip install scipy`', category=RuntimeWarning)
    else:
        default_solver = available_solvers[0]


_init_standard_solvers()


@contextmanager
def set_default_solver(newsolver):
    """Change the default solver within context.

    Intended usage

        costs = ...
        mysolver = lambda x: ... # solver code that returns pairings

        with lap.set_default_solver(mysolver):
            rids, cids = lap.linear_sum_assignment(costs)

    Params
    ------
    newsolver : callable or str
        new solver function
    """

    global default_solver  # pylint: disable=global-statement

    oldsolver = default_solver
    try:
        default_solver = newsolver
        yield
    finally:
        default_solver = oldsolver
