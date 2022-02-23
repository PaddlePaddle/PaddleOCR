# Wrapper for the shortest augmenting path algorithm for solving the
# rectangular linear sum assignment problem.  The original code was an
# implementation of the Hungarian algorithm (Kuhn-Munkres) taken from
# scikit-learn, based on original code by Brian Clapper and adapted to NumPy
# by Gael Varoquaux. Further improvements by Ben Root, Vlad Niculae, Lars
# Buitinck, and Peter Larsen.
#
# Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
# Author: Brian M. Clapper, Gael Varoquaux
# License: 3-clause BSD

import numpy as np
from . import _lsap_module


def linear_sum_assignment(cost_matrix, maximize=False):
    """Solve the linear sum assignment problem.

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    maximize : bool (default: False)
        Calculates a maximum weight matching if true.

    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.

    See Also
    --------
    scipy.sparse.csgraph.min_weight_full_bipartite_matching : for sparse inputs

    Notes
    -----

    The linear sum assignment problem [1]_ is also known as minimum weight
    matching in bipartite graphs. A problem instance is described by a matrix
    C, where each C[i,j] is the cost of matching vertex i of the first partite
    set (a "worker") and vertex j of the second set (a "job"). The goal is to
    find a complete assignment of workers to jobs of minimal cost.

    Formally, let X be a boolean matrix where :math:`X[i,j] = 1` iff row i is
    assigned to column j. Then the optimal assignment has cost

    .. math::
        \\min \\sum_i \\sum_j C_{i,j} X_{i,j}

    where, in the case where the matrix X is square, each row is assigned to
    exactly one column, and each column to exactly one row.

    This function can also solve a generalization of the classic assignment
    problem where the cost matrix is rectangular. If it has more rows than
    columns, then not every row needs to be assigned to a column, and vice
    versa.

    This implementation is a modified Jonker-Volgenant algorithm with no
    initialization, described in ref. [2]_.

    .. versionadded:: 0.17.0

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Assignment_problem

    .. [2] DF Crouse. On implementing 2D rectangular assignment algorithms.
           *IEEE Transactions on Aerospace and Electronic Systems*,
           52(4):1679-1696, August 2016, :doi:`10.1109/TAES.2016.140952`

    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> from scipy.optimize import linear_sum_assignment
    >>> row_ind, col_ind = linear_sum_assignment(cost)
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind].sum()
    5
    """
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.ndim != 2:
        raise ValueError("expected a matrix (2-D array), got a %r array"
                         % (cost_matrix.shape,))

    if not (np.issubdtype(cost_matrix.dtype, np.number) or
            cost_matrix.dtype == np.dtype(np.bool_)):
        raise ValueError("expected a matrix containing numerical entries, got %s"
                         % (cost_matrix.dtype,))

    if maximize:
        cost_matrix = -cost_matrix

    return _lsap_module.calculate_assignment(cost_matrix)
