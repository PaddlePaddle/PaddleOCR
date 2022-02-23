"""Revised simplex method for linear programming

The *revised simplex* method uses the method described in [1]_, except
that a factorization [2]_ of the basis matrix, rather than its inverse,
is efficiently maintained and used to solve the linear systems at each
iteration of the algorithm.

.. versionadded:: 1.3.0

References
----------
.. [1] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
.. [2] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.

"""
# Author: Matt Haberland

import numpy as np
from scipy.linalg import solve
from .optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from scipy.linalg import LinAlgError
from numpy.linalg.linalg import LinAlgError as LinAlgError2
from ._linprog_util import _postsolve
from .optimize import OptimizeResult


def _phase_one(A, b, x0, callback, postsolve_args, maxiter, tol, disp,
               maxupdate, mast, pivot):
    """
    The purpose of phase one is to find an initial basic feasible solution
    (BFS) to the original problem.

    Generates an auxiliary problem with a trivial BFS and an objective that
    minimizes infeasibility of the original problem. Solves the auxiliary
    problem using the main simplex routine (phase two). This either yields
    a BFS to the original problem or determines that the original problem is
    infeasible. If feasible, phase one detects redundant rows in the original
    constraint matrix and removes them, then chooses additional indices as
    necessary to complete a basis/BFS for the original problem.
    """

    m, n = A.shape
    status = 0

    # generate auxiliary problem to get initial BFS
    A, b, c, basis, x, status = _generate_auxiliary_problem(A, b, x0, tol)

    if status == 6:
        residual = c.dot(x)
        iter_k = 0
        return x, basis, A, b, residual, status, iter_k

    # solve auxiliary problem
    phase_one_n = n
    iter_k = 0
    x, basis, status, iter_k = _phase_two(c, A, x, basis, callback,
                                          postsolve_args,
                                          maxiter, tol, disp,
                                          maxupdate, mast, pivot,
                                          iter_k, phase_one_n)

    # check for infeasibility
    residual = c.dot(x)
    if status == 0 and residual > tol:
        status = 2

    # drive artificial variables out of basis
    # TODO: test redundant row removal better
    # TODO: make solve more efficient with BGLU? This could take a while.
    keep_rows = np.ones(m, dtype=bool)
    for basis_column in basis[basis >= n]:
        B = A[:, basis]
        try:
            basis_finder = np.abs(solve(B, A))  # inefficient
            pertinent_row = np.argmax(basis_finder[:, basis_column])
            eligible_columns = np.ones(n, dtype=bool)
            eligible_columns[basis[basis < n]] = 0
            eligible_column_indices = np.where(eligible_columns)[0]
            index = np.argmax(basis_finder[:, :n]
                              [pertinent_row, eligible_columns])
            new_basis_column = eligible_column_indices[index]
            if basis_finder[pertinent_row, new_basis_column] < tol:
                keep_rows[pertinent_row] = False
            else:
                basis[basis == basis_column] = new_basis_column
        except (LinAlgError, LinAlgError2):
            status = 4

    # form solution to original problem
    A = A[keep_rows, :n]
    basis = basis[keep_rows]
    x = x[:n]
    m = A.shape[0]
    return x, basis, A, b, residual, status, iter_k


def _get_more_basis_columns(A, basis):
    """
    Called when the auxiliary problem terminates with artificial columns in
    the basis, which must be removed and replaced with non-artificial
    columns. Finds additional columns that do not make the matrix singular.
    """
    m, n = A.shape

    # options for inclusion are those that aren't already in the basis
    a = np.arange(m+n)
    bl = np.zeros(len(a), dtype=bool)
    bl[basis] = 1
    options = a[~bl]
    options = options[options < n]  # and they have to be non-artificial

    # form basis matrix
    B = np.zeros((m, m))
    B[:, 0:len(basis)] = A[:, basis]

    if (basis.size > 0 and
            np.linalg.matrix_rank(B[:, :len(basis)]) < len(basis)):
        raise Exception("Basis has dependent columns")

    rank = 0  # just enter the loop
    for i in range(n):  # somewhat arbitrary, but we need another way out
        # permute the options, and take as many as needed
        new_basis = np.random.permutation(options)[:m-len(basis)]
        B[:, len(basis):] = A[:, new_basis]  # update the basis matrix
        rank = np.linalg.matrix_rank(B)      # check the rank
        if rank == m:
            break

    return np.concatenate((basis, new_basis))


def _generate_auxiliary_problem(A, b, x0, tol):
    """
    Modifies original problem to create an auxiliary problem with a trivial
    initial basic feasible solution and an objective that minimizes
    infeasibility in the original problem.

    Conceptually, this is done by stacking an identity matrix on the right of
    the original constraint matrix, adding artificial variables to correspond
    with each of these new columns, and generating a cost vector that is all
    zeros except for ones corresponding with each of the new variables.

    A initial basic feasible solution is trivial: all variables are zero
    except for the artificial variables, which are set equal to the
    corresponding element of the right hand side `b`.

    Runnning the simplex method on this auxiliary problem drives all of the
    artificial variables - and thus the cost - to zero if the original problem
    is feasible. The original problem is declared infeasible otherwise.

    Much of the complexity below is to improve efficiency by using singleton
    columns in the original problem where possible, thus generating artificial
    variables only as necessary, and using an initial 'guess' basic feasible
    solution.
    """
    status = 0
    m, n = A.shape

    if x0 is not None:
        x = x0
    else:
        x = np.zeros(n)

    r = b - A@x  # residual; this must be all zeros for feasibility

    A[r < 0] = -A[r < 0]  # express problem with RHS positive for trivial BFS
    b[r < 0] = -b[r < 0]  # to the auxiliary problem
    r[r < 0] *= -1

    # Rows which we will need to find a trivial way to zero.
    # This should just be the rows where there is a nonzero residual.
    # But then we would not necessarily have a column singleton in every row.
    # This makes it difficult to find an initial basis.
    if x0 is None:
        nonzero_constraints = np.arange(m)
    else:
        nonzero_constraints = np.where(r > tol)[0]

    # these are (at least some of) the initial basis columns
    basis = np.where(np.abs(x) > tol)[0]

    if len(nonzero_constraints) == 0 and len(basis) <= m:  # already a BFS
        c = np.zeros(n)
        basis = _get_more_basis_columns(A, basis)
        return A, b, c, basis, x, status
    elif (len(nonzero_constraints) > m - len(basis) or
          np.any(x < 0)):  # can't get trivial BFS
        c = np.zeros(n)
        status = 6
        return A, b, c, basis, x, status

    # chooses existing columns appropriate for inclusion in initial basis
    cols, rows = _select_singleton_columns(A, r)

    # find the rows we need to zero that we _can_ zero with column singletons
    i_tofix = np.isin(rows, nonzero_constraints)
    # these columns can't already be in the basis, though
    # we are going to add them to the basis and change the corresponding x val
    i_notinbasis = np.logical_not(np.isin(cols, basis))
    i_fix_without_aux = np.logical_and(i_tofix, i_notinbasis)
    rows = rows[i_fix_without_aux]
    cols = cols[i_fix_without_aux]

    # indices of the rows we can only zero with auxiliary variable
    # these rows will get a one in each auxiliary column
    arows = nonzero_constraints[np.logical_not(
                                np.isin(nonzero_constraints, rows))]
    n_aux = len(arows)
    acols = n + np.arange(n_aux)          # indices of auxiliary columns

    basis_ng = np.concatenate((cols, acols))   # basis columns not from guess
    basis_ng_rows = np.concatenate((rows, arows))  # rows we need to zero

    # add auxiliary singleton columns
    A = np.hstack((A, np.zeros((m, n_aux))))
    A[arows, acols] = 1

    # generate initial BFS
    x = np.concatenate((x, np.zeros(n_aux)))
    x[basis_ng] = r[basis_ng_rows]/A[basis_ng_rows, basis_ng]

    # generate costs to minimize infeasibility
    c = np.zeros(n_aux + n)
    c[acols] = 1

    # basis columns correspond with nonzeros in guess, those with column
    # singletons we used to zero remaining constraints, and any additional
    # columns to get a full set (m columns)
    basis = np.concatenate((basis, basis_ng))
    basis = _get_more_basis_columns(A, basis)  # add columns as needed

    return A, b, c, basis, x, status


def _select_singleton_columns(A, b):
    """
    Finds singleton columns for which the singleton entry is of the same sign
    as the right-hand side; these columns are eligible for inclusion in an
    initial basis. Determines the rows in which the singleton entries are
    located. For each of these rows, returns the indices of the one singleton
    column and its corresponding row.
    """
    # find indices of all singleton columns and corresponding row indicies
    column_indices = np.nonzero(np.sum(np.abs(A) != 0, axis=0) == 1)[0]
    columns = A[:, column_indices]          # array of singleton columns
    row_indices = np.zeros(len(column_indices), dtype=int)
    nonzero_rows, nonzero_columns = np.nonzero(columns)
    row_indices[nonzero_columns] = nonzero_rows   # corresponding row indicies

    # keep only singletons with entries that have same sign as RHS
    # this is necessary because all elements of BFS must be non-negative
    same_sign = A[row_indices, column_indices]*b[row_indices] >= 0
    column_indices = column_indices[same_sign][::-1]
    row_indices = row_indices[same_sign][::-1]
    # Reversing the order so that steps below select rightmost columns
    # for initial basis, which will tend to be slack variables. (If the
    # guess corresponds with a basic feasible solution but a constraint
    # is not satisfied with the corresponding slack variable zero, the slack
    # variable must be basic.)

    # for each row, keep rightmost singleton column with an entry in that row
    unique_row_indices, first_columns = np.unique(row_indices,
                                                  return_index=True)
    return column_indices[first_columns], unique_row_indices


def _find_nonzero_rows(A, tol):
    """
    Returns logical array indicating the locations of rows with at least
    one nonzero element.
    """
    return np.any(np.abs(A) > tol, axis=1)


def _select_enter_pivot(c_hat, bl, a, rule="bland", tol=1e-12):
    """
    Selects a pivot to enter the basis. Currently Bland's rule - the smallest
    index that has a negative reduced cost - is the default.
    """
    if rule.lower() == "mrc":  # index with minimum reduced cost
        return a[~bl][np.argmin(c_hat)]
    else:  # smallest index w/ negative reduced cost
        return a[~bl][c_hat < -tol][0]


def _display_iter(phase, iteration, slack, con, fun):
    """
    Print indicators of optimization status to the console.
    """
    header = True if not iteration % 20 else False

    if header:
        print("Phase",
              "Iteration",
              "Minimum Slack      ",
              "Constraint Residual",
              "Objective          ")

    # :<X.Y left aligns Y digits in X digit spaces
    fmt = '{0:<6}{1:<10}{2:<20.13}{3:<20.13}{4:<20.13}'
    try:
        slack = np.min(slack)
    except ValueError:
        slack = "NA"
    print(fmt.format(phase, iteration, slack, np.linalg.norm(con), fun))


def _display_and_callback(phase_one_n, x, postsolve_args, status,
                          iteration, disp, callback):
    if phase_one_n is not None:
        phase = 1
        x_postsolve = x[:phase_one_n]
    else:
        phase = 2
        x_postsolve = x
    x_o, fun, slack, con = _postsolve(x_postsolve,
                                      postsolve_args)

    if callback is not None:
        res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
                              'con': con, 'nit': iteration,
                              'phase': phase, 'complete': False,
                              'status': status, 'message': "",
                              'success': False})
        callback(res)
    if disp:
        _display_iter(phase, iteration, slack, con, fun)


def _phase_two(c, A, x, b, callback, postsolve_args, maxiter, tol, disp,
               maxupdate, mast, pivot, iteration=0, phase_one_n=None):
    """
    The heart of the simplex method. Beginning with a basic feasible solution,
    moves to adjacent basic feasible solutions successively lower reduced cost.
    Terminates when there are no basic feasible solutions with lower reduced
    cost or if the problem is determined to be unbounded.

    This implementation follows the revised simplex method based on LU
    decomposition. Rather than maintaining a tableau or an inverse of the
    basis matrix, we keep a factorization of the basis matrix that allows
    efficient solution of linear systems while avoiding stability issues
    associated with inverted matrices.
    """
    m, n = A.shape
    status = 0
    a = np.arange(n)                    # indices of columns of A
    ab = np.arange(m)                   # indices of columns of B
    if maxupdate:
        # basis matrix factorization object; similar to B = A[:, b]
        B = BGLU(A, b, maxupdate, mast)
    else:
        B = LU(A, b)

    for iteration in range(iteration, maxiter):

        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status,
                                  iteration, disp, callback)

        bl = np.zeros(len(a), dtype=bool)
        bl[b] = 1

        xb = x[b]       # basic variables
        cb = c[b]       # basic costs

        try:
            v = B.solve(cb, transposed=True)    # similar to v = solve(B.T, cb)
        except LinAlgError:
            status = 4
            break

        # TODO: cythonize?
        c_hat = c - v.dot(A)    # reduced cost
        c_hat = c_hat[~bl]
        # Above is much faster than:
        # N = A[:, ~bl]                 # slow!
        # c_hat = c[~bl] - v.T.dot(N)
        # Can we perform the multiplication only on the nonbasic columns?

        if np.all(c_hat >= -tol):  # all reduced costs positive -> terminate
            break

        j = _select_enter_pivot(c_hat, bl, a, rule=pivot, tol=tol)
        u = B.solve(A[:, j])        # similar to u = solve(B, A[:, j])

        i = u > tol                 # if none of the u are positive, unbounded
        if not np.any(i):
            status = 3
            break

        th = xb[i]/u[i]
        l = np.argmin(th)           # implicitly selects smallest subscript
        th_star = th[l]             # step size

        x[b] = x[b] - th_star*u     # take step
        x[j] = th_star
        B.update(ab[i][l], j)       # modify basis
        b = B.b                     # similar to b[ab[i][l]] =

    else:
        # If the end of the for loop is reached (without a break statement),
        # then another step has been taken, so the iteration counter should
        # increment, info should be displayed, and callback should be called.
        iteration += 1
        status = 1
        if disp or callback is not None:
            _display_and_callback(phase_one_n, x, postsolve_args, status,
                                  iteration, disp, callback)

    return x, b, status, iteration


def _linprog_rs(c, c0, A, b, x0, callback, postsolve_args,
                maxiter=5000, tol=1e-12, disp=False,
                maxupdate=10, mast=False, pivot="mrc",
                **unknown_options):
    """
    Solve the following linear programming problem via a two-phase
    revised simplex algorithm.::

        minimize:     c @ x

        subject to:  A @ x == b
                     0 <= x < oo

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Currently unused.)
    A : 2-D array
        2-D array which, when matrix-multiplied by ``x``, gives the values of
        the equality constraints at ``x``.
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    x0 : 1-D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm. For the revised simplex method, these must
        correspond with a basic feasible solution.
    callback : callable, optional
        If a callback function is provided, it will be called within each
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

            x : 1-D array
                Current solution vector.
            fun : float
                Current value of the objective function ``c @ x``.
            success : bool
                True only when an algorithm has completed successfully,
                so this is always False as the callback function is called
                only while the algorithm is still iterating.
            slack : 1-D array
                The values of the slack variables. Each slack variable
                corresponds to an inequality constraint. If the slack is zero,
                the corresponding constraint is active.
            con : 1-D array
                The (nominally zero) residuals of the equality constraints,
                that is, ``b - A_eq @ x``.
            phase : int
                The phase of the algorithm being executed.
            status : int
                For revised simplex, this is always 0 because if a different
                status is detected, the algorithm terminates.
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem.

    Options
    -------
    maxiter : int
       The maximum number of iterations to perform in either phase.
    tol : float
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    maxupdate : int
        The maximum number of updates performed on the LU factorization.
        After this many updates is reached, the basis matrix is factorized
        from scratch.
    mast : bool
        Minimize Amortized Solve Time. If enabled, the average time to solve
        a linear system using the basis factorization is measured. Typically,
        the average solve time will decrease with each successive solve after
        initial factorization, as factorization takes much more time than the
        solve operation (and updates). Eventually, however, the updated
        factorization becomes sufficiently complex that the average solve time
        begins to increase. When this is detected, the basis is refactorized
        from scratch. Enable this option to maximize speed at the risk of
        nondeterministic behavior. Ignored if ``maxupdate`` is 0.
    pivot : "mrc" or "bland"
        Pivot rule: Minimum Reduced Cost (default) or Bland's rule. Choose
        Bland's rule if iteration limit is reached and cycling is suspected.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

    Returns
    -------
    x : 1-D array
        Solution vector.
    status : int
        An integer representing the exit status of the optimization::

         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Numerical difficulties encountered
         5 : No constraints; turn presolve on
         6 : Guess x0 cannot be converted to a basic feasible solution

    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem.
    """

    _check_unknown_options(unknown_options)

    messages = ["Optimization terminated successfully.",
                "Iteration limit reached.",
                "The problem appears infeasible, as the phase one auxiliary "
                "problem terminated successfully with a residual of {0:.1e}, "
                "greater than the tolerance {1} required for the solution to "
                "be considered feasible. Consider increasing the tolerance to "
                "be greater than {0:.1e}. If this tolerance is unnaceptably "
                "large, the problem is likely infeasible.",
                "The problem is unbounded, as the simplex algorithm found "
                "a basic feasible solution from which there is a direction "
                "with negative reduced cost in which all decision variables "
                "increase.",
                "Numerical difficulties encountered; consider trying "
                "method='interior-point'.",
                "Problems with no constraints are trivially solved; please "
                "turn presolve on.",
                "The guess x0 cannot be converted to a basic feasible "
                "solution. "
                ]

    if A.size == 0:  # address test_unbounded_below_no_presolve_corrected
        return np.zeros(c.shape), 5, messages[5], 0

    x, basis, A, b, residual, status, iteration = (
        _phase_one(A, b, x0, callback, postsolve_args,
                   maxiter, tol, disp, maxupdate, mast, pivot))

    if status == 0:
        x, basis, status, iteration = _phase_two(c, A, x, basis, callback,
                                                 postsolve_args,
                                                 maxiter, tol, disp,
                                                 maxupdate, mast, pivot,
                                                 iteration)

    return x, status, messages[status].format(residual, tol), iteration
