"""Linear least squares with bound constraints on independent variables."""
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.optimize import OptimizeResult

from .common import in_bounds, compute_grad
from .trf_linear import trf_linear
from .bvls import bvls


def prepare_bounds(bounds, n):
    lb, ub = [np.asarray(b, dtype=float) for b in bounds]

    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


TERMINATION_MESSAGES = {
    -1: "The algorithm was not able to make progress on the last iteration.",
    0: "The maximum number of iterations is exceeded.",
    1: "The first-order optimality measure is less than `tol`.",
    2: "The relative change of the cost function is less than `tol`.",
    3: "The unconstrained solution is optimal."
}


def lsq_linear(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
               lsq_solver=None, lsmr_tol=None, max_iter=None, verbose=0):
    r"""Solve a linear least-squares problem with bounds on the variables.

    Given a m-by-n design matrix A and a target vector b with m elements,
    `lsq_linear` solves the following optimization problem::

        minimize 0.5 * ||A x - b||**2
        subject to lb <= x <= ub

    This optimization problem is convex, hence a found minimum (if iterations
    have converged) is guaranteed to be global.

    Parameters
    ----------
    A : array_like, sparse matrix of LinearOperator, shape (m, n)
        Design matrix. Can be `scipy.sparse.linalg.LinearOperator`.
    b : array_like, shape (m,)
        Target vector.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must have shape (n,) or be a scalar, in the latter
        case a bound will be the same for all variables. Use ``np.inf`` with
        an appropriate sign to disable bounds on all or some variables.
    method : 'trf' or 'bvls', optional
        Method to perform minimization.

            * 'trf' : Trust Region Reflective algorithm adapted for a linear
              least-squares problem. This is an interior-point-like method
              and the required number of iterations is weakly correlated with
              the number of variables.
            * 'bvls' : Bounded-variable least-squares algorithm. This is
              an active set method, which requires the number of iterations
              comparable to the number of variables. Can't be used when `A` is
              sparse or LinearOperator.

        Default is 'trf'.
    tol : float, optional
        Tolerance parameter. The algorithm terminates if a relative change
        of the cost function is less than `tol` on the last iteration.
        Additionally, the first-order optimality measure is considered:

            * ``method='trf'`` terminates if the uniform norm of the gradient,
              scaled to account for the presence of the bounds, is less than
              `tol`.
            * ``method='bvls'`` terminates if Karush-Kuhn-Tucker conditions
              are satisfied within `tol` tolerance.

    lsq_solver : {None, 'exact', 'lsmr'}, optional
        Method of solving unbounded least-squares problems throughout
        iterations:

            * 'exact' : Use dense QR or SVD decomposition approach. Can't be
              used when `A` is sparse or LinearOperator.
            * 'lsmr' : Use `scipy.sparse.linalg.lsmr` iterative procedure
              which requires only matrix-vector product evaluations. Can't
              be used with ``method='bvls'``.

        If None (default), the solver is chosen based on type of `A`.
    lsmr_tol : None, float or 'auto', optional
        Tolerance parameters 'atol' and 'btol' for `scipy.sparse.linalg.lsmr`
        If None (default), it is set to ``1e-2 * tol``. If 'auto', the
        tolerance will be adjusted based on the optimality of the current
        iterate, which can speed up the optimization process, but is not always
        reliable.
    max_iter : None or int, optional
        Maximum number of iterations before termination. If None (default), it
        is set to 100 for ``method='trf'`` or to the number of variables for
        ``method='bvls'`` (not counting iterations for 'bvls' initialization).
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.

    Returns
    -------
    OptimizeResult with the following fields defined:
    x : ndarray, shape (n,)
        Solution found.
    cost : float
        Value of the cost function at the solution.
    fun : ndarray, shape (m,)
        Vector of residuals at the solution.
    optimality : float
        First-order optimality measure. The exact meaning depends on `method`,
        refer to the description of `tol` parameter.
    active_mask : ndarray of int, shape (n,)
        Each component shows whether a corresponding constraint is active
        (that is, whether a variable is at the bound):

            *  0 : a constraint is not active.
            * -1 : a lower bound is active.
            *  1 : an upper bound is active.

        Might be somewhat arbitrary for the `trf` method as it generates a
        sequence of strictly feasible iterates and active_mask is determined
        within a tolerance threshold.
    nit : int
        Number of iterations. Zero if the unconstrained solution is optimal.
    status : int
        Reason for algorithm termination:

            * -1 : the algorithm was not able to make progress on the last
              iteration.
            *  0 : the maximum number of iterations is exceeded.
            *  1 : the first-order optimality measure is less than `tol`.
            *  2 : the relative change of the cost function is less than `tol`.
            *  3 : the unconstrained solution is optimal.

    message : str
        Verbal description of the termination reason.
    success : bool
        True if one of the convergence criteria is satisfied (`status` > 0).

    See Also
    --------
    nnls : Linear least squares with non-negativity constraint.
    least_squares : Nonlinear least squares with bounds on the variables.

    Notes
    -----
    The algorithm first computes the unconstrained least-squares solution by
    `numpy.linalg.lstsq` or `scipy.sparse.linalg.lsmr` depending on
    `lsq_solver`. This solution is returned as optimal if it lies within the
    bounds.

    Method 'trf' runs the adaptation of the algorithm described in [STIR]_ for
    a linear least-squares problem. The iterations are essentially the same as
    in the nonlinear least-squares algorithm, but as the quadratic function
    model is always accurate, we don't need to track or modify the radius of
    a trust region. The line search (backtracking) is used as a safety net
    when a selected step does not decrease the cost function. Read more
    detailed description of the algorithm in `scipy.optimize.least_squares`.

    Method 'bvls' runs a Python implementation of the algorithm described in
    [BVLS]_. The algorithm maintains active and free sets of variables, on
    each iteration chooses a new variable to move from the active set to the
    free set and then solves the unconstrained least-squares problem on free
    variables. This algorithm is guaranteed to give an accurate solution
    eventually, but may require up to n iterations for a problem with n
    variables. Additionally, an ad-hoc initialization procedure is
    implemented, that determines which variables to set free or active
    initially. It takes some number of iterations before actual BVLS starts,
    but can significantly reduce the number of further iterations.

    References
    ----------
    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
              and Conjugate Gradient Method for Large-Scale Bound-Constrained
              Minimization Problems," SIAM Journal on Scientific Computing,
              Vol. 21, Number 1, pp 1-23, 1999.
    .. [BVLS] P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares:
              an Algorithm and Applications", Computational Statistics, 10,
              129-141, 1995.

    Examples
    --------
    In this example, a problem with a large sparse matrix and bounds on the
    variables is solved.

    >>> from scipy.sparse import rand
    >>> from scipy.optimize import lsq_linear
    >>> rng = np.random.default_rng()
    ...
    >>> m = 20000
    >>> n = 10000
    ...
    >>> A = rand(m, n, density=1e-4, random_state=rng)
    >>> b = rng.standard_normal(m)
    ...
    >>> lb = rng.standard_normal(n)
    >>> ub = lb + 1
    ...
    >>> res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
    # may vary
    The relative change of the cost function is less than `tol`.
    Number of iterations 16, initial cost 1.5039e+04, final cost 1.1112e+04,
    first-order optimality 4.66e-08.
    """
    if method not in ['trf', 'bvls']:
        raise ValueError("`method` must be 'trf' or 'bvls'")

    if lsq_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`solver` must be None, 'exact' or 'lsmr'.")

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if issparse(A):
        A = csr_matrix(A)
    elif not isinstance(A, LinearOperator):
        A = np.atleast_2d(np.asarray(A))

    if method == 'bvls':
        if lsq_solver == 'lsmr':
            raise ValueError("method='bvls' can't be used with "
                             "lsq_solver='lsmr'")

        if not isinstance(A, np.ndarray):
            raise ValueError("method='bvls' can't be used with `A` being "
                             "sparse or LinearOperator.")

    if lsq_solver is None:
        if isinstance(A, np.ndarray):
            lsq_solver = 'exact'
        else:
            lsq_solver = 'lsmr'
    elif lsq_solver == 'exact' and not isinstance(A, np.ndarray):
        raise ValueError("`exact` solver can't be used when `A` is "
                         "sparse or LinearOperator.")

    if len(A.shape) != 2:  # No ndim for LinearOperator.
        raise ValueError("`A` must have at most 2 dimensions.")

    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")

    if max_iter is not None and max_iter <= 0:
        raise ValueError("`max_iter` must be None or positive integer.")

    m, n = A.shape

    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("`b` must have at most 1 dimension.")

    if b.size != m:
        raise ValueError("Inconsistent shapes between `A` and `b`.")

    lb, ub = prepare_bounds(bounds, n)

    if lb.shape != (n,) and ub.shape != (n,):
        raise ValueError("Bounds have wrong shape.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")

    if lsq_solver == 'exact':
        x_lsq = np.linalg.lstsq(A, b, rcond=-1)[0]
    elif lsq_solver == 'lsmr':
        x_lsq = lsmr(A, b, atol=tol, btol=tol)[0]

    if in_bounds(x_lsq, lb, ub):
        r = A @ x_lsq - b
        cost = 0.5 * np.dot(r, r)
        termination_status = 3
        termination_message = TERMINATION_MESSAGES[termination_status]
        g = compute_grad(A, r)
        g_norm = norm(g, ord=np.inf)

        if verbose > 0:
            print(termination_message)
            print("Final cost {0:.4e}, first-order optimality {1:.2e}"
                  .format(cost, g_norm))

        return OptimizeResult(
            x=x_lsq, fun=r, cost=cost, optimality=g_norm,
            active_mask=np.zeros(n), nit=0, status=termination_status,
            message=termination_message, success=True)

    if method == 'trf':
        res = trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol,
                         max_iter, verbose)
    elif method == 'bvls':
        res = bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose)

    res.message = TERMINATION_MESSAGES[res.status]
    res.success = res.status > 0

    if verbose > 0:
        print(res.message)
        print("Number of iterations {0}, initial cost {1:.4e}, "
              "final cost {2:.4e}, first-order optimality {3:.2e}."
              .format(res.nit, res.initial_cost, res.cost, res.optimality))

    del res.initial_cost

    return res
