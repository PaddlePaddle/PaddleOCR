"""The adaptation of Trust Region Reflective algorithm for a linear
least-squares problem."""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult

from .givens_elimination import givens_elimination
from .common import (
    EPS, step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
    minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
    print_header_linear, print_iteration_linear, compute_grad,
    regularized_lsq_operator, right_multiplied_operator)


def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
    """Solve regularized least squares using information from QR-decomposition.

    The initial problem is to solve the following system in a least-squares
    sense:
    ::

        A x = b
        D x = 0

    where D is diagonal matrix. The method is based on QR decomposition
    of the form A P = Q R, where P is a column permutation matrix, Q is an
    orthogonal matrix and R is an upper triangular matrix.

    Parameters
    ----------
    m, n : int
        Initial shape of A.
    R : ndarray, shape (n, n)
        Upper triangular matrix from QR decomposition of A.
    QTb : ndarray, shape (n,)
        First n components of Q^T b.
    perm : ndarray, shape (n,)
        Array defining column permutation of A, such that ith column of
        P is perm[i]-th column of identity matrix.
    diag : ndarray, shape (n,)
        Array containing diagonal elements of D.

    Returns
    -------
    x : ndarray, shape (n,)
        Found least-squares solution.
    """
    if copy_R:
        R = R.copy()
    v = QTb.copy()

    givens_elimination(R, v, diag[perm])

    abs_diag_R = np.abs(np.diag(R))
    threshold = EPS * max(m, n) * np.max(abs_diag_R)
    nns, = np.nonzero(abs_diag_R > threshold)

    R = R[np.ix_(nns, nns)]
    v = v[nns]

    x = np.zeros(n)
    x[perm[nns]] = solve_triangular(R, v)

    return x


def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """Find an appropriate step size using backtracking line search."""
    alpha = 1
    while True:
        x_new, _ = reflective_transformation(x + alpha * p, lb, ub)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)
        if cost_change > -0.1 * alpha * p_dot_g:
            break
        alpha *= 0.5

    active = find_active_constraints(x_new, lb, ub)
    if np.any(active != 0):
        x_new, _ = reflective_transformation(x + theta * alpha * p, lb, ub)
        x_new = make_strictly_feasible(x_new, lb, ub, rstep=0)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)

    return x, step, cost_change


def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        return p

    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Find the step size along reflected direction.
    r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Stay interior.
    r_stride_l = (1 - theta) * r_stride_u
    r_stride_u *= theta

    if r_stride_u > 0:
        a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h = p_h + r_h * r_stride
        r = d * r_h
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p_h *= theta
    p *= theta
    p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)

    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    else:
        return ag


def trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol, max_iter,
               verbose):
    m, n = A.shape
    x, _ = reflective_transformation(x_lsq, lb, ub)
    x = make_strictly_feasible(x, lb, ub, rstep=0.1)

    if lsq_solver == 'exact':
        QT, R, perm = qr(A, mode='economic', pivoting=True)
        QT = QT.T

        if m < n:
            R = np.vstack((R, np.zeros((n - m, n))))

        QTr = np.zeros(n)
        k = min(m, n)
    elif lsq_solver == 'lsmr':
        r_aug = np.zeros(m + n)
        auto_lsmr_tol = False
        if lsmr_tol is None:
            lsmr_tol = 1e-2 * tol
        elif lsmr_tol == 'auto':
            auto_lsmr_tol = True

    r = A.dot(x) - b
    g = compute_grad(A, r)
    cost = 0.5 * np.dot(r, r)
    initial_cost = cost

    termination_status = None
    step_norm = None
    cost_change = None

    if max_iter is None:
        max_iter = 100

    if verbose == 2:
        print_header_linear()

    for iteration in range(max_iter):
        v, dv = CL_scaling_vector(x, g, lb, ub)
        g_scaled = g * v
        g_norm = norm(g_scaled, ord=np.inf)
        if g_norm < tol:
            termination_status = 1

        if verbose == 2:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, g_norm)

        if termination_status is not None:
            break

        diag_h = g * dv
        diag_root_h = diag_h ** 0.5
        d = v ** 0.5
        g_h = d * g

        A_h = right_multiplied_operator(A, d)
        if lsq_solver == 'exact':
            QTr[:k] = QT.dot(r)
            p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm,
                                           diag_root_h, copy_R=False)
        elif lsq_solver == 'lsmr':
            lsmr_op = regularized_lsq_operator(A_h, diag_root_h)
            r_aug[:m] = r
            if auto_lsmr_tol:
                eta = 1e-2 * min(0.5, g_norm)
                lsmr_tol = max(EPS, min(0.1, eta * g_norm))
            p_h = -lsmr(lsmr_op, r_aug, atol=lsmr_tol, btol=lsmr_tol)[0]

        p = d * p_h

        p_dot_g = np.dot(p, g)
        if p_dot_g > 0:
            termination_status = -1

        theta = 1 - min(0.005, g_norm)
        step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
        cost_change = -evaluate_quadratic(A, g, step)

        # Perhaps almost never executed, the idea is that `p` is descent
        # direction thus we must find acceptable cost decrease using simple
        # "backtracking", otherwise the algorithm's logic would break.
        if cost_change < 0:
            x, step, cost_change = backtracking(
                A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            x = make_strictly_feasible(x + step, lb, ub, rstep=0)

        step_norm = norm(step)
        r = A.dot(x) - b
        g = compute_grad(A, r)

        if cost_change < tol * cost:
            termination_status = 2

        cost = 0.5 * np.dot(r, r)

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=tol)

    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=g_norm, active_mask=active_mask,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
