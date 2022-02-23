"""
Dogleg algorithm with rectangular trust regions for least-squares minimization.

The description of the algorithm can be found in [Voglis]_. The algorithm does
trust-region iterations, but the shape of trust regions is rectangular as
opposed to conventional elliptical. The intersection of a trust region and
an initial feasible region is again some rectangle. Thus, on each iteration a
bound-constrained quadratic optimization problem is solved.

A quadratic problem is solved by well-known dogleg approach, where the
function is minimized along piecewise-linear "dogleg" path [NumOpt]_,
Chapter 4. If Jacobian is not rank-deficient then the function is decreasing
along this path, and optimization amounts to simply following along this
path as long as a point stays within the bounds. A constrained Cauchy step
(along the anti-gradient) is considered for safety in rank deficient cases,
in this situations the convergence might be slow.

If during iterations some variable hit the initial bound and the component
of anti-gradient points outside the feasible region, then a next dogleg step
won't make any progress. At this state such variables satisfy first-order
optimality conditions and they are excluded before computing a next dogleg
step.

Gauss-Newton step can be computed exactly by `numpy.linalg.lstsq` (for dense
Jacobian matrices) or by iterative procedure `scipy.sparse.linalg.lsmr` (for
dense and sparse matrices, or Jacobian being LinearOperator). The second
option allows to solve very large problems (up to couple of millions of
residuals on a regular PC), provided the Jacobian matrix is sufficiently
sparse. But note that dogbox is not very good for solving problems with
large number of constraints, because of variables exclusion-inclusion on each
iteration (a required number of function evaluations might be high or accuracy
of a solution will be poor), thus its large-scale usage is probably limited
to unconstrained problems.

References
----------
.. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region Dogleg
            Approach for Unconstrained and Bound Constrained Nonlinear
            Optimization", WSEAS International Conference on Applied
            Mathematics, Corfu, Greece, 2004.
.. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization, 2nd edition".
"""
import numpy as np
from numpy.linalg import lstsq, norm

from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy.optimize import OptimizeResult

from .common import (
    step_size_to_bound, in_bounds, update_tr_radius, evaluate_quadratic,
    build_quadratic_1d, minimize_quadratic_1d, compute_grad,
    compute_jac_scale, check_termination, scale_for_robust_loss_function,
    print_header_nonlinear, print_iteration_nonlinear)


def lsmr_operator(Jop, d, active_set):
    """Compute LinearOperator to use in LSMR by dogbox algorithm.

    `active_set` mask is used to excluded active variables from computations
    of matrix-vector products.
    """
    m, n = Jop.shape

    def matvec(x):
        x_free = x.ravel().copy()
        x_free[active_set] = 0
        return Jop.matvec(x * d)

    def rmatvec(x):
        r = d * Jop.rmatvec(x)
        r[active_set] = 0
        return r

    return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=float)


def find_intersection(x, tr_bounds, lb, ub):
    """Find intersection of trust-region bounds and initial bounds.

    Returns
    -------
    lb_total, ub_total : ndarray with shape of x
        Lower and upper bounds of the intersection region.
    orig_l, orig_u : ndarray of bool with shape of x
        True means that an original bound is taken as a corresponding bound
        in the intersection region.
    tr_l, tr_u : ndarray of bool with shape of x
        True means that a trust-region bound is taken as a corresponding bound
        in the intersection region.
    """
    lb_centered = lb - x
    ub_centered = ub - x

    lb_total = np.maximum(lb_centered, -tr_bounds)
    ub_total = np.minimum(ub_centered, tr_bounds)

    orig_l = np.equal(lb_total, lb_centered)
    orig_u = np.equal(ub_total, ub_centered)

    tr_l = np.equal(lb_total, -tr_bounds)
    tr_u = np.equal(ub_total, tr_bounds)

    return lb_total, ub_total, orig_l, orig_u, tr_l, tr_u


def dogleg_step(x, newton_step, g, a, b, tr_bounds, lb, ub):
    """Find dogleg step in a rectangular region.

    Returns
    -------
    step : ndarray, shape (n,)
        Computed dogleg step.
    bound_hits : ndarray of int, shape (n,)
        Each component shows whether a corresponding variable hits the
        initial bound after the step is taken:
            *  0 - a variable doesn't hit the bound.
            * -1 - lower bound is hit.
            *  1 - upper bound is hit.
    tr_hit : bool
        Whether the step hit the boundary of the trust-region.
    """
    lb_total, ub_total, orig_l, orig_u, tr_l, tr_u = find_intersection(
        x, tr_bounds, lb, ub
    )
    bound_hits = np.zeros_like(x, dtype=int)

    if in_bounds(newton_step, lb_total, ub_total):
        return newton_step, bound_hits, False

    to_bounds, _ = step_size_to_bound(np.zeros_like(x), -g, lb_total, ub_total)

    # The classical dogleg algorithm would check if Cauchy step fits into
    # the bounds, and just return it constrained version if not. But in a
    # rectangular trust region it makes sense to try to improve constrained
    # Cauchy step too. Thus, we don't distinguish these two cases.

    cauchy_step = -minimize_quadratic_1d(a, b, 0, to_bounds)[0] * g

    step_diff = newton_step - cauchy_step
    step_size, hits = step_size_to_bound(cauchy_step, step_diff,
                                         lb_total, ub_total)
    bound_hits[(hits < 0) & orig_l] = -1
    bound_hits[(hits > 0) & orig_u] = 1
    tr_hit = np.any((hits < 0) & tr_l | (hits > 0) & tr_u)

    return cauchy_step + step_size * step_diff, bound_hits, tr_hit


def dogbox(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale,
           loss_function, tr_solver, tr_options, verbose):
    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv, ord=np.inf)
    if Delta == 0:
        Delta = 1.0

    on_bound = np.zeros_like(x0, dtype=int)
    on_bound[np.equal(x0, lb)] = -1
    on_bound[np.equal(x0, ub)] = 1

    x = x0
    step = np.empty_like(x0)

    if max_nfev is None:
        max_nfev = x0.size * 100

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    if verbose == 2:
        print_header_nonlinear()

    while True:
        active_set = on_bound * g < 0
        free_set = ~active_set

        g_free = g[free_set]
        g_full = g.copy()
        g[active_set] = 0

        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                      step_norm, g_norm)

        if termination_status is not None or nfev == max_nfev:
            break

        x_free = x[free_set]
        lb_free = lb[free_set]
        ub_free = ub[free_set]
        scale_free = scale[free_set]

        # Compute (Gauss-)Newton and build quadratic model for Cauchy step.
        if tr_solver == 'exact':
            J_free = J[:, free_set]
            newton_step = lstsq(J_free, -f, rcond=-1)[0]

            # Coefficients for the quadratic model along the anti-gradient.
            a, b = build_quadratic_1d(J_free, g_free, -g_free)
        elif tr_solver == 'lsmr':
            Jop = aslinearoperator(J)

            # We compute lsmr step in scaled variables and then
            # transform back to normal variables, if lsmr would give exact lsq
            # solution, this would be equivalent to not doing any
            # transformations, but from experience it's better this way.

            # We pass active_set to make computations as if we selected
            # the free subset of J columns, but without actually doing any
            # slicing, which is expensive for sparse matrices and impossible
            # for LinearOperator.

            lsmr_op = lsmr_operator(Jop, scale, active_set)
            newton_step = -lsmr(lsmr_op, f, **tr_options)[0][free_set]
            newton_step *= scale_free

            # Components of g for active variables were zeroed, so this call
            # is correct and equivalent to using J_free and g_free.
            a, b = build_quadratic_1d(Jop, g, -g)

        actual_reduction = -1.0
        while actual_reduction <= 0 and nfev < max_nfev:
            tr_bounds = Delta * scale_free

            step_free, on_bound_free, tr_hit = dogleg_step(
                x_free, newton_step, g_free, a, b, tr_bounds, lb_free, ub_free)

            step.fill(0.0)
            step[free_set] = step_free

            if tr_solver == 'exact':
                predicted_reduction = -evaluate_quadratic(J_free, g_free,
                                                          step_free)
            elif tr_solver == 'lsmr':
                predicted_reduction = -evaluate_quadratic(Jop, g, step)

            # gh11403 ensure that solution is fully within bounds.
            x_new = np.clip(x + step, lb, ub)

            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step * scale_inv, ord=np.inf)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, tr_hit
            )

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)

            if termination_status is not None:
                break

        if actual_reduction > 0:
            on_bound[free_set] = on_bound_free

            x = x_new
            # Set variables exactly at the boundary.
            mask = on_bound == -1
            x[mask] = lb[mask]
            mask = on_bound == 1
            x[mask] = ub[mask]

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g_full, optimality=g_norm,
        active_mask=on_bound, nfev=nfev, njev=njev, status=termination_status)
