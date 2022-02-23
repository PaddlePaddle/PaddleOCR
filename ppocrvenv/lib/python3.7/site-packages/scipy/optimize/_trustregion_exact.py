"""Nearly exact trust-region optimization subproblem."""
import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
                          cho_solve)
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

__all__ = ['_minimize_trustregion_exact',
           'estimate_smallest_singular_value',
           'singular_leading_submatrix',
           'IterativeSubproblem']


def _minimize_trustregion_exact(fun, x0, args=(), jac=None, hess=None,
                                **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    a nearly exact trust-region algorithm.

    Options
    -------
    initial_tr_radius : float
        Initial trust-region radius.
    max_tr_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than ``gtol`` before successful
        termination.
    """

    if jac is None:
        raise ValueError('Jacobian is required for trust region '
                         'exact minimization.')
    if hess is None:
        raise ValueError('Hessian matrix is required for trust region '
                         'exact minimization.')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  subproblem=IterativeSubproblem,
                                  **trust_region_options)


def estimate_smallest_singular_value(U):
    """Given upper triangular matrix ``U`` estimate the smallest singular
    value and the correspondent right singular vector in O(n**2) operations.

    Parameters
    ----------
    U : ndarray
        Square upper triangular matrix.

    Returns
    -------
    s_min : float
        Estimated smallest singular value of the provided matrix.
    z_min : ndarray
        Estimatied right singular vector.

    Notes
    -----
    The procedure is based on [1]_ and is done in two steps. First, it finds
    a vector ``e`` with components selected from {+1, -1} such that the
    solution ``w`` from the system ``U.T w = e`` is as large as possible.
    Next it estimate ``U v = w``. The smallest singular value is close
    to ``norm(w)/norm(v)`` and the right singular vector is close
    to ``v/norm(v)``.

    The estimation will be better more ill-conditioned is the matrix.

    References
    ----------
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
           An estimate for the condition number of a matrix.  1979.
           SIAM Journal on Numerical Analysis, 16(2), 368-375.
    """

    U = np.atleast_2d(U)
    m, n = U.shape

    if m != n:
        raise ValueError("A square triangular matrix should be provided.")

    # A vector `e` with components selected from {+1, -1}
    # is selected so that the solution `w` to the system
    # `U.T w = e` is as large as possible. Implementation
    # based on algorithm 3.5.1, p. 142, from reference [2]
    # adapted for lower triangular matrix.

    p = np.zeros(n)
    w = np.empty(n)

    # Implemented according to:  Golub, G. H., Van Loan, C. F. (2013).
    # "Matrix computations". Forth Edition. JHU press. pp. 140-142.
    for k in range(n):
        wp = (1-p[k]) / U.T[k, k]
        wm = (-1-p[k]) / U.T[k, k]
        pp = p[k+1:] + U.T[k+1:, k]*wp
        pm = p[k+1:] + U.T[k+1:, k]*wm

        if abs(wp) + norm(pp, 1) >= abs(wm) + norm(pm, 1):
            w[k] = wp
            p[k+1:] = pp
        else:
            w[k] = wm
            p[k+1:] = pm

    # The system `U v = w` is solved using backward substitution.
    v = solve_triangular(U, w)

    v_norm = norm(v)
    w_norm = norm(w)

    # Smallest singular value
    s_min = w_norm / v_norm

    # Associated vector
    z_min = v / v_norm

    return s_min, z_min


def gershgorin_bounds(H):
    """
    Given a square matrix ``H`` compute upper
    and lower bounds for its eigenvalues (Gregoshgorin Bounds).
    Defined ref. [1].

    References
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """

    H_diag = np.diag(H)
    H_diag_abs = np.abs(H_diag)
    H_row_sums = np.sum(np.abs(H), axis=1)
    lb = np.min(H_diag + H_diag_abs - H_row_sums)
    ub = np.max(H_diag - H_diag_abs + H_row_sums)

    return lb, ub


def singular_leading_submatrix(A, U, k):
    """
    Compute term that makes the leading ``k`` by ``k``
    submatrix from ``A`` singular.

    Parameters
    ----------
    A : ndarray
        Symmetric matrix that is not positive definite.
    U : ndarray
        Upper triangular matrix resulting of an incomplete
        Cholesky decomposition of matrix ``A``.
    k : int
        Positive integer such that the leading k by k submatrix from
        `A` is the first non-positive definite leading submatrix.

    Returns
    -------
    delta : float
        Amount that should be added to the element (k, k) of the
        leading k by k submatrix of ``A`` to make it singular.
    v : ndarray
        A vector such that ``v.T B v = 0``. Where B is the matrix A after
        ``delta`` is added to its element (k, k).
    """

    # Compute delta
    delta = np.sum(U[:k-1, k-1]**2) - A[k-1, k-1]

    n = len(A)

    # Inicialize v
    v = np.zeros(n)
    v[k-1] = 1

    # Compute the remaining values of v by solving a triangular system.
    if k != 1:
        v[:k-1] = solve_triangular(U[:k-1, :k-1], -U[:k-1, k-1])

    return delta, v


class IterativeSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by nearly exact iterative method.

    Notes
    -----
    This subproblem solver was based on [1]_, [2]_ and [3]_,
    which implement similar algorithms. The algorithm is basically
    that of [1]_ but ideas from [2]_ and [3]_ were also used.

    References
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.
    """

    # UPDATE_COEFF appears in reference [1]_
    # in formula 7.3.14 (p. 190) named as "theta".
    # As recommended there it value is fixed in 0.01.
    UPDATE_COEFF = 0.01

    EPS = np.finfo(float).eps

    def __init__(self, x, fun, jac, hess, hessp=None,
                 k_easy=0.1, k_hard=0.2):

        super().__init__(x, fun, jac, hess)

        # When the trust-region shrinks in two consecutive
        # calculations (``tr_radius < previous_tr_radius``)
        # the lower bound ``lambda_lb`` may be reused,
        # facilitating  the convergence. To indicate no
        # previous value is known at first ``previous_tr_radius``
        # is set to -1  and ``lambda_lb`` to None.
        self.previous_tr_radius = -1
        self.lambda_lb = None

        self.niter = 0

        # ``k_easy`` and ``k_hard`` are parameters used
        # to determine the stop criteria to the iterative
        # subproblem solver. Take a look at pp. 194-197
        # from reference _[1] for a more detailed description.
        self.k_easy = k_easy
        self.k_hard = k_hard

        # Get Lapack function for cholesky decomposition.
        # The implemented SciPy wrapper does not return
        # the incomplete factorization needed by the method.
        self.cholesky, = get_lapack_funcs(('potrf',), (self.hess,))

        # Get info about Hessian
        self.dimension = len(self.hess)
        self.hess_gershgorin_lb,\
            self.hess_gershgorin_ub = gershgorin_bounds(self.hess)
        self.hess_inf = norm(self.hess, np.Inf)
        self.hess_fro = norm(self.hess, 'fro')

        # A constant such that for vectors smaler than that
        # backward substituition is not reliable. It was stabilished
        # based on Golub, G. H., Van Loan, C. F. (2013).
        # "Matrix computations". Forth Edition. JHU press., p.165.
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf

    def _initial_values(self, tr_radius):
        """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """

        # Upper bound for the damping factor
        lambda_ub = max(0, self.jac_mag/tr_radius + min(-self.hess_gershgorin_lb,
                                                        self.hess_fro,
                                                        self.hess_inf))

        # Lower bound for the damping factor
        lambda_lb = max(0, -min(self.hess.diagonal()),
                        self.jac_mag/tr_radius - min(self.hess_gershgorin_ub,
                                                     self.hess_fro,
                                                     self.hess_inf))

        # Improve bounds with previous info
        if tr_radius < self.previous_tr_radius:
            lambda_lb = max(self.lambda_lb, lambda_lb)

        # Initial guess for the damping factor
        if lambda_lb == 0:
            lambda_initial = 0
        else:
            lambda_initial = max(np.sqrt(lambda_lb * lambda_ub),
                                 lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

        return lambda_initial, lambda_lb, lambda_ub

    def solve(self, tr_radius):
        """Solve quadratic subproblem"""

        lambda_current, lambda_lb, lambda_ub = self._initial_values(tr_radius)
        n = self.dimension
        hits_boundary = True
        already_factorized = False
        self.niter = 0

        while True:

            # Compute Cholesky factorization
            if already_factorized:
                already_factorized = False
            else:
                H = self.hess+lambda_current*np.eye(n)
                U, info = self.cholesky(H, lower=False,
                                        overwrite_a=False,
                                        clean=True)

            self.niter += 1

            # Check if factorization succeeded
            if info == 0 and self.jac_mag > self.CLOSE_TO_ZERO:
                # Successful factorization

                # Solve `U.T U p = s`
                p = cho_solve((U, False), -self.jac)

                p_norm = norm(p)

                # Check for interior convergence
                if p_norm <= tr_radius and lambda_current == 0:
                    hits_boundary = False
                    break

                # Solve `U.T w = p`
                w = solve_triangular(U, p, trans='T')

                w_norm = norm(w)

                # Compute Newton step accordingly to
                # formula (4.44) p.87 from ref [2]_.
                delta_lambda = (p_norm/w_norm)**2 * (p_norm-tr_radius)/tr_radius
                lambda_new = lambda_current + delta_lambda

                if p_norm < tr_radius:  # Inside boundary
                    s_min, z_min = estimate_smallest_singular_value(U)

                    ta, tb = self.get_boundaries_intersections(p, z_min,
                                                               tr_radius)

                    # Choose `step_len` with the smallest magnitude.
                    # The reason for this choice is explained at
                    # ref [3]_, p. 6 (Immediately before the formula
                    # for `tau`).
                    step_len = min([ta, tb], key=abs)

                    # Compute the quadratic term  (p.T*H*p)
                    quadratic_term = np.dot(p, np.dot(H, p))

                    # Check stop criteria
                    relative_error = (step_len**2 * s_min**2) / (quadratic_term + lambda_current*tr_radius**2)
                    if relative_error <= self.k_hard:
                        p += step_len * z_min
                        break

                    # Update uncertanty bounds
                    lambda_ub = lambda_current
                    lambda_lb = max(lambda_lb, lambda_current - s_min**2)

                    # Compute Cholesky factorization
                    H = self.hess + lambda_new*np.eye(n)
                    c, info = self.cholesky(H, lower=False,
                                            overwrite_a=False,
                                            clean=True)

                    # Check if the factorization have succeeded
                    #
                    if info == 0:  # Successful factorization
                        # Update damping factor
                        lambda_current = lambda_new
                        already_factorized = True
                    else:  # Unsuccessful factorization
                        # Update uncertanty bounds
                        lambda_lb = max(lambda_lb, lambda_new)

                        # Update damping factor
                        lambda_current = max(np.sqrt(lambda_lb * lambda_ub),
                                             lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

                else:  # Outside boundary
                    # Check stop criteria
                    relative_error = abs(p_norm - tr_radius) / tr_radius
                    if relative_error <= self.k_easy:
                        break

                    # Update uncertanty bounds
                    lambda_lb = lambda_current

                    # Update damping factor
                    lambda_current = lambda_new

            elif info == 0 and self.jac_mag <= self.CLOSE_TO_ZERO:
                # jac_mag very close to zero

                # Check for interior convergence
                if lambda_current == 0:
                    p = np.zeros(n)
                    hits_boundary = False
                    break

                s_min, z_min = estimate_smallest_singular_value(U)
                step_len = tr_radius

                # Check stop criteria
                if step_len**2 * s_min**2 <= self.k_hard * lambda_current * tr_radius**2:
                    p = step_len * z_min
                    break

                # Update uncertanty bounds
                lambda_ub = lambda_current
                lambda_lb = max(lambda_lb, lambda_current - s_min**2)

                # Update damping factor
                lambda_current = max(np.sqrt(lambda_lb * lambda_ub),
                                     lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

            else:  # Unsuccessful factorization

                # Compute auxiliary terms
                delta, v = singular_leading_submatrix(H, U, info)
                v_norm = norm(v)

                # Update uncertanty interval
                lambda_lb = max(lambda_lb, lambda_current + delta/v_norm**2)

                # Update damping factor
                lambda_current = max(np.sqrt(lambda_lb * lambda_ub),
                                     lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))

        self.lambda_lb = lambda_lb
        self.lambda_current = lambda_current
        self.previous_tr_radius = tr_radius

        return p, hits_boundary
