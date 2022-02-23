from ._trustregion import (_minimize_trust_region)
from ._trlib import (get_trlib_quadratic_subproblem)

__all__ = ['_minimize_trust_krylov']

def _minimize_trust_krylov(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           inexact=True, **trust_region_options):
    """
    Minimization of a scalar function of one or more variables using
    a nearly exact trust-region algorithm that only requires matrix
    vector products with the hessian matrix.

    .. versionadded:: 1.0.0

    Options
    -------
    inexact : bool, optional
        Accuracy to solve subproblems. If True requires less nonlinear
        iterations, but more vector products.
    """

    if jac is None:
        raise ValueError('Jacobian is required for trust region ',
                         'exact minimization.')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Krylov trust-region minimization')

    # tol_rel specifies the termination tolerance relative to the initial
    # gradient norm in the Krylov subspace iteration.

    # - tol_rel_i specifies the tolerance for interior convergence.
    # - tol_rel_b specifies the tolerance for boundary convergence.
    #   in nonlinear programming applications it is not necessary to solve
    #   the boundary case as exact as the interior case.

    # - setting tol_rel_i=-2 leads to a forcing sequence in the Krylov
    #   subspace iteration leading to quadratic convergence if eventually
    #   the trust region stays inactive.
    # - setting tol_rel_b=-3 leads to a forcing sequence in the Krylov
    #   subspace iteration leading to superlinear convergence as long
    #   as the iterates hit the trust region boundary.

    # For details consult the documentation of trlib_krylov_min
    # in _trlib/trlib_krylov.h
    #
    # Optimality of this choice of parameters among a range of possibilities
    # has been tested on the unconstrained subset of the CUTEst library.

    if inexact:
        return _minimize_trust_region(fun, x0, args=args, jac=jac,
                                      hess=hess, hessp=hessp,
                                      subproblem=get_trlib_quadratic_subproblem(
                                          tol_rel_i=-2.0, tol_rel_b=-3.0,
                                          disp=trust_region_options.get('disp', False)
                                          ),
                                      **trust_region_options)
    else:
        return _minimize_trust_region(fun, x0, args=args, jac=jac,
                                      hess=hess, hessp=hessp,
                                      subproblem=get_trlib_quadratic_subproblem(
                                          tol_rel_i=1e-8, tol_rel_b=1e-6,
                                          disp=trust_region_options.get('disp', False)
                                          ),
                                      **trust_region_options)
