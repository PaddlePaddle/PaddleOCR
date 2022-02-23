"""HiGHS Linear Optimization Methods

Interface to HiGHS linear optimization software.
https://www.maths.ed.ac.uk/hall/HiGHS/

.. versionadded:: 1.5.0

References
----------
.. [1] Q. Huangfu and J.A.J. Hall. "Parallelizing the dual revised simplex
           method." Mathematical Programming Computation, 10 (1), 119-142,
           2018. DOI: 10.1007/s12532-017-0130-5

"""

import inspect
import numpy as np
from .optimize import _check_unknown_options, OptimizeWarning, OptimizeResult
from warnings import warn
from ._highs._highs_wrapper import _highs_wrapper
from ._highs._highs_constants import (
    CONST_I_INF,
    CONST_INF,
    MESSAGE_LEVEL_MINIMAL,

    MODEL_STATUS_NOTSET,
    MODEL_STATUS_LOAD_ERROR,
    MODEL_STATUS_MODEL_ERROR,
    MODEL_STATUS_PRESOLVE_ERROR,
    MODEL_STATUS_SOLVE_ERROR,
    MODEL_STATUS_POSTSOLVE_ERROR,
    MODEL_STATUS_MODEL_EMPTY,
    MODEL_STATUS_PRIMAL_INFEASIBLE,
    MODEL_STATUS_PRIMAL_UNBOUNDED,
    MODEL_STATUS_OPTIMAL,
    MODEL_STATUS_REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND
    as MODEL_STATUS_RDOVUB,
    MODEL_STATUS_REACHED_TIME_LIMIT,
    MODEL_STATUS_REACHED_ITERATION_LIMIT,
    MODEL_STATUS_PRIMAL_DUAL_INFEASIBLE,
    MODEL_STATUS_DUAL_INFEASIBLE,

    HIGHS_SIMPLEX_STRATEGY_CHOOSE,
    HIGHS_SIMPLEX_STRATEGY_DUAL,

    HIGHS_SIMPLEX_CRASH_STRATEGY_OFF,

    HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_CHOOSE,
    HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_DANTZIG,
    HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_DEVEX,
    HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE,
)
from scipy.sparse import csc_matrix, vstack, issparse


def _replace_inf(x):
    # Replace `np.inf` with CONST_INF
    infs = np.isinf(x)
    x[infs] = np.sign(x[infs])*CONST_INF
    return x


def _convert_to_highs_enum(option, option_str, choices):
    # If option is in the choices we can look it up, if not use
    # the default value taken from function signature and warn:
    try:
        return choices[option.lower()]
    except AttributeError:
        return choices[option]
    except KeyError:
        sig = inspect.signature(_linprog_highs)
        default_str = sig.parameters[option_str].default
        warn(f"Option {option_str} is {option}, but only values in "
             f"{set(choices.keys())} are allowed. Using default: "
             f"{default_str}.",
             OptimizeWarning, stacklevel=3)
        return choices[default_str]


def _linprog_highs(lp, solver, time_limit=None, presolve=True,
                   disp=False, maxiter=None,
                   dual_feasibility_tolerance=None,
                   primal_feasibility_tolerance=None,
                   ipm_optimality_tolerance=None,
                   simplex_dual_edge_weight_strategy=None,
                   **unknown_options):
    r"""
    Solve the following linear programming problem using one of the HiGHS
    solvers:

    User-facing documentation is in _linprog_doc.py.

    Parameters
    ----------
    lp :  _LPProblem
        A ``scipy.optimize._linprog_util._LPProblem`` ``namedtuple``.
    solver : "ipm" or "simplex" or None
        Which HiGHS solver to use.  If ``None``, "simplex" will be used.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase. For
        ``solver='ipm'``, this does not include the number of crossover
        iterations.  Default is the largest possible value for an ``int``
        on the platform.
    disp : bool
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration; default ``False``.
    time_limit : float
        The maximum time in seconds allotted to solve the problem; default is
        the largest possible value for a ``double`` on the platform.
    presolve : bool
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if presolve is
        to be disabled.
    dual_feasibility_tolerance : double
        Dual feasibility tolerance.  Default is 1e-07.
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance when ``solver='ipm'``.
    primal_feasibility_tolerance : double
        Primal feasibility tolerance.  Default is 1e-07.
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance when ``solver='ipm'``.
    ipm_optimality_tolerance : double
        Optimality tolerance for ``solver='ipm'``.  Default is 1e-08.
        Minimum possible value is 1e-12 and must be smaller than the largest
        possible value for a ``double`` on the platform.
    simplex_dual_edge_weight_strategy : str (default: None)
        Strategy for simplex dual edge weights. The default, ``None``,
        automatically selects one of the following.

        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.

        ``'devex'`` uses the strategy described in [15]_.

        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.

        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.

        Curently, using ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.

    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing all
        unused options.

    Returns
    -------
    sol : dict
        A dictionary consisting of the fields:

            x : 1D array
                The values of the decision variables that minimizes the
                objective function while satisfying the constraints.
            fun : float
                The optimal value of the objective function ``c @ x``.
            slack : 1D array
                The (nominally positive) values of the slack,
                ``b_ub - A_ub @ x``.
            con : 1D array
                The (nominally zero) residuals of the equality constraints,
                ``b_eq - A_eq @ x``.
            success : bool
                ``True`` when the algorithm succeeds in finding an optimal
                solution.
            status : int
                An integer representing the exit status of the algorithm.

                ``0`` : Optimization terminated successfully.

                ``1`` : Iteration or time limit reached.

                ``2`` : Problem appears to be infeasible.

                ``3`` : Problem appears to be unbounded.

                ``4`` : The HiGHS solver ran into a problem.

            message : str
                A string descriptor of the exit status of the algorithm.
            nit : int
                The total number of iterations performed.
                For ``solver='simplex'``, this includes iterations in all
                phases. For ``solver='ipm'``, this does not include
                crossover iterations.
            crossover_nit : int
                The number of primal/dual pushes performed during the
                crossover routine for ``solver='ipm'``.  This is ``0``
                for ``solver='simplex'``.
            ineqlin : OptimizeResult
                Solution and sensitivity information corresponding to the
                inequality constraints, `b_ub`. A dictionary consisting of the
                fields:

                residual : np.ndnarray
                    The (nominally positive) values of the slack variables,
                    ``b_ub - A_ub @ x``.  This quantity is also commonly
                    referred to as "slack".

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the right-hand side of the
                    inequality constraints, `b_ub`.

            eqlin : OptimizeResult
                Solution and sensitivity information corresponding to the
                equality constraints, `b_eq`.  A dictionary consisting of the
                fields:

                residual : np.ndarray
                    The (nominally zero) residuals of the equality constraints,
                    ``b_eq - A_eq @ x``.

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the right-hand side of the
                    equality constraints, `b_eq`.

            lower, upper : OptimizeResult
                Solution and sensitivity information corresponding to the
                lower and upper bounds on decision variables, `bounds`.

                residual : np.ndarray
                    The (nominally positive) values of the quantity
                    ``x - lb`` (lower) or ``ub - x`` (upper).

                marginals : np.ndarray
                    The sensitivity (partial derivative) of the objective
                    function with respect to the lower and upper
                    `bounds`.

    Notes
    -----
    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """

    _check_unknown_options(unknown_options)

    # Map options to HiGHS enum values
    simplex_dual_edge_weight_strategy_enum = _convert_to_highs_enum(
        simplex_dual_edge_weight_strategy,
        'simplex_dual_edge_weight_strategy',
        choices={'dantzig': HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_DANTZIG,
                 'devex': HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_DEVEX,
                 'steepest-devex': HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_CHOOSE,
                 'steepest':
                 HIGHS_SIMPLEX_DUAL_EDGE_WEIGHT_STRATEGY_STEEPEST_EDGE,
                 None: None})

    statuses = {
        MODEL_STATUS_NOTSET: (
            4,
            'HiGHS Status Code 0: HighsModelStatusNOTSET',
        ),
        MODEL_STATUS_LOAD_ERROR: (
            4,
            'HiGHS Status Code 1: HighsModelStatusLOAD_ERROR',
        ),
        MODEL_STATUS_MODEL_ERROR: (
            2,
            'HiGHS Status Code 2: HighsModelStatusMODEL_ERROR',
        ),
        MODEL_STATUS_PRESOLVE_ERROR: (
            4,
            'HiGHS Status Code 4: HighsModelStatusPRESOLVE_ERROR',
        ),
        MODEL_STATUS_SOLVE_ERROR: (
            4,
            'HiGHS Status Code 5: HighsModelStatusSOLVE_ERROR',
        ),
        MODEL_STATUS_POSTSOLVE_ERROR: (
            4,
            'HiGHS Status Code 6: HighsModelStatusPOSTSOLVE_ERROR',
        ),
        MODEL_STATUS_MODEL_EMPTY: (
            4,
            'HiGHS Status Code 3: HighsModelStatusMODEL_EMPTY',
        ),
        MODEL_STATUS_RDOVUB: (
            4,
            'HiGHS Status Code 10: '
            'HighsModelStatusREACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND',
        ),
        MODEL_STATUS_PRIMAL_INFEASIBLE: (
            2,
            "The problem is infeasible.",
        ),
        MODEL_STATUS_PRIMAL_UNBOUNDED: (
            3,
            "The problem is unbounded.",
        ),
        MODEL_STATUS_OPTIMAL: (
            0,
            "Optimization terminated successfully.",
        ),
        MODEL_STATUS_REACHED_TIME_LIMIT: (
            1,
            "Time limit reached.",
        ),
        MODEL_STATUS_REACHED_ITERATION_LIMIT: (
            1,
            "Iteration limit reached.",
        ),
        MODEL_STATUS_PRIMAL_DUAL_INFEASIBLE: (
            2,
            "The problem is primal/dual infeasible.",
        ),
        MODEL_STATUS_DUAL_INFEASIBLE: (
            2,
            "The problem is dual infeasible.",
        ),
    }

    c, A_ub, b_ub, A_eq, b_eq, bounds, x0 = lp

    lb, ub = bounds.T.copy()  # separate bounds, copy->C-cntgs
    # highs_wrapper solves LHS <= A*x <= RHS, not equality constraints
    lhs_ub = -np.ones_like(b_ub)*np.inf  # LHS of UB constraints is -inf
    rhs_ub = b_ub  # RHS of UB constraints is b_ub
    lhs_eq = b_eq  # Equality constaint is inequality
    rhs_eq = b_eq  # constraint with LHS=RHS
    lhs = np.concatenate((lhs_ub, lhs_eq))
    rhs = np.concatenate((rhs_ub, rhs_eq))

    if issparse(A_ub) or issparse(A_eq):
        A = vstack((A_ub, A_eq))
    else:
        A = np.vstack((A_ub, A_eq))
    A = csc_matrix(A)

    options = {
        'presolve': presolve,
        'sense': 1,  # minimization
        'solver': solver,
        'time_limit': time_limit,
        'message_level': MESSAGE_LEVEL_MINIMAL * disp,
        'dual_feasibility_tolerance': dual_feasibility_tolerance,
        'ipm_optimality_tolerance': ipm_optimality_tolerance,
        'primal_feasibility_tolerance': primal_feasibility_tolerance,
        'simplex_dual_edge_weight_strategy':
            simplex_dual_edge_weight_strategy_enum,
        'simplex_strategy': HIGHS_SIMPLEX_STRATEGY_DUAL,
        'simplex_crash_strategy': HIGHS_SIMPLEX_CRASH_STRATEGY_OFF,
        'ipm_iteration_limit': maxiter,
        'simplex_iteration_limit': maxiter,
    }

    # np.inf doesn't work; use very large constant
    rhs = _replace_inf(rhs)
    lhs = _replace_inf(lhs)
    lb = _replace_inf(lb)
    ub = _replace_inf(ub)

    res = _highs_wrapper(c, A.indptr, A.indices, A.data, lhs, rhs,
                         lb, ub, options)

    # HiGHS represents constraints as lhs/rhs, so
    # Ax + s = b => Ax = b - s
    # and we need to split up s by A_ub and A_eq
    if 'slack' in res:
        slack = res['slack']
        con = np.array(slack[len(b_ub):])
        slack = np.array(slack[:len(b_ub)])
    else:
        slack, con = None, None

    # lagrange multipliers for equalities/inequalities and upper/lower bounds
    if 'lambda' in res:
        lamda = res['lambda']
        marg_ineqlin = np.array(lamda[:len(b_ub)])
        marg_eqlin = np.array(lamda[len(b_ub):])
        marg_upper = res['marg_bnds'][1, :]
        marg_lower = res['marg_bnds'][0, :]
    else:
        marg_ineqlin, marg_eqlin = None, None
        marg_upper, marg_lower = None, None

    # this needs to be updated if we start choosing the solver intelligently
    solvers = {"ipm": "highs-ipm", "simplex": "highs-ds", None: "highs-ds"}

    # HiGHS will report OPTIMAL if the scaled model is solved to optimality
    # even if the unscaled original model is infeasible;
    # Catch that case here and provide a more useful message
    if ((res['status'] == MODEL_STATUS_OPTIMAL) and
            (res['unscaled_status'] != res['status'])):
        _unscaled_status, unscaled_message = statuses[res["unscaled_status"]]
        status, message = 4, ('An optimal solution to the scaled model was '
                              f'found but was {unscaled_message} in the '
                              'unscaled model. For more information run with '
                              'the option `disp: True`.')
    else:
        status, message = statuses[res['status']]

    x = np.array(res['x']) if 'x' in res else None
    sol = {'x': x,
           'slack': slack,
           'con': con,
           'ineqlin': OptimizeResult({
               'residual': slack,
               'marginals': marg_ineqlin,
           }),
           'eqlin': OptimizeResult({
               'residual': con,
               'marginals': marg_eqlin,
           }),
           'lower': OptimizeResult({
               'residual': None if x is None else x - lb,
               'marginals': marg_lower,
           }),
           'upper': OptimizeResult({
               'residual': None if x is None else ub - x,
               'marginals': marg_upper
            }),
           'fun': res.get('fun'),
           'status': status,
           'success': res['status'] == MODEL_STATUS_OPTIMAL,
           'message': message,
           'nit': res.get('simplex_nit', 0) or res.get('ipm_nit', 0),
           'crossover_nit': res.get('crossover_nit'),
           }

    return sol
