"""
A top-level linear programming interface. Currently this interface solves
linear programming problems via the Simplex and Interior-Point methods.

.. versionadded:: 0.15.0

Functions
---------
.. autosummary::
   :toctree: generated/

    linprog
    linprog_verbose_callback
    linprog_terse_callback

"""

import numpy as np

from .optimize import OptimizeResult, OptimizeWarning
from warnings import warn
from ._linprog_highs import _linprog_highs
from ._linprog_ip import _linprog_ip
from ._linprog_simplex import _linprog_simplex
from ._linprog_rs import _linprog_rs
from ._linprog_doc import (_linprog_highs_doc, _linprog_ip_doc,
                           _linprog_rs_doc, _linprog_simplex_doc,
                           _linprog_highs_ipm_doc, _linprog_highs_ds_doc)
from ._linprog_util import (
    _parse_linprog, _presolve, _get_Abc, _LPProblem, _autoscale,
    _postsolve, _check_result, _display_summary)
from copy import deepcopy

__all__ = ['linprog', 'linprog_verbose_callback', 'linprog_terse_callback']

__docformat__ = "restructuredtext en"

LINPROG_METHODS = ['simplex', 'revised simplex', 'interior-point', 'highs', 'highs-ds', 'highs-ipm']


def linprog_verbose_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces detailed output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    x = res['x']
    fun = res['fun']
    phase = res['phase']
    status = res['status']
    nit = res['nit']
    message = res['message']
    complete = res['complete']

    saved_printoptions = np.get_printoptions()
    np.set_printoptions(linewidth=500,
                        formatter={'float': lambda x: "{0: 12.4f}".format(x)})
    if status:
        print('--------- Simplex Early Exit -------\n'.format(nit))
        print('The simplex method exited early with status {0:d}'.format(status))
        print(message)
    elif complete:
        print('--------- Simplex Complete --------\n')
        print('Iterations required: {}'.format(nit))
    else:
        print('--------- Iteration {0:d}  ---------\n'.format(nit))

    if nit > 0:
        if phase == 1:
            print('Current Pseudo-Objective Value:')
        else:
            print('Current Objective Value:')
        print('f = ', fun)
        print()
        print('Current Solution Vector:')
        print('x = ', x)
        print()

    np.set_printoptions(**saved_printoptions)


def linprog_terse_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces brief output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``.
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    nit = res['nit']
    x = res['x']

    if nit == 0:
        print("Iter:   X:")
    print("{0: <5d}   ".format(nit), end="")
    print(x)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
            bounds=None, method='interior-point', callback=None,
            options=None, x0=None):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints.

    Linear programming solves problems of the following form:

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str, optional
        The algorithm used to solve the standard form problem.
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are supported.
    callback : callable, optional
        If a callback function is provided, it will be called at least once per
        iteration of the algorithm. The callback function must accept a single
        `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The current solution vector.
        fun : float
            The current value of the objective function ``c @ x``.
        success : bool
            ``True`` when the algorithm has completed successfully.
        slack : 1-D array
            The (nominally positive) values of the slack,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        phase : int
            The phase of the algorithm being executed.
        status : int
            An integer representing the status of the algorithm.

            ``0`` : Optimization proceeding nominally.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

            nit : int
                The current iteration number.
            message : str
                A string descriptor of the algorithm status.

        Callback functions are not currently supported by the HiGHS methods.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        options:

        maxiter : int
            Maximum number of iterations to perform.
            Default: see method-specific documentation.
        disp : bool
            Set to ``True`` to print convergence messages.
            Default: ``False``.
        presolve : bool
            Set to ``False`` to disable automatic presolve.
            Default: ``True``.

        All methods except the HiGHS solvers also accept:

        tol : float
            A tolerance which determines when a residual is "close enough" to
            zero to be considered exactly zero.
        autoscale : bool
            Set to ``True`` to automatically perform equilibration.
            Consider using this option if the numerical values in the
            constraints are separated by several orders of magnitude.
            Default: ``False``.
        rr : bool
            Set to ``False`` to disable automatic redundancy removal.
            Default: ``True``.
        rr_method : string
            Method used to identify and remove redundant rows from the
            equality constraint matrix after presolve. For problems with
            dense input, the available methods for redundancy removal are:

            "SVD":
                Repeatedly performs singular value decomposition on
                the matrix, detecting redundant rows based on nonzeros
                in the left singular vectors that correspond with
                zero singular values. May be fast when the matrix is
                nearly full rank.
            "pivot":
                Uses the algorithm presented in [5]_ to identify
                redundant rows.
            "ID":
                Uses a randomized interpolative decomposition.
                Identifies columns of the matrix transpose not used in
                a full-rank interpolative decomposition of the matrix.
            None:
                Uses "svd" if the matrix is nearly full rank, that is,
                the difference between the matrix rank and the number
                of rows is less than five. If not, uses "pivot". The
                behavior of this default is subject to change without
                prior notice.

            Default: None.
            For problems with sparse input, this option is ignored, and the
            pivot-based algorithm presented in [5]_ is used.

        For method-specific options, see
        :func:`show_options('linprog') <show_options>`.

    x0 : 1-D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        'revised simplex' method, and can only be used if `x0` represents a
        basic feasible solution.


    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The total number of iterations performed in all phases.
        message : str
            A string descriptor of the exit status of the algorithm.

    See Also
    --------
    show_options : Additional options accepted by the solvers.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    `'highs-ds'` and
    `'highs-ipm'` are interfaces to the
    HiGHS simplex and interior-point method solvers [13]_, respectively.
    `'highs'` chooses between
    the two automatically. These are the fastest linear
    programming solvers in SciPy, especially for large, sparse problems;
    which of these two is faster is problem-dependent.
    `'interior-point'` is the default
    as it was the fastest and most robust method before the recent
    addition of the HiGHS solvers.
    `'revised simplex'` is more
    accurate than interior-point for the problems it solves.
    `'simplex'` is the legacy method and is
    included for backwards compatibility and educational purposes.

    Method *highs-ds* is a wrapper of the C++ high performance dual
    revised simplex implementation (HSOL) [13]_, [14]_. Method *highs-ipm*
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver. Method *highs* chooses between the two automatically.
    For new code involving `linprog`, we recommend explicitly choosing one of
    these three method values.

    .. versionadded:: 1.6.0

    Method *interior-point* uses the primal-dual path following algorithm
    as outlined in [4]_. This algorithm supports sparse constraint matrices and
    is typically faster than the simplex methods, especially for large, sparse
    problems. Note, however, that the solution returned may be slightly less
    accurate than those of the simplex methods and will not, in general,
    correspond with a vertex of the polytope defined by the constraints.

    .. versionadded:: 1.0.0

    Method *revised simplex* uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    .. versionadded:: 1.3.0

    Method *simplex* uses a traditional, full-tableau implementation of
    Dantzig's simplex algorithm [1]_, [2]_ (*not* the
    Nelder-Mead simplex). This algorithm is included for backwards
    compatibility and educational purposes.

    .. versionadded:: 0.15.0

    Before applying *interior-point*, *revised simplex*, or *simplex*,
    a presolve procedure based on [8]_ attempts
    to identify trivial infeasibilities, trivial unboundedness, and potential
    problem simplifications. Specifically, it checks for:

    - rows of zeros in ``A_eq`` or ``A_ub``, representing trivial constraints;
    - columns of zeros in ``A_eq`` `and` ``A_ub``, representing unconstrained
      variables;
    - column singletons in ``A_eq``, representing fixed variables; and
    - column singletons in ``A_ub``, representing simple bounds.

    If presolve reveals that the problem is unbounded (e.g. an unconstrained
    and unbounded variable has negative cost) or infeasible (e.g., a row of
    zeros in ``A_eq`` corresponds with a nonzero in ``b_eq``), the solver
    terminates with the appropriate status code. Note that presolve terminates
    as soon as any sign of unboundedness is detected; consequently, a problem
    may be reported as unbounded when in reality the problem is infeasible
    (but infeasibility has not been detected yet). Therefore, if it is
    important to know whether the problem is actually infeasible, solve the
    problem again with option ``presolve=False``.

    If neither infeasibility nor unboundedness are detected in a single pass
    of the presolve, bounds are tightened where possible and fixed
    variables are removed from the problem. Then, linearly dependent rows
    of the ``A_eq`` matrix are removed, (unless they represent an
    infeasibility) to avoid numerical difficulties in the primary solve
    routine. Note that rows that are nearly linearly dependent (within a
    prescribed tolerance) may also be removed, which can change the optimal
    solution in rare cases. If this is a concern, eliminate redundancy from
    your problem formulation and run with option ``rr=False`` or
    ``presolve=False``.

    Several potential improvements can be made here: additional presolve
    checks outlined in [8]_ should be implemented, the presolve routine should
    be run multiple times (until no further simplifications can be made), and
    more of the efficiency improvements from [5]_ should be implemented in the
    redundancy removal routines.

    After presolve, the problem is transformed to standard form by converting
    the (tightened) simple bounds to upper bound constraints, introducing
    non-negative slack variables for inequality constraints, and expressing
    unbounded variables as the difference between two non-negative variables.
    Optionally, the problem is automatically scaled via equilibration [12]_.
    The selected algorithm solves the standard form problem, and a
    postprocessing routine converts the result to a solution to the original
    problem.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [5] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [7] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."
           Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at
           http://www.4er.org/CourseNotes/Book%20B/B-III.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point
            methods for large scale linear programming. HEC/Universite de
            Geneve, 1996.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
    .. [12] Tomlin, J. A. "On scaling linear programming problems."
            Mathematical Programming Study 4 (1975): 146-166.
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
            "HiGHS - high performance software for linear optimization."
            Accessed 4/16/2020 at https://www.maths.ed.ac.uk/hall/HiGHS/#guide
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
            simplex method." Mathematical Programming Computation, 10 (1),
            119-142, 2018. DOI: 10.1007/s12532-017-0130-5

    Examples
    --------
    Consider the following problem:

    .. math::

        \min_{x_0, x_1} \ -x_0 + 4x_1 & \\
        \mbox{such that} \ -3x_0 + x_1 & \leq 6,\\
        -x_0 - 2x_1 & \geq -4,\\
        x_1 & \geq -3.

    The problem is not presented in the form accepted by `linprog`. This is
    easily remedied by converting the "greater than" inequality
    constraint to a "less than" inequality constraint by
    multiplying both sides by a factor of :math:`-1`. Note also that the last
    constraint is really the simple bound :math:`-3 \leq x_1 \leq \infty`.
    Finally, since there are no bounds on :math:`x_0`, we must explicitly
    specify the bounds :math:`-\infty \leq x_0 \leq \infty`, as the
    default is for variables to be non-negative. After collecting coeffecients
    into arrays and tuples, the input for this problem is:

    >>> c = [-1, 4]
    >>> A = [[-3, 1], [1, 2]]
    >>> b = [6, 4]
    >>> x0_bounds = (None, None)
    >>> x1_bounds = (-3, None)
    >>> from scipy.optimize import linprog
    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

    Note that the default method for `linprog` is 'interior-point', which is
    approximate by nature.

    >>> print(res)
         con: array([], dtype=float64)
         fun: -21.99999984082494 # may vary
     message: 'Optimization terminated successfully.'
         nit: 6 # may vary
       slack: array([3.89999997e+01, 8.46872439e-08] # may vary
      status: 0
     success: True
           x: array([ 9.99999989, -2.99999999]) # may vary

    If you need greater accuracy, try 'revised simplex'.

    >>> res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='revised simplex')
    >>> print(res)
         con: array([], dtype=float64)
         fun: -22.0 # may vary
     message: 'Optimization terminated successfully.'
         nit: 1 # may vary
       slack: array([39.,  0.]) # may vary
      status: 0
     success: True
           x: array([10., -3.]) # may vary

    """

    meth = method.lower()
    methods = {"simplex", "revised simplex", "interior-point",
               "highs", "highs-ds", "highs-ipm"}

    if meth not in methods:
        raise ValueError(f"Unknown solver '{method}'")

    if x0 is not None and meth != "revised simplex":
        warning_message = "x0 is used only when method is 'revised simplex'. "
        warn(warning_message, OptimizeWarning)

    lp = _LPProblem(c, A_ub, b_ub, A_eq, b_eq, bounds, x0)
    lp, solver_options = _parse_linprog(lp, options, meth)
    tol = solver_options.get('tol', 1e-9)

    # Give unmodified problem to HiGHS
    if meth.startswith('highs'):
        if callback is not None:
            raise NotImplementedError("HiGHS solvers do not support the "
                                      "callback interface.")
        highs_solvers = {'highs-ipm': 'ipm', 'highs-ds': 'simplex',
                         'highs': None}

        sol = _linprog_highs(lp, solver=highs_solvers[meth],
                             **solver_options)
        sol['status'], sol['message'] = (
            _check_result(sol['x'], sol['fun'], sol['status'], sol['slack'],
                          sol['con'], lp.bounds, tol, sol['message']))
        sol['success'] = sol['status'] == 0
        return OptimizeResult(sol)

    iteration = 0
    complete = False  # will become True if solved in presolve
    undo = []

    # Keep the original arrays to calculate slack/residuals for original
    # problem.
    lp_o = deepcopy(lp)

    # Solve trivial problem, eliminate variables, tighten bounds, etc.
    rr_method = solver_options.pop('rr_method', None)  # need to pop these;
    rr = solver_options.pop('rr', True)  # they're not passed to methods
    c0 = 0  # we might get a constant term in the objective
    if solver_options.pop('presolve', True):
        (lp, c0, x, undo, complete, status, message) = _presolve(lp, rr,
                                                                 rr_method,
                                                                 tol)

    C, b_scale = 1, 1  # for trivial unscaling if autoscale is not used
    postsolve_args = (lp_o._replace(bounds=lp.bounds), undo, C, b_scale)

    if not complete:
        A, b, c, c0, x0 = _get_Abc(lp, c0)
        if solver_options.pop('autoscale', False):
            A, b, c, x0, C, b_scale = _autoscale(A, b, c, x0)
            postsolve_args = postsolve_args[:-2] + (C, b_scale)

        if meth == 'simplex':
            x, status, message, iteration = _linprog_simplex(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'interior-point':
            x, status, message, iteration = _linprog_ip(
                c, c0=c0, A=A, b=b, callback=callback,
                postsolve_args=postsolve_args, **solver_options)
        elif meth == 'revised simplex':
            x, status, message, iteration = _linprog_rs(
                c, c0=c0, A=A, b=b, x0=x0, callback=callback,
                postsolve_args=postsolve_args, **solver_options)

    # Eliminate artificial variables, re-introduce presolved variables, etc.
    disp = solver_options.get('disp', False)

    x, fun, slack, con = _postsolve(x, postsolve_args, complete)

    status, message = _check_result(x, fun, status, slack, con, lp_o.bounds, tol, message)

    if disp:
        _display_summary(message, status, fun, iteration)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        'success': status == 0}

    return OptimizeResult(sol)
