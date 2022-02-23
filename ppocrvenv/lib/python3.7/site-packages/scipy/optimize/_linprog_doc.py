# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:49:17 2020

@author: matth
"""


def _linprog_highs_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                       bounds=None, method='highs', callback=None,
                       maxiter=None, disp=False, presolve=True,
                       time_limit=None,
                       dual_feasibility_tolerance=None,
                       primal_feasibility_tolerance=None,
                       ipm_optimality_tolerance=None,
                       simplex_dual_edge_weight_strategy=None,
                       **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using one of the HiGHS solvers.

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
    method : str

        This is the method-specific documentation for 'highs', which chooses
        automatically between
        :ref:`'highs-ds' <optimize.linprog-highs-ds>` and
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase.
        For :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`, this does not
        include the number of crossover iterations. Default is the largest
        possible value for an ``int`` on the platform.
    disp : bool (default: ``False``)
        Set to ``True`` if indicators of optimization status are to be
        printed to the console during optimization.
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    dual_feasibility_tolerance : double (default: 1e-07)
        Dual feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    primal_feasibility_tolerance : double (default: 1e-07)
        Primal feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    ipm_optimality_tolerance : double (default: ``1e-08``)
        Optimality tolerance for
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        Minimum allowable value is 1e-12.
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

        Curently, ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

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
            For the HiGHS simplex method, this includes iterations in all
            phases. For the HiGHS interior-point method, this does not include
            crossover iterations.
        crossover_nit : int
            The number of primal/dual pushes performed during the
            crossover routine for the HiGHS interior-point method.
            This is ``0`` for the HiGHS simplex method.
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

    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>` is a wrapper
    of the C++ high performance dual revised simplex implementation (HSOL)
    [13]_, [14]_. Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver. Method :ref:`'highs' <optimize.linprog-highs>` chooses
    between the two automatically. For new code involving `linprog`, we
    recommend explicitly choosing one of these three method values instead of
    :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
    :ref:`'simplex' <optimize.linprog-simplex>` (legacy).

    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           Accessed 4/16/2020 at https://www.maths.ed.ac.uk/hall/HiGHS/#guide
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """
    pass


def _linprog_highs_ds_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                          bounds=None, method='highs-ds', callback=None,
                          maxiter=None, disp=False, presolve=True,
                          time_limit=None,
                          dual_feasibility_tolerance=None,
                          primal_feasibility_tolerance=None,
                          simplex_dual_edge_weight_strategy=None,
                          **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the HiGHS dual simplex solver.

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
    method : str

        This is the method-specific documentation for 'highs-ds'.
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase.
        Default is the largest possible value for an ``int`` on the platform.
    disp : bool (default: ``False``)
        Set to ``True`` if indicators of optimization status are to be
        printed to the console during optimization.
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    dual_feasibility_tolerance : double (default: 1e-07)
        Dual feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
    primal_feasibility_tolerance : double (default: 1e-07)
        Primal feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
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

        Curently, ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

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
            The total number of iterations performed. This includes iterations
            in all phases.
        crossover_nit : int
            This is always ``0`` for the HiGHS simplex method.
            For the HiGHS interior-point method, this is the number of
            primal/dual pushes performed during the crossover routine.
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

    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>` is a wrapper
    of the C++ high performance dual revised simplex implementation (HSOL)
    [13]_, [14]_. Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver. Method :ref:`'highs' <optimize.linprog-highs>` chooses
    between the two automatically. For new code involving `linprog`, we
    recommend explicitly choosing one of these three method values instead of
    :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
    :ref:`'simplex' <optimize.linprog-simplex>` (legacy).

    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           Accessed 4/16/2020 at https://www.maths.ed.ac.uk/hall/HiGHS/#guide
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """
    pass


def _linprog_highs_ipm_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                           bounds=None, method='highs-ipm', callback=None,
                           maxiter=None, disp=False, presolve=True,
                           time_limit=None,
                           dual_feasibility_tolerance=None,
                           primal_feasibility_tolerance=None,
                           ipm_optimality_tolerance=None,
                           **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the HiGHS interior point solver.

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
    method : str

        This is the method-specific documentation for 'highs-ipm'.
        :ref:`'highs-ipm' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase.
        For :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`, this does not
        include the number of crossover iterations. Default is the largest
        possible value for an ``int`` on the platform.
    disp : bool (default: ``False``)
        Set to ``True`` if indicators of optimization status are to be
        printed to the console during optimization.
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    dual_feasibility_tolerance : double (default: 1e-07)
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    primal_feasibility_tolerance : double (default: 1e-07)
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    ipm_optimality_tolerance : double (default: ``1e-08``)
        Optimality tolerance for
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        Minimum allowable value is 1e-12.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

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
            For the HiGHS interior-point method, this does not include
            crossover iterations.
        crossover_nit : int
            The number of primal/dual pushes performed during the
            crossover routine for the HiGHS interior-point method.
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

    Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
    is a wrapper of a C++ implementation of an **i**\ nterior-\ **p**\ oint
    **m**\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver.
    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>` is a wrapper
    of the C++ high performance dual revised simplex implementation (HSOL)
    [13]_, [14]_. Method :ref:`'highs' <optimize.linprog-highs>` chooses
    between the two automatically. For new code involving `linprog`, we
    recommend explicitly choosing one of these three method values instead of
    :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
    :ref:`'simplex' <optimize.linprog-simplex>` (legacy).

    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           Accessed 4/16/2020 at https://www.maths.ed.ac.uk/hall/HiGHS/#guide
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    """
    pass


def _linprog_ip_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                    bounds=None, method='interior-point', callback=None,
                    maxiter=1000, disp=False, presolve=True,
                    tol=1e-8, autoscale=False, rr=True,
                    alpha0=.99995, beta=0.1, sparse=False,
                    lstsq=False, sym_pos=True, cholesky=True, pc=True,
                    ip=False, permc_spec='MMD_AT_PLUS_A', **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the interior-point method of
    [4]_.

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
    method : str
        This is the method-specific documentation for 'interior-point'.
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.
    callback : callable, optional
        Callback function to be executed once per iteration.

    Options
    -------
    maxiter : int (default: 1000)
        The maximum number of iterations of the algorithm.
    disp : bool (default: False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    presolve : bool (default: True)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    tol : float (default: 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [4]_ Section 4.5.
    autoscale : bool (default: False)
        Set to ``True`` to automatically perform equilibration.
        Consider using this option if the numerical values in the
        constraints are separated by several orders of magnitude.
    rr : bool (default: True)
        Set to ``False`` to disable automatic redundancy removal.
    alpha0 : float (default: 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see :math:`\beta_{3}` of [4]_ Table 8.1.
    beta : float (default: 0.1)
        The desired reduction of the path parameter :math:`\mu` (see [6]_)
        when Mehrota's predictor-corrector is not in use (uncommon).
    sparse : bool (default: False)
        Set to ``True`` if the problem is to be treated as sparse after
        presolve. If either ``A_eq`` or ``A_ub`` is a sparse matrix,
        this option will automatically be set ``True``, and the problem
        will be treated as sparse even during presolve. If your constraint
        matrices contain mostly zeros and the problem is not very small (less
        than about 100 constraints or variables), consider setting ``True``
        or providing ``A_eq`` and ``A_ub`` as sparse matrices.
    lstsq : bool (default: ``False``)
        Set to ``True`` if the problem is expected to be very poorly
        conditioned. This should always be left ``False`` unless severe
        numerical difficulties are encountered. Leave this at the default
        unless you receive a warning message suggesting otherwise.
    sym_pos : bool (default: True)
        Leave ``True`` if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix
        (almost always). Leave this at the default unless you receive
        a warning message suggesting otherwise.
    cholesky : bool (default: True)
        Set to ``True`` if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is typically faster for problems
        that are numerically well-behaved.
    pc : bool (default: True)
        Leave ``True`` if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : bool (default: False)
        Set to ``True`` if the improved initial point suggestion due to [4]_
        Section 4.3 is desired. Whether this is beneficial or not
        depends on the problem.
    permc_spec : str (default: 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``, and no SuiteSparse.)
        A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

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

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed in all phases.


    Notes
    -----
    This method implements the algorithm outlined in [4]_ with ideas from [8]_
    and a structure inspired by the simpler methods of [6]_.

    The primal-dual path following method begins with initial 'guesses' of
    the primal and dual variables of the standard form problem and iteratively
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the
    problem with a gradually reduced logarithmic barrier term added to the
    objective. This particular implementation uses a homogeneous self-dual
    formulation, which provides certificates of infeasibility or unboundedness
    where applicable.

    The default initial point for the primal and dual variables is that
    defined in [4]_ Section 4.4 Equation 8.22. Optionally (by setting initial
    point option ``ip=True``), an alternate (potentially improved) starting
    point can be calculated according to the additional recommendations of
    [4]_ Section 4.4.

    A search direction is calculated using the predictor-corrector method
    (single correction) proposed by Mehrota and detailed in [4]_ Section 4.1.
    (A potential improvement would be to implement the method of multiple
    corrections described in [4]_ Section 4.2.) In practice, this is
    accomplished by solving the normal equations, [4]_ Section 5.1 Equations
    8.31 and 8.32, derived from the Newton equations [4]_ Section 5 Equations
    8.25 (compare to [4]_ Section 4 Equations 8.6-8.8). The advantage of
    solving the normal equations rather than 8.25 directly is that the
    matrices involved are symmetric positive definite, so Cholesky
    decomposition can be used rather than the more expensive LU factorization.

    With default options, the solver used to perform the factorization depends
    on third-party software availability and the conditioning of the problem.

    For dense problems, solvers are tried in the following order:

    1. ``scipy.linalg.cho_factor``

    2. ``scipy.linalg.solve`` with option ``sym_pos=True``

    3. ``scipy.linalg.solve`` with option ``sym_pos=False``

    4. ``scipy.linalg.lstsq``

    For sparse problems:

    1. ``sksparse.cholmod.cholesky`` (if scikit-sparse and SuiteSparse are
       installed)

    2. ``scipy.sparse.linalg.factorized`` (if scikit-umfpack and SuiteSparse
       are installed)

    3. ``scipy.sparse.linalg.splu`` (which uses SuperLU distributed with SciPy)

    4. ``scipy.sparse.linalg.lsqr``

    If the solver fails for any reason, successively more robust (but slower)
    solvers are attempted in the order indicated. Attempting, failing, and
    re-starting factorization can be time consuming, so if the problem is
    numerically challenging, options can be set to  bypass solvers that are
    failing. Setting ``cholesky=False`` skips to solver 2,
    ``sym_pos=False`` skips to solver 3, and ``lstsq=True`` skips
    to solver 4 for both sparse and dense problems.

    Potential improvements for combatting issues associated with dense
    columns in otherwise sparse problems are outlined in [4]_ Section 5.3 and
    [10]_ Section 4.1-4.2; the latter also discusses the alleviation of
    accuracy issues associated with the substitution approach to free
    variables.

    After calculating the search direction, the maximum possible step size
    that does not activate the non-negativity constraints is calculated, and
    the smaller of this step size and unity is applied (as in [4]_ Section
    4.1.) [4]_ Section 4.3 suggests improvements for choosing the step size.

    The new point is tested according to the termination conditions of [4]_
    Section 4.5. The same tolerance, which can be set using the ``tol`` option,
    is used for all checks. (A potential improvement would be to expose
    the different tolerances to be set independently.) If optimality,
    unboundedness, or infeasibility is detected, the solve procedure
    terminates; otherwise it repeats.

    Whereas the top level ``linprog`` module expects a problem of form:

    Minimize::

        c @ x

    Subject to::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub

    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``. The problem
    is automatically converted to the form:

    Minimize::

        c @ x

    Subject to::

        A @ x == b
            x >= 0

    for solution. That is, the original problem contains equality, upper-bound
    and variable constraints whereas the method specific solver requires
    equality constraints and variable non-negativity. ``linprog`` converts the
    original problem to standard form by converting the simple bounds to upper
    bound constraints, introducing non-negative slack variables for inequality
    constraints, and expressing unbounded variables as the difference between
    two non-negative variables. The problem is converted back to the original
    form before results are reported.

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [6] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [8] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [10] Andersen, Erling D., et al. Implementation of interior point
            methods for large scale linear programming. HEC/Universite de
            Geneve, 1996.
    """
    pass


def _linprog_rs_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                    bounds=None, method='interior-point', callback=None,
                    x0=None, maxiter=5000, disp=False, presolve=True,
                    tol=1e-12, autoscale=False, rr=True, maxupdate=10,
                    mast=False, pivot="mrc", **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the revised simplex method.

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
    method : str
        This is the method-specific documentation for 'revised simplex'.
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        and :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.
    callback : callable, optional
        Callback function to be executed once per iteration.
    x0 : 1-D array, optional
        Guess values of the decision variables, which will be refined by
        the optimization algorithm. This argument is currently used only by the
        'revised simplex' method, and can only be used if `x0` represents a
        basic feasible solution.

    Options
    -------
    maxiter : int (default: 5000)
       The maximum number of iterations to perform in either phase.
    disp : bool (default: False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    presolve : bool (default: True)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    tol : float (default: 1e-12)
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    autoscale : bool (default: False)
        Set to ``True`` to automatically perform equilibration.
        Consider using this option if the numerical values in the
        constraints are separated by several orders of magnitude.
    rr : bool (default: True)
        Set to ``False`` to disable automatic redundancy removal.
    maxupdate : int (default: 10)
        The maximum number of updates performed on the LU factorization.
        After this many updates is reached, the basis matrix is factorized
        from scratch.
    mast : bool (default: False)
        Minimize Amortized Solve Time. If enabled, the average time to solve
        a linear system using the basis factorization is measured. Typically,
        the average solve time will decrease with each successive solve after
        initial factorization, as factorization takes much more time than the
        solve operation (and updates). Eventually, however, the updated
        factorization becomes sufficiently complex that the average solve time
        begins to increase. When this is detected, the basis is refactorized
        from scratch. Enable this option to maximize speed at the risk of
        nondeterministic behavior. Ignored if ``maxupdate`` is 0.
    pivot : "mrc" or "bland" (default: "mrc")
        Pivot rule: Minimum Reduced Cost ("mrc") or Bland's rule ("bland").
        Choose Bland's rule if iteration limit is reached and cycling is
        suspected.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

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

            ``5`` : Problem has no constraints; turn presolve on.

            ``6`` : Invalid guess provided.

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed in all phases.


    Notes
    -----
    Method *revised simplex* uses the revised simplex method as described in
    [9]_, except that a factorization [11]_ of the basis matrix, rather than
    its inverse, is efficiently maintained and used to solve the linear systems
    at each iteration of the algorithm.

    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [11] Bartels, Richard H. "A stabilization of the simplex method."
            Journal in  Numerische Mathematik 16.5 (1971): 414-434.
    """
    pass


def _linprog_simplex_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                         bounds=None, method='interior-point', callback=None,
                         maxiter=5000, disp=False, presolve=True,
                         tol=1e-12, autoscale=False, rr=True, bland=False,
                         **unknown_options):
    r"""
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using the tableau-based simplex method.

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
    method : str
        This is the method-specific documentation for 'simplex'.
        :ref:`'highs' <optimize.linprog-highs>`,
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`,
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`,
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        and :ref:`'revised simplex' <optimize.linprog-revised_simplex>`
        are also available.
    callback : callable, optional
        Callback function to be executed once per iteration.

    Options
    -------
    maxiter : int (default: 5000)
       The maximum number of iterations to perform in either phase.
    disp : bool (default: False)
        Set to ``True`` if indicators of optimization status are to be printed
        to the console each iteration.
    presolve : bool (default: True)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    tol : float (default: 1e-12)
        The tolerance which determines when a solution is "close enough" to
        zero in Phase 1 to be considered a basic feasible solution or close
        enough to positive to serve as an optimal solution.
    autoscale : bool (default: False)
        Set to ``True`` to automatically perform equilibration.
        Consider using this option if the numerical values in the
        constraints are separated by several orders of magnitude.
    rr : bool (default: True)
        Set to ``False`` to disable automatic redundancy removal.
    bland : bool
        If True, use Bland's anti-cycling rule [3]_ to choose pivots to
        prevent cycling. If False, choose pivots which should lead to a
        converged solution more quickly. The latter method is subject to
        cycling (non-convergence) in rare instances.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        `unknown_options` is non-empty a warning is issued listing all
        unused options.

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

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed in all phases.

    References
    ----------
    .. [1] Dantzig, George B., Linear programming and extensions. Rand
           Corporation Research Study Princeton Univ. Press, Princeton, NJ,
           1963
    .. [2] Hillier, S.H. and Lieberman, G.J. (1995), "Introduction to
           Mathematical Programming", McGraw-Hill, Chapter 4.
    .. [3] Bland, Robert G. New finite pivoting rules for the simplex method.
           Mathematics of Operations Research (2), 1977: pp. 103-107.
    """
    pass
