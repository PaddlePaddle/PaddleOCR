"""
Unified interfaces to minimization algorithms.

Functions
---------
- minimize : minimization of a function of several variables.
- minimize_scalar : minimization of a function of one variable.
"""

__all__ = ['minimize', 'minimize_scalar']


from warnings import warn

import numpy as np


# unconstrained minimization
from .optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
                       _minimize_bfgs, _minimize_newtoncg,
                       _minimize_scalar_brent, _minimize_scalar_bounded,
                       _minimize_scalar_golden, MemoizeJac)
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr

# constrained minimization
from .lbfgsb import _minimize_lbfgsb
from .tnc import _minimize_tnc
from .cobyla import _minimize_cobyla
from .slsqp import _minimize_slsqp
from ._constraints import (old_bound_to_new, new_bounds_to_old,
                           old_constraint_to_new, new_constraint_to_old,
                           NonlinearConstraint, LinearConstraint, Bounds)
from ._differentiable_functions import FD_METHODS

MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr',
                    'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

MINIMIZE_SCALAR_METHODS = ['brent', 'bounded', 'golden']

def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    """Minimization of scalar function of one or more variables.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        Type of solver.  Should be one of

            - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
            - custom - a callable object (added in version 0.14.0),
              see below for description.

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
        trust-exact and trust-constr.
        If it is a callable, it should be a function that returns the gradient
        vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters. If `jac` is a Boolean and is True, `fun` is
        assumed to return a tuple ``(f, g)`` containing the objective
        function and the gradient.
        Methods 'Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact', and
        'trust-krylov' require that either a callable be supplied, or that
        `fun` return the objective and gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
        trust-ncg, trust-krylov, trust-exact and trust-constr. If it is
        callable, it should return the Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        where x is a (n,) ndarray and `args` is a tuple with the fixed
        parameters. LinearOperator and sparse matrix returns are only allowed
        for 'trust-constr' method. Alternatively, the keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme
        for numerical estimation. Or, objects implementing the
        `HessianUpdateStrategy` interface can be used to approximate
        the Hessian. Available quasi-Newton methods implementing
        this interface are:

            - `BFGS`;
            - `SR1`.

        Whenever the gradient is estimated via finite-differences,
        the Hessian cannot be estimated with options
        {'2-point', '3-point', 'cs'} and needs to be
        estimated using one of the quasi-Newton strategies.
        'trust-exact' cannot use a finite-difference scheme, and must be used
        with a callable returning an (n, n) array.
    hessp : callable, optional
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, trust-ncg, trust-krylov, trust-constr.
        Only one of `hessp` or `hess` needs to be given.  If `hess` is
        provided, then `hessp` will be ignored.  `hessp` must compute the
        Hessian times an arbitrary vector:

            ``hessp(x, p, *args) ->  ndarray shape (n,)``

        where x is a (n,) ndarray, p is an arbitrary vector with
        dimension (n,) and `args` is a tuple with the fixed
        parameters.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and
        trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition (only for COBYLA, SLSQP and trust-constr).

        Constraints for 'trust-constr' are defined as a single object or a
        list of objects specifying constraints to the optimization problem.
        Available constraints are:

            - `LinearConstraint`
            - `NonlinearConstraint`

        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
        Each dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
        Note that COBYLA only supports inequality constraints.
    tol : float, optional
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options()`.
    callback : callable, optional
        Called after each iteration. For 'trust-constr' it is a callable with
        the signature:

            ``callback(xk, OptimizeResult state) -> bool``

        where ``xk`` is the current parameter vector. and ``state``
        is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True
        the algorithm execution is terminated.
        For all the other methods, the signature is:

            ``callback(xk)``

        where ``xk`` is the current parameter vector.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    minimize_scalar : Interface to minimization algorithms for scalar
        univariate functions
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *BFGS*.

    **Unconstrained minimization**

    Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate
    gradient algorithm by Polak and Ribiere, a variant of the
    Fletcher-Reeves method described in [5]_ pp.120-122. Only the
    first derivatives are used.

    Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton
    method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_
    pp. 136. It uses the first derivatives only. BFGS has proven good
    performance even for non-smooth optimizations. This method also
    returns an approximation of the Hessian inverse, stored as
    `hess_inv` in the OptimizeResult object.

    Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a
    Newton-CG algorithm [5]_ pp. 168 (also known as the truncated
    Newton method). It uses a CG method to the compute the search
    direction. See also *TNC* method for a box-constrained
    minimization with a similar algorithm. Suitable for large-scale
    problems.

    Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg
    trust-region algorithm [5]_ for unconstrained minimization. This
    algorithm requires the gradient and Hessian; furthermore the
    Hessian is required to be positive definite.

    Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the
    Newton conjugate gradient trust-region algorithm [5]_ for
    unconstrained minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.

    Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses
    the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained
    minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.
    On indefinite problems it requires usually less iterations than the
    `trust-ncg` method and is recommended for medium and large-scale problems.

    Method :ref:`trust-exact <optimize.minimize-trustexact>`
    is a trust-region method for unconstrained minimization in which
    quadratic subproblems are solved almost exactly [13]_. This
    algorithm requires the gradient and the Hessian (which is
    *not* required to be positive definite). It is, in many
    situations, the Newton method to converge in fewer iteraction
    and the most recommended for small and medium-size problems.

    **Bound-Constrained minimization**

    Method :ref:`Nelder-Mead <optimize.minimize-neldermead>` uses the
    Simplex algorithm [1]_, [2]_. This algorithm is robust in many
    applications. However, if numerical computation of derivative can be
    trusted, other algorithms using the first and/or second derivatives
    information might be preferred for their better performance in
    general.

    Method :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` uses the L-BFGS-B
    algorithm [6]_, [7]_ for bound constrained minimization.

    Method :ref:`Powell <optimize.minimize-powell>` is a modification
    of Powell's method [3]_, [4]_ which is a conjugate direction
    method. It performs sequential one-dimensional minimizations along
    each vector of the directions set (`direc` field in `options` and
    `info`), which is updated at each iteration of the main
    minimization loop. The function need not be differentiable, and no
    derivatives are taken. If bounds are not provided, then an
    unbounded line search will be used. If bounds are provided and
    the initial guess is within the bounds, then every function
    evaluation throughout the minimization procedure will be within
    the bounds. If bounds are provided, the initial guess is outside
    the bounds, and `direc` is full rank (default has full rank), then
    some function evaluations during the first iteration may be
    outside the bounds, but every function evaluation after the first
    iteration will be within the bounds. If `direc` is not full rank,
    then some parameters may not be optimized and the solution is not
    guaranteed to be within the bounds.

    Method :ref:`TNC <optimize.minimize-tnc>` uses a truncated Newton
    algorithm [5]_, [8]_ to minimize a function with variables subject
    to bounds. This algorithm uses gradient information; it is also
    called Newton Conjugate-Gradient. It differs from the *Newton-CG*
    method described above as it wraps a C implementation and allows
    each variable to be given upper and lower bounds.

    **Constrained Minimization**

    Method :ref:`COBYLA <optimize.minimize-cobyla>` uses the
    Constrained Optimization BY Linear Approximation (COBYLA) method
    [9]_, [10]_, [11]_. The algorithm is based on linear
    approximations to the objective function and each constraint. The
    method wraps a FORTRAN implementation of the algorithm. The
    constraints functions 'fun' may return either a single number
    or an array or list of numbers.

    Method :ref:`SLSQP <optimize.minimize-slsqp>` uses Sequential
    Least SQuares Programming to minimize a function of several
    variables with any combination of bounds, equality and inequality
    constraints. The method wraps the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft [12]_. Note that the
    wrapper handles infinite values in bounds by converting them into
    large floating values.

    Method :ref:`trust-constr <optimize.minimize-trustconstr>` is a
    trust-region algorithm for constrained optimization. It swiches
    between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in [17]_ and in [5]_, p. 549. When
    inequality constraints  are imposed as well, it swiches to the trust-region
    interior point  method described in [16]_. This interior point algorithm,
    in turn, solves inequality constraints by introducing slack variables
    and solving a sequence of equality-constrained barrier problems
    for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

    **Finite-Difference Options**

    For Method :ref:`trust-constr <optimize.minimize-trustconstr>`
    the gradient and the Hessian may be approximated using
    three finite-difference schemes: {'2-point', '3-point', 'cs'}.
    The scheme 'cs' is, potentially, the most accurate but it
    requires the function to correctly handles complex inputs and to
    be differentiable in the complex plane. The scheme '3-point' is more
    accurate than '2-point' but requires twice as many operations.

    **Custom minimizers**

    It may be useful to pass a custom minimization method, for example
    when using a frontend to this method such as `scipy.optimize.basinhopping`
    or a different library.  You can simply pass a callable as the ``method``
    parameter.

    The callable is called as ``method(fun, x0, args, **kwargs, **options)``
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `callback`, `hess`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  Also, if
    `jac` has been passed as a bool type, `jac` and `fun` are mangled so that
    `fun` returns just the function values and `jac` is converted to a function
    returning the Jacobian.  The method shall return an `OptimizeResult`
    object.

    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method.  You can find an example in the scipy.optimize tutorial.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function
        Minimization. The Computer Journal 7: 308-13.
    .. [2] Wright M H. 1996. Direct search methods: Once scorned, now
        respectable, in Numerical Analysis 1995: Proceedings of the 1995
        Dundee Biennial Conference in Numerical Analysis (Eds. D F
        Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.
        191-208.
    .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of
       a function of several variables without calculating derivatives. The
       Computer Journal 7: 155-162.
    .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.
       Numerical Recipes (any edition), Cambridge University Press.
    .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
       Springer New York.
    .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
       Algorithm for Bound Constrained Optimization. SIAM Journal on
       Scientific and Statistical Computing 16 (5): 1190-1208.
    .. [7] Zhu, C and R H Byrd and J Nocedal. 1997. L-BFGS-B: Algorithm
       778: L-BFGS-B, FORTRAN routines for large scale bound constrained
       optimization. ACM Transactions on Mathematical Software 23 (4):
       550-560.
    .. [8] Nash, S G. Newton-Type Minimization Via the Lanczos Method.
       1984. SIAM Journal of Numerical Analysis 21: 770-778.
    .. [9] Powell, M J D. A direct search optimization method that models
       the objective and constraint functions by linear interpolation.
       1994. Advances in Optimization and Numerical Analysis, eds. S. Gomez
       and J-P Hennart, Kluwer Academic (Dordrecht), 51-67.
    .. [10] Powell M J D. Direct search algorithms for optimization
       calculations. 1998. Acta Numerica 7: 287-336.
    .. [11] Powell M J D. A view of algorithms for optimization without
       derivatives. 2007.Cambridge University Technical Report DAMTP
       2007/NA03
    .. [12] Kraft, D. A software package for sequential quadratic
       programming. 1988. Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace
       Center -- Institute for Flight Mechanics, Koln, Germany.
    .. [13] Conn, A. R., Gould, N. I., and Toint, P. L.
       Trust region methods. 2000. Siam. pp. 169-200.
    .. [14] F. Lenders, C. Kirches, A. Potschka: "trlib: A vector-free
       implementation of the GLTR method for iterative solution of
       the trust region problem", :arxiv:`1611.04718`
    .. [15] N. Gould, S. Lucidi, M. Roma, P. Toint: "Solving the
       Trust-Region Subproblem using the Lanczos Method",
       SIAM J. Optim., 9(2), 504--525, (1999).
    .. [16] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal. 1999.
        An interior point algorithm for large-scale nonlinear  programming.
        SIAM Journal on Optimization 9.4: 877-900.
    .. [17] Lalee, Marucha, Jorge Nocedal, and Todd Plantega. 1998. On the
        implementation of an algorithm for large-scale equality constrained
        optimization. SIAM Journal on Optimization 8.3: 682-706.

    Examples
    --------
    Let us consider the problem of minimizing the Rosenbrock function. This
    function (and its respective derivatives) is implemented in `rosen`
    (resp. `rosen_der`, `rosen_hess`) in the `scipy.optimize`.

    >>> from scipy.optimize import minimize, rosen, rosen_der

    A simple application of the *Nelder-Mead* method is:

    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])

    Now using the *BFGS* algorithm, using the first derivative and a few
    options:

    >>> res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
    ...                options={'gtol': 1e-6, 'disp': True})
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 26
             Function evaluations: 31
             Gradient evaluations: 31
    >>> res.x
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> print(res.message)
    Optimization terminated successfully.
    >>> res.hess_inv
    array([[ 0.00749589,  0.01255155,  0.02396251,  0.04750988,  0.09495377],  # may vary
           [ 0.01255155,  0.02510441,  0.04794055,  0.09502834,  0.18996269],
           [ 0.02396251,  0.04794055,  0.09631614,  0.19092151,  0.38165151],
           [ 0.04750988,  0.09502834,  0.19092151,  0.38341252,  0.7664427 ],
           [ 0.09495377,  0.18996269,  0.38165151,  0.7664427,   1.53713523]])


    Next, consider a minimization problem with several constraints (namely
    Example 16.4 from [5]_). The objective function is:

    >>> fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

    There are three constraints defined as:

    >>> cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    ...         {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

    And variables must be positive, hence the following bounds:

    >>> bnds = ((0, None), (0, None))

    The optimization problem is solved using the SLSQP method as:

    >>> res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
    ...                constraints=cons)

    It should converge to the theoretical solution (1.4 ,1.7).

    """
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if method is None:
        # Select automatically
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()

    if options is None:
        options = {}
    # check if optional parameters are supported by the selected method
    # - jac
    if meth in ('nelder-mead', 'powell', 'cobyla') and bool(jac):
        warn('Method %s does not use gradient information (jac).' % method,
             RuntimeWarning)
    # - hess
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', 'trust-exact', '_custom') and hess is not None:
        warn('Method %s does not use Hessian information (hess).' % method,
             RuntimeWarning)
    # - hessp
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', '_custom') \
       and hessp is not None:
        warn('Method %s does not use Hessian-vector product '
             'information (hessp).' % method, RuntimeWarning)
    # - constraints or bounds
    if (meth in ('cg', 'bfgs', 'newton-cg', 'dogleg', 'trust-ncg')
            and (bounds is not None or np.any(constraints))):
        warn('Method %s cannot handle constraints nor bounds.' % method,
             RuntimeWarning)
    if meth in ('nelder-mead', 'l-bfgs-b', 'tnc', 'powell') and np.any(constraints):
        warn('Method %s cannot handle constraints.' % method,
             RuntimeWarning)
    if meth == 'cobyla' and bounds is not None:
        warn('Method %s cannot handle bounds.' % method,
             RuntimeWarning)
    # - callback
    if (meth in ('cobyla',) and callback is not None):
        warn('Method %s does not support callback.' % method, RuntimeWarning)
    # - return_all
    if (meth in ('l-bfgs-b', 'tnc', 'cobyla', 'slsqp') and
            options.get('return_all', False)):
        warn('Method %s does not support the return_all option.' % method,
             RuntimeWarning)

    # check gradient vector
    if callable(jac):
        pass
    elif jac is True:
        # fun returns func and grad
        fun = MemoizeJac(fun)
        jac = fun.derivative
    elif (jac in FD_METHODS and
          meth in ['trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp']):
        # finite differences with relative step
        pass
    elif meth in ['trust-constr']:
        # default jac calculation for this method
        jac = '2-point'
    elif jac is None or bool(jac) is False:
        # this will cause e.g. LBFGS to use forward difference, absolute step
        jac = None
    else:
        # default if jac option is not understood
        jac = None

    # set default tolerances
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
                    'trust-ncg', 'trust-exact', 'trust-krylov'):
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            options.setdefault('tol', tol)
        if meth == 'trust-constr':
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)

    if meth == '_custom':
        # custom method called before bounds and constraints are 'standardised'
        # custom method should be able to accept whatever bounds/constraints
        # are provided to it.
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                      bounds=bounds, constraints=constraints,
                      callback=callback, **options)

    if bounds is not None:
        bounds = standardize_bounds(bounds, x0, meth)

    if constraints is not None:
        constraints = standardize_constraints(constraints, x0, meth)

    if meth == 'nelder-mead':
        return _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
                                    **options)
    elif meth == 'powell':
        return _minimize_powell(fun, x0, args, callback, bounds, **options)
    elif meth == 'cg':
        return _minimize_cg(fun, x0, args, jac, callback, **options)
    elif meth == 'bfgs':
        return _minimize_bfgs(fun, x0, args, jac, callback, **options)
    elif meth == 'newton-cg':
        return _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback,
                                  **options)
    elif meth == 'l-bfgs-b':
        return _minimize_lbfgsb(fun, x0, args, jac, bounds,
                                callback=callback, **options)
    elif meth == 'tnc':
        return _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
                             **options)
    elif meth == 'cobyla':
        return _minimize_cobyla(fun, x0, args, constraints, **options)
    elif meth == 'slsqp':
        return _minimize_slsqp(fun, x0, args, jac, bounds,
                               constraints, callback=callback, **options)
    elif meth == 'trust-constr':
        return _minimize_trustregion_constr(fun, x0, args, jac, hess, hessp,
                                            bounds, constraints,
                                            callback=callback, **options)
    elif meth == 'dogleg':
        return _minimize_dogleg(fun, x0, args, jac, hess,
                                callback=callback, **options)
    elif meth == 'trust-ncg':
        return _minimize_trust_ncg(fun, x0, args, jac, hess, hessp,
                                   callback=callback, **options)
    elif meth == 'trust-krylov':
        return _minimize_trust_krylov(fun, x0, args, jac, hess, hessp,
                                      callback=callback, **options)
    elif meth == 'trust-exact':
        return _minimize_trustregion_exact(fun, x0, args, jac, hess,
                                           callback=callback, **options)
    else:
        raise ValueError('Unknown solver %s' % method)


def minimize_scalar(fun, bracket=None, bounds=None, args=(),
                    method='brent', tol=None, options=None):
    """Minimization of scalar function of one variable.

    Parameters
    ----------
    fun : callable
        Objective function.
        Scalar function, must return a scalar.
    bracket : sequence, optional
        For methods 'brent' and 'golden', `bracket` defines the bracketing
        interval and can either have three items ``(a, b, c)`` so that
        ``a < b < c`` and ``fun(b) < fun(a), fun(c)`` or two items ``a`` and
        ``c`` which are assumed to be a starting interval for a downhill
        bracket search (see `bracket`); it doesn't always mean that the
        obtained solution will satisfy ``a <= x <= c``.
    bounds : sequence, optional
        For method 'bounded', `bounds` is mandatory and must have two items
        corresponding to the optimization bounds.
    args : tuple, optional
        Extra arguments passed to the objective function.
    method : str or callable, optional
        Type of solver.  Should be one of:

            - 'Brent'     :ref:`(see here) <optimize.minimize_scalar-brent>`
            - 'Bounded'   :ref:`(see here) <optimize.minimize_scalar-bounded>`
            - 'Golden'    :ref:`(see here) <optimize.minimize_scalar-golden>`
            - custom - a callable object (added in version 0.14.0), see below

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options.

            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.

        See :func:`show_options()` for solver-specific options.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    See also
    --------
    minimize : Interface to minimization algorithms for scalar multivariate
        functions
    show_options : Additional options accepted by the solvers

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *Brent*.

    Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's
    algorithm to find a local minimum.  The algorithm uses inverse
    parabolic interpolation when possible to speed up convergence of
    the golden section method.

    Method :ref:`Golden <optimize.minimize_scalar-golden>` uses the
    golden section search technique. It uses analog of the bisection
    method to decrease the bracketed interval. It is usually
    preferable to use the *Brent* method.

    Method :ref:`Bounded <optimize.minimize_scalar-bounded>` can
    perform bounded minimization. It uses the Brent method to find a
    local minimum in the interval x1 < xopt < x2.

    **Custom minimizers**

    It may be useful to pass a custom minimization method, for example
    when using some library frontend to minimize_scalar. You can simply
    pass a callable as the ``method`` parameter.

    The callable is called as ``method(fun, args, **kwargs, **options)``
    where ``kwargs`` corresponds to any other parameters passed to `minimize`
    (such as `bracket`, `tol`, etc.), except the `options` dict, which has
    its contents also passed as `method` parameters pair by pair.  The method
    shall return an `OptimizeResult` object.

    The provided `method` callable must be able to accept (and possibly ignore)
    arbitrary parameters; the set of parameters accepted by `minimize` may
    expand in future versions and then these parameters will be passed to
    the method. You can find an example in the scipy.optimize tutorial.

    .. versionadded:: 0.11.0

    Examples
    --------
    Consider the problem of minimizing the following function.

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    Using the *Brent* method, we find the local minimum as:

    >>> from scipy.optimize import minimize_scalar
    >>> res = minimize_scalar(f)
    >>> res.x
    1.28077640403

    Using the *Bounded* method, we find a local minimum with specified
    bounds as:

    >>> res = minimize_scalar(f, bounds=(-3, -1), method='bounded')
    >>> res.x
    -2.0000002026

    """
    if not isinstance(args, tuple):
        args = (args,)

    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()
    if options is None:
        options = {}

    if tol is not None:
        options = dict(options)
        if meth == 'bounded' and 'xatol' not in options:
            warn("Method 'bounded' does not support relative tolerance in x; "
                 "defaulting to absolute tolerance.", RuntimeWarning)
            options['xatol'] = tol
        elif meth == '_custom':
            options.setdefault('tol', tol)
        else:
            options.setdefault('xtol', tol)

    if meth == '_custom':
        return method(fun, args=args, bracket=bracket, bounds=bounds, **options)
    elif meth == 'brent':
        return _minimize_scalar_brent(fun, bracket, args, **options)
    elif meth == 'bounded':
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for '
                             'method `bounded`.')
        # replace boolean "disp" option, if specified, by an integer value, as
        # expected by _minimize_scalar_bounded()
        disp = options.get('disp')
        if isinstance(disp, bool):
            options['disp'] = 2 * int(disp)
        return _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        return _minimize_scalar_golden(fun, bracket, args, **options)
    else:
        raise ValueError('Unknown solver %s' % method)


def standardize_bounds(bounds, x0, meth):
    """Converts bounds to the form required by the solver."""
    if meth in {'trust-constr', 'powell', 'nelder-mead'}:
        if not isinstance(bounds, Bounds):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
    elif meth in ('l-bfgs-b', 'tnc', 'slsqp'):
        if isinstance(bounds, Bounds):
            bounds = new_bounds_to_old(bounds.lb, bounds.ub, x0.shape[0])
    return bounds


def standardize_constraints(constraints, x0, meth):
    """Converts constraints to the form required by the solver."""
    all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
    new_constraint_types = all_constraint_types[:-1]
    if isinstance(constraints, all_constraint_types):
        constraints = [constraints]
    constraints = list(constraints)  # ensure it's a mutable sequence

    if meth == 'trust-constr':
        for i, con in enumerate(constraints):
            if not isinstance(con, new_constraint_types):
                constraints[i] = old_constraint_to_new(i, con)
    else:
        # iterate over copy, changing original
        for i, con in enumerate(list(constraints)):
            if isinstance(con, new_constraint_types):
                old_constraints = new_constraint_to_old(con, x0)
                constraints[i] = old_constraints[0]
                constraints.extend(old_constraints[1:])  # appends 1 if present

    return constraints
