"""
Interface to Constrained Optimization By Linear Approximation

Functions
---------
.. autosummary::
   :toctree: generated/

    fmin_cobyla

"""

import functools
from threading import RLock

import numpy as np
from scipy.optimize import _cobyla
from .optimize import OptimizeResult, _check_unknown_options
try:
    from itertools import izip
except ImportError:
    izip = zip

__all__ = ['fmin_cobyla']

# Workarund as _cobyla.minimize is not threadsafe
# due to an unknown f2py bug and can segfault,
# see gh-9658.
_module_lock = RLock()
def synchronized(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _module_lock:
            return func(*args, **kwargs)
    return wrapper

@synchronized
def fmin_cobyla(func, x0, cons, args=(), consargs=None, rhobeg=1.0,
                rhoend=1e-4, maxfun=1000, disp=None, catol=2e-4):
    """
    Minimize a function using the Constrained Optimization By Linear
    Approximation (COBYLA) method. This method wraps a FORTRAN
    implementation of the algorithm.

    Parameters
    ----------
    func : callable
        Function to minimize. In the form func(x, \\*args).
    x0 : ndarray
        Initial guess.
    cons : sequence
        Constraint functions; must all be ``>=0`` (a single function
        if only 1 constraint). Each function takes the parameters `x`
        as its first argument, and it can return either a single number or
        an array or list of numbers.
    args : tuple, optional
        Extra arguments to pass to function.
    consargs : tuple, optional
        Extra arguments to pass to constraint functions (default of None means
        use same extra arguments as those passed to func).
        Use ``()`` for no extra arguments.
    rhobeg : float, optional
        Reasonable initial changes to the variables.
    rhoend : float, optional
        Final accuracy in the optimization (not precisely guaranteed). This
        is a lower bound on the size of the trust region.
    disp : {0, 1, 2, 3}, optional
        Controls the frequency of output; 0 implies no output.
    maxfun : int, optional
        Maximum number of function evaluations.
    catol : float, optional
        Absolute tolerance for constraint violations.

    Returns
    -------
    x : ndarray
        The argument that minimises `f`.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'COBYLA' `method` in particular.

    Notes
    -----
    This algorithm is based on linear approximations to the objective
    function and each constraint. We briefly describe the algorithm.

    Suppose the function is being minimized over k variables. At the
    jth iteration the algorithm has k+1 points v_1, ..., v_(k+1),
    an approximate solution x_j, and a radius RHO_j.
    (i.e., linear plus a constant) approximations to the objective
    function and constraint functions such that their function values
    agree with the linear approximation on the k+1 points v_1,.., v_(k+1).
    This gives a linear program to solve (where the linear approximations
    of the constraint functions are constrained to be non-negative).

    However, the linear approximations are likely only good
    approximations near the current simplex, so the linear program is
    given the further requirement that the solution, which
    will become x_(j+1), must be within RHO_j from x_j. RHO_j only
    decreases, never increases. The initial RHO_j is rhobeg and the
    final RHO_j is rhoend. In this way COBYLA's iterations behave
    like a trust region algorithm.

    Additionally, the linear program may be inconsistent, or the
    approximation may give poor improvement. For details about
    how these issues are resolved, as well as how the points v_i are
    updated, refer to the source code or the references below.


    References
    ----------
    Powell M.J.D. (1994), "A direct search optimization method that models
    the objective and constraint functions by linear interpolation.", in
    Advances in Optimization and Numerical Analysis, eds. S. Gomez and
    J-P Hennart, Kluwer Academic (Dordrecht), pp. 51-67

    Powell M.J.D. (1998), "Direct search algorithms for optimization
    calculations", Acta Numerica 7, 287-336

    Powell M.J.D. (2007), "A view of algorithms for optimization without
    derivatives", Cambridge University Technical Report DAMTP 2007/NA03


    Examples
    --------
    Minimize the objective function f(x,y) = x*y subject
    to the constraints x**2 + y**2 < 1 and y > 0::

        >>> def objective(x):
        ...     return x[0]*x[1]
        ...
        >>> def constr1(x):
        ...     return 1 - (x[0]**2 + x[1]**2)
        ...
        >>> def constr2(x):
        ...     return x[1]
        ...
        >>> from scipy.optimize import fmin_cobyla
        >>> fmin_cobyla(objective, [0.0, 0.1], [constr1, constr2], rhoend=1e-7)
        array([-0.70710685,  0.70710671])

    The exact solution is (-sqrt(2)/2, sqrt(2)/2).



    """
    err = "cons must be a sequence of callable functions or a single"\
          " callable function."
    try:
        len(cons)
    except TypeError as e:
        if callable(cons):
            cons = [cons]
        else:
            raise TypeError(err) from e
    else:
        for thisfunc in cons:
            if not callable(thisfunc):
                raise TypeError(err)

    if consargs is None:
        consargs = args

    # build constraints
    con = tuple({'type': 'ineq', 'fun': c, 'args': consargs} for c in cons)

    # options
    opts = {'rhobeg': rhobeg,
            'tol': rhoend,
            'disp': disp,
            'maxiter': maxfun,
            'catol': catol}

    sol = _minimize_cobyla(func, x0, args, constraints=con,
                           **opts)
    if disp and not sol['success']:
        print("COBYLA failed to find a solution: %s" % (sol.message,))
    return sol['x']

@synchronized
def _minimize_cobyla(fun, x0, args=(), constraints=(),
                     rhobeg=1.0, tol=1e-4, maxiter=1000,
                     disp=False, catol=2e-4, **unknown_options):
    """
    Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Linear Approximation (COBYLA) algorithm.

    Options
    -------
    rhobeg : float
        Reasonable initial changes to the variables.
    tol : float
        Final accuracy in the optimization (not precisely guaranteed).
        This is a lower bound on the size of the trust region.
    disp : bool
        Set to True to print convergence messages. If False,
        `verbosity` is ignored as set to 0.
    maxiter : int
        Maximum number of function evaluations.
    catol : float
        Tolerance (absolute) for constraint violations

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    rhoend = tol
    iprint = int(bool(disp))

    # check constraints
    if isinstance(constraints, dict):
        constraints = (constraints, )

    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype != 'ineq':
                raise ValueError("Constraints of type '%s' not handled by "
                                 "COBYLA." % con['type'])

        # check function
        if 'fun' not in con:
            raise KeyError('Constraint %d has no function defined.' % ic)

        # check extra arguments
        if 'args' not in con:
            con['args'] = ()

    # m is the total number of constraint values
    # it takes into account that some constraints may be vector-valued
    cons_lengths = []
    for c in constraints:
        f = c['fun'](x0, *c['args'])
        try:
            cons_length = len(f)
        except TypeError:
            cons_length = 1
        cons_lengths.append(cons_length)
    m = sum(cons_lengths)

    def calcfc(x, con):
        f = fun(np.copy(x), *args)
        i = 0
        for size, c in izip(cons_lengths, constraints):
            con[i: i + size] = c['fun'](x, *c['args'])
            i += size
        return f

    info = np.zeros(4, np.float64)
    xopt, info = _cobyla.minimize(calcfc, m=m, x=np.copy(x0), rhobeg=rhobeg,
                                  rhoend=rhoend, iprint=iprint, maxfun=maxfun,
                                  dinfo=info)

    if info[3] > catol:
        # Check constraint violation
        info[0] = 4

    return OptimizeResult(x=xopt,
                          status=int(info[0]),
                          success=info[0] == 1,
                          message={1: 'Optimization terminated successfully.',
                                   2: 'Maximum number of function evaluations '
                                      'has been exceeded.',
                                   3: 'Rounding errors are becoming damaging '
                                      'in COBYLA subroutine.',
                                   4: 'Did not converge to a solution '
                                      'satisfying the constraints. See '
                                      '`maxcv` for magnitude of violation.',
                                   5: 'NaN result encountered.'
                                   }.get(info[0], 'Unknown exit status.'),
                          nfev=int(info[1]),
                          fun=info[2],
                          maxcv=info[3])
