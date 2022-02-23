"""
Unified interfaces to root finding algorithms for real or complex
scalar functions.

Functions
---------
- root : find a root of a scalar function.
"""
import numpy as np

from . import zeros as optzeros

__all__ = ['root_scalar']

ROOT_SCALAR_METHODS = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748',
                       'newton', 'secant', 'halley']


class MemoizeDer:
    """Decorator that caches the value and derivative(s) of function each
    time it is called.

    This is a simplistic memoizer that calls and caches a single value
    of `f(x, *args)`.
    It assumes that `args` does not change between invocations.
    It supports the use case of a root-finder where `args` is fixed,
    `x` changes, and only rarely, if at all, does x assume the same value
    more than once."""
    def __init__(self, fun):
        self.fun = fun
        self.vals = None
        self.x = None
        self.n_calls = 0

    def __call__(self, x, *args):
        r"""Calculate f or use cached value if available"""
        # Derivative may be requested before the function itself, always check
        if self.vals is None or x != self.x:
            fg = self.fun(x, *args)
            self.x = x
            self.n_calls += 1
            self.vals = fg[:]
        return self.vals[0]

    def fprime(self, x, *args):
        r"""Calculate f' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[1]

    def fprime2(self, x, *args):
        r"""Calculate f'' or use a cached value if available"""
        if self.vals is None or x != self.x:
            self(x, *args)
        return self.vals[2]

    def ncalls(self):
        return self.n_calls


def root_scalar(f, args=(), method=None, bracket=None,
                fprime=None, fprime2=None,
                x0=None, x1=None,
                xtol=None, rtol=None, maxiter=None,
                options=None):
    """
    Find a root of a scalar function.

    Parameters
    ----------
    f : callable
        A function to find a root of.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative(s).
    method : str, optional
        Type of solver.  Should be one of

            - 'bisect'    :ref:`(see here) <optimize.root_scalar-bisect>`
            - 'brentq'    :ref:`(see here) <optimize.root_scalar-brentq>`
            - 'brenth'    :ref:`(see here) <optimize.root_scalar-brenth>`
            - 'ridder'    :ref:`(see here) <optimize.root_scalar-ridder>`
            - 'toms748'    :ref:`(see here) <optimize.root_scalar-toms748>`
            - 'newton'    :ref:`(see here) <optimize.root_scalar-newton>`
            - 'secant'    :ref:`(see here) <optimize.root_scalar-secant>`
            - 'halley'    :ref:`(see here) <optimize.root_scalar-halley>`

    bracket: A sequence of 2 floats, optional
        An interval bracketing a root.  `f(x, *args)` must have different
        signs at the two endpoints.
    x0 : float, optional
        Initial guess.
    x1 : float, optional
        A second guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the derivative.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, optional
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of the objective function and of the
        first and second derivatives.
        `fprime2` can also be a callable returning the second derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options : dict, optional
        A dictionary of solver options. E.g., ``k``, see
        :obj:`show_options()` for details.

    Returns
    -------
    sol : RootResults
        The solution represented as a ``RootResults`` object.
        Important attributes are: ``root`` the solution , ``converged`` a
        boolean flag indicating if the algorithm exited successfully and
        ``flag`` which describes the cause of the termination. See
        `RootResults` for a description of other attributes.

    See also
    --------
    show_options : Additional options accepted by the solvers
    root : Find a root of a vector function.

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter.

    The default is to use the best method available for the situation
    presented.
    If a bracket is provided, it may use one of the bracketing methods.
    If a derivative and an initial value are specified, it may
    select one of the derivative-based methods.
    If no method is judged applicable, it will raise an Exception.


    Examples
    --------

    Find the root of a simple cubic

    >>> from scipy import optimize
    >>> def f(x):
    ...     return (x**3 - 1)  # only one real root at x = 1

    >>> def fprime(x):
    ...     return 3*x**2

    The `brentq` method takes as input a bracket

    >>> sol = optimize.root_scalar(f, bracket=[0, 3], method='brentq')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 10, 11)

    The `newton` method takes as input a single point and uses the derivative(s)

    >>> sol = optimize.root_scalar(f, x0=0.2, fprime=fprime, method='newton')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 22)

    The function can provide the value and derivative(s) in a single call.

    >>> def f_p_pp(x):
    ...     return (x**3 - 1), 3*x**2, 6*x

    >>> sol = optimize.root_scalar(f_p_pp, x0=0.2, fprime=True, method='newton')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 11, 11)

    >>> sol = optimize.root_scalar(f_p_pp, x0=0.2, fprime=True, fprime2=True, method='halley')
    >>> sol.root, sol.iterations, sol.function_calls
    (1.0, 7, 8)


    """
    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}

    # fun also returns the derivative(s)
    is_memoized = False
    if fprime2 is not None and not callable(fprime2):
        if bool(fprime2):
            f = MemoizeDer(f)
            is_memoized = True
            fprime2 = f.fprime2
            fprime = f.fprime
        else:
            fprime2 = None
    if fprime is not None and not callable(fprime):
        if bool(fprime):
            f = MemoizeDer(f)
            is_memoized = True
            fprime = f.fprime
        else:
            fprime = None

    # respect solver-specific default tolerances - only pass in if actually set
    kwargs = {}
    for k in ['xtol', 'rtol', 'maxiter']:
        v = locals().get(k)
        if v is not None:
            kwargs[k] = v

    # Set any solver-specific options
    if options:
        kwargs.update(options)
    # Always request full_output from the underlying method as _root_scalar
    # always returns a RootResults object
    kwargs.update(full_output=True, disp=False)

    # Pick a method if not specified.
    # Use the "best" method available for the situation.
    if not method:
        if bracket:
            method = 'brentq'
        elif x0 is not None:
            if fprime:
                if fprime2:
                    method = 'halley'
                else:
                    method = 'newton'
            else:
                method = 'secant'
    if not method:
        raise ValueError('Unable to select a solver as neither bracket '
                         'nor starting point provided.')

    meth = method.lower()
    map2underlying = {'halley': 'newton', 'secant': 'newton'}

    try:
        methodc = getattr(optzeros, map2underlying.get(meth, meth))
    except AttributeError as e:
        raise ValueError('Unknown solver %s' % meth) from e

    if meth in ['bisect', 'ridder', 'brentq', 'brenth', 'toms748']:
        if not isinstance(bracket, (list, tuple, np.ndarray)):
            raise ValueError('Bracket needed for %s' % method)

        a, b = bracket[:2]
        r, sol = methodc(f, a, b, args=args, **kwargs)
    elif meth in ['secant']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if x1 is None:
            raise ValueError('x1 must not be None for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=None, fprime2=None,
                         x1=x1, **kwargs)
    elif meth in ['newton']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if not fprime:
            raise ValueError('fprime must be specified for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=None,
                         **kwargs)
    elif meth in ['halley']:
        if x0 is None:
            raise ValueError('x0 must not be None for %s' % method)
        if not fprime:
            raise ValueError('fprime must be specified for %s' % method)
        if not fprime2:
            raise ValueError('fprime2 must be specified for %s' % method)
        if 'xtol' in kwargs:
            kwargs['tol'] = kwargs.pop('xtol')
        r, sol = methodc(f, x0, args=args, fprime=fprime, fprime2=fprime2, **kwargs)
    else:
        raise ValueError('Unknown solver %s' % method)

    if is_memoized:
        # Replace the function_calls count with the memoized count.
        # Avoids double and triple-counting.
        n_calls = f.n_calls
        sol.function_calls = n_calls

    return sol


def _root_scalar_brentq_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above

    """
    pass


def _root_scalar_brenth_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass

def _root_scalar_toms748_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_secant_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    x1 : float, required
        A second guess.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_newton_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivative.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, optional
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_halley_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    x0 : float, required
        Initial guess.
    fprime : bool or callable, required
        If `fprime` is a boolean and is True, `f` is assumed to return the
        value of derivative along with the objective function.
        `fprime` can also be a callable returning the derivative of `f`. In
        this case, it must accept the same arguments as `f`.
    fprime2 : bool or callable, required
        If `fprime2` is a boolean and is True, `f` is assumed to return the
        value of 1st and 2nd derivatives along with the objective function.
        `fprime2` can also be a callable returning the 2nd derivative of `f`.
        In this case, it must accept the same arguments as `f`.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_ridder_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass


def _root_scalar_bisect_doc():
    r"""
    Options
    -------
    args : tuple, optional
        Extra arguments passed to the objective function.
    xtol : float, optional
        Tolerance (absolute) for termination.
    rtol : float, optional
        Tolerance (relative) for termination.
    maxiter : int, optional
        Maximum number of iterations.
    options: dict, optional
        Specifies any method-specific options not covered above.

    """
    pass
