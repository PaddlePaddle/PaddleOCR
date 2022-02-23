# Authors: Pearu Peterson, Pauli Virtanen, John Travers
"""
First-order ODE integrators.

User-friendly interface to various numerical integrators for solving a
system of first order ODEs with prescribed initial conditions::

    d y(t)[i]
    ---------  = f(t,y(t))[i],
       d t

    y(t=0)[i] = y0[i],

where::

    i = 0, ..., len(y0) - 1

class ode
---------

A generic interface class to numeric integrators. It has the following
methods::

    integrator = ode(f, jac=None)
    integrator = integrator.set_integrator(name, **params)
    integrator = integrator.set_initial_value(y0, t0=0.0)
    integrator = integrator.set_f_params(*args)
    integrator = integrator.set_jac_params(*args)
    y1 = integrator.integrate(t1, step=False, relax=False)
    flag = integrator.successful()

class complex_ode
-----------------

This class has the same generic interface as ode, except it can handle complex
f, y and Jacobians by transparently translating them into the equivalent
real-valued system. It supports the real-valued solvers (i.e., not zvode) and is
an alternative to ode with the zvode solver, sometimes performing better.
"""
# XXX: Integrators must have:
# ===========================
# cvode - C version of vode and vodpk with many improvements.
#   Get it from http://www.netlib.org/ode/cvode.tar.gz.
#   To wrap cvode to Python, one must write the extension module by
#   hand. Its interface is too much 'advanced C' that using f2py
#   would be too complicated (or impossible).
#
# How to define a new integrator:
# ===============================
#
# class myodeint(IntegratorBase):
#
#     runner = <odeint function> or None
#
#     def __init__(self,...):                           # required
#         <initialize>
#
#     def reset(self,n,has_jac):                        # optional
#         # n - the size of the problem (number of equations)
#         # has_jac - whether user has supplied its own routine for Jacobian
#         <allocate memory,initialize further>
#
#     def run(self,f,jac,y0,t0,t1,f_params,jac_params): # required
#         # this method is called to integrate from t=t0 to t=t1
#         # with initial condition y0. f and jac are user-supplied functions
#         # that define the problem. f_params,jac_params are additional
#         # arguments
#         # to these functions.
#         <calculate y1>
#         if <calculation was unsuccessful>:
#             self.success = 0
#         return t1,y1
#
#     # In addition, one can define step() and run_relax() methods (they
#     # take the same arguments as run()) if the integrator can support
#     # these features (see IntegratorBase doc strings).
#
# if myodeint.runner:
#     IntegratorBase.integrator_classes.append(myodeint)

__all__ = ['ode', 'complex_ode']
__version__ = "$Id$"
__docformat__ = "restructuredtext en"

import re
import warnings

from numpy import asarray, array, zeros, isscalar, real, imag, vstack

from . import vode as _vode
from . import _dop
from . import lsoda as _lsoda


_dop_int_dtype = _dop.types.intvar.dtype
_vode_int_dtype = _vode.types.intvar.dtype
_lsoda_int_dtype = _lsoda.types.intvar.dtype


# ------------------------------------------------------------------------------
# User interface
# ------------------------------------------------------------------------------


class ode:
    """
    A generic interface class to numeric integrators.

    Solve an equation system :math:`y'(t) = f(t,y)` with (optional) ``jac = df/dy``.

    *Note*: The first two arguments of ``f(t, y, ...)`` are in the
    opposite order of the arguments in the system definition function used
    by `scipy.integrate.odeint`.

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        Right-hand side of the differential equation. t is a scalar,
        ``y.shape == (n,)``.
        ``f_args`` is set by calling ``set_f_params(*args)``.
        `f` should return a scalar, array or list (not a tuple).
    jac : callable ``jac(t, y, *jac_args)``, optional
        Jacobian of the right-hand side, ``jac[i,j] = d f[i] / d y[j]``.
        ``jac_args`` is set by calling ``set_jac_params(*args)``.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current variable values.

    See also
    --------
    odeint : an integrator with a simpler interface based on lsoda from ODEPACK
    quad : for finding the area under a curve

    Notes
    -----
    Available integrators are listed below. They can be selected using
    the `set_integrator` method.

    "vode"

        Real-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        implicit Adams method (for non-stiff problems) and a method based on
        backward differentiation formulas (BDF) (for stiff problems).

        Source: http://www.netlib.org/ode/vode.f

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "vode" integrator at the same time.

        This integrator accepts the following parameters in `set_integrator`
        method of the `ode` class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - lband : None or int
        - uband : None or int
          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
          Setting these requires your jac routine to return the jacobian
          in packed format, jac_packed[i-j+uband, j] = jac[i,j]. The
          dimension of the matrix must be (lband+uband+1, len(y)).
        - method: 'adams' or 'bdf'
          Which solver to use, Adams (non-stiff) or BDF (stiff)
        - with_jacobian : bool
          This option is only considered when the user has not supplied a
          Jacobian function and has not indicated (by setting either band)
          that the Jacobian is banded. In this case, `with_jacobian` specifies
          whether the iteration method of the ODE solver's correction step is
          chord iteration with an internally generated full Jacobian or
          functional iteration with no Jacobian.
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - min_step : float
        - max_step : float
          Limits for the step sizes used by the integrator.
        - order : int
          Maximum order used by the integrator,
          order <= 12 for Adams, <= 5 for BDF.

    "zvode"

        Complex-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        implicit Adams method (for non-stiff problems) and a method based on
        backward differentiation formulas (BDF) (for stiff problems).

        Source: http://www.netlib.org/ode/zvode.f

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "zvode" integrator at the same time.

        This integrator accepts the same parameters in `set_integrator`
        as the "vode" solver.

        .. note::

            When using ZVODE for a stiff system, it should only be used for
            the case in which the function f is analytic, that is, when each f(i)
            is an analytic function of each y(j). Analyticity means that the
            partial derivative df(i)/dy(j) is a unique complex number, and this
            fact is critical in the way ZVODE solves the dense or banded linear
            systems that arise in the stiff case. For a complex stiff ODE system
            in which f is not analytic, ZVODE is likely to have convergence
            failures, and for this problem one should instead use DVODE on the
            equivalent real system (in the real and imaginary parts of y).

    "lsoda"

        Real-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        automatic method switching between implicit Adams method (for non-stiff
        problems) and a method based on backward differentiation formulas (BDF)
        (for stiff problems).

        Source: http://www.netlib.org/odepack

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "lsoda" integrator at the same time.

        This integrator accepts the following parameters in `set_integrator`
        method of the `ode` class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - lband : None or int
        - uband : None or int
          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
          Setting these requires your jac routine to return the jacobian
          in packed format, jac_packed[i-j+uband, j] = jac[i,j].
        - with_jacobian : bool
          *Not used.*
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - min_step : float
        - max_step : float
          Limits for the step sizes used by the integrator.
        - max_order_ns : int
          Maximum order used in the nonstiff case (default 12).
        - max_order_s : int
          Maximum order used in the stiff case (default 5).
        - max_hnil : int
          Maximum number of messages reporting too small step size (t + h = t)
          (default 0)
        - ixpr : int
          Whether to generate extra printing at method switches (default False).

    "dopri5"

        This is an explicit runge-kutta method of order (4)5 due to Dormand &
        Prince (with stepsize control and dense output).

        Authors:

            E. Hairer and G. Wanner
            Universite de Geneve, Dept. de Mathematiques
            CH-1211 Geneve 24, Switzerland
            e-mail:  ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch

        This code is described in [HNW93]_.

        This integrator accepts the following parameters in set_integrator()
        method of the ode class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - max_step : float
        - safety : float
          Safety factor on new step selection (default 0.9)
        - ifactor : float
        - dfactor : float
          Maximum factor to increase/decrease step size by in one step
        - beta : float
          Beta parameter for stabilised step size control.
        - verbosity : int
          Switch for printing messages (< 0 for no messages).

    "dop853"

        This is an explicit runge-kutta method of order 8(5,3) due to Dormand
        & Prince (with stepsize control and dense output).

        Options and references the same as "dopri5".

    Examples
    --------

    A problem to integrate and the corresponding jacobian:

    >>> from scipy.integrate import ode
    >>>
    >>> y0, t0 = [1.0j, 2.0], 0
    >>>
    >>> def f(t, y, arg1):
    ...     return [1j*arg1*y[0] + y[1], -arg1*y[1]**2]
    >>> def jac(t, y, arg1):
    ...     return [[1j*arg1, 1], [0, -arg1*2*y[1]]]

    The integration:

    >>> r = ode(f, jac).set_integrator('zvode', method='bdf')
    >>> r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
    >>> t1 = 10
    >>> dt = 1
    >>> while r.successful() and r.t < t1:
    ...     print(r.t+dt, r.integrate(r.t+dt))
    1 [-0.71038232+0.23749653j  0.40000271+0.j        ]
    2.0 [0.19098503-0.52359246j 0.22222356+0.j        ]
    3.0 [0.47153208+0.52701229j 0.15384681+0.j        ]
    4.0 [-0.61905937+0.30726255j  0.11764744+0.j        ]
    5.0 [0.02340997-0.61418799j 0.09523835+0.j        ]
    6.0 [0.58643071+0.339819j 0.08000018+0.j      ]
    7.0 [-0.52070105+0.44525141j  0.06896565+0.j        ]
    8.0 [-0.15986733-0.61234476j  0.06060616+0.j        ]
    9.0 [0.64850462+0.15048982j 0.05405414+0.j        ]
    10.0 [-0.38404699+0.56382299j  0.04878055+0.j        ]

    References
    ----------
    .. [HNW93] E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary
        Differential Equations i. Nonstiff Problems. 2nd edition.
        Springer Series in Computational Mathematics,
        Springer-Verlag (1993)

    """

    def __init__(self, f, jac=None):
        self.stiff = 0
        self.f = f
        self.jac = jac
        self.f_params = ()
        self.jac_params = ()
        self._y = []

    @property
    def y(self):
        return self._y

    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        if isscalar(y):
            y = [y]
        n_prev = len(self._y)
        if not n_prev:
            self.set_integrator('')  # find first available integrator
        self._y = asarray(y, self._integrator.scalar)
        self.t = t
        self._integrator.reset(len(self._y), self.jac is not None)
        return self

    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator.
        integrator_params
            Additional parameters for the integrator.
        """
        integrator = find_integrator(name)
        if integrator is None:
            # FIXME: this really should be raise an exception. Will that break
            # any code?
            warnings.warn('No integrator name match with %r or is not '
                          'available.' % name)
        else:
            self._integrator = integrator(**integrator_params)
            if not len(self._y):
                self.t = 0.0
                self._y = array([0.0], self._integrator.scalar)
            self._integrator.reset(len(self._y), self.jac is not None)
        return self

    def integrate(self, t, step=False, relax=False):
        """Find y=y(t), set y as an initial condition, and return y.

        Parameters
        ----------
        t : float
            The endpoint of the integration step.
        step : bool
            If True, and if the integrator supports the step method,
            then perform a single integration step and return.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.
        relax : bool
            If True and if the integrator supports the run_relax method,
            then integrate until t_1 >= t and return. ``relax`` is not
            referenced if ``step=True``.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.

        Returns
        -------
        y : float
            The integrated value at t
        """
        if step and self._integrator.supports_step:
            mth = self._integrator.step
        elif relax and self._integrator.supports_run_relax:
            mth = self._integrator.run_relax
        else:
            mth = self._integrator.run

        try:
            self._y, self.t = mth(self.f, self.jac or (lambda: None),
                                  self._y, self.t, t,
                                  self.f_params, self.jac_params)
        except SystemError as e:
            # f2py issue with tuple returns, see ticket 1187.
            raise ValueError(
                'Function to integrate must not return a tuple.'
            ) from e

        return self._y

    def successful(self):
        """Check if integration was successful."""
        try:
            self._integrator
        except AttributeError:
            self.set_integrator('')
        return self._integrator.success == 1

    def get_return_code(self):
        """Extracts the return code for the integration to enable better control
        if the integration fails.

        In general, a return code > 0 implies success, while a return code < 0
        implies failure.

        Notes
        -----
        This section describes possible return codes and their meaning, for available
        integrators that can be selected by `set_integrator` method.

        "vode"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call. (Perhaps wrong MF.)
        -2           Excess accuracy requested. (Tolerances too small.)
        -3           Illegal input detected. (See printed message.)
        -4           Repeated error test failures. (Check all input.)
        -5           Repeated convergence failures. (Perhaps bad Jacobian
                     supplied or wrong choice of MF or tolerances.)
        -6           Error weight became zero during problem. (Solution
                     component i vanished, and ATOL or ATOL(i) = 0.)
        ===========  =======

        "zvode"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call. (Perhaps wrong MF.)
        -2           Excess accuracy requested. (Tolerances too small.)
        -3           Illegal input detected. (See printed message.)
        -4           Repeated error test failures. (Check all input.)
        -5           Repeated convergence failures. (Perhaps bad Jacobian
                     supplied or wrong choice of MF or tolerances.)
        -6           Error weight became zero during problem. (Solution
                     component i vanished, and ATOL or ATOL(i) = 0.)
        ===========  =======

        "dopri5"

        ===========  =======
        Return Code  Message
        ===========  =======
        1            Integration successful.
        2            Integration successful (interrupted by solout).
        -1           Input is not consistent.
        -2           Larger nsteps is needed.
        -3           Step size becomes too small.
        -4           Problem is probably stiff (interrupted).
        ===========  =======

        "dop853"

        ===========  =======
        Return Code  Message
        ===========  =======
        1            Integration successful.
        2            Integration successful (interrupted by solout).
        -1           Input is not consistent.
        -2           Larger nsteps is needed.
        -3           Step size becomes too small.
        -4           Problem is probably stiff (interrupted).
        ===========  =======

        "lsoda"

        ===========  =======
        Return Code  Message
        ===========  =======
        2            Integration successful.
        -1           Excess work done on this call (perhaps wrong Dfun type).
        -2           Excess accuracy requested (tolerances too small).
        -3           Illegal input detected (internal error).
        -4           Repeated error test failures (internal error).
        -5           Repeated convergence failures (perhaps bad Jacobian or tolerances).
        -6           Error weight became zero during problem.
        -7           Internal workspace insufficient to finish (internal error).
        ===========  =======
        """
        try:
            self._integrator
        except AttributeError:
            self.set_integrator('')
        return self._integrator.istate

    def set_f_params(self, *args):
        """Set extra parameters for user-supplied function f."""
        self.f_params = args
        return self

    def set_jac_params(self, *args):
        """Set extra parameters for user-supplied function jac."""
        self.jac_params = args
        return self

    def set_solout(self, solout):
        """
        Set callable to be called at every successful integration step.

        Parameters
        ----------
        solout : callable
            ``solout(t, y)`` is called at each internal integrator step,
            t is a scalar providing the current independent position
            y is the current soloution ``y.shape == (n,)``
            solout should return -1 to stop integration
            otherwise it should return None or 0

        """
        if self._integrator.supports_solout:
            self._integrator.set_solout(solout)
            if self._y is not None:
                self._integrator.reset(len(self._y), self.jac is not None)
        else:
            raise ValueError("selected integrator does not support solout,"
                             " choose another one")


def _transform_banded_jac(bjac):
    """
    Convert a real matrix of the form (for example)

        [0 0 A B]        [0 0 0 B]
        [0 0 C D]        [0 0 A D]
        [E F G H]   to   [0 F C H]
        [I J K L]        [E J G L]
                         [I 0 K 0]

    That is, every other column is shifted up one.
    """
    # Shift every other column.
    newjac = zeros((bjac.shape[0] + 1, bjac.shape[1]))
    newjac[1:, ::2] = bjac[:, ::2]
    newjac[:-1, 1::2] = bjac[:, 1::2]
    return newjac


class complex_ode(ode):
    """
    A wrapper of ode for complex systems.

    This functions similarly as `ode`, but re-maps a complex-valued
    equation system to a real-valued one before using the integrators.

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        Rhs of the equation. t is a scalar, ``y.shape == (n,)``.
        ``f_args`` is set by calling ``set_f_params(*args)``.
    jac : callable ``jac(t, y, *jac_args)``
        Jacobian of the rhs, ``jac[i,j] = d f[i] / d y[j]``.
        ``jac_args`` is set by calling ``set_f_params(*args)``.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current variable values.

    Examples
    --------
    For usage examples, see `ode`.

    """

    def __init__(self, f, jac=None):
        self.cf = f
        self.cjac = jac
        if jac is None:
            ode.__init__(self, self._wrap, None)
        else:
            ode.__init__(self, self._wrap, self._wrap_jac)

    def _wrap(self, t, y, *f_args):
        f = self.cf(*((t, y[::2] + 1j * y[1::2]) + f_args))
        # self.tmp is a real-valued array containing the interleaved
        # real and imaginary parts of f.
        self.tmp[::2] = real(f)
        self.tmp[1::2] = imag(f)
        return self.tmp

    def _wrap_jac(self, t, y, *jac_args):
        # jac is the complex Jacobian computed by the user-defined function.
        jac = self.cjac(*((t, y[::2] + 1j * y[1::2]) + jac_args))

        # jac_tmp is the real version of the complex Jacobian.  Each complex
        # entry in jac, say 2+3j, becomes a 2x2 block of the form
        #     [2 -3]
        #     [3  2]
        jac_tmp = zeros((2 * jac.shape[0], 2 * jac.shape[1]))
        jac_tmp[1::2, 1::2] = jac_tmp[::2, ::2] = real(jac)
        jac_tmp[1::2, ::2] = imag(jac)
        jac_tmp[::2, 1::2] = -jac_tmp[1::2, ::2]

        ml = getattr(self._integrator, 'ml', None)
        mu = getattr(self._integrator, 'mu', None)
        if ml is not None or mu is not None:
            # Jacobian is banded.  The user's Jacobian function has computed
            # the complex Jacobian in packed format.  The corresponding
            # real-valued version has every other column shifted up.
            jac_tmp = _transform_banded_jac(jac_tmp)

        return jac_tmp

    @property
    def y(self):
        return self._y[::2] + 1j * self._y[1::2]

    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator
        integrator_params
            Additional parameters for the integrator.
        """
        if name == 'zvode':
            raise ValueError("zvode must be used with ode, not complex_ode")

        lband = integrator_params.get('lband')
        uband = integrator_params.get('uband')
        if lband is not None or uband is not None:
            # The Jacobian is banded.  Override the user-supplied bandwidths
            # (which are for the complex Jacobian) with the bandwidths of
            # the corresponding real-valued Jacobian wrapper of the complex
            # Jacobian.
            integrator_params['lband'] = 2 * (lband or 0) + 1
            integrator_params['uband'] = 2 * (uband or 0) + 1

        return ode.set_integrator(self, name, **integrator_params)

    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        y = asarray(y)
        self.tmp = zeros(y.size * 2, 'float')
        self.tmp[::2] = real(y)
        self.tmp[1::2] = imag(y)
        return ode.set_initial_value(self, self.tmp, t)

    def integrate(self, t, step=False, relax=False):
        """Find y=y(t), set y as an initial condition, and return y.

        Parameters
        ----------
        t : float
            The endpoint of the integration step.
        step : bool
            If True, and if the integrator supports the step method,
            then perform a single integration step and return.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.
        relax : bool
            If True and if the integrator supports the run_relax method,
            then integrate until t_1 >= t and return. ``relax`` is not
            referenced if ``step=True``.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.

        Returns
        -------
        y : float
            The integrated value at t
        """
        y = ode.integrate(self, t, step, relax)
        return y[::2] + 1j * y[1::2]

    def set_solout(self, solout):
        """
        Set callable to be called at every successful integration step.

        Parameters
        ----------
        solout : callable
            ``solout(t, y)`` is called at each internal integrator step,
            t is a scalar providing the current independent position
            y is the current soloution ``y.shape == (n,)``
            solout should return -1 to stop integration
            otherwise it should return None or 0

        """
        if self._integrator.supports_solout:
            self._integrator.set_solout(solout, complex=True)
        else:
            raise TypeError("selected integrator does not support solouta,"
                            + "choose another one")


# ------------------------------------------------------------------------------
# ODE integrators
# ------------------------------------------------------------------------------

def find_integrator(name):
    for cl in IntegratorBase.integrator_classes:
        if re.match(name, cl.__name__, re.I):
            return cl
    return None


class IntegratorConcurrencyError(RuntimeError):
    """
    Failure due to concurrent usage of an integrator that can be used
    only for a single problem at a time.

    """

    def __init__(self, name):
        msg = ("Integrator `%s` can be used to solve only a single problem "
               "at a time. If you want to integrate multiple problems, "
               "consider using a different integrator "
               "(see `ode.set_integrator`)") % name
        RuntimeError.__init__(self, msg)


class IntegratorBase:
    runner = None  # runner is None => integrator is not available
    success = None  # success==1 if integrator was called successfully
    istate = None  # istate > 0 means success, istate < 0 means failure
    supports_run_relax = None
    supports_step = None
    supports_solout = False
    integrator_classes = []
    scalar = float

    def acquire_new_handle(self):
        # Some of the integrators have internal state (ancient
        # Fortran...), and so only one instance can use them at a time.
        # We keep track of this, and fail when concurrent usage is tried.
        self.__class__.active_global_handle += 1
        self.handle = self.__class__.active_global_handle

    def check_handle(self):
        if self.handle is not self.__class__.active_global_handle:
            raise IntegratorConcurrencyError(self.__class__.__name__)

    def reset(self, n, has_jac):
        """Prepare integrator for call: allocate memory, set flags, etc.
        n - number of equations.
        has_jac - if user has supplied function for evaluating Jacobian.
        """

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Integrate from t=t0 to t=t1 using y0 as an initial condition.
        Return 2-tuple (y1,t1) where y1 is the result and t=t1
        defines the stoppage coordinate of the result.
        """
        raise NotImplementedError('all integrators must define '
                                  'run(f, jac, t0, t1, y0, f_params, jac_params)')

    def step(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Make one integration step and return (y1,t1)."""
        raise NotImplementedError('%s does not support step() method' %
                                  self.__class__.__name__)

    def run_relax(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Integrate from t=t0 to t>=t1 and return (y1,t)."""
        raise NotImplementedError('%s does not support run_relax() method' %
                                  self.__class__.__name__)

    # XXX: __str__ method for getting visual state of the integrator


def _vode_banded_jac_wrapper(jacfunc, ml, jac_params):
    """
    Wrap a banded Jacobian function with a function that pads
    the Jacobian with `ml` rows of zeros.
    """

    def jac_wrapper(t, y):
        jac = asarray(jacfunc(t, y, *jac_params))
        padded_jac = vstack((jac, zeros((ml, jac.shape[1]))))
        return padded_jac

    return jac_wrapper


class vode(IntegratorBase):
    runner = getattr(_vode, 'dvode', None)

    messages = {-1: 'Excess work done on this call. (Perhaps wrong MF.)',
                -2: 'Excess accuracy requested. (Tolerances too small.)',
                -3: 'Illegal input detected. (See printed message.)',
                -4: 'Repeated error test failures. (Check all input.)',
                -5: 'Repeated convergence failures. (Perhaps bad'
                    ' Jacobian supplied or wrong choice of MF or tolerances.)',
                -6: 'Error weight became zero during problem. (Solution'
                    ' component i vanished, and ATOL or ATOL(i) = 0.)'
                }
    supports_run_relax = 1
    supports_step = 1
    active_global_handle = 0

    def __init__(self,
                 method='adams',
                 with_jacobian=False,
                 rtol=1e-6, atol=1e-12,
                 lband=None, uband=None,
                 order=12,
                 nsteps=500,
                 max_step=0.0,  # corresponds to infinite
                 min_step=0.0,
                 first_step=0.0,  # determined by solver
                 ):

        if re.match(method, r'adams', re.I):
            self.meth = 1
        elif re.match(method, r'bdf', re.I):
            self.meth = 2
        else:
            raise ValueError('Unknown integration method %s' % method)
        self.with_jacobian = with_jacobian
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband

        self.order = order
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.success = 1

        self.initialized = False

    def _determine_mf_and_set_bands(self, has_jac):
        """
        Determine the `MF` parameter (Method Flag) for the Fortran subroutine `dvode`.

        In the Fortran code, the legal values of `MF` are:
            10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            -11, -12, -14, -15, -21, -22, -24, -25
        but this Python wrapper does not use negative values.

        Returns

            mf  = 10*self.meth + miter

        self.meth is the linear multistep method:
            self.meth == 1:  method="adams"
            self.meth == 2:  method="bdf"

        miter is the correction iteration method:
            miter == 0:  Functional iteraton; no Jacobian involved.
            miter == 1:  Chord iteration with user-supplied full Jacobian.
            miter == 2:  Chord iteration with internally computed full Jacobian.
            miter == 3:  Chord iteration with internally computed diagonal Jacobian.
            miter == 4:  Chord iteration with user-supplied banded Jacobian.
            miter == 5:  Chord iteration with internally computed banded Jacobian.

        Side effects: If either self.mu or self.ml is not None and the other is None,
        then the one that is None is set to 0.
        """

        jac_is_banded = self.mu is not None or self.ml is not None
        if jac_is_banded:
            if self.mu is None:
                self.mu = 0
            if self.ml is None:
                self.ml = 0

        # has_jac is True if the user provided a Jacobian function.
        if has_jac:
            if jac_is_banded:
                miter = 4
            else:
                miter = 1
        else:
            if jac_is_banded:
                if self.ml == self.mu == 0:
                    miter = 3  # Chord iteration with internal diagonal Jacobian.
                else:
                    miter = 5  # Chord iteration with internal banded Jacobian.
            else:
                # self.with_jacobian is set by the user in the call to ode.set_integrator.
                if self.with_jacobian:
                    miter = 2  # Chord iteration with internal full Jacobian.
                else:
                    miter = 0  # Functional iteraton; no Jacobian involved.

        mf = 10 * self.meth + miter
        return mf

    def reset(self, n, has_jac):
        mf = self._determine_mf_and_set_bands(has_jac)

        if mf == 10:
            lrw = 20 + 16 * n
        elif mf in [11, 12]:
            lrw = 22 + 16 * n + 2 * n * n
        elif mf == 13:
            lrw = 22 + 17 * n
        elif mf in [14, 15]:
            lrw = 22 + 18 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf == 20:
            lrw = 20 + 9 * n
        elif mf in [21, 22]:
            lrw = 22 + 9 * n + 2 * n * n
        elif mf == 23:
            lrw = 22 + 10 * n
        elif mf in [24, 25]:
            lrw = 22 + 11 * n + (3 * self.ml + 2 * self.mu) * n
        else:
            raise ValueError('Unexpected mf=%s' % mf)

        if mf % 10 in [0, 3]:
            liw = 30
        else:
            liw = 30 + n

        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork

        iwork = zeros((liw,), _vode_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2  # mxhnil
        self.iwork = iwork

        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.rwork, self.iwork, mf]
        self.success = 1
        self.initialized = False

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()
        else:
            self.initialized = True
            self.acquire_new_handle()

        if self.ml is not None and self.ml > 0:
            # Banded Jacobian. Wrap the user-provided function with one
            # that pads the Jacobian array with the extra `self.ml` rows
            # required by the f2py-generated wrapper.
            jac = _vode_banded_jac_wrapper(jac, self.ml, jac_params)

        args = ((f, jac, y0, t0, t1) + tuple(self.call_args) +
                (f_params, jac_params))
        y1, t, istate = self.runner(*args)
        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = 'Unexpected istate={:d}'.format(istate)
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
                          self.messages.get(istate, unexpected_istate_msg)))
            self.success = 0
        else:
            self.call_args[3] = 2  # upgrade istate from 1 to 2
            self.istate = 2
        return y1, t

    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

    def run_relax(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 3
        r = self.run(*args)
        self.call_args[2] = itask
        return r


if vode.runner is not None:
    IntegratorBase.integrator_classes.append(vode)


class zvode(vode):
    runner = getattr(_vode, 'zvode', None)

    supports_run_relax = 1
    supports_step = 1
    scalar = complex
    active_global_handle = 0

    def reset(self, n, has_jac):
        mf = self._determine_mf_and_set_bands(has_jac)

        if mf in (10,):
            lzw = 15 * n
        elif mf in (11, 12):
            lzw = 15 * n + 2 * n ** 2
        elif mf in (-11, -12):
            lzw = 15 * n + n ** 2
        elif mf in (13,):
            lzw = 16 * n
        elif mf in (14, 15):
            lzw = 17 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf in (-14, -15):
            lzw = 16 * n + (2 * self.ml + self.mu) * n
        elif mf in (20,):
            lzw = 8 * n
        elif mf in (21, 22):
            lzw = 8 * n + 2 * n ** 2
        elif mf in (-21, -22):
            lzw = 8 * n + n ** 2
        elif mf in (23,):
            lzw = 9 * n
        elif mf in (24, 25):
            lzw = 10 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf in (-24, -25):
            lzw = 9 * n + (2 * self.ml + self.mu) * n

        lrw = 20 + n

        if mf % 10 in (0, 3):
            liw = 30
        else:
            liw = 30 + n

        zwork = zeros((lzw,), complex)
        self.zwork = zwork

        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork

        iwork = zeros((liw,), _vode_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2  # mxhnil
        self.iwork = iwork

        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.zwork, self.rwork, self.iwork, mf]
        self.success = 1
        self.initialized = False


if zvode.runner is not None:
    IntegratorBase.integrator_classes.append(zvode)


class dopri5(IntegratorBase):
    runner = getattr(_dop, 'dopri5', None)
    name = 'dopri5'
    supports_solout = True

    messages = {1: 'computation successful',
                2: 'computation successful (interrupted by solout)',
                -1: 'input is not consistent',
                -2: 'larger nsteps is needed',
                -3: 'step size becomes too small',
                -4: 'problem is probably stiff (interrupted)',
                }

    def __init__(self,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 max_step=0.0,
                 first_step=0.0,  # determined by solver
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 beta=0.0,
                 method=None,
                 verbosity=-1,  # no messages if negative
                 ):
        self.rtol = rtol
        self.atol = atol
        self.nsteps = nsteps
        self.max_step = max_step
        self.first_step = first_step
        self.safety = safety
        self.ifactor = ifactor
        self.dfactor = dfactor
        self.beta = beta
        self.verbosity = verbosity
        self.success = 1
        self.set_solout(None)

    def set_solout(self, solout, complex=False):
        self.solout = solout
        self.solout_cmplx = complex
        if solout is None:
            self.iout = 0
        else:
            self.iout = 1

    def reset(self, n, has_jac):
        work = zeros((8 * n + 21,), float)
        work[1] = self.safety
        work[2] = self.dfactor
        work[3] = self.ifactor
        work[4] = self.beta
        work[5] = self.max_step
        work[6] = self.first_step
        self.work = work
        iwork = zeros((21,), _dop_int_dtype)
        iwork[0] = self.nsteps
        iwork[2] = self.verbosity
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.iout, self.work, self.iwork]
        self.success = 1

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        x, y, iwork, istate = self.runner(*((f, t0, y0, t1) +
                                          tuple(self.call_args) + (f_params,)))
        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = 'Unexpected istate={:d}'.format(istate)
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
                          self.messages.get(istate, unexpected_istate_msg)))
            self.success = 0
        return y, x

    def _solout(self, nr, xold, x, y, nd, icomp, con):
        if self.solout is not None:
            if self.solout_cmplx:
                y = y[::2] + 1j * y[1::2]
            return self.solout(x, y)
        else:
            return 1


if dopri5.runner is not None:
    IntegratorBase.integrator_classes.append(dopri5)


class dop853(dopri5):
    runner = getattr(_dop, 'dop853', None)
    name = 'dop853'

    def __init__(self,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 max_step=0.0,
                 first_step=0.0,  # determined by solver
                 safety=0.9,
                 ifactor=6.0,
                 dfactor=0.3,
                 beta=0.0,
                 method=None,
                 verbosity=-1,  # no messages if negative
                 ):
        super().__init__(rtol, atol, nsteps, max_step, first_step, safety,
                         ifactor, dfactor, beta, method, verbosity)

    def reset(self, n, has_jac):
        work = zeros((11 * n + 21,), float)
        work[1] = self.safety
        work[2] = self.dfactor
        work[3] = self.ifactor
        work[4] = self.beta
        work[5] = self.max_step
        work[6] = self.first_step
        self.work = work
        iwork = zeros((21,), _dop_int_dtype)
        iwork[0] = self.nsteps
        iwork[2] = self.verbosity
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.iout, self.work, self.iwork]
        self.success = 1


if dop853.runner is not None:
    IntegratorBase.integrator_classes.append(dop853)


class lsoda(IntegratorBase):
    runner = getattr(_lsoda, 'lsoda', None)
    active_global_handle = 0

    messages = {
        2: "Integration successful.",
        -1: "Excess work done on this call (perhaps wrong Dfun type).",
        -2: "Excess accuracy requested (tolerances too small).",
        -3: "Illegal input detected (internal error).",
        -4: "Repeated error test failures (internal error).",
        -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
        -6: "Error weight became zero during problem.",
        -7: "Internal workspace insufficient to finish (internal error)."
    }

    def __init__(self,
                 with_jacobian=False,
                 rtol=1e-6, atol=1e-12,
                 lband=None, uband=None,
                 nsteps=500,
                 max_step=0.0,  # corresponds to infinite
                 min_step=0.0,
                 first_step=0.0,  # determined by solver
                 ixpr=0,
                 max_hnil=0,
                 max_order_ns=12,
                 max_order_s=5,
                 method=None
                 ):

        self.with_jacobian = with_jacobian
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband

        self.max_order_ns = max_order_ns
        self.max_order_s = max_order_s
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.ixpr = ixpr
        self.max_hnil = max_hnil
        self.success = 1

        self.initialized = False

    def reset(self, n, has_jac):
        # Calculate parameters for Fortran subroutine dvode.
        if has_jac:
            if self.mu is None and self.ml is None:
                jt = 1
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 4
        else:
            if self.mu is None and self.ml is None:
                jt = 2
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 5
        lrn = 20 + (self.max_order_ns + 4) * n
        if jt in [1, 2]:
            lrs = 22 + (self.max_order_s + 4) * n + n * n
        elif jt in [4, 5]:
            lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
        else:
            raise ValueError('Unexpected jt=%s' % jt)
        lrw = max(lrn, lrs)
        liw = 20 + n
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), _lsoda_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.ixpr
        iwork[5] = self.nsteps
        iwork[6] = self.max_hnil
        iwork[7] = self.max_order_ns
        iwork[8] = self.max_order_s
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.rwork, self.iwork, jt]
        self.success = 1
        self.initialized = False

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()
        else:
            self.initialized = True
            self.acquire_new_handle()
        args = [f, y0, t0, t1] + self.call_args[:-1] + \
               [jac, self.call_args[-1], f_params, 0, jac_params]
        y1, t, istate = self.runner(*args)
        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = 'Unexpected istate={:d}'.format(istate)
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
                          self.messages.get(istate, unexpected_istate_msg)))
            self.success = 0
        else:
            self.call_args[3] = 2  # upgrade istate from 1 to 2
            self.istate = 2
        return y1, t

    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

    def run_relax(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 3
        r = self.run(*args)
        self.call_args[2] = itask
        return r


if lsoda.runner:
    IntegratorBase.integrator_classes.append(lsoda)
