import numpy as np
from scipy.integrate import ode
from .common import validate_tol, validate_first_step, warn_extraneous
from .base import OdeSolver, DenseOutput


class LSODA(OdeSolver):
    """Adams/BDF method with automatic stiffness detection and switching.

    This is a wrapper to the Fortran solver from ODEPACK [1]_. It switches
    automatically between the nonstiff Adams method and the stiff BDF method.
    The method was originally detailed in [2]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below). The
        vectorized implementation allows a faster approximation of the Jacobian
        by finite differences (required for this solver).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    min_step : float, optional
        Minimum allowed step size. Default is 0.0, i.e., the step size is not
        bounded and determined solely by the solver.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : None or callable, optional
        Jacobian matrix of the right-hand side of the system with respect to
        ``y``. The Jacobian matrix has shape (n, n) and its element (i, j) is
        equal to ``d f_i / d y_j``. The function will be called as
        ``jac(t, y)``. If None (default), the Jacobian will be
        approximated by finite differences. It is generally recommended to
        provide the Jacobian rather than relying on a finite-difference
        approximation.
    lband, uband : int or None
        Parameters defining the bandwidth of the Jacobian,
        i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``. Setting
        these requires your jac routine to return the Jacobian in the packed format:
        the returned array must have ``n`` columns and ``uband + lband + 1``
        rows in which Jacobian diagonals are written. Specifically
        ``jac_packed[uband + i - j , j] = jac[i, j]``. The same format is used
        in `scipy.linalg.solve_banded` (check for an illustration).
        These parameters can be also used with ``jac=None`` to reduce the
        number of Jacobian elements estimated by finite differences.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. A vectorized
        implementation offers no advantages for this solver. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.

    References
    ----------
    .. [1] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [2] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    """
    def __init__(self, fun, t0, y0, t_bound, first_step=None, min_step=0.0,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, jac=None, lband=None,
                 uband=None, vectorized=False, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)

        if first_step is None:
            first_step = 0  # LSODA value for automatic selection.
        else:
            first_step = validate_first_step(first_step, t0, t_bound)

        first_step *= self.direction

        if max_step == np.inf:
            max_step = 0  # LSODA value for infinity.
        elif max_step <= 0:
            raise ValueError("`max_step` must be positive.")

        if min_step < 0:
            raise ValueError("`min_step` must be nonnegative.")

        rtol, atol = validate_tol(rtol, atol, self.n)

        solver = ode(self.fun, jac)
        solver.set_integrator('lsoda', rtol=rtol, atol=atol, max_step=max_step,
                              min_step=min_step, first_step=first_step,
                              lband=lband, uband=uband)
        solver.set_initial_value(y0, t0)

        # Inject t_bound into rwork array as needed for itask=5.
        solver._integrator.rwork[0] = self.t_bound
        solver._integrator.call_args[4] = solver._integrator.rwork

        self._lsoda_solver = solver

    def _step_impl(self):
        solver = self._lsoda_solver
        integrator = solver._integrator

        # From lsoda.step and lsoda.integrate itask=5 means take a single
        # step and do not go past t_bound.
        itask = integrator.call_args[2]
        integrator.call_args[2] = 5
        solver._y, solver.t = integrator.run(
            solver.f, solver.jac or (lambda: None), solver._y, solver.t,
            self.t_bound, solver.f_params, solver.jac_params)
        integrator.call_args[2] = itask

        if solver.successful():
            self.t = solver.t
            self.y = solver._y
            # From LSODA Fortran source njev is equal to nlu.
            self.njev = integrator.iwork[12]
            self.nlu = integrator.iwork[12]
            return True, None
        else:
            return False, 'Unexpected istate in LSODA.'

    def _dense_output_impl(self):
        iwork = self._lsoda_solver._integrator.iwork
        rwork = self._lsoda_solver._integrator.rwork

        order = iwork[14]
        h = rwork[11]
        yh = np.reshape(rwork[20:20 + (order + 1) * self.n],
                        (self.n, order + 1), order='F').copy()

        return LsodaDenseOutput(self.t_old, self.t, h, order, yh)


class LsodaDenseOutput(DenseOutput):
    def __init__(self, t_old, t, h, order, yh):
        super().__init__(t_old, t)
        self.h = h
        self.yh = yh
        self.p = np.arange(order + 1)

    def _call_impl(self, t):
        if t.ndim == 0:
            x = ((t - self.t) / self.h) ** self.p
        else:
            x = ((t - self.t) / self.h) ** self.p[:, None]

        return np.dot(self.yh, x)
