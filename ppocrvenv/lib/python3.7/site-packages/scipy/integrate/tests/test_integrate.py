# Authors: Nils Wagner, Ed Schofield, Pauli Virtanen, John Travers
"""
Tests for numerical integration.
"""
import sys
import platform

import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
                   allclose)

from numpy.testing import (
    assert_, assert_array_almost_equal,
    assert_allclose, assert_array_equal, assert_equal, assert_warns)
import pytest
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode

IS_MACOS_ARM64 =  sys.platform == 'darwin' and platform.machine() == 'arm64'


#------------------------------------------------------------------------------
# Test ODE integrators
#------------------------------------------------------------------------------


class TestOdeint:
    # Check integrate.odeint

    def _do_problem(self, problem):
        t = arange(0.0, problem.stop_t, 0.05)

        # Basic case
        z, infodict = odeint(problem.f, problem.z0, t, full_output=True)
        assert_(problem.verify(z, t))

        # Use tfirst=True
        z, infodict = odeint(lambda t, y: problem.f(y, t), problem.z0, t,
                             full_output=True, tfirst=True)
        assert_(problem.verify(z, t))

        if hasattr(problem, 'jac'):
            # Use Dfun
            z, infodict = odeint(problem.f, problem.z0, t, Dfun=problem.jac,
                                 full_output=True)
            assert_(problem.verify(z, t))

            # Use Dfun and tfirst=True
            z, infodict = odeint(lambda t, y: problem.f(y, t), problem.z0, t,
                                 Dfun=lambda t, y: problem.jac(y, t),
                                 full_output=True, tfirst=True)
            assert_(problem.verify(z, t))

    def test_odeint(self):
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem)


class TestODEClass:

    ode_class = None   # Set in subclass.

    def _do_problem(self, problem, integrator, method='adams'):

        # ode has callback arguments in different order than odeint
        f = lambda t, z: problem.f(z, t)
        jac = None
        if hasattr(problem, 'jac'):
            jac = lambda t, z: problem.jac(z, t)

        integrator_params = {}
        if problem.lband is not None or problem.uband is not None:
            integrator_params['uband'] = problem.uband
            integrator_params['lband'] = problem.lband

        ig = self.ode_class(f, jac)
        ig.set_integrator(integrator,
                          atol=problem.atol/10,
                          rtol=problem.rtol/10,
                          method=method,
                          **integrator_params)

        ig.set_initial_value(problem.z0, t=0.0)
        z = ig.integrate(problem.stop_t)

        assert_array_equal(z, ig.y)
        assert_(ig.successful(), (problem, method))
        assert_(ig.get_return_code() > 0, (problem, method))
        assert_(problem.verify(array([z]), problem.stop_t), (problem, method))


class TestOde(TestODEClass):

    ode_class = ode

    def test_vode(self):
        # Check the vode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if not problem.stiff:
                self._do_problem(problem, 'vode', 'adams')
            self._do_problem(problem, 'vode', 'bdf')

    @pytest.mark.skipif(IS_MACOS_ARM64, reason="zvode failing, see gh-15077")
    def test_zvode(self):
        # Check the zvode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                self._do_problem(problem, 'zvode', 'adams')
            self._do_problem(problem, 'zvode', 'bdf')

    def test_lsoda(self):
        # Check the lsoda solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        # Check the dopri5 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        # Check the dop853 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.cmplx:
                continue
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dop853')

    def test_concurrent_fail(self):
        for sol in ('vode', 'zvode', 'lsoda'):
            f = lambda t, y: 1.0

            r = ode(f).set_integrator(sol)
            r.set_initial_value(0, 0)

            r2 = ode(f).set_integrator(sol)
            r2.set_initial_value(0, 0)

            r.integrate(r.t + 0.1)
            r2.integrate(r2.t + 0.1)

            assert_raises(RuntimeError, r.integrate, r.t + 0.1)

    @pytest.mark.skipif(IS_MACOS_ARM64, reason="zvode failing, see gh-15077")
    def test_concurrent_ok(self):
        f = lambda t, y: 1.0

        for k in range(3):
            for sol in ('vode', 'zvode', 'lsoda', 'dopri5', 'dop853'):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r2.integrate(r2.t + 0.1)

                assert_allclose(r.y, 0.1)
                assert_allclose(r2.y, 0.2)

            for sol in ('dopri5', 'dop853'):
                r = ode(f).set_integrator(sol)
                r.set_initial_value(0, 0)

                r2 = ode(f).set_integrator(sol)
                r2.set_initial_value(0, 0)

                r.integrate(r.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)
                r.integrate(r.t + 0.1)
                r2.integrate(r2.t + 0.1)

                assert_allclose(r.y, 0.3)
                assert_allclose(r2.y, 0.2)


class TestComplexOde(TestODEClass):

    ode_class = complex_ode

    def test_vode(self):
        # Check the vode solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if not problem.stiff:
                self._do_problem(problem, 'vode', 'adams')
            else:
                self._do_problem(problem, 'vode', 'bdf')

    def test_lsoda(self):
        # Check the lsoda solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            self._do_problem(problem, 'lsoda')

    def test_dopri5(self):
        # Check the dopri5 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dopri5')

    def test_dop853(self):
        # Check the dop853 solver
        for problem_cls in PROBLEMS:
            problem = problem_cls()
            if problem.stiff:
                continue
            if hasattr(problem, 'jac'):
                continue
            self._do_problem(problem, 'dop853')


class TestSolout:
    # Check integrate.ode correctly handles solout for dopri5 and dop853
    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)

    def test_solout(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_test(integrator)

    def _run_solout_after_initial_test(self, integrator):
        # Check if solout works even if it is set after the initial value.
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_initial_value(y0, t0)
        ig.set_solout(solout)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)

    def test_solout_after_initial(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_after_initial_test(integrator)

    def _run_solout_break_test(self, integrator):
        # Check correct usage of stopping via solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 10.0
        y0 = [1.0, 2.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())
            if t > tend/2.0:
                return -1

        def rhs(t, y):
            return [y[0] + y[1], -y[1]**2]

        ig = ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_(ts[-1] > tend/2.0)
        assert_(ts[-1] < tend)

    def test_solout_break(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_break_test(integrator)


class TestComplexSolout:
    # Check integrate.ode correctly handles solout for dopri5 and dop853
    def _run_solout_test(self, integrator):
        # Check correct usage of solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 20.0
        y0 = [0.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())

        def rhs(t, y):
            return [1.0/(t - 10.0 - 1j)]

        ig = complex_ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)

    def test_solout(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_test(integrator)

    def _run_solout_break_test(self, integrator):
        # Check correct usage of stopping via solout
        ts = []
        ys = []
        t0 = 0.0
        tend = 20.0
        y0 = [0.0]

        def solout(t, y):
            ts.append(t)
            ys.append(y.copy())
            if t > tend/2.0:
                return -1

        def rhs(t, y):
            return [1.0/(t - 10.0 - 1j)]

        ig = complex_ode(rhs).set_integrator(integrator)
        ig.set_solout(solout)
        ig.set_initial_value(y0, t0)
        ret = ig.integrate(tend)
        assert_array_equal(ys[0], y0)
        assert_array_equal(ys[-1], ret)
        assert_equal(ts[0], t0)
        assert_(ts[-1] > tend/2.0)
        assert_(ts[-1] < tend)

    def test_solout_break(self):
        for integrator in ('dopri5', 'dop853'):
            self._run_solout_break_test(integrator)


#------------------------------------------------------------------------------
# Test problems
#------------------------------------------------------------------------------


class ODE:
    """
    ODE problem
    """
    stiff = False
    cmplx = False
    stop_t = 1
    z0 = []

    lband = None
    uband = None

    atol = 1e-6
    rtol = 1e-5


class SimpleOscillator(ODE):
    r"""
    Free vibration of a simple oscillator::
        m \ddot{u} + k u = 0, u(0) = u_0 \dot{u}(0) \dot{u}_0
    Solution::
        u(t) = u_0*cos(sqrt(k/m)*t)+\dot{u}_0*sin(sqrt(k/m)*t)/sqrt(k/m)
    """
    stop_t = 1 + 0.09
    z0 = array([1.0, 0.1], float)

    k = 4.0
    m = 1.0

    def f(self, z, t):
        tmp = zeros((2, 2), float)
        tmp[0, 1] = 1.0
        tmp[1, 0] = -self.k / self.m
        return dot(tmp, z)

    def verify(self, zs, t):
        omega = sqrt(self.k / self.m)
        u = self.z0[0]*cos(omega*t) + self.z0[1]*sin(omega*t)/omega
        return allclose(u, zs[:, 0], atol=self.atol, rtol=self.rtol)


class ComplexExp(ODE):
    r"""The equation :lm:`\dot u = i u`"""
    stop_t = 1.23*pi
    z0 = exp([1j, 2j, 3j, 4j, 5j])
    cmplx = True

    def f(self, z, t):
        return 1j*z

    def jac(self, z, t):
        return 1j*eye(5)

    def verify(self, zs, t):
        u = self.z0 * exp(1j*t)
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


class Pi(ODE):
    r"""Integrate 1/(t + 1j) from t=-10 to t=10"""
    stop_t = 20
    z0 = [0]
    cmplx = True

    def f(self, z, t):
        return array([1./(t - 10 + 1j)])

    def verify(self, zs, t):
        u = -2j * np.arctan(10)
        return allclose(u, zs[-1, :], atol=self.atol, rtol=self.rtol)


class CoupledDecay(ODE):
    r"""
    3 coupled decays suited for banded treatment
    (banded mode makes it necessary when N>>3)
    """

    stiff = True
    stop_t = 0.5
    z0 = [5.0, 7.0, 13.0]
    lband = 1
    uband = 0

    lmbd = [0.17, 0.23, 0.29]  # fictitious decay constants

    def f(self, z, t):
        lmbd = self.lmbd
        return np.array([-lmbd[0]*z[0],
                         -lmbd[1]*z[1] + lmbd[0]*z[0],
                         -lmbd[2]*z[2] + lmbd[1]*z[1]])

    def jac(self, z, t):
        # The full Jacobian is
        #
        #    [-lmbd[0]      0         0   ]
        #    [ lmbd[0]  -lmbd[1]      0   ]
        #    [    0      lmbd[1]  -lmbd[2]]
        #
        # The lower and upper bandwidths are lband=1 and uband=0, resp.
        # The representation of this array in packed format is
        #
        #    [-lmbd[0]  -lmbd[1]  -lmbd[2]]
        #    [ lmbd[0]   lmbd[1]      0   ]

        lmbd = self.lmbd
        j = np.zeros((self.lband + self.uband + 1, 3), order='F')

        def set_j(ri, ci, val):
            j[self.uband + ri - ci, ci] = val
        set_j(0, 0, -lmbd[0])
        set_j(1, 0, lmbd[0])
        set_j(1, 1, -lmbd[1])
        set_j(2, 1, lmbd[1])
        set_j(2, 2, -lmbd[2])
        return j

    def verify(self, zs, t):
        # Formulae derived by hand
        lmbd = np.array(self.lmbd)
        d10 = lmbd[1] - lmbd[0]
        d21 = lmbd[2] - lmbd[1]
        d20 = lmbd[2] - lmbd[0]
        e0 = np.exp(-lmbd[0] * t)
        e1 = np.exp(-lmbd[1] * t)
        e2 = np.exp(-lmbd[2] * t)
        u = np.vstack((
            self.z0[0] * e0,
            self.z0[1] * e1 + self.z0[0] * lmbd[0] / d10 * (e0 - e1),
            self.z0[2] * e2 + self.z0[1] * lmbd[1] / d21 * (e1 - e2) +
            lmbd[1] * lmbd[0] * self.z0[0] / d10 *
            (1 / d20 * (e0 - e2) - 1 / d21 * (e1 - e2)))).transpose()
        return allclose(u, zs, atol=self.atol, rtol=self.rtol)


PROBLEMS = [SimpleOscillator, ComplexExp, Pi, CoupledDecay]

#------------------------------------------------------------------------------


def f(t, x):
    dxdt = [x[1], -x[0]]
    return dxdt


def jac(t, x):
    j = array([[0.0, 1.0],
               [-1.0, 0.0]])
    return j


def f1(t, x, omega):
    dxdt = [omega*x[1], -omega*x[0]]
    return dxdt


def jac1(t, x, omega):
    j = array([[0.0, omega],
               [-omega, 0.0]])
    return j


def f2(t, x, omega1, omega2):
    dxdt = [omega1*x[1], -omega2*x[0]]
    return dxdt


def jac2(t, x, omega1, omega2):
    j = array([[0.0, omega1],
               [-omega2, 0.0]])
    return j


def fv(t, x, omega):
    dxdt = [omega[0]*x[1], -omega[1]*x[0]]
    return dxdt


def jacv(t, x, omega):
    j = array([[0.0, omega[0]],
               [-omega[1], 0.0]])
    return j


class ODECheckParameterUse:
    """Call an ode-class solver with several cases of parameter use."""

    # solver_name must be set before tests can be run with this class.

    # Set these in subclasses.
    solver_name = ''
    solver_uses_jac = False

    def _get_solver(self, f, jac):
        solver = ode(f, jac)
        if self.solver_uses_jac:
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7,
                                  with_jacobian=self.solver_uses_jac)
        else:
            # XXX Shouldn't set_integrator *always* accept the keyword arg
            # 'with_jacobian', and perhaps raise an exception if it is set
            # to True if the solver can't actually use it?
            solver.set_integrator(self.solver_name, atol=1e-9, rtol=1e-7)
        return solver

    def _check_solver(self, solver):
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        solver.integrate(pi)
        assert_array_almost_equal(solver.y, [-1.0, 0.0])

    @pytest.mark.skipif(IS_MACOS_ARM64, reason="zvode failing, see gh-15077")
    def test_no_params(self):
        solver = self._get_solver(f, jac)
        self._check_solver(solver)

    @pytest.mark.skipif(IS_MACOS_ARM64, reason="zvode failing, see gh-15077")
    def test_one_scalar_param(self):
        solver = self._get_solver(f1, jac1)
        omega = 1.0
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    @pytest.mark.skipif(IS_MACOS_ARM64, reason="zvode failing, see gh-15077")
    def test_two_scalar_params(self):
        solver = self._get_solver(f2, jac2)
        omega1 = 1.0
        omega2 = 1.0
        solver.set_f_params(omega1, omega2)
        if self.solver_uses_jac:
            solver.set_jac_params(omega1, omega2)
        self._check_solver(solver)

    @pytest.mark.skipif(IS_MACOS_ARM64, reason="zvode failing, see gh-15077")
    def test_vector_param(self):
        solver = self._get_solver(fv, jacv)
        omega = [1.0, 1.0]
        solver.set_f_params(omega)
        if self.solver_uses_jac:
            solver.set_jac_params(omega)
        self._check_solver(solver)

    def test_warns_on_failure(self):
        # Set nsteps small to ensure failure
        solver = self._get_solver(f, jac)
        solver.set_integrator(self.solver_name, nsteps=1)
        ic = [1.0, 0.0]
        solver.set_initial_value(ic, 0.0)
        assert_warns(UserWarning, solver.integrate, pi)


class TestDOPRI5CheckParameterUse(ODECheckParameterUse):
    solver_name = 'dopri5'
    solver_uses_jac = False


class TestDOP853CheckParameterUse(ODECheckParameterUse):
    solver_name = 'dop853'
    solver_uses_jac = False


class TestVODECheckParameterUse(ODECheckParameterUse):
    solver_name = 'vode'
    solver_uses_jac = True


class TestZVODECheckParameterUse(ODECheckParameterUse):
    solver_name = 'zvode'
    solver_uses_jac = True


class TestLSODACheckParameterUse(ODECheckParameterUse):
    solver_name = 'lsoda'
    solver_uses_jac = True


def test_odeint_trivial_time():
    # Test that odeint succeeds when given a single time point
    # and full_output=True.  This is a regression test for gh-4282.
    y0 = 1
    t = [0]
    y, info = odeint(lambda y, t: -y, y0, t, full_output=True)
    assert_array_equal(y, np.array([[y0]]))


def test_odeint_banded_jacobian():
    # Test the use of the `Dfun`, `ml` and `mu` options of odeint.

    def func(y, t, c):
        return c.dot(y)

    def jac(y, t, c):
        return c

    def jac_transpose(y, t, c):
        return c.T.copy(order='C')

    def bjac_rows(y, t, c):
        jac = np.row_stack((np.r_[0, np.diag(c, 1)],
                            np.diag(c),
                            np.r_[np.diag(c, -1), 0],
                            np.r_[np.diag(c, -2), 0, 0]))
        return jac

    def bjac_cols(y, t, c):
        return bjac_rows(y, t, c).T.copy(order='C')

    c = array([[-205, 0.01, 0.00, 0.0],
               [0.1, -2.50, 0.02, 0.0],
               [1e-3, 0.01, -2.0, 0.01],
               [0.00, 0.00, 0.1, -1.0]])

    y0 = np.ones(4)
    t = np.array([0, 5, 10, 100])

    # Use the full Jacobian.
    sol1, info1 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=jac)

    # Use the transposed full Jacobian, with col_deriv=True.
    sol2, info2 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=jac_transpose, col_deriv=True)

    # Use the banded Jacobian.
    sol3, info3 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=bjac_rows, ml=2, mu=1)

    # Use the transposed banded Jacobian, with col_deriv=True.
    sol4, info4 = odeint(func, y0, t, args=(c,), full_output=True,
                         atol=1e-13, rtol=1e-11, mxstep=10000,
                         Dfun=bjac_cols, ml=2, mu=1, col_deriv=True)

    assert_allclose(sol1, sol2, err_msg="sol1 != sol2")
    assert_allclose(sol1, sol3, atol=1e-12, err_msg="sol1 != sol3")
    assert_allclose(sol3, sol4, err_msg="sol3 != sol4")

    # Verify that the number of jacobian evaluations was the same for the
    # calls of odeint with a full jacobian and with a banded jacobian. This is
    # a regression test--there was a bug in the handling of banded jacobians
    # that resulted in an incorrect jacobian matrix being passed to the LSODA
    # code.  That would cause errors or excessive jacobian evaluations.
    assert_array_equal(info1['nje'], info2['nje'])
    assert_array_equal(info3['nje'], info4['nje'])

    # Test the use of tfirst
    sol1ty, info1ty = odeint(lambda t, y, c: func(y, t, c), y0, t, args=(c,),
                             full_output=True, atol=1e-13, rtol=1e-11,
                             mxstep=10000,
                             Dfun=lambda t, y, c: jac(y, t, c), tfirst=True)
    # The code should execute the exact same sequence of floating point
    # calculations, so these should be exactly equal. We'll be safe and use
    # a small tolerance.
    assert_allclose(sol1, sol1ty, rtol=1e-12, err_msg="sol1 != sol1ty")


def test_odeint_errors():
    def sys1d(x, t):
        return -100*x

    def bad1(x, t):
        return 1.0/0

    def bad2(x, t):
        return "foo"

    def bad_jac1(x, t):
        return 1.0/0

    def bad_jac2(x, t):
        return [["foo"]]

    def sys2d(x, t):
        return [-100*x[0], -0.1*x[1]]

    def sys2d_bad_jac(x, t):
        return [[1.0/0, 0], [0, -0.1]]

    assert_raises(ZeroDivisionError, odeint, bad1, 1.0, [0, 1])
    assert_raises(ValueError, odeint, bad2, 1.0, [0, 1])

    assert_raises(ZeroDivisionError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac1)
    assert_raises(ValueError, odeint, sys1d, 1.0, [0, 1], Dfun=bad_jac2)

    assert_raises(ZeroDivisionError, odeint, sys2d, [1.0, 1.0], [0, 1],
                  Dfun=sys2d_bad_jac)


def test_odeint_bad_shapes():
    # Tests of some errors that can occur with odeint.

    def badrhs(x, t):
        return [1, -1]

    def sys1(x, t):
        return -100*x

    def badjac(x, t):
        return [[0, 0, 0]]

    # y0 must be at most 1-d.
    bad_y0 = [[0, 0], [0, 0]]
    assert_raises(ValueError, odeint, sys1, bad_y0, [0, 1])

    # t must be at most 1-d.
    bad_t = [[0, 1], [2, 3]]
    assert_raises(ValueError, odeint, sys1, [10.0], bad_t)

    # y0 is 10, but badrhs(x, t) returns [1, -1].
    assert_raises(RuntimeError, odeint, badrhs, 10, [0, 1])

    # shape of array returned by badjac(x, t) is not correct.
    assert_raises(RuntimeError, odeint, sys1, [10, 10], [0, 1], Dfun=badjac)


def test_repeated_t_values():
    """Regression test for gh-8217."""

    def func(x, t):
        return -0.25*x

    t = np.zeros(10)
    sol = odeint(func, [1.], t)
    assert_array_equal(sol, np.ones((len(t), 1)))

    tau = 4*np.log(2)
    t = [0]*9 + [tau, 2*tau, 2*tau, 3*tau]
    sol = odeint(func, [1, 2], t, rtol=1e-12, atol=1e-12)
    expected_sol = np.array([[1.0, 2.0]]*9 +
                            [[0.5, 1.0],
                             [0.25, 0.5],
                             [0.25, 0.5],
                             [0.125, 0.25]])
    assert_allclose(sol, expected_sol)

    # Edge case: empty t sequence.
    sol = odeint(func, [1.], [])
    assert_array_equal(sol, np.array([], dtype=np.float64).reshape((0, 1)))

    # t values are not monotonic.
    assert_raises(ValueError, odeint, func, [1.], [0, 1, 0.5, 0])
    assert_raises(ValueError, odeint, func, [1, 2, 3], [0, -1, -2, 3])
