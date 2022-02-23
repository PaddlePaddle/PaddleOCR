"""
Unit tests for optimization routines from minpack.py.
"""
import warnings

from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose,
                           assert_warns, suppress_warnings)
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool

from scipy import optimize
from scipy.special import lambertw
from scipy.optimize.minpack import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning


class ReturnShape:
    """This class exists to create a callable that does not have a '__name__' attribute.

    __init__ takes the argument 'shape', which should be a tuple of ints. When an instance
    is called with a single argument 'x', it returns numpy.ones(shape).
    """

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        return np.ones(self.shape)


def dummy_func(x, shape):
    """A function that returns an array of ones of the given shape.
    `x` is ignored.
    """
    return np.ones(shape)


def sequence_parallel(fs):
    with ThreadPool(len(fs)) as pool:
        return pool.map(lambda f: f(), fs)


# Function and Jacobian for tests of solvers for systems of nonlinear
# equations


def pressure_network(flow_rates, Qtot, k):
    """Evaluate non-linear equation system representing
    the pressures and flows in a system of n parallel pipes::

        f_i = P_i - P_0, for i = 1..n
        f_0 = sum(Q_i) - Qtot

    where Q_i is the flow rate in pipe i and P_i the pressure in that pipe.
    Pressure is modeled as a P=kQ**2 where k is a valve coefficient and
    Q is the flow rate.

    Parameters
    ----------
    flow_rates : float
        A 1-D array of n flow rates [kg/s].
    k : float
        A 1-D array of n valve coefficients [1/kg m].
    Qtot : float
        A scalar, the total input flow rate [kg/s].

    Returns
    -------
    F : float
        A 1-D array, F[i] == f_i.

    """
    P = k * flow_rates**2
    F = np.hstack((P[1:] - P[0], flow_rates.sum() - Qtot))
    return F


def pressure_network_jacobian(flow_rates, Qtot, k):
    """Return the jacobian of the equation system F(flow_rates)
    computed by `pressure_network` with respect to
    *flow_rates*. See `pressure_network` for the detailed
    description of parrameters.

    Returns
    -------
    jac : float
        *n* by *n* matrix ``df_i/dQ_i`` where ``n = len(flow_rates)``
        and *f_i* and *Q_i* are described in the doc for `pressure_network`
    """
    n = len(flow_rates)
    pdiff = np.diag(flow_rates[1:] * 2 * k[1:] - 2 * flow_rates[0] * k[0])

    jac = np.empty((n, n))
    jac[:n-1, :n-1] = pdiff * 0
    jac[:n-1, n-1] = 0
    jac[n-1, :] = np.ones(n)

    return jac


def pressure_network_fun_and_grad(flow_rates, Qtot, k):
    return (pressure_network(flow_rates, Qtot, k),
            pressure_network_jacobian(flow_rates, Qtot, k))


class TestFSolve:
    def test_pressure_network_no_gradient(self):
        # fsolve without gradient, equal pipes -> equal flows.
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows, info, ier, mesg = optimize.fsolve(
            pressure_network, initial_guess, args=(Qtot, k),
            full_output=True)
        assert_array_almost_equal(final_flows, np.ones(4))
        assert_(ier == 1, mesg)

    def test_pressure_network_with_gradient(self):
        # fsolve with gradient, equal pipes -> equal flows
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows = optimize.fsolve(
            pressure_network, initial_guess, args=(Qtot, k),
            fprime=pressure_network_jacobian)
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_wrong_shape_func_callable(self):
        func = ReturnShape(1)
        # x0 is a list of two elements, but func will return an array with
        # length 1, so this should result in a TypeError.
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.fsolve, func, x0)

    def test_wrong_shape_func_function(self):
        # x0 is a list of two elements, but func will return an array with
        # length 1, so this should result in a TypeError.
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.fsolve, dummy_func, x0, args=((1,),))

    def test_wrong_shape_fprime_callable(self):
        func = ReturnShape(1)
        deriv_func = ReturnShape((2,2))
        assert_raises(TypeError, optimize.fsolve, func, x0=[0,1], fprime=deriv_func)

    def test_wrong_shape_fprime_function(self):
        func = lambda x: dummy_func(x, (2,))
        deriv_func = lambda x: dummy_func(x, (3,3))
        assert_raises(TypeError, optimize.fsolve, func, x0=[0,1], fprime=deriv_func)

    def test_func_can_raise(self):
        def func(*args):
            raise ValueError('I raised')

        with assert_raises(ValueError, match='I raised'):
            optimize.fsolve(func, x0=[0])

    def test_Dfun_can_raise(self):
        func = lambda x: x - np.array([10])

        def deriv_func(*args):
            raise ValueError('I raised')

        with assert_raises(ValueError, match='I raised'):
            optimize.fsolve(func, x0=[0], fprime=deriv_func)

    def test_float32(self):
        func = lambda x: np.array([x[0] - 100, x[1] - 1000], dtype=np.float32)**2
        p = optimize.fsolve(func, np.array([1, 1], np.float32))
        assert_allclose(func(p), [0, 0], atol=1e-3)

    def test_reentrant_func(self):
        def func(*args):
            self.test_pressure_network_no_gradient()
            return pressure_network(*args)

        # fsolve without gradient, equal pipes -> equal flows.
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows, info, ier, mesg = optimize.fsolve(
            func, initial_guess, args=(Qtot, k),
            full_output=True)
        assert_array_almost_equal(final_flows, np.ones(4))
        assert_(ier == 1, mesg)

    def test_reentrant_Dfunc(self):
        def deriv_func(*args):
            self.test_pressure_network_with_gradient()
            return pressure_network_jacobian(*args)

        # fsolve with gradient, equal pipes -> equal flows
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows = optimize.fsolve(
            pressure_network, initial_guess, args=(Qtot, k),
            fprime=deriv_func)
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_concurrent_no_gradient(self):
        return sequence_parallel([self.test_pressure_network_no_gradient] * 10)

    def test_concurrent_with_gradient(self):
        return sequence_parallel([self.test_pressure_network_with_gradient] * 10)


class TestRootHybr:
    def test_pressure_network_no_gradient(self):
        # root/hybr without gradient, equal pipes -> equal flows
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows = optimize.root(pressure_network, initial_guess,
                                    method='hybr', args=(Qtot, k)).x
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_pressure_network_with_gradient(self):
        # root/hybr with gradient, equal pipes -> equal flows
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([[2., 0., 2., 0.]])
        final_flows = optimize.root(pressure_network, initial_guess,
                                    args=(Qtot, k), method='hybr',
                                    jac=pressure_network_jacobian).x
        assert_array_almost_equal(final_flows, np.ones(4))

    def test_pressure_network_with_gradient_combined(self):
        # root/hybr with gradient and function combined, equal pipes -> equal
        # flows
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows = optimize.root(pressure_network_fun_and_grad,
                                    initial_guess, args=(Qtot, k),
                                    method='hybr', jac=True).x
        assert_array_almost_equal(final_flows, np.ones(4))


class TestRootLM:
    def test_pressure_network_no_gradient(self):
        # root/lm without gradient, equal pipes -> equal flows
        k = np.full(4, 0.5)
        Qtot = 4
        initial_guess = array([2., 0., 2., 0.])
        final_flows = optimize.root(pressure_network, initial_guess,
                                    method='lm', args=(Qtot, k)).x
        assert_array_almost_equal(final_flows, np.ones(4))


class TestLeastSq:
    def setup_method(self):
        x = np.linspace(0, 10, 40)
        a,b,c = 3.1, 42, -304.2
        self.x = x
        self.abc = a,b,c
        y_true = a*x**2 + b*x + c
        np.random.seed(0)
        self.y_meas = y_true + 0.01*np.random.standard_normal(y_true.shape)

    def residuals(self, p, y, x):
        a,b,c = p
        err = y-(a*x**2 + b*x + c)
        return err

    def residuals_jacobian(self, _p, _y, x):
        return -np.vstack([x**2, x, np.ones_like(x)]).T

    def test_basic(self):
        p0 = array([0,0,0])
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x))
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # low precision due to random
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_basic_with_gradient(self):
        p0 = array([0,0,0])
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x),
                                  Dfun=self.residuals_jacobian)
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # low precision due to random
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_full_output(self):
        p0 = array([[0,0,0]])
        full_output = leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert_(ier in (1,2,3,4), 'solution not found: %s' % mesg)

    def test_input_untouched(self):
        p0 = array([0,0,0],dtype=float64)
        p0_copy = array(p0, copy=True)
        full_output = leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert_(ier in (1,2,3,4), 'solution not found: %s' % mesg)
        assert_array_equal(p0, p0_copy)

    def test_wrong_shape_func_callable(self):
        func = ReturnShape(1)
        # x0 is a list of two elements, but func will return an array with
        # length 1, so this should result in a TypeError.
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.leastsq, func, x0)

    def test_wrong_shape_func_function(self):
        # x0 is a list of two elements, but func will return an array with
        # length 1, so this should result in a TypeError.
        x0 = [1.5, 2.0]
        assert_raises(TypeError, optimize.leastsq, dummy_func, x0, args=((1,),))

    def test_wrong_shape_Dfun_callable(self):
        func = ReturnShape(1)
        deriv_func = ReturnShape((2,2))
        assert_raises(TypeError, optimize.leastsq, func, x0=[0,1], Dfun=deriv_func)

    def test_wrong_shape_Dfun_function(self):
        func = lambda x: dummy_func(x, (2,))
        deriv_func = lambda x: dummy_func(x, (3,3))
        assert_raises(TypeError, optimize.leastsq, func, x0=[0,1], Dfun=deriv_func)

    def test_float32(self):
        # Regression test for gh-1447
        def func(p,x,y):
            q = p[0]*np.exp(-(x-p[1])**2/(2.0*p[2]**2))+p[3]
            return q - y

        x = np.array([1.475,1.429,1.409,1.419,1.455,1.519,1.472, 1.368,1.286,
                       1.231], dtype=np.float32)
        y = np.array([0.0168,0.0193,0.0211,0.0202,0.0171,0.0151,0.0185,0.0258,
                      0.034,0.0396], dtype=np.float32)
        p0 = np.array([1.0,1.0,1.0,1.0])
        p1, success = optimize.leastsq(func, p0, args=(x,y))

        assert_(success in [1,2,3,4])
        assert_((func(p1,x,y)**2).sum() < 1e-4 * (func(p0,x,y)**2).sum())

    def test_func_can_raise(self):
        def func(*args):
            raise ValueError('I raised')

        with assert_raises(ValueError, match='I raised'):
            optimize.leastsq(func, x0=[0])

    def test_Dfun_can_raise(self):
        func = lambda x: x - np.array([10])

        def deriv_func(*args):
            raise ValueError('I raised')

        with assert_raises(ValueError, match='I raised'):
            optimize.leastsq(func, x0=[0], Dfun=deriv_func)

    def test_reentrant_func(self):
        def func(*args):
            self.test_basic()
            return self.residuals(*args)

        p0 = array([0,0,0])
        params_fit, ier = leastsq(func, p0,
                                  args=(self.y_meas, self.x))
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # low precision due to random
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_reentrant_Dfun(self):
        def deriv_func(*args):
            self.test_basic()
            return self.residuals_jacobian(*args)

        p0 = array([0,0,0])
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x),
                                  Dfun=deriv_func)
        assert_(ier in (1,2,3,4), 'solution not found (ier=%d)' % ier)
        # low precision due to random
        assert_array_almost_equal(params_fit, self.abc, decimal=2)

    def test_concurrent_no_gradient(self):
        return sequence_parallel([self.test_basic] * 10)

    def test_concurrent_with_gradient(self):
        return sequence_parallel([self.test_basic_with_gradient] * 10)

    def test_func_input_output_length_check(self):

        def func(x):
            return 2 * (x[0] - 3) ** 2 + 1

        with assert_raises(TypeError,
                           match='Improper input: func input vector length N='):
            optimize.leastsq(func, x0=[0, 1])


class TestCurveFit:
    def setup_method(self):
        self.y = array([1.0, 3.2, 9.5, 13.7])
        self.x = array([1.0, 2.0, 3.0, 4.0])

    def test_one_argument(self):
        def func(x,a):
            return x**a
        popt, pcov = curve_fit(func, self.x, self.y)
        assert_(len(popt) == 1)
        assert_(pcov.shape == (1,1))
        assert_almost_equal(popt[0], 1.9149, decimal=4)
        assert_almost_equal(pcov[0,0], 0.0016, decimal=4)

        # Test if we get the same with full_output. Regression test for #1415.
        # Also test if check_finite can be turned off.
        res = curve_fit(func, self.x, self.y,
                        full_output=1, check_finite=False)
        (popt2, pcov2, infodict, errmsg, ier) = res
        assert_array_almost_equal(popt, popt2)

    def test_two_argument(self):
        def func(x, a, b):
            return b*x**a
        popt, pcov = curve_fit(func, self.x, self.y)
        assert_(len(popt) == 2)
        assert_(pcov.shape == (2,2))
        assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
        assert_array_almost_equal(pcov, [[0.0852, -0.1260], [-0.1260, 0.1912]],
                                  decimal=4)

    def test_func_is_classmethod(self):
        class test_self:
            """This class tests if curve_fit passes the correct number of
               arguments when the model function is a class instance method.
            """

            def func(self, x, a, b):
                return b * x**a

        test_self_inst = test_self()
        popt, pcov = curve_fit(test_self_inst.func, self.x, self.y)
        assert_(pcov.shape == (2,2))
        assert_array_almost_equal(popt, [1.7989, 1.1642], decimal=4)
        assert_array_almost_equal(pcov, [[0.0852, -0.1260], [-0.1260, 0.1912]],
                                  decimal=4)

    def test_regression_2639(self):
        # This test fails if epsfcn in leastsq is too large.
        x = [574.14200000000005, 574.154, 574.16499999999996,
             574.17700000000002, 574.18799999999999, 574.19899999999996,
             574.21100000000001, 574.22199999999998, 574.23400000000004,
             574.245]
        y = [859.0, 997.0, 1699.0, 2604.0, 2013.0, 1964.0, 2435.0,
             1550.0, 949.0, 841.0]
        guess = [574.1861428571428, 574.2155714285715, 1302.0, 1302.0,
                 0.0035019999999983615, 859.0]
        good = [5.74177150e+02, 5.74209188e+02, 1.74187044e+03, 1.58646166e+03,
                1.0068462e-02, 8.57450661e+02]

        def f_double_gauss(x, x0, x1, A0, A1, sigma, c):
            return (A0*np.exp(-(x-x0)**2/(2.*sigma**2))
                    + A1*np.exp(-(x-x1)**2/(2.*sigma**2)) + c)
        popt, pcov = curve_fit(f_double_gauss, x, y, guess, maxfev=10000)
        assert_allclose(popt, good, rtol=1e-5)

    def test_pcov(self):
        xdata = np.array([0, 1, 2, 3, 4, 5])
        ydata = np.array([1, 1, 5, 7, 8, 12])
        sigma = np.array([1, 2, 1, 2, 1, 2])

        def f(x, a, b):
            return a*x + b

        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma,
                                   method=method)
            perr_scaled = np.sqrt(np.diag(pcov))
            assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=1e-3)

            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3*sigma,
                                   method=method)
            perr_scaled = np.sqrt(np.diag(pcov))
            assert_allclose(perr_scaled, [0.20659803, 0.57204404], rtol=1e-3)

            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=sigma,
                                   absolute_sigma=True, method=method)
            perr = np.sqrt(np.diag(pcov))
            assert_allclose(perr, [0.30714756, 0.85045308], rtol=1e-3)

            popt, pcov = curve_fit(f, xdata, ydata, p0=[2, 0], sigma=3*sigma,
                                   absolute_sigma=True, method=method)
            perr = np.sqrt(np.diag(pcov))
            assert_allclose(perr, [3*0.30714756, 3*0.85045308], rtol=1e-3)

        # infinite variances

        def f_flat(x, a, b):
            return a*x

        pcov_expected = np.array([np.inf]*4).reshape(2, 2)

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning,
                       "Covariance of the parameters could not be estimated")
            popt, pcov = curve_fit(f_flat, xdata, ydata, p0=[2, 0], sigma=sigma)
            popt1, pcov1 = curve_fit(f, xdata[:2], ydata[:2], p0=[2, 0])

        assert_(pcov.shape == (2, 2))
        assert_array_equal(pcov, pcov_expected)

        assert_(pcov1.shape == (2, 2))
        assert_array_equal(pcov1, pcov_expected)

    def test_array_like(self):
        # Test sequence input. Regression test for gh-3037.
        def f_linear(x, a, b):
            return a*x + b

        x = [1, 2, 3, 4]
        y = [3, 5, 7, 9]
        assert_allclose(curve_fit(f_linear, x, y)[0], [2, 1], atol=1e-10)

    def test_indeterminate_covariance(self):
        # Test that a warning is returned when pcov is indeterminate
        xdata = np.array([1, 2, 3, 4, 5, 6])
        ydata = np.array([1, 2, 3, 4, 5.5, 6])
        assert_warns(OptimizeWarning, curve_fit,
                     lambda x, a, b: a*x, xdata, ydata)

    def test_NaN_handling(self):
        # Test for correct handling of NaNs in input data: gh-3422

        # create input with NaNs
        xdata = np.array([1, np.nan, 3])
        ydata = np.array([1, 2, 3])

        assert_raises(ValueError, curve_fit,
                      lambda x, a, b: a*x + b, xdata, ydata)
        assert_raises(ValueError, curve_fit,
                      lambda x, a, b: a*x + b, ydata, xdata)

        assert_raises(ValueError, curve_fit, lambda x, a, b: a*x + b,
                      xdata, ydata, **{"check_finite": True})

    def test_empty_inputs(self):
        # Test both with and without bounds (regression test for gh-9864)
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [], [])
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [], [],
                      bounds=(1, 2))
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [1], [])
        assert_raises(ValueError, curve_fit, lambda x, a: a*x, [2], [],
                      bounds=(1, 2))

    def test_function_zero_params(self):
        # Fit args is zero, so "Unable to determine number of fit parameters."
        assert_raises(ValueError, curve_fit, lambda x: x, [1, 2], [3, 4])

    def test_None_x(self):  # Added in GH10196
        popt, pcov = curve_fit(lambda _, a: a * np.arange(10),
                               None, 2 * np.arange(10))
        assert_allclose(popt, [2.])

    def test_method_argument(self):
        def f(x, a, b):
            return a * np.exp(-b*x)

        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        for method in ['trf', 'dogbox', 'lm', None]:
            popt, pcov = curve_fit(f, xdata, ydata, method=method)
            assert_allclose(popt, [2., 2.])

        assert_raises(ValueError, curve_fit, f, xdata, ydata, method='unknown')

    def test_bounds(self):
        def f(x, a, b):
            return a * np.exp(-b*x)

        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        # The minimum w/out bounds is at [2., 2.],
        # and with bounds it's at [1.5, smth].
        bounds = ([1., 0], [1.5, 3.])
        for method in [None, 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, bounds=bounds,
                                   method=method)
            assert_allclose(popt[0], 1.5)

        # With bounds, the starting estimate is feasible.
        popt, pcov = curve_fit(f, xdata, ydata, method='trf',
                               bounds=([0., 0], [0.6, np.inf]))
        assert_allclose(popt[0], 0.6)

        # method='lm' doesn't support bounds.
        assert_raises(ValueError, curve_fit, f, xdata, ydata, bounds=bounds,
                      method='lm')

    def test_bounds_p0(self):
        # This test is for issue #5719. The problem was that an initial guess
        # was ignored when 'trf' or 'dogbox' methods were invoked.
        def f(x, a):
            return np.sin(x + a)

        xdata = np.linspace(-2*np.pi, 2*np.pi, 40)
        ydata = np.sin(xdata)
        bounds = (-3 * np.pi, 3 * np.pi)
        for method in ['trf', 'dogbox']:
            popt_1, _ = curve_fit(f, xdata, ydata, p0=2.1*np.pi)
            popt_2, _ = curve_fit(f, xdata, ydata, p0=2.1*np.pi,
                                  bounds=bounds, method=method)

            # If the initial guess is ignored, then popt_2 would be close 0.
            assert_allclose(popt_1, popt_2)

    def test_jac(self):
        # Test that Jacobian callable is handled correctly and
        # weighted if sigma is provided.
        def f(x, a, b):
            return a * np.exp(-b*x)

        def jac(x, a, b):
            e = np.exp(-b*x)
            return np.vstack((e, -a * x * e)).T

        xdata = np.linspace(0, 1, 11)
        ydata = f(xdata, 2., 2.)

        # Test numerical options for least_squares backend.
        for method in ['trf', 'dogbox']:
            for scheme in ['2-point', '3-point', 'cs']:
                popt, pcov = curve_fit(f, xdata, ydata, jac=scheme,
                                       method=method)
                assert_allclose(popt, [2, 2])

        # Test the analytic option.
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, method=method, jac=jac)
            assert_allclose(popt, [2, 2])

        # Now add an outlier and provide sigma.
        ydata[5] = 100
        sigma = np.ones(xdata.shape[0])
        sigma[5] = 200
        for method in ['lm', 'trf', 'dogbox']:
            popt, pcov = curve_fit(f, xdata, ydata, sigma=sigma, method=method,
                                   jac=jac)
            # Still the optimization process is influenced somehow,
            # have to set rtol=1e-3.
            assert_allclose(popt, [2, 2], rtol=1e-3)

    def test_maxfev_and_bounds(self):
        # gh-6340: with no bounds, curve_fit accepts parameter maxfev (via leastsq)
        # but with bounds, the parameter is `max_nfev` (via least_squares)
        x = np.arange(0, 10)
        y = 2*x
        popt1, _ = curve_fit(lambda x,p: p*x, x, y, bounds=(0, 3), maxfev=100)
        popt2, _ = curve_fit(lambda x,p: p*x, x, y, bounds=(0, 3), max_nfev=100)

        assert_allclose(popt1, 2, atol=1e-14)
        assert_allclose(popt2, 2, atol=1e-14)

    def test_curvefit_simplecovariance(self):

        def func(x, a, b):
            return a * np.exp(-b*x)

        def jac(x, a, b):
            e = np.exp(-b*x)
            return np.vstack((e, -a * x * e)).T

        np.random.seed(0)
        xdata = np.linspace(0, 4, 50)
        y = func(xdata, 2.5, 1.3)
        ydata = y + 0.2 * np.random.normal(size=len(xdata))

        sigma = np.zeros(len(xdata)) + 0.2
        covar = np.diag(sigma**2)

        for jac1, jac2 in [(jac, jac), (None, None)]:
            for absolute_sigma in [False, True]:
                popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma,
                        jac=jac1, absolute_sigma=absolute_sigma)
                popt2, pcov2 = curve_fit(func, xdata, ydata, sigma=covar,
                        jac=jac2, absolute_sigma=absolute_sigma)

                assert_allclose(popt1, popt2, atol=1e-14)
                assert_allclose(pcov1, pcov2, atol=1e-14)

    def test_curvefit_covariance(self):

        def funcp(x, a, b):
            rotn = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0], [1./np.sqrt(2), 1./np.sqrt(2), 0], [0, 0, 1.0]])
            return rotn.dot(a * np.exp(-b*x))

        def jacp(x, a, b):
            rotn = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0], [1./np.sqrt(2), 1./np.sqrt(2), 0], [0, 0, 1.0]])
            e = np.exp(-b*x)
            return rotn.dot(np.vstack((e, -a * x * e)).T)

        def func(x, a, b):
            return a * np.exp(-b*x)

        def jac(x, a, b):
            e = np.exp(-b*x)
            return np.vstack((e, -a * x * e)).T

        np.random.seed(0)
        xdata = np.arange(1, 4)
        y = func(xdata, 2.5, 1.0)
        ydata = y + 0.2 * np.random.normal(size=len(xdata))
        sigma = np.zeros(len(xdata)) + 0.2
        covar = np.diag(sigma**2)
        # Get a rotation matrix, and obtain ydatap = R ydata
        # Chisq = ydata^T C^{-1} ydata
        #       = ydata^T R^T R C^{-1} R^T R ydata
        #       = ydatap^T Cp^{-1} ydatap
        # Cp^{-1} = R C^{-1} R^T
        # Cp      = R C R^T, since R^-1 = R^T
        rotn = np.array([[1./np.sqrt(2), -1./np.sqrt(2), 0], [1./np.sqrt(2), 1./np.sqrt(2), 0], [0, 0, 1.0]])
        ydatap = rotn.dot(ydata)
        covarp = rotn.dot(covar).dot(rotn.T)

        for jac1, jac2 in [(jac, jacp), (None, None)]:
            for absolute_sigma in [False, True]:
                popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma,
                        jac=jac1, absolute_sigma=absolute_sigma)
                popt2, pcov2 = curve_fit(funcp, xdata, ydatap, sigma=covarp,
                        jac=jac2, absolute_sigma=absolute_sigma)

                assert_allclose(popt1, popt2, rtol=1.2e-7, atol=1e-14)
                assert_allclose(pcov1, pcov2, rtol=1.2e-7, atol=1e-14)

    def test_dtypes(self):
        # regression test for gh-9581: curve_fit fails if x and y dtypes differ
        x = np.arange(-3, 5)
        y = 1.5*x + 3.0 + 0.5*np.sin(x)

        def func(x, a, b):
            return a*x + b

        for method in ['lm', 'trf', 'dogbox']:
            for dtx in [np.float32, np.float64]:
                for dty in [np.float32, np.float64]:
                    x = x.astype(dtx)
                    y = y.astype(dty)

                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    p, cov = curve_fit(func, x, y, method=method)

                    assert np.isfinite(cov).all()
                    assert not np.allclose(p, 1)   # curve_fit's initial value

    def test_dtypes2(self):
        # regression test for gh-7117: curve_fit fails if
        # both inputs are float32
        def hyperbola(x, s_1, s_2, o_x, o_y, c):
            b_2 = (s_1 + s_2) / 2
            b_1 = (s_2 - s_1) / 2
            return o_y + b_1*(x-o_x) + b_2*np.sqrt((x-o_x)**2 + c**2/4)

        min_fit = np.array([-3.0, 0.0, -2.0, -10.0, 0.0])
        max_fit = np.array([0.0, 3.0, 3.0, 0.0, 10.0])
        guess = np.array([-2.5/3.0, 4/3.0, 1.0, -4.0, 0.5])

        params = [-2, .4, -1, -5, 9.5]
        xdata = np.array([-32, -16, -8, 4, 4, 8, 16, 32])
        ydata = hyperbola(xdata, *params)

        # run optimization twice, with xdata being float32 and float64
        popt_64, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess,
                               bounds=(min_fit, max_fit))

        xdata = xdata.astype(np.float32)
        ydata = hyperbola(xdata, *params)

        popt_32, _ = curve_fit(f=hyperbola, xdata=xdata, ydata=ydata, p0=guess,
                               bounds=(min_fit, max_fit))

        assert_allclose(popt_32, popt_64, atol=2e-5)

    def test_broadcast_y(self):
        xdata = np.arange(10)
        target = 4.7 * xdata ** 2 + 3.5 * xdata + np.random.rand(len(xdata))
        fit_func = lambda x, a, b: a*x**2 + b*x - target
        for method in ['lm', 'trf', 'dogbox']:
            popt0, pcov0 = curve_fit(fit_func,
                                     xdata=xdata,
                                     ydata=np.zeros_like(xdata),
                                     method=method)
            popt1, pcov1 = curve_fit(fit_func,
                                     xdata=xdata,
                                     ydata=0,
                                     method=method)
            assert_allclose(pcov0, pcov1)

    def test_args_in_kwargs(self):
        # Ensure that `args` cannot be passed as keyword argument to `curve_fit`

        def func(x, a, b):
            return a * x + b

        with assert_raises(ValueError):
            curve_fit(func,
                      xdata=[1, 2, 3, 4],
                      ydata=[5, 9, 13, 17],
                      p0=[1],
                      args=(1,))

    def test_data_point_number_validation(self):
        def func(x, a, b, c, d, e):
            return a * np.exp(-b * x) + c + d + e

        with assert_raises(TypeError, match="The number of func parameters="):
            curve_fit(func,
                      xdata=[1, 2, 3, 4],
                      ydata=[5, 9, 13, 17])


class TestFixedPoint:

    def test_scalar_trivial(self):
        # f(x) = 2x; fixed point should be x=0
        def func(x):
            return 2.0*x
        x0 = 1.0
        x = fixed_point(func, x0)
        assert_almost_equal(x, 0.0)

    def test_scalar_basic1(self):
        # f(x) = x**2; x0=1.05; fixed point should be x=1
        def func(x):
            return x**2
        x0 = 1.05
        x = fixed_point(func, x0)
        assert_almost_equal(x, 1.0)

    def test_scalar_basic2(self):
        # f(x) = x**0.5; x0=1.05; fixed point should be x=1
        def func(x):
            return x**0.5
        x0 = 1.05
        x = fixed_point(func, x0)
        assert_almost_equal(x, 1.0)

    def test_array_trivial(self):
        def func(x):
            return 2.0*x
        x0 = [0.3, 0.15]
        with np.errstate(all='ignore'):
            x = fixed_point(func, x0)
        assert_almost_equal(x, [0.0, 0.0])

    def test_array_basic1(self):
        # f(x) = c * x**2; fixed point should be x=1/c
        def func(x, c):
            return c * x**2
        c = array([0.75, 1.0, 1.25])
        x0 = [1.1, 1.15, 0.9]
        with np.errstate(all='ignore'):
            x = fixed_point(func, x0, args=(c,))
        assert_almost_equal(x, 1.0/c)

    def test_array_basic2(self):
        # f(x) = c * x**0.5; fixed point should be x=c**2
        def func(x, c):
            return c * x**0.5
        c = array([0.75, 1.0, 1.25])
        x0 = [0.8, 1.1, 1.1]
        x = fixed_point(func, x0, args=(c,))
        assert_almost_equal(x, c**2)

    def test_lambertw(self):
        # python-list/2010-December/594592.html
        xxroot = fixed_point(lambda xx: np.exp(-2.0*xx)/2.0, 1.0,
                args=(), xtol=1e-12, maxiter=500)
        assert_allclose(xxroot, np.exp(-2.0*xxroot)/2.0)
        assert_allclose(xxroot, lambertw(1)/2)

    def test_no_acceleration(self):
        # github issue 5460
        ks = 2
        kl = 6
        m = 1.3
        n0 = 1.001
        i0 = ((m-1)/m)*(kl/ks/m)**(1/(m-1))

        def func(n):
            return np.log(kl/ks/n) / np.log((i0*n/(n - 1))) + 1

        n = fixed_point(func, n0, method='iteration')
        assert_allclose(n, m)
