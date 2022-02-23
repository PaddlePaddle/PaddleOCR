import pytest

from math import sqrt, exp, sin, cos
from functools import lru_cache

from numpy.testing import (assert_warns, assert_,
                           assert_allclose,
                           assert_equal,
                           assert_array_equal,
                           suppress_warnings)
import numpy as np
from numpy import finfo, power, nan, isclose


from scipy.optimize import zeros, newton, root_scalar

from scipy._lib._util import getfullargspec_no_self as _getfullargspec

# Import testing parameters
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions, fstrings as tstutils_fstrings

TOL = 4*np.finfo(float).eps  # tolerance

_FLOAT_EPS = finfo(float).eps

# A few test functions used frequently:
# # A simple quadratic, (x-1)^2 - 1
def f1(x):
    return x ** 2 - 2 * x - 1


def f1_1(x):
    return 2 * x - 2


def f1_2(x):
    return 2.0 + 0 * x


def f1_and_p_and_pp(x):
    return f1(x), f1_1(x), f1_2(x)


# Simple transcendental function
def f2(x):
    return exp(x) - cos(x)


def f2_1(x):
    return exp(x) + sin(x)


def f2_2(x):
    return exp(x) + cos(x)


# lru cached function
@lru_cache()
def f_lrucached(x):
    return x


class TestBasic:

    def run_check_by_name(self, name, smoothness=0, **kwargs):
        a = .5
        b = sqrt(3)
        xtol = 4*np.finfo(float).eps
        rtol = 4*np.finfo(float).eps
        for function, fname in zip(tstutils_functions, tstutils_fstrings):
            if smoothness > 0 and fname in ['f4', 'f5', 'f6']:
                continue
            r = root_scalar(function, method=name, bracket=[a, b], x0=a,
                            xtol=xtol, rtol=rtol, **kwargs)
            zero = r.root
            assert_(r.converged)
            assert_allclose(zero, 1.0, atol=xtol, rtol=rtol,
                            err_msg='method %s, function %s' % (name, fname))

    def run_check(self, method, name):
        a = .5
        b = sqrt(3)
        xtol = 4 * _FLOAT_EPS
        rtol = 4 * _FLOAT_EPS
        for function, fname in zip(tstutils_functions, tstutils_fstrings):
            zero, r = method(function, a, b, xtol=xtol, rtol=rtol,
                             full_output=True)
            assert_(r.converged)
            assert_allclose(zero, 1.0, atol=xtol, rtol=rtol,
                            err_msg='method %s, function %s' % (name, fname))

    def run_check_lru_cached(self, method, name):
        # check that https://github.com/scipy/scipy/issues/10846 is fixed
        a = -1
        b = 1
        zero, r = method(f_lrucached, a, b, full_output=True)
        assert_(r.converged)
        assert_allclose(zero, 0,
                        err_msg='method %s, function %s' % (name, 'f_lrucached'))

    def _run_one_test(self, tc, method, sig_args_keys=None,
                      sig_kwargs_keys=None, **kwargs):
        method_args = []
        for k in sig_args_keys or []:
            if k not in tc:
                # If a,b not present use x0, x1. Similarly for f and func
                k = {'a': 'x0', 'b': 'x1', 'func': 'f'}.get(k, k)
            method_args.append(tc[k])

        method_kwargs = dict(**kwargs)
        method_kwargs.update({'full_output': True, 'disp': False})
        for k in sig_kwargs_keys or []:
            method_kwargs[k] = tc[k]

        root = tc.get('root')
        func_args = tc.get('args', ())

        try:
            r, rr = method(*method_args, args=func_args, **method_kwargs)
            return root, rr, tc
        except Exception:
            return root, zeros.RootResults(nan, -1, -1, zeros._EVALUEERR), tc

    def run_tests(self, tests, method, name,
                  xtol=4 * _FLOAT_EPS, rtol=4 * _FLOAT_EPS,
                  known_fail=None, **kwargs):
        r"""Run test-cases using the specified method and the supplied signature.

        Extract the arguments for the method call from the test case
        dictionary using the supplied keys for the method's signature."""
        # The methods have one of two base signatures:
        # (f, a, b, **kwargs)  # newton
        # (func, x0, **kwargs)  # bisect/brentq/...
        sig = _getfullargspec(method)  # FullArgSpec with args, varargs, varkw, defaults, ...
        assert_(not sig.kwonlyargs)
        nDefaults = len(sig.defaults)
        nRequired = len(sig.args) - nDefaults
        sig_args_keys = sig.args[:nRequired]
        sig_kwargs_keys = []
        if name in ['secant', 'newton', 'halley']:
            if name in ['newton', 'halley']:
                sig_kwargs_keys.append('fprime')
                if name in ['halley']:
                    sig_kwargs_keys.append('fprime2')
            kwargs['tol'] = xtol
        else:
            kwargs['xtol'] = xtol
            kwargs['rtol'] = rtol

        results = [list(self._run_one_test(
            tc, method, sig_args_keys=sig_args_keys,
            sig_kwargs_keys=sig_kwargs_keys, **kwargs)) for tc in tests]
        # results= [[true root, full output, tc], ...]

        known_fail = known_fail or []
        notcvgd = [elt for elt in results if not elt[1].converged]
        notcvgd = [elt for elt in notcvgd if elt[-1]['ID'] not in known_fail]
        notcvged_IDS = [elt[-1]['ID'] for elt in notcvgd]
        assert_equal([len(notcvged_IDS), notcvged_IDS], [0, []])

        # The usable xtol and rtol depend on the test
        tols = {'xtol': 4 * _FLOAT_EPS, 'rtol': 4 * _FLOAT_EPS}
        tols.update(**kwargs)
        rtol = tols['rtol']
        atol = tols.get('tol', tols['xtol'])

        cvgd = [elt for elt in results if elt[1].converged]
        approx = [elt[1].root for elt in cvgd]
        correct = [elt[0] for elt in cvgd]
        notclose = [[a] + elt for a, c, elt in zip(approx, correct, cvgd) if
                    not isclose(a, c, rtol=rtol, atol=atol)
                    and elt[-1]['ID'] not in known_fail]
        # Evaluate the function and see if is 0 at the purported root
        fvs = [tc['f'](aroot, *(tc['args'])) for aroot, c, fullout, tc in notclose]
        notclose = [[fv] + elt for fv, elt in zip(fvs, notclose) if fv != 0]
        assert_equal([notclose, len(notclose)], [[], 0])

    def run_collection(self, collection, method, name, smoothness=None,
                       known_fail=None,
                       xtol=4 * _FLOAT_EPS, rtol=4 * _FLOAT_EPS,
                       **kwargs):
        r"""Run a collection of tests using the specified method.

        The name is used to determine some optional arguments."""
        tests = get_tests(collection, smoothness=smoothness)
        self.run_tests(tests, method, name, xtol=xtol, rtol=rtol,
                       known_fail=known_fail, **kwargs)

    def test_bisect(self):
        self.run_check(zeros.bisect, 'bisect')
        self.run_check_lru_cached(zeros.bisect, 'bisect')
        self.run_check_by_name('bisect')
        self.run_collection('aps', zeros.bisect, 'bisect', smoothness=1)

    def test_ridder(self):
        self.run_check(zeros.ridder, 'ridder')
        self.run_check_lru_cached(zeros.ridder, 'ridder')
        self.run_check_by_name('ridder')
        self.run_collection('aps', zeros.ridder, 'ridder', smoothness=1)

    def test_brentq(self):
        self.run_check(zeros.brentq, 'brentq')
        self.run_check_lru_cached(zeros.brentq, 'brentq')
        self.run_check_by_name('brentq')
        # Brentq/h needs a lower tolerance to be specified
        self.run_collection('aps', zeros.brentq, 'brentq', smoothness=1,
                            xtol=1e-14, rtol=1e-14)

    def test_brenth(self):
        self.run_check(zeros.brenth, 'brenth')
        self.run_check_lru_cached(zeros.brenth, 'brenth')
        self.run_check_by_name('brenth')
        self.run_collection('aps', zeros.brenth, 'brenth', smoothness=1,
                            xtol=1e-14, rtol=1e-14)

    def test_toms748(self):
        self.run_check(zeros.toms748, 'toms748')
        self.run_check_lru_cached(zeros.toms748, 'toms748')
        self.run_check_by_name('toms748')
        self.run_collection('aps', zeros.toms748, 'toms748', smoothness=1)

    def test_newton_collections(self):
        known_fail = ['aps.13.00']
        known_fail += ['aps.12.05', 'aps.12.17']  # fails under Windows Py27
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'newton',
                                smoothness=2, known_fail=known_fail)

    def test_halley_collections(self):
        known_fail = ['aps.12.06', 'aps.12.07', 'aps.12.08', 'aps.12.09',
                      'aps.12.10', 'aps.12.11', 'aps.12.12', 'aps.12.13',
                      'aps.12.14', 'aps.12.15', 'aps.12.16', 'aps.12.17',
                      'aps.12.18', 'aps.13.00']
        for collection in ['aps', 'complex']:
            self.run_collection(collection, zeros.newton, 'halley',
                                smoothness=2, known_fail=known_fail)

    @staticmethod
    def f1(x):
        return x**2 - 2*x - 1  # == (x-1)**2 - 2

    @staticmethod
    def f1_1(x):
        return 2*x - 2

    @staticmethod
    def f1_2(x):
        return 2.0 + 0*x

    @staticmethod
    def f2(x):
        return exp(x) - cos(x)

    @staticmethod
    def f2_1(x):
        return exp(x) + sin(x)

    @staticmethod
    def f2_2(x):
        return exp(x) + cos(x)

    def test_newton(self):
        for f, f_1, f_2 in [(self.f1, self.f1_1, self.f1_2),
                            (self.f2, self.f2_1, self.f2_2)]:
            x = zeros.newton(f, 3, tol=1e-6)
            assert_allclose(f(x), 0, atol=1e-6)
            x = zeros.newton(f, 3, x1=5, tol=1e-6)  # secant, x0 and x1
            assert_allclose(f(x), 0, atol=1e-6)
            x = zeros.newton(f, 3, fprime=f_1, tol=1e-6)   # newton
            assert_allclose(f(x), 0, atol=1e-6)
            x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-6)  # halley
            assert_allclose(f(x), 0, atol=1e-6)

    def test_newton_by_name(self):
        r"""Invoke newton through root_scalar()"""
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='newton', x0=3, fprime=f_1, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)

    def test_secant_by_name(self):
        r"""Invoke secant through root_scalar()"""
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='secant', x0=3, x1=2, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)
            r = root_scalar(f, method='secant', x0=3, x1=5, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)

    def test_halley_by_name(self):
        r"""Invoke halley through root_scalar()"""
        for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
            r = root_scalar(f, method='halley', x0=3,
                            fprime=f_1, fprime2=f_2, xtol=1e-6)
            assert_allclose(f(r.root), 0, atol=1e-6)

    def test_root_scalar_fail(self):
        with pytest.raises(ValueError):
            root_scalar(f1, method='secant', x0=3, xtol=1e-6)  # no x1
        with pytest.raises(ValueError):
            root_scalar(f1, method='newton', x0=3, xtol=1e-6)  # no fprime
        with pytest.raises(ValueError):
            root_scalar(f1, method='halley', fprime=f1_1, x0=3, xtol=1e-6)  # no fprime2
        with pytest.raises(ValueError):
            root_scalar(f1, method='halley', fprime2=f1_2, x0=3, xtol=1e-6)  # no fprime

    def test_array_newton(self):
        """test newton with array"""

        def f1(x, *a):
            b = a[0] + x * a[3]
            return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x

        def f1_1(x, *a):
            b = a[3] / a[5]
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1

        def f1_2(x, *a):
            b = a[3] / a[5]
            return -a[2] * np.exp(a[0] / a[5] + x * b) * b**2

        a0 = np.array([
            5.32725221, 5.48673747, 5.49539973,
            5.36387202, 4.80237316, 1.43764452,
            5.23063958, 5.46094772, 5.50512718,
            5.42046290
        ])
        a1 = (np.sin(range(10)) + 1.0) * 7.0
        args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
        x0 = [7.0] * 10
        x = zeros.newton(f1, x0, f1_1, args)
        x_expected = (
            6.17264965, 11.7702805, 12.2219954,
            7.11017681, 1.18151293, 0.143707955,
            4.31928228, 10.5419107, 12.7552490,
            8.91225749
        )
        assert_allclose(x, x_expected)
        # test halley's
        x = zeros.newton(f1, x0, f1_1, args, fprime2=f1_2)
        assert_allclose(x, x_expected)
        # test secant
        x = zeros.newton(f1, x0, args=args)
        assert_allclose(x, x_expected)

    def test_array_newton_complex(self):
        def f(x):
            return x + 1+1j

        def fprime(x):
            return 1.0

        t = np.full(4, 1j)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.)

        # should work even if x0 is not complex
        t = np.ones(4)
        x = zeros.newton(f, t, fprime=fprime)
        assert_allclose(f(x), 0.)

        x = zeros.newton(f, t)
        assert_allclose(f(x), 0.)

    def test_array_secant_active_zero_der(self):
        """test secant doesn't continue to iterate zero derivatives"""
        x = zeros.newton(lambda x, *a: x*x - a[0], x0=[4.123, 5],
                         args=[np.array([17, 25])])
        assert_allclose(x, (4.123105625617661, 5.0))

    def test_array_newton_integers(self):
        # test secant with float
        x = zeros.newton(lambda y, z: z - y ** 2, [4.0] * 2,
                         args=([15.0, 17.0],))
        assert_allclose(x, (3.872983346207417, 4.123105625617661))
        # test integer becomes float
        x = zeros.newton(lambda y, z: z - y ** 2, [4] * 2, args=([15, 17],))
        assert_allclose(x, (3.872983346207417, 4.123105625617661))

    def test_array_newton_zero_der_failures(self):
        # test derivative zero warning
        assert_warns(RuntimeWarning, zeros.newton,
                     lambda y: y**2 - 2, [0., 0.], lambda y: 2 * y)
        # test failures and zero_der
        with pytest.warns(RuntimeWarning):
            results = zeros.newton(lambda y: y**2 - 2, [0., 0.],
                                   lambda y: 2*y, full_output=True)
            assert_allclose(results.root, 0)
            assert results.zero_der.all()
            assert not results.converged.any()

    def test_newton_combined(self):
        f1 = lambda x: x**2 - 2*x - 1
        f1_1 = lambda x: 2*x - 2
        f1_2 = lambda x: 2.0 + 0*x

        def f1_and_p_and_pp(x):
            return x**2 - 2*x-1, 2*x-2, 2.0

        sol0 = root_scalar(f1, method='newton', x0=3, fprime=f1_1)
        sol = root_scalar(f1_and_p_and_pp, method='newton', x0=3, fprime=True)
        assert_allclose(sol0.root, sol.root, atol=1e-8)
        assert_equal(2*sol.function_calls, sol0.function_calls)

        sol0 = root_scalar(f1, method='halley', x0=3, fprime=f1_1, fprime2=f1_2)
        sol = root_scalar(f1_and_p_and_pp, method='halley', x0=3, fprime2=True)
        assert_allclose(sol0.root, sol.root, atol=1e-8)
        assert_equal(3*sol.function_calls, sol0.function_calls)

    def test_newton_full_output(self):
        # Test the full_output capability, both when converging and not.
        # Use simple polynomials, to avoid hitting platform dependencies
        # (e.g., exp & trig) in number of iterations

        x0 = 3
        expected_counts = [(6, 7), (5, 10), (3, 9)]

        for derivs in range(3):
            kwargs = {'tol': 1e-6, 'full_output': True, }
            for k, v in [['fprime', self.f1_1], ['fprime2', self.f1_2]][:derivs]:
                kwargs[k] = v

            x, r = zeros.newton(self.f1, x0, disp=False, **kwargs)
            assert_(r.converged)
            assert_equal(x, r.root)
            assert_equal((r.iterations, r.function_calls), expected_counts[derivs])
            if derivs == 0:
                assert(r.function_calls <= r.iterations + 1)
            else:
                assert_equal(r.function_calls, (derivs + 1) * r.iterations)

            # Now repeat, allowing one fewer iteration to force convergence failure
            iters = r.iterations - 1
            x, r = zeros.newton(self.f1, x0, maxiter=iters, disp=False, **kwargs)
            assert_(not r.converged)
            assert_equal(x, r.root)
            assert_equal(r.iterations, iters)

            if derivs == 1:
                # Check that the correct Exception is raised and
                # validate the start of the message.
                with pytest.raises(
                    RuntimeError,
                    match='Failed to converge after %d iterations, value is .*' % (iters)):
                    x, r = zeros.newton(self.f1, x0, maxiter=iters, disp=True, **kwargs)

    def test_deriv_zero_warning(self):
        func = lambda x: x**2 - 2.0
        dfunc = lambda x: 2*x
        assert_warns(RuntimeWarning, zeros.newton, func, 0.0, dfunc, disp=False)
        with pytest.raises(RuntimeError, match='Derivative was zero'):
            zeros.newton(func, 0.0, dfunc)

    def test_newton_does_not_modify_x0(self):
        # https://github.com/scipy/scipy/issues/9964
        x0 = np.array([0.1, 3])
        x0_copy = x0.copy()  # Copy to test for equality.
        newton(np.sin, x0, np.cos)
        assert_array_equal(x0, x0_copy)

    def test_maxiter_int_check(self):
        for method in [zeros.bisect, zeros.newton, zeros.ridder, zeros.brentq,
                       zeros.brenth, zeros.toms748]:
            with pytest.raises(TypeError,
                    match="'float' object cannot be interpreted as an integer"):
                method(f1, 0.0, 1.0, maxiter=72.45)


def test_gh_5555():
    root = 0.1

    def f(x):
        return x - root

    methods = [zeros.bisect, zeros.ridder]
    xtol = rtol = TOL
    for method in methods:
        res = method(f, -1e8, 1e7, xtol=xtol, rtol=rtol)
        assert_allclose(root, res, atol=xtol, rtol=rtol,
                        err_msg='method %s' % method.__name__)


def test_gh_5557():
    # Show that without the changes in 5557 brentq and brenth might
    # only achieve a tolerance of 2*(xtol + rtol*|res|).

    # f linearly interpolates (0, -0.1), (0.5, -0.1), and (1,
    # 0.4). The important parts are that |f(0)| < |f(1)| (so that
    # brent takes 0 as the initial guess), |f(0)| < atol (so that
    # brent accepts 0 as the root), and that the exact root of f lies
    # more than atol away from 0 (so that brent doesn't achieve the
    # desired tolerance).
    def f(x):
        if x < 0.5:
            return -0.1
        else:
            return x - 0.6

    atol = 0.51
    rtol = 4 * _FLOAT_EPS
    methods = [zeros.brentq, zeros.brenth]
    for method in methods:
        res = method(f, 0, 1, xtol=atol, rtol=rtol)
        assert_allclose(0.6, res, atol=atol, rtol=rtol)


class TestRootResults:
    def test_repr(self):
        r = zeros.RootResults(root=1.0,
                              iterations=44,
                              function_calls=46,
                              flag=0)
        expected_repr = ("      converged: True\n           flag: 'converged'"
                         "\n function_calls: 46\n     iterations: 44\n"
                         "           root: 1.0")
        assert_equal(repr(r), expected_repr)


def test_complex_halley():
    """Test Halley's works with complex roots"""
    def f(x, *a):
        return a[0] * x**2 + a[1] * x + a[2]

    def f_1(x, *a):
        return 2 * a[0] * x + a[1]

    def f_2(x, *a):
        retval = 2 * a[0]
        try:
            size = len(x)
        except TypeError:
            return retval
        else:
            return [retval] * size

    z = complex(1.0, 2.0)
    coeffs = (2.0, 3.0, 4.0)
    y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-6)
    # (-0.75000000000000078+1.1989578808281789j)
    assert_allclose(f(y, *coeffs), 0, atol=1e-6)
    z = [z] * 10
    coeffs = (2.0, 3.0, 4.0)
    y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-6)
    assert_allclose(f(y, *coeffs), 0, atol=1e-6)


def test_zero_der_nz_dp():
    """Test secant method with a non-zero dp, but an infinite newton step"""
    # pick a symmetrical functions and choose a point on the side that with dx
    # makes a secant that is a flat line with zero slope, EG: f = (x - 100)**2,
    # which has a root at x = 100 and is symmetrical around the line x = 100
    # we have to pick a really big number so that it is consistently true
    # now find a point on each side so that the secant has a zero slope
    dx = np.finfo(float).eps ** 0.33
    # 100 - p0 = p1 - 100 = p0 * (1 + dx) + dx - 100
    # -> 200 = p0 * (2 + dx) + dx
    p0 = (200.0 - dx) / (2.0 + dx)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "RMS of")
        x = zeros.newton(lambda y: (y - 100.0)**2, x0=[p0] * 10)
    assert_allclose(x, [100] * 10)
    # test scalar cases too
    p0 = (2.0 - 1e-4) / (2.0 + 1e-4)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Tolerance of")
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=False)
    assert_allclose(x, 1)
    with pytest.raises(RuntimeError, match='Tolerance of'):
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=True)
    p0 = (-2.0 + 1e-4) / (2.0 + 1e-4)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, "Tolerance of")
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=False)
    assert_allclose(x, -1)
    with pytest.raises(RuntimeError, match='Tolerance of'):
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=True)


def test_array_newton_failures():
    """Test that array newton fails as expected"""
    # p = 0.68  # [MPa]
    # dp = -0.068 * 1e6  # [Pa]
    # T = 323  # [K]
    diameter = 0.10  # [m]
    # L = 100  # [m]
    roughness = 0.00015  # [m]
    rho = 988.1  # [kg/m**3]
    mu = 5.4790e-04  # [Pa*s]
    u = 2.488  # [m/s]
    reynolds_number = rho * u * diameter / mu  # Reynolds number

    def colebrook_eqn(darcy_friction, re, dia):
        return (1 / np.sqrt(darcy_friction) +
                2 * np.log10(roughness / 3.7 / dia +
                             2.51 / re / np.sqrt(darcy_friction)))

    # only some failures
    with pytest.warns(RuntimeWarning):
        result = zeros.newton(
            colebrook_eqn, x0=[0.01, 0.2, 0.02223, 0.3], maxiter=2,
            args=[reynolds_number, diameter], full_output=True
        )
        assert not result.converged.all()
    # they all fail
    with pytest.raises(RuntimeError):
        result = zeros.newton(
            colebrook_eqn, x0=[0.01] * 2, maxiter=2,
            args=[reynolds_number, diameter], full_output=True
        )


# this test should **not** raise a RuntimeWarning
def test_gh8904_zeroder_at_root_fails():
    """Test that Newton or Halley don't warn if zero derivative at root"""

    # a function that has a zero derivative at it's root
    def f_zeroder_root(x):
        return x**3 - x**2

    # should work with secant
    r = zeros.newton(f_zeroder_root, x0=0)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    # test again with array
    r = zeros.newton(f_zeroder_root, x0=[0]*10)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

    # 1st derivative
    def fder(x):
        return 3 * x**2 - 2 * x

    # 2nd derivative
    def fder2(x):
        return 6*x - 2

    # should work with newton and halley
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder,
                     fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    # test again with array
    r = zeros.newton(f_zeroder_root, x0=[0]*10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0]*10, fprime=fder,
                     fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

    # also test that if a root is found we do not raise RuntimeWarning even if
    # the derivative is zero, EG: at x = 0.5, then fval = -0.125 and
    # fder = -0.25 so the next guess is 0.5 - (-0.125/-0.5) = 0 which is the
    # root, but if the solver continued with that guess, then it will calculate
    # a zero derivative, so it should return the root w/o RuntimeWarning
    r = zeros.newton(f_zeroder_root, x0=0.5, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    # test again with array
    r = zeros.newton(f_zeroder_root, x0=[0.5]*10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    # doesn't apply to halley


def test_gh_8881():
    r"""Test that Halley's method realizes that the 2nd order adjustment
    is too big and drops off to the 1st order adjustment."""
    n = 9

    def f(x):
        return power(x, 1.0/n) - power(n, 1.0/n)

    def fp(x):
        return power(x, (1.0-n)/n)/n

    def fpp(x):
        return power(x, (1.0-2*n)/n) * (1.0/n) * (1.0-n)/n

    x0 = 0.1
    # The root is at x=9.
    # The function has positive slope, x0 < root.
    # Newton succeeds in 8 iterations
    rt, r = newton(f, x0, fprime=fp, full_output=True)
    assert(r.converged)
    # Before the Issue 8881/PR 8882, halley would send x in the wrong direction.
    # Check that it now succeeds.
    rt, r = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert(r.converged)


def test_gh_9608_preserve_array_shape():
    """
    Test that shape is preserved for array inputs even if fprime or fprime2 is
    scalar
    """
    def f(x):
        return x**2

    def fp(x):
        return 2 * x

    def fpp(x):
        return 2

    x0 = np.array([-2], dtype=np.float32)
    rt, r = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert(r.converged)

    x0_array = np.array([-2, -3], dtype=np.float32)
    # This next invocation should fail
    with pytest.raises(IndexError):
        result = zeros.newton(
            f, x0_array, fprime=fp, fprime2=fpp, full_output=True
        )

    def fpp_array(x):
        return np.full(np.shape(x), 2, dtype=np.float32)

    result = zeros.newton(
        f, x0_array, fprime=fp, fprime2=fpp_array, full_output=True
    )
    assert result.converged.all()


@pytest.mark.parametrize(
    "maximum_iterations,flag_expected",
    [(10, zeros.CONVERR), (100, zeros.CONVERGED)])
def test_gh9254_flag_if_maxiter_exceeded(maximum_iterations, flag_expected):
    """
    Test that if the maximum iterations is exceeded that the flag is not
    converged.
    """
    result = zeros.brentq(
        lambda x: ((1.2*x - 2.3)*x + 3.4)*x - 4.5,
        -30, 30, (), 1e-6, 1e-6, maximum_iterations,
        full_output=True, disp=False)
    assert result[1].flag == flag_expected
    if flag_expected == zeros.CONVERR:
        # didn't converge because exceeded maximum iterations
        assert result[1].iterations == maximum_iterations
    elif flag_expected == zeros.CONVERGED:
        # converged before maximum iterations
        assert result[1].iterations < maximum_iterations


def test_gh9551_raise_error_if_disp_true():
    """Test that if disp is true then zero derivative raises RuntimeError"""

    def f(x):
        return x*x + 1

    def f_p(x):
        return 2*x

    assert_warns(RuntimeWarning, zeros.newton, f, 1.0, f_p, disp=False)
    with pytest.raises(
            RuntimeError,
            match=r'^Derivative was zero\. Failed to converge after \d+ iterations, value is [+-]?\d*\.\d+\.$'):
        zeros.newton(f, 1.0, f_p)
    root = zeros.newton(f, complex(10.0, 10.0), f_p)
    assert_allclose(root, complex(0.0, 1.0))
