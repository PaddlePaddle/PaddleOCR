import os
import sys
import time

import numpy as np
from numpy.testing import assert_
import pytest

from scipy.special._testutils import assert_func_equal

try:
    import mpmath
except ImportError:
    pass


# ------------------------------------------------------------------------------
# Machinery for systematic tests with mpmath
# ------------------------------------------------------------------------------

class Arg:
    """Generate a set of numbers on the real axis, concentrating on
    'interesting' regions and covering all orders of magnitude.

    """

    def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
        if a > b:
            raise ValueError("a should be less than or equal to b")
        if a == -np.inf:
            a = -0.5*np.finfo(float).max
        if b == np.inf:
            b = 0.5*np.finfo(float).max
        self.a, self.b = a, b

        self.inclusive_a, self.inclusive_b = inclusive_a, inclusive_b

    def _positive_values(self, a, b, n):
        if a < 0:
            raise ValueError("a should be positive")

        # Try to put half of the points into a linspace between a and
        # 10 the other half in a logspace.
        if n % 2 == 0:
            nlogpts = n//2
            nlinpts = nlogpts
        else:
            nlogpts = n//2
            nlinpts = nlogpts + 1

        if a >= 10:
            # Outside of linspace range; just return a logspace.
            pts = np.logspace(np.log10(a), np.log10(b), n)
        elif a > 0 and b < 10:
            # Outside of logspace range; just return a linspace
            pts = np.linspace(a, b, n)
        elif a > 0:
            # Linspace between a and 10 and a logspace between 10 and
            # b.
            linpts = np.linspace(a, 10, nlinpts, endpoint=False)
            logpts = np.logspace(1, np.log10(b), nlogpts)
            pts = np.hstack((linpts, logpts))
        elif a == 0 and b <= 10:
            # Linspace between 0 and b and a logspace between 0 and
            # the smallest positive point of the linspace
            linpts = np.linspace(0, b, nlinpts)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts = np.logspace(-30, right, nlogpts, endpoint=False)
            pts = np.hstack((logpts, linpts))
        else:
            # Linspace between 0 and 10, logspace between 0 and the
            # smallest positive point of the linspace, and a logspace
            # between 10 and b.
            if nlogpts % 2 == 0:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1
            else:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1 + 1
            linpts = np.linspace(0, 10, nlinpts, endpoint=False)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts1 = np.logspace(-30, right, nlogpts1, endpoint=False)
            logpts2 = np.logspace(1, np.log10(b), nlogpts2)
            pts = np.hstack((logpts1, linpts, logpts2))

        return np.sort(pts)

    def values(self, n):
        """Return an array containing n numbers."""
        a, b = self.a, self.b
        if a == b:
            return np.zeros(n)

        if not self.inclusive_a:
            n += 1
        if not self.inclusive_b:
            n += 1

        if n % 2 == 0:
            n1 = n//2
            n2 = n1
        else:
            n1 = n//2
            n2 = n1 + 1

        if a >= 0:
            pospts = self._positive_values(a, b, n)
            negpts = []
        elif b <= 0:
            pospts = []
            negpts = -self._positive_values(-b, -a, n)
        else:
            pospts = self._positive_values(0, b, n1)
            negpts = -self._positive_values(0, -a, n2 + 1)
            # Don't want to get zero twice
            negpts = negpts[1:]
        pts = np.hstack((negpts[::-1], pospts))

        if not self.inclusive_a:
            pts = pts[1:]
        if not self.inclusive_b:
            pts = pts[:-1]
        return pts


class FixedArg:
    def __init__(self, values):
        self._values = np.asarray(values)

    def values(self, n):
        return self._values


class ComplexArg:
    def __init__(self, a=complex(-np.inf, -np.inf), b=complex(np.inf, np.inf)):
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)

    def values(self, n):
        m = int(np.floor(np.sqrt(n)))
        x = self.real.values(m)
        y = self.imag.values(m + 1)
        return (x[:,None] + 1j*y[None,:]).ravel()


class IntArg:
    def __init__(self, a=-1000, b=1000):
        self.a = a
        self.b = b

    def values(self, n):
        v1 = Arg(self.a, self.b).values(max(1 + n//2, n-5)).astype(int)
        v2 = np.arange(-5, 5)
        v = np.unique(np.r_[v1, v2])
        v = v[(v >= self.a) & (v < self.b)]
        return v


def get_args(argspec, n):
    if isinstance(argspec, np.ndarray):
        args = argspec.copy()
    else:
        nargs = len(argspec)
        ms = np.asarray([1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec])
        ms = (n**(ms/sum(ms))).astype(int) + 1

        args = [spec.values(m) for spec, m in zip(argspec, ms)]
        args = np.array(np.broadcast_arrays(*np.ix_(*args))).reshape(nargs, -1).T

    return args


class MpmathData:
    def __init__(self, scipy_func, mpmath_func, arg_spec, name=None,
                 dps=None, prec=None, n=None, rtol=1e-7, atol=1e-300,
                 ignore_inf_sign=False, distinguish_nan_and_inf=True,
                 nan_ok=True, param_filter=None):

        # mpmath tests are really slow (see gh-6989).  Use a small number of
        # points by default, increase back to 5000 (old default) if XSLOW is
        # set
        if n is None:
            try:
                is_xslow = int(os.environ.get('SCIPY_XSLOW', '0'))
            except ValueError:
                is_xslow = False

            n = 5000 if is_xslow else 500

        self.scipy_func = scipy_func
        self.mpmath_func = mpmath_func
        self.arg_spec = arg_spec
        self.dps = dps
        self.prec = prec
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.ignore_inf_sign = ignore_inf_sign
        self.nan_ok = nan_ok
        if isinstance(self.arg_spec, np.ndarray):
            self.is_complex = np.issubdtype(self.arg_spec.dtype, np.complexfloating)
        else:
            self.is_complex = any([isinstance(arg, ComplexArg) for arg in self.arg_spec])
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf
        if not name or name == '<lambda>':
            name = getattr(scipy_func, '__name__', None)
        if not name or name == '<lambda>':
            name = getattr(mpmath_func, '__name__', None)
        self.name = name
        self.param_filter = param_filter

    def check(self):
        np.random.seed(1234)

        # Generate values for the arguments
        argarr = get_args(self.arg_spec, self.n)

        # Check
        old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
        try:
            if self.dps is not None:
                dps_list = [self.dps]
            else:
                dps_list = [20]
            if self.prec is not None:
                mpmath.mp.prec = self.prec

            # Proper casting of mpmath input and output types. Using
            # native mpmath types as inputs gives improved precision
            # in some cases.
            if np.issubdtype(argarr.dtype, np.complexfloating):
                pytype = mpc2complex

                def mptype(x):
                    return mpmath.mpc(complex(x))
            else:
                def mptype(x):
                    return mpmath.mpf(float(x))

                def pytype(x):
                    if abs(x.imag) > 1e-16*(1 + abs(x.real)):
                        return np.nan
                    else:
                        return mpf2float(x.real)

            # Try out different dps until one (or none) works
            for j, dps in enumerate(dps_list):
                mpmath.mp.dps = dps

                try:
                    assert_func_equal(self.scipy_func,
                                      lambda *a: pytype(self.mpmath_func(*map(mptype, a))),
                                      argarr,
                                      vectorized=False,
                                      rtol=self.rtol, atol=self.atol,
                                      ignore_inf_sign=self.ignore_inf_sign,
                                      distinguish_nan_and_inf=self.distinguish_nan_and_inf,
                                      nan_ok=self.nan_ok,
                                      param_filter=self.param_filter)
                    break
                except AssertionError:
                    if j >= len(dps_list)-1:
                        # reraise the Exception
                        tp, value, tb = sys.exc_info()
                        if value.__traceback__ is not tb:
                            raise value.with_traceback(tb)
                        raise value
        finally:
            mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec

    def __repr__(self):
        if self.is_complex:
            return "<MpmathData: %s (complex)>" % (self.name,)
        else:
            return "<MpmathData: %s>" % (self.name,)


def assert_mpmath_equal(*a, **kw):
    d = MpmathData(*a, **kw)
    d.check()


def nonfunctional_tooslow(func):
    return pytest.mark.skip(reason="    Test not yet functional (too slow), needs more work.")(func)


# ------------------------------------------------------------------------------
# Tools for dealing with mpmath quirks
# ------------------------------------------------------------------------------

def mpf2float(x):
    """
    Convert an mpf to the nearest floating point number. Just using
    float directly doesn't work because of results like this:

    with mp.workdps(50):
        float(mpf("0.99999999999999999")) = 0.9999999999999999

    """
    return float(mpmath.nstr(x, 17, min_fixed=0, max_fixed=0))


def mpc2complex(x):
    return complex(mpf2float(x.real), mpf2float(x.imag))


def trace_args(func):
    def tofloat(x):
        if isinstance(x, mpmath.mpc):
            return complex(x)
        else:
            return float(x)

    def wrap(*a, **kw):
        sys.stderr.write("%r: " % (tuple(map(tofloat, a)),))
        sys.stderr.flush()
        try:
            r = func(*a, **kw)
            sys.stderr.write("-> %r" % r)
        finally:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return r
    return wrap


try:
    import posix
    import signal
    POSIX = ('setitimer' in dir(signal))
except ImportError:
    POSIX = False


class TimeoutError(Exception):
    pass


def time_limited(timeout=0.5, return_val=np.nan, use_sigalrm=True):
    """
    Decorator for setting a timeout for pure-Python functions.

    If the function does not return within `timeout` seconds, the
    value `return_val` is returned instead.

    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is
    used. Do not use this with threads: the SIGALRM implementation
    does probably not work well. The settrace implementation only
    traces the current thread.

    The settrace implementation slows down execution speed. Slowdown
    by a factor around 10 is probably typical.
    """
    if POSIX and use_sigalrm:
        def sigalrm_handler(signum, frame):
            raise TimeoutError()

        def deco(func):
            def wrap(*a, **kw):
                old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    return return_val
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrap
    else:
        def deco(func):
            def wrap(*a, **kw):
                start_time = time.time()

                def trace(frame, event, arg):
                    if time.time() - start_time > timeout:
                        raise TimeoutError()
                    return trace
                sys.settrace(trace)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    sys.settrace(None)
                    return return_val
                finally:
                    sys.settrace(None)
            return wrap
    return deco


def exception_to_nan(func):
    """Decorate function to return nan if it raises an exception"""
    def wrap(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception:
            return np.nan
    return wrap


def inf_to_nan(func):
    """Decorate function to return nan if it returns inf"""
    def wrap(*a, **kw):
        v = func(*a, **kw)
        if not np.isfinite(v):
            return np.nan
        return v
    return wrap


def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
    """
    Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it
    can be done to higher precision than double.

    """
    try:
        len(res)
    except TypeError:
        res = list(res)

    n = len(std)
    if len(res) != n:
        raise AssertionError("Lengths of inputs not equal.")

    failures = []
    for k in range(n):
        try:
            assert_(mpmath.fabs(res[k] - std[k]) <= atol + rtol*mpmath.fabs(std[k]))
        except AssertionError:
            failures.append(k)

    ndigits = int(abs(np.log10(rtol)))
    msg = [""]
    msg.append("Bad results ({} out of {}) for the following points:"
               .format(len(failures), n))
    for k in failures:
        resrep = mpmath.nstr(res[k], ndigits, min_fixed=0, max_fixed=0)
        stdrep = mpmath.nstr(std[k], ndigits, min_fixed=0, max_fixed=0)
        if std[k] == 0:
            rdiff = "inf"
        else:
            rdiff = mpmath.fabs((res[k] - std[k])/std[k])
            rdiff = mpmath.nstr(rdiff, 3)
        msg.append("{}: {} != {} (rdiff {})".format(k, resrep, stdrep, rdiff))
    if failures:
        assert_(False, "\n".join(msg))
