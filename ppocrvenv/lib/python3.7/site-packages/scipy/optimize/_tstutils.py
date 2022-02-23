r"""
Parameters used in test and benchmark methods.

Collections of test cases suitable for testing 1-D root-finders
  'original': The original benchmarking functions.
     Real-valued functions of real-valued inputs on an interval
     with a zero.
     f1, .., f3 are continuous and infinitely differentiable
     f4 has a left- and right- discontinuity at the root
     f5 has a root at 1 replacing a 1st order pole
     f6 is randomly positive on one side of the root,
     randomly negative on the other.
     f4 - f6 are not continuous at the root.

  'aps': The test problems in the 1995 paper
     TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
     by Alefeld, Potra and Shi. Real-valued functions of
     real-valued inputs on an interval with a zero.
     Suitable for methods which start with an enclosing interval, and
     derivatives up to 2nd order.

  'complex': Some complex-valued functions of complex-valued inputs.
     No enclosing bracket is provided.
     Suitable for methods which use one or more starting values, and
     derivatives up to 2nd order.

  The test cases are provided as a list of dictionaries. The dictionary
  keys will be a subset of:
  ["f", "fprime", "fprime2", "args", "bracket", "smoothness",
  "a", "b", "x0", "x1", "root", "ID"]
"""

# Sources:
#  [1] Alefeld, G. E. and Potra, F. A. and Shi, Yixun,
#      "Algorithm 748: Enclosing Zeros of Continuous Functions",
#      ACM Trans. Math. Softw. Volume 221(1995)
#       doi = {10.1145/210089.210111},

from random import random

import numpy as np

from scipy.optimize import zeros as cc

# "description" refers to the original functions
description = """
f2 is a symmetric parabola, x**2 - 1
f3 is a quartic polynomial with large hump in interval
f4 is step function with a discontinuity at 1
f5 is a hyperbola with vertical asymptote at 1
f6 has random values positive to left of 1, negative to right

Of course, these are not real problems. They just test how the
'good' solvers behave in bad circumstances where bisection is
really the best. A good solver should not be much worse than
bisection in such circumstance, while being faster for smooth
monotone sorts of functions.
"""


def f1(x):
    r"""f1 is a quadratic with roots at 0 and 1"""
    return x * (x - 1.)


def f1_fp(x):
    return 2 * x - 1


def f1_fpp(x):
    return 2


def f2(x):
    r"""f2 is a symmetric parabola, x**2 - 1"""
    return x**2 - 1


def f2_fp(x):
    return 2 * x


def f2_fpp(x):
    return 2


def f3(x):
    r"""A quartic with roots at 0, 1, 2 and 3"""
    return x * (x - 1.) * (x - 2.) * (x - 3.)  # x**4 - 6x**3 + 11x**2 - 6x


def f3_fp(x):
    return 4 * x**3 - 18 * x**2 + 22 * x - 6


def f3_fpp(x):
    return 12 * x**2 - 36 * x + 22


def f4(x):
    r"""Piecewise linear, left- and right- discontinuous at x=1, the root."""
    if x > 1:
        return 1.0 + .1 * x
    if x < 1:
        return -1.0 + .1 * x
    return 0


def f5(x):
    r"""Hyperbola with a pole at x=1, but pole replaced with 0. Not continuous at root."""
    if x != 1:
        return 1.0 / (1. - x)
    return 0


# f6(x) returns random value. Without memoization, calling twice with the
# same x returns different values, hence a "random value", not a
# "function with random values"
_f6_cache = {}
def f6(x):
    v = _f6_cache.get(x, None)
    if v is None:
        if x > 1:
            v = random()
        elif x < 1:
            v = -random()
        else:
            v = 0
        _f6_cache[x] = v
    return v


# Each Original test case has
# - a function and its two derivatives,
# - additional arguments,
# - a bracket enclosing a root,
# - the order of differentiability (smoothness) on this interval
# - a starting value for methods which don't require a bracket
# - the root (inside the bracket)
# - an Identifier of the test case

_ORIGINAL_TESTS_KEYS = ["f", "fprime", "fprime2", "args", "bracket", "smoothness", "x0", "root", "ID"]
_ORIGINAL_TESTS = [
    [f1, f1_fp, f1_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.01.00"],
    [f2, f2_fp, f2_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.02.00"],
    [f3, f3_fp, f3_fpp, (), [0.5, np.sqrt(3)], np.inf, 0.6, 1.0, "original.03.00"],
    [f4, None, None, (), [0.5, np.sqrt(3)], -1, 0.6, 1.0, "original.04.00"],
    [f5, None, None, (), [0.5, np.sqrt(3)], -1, 0.6, 1.0, "original.05.00"],
    [f6, None, None, (), [0.5, np.sqrt(3)], -np.inf, 0.6, 1.0, "original.05.00"]
]

_ORIGINAL_TESTS_DICTS = [dict(zip(_ORIGINAL_TESTS_KEYS, testcase)) for testcase in _ORIGINAL_TESTS]

#   ##################
#   "APS" test cases
#   Functions and test cases that appear in [1]


def aps01_f(x):
    r"""Straightforward sum of trigonometric function and polynomial"""
    return np.sin(x) - x / 2


def aps01_fp(x):
    return np.cos(x) - 1.0 / 2


def aps01_fpp(x):
    return -np.sin(x)


def aps02_f(x):
    r"""poles at x=n**2, 1st and 2nd derivatives at root are also close to 0"""
    ii = np.arange(1, 21)
    return -2 * np.sum((2 * ii - 5)**2 / (x - ii**2)**3)


def aps02_fp(x):
    ii = np.arange(1, 21)
    return 6 * np.sum((2 * ii - 5)**2 / (x - ii**2)**4)


def aps02_fpp(x):
    ii = np.arange(1, 21)
    return 24 * np.sum((2 * ii - 5)**2 / (x - ii**2)**5)


def aps03_f(x, a, b):
    r"""Rapidly changing at the root"""
    return a * x * np.exp(b * x)


def aps03_fp(x, a, b):
    return a * (b * x + 1) * np.exp(b * x)


def aps03_fpp(x, a, b):
    return a * (b * (b * x + 1) + b) * np.exp(b * x)


def aps04_f(x, n, a):
    r"""Medium-degree polynomial"""
    return x**n - a


def aps04_fp(x, n, a):
    return n * x**(n - 1)


def aps04_fpp(x, n, a):
    return n * (n - 1) * x**(n - 2)


def aps05_f(x):
    r"""Simple Trigonometric function"""
    return np.sin(x) - 1.0 / 2


def aps05_fp(x):
    return np.cos(x)


def aps05_fpp(x):
    return -np.sin(x)


def aps06_f(x, n):
    r"""Exponential rapidly changing from -1 to 1 at x=0"""
    return 2 * x * np.exp(-n) - 2 * np.exp(-n * x) + 1


def aps06_fp(x, n):
    return 2 * np.exp(-n) + 2 * n * np.exp(-n * x)


def aps06_fpp(x, n):
    return -2 * n * n * np.exp(-n * x)


def aps07_f(x, n):
    r"""Upside down parabola with parametrizable height"""
    return (1 + (1 - n)**2) * x - (1 - n * x)**2


def aps07_fp(x, n):
    return (1 + (1 - n)**2) + 2 * n * (1 - n * x)


def aps07_fpp(x, n):
    return -2 * n * n


def aps08_f(x, n):
    r"""Degree n polynomial"""
    return x * x - (1 - x)**n


def aps08_fp(x, n):
    return 2 * x + n * (1 - x)**(n - 1)


def aps08_fpp(x, n):
    return 2 - n * (n - 1) * (1 - x)**(n - 2)


def aps09_f(x, n):
    r"""Upside down quartic with parametrizable height"""
    return (1 + (1 - n)**4) * x - (1 - n * x)**4


def aps09_fp(x, n):
    return (1 + (1 - n)**4) + 4 * n * (1 - n * x)**3


def aps09_fpp(x, n):
    return -12 * n * (1 - n * x)**2


def aps10_f(x, n):
    r"""Exponential plus a polynomial"""
    return np.exp(-n * x) * (x - 1) + x**n


def aps10_fp(x, n):
    return np.exp(-n * x) * (-n * (x - 1) + 1) + n * x**(n - 1)


def aps10_fpp(x, n):
    return np.exp(-n * x) * (-n * (-n * (x - 1) + 1) + -n * x) + n * (n - 1) * x**(n - 2)


def aps11_f(x, n):
    r"""Rational function with a zero at x=1/n and a pole at x=0"""
    return (n * x - 1) / ((n - 1) * x)


def aps11_fp(x, n):
    return 1 / (n - 1) / x**2


def aps11_fpp(x, n):
    return -2 / (n - 1) / x**3


def aps12_f(x, n):
    r"""nth root of x, with a zero at x=n"""
    return np.power(x, 1.0 / n) - np.power(n, 1.0 / n)


def aps12_fp(x, n):
    return np.power(x, (1.0 - n) / n) / n


def aps12_fpp(x, n):
    return np.power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n


_MAX_EXPABLE = np.log(np.finfo(float).max)


def aps13_f(x):
    r"""Function with *all* derivatives 0 at the root"""
    if x == 0:
        return 0
    # x2 = 1.0/x**2
    # if x2 > 708:
    #     return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return x / np.exp(y)


def aps13_fp(x):
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return (1 + 2 / x**2) / np.exp(y)


def aps13_fpp(x):
    if x == 0:
        return 0
    y = 1 / x**2
    if y > _MAX_EXPABLE:
        return 0
    return 2 * (2 - x**2) / x**5 / np.exp(y)


def aps14_f(x, n):
    r"""0 for negative x-values, trigonometric+linear for x positive"""
    if x <= 0:
        return -n / 20.0
    return n / 20.0 * (x / 1.5 + np.sin(x) - 1)


def aps14_fp(x, n):
    if x <= 0:
        return 0
    return n / 20.0 * (1.0 / 1.5 + np.cos(x))


def aps14_fpp(x, n):
    if x <= 0:
        return 0
    return -n / 20.0 * (np.sin(x))


def aps15_f(x, n):
    r"""piecewise linear, constant outside of [0, 0.002/(1+n)]"""
    if x < 0:
        return -0.859
    if x > 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) - 1.859


def aps15_fp(x, n):
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000


def aps15_fpp(x, n):
    if not 0 <= x <= 2 * 1e-3 / (1 + n):
        return np.e - 1.859
    return np.exp((n + 1) * x / 2 * 1000) * (n + 1) / 2 * 1000 * (n + 1) / 2 * 1000


# Each APS test case has
# - a function and its two derivatives,
# - additional arguments,
# - a bracket enclosing a root,
# - the order of differentiability of the the function on this interval
# - a starting value for methods which don't require a bracket
# - the root (inside the bracket)
# - an Identifier of the test case
#
# Algorithm 748 is a bracketing algorithm so a bracketing interval was provided
# in [1] for each test case. Newton and Halley methods need a single
# starting point x0, which was chosen to be near the middle of the interval,
# unless that would have made the problem too easy.

_APS_TESTS_KEYS = ["f", "fprime", "fprime2", "args", "bracket", "smoothness", "x0", "root", "ID"]
_APS_TESTS = [
    [aps01_f, aps01_fp, aps01_fpp, (), [np.pi / 2, np.pi], np.inf, 3, 1.89549426703398094e+00, "aps.01.00"],
    [aps02_f, aps02_fp, aps02_fpp, (), [1 + 1e-9, 4 - 1e-9], np.inf, 2, 3.02291534727305677e+00, "aps.02.00"],
    [aps02_f, aps02_fp, aps02_fpp, (), [4 + 1e-9, 9 - 1e-9], np.inf, 5, 6.68375356080807848e+00, "aps.02.01"],
    [aps02_f, aps02_fp, aps02_fpp, (), [9 + 1e-9, 16 - 1e-9], np.inf, 10, 1.12387016550022114e+01, "aps.02.02"],
    [aps02_f, aps02_fp, aps02_fpp, (), [16 + 1e-9, 25 - 1e-9], np.inf, 17, 1.96760000806234103e+01, "aps.02.03"],
    [aps02_f, aps02_fp, aps02_fpp, (), [25 + 1e-9, 36 - 1e-9], np.inf, 26, 2.98282273265047557e+01, "aps.02.04"],
    [aps02_f, aps02_fp, aps02_fpp, (), [36 + 1e-9, 49 - 1e-9], np.inf, 37, 4.19061161952894139e+01, "aps.02.05"],
    [aps02_f, aps02_fp, aps02_fpp, (), [49 + 1e-9, 64 - 1e-9], np.inf, 50, 5.59535958001430913e+01, "aps.02.06"],
    [aps02_f, aps02_fp, aps02_fpp, (), [64 + 1e-9, 81 - 1e-9], np.inf, 65, 7.19856655865877997e+01, "aps.02.07"],
    [aps02_f, aps02_fp, aps02_fpp, (), [81 + 1e-9, 100 - 1e-9], np.inf, 82, 9.00088685391666701e+01, "aps.02.08"],
    [aps02_f, aps02_fp, aps02_fpp, (), [100 + 1e-9, 121 - 1e-9], np.inf, 101, 1.10026532748330197e+02, "aps.02.09"],
    [aps03_f, aps03_fp, aps03_fpp, (-40, -1), [-9, 31], np.inf, -2, 0, "aps.03.00"],
    [aps03_f, aps03_fp, aps03_fpp, (-100, -2), [-9, 31], np.inf, -2, 0, "aps.03.01"],
    [aps03_f, aps03_fp, aps03_fpp, (-200, -3), [-9, 31], np.inf, -2, 0, "aps.03.02"],
    [aps04_f, aps04_fp, aps04_fpp, (4, 0.2), [0, 5], np.inf, 2.5, 6.68740304976422006e-01, "aps.04.00"],
    [aps04_f, aps04_fp, aps04_fpp, (6, 0.2), [0, 5], np.inf, 2.5, 7.64724491331730039e-01, "aps.04.01"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 0.2), [0, 5], np.inf, 2.5, 8.17765433957942545e-01, "aps.04.02"],
    [aps04_f, aps04_fp, aps04_fpp, (10, 0.2), [0, 5], np.inf, 2.5, 8.51339922520784609e-01, "aps.04.03"],
    [aps04_f, aps04_fp, aps04_fpp, (12, 0.2), [0, 5], np.inf, 2.5, 8.74485272221167897e-01, "aps.04.04"],
    [aps04_f, aps04_fp, aps04_fpp, (4, 1), [0, 5], np.inf, 2.5, 1, "aps.04.05"],
    [aps04_f, aps04_fp, aps04_fpp, (6, 1), [0, 5], np.inf, 2.5, 1, "aps.04.06"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 1), [0, 5], np.inf, 2.5, 1, "aps.04.07"],
    [aps04_f, aps04_fp, aps04_fpp, (10, 1), [0, 5], np.inf, 2.5, 1, "aps.04.08"],
    [aps04_f, aps04_fp, aps04_fpp, (12, 1), [0, 5], np.inf, 2.5, 1, "aps.04.09"],
    [aps04_f, aps04_fp, aps04_fpp, (8, 1), [-0.95, 4.05], np.inf, 1.5, 1, "aps.04.10"],
    [aps04_f, aps04_fp, aps04_fpp, (10, 1), [-0.95, 4.05], np.inf, 1.5, 1, "aps.04.11"],
    [aps04_f, aps04_fp, aps04_fpp, (12, 1), [-0.95, 4.05], np.inf, 1.5, 1, "aps.04.12"],
    [aps04_f, aps04_fp, aps04_fpp, (14, 1), [-0.95, 4.05], np.inf, 1.5, 1, "aps.04.13"],
    [aps05_f, aps05_fp, aps05_fpp, (), [0, 1.5], np.inf, 1.3, np.pi / 6, "aps.05.00"],
    [aps06_f, aps06_fp, aps06_fpp, (1,), [0, 1], np.inf, 0.5, 4.22477709641236709e-01, "aps.06.00"],
    [aps06_f, aps06_fp, aps06_fpp, (2,), [0, 1], np.inf, 0.5, 3.06699410483203705e-01, "aps.06.01"],
    [aps06_f, aps06_fp, aps06_fpp, (3,), [0, 1], np.inf, 0.5, 2.23705457654662959e-01, "aps.06.02"],
    [aps06_f, aps06_fp, aps06_fpp, (4,), [0, 1], np.inf, 0.5, 1.71719147519508369e-01, "aps.06.03"],
    [aps06_f, aps06_fp, aps06_fpp, (5,), [0, 1], np.inf, 0.4, 1.38257155056824066e-01, "aps.06.04"],
    [aps06_f, aps06_fp, aps06_fpp, (20,), [0, 1], np.inf, 0.1, 3.46573590208538521e-02, "aps.06.05"],
    [aps06_f, aps06_fp, aps06_fpp, (40,), [0, 1], np.inf, 5e-02, 1.73286795139986315e-02, "aps.06.06"],
    [aps06_f, aps06_fp, aps06_fpp, (60,), [0, 1], np.inf, 1.0 / 30, 1.15524530093324210e-02, "aps.06.07"],
    [aps06_f, aps06_fp, aps06_fpp, (80,), [0, 1], np.inf, 2.5e-02, 8.66433975699931573e-03, "aps.06.08"],
    [aps06_f, aps06_fp, aps06_fpp, (100,), [0, 1], np.inf, 2e-02, 6.93147180559945415e-03, "aps.06.09"],
    [aps07_f, aps07_fp, aps07_fpp, (5,), [0, 1], np.inf, 0.4, 3.84025518406218985e-02, "aps.07.00"],
    [aps07_f, aps07_fp, aps07_fpp, (10,), [0, 1], np.inf, 0.4, 9.90000999800049949e-03, "aps.07.01"],
    [aps07_f, aps07_fp, aps07_fpp, (20,), [0, 1], np.inf, 0.4, 2.49375003906201174e-03, "aps.07.02"],
    [aps08_f, aps08_fp, aps08_fpp, (2,), [0, 1], np.inf, 0.9, 0.5, "aps.08.00"],
    [aps08_f, aps08_fp, aps08_fpp, (5,), [0, 1], np.inf, 0.9, 3.45954815848242059e-01, "aps.08.01"],
    [aps08_f, aps08_fp, aps08_fpp, (10,), [0, 1], np.inf, 0.9, 2.45122333753307220e-01, "aps.08.02"],
    [aps08_f, aps08_fp, aps08_fpp, (15,), [0, 1], np.inf, 0.9, 1.95547623536565629e-01, "aps.08.03"],
    [aps08_f, aps08_fp, aps08_fpp, (20,), [0, 1], np.inf, 0.9, 1.64920957276440960e-01, "aps.08.04"],
    [aps09_f, aps09_fp, aps09_fpp, (1,), [0, 1], np.inf, 0.5, 2.75508040999484394e-01, "aps.09.00"],
    [aps09_f, aps09_fp, aps09_fpp, (2,), [0, 1], np.inf, 0.5, 1.37754020499742197e-01, "aps.09.01"],
    [aps09_f, aps09_fp, aps09_fpp, (4,), [0, 1], np.inf, 0.5, 1.03052837781564422e-02, "aps.09.02"],
    [aps09_f, aps09_fp, aps09_fpp, (5,), [0, 1], np.inf, 0.5, 3.61710817890406339e-03, "aps.09.03"],
    [aps09_f, aps09_fp, aps09_fpp, (8,), [0, 1], np.inf, 0.5, 4.10872918496395375e-04, "aps.09.04"],
    [aps09_f, aps09_fp, aps09_fpp, (15,), [0, 1], np.inf, 0.5, 2.59895758929076292e-05, "aps.09.05"],
    [aps09_f, aps09_fp, aps09_fpp, (20,), [0, 1], np.inf, 0.5, 7.66859512218533719e-06, "aps.09.06"],
    [aps10_f, aps10_fp, aps10_fpp, (1,), [0, 1], np.inf, 0.9, 4.01058137541547011e-01, "aps.10.00"],
    [aps10_f, aps10_fp, aps10_fpp, (5,), [0, 1], np.inf, 0.9, 5.16153518757933583e-01, "aps.10.01"],
    [aps10_f, aps10_fp, aps10_fpp, (10,), [0, 1], np.inf, 0.9, 5.39522226908415781e-01, "aps.10.02"],
    [aps10_f, aps10_fp, aps10_fpp, (15,), [0, 1], np.inf, 0.9, 5.48182294340655241e-01, "aps.10.03"],
    [aps10_f, aps10_fp, aps10_fpp, (20,), [0, 1], np.inf, 0.9, 5.52704666678487833e-01, "aps.10.04"],
    [aps11_f, aps11_fp, aps11_fpp, (2,), [0.01, 1], np.inf, 1e-02, 1.0 / 2, "aps.11.00"],
    [aps11_f, aps11_fp, aps11_fpp, (5,), [0.01, 1], np.inf, 1e-02, 1.0 / 5, "aps.11.01"],
    [aps11_f, aps11_fp, aps11_fpp, (15,), [0.01, 1], np.inf, 1e-02, 1.0 / 15, "aps.11.02"],
    [aps11_f, aps11_fp, aps11_fpp, (20,), [0.01, 1], np.inf, 1e-02, 1.0 / 20, "aps.11.03"],
    [aps12_f, aps12_fp, aps12_fpp, (2,), [1, 100], np.inf, 1.1, 2, "aps.12.00"],
    [aps12_f, aps12_fp, aps12_fpp, (3,), [1, 100], np.inf, 1.1, 3, "aps.12.01"],
    [aps12_f, aps12_fp, aps12_fpp, (4,), [1, 100], np.inf, 1.1, 4, "aps.12.02"],
    [aps12_f, aps12_fp, aps12_fpp, (5,), [1, 100], np.inf, 1.1, 5, "aps.12.03"],
    [aps12_f, aps12_fp, aps12_fpp, (6,), [1, 100], np.inf, 1.1, 6, "aps.12.04"],
    [aps12_f, aps12_fp, aps12_fpp, (7,), [1, 100], np.inf, 1.1, 7, "aps.12.05"],
    [aps12_f, aps12_fp, aps12_fpp, (9,), [1, 100], np.inf, 1.1, 9, "aps.12.06"],
    [aps12_f, aps12_fp, aps12_fpp, (11,), [1, 100], np.inf, 1.1, 11, "aps.12.07"],
    [aps12_f, aps12_fp, aps12_fpp, (13,), [1, 100], np.inf, 1.1, 13, "aps.12.08"],
    [aps12_f, aps12_fp, aps12_fpp, (15,), [1, 100], np.inf, 1.1, 15, "aps.12.09"],
    [aps12_f, aps12_fp, aps12_fpp, (17,), [1, 100], np.inf, 1.1, 17, "aps.12.10"],
    [aps12_f, aps12_fp, aps12_fpp, (19,), [1, 100], np.inf, 1.1, 19, "aps.12.11"],
    [aps12_f, aps12_fp, aps12_fpp, (21,), [1, 100], np.inf, 1.1, 21, "aps.12.12"],
    [aps12_f, aps12_fp, aps12_fpp, (23,), [1, 100], np.inf, 1.1, 23, "aps.12.13"],
    [aps12_f, aps12_fp, aps12_fpp, (25,), [1, 100], np.inf, 1.1, 25, "aps.12.14"],
    [aps12_f, aps12_fp, aps12_fpp, (27,), [1, 100], np.inf, 1.1, 27, "aps.12.15"],
    [aps12_f, aps12_fp, aps12_fpp, (29,), [1, 100], np.inf, 1.1, 29, "aps.12.16"],
    [aps12_f, aps12_fp, aps12_fpp, (31,), [1, 100], np.inf, 1.1, 31, "aps.12.17"],
    [aps12_f, aps12_fp, aps12_fpp, (33,), [1, 100], np.inf, 1.1, 33, "aps.12.18"],
    [aps13_f, aps13_fp, aps13_fpp, (), [-1, 4], np.inf, 1.5, 1.54720911915117165e-02, "aps.13.00"],
    [aps14_f, aps14_fp, aps14_fpp, (1,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.00"],
    [aps14_f, aps14_fp, aps14_fpp, (2,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.01"],
    [aps14_f, aps14_fp, aps14_fpp, (3,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.02"],
    [aps14_f, aps14_fp, aps14_fpp, (4,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.03"],
    [aps14_f, aps14_fp, aps14_fpp, (5,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.04"],
    [aps14_f, aps14_fp, aps14_fpp, (6,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.05"],
    [aps14_f, aps14_fp, aps14_fpp, (7,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.06"],
    [aps14_f, aps14_fp, aps14_fpp, (8,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.07"],
    [aps14_f, aps14_fp, aps14_fpp, (9,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.08"],
    [aps14_f, aps14_fp, aps14_fpp, (10,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.09"],
    [aps14_f, aps14_fp, aps14_fpp, (11,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.10"],
    [aps14_f, aps14_fp, aps14_fpp, (12,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.11"],
    [aps14_f, aps14_fp, aps14_fpp, (13,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.12"],
    [aps14_f, aps14_fp, aps14_fpp, (14,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.13"],
    [aps14_f, aps14_fp, aps14_fpp, (15,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.14"],
    [aps14_f, aps14_fp, aps14_fpp, (16,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.15"],
    [aps14_f, aps14_fp, aps14_fpp, (17,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.16"],
    [aps14_f, aps14_fp, aps14_fpp, (18,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.17"],
    [aps14_f, aps14_fp, aps14_fpp, (19,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.18"],
    [aps14_f, aps14_fp, aps14_fpp, (20,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.19"],
    [aps14_f, aps14_fp, aps14_fpp, (21,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.20"],
    [aps14_f, aps14_fp, aps14_fpp, (22,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.21"],
    [aps14_f, aps14_fp, aps14_fpp, (23,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.22"],
    [aps14_f, aps14_fp, aps14_fpp, (24,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.23"],
    [aps14_f, aps14_fp, aps14_fpp, (25,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.24"],
    [aps14_f, aps14_fp, aps14_fpp, (26,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.25"],
    [aps14_f, aps14_fp, aps14_fpp, (27,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.26"],
    [aps14_f, aps14_fp, aps14_fpp, (28,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.27"],
    [aps14_f, aps14_fp, aps14_fpp, (29,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.28"],
    [aps14_f, aps14_fp, aps14_fpp, (30,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.29"],
    [aps14_f, aps14_fp, aps14_fpp, (31,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.30"],
    [aps14_f, aps14_fp, aps14_fpp, (32,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.31"],
    [aps14_f, aps14_fp, aps14_fpp, (33,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.32"],
    [aps14_f, aps14_fp, aps14_fpp, (34,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.33"],
    [aps14_f, aps14_fp, aps14_fpp, (35,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.34"],
    [aps14_f, aps14_fp, aps14_fpp, (36,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.35"],
    [aps14_f, aps14_fp, aps14_fpp, (37,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.36"],
    [aps14_f, aps14_fp, aps14_fpp, (38,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.37"],
    [aps14_f, aps14_fp, aps14_fpp, (39,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.38"],
    [aps14_f, aps14_fp, aps14_fpp, (40,), [-1000, np.pi / 2], 0, 1, 6.23806518961612433e-01, "aps.14.39"],
    [aps15_f, aps15_fp, aps15_fpp, (20,), [-1000, 1e-4], 0, -2, 5.90513055942197166e-05, "aps.15.00"],
    [aps15_f, aps15_fp, aps15_fpp, (21,), [-1000, 1e-4], 0, -2, 5.63671553399369967e-05, "aps.15.01"],
    [aps15_f, aps15_fp, aps15_fpp, (22,), [-1000, 1e-4], 0, -2, 5.39164094555919196e-05, "aps.15.02"],
    [aps15_f, aps15_fp, aps15_fpp, (23,), [-1000, 1e-4], 0, -2, 5.16698923949422470e-05, "aps.15.03"],
    [aps15_f, aps15_fp, aps15_fpp, (24,), [-1000, 1e-4], 0, -2, 4.96030966991445609e-05, "aps.15.04"],
    [aps15_f, aps15_fp, aps15_fpp, (25,), [-1000, 1e-4], 0, -2, 4.76952852876389951e-05, "aps.15.05"],
    [aps15_f, aps15_fp, aps15_fpp, (26,), [-1000, 1e-4], 0, -2, 4.59287932399486662e-05, "aps.15.06"],
    [aps15_f, aps15_fp, aps15_fpp, (27,), [-1000, 1e-4], 0, -2, 4.42884791956647841e-05, "aps.15.07"],
    [aps15_f, aps15_fp, aps15_fpp, (28,), [-1000, 1e-4], 0, -2, 4.27612902578832391e-05, "aps.15.08"],
    [aps15_f, aps15_fp, aps15_fpp, (29,), [-1000, 1e-4], 0, -2, 4.13359139159538030e-05, "aps.15.09"],
    [aps15_f, aps15_fp, aps15_fpp, (30,), [-1000, 1e-4], 0, -2, 4.00024973380198076e-05, "aps.15.10"],
    [aps15_f, aps15_fp, aps15_fpp, (31,), [-1000, 1e-4], 0, -2, 3.87524192962066869e-05, "aps.15.11"],
    [aps15_f, aps15_fp, aps15_fpp, (32,), [-1000, 1e-4], 0, -2, 3.75781035599579910e-05, "aps.15.12"],
    [aps15_f, aps15_fp, aps15_fpp, (33,), [-1000, 1e-4], 0, -2, 3.64728652199592355e-05, "aps.15.13"],
    [aps15_f, aps15_fp, aps15_fpp, (34,), [-1000, 1e-4], 0, -2, 3.54307833565318273e-05, "aps.15.14"],
    [aps15_f, aps15_fp, aps15_fpp, (35,), [-1000, 1e-4], 0, -2, 3.44465949299614980e-05, "aps.15.15"],
    [aps15_f, aps15_fp, aps15_fpp, (36,), [-1000, 1e-4], 0, -2, 3.35156058778003705e-05, "aps.15.16"],
    [aps15_f, aps15_fp, aps15_fpp, (37,), [-1000, 1e-4], 0, -2, 3.26336162494372125e-05, "aps.15.17"],
    [aps15_f, aps15_fp, aps15_fpp, (38,), [-1000, 1e-4], 0, -2, 3.17968568584260013e-05, "aps.15.18"],
    [aps15_f, aps15_fp, aps15_fpp, (39,), [-1000, 1e-4], 0, -2, 3.10019354369653455e-05, "aps.15.19"],
    [aps15_f, aps15_fp, aps15_fpp, (40,), [-1000, 1e-4], 0, -2, 3.02457906702100968e-05, "aps.15.20"],
    [aps15_f, aps15_fp, aps15_fpp, (100,), [-1000, 1e-4], 0, -2, 1.22779942324615231e-05, "aps.15.21"],
    [aps15_f, aps15_fp, aps15_fpp, (200,), [-1000, 1e-4], 0, -2, 6.16953939044086617e-06, "aps.15.22"],
    [aps15_f, aps15_fp, aps15_fpp, (300,), [-1000, 1e-4], 0, -2, 4.11985852982928163e-06, "aps.15.23"],
    [aps15_f, aps15_fp, aps15_fpp, (400,), [-1000, 1e-4], 0, -2, 3.09246238772721682e-06, "aps.15.24"],
    [aps15_f, aps15_fp, aps15_fpp, (500,), [-1000, 1e-4], 0, -2, 2.47520442610501789e-06, "aps.15.25"],
    [aps15_f, aps15_fp, aps15_fpp, (600,), [-1000, 1e-4], 0, -2, 2.06335676785127107e-06, "aps.15.26"],
    [aps15_f, aps15_fp, aps15_fpp, (700,), [-1000, 1e-4], 0, -2, 1.76901200781542651e-06, "aps.15.27"],
    [aps15_f, aps15_fp, aps15_fpp, (800,), [-1000, 1e-4], 0, -2, 1.54816156988591016e-06, "aps.15.28"],
    [aps15_f, aps15_fp, aps15_fpp, (900,), [-1000, 1e-4], 0, -2, 1.37633453660223511e-06, "aps.15.29"],
    [aps15_f, aps15_fp, aps15_fpp, (1000,), [-1000, 1e-4], 0, -2, 1.23883857889971403e-06, "aps.15.30"]
]

_APS_TESTS_DICTS = [dict(zip(_APS_TESTS_KEYS, testcase)) for testcase in _APS_TESTS]


#   ##################
#   "complex" test cases
#   A few simple, complex-valued, functions, defined on the complex plane.


def cplx01_f(z, n, a):
    r"""z**n-a:  Use to find the nth root of a"""
    return z**n - a


def cplx01_fp(z, n, a):
    return n * z**(n - 1)


def cplx01_fpp(z, n, a):
    return n * (n - 1) * z**(n - 2)


def cplx02_f(z, a):
    r"""e**z - a: Use to find the log of a"""
    return np.exp(z) - a


def cplx02_fp(z, a):
    return np.exp(z)


def cplx02_fpp(z, a):
    return np.exp(z)


# Each "complex" test case has
# - a function and its two derivatives,
# - additional arguments,
# - the order of differentiability of the the function on this interval
# - two starting values x0 and x1
# - the root
# - an Identifier of the test case
#
# Algorithm 748 is a bracketing algorithm so a bracketing interval was provided
# in [1] for each test case. Newton and Halley need a single starting point
# x0, which was chosen to be near the middle of the interval, unless that
# would make the problem too easy.


_COMPLEX_TESTS_KEYS = ["f", "fprime", "fprime2", "args", "smoothness", "x0", "x1", "root", "ID"]
_COMPLEX_TESTS = [
    [cplx01_f, cplx01_fp, cplx01_fpp, (2, -1), np.inf, (1 + 1j), (0.5 + 0.5j), 1j, "complex.01.00"],
    [cplx01_f, cplx01_fp, cplx01_fpp, (3, 1), np.inf, (-1 + 1j), (-0.5 + 2.0j), (-0.5 + np.sqrt(3) / 2 * 1.0j),
     "complex.01.01"],
    [cplx01_f, cplx01_fp, cplx01_fpp, (3, -1), np.inf, 1j, (0.5 + 0.5j), (0.5 + np.sqrt(3) / 2 * 1.0j),
     "complex.01.02"],
    [cplx01_f, cplx01_fp, cplx01_fpp, (3, 8), np.inf, 5, 4, 2, "complex.01.03"],
    [cplx02_f, cplx02_fp, cplx02_fpp, (-1,), np.inf, (1 + 2j), (0.5 + 0.5j), np.pi * 1.0j, "complex.02.00"],
    [cplx02_f, cplx02_fp, cplx02_fpp, (1j,), np.inf, (1 + 2j), (0.5 + 0.5j), np.pi * 0.5j, "complex.02.01"],
]

_COMPLEX_TESTS_DICTS = [dict(zip(_COMPLEX_TESTS_KEYS, testcase)) for testcase in _COMPLEX_TESTS]


def _add_a_b(tests):
    r"""Add "a" and "b" keys to each test from the "bracket" value"""
    for d in tests:
        for k, v in zip(['a', 'b'], d.get('bracket', [])):
            d[k] = v


_add_a_b(_ORIGINAL_TESTS_DICTS)
_add_a_b(_APS_TESTS_DICTS)
_add_a_b(_COMPLEX_TESTS_DICTS)


def get_tests(collection='original', smoothness=None):
    r"""Return the requested collection of test cases, as an array of dicts with subset-specific keys

    Allowed values of collection:
    'original': The original benchmarking functions.
         Real-valued functions of real-valued inputs on an interval with a zero.
         f1, .., f3 are continuous and infinitely differentiable
         f4 has a single discontinuity at the root
         f5 has a root at 1 replacing a 1st order pole
         f6 is randomly positive on one side of the root, randomly negative on the other
    'aps': The test problems in the TOMS "Algorithm 748: Enclosing Zeros of Continuous Functions"
         paper by Alefeld, Potra and Shi. Real-valued functions of
         real-valued inputs on an interval with a zero.
         Suitable for methods which start with an enclosing interval, and
         derivatives up to 2nd order.
    'complex': Some complex-valued functions of complex-valued inputs.
         No enclosing bracket is provided.
         Suitable for methods which use one or more starting values, and
         derivatives up to 2nd order.

    The dictionary keys will be a subset of
    ["f", "fprime", "fprime2", "args", "bracket", "a", b", "smoothness", "x0", "x1", "root", "ID"]
     """
    collection = collection or "original"
    subsets = {"aps": _APS_TESTS_DICTS,
               "complex": _COMPLEX_TESTS_DICTS,
               "original": _ORIGINAL_TESTS_DICTS}
    tests = subsets.get(collection, [])
    if smoothness is not None:
        tests = [tc for tc in tests if tc['smoothness'] >= smoothness]
    return tests


# Backwards compatibility
methods = [cc.bisect, cc.ridder, cc.brenth, cc.brentq]
mstrings = ['cc.bisect', 'cc.ridder', 'cc.brenth', 'cc.brentq']
functions = [f2, f3, f4, f5, f6]
fstrings = ['f2', 'f3', 'f4', 'f5', 'f6']
