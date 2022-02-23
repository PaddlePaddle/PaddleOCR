"""
Test Cython optimize zeros API functions: ``bisect``, ``ridder``, ``brenth``,
and ``brentq`` in `scipy.optimize.cython_optimize`, by finding the roots of a
3rd order polynomial given a sequence of constant terms, ``a0``, and fixed 1st,
2nd, and 3rd order terms in ``args``.

.. math::

    f(x, a0, args) =  ((args[2]*x + args[1])*x + args[0])*x + a0

The 3rd order polynomial function is written in Cython and called in a Python
wrapper named after the zero function. See the private ``_zeros`` Cython module
in `scipy.optimize.cython_optimze` for more information.
"""

import numpy.testing as npt
from scipy.optimize.cython_optimize import _zeros

# CONSTANTS
# Solve x**3 - A0 = 0  for A0 = [2.0, 2.1, ..., 2.9].
# The ARGS have 3 elements just to show how this could be done for any cubic
# polynomial.
A0 = tuple(-2.0 - x/10.0 for x in range(10))  # constant term
ARGS = (0.0, 0.0, 1.0)  # 1st, 2nd, and 3rd order terms
XLO, XHI = 0.0, 2.0  # first and second bounds of zeros functions
# absolute and relative tolerances and max iterations for zeros functions
XTOL, RTOL, MITR = 0.001, 0.001, 10
EXPECTED = [(-a0) ** (1.0/3.0) for a0 in A0]
# = [1.2599210498948732,
#    1.2805791649874942,
#    1.300591446851387,
#    1.3200061217959123,
#    1.338865900164339,
#    1.3572088082974532,
#    1.375068867074141,
#    1.3924766500838337,
#    1.4094597464129783,
#    1.4260431471424087]


# test bisect
def test_bisect():
    npt.assert_allclose(
        EXPECTED,
        list(
            _zeros.loop_example('bisect', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)
        ),
        rtol=RTOL, atol=XTOL
    )


# test ridder
def test_ridder():
    npt.assert_allclose(
        EXPECTED,
        list(
            _zeros.loop_example('ridder', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)
        ),
        rtol=RTOL, atol=XTOL
    )


# test brenth
def test_brenth():
    npt.assert_allclose(
        EXPECTED,
        list(
            _zeros.loop_example('brenth', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)
        ),
        rtol=RTOL, atol=XTOL
    )


# test brentq
def test_brentq():
    npt.assert_allclose(
        EXPECTED,
        list(
            _zeros.loop_example('brentq', A0, ARGS, XLO, XHI, XTOL, RTOL, MITR)
        ),
        rtol=RTOL, atol=XTOL
    )


# test brentq with full output
def test_brentq_full_output():
    output = _zeros.full_output_example(
        (A0[0],) + ARGS, XLO, XHI, XTOL, RTOL, MITR)
    npt.assert_allclose(EXPECTED[0], output['root'], rtol=RTOL, atol=XTOL)
    npt.assert_equal(6, output['iterations'])
    npt.assert_equal(7, output['funcalls'])
    npt.assert_equal(0, output['error_num'])
