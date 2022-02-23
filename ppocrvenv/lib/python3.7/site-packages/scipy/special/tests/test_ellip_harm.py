#
# Tests for the Ellipsoidal Harmonic Function,
# Distributed under the same license as SciPy itself.
#

import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
                           assert_, suppress_warnings)
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi


def test_ellip_potential():
    def change_coefficient(lambda1, mu, nu, h2, k2):
        x = sqrt(lambda1**2*mu**2*nu**2/(h2*k2))
        y = sqrt((lambda1**2 - h2)*(mu**2 - h2)*(h2 - nu**2)/(h2*(k2 - h2)))
        z = sqrt((lambda1**2 - k2)*(k2 - mu**2)*(k2 - nu**2)/(k2*(k2 - h2)))
        return x, y, z

    def solid_int_ellip(lambda1, mu, nu, n, p, h2, k2):
        return (ellip_harm(h2, k2, n, p, lambda1)*ellip_harm(h2, k2, n, p, mu)
               * ellip_harm(h2, k2, n, p, nu))

    def solid_int_ellip2(lambda1, mu, nu, n, p, h2, k2):
        return (ellip_harm_2(h2, k2, n, p, lambda1)
                * ellip_harm(h2, k2, n, p, mu)*ellip_harm(h2, k2, n, p, nu))

    def summation(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
        tol = 1e-8
        sum1 = 0
        for n in range(20):
            xsum = 0
            for p in range(1, 2*n+2):
                xsum += (4*pi*(solid_int_ellip(lambda2, mu2, nu2, n, p, h2, k2)
                    * solid_int_ellip2(lambda1, mu1, nu1, n, p, h2, k2)) /
                    (ellip_normal(h2, k2, n, p)*(2*n + 1)))
            if abs(xsum) < 0.1*tol*abs(sum1):
                break
            sum1 += xsum
        return sum1, xsum

    def potential(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
        x1, y1, z1 = change_coefficient(lambda1, mu1, nu1, h2, k2)
        x2, y2, z2 = change_coefficient(lambda2, mu2, nu2, h2, k2)
        res = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return 1/res

    pts = [
        (120, sqrt(19), 2, 41, sqrt(17), 2, 15, 25),
        (120, sqrt(16), 3.2, 21, sqrt(11), 2.9, 11, 20),
       ]

    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        sup.filter(IntegrationWarning, "The maximum number of subdivisions")

        for p in pts:
            err_msg = repr(p)
            exact = potential(*p)
            result, last_term = summation(*p)
            assert_allclose(exact, result, atol=0, rtol=1e-8, err_msg=err_msg)
            assert_(abs(result - exact) < 10*abs(last_term), err_msg)


def test_ellip_norm():

    def G01(h2, k2):
        return 4*pi

    def G11(h2, k2):
        return 4*pi*h2*k2/3

    def G12(h2, k2):
        return 4*pi*h2*(k2 - h2)/3

    def G13(h2, k2):
        return 4*pi*k2*(k2 - h2)/3

    def G22(h2, k2):
        res = (2*(h2**4 + k2**4) - 4*h2*k2*(h2**2 + k2**2) + 6*h2**2*k2**2 +
        sqrt(h2**2 + k2**2 - h2*k2)*(-2*(h2**3 + k2**3) + 3*h2*k2*(h2 + k2)))
        return 16*pi/405*res

    def G21(h2, k2):
        res = (2*(h2**4 + k2**4) - 4*h2*k2*(h2**2 + k2**2) + 6*h2**2*k2**2
        + sqrt(h2**2 + k2**2 - h2*k2)*(2*(h2**3 + k2**3) - 3*h2*k2*(h2 + k2)))
        return 16*pi/405*res

    def G23(h2, k2):
        return 4*pi*h2**2*k2*(k2 - h2)/15

    def G24(h2, k2):
        return 4*pi*h2*k2**2*(k2 - h2)/15

    def G25(h2, k2):
        return 4*pi*h2*k2*(k2 - h2)**2/15

    def G32(h2, k2):
        res = (16*(h2**4 + k2**4) - 36*h2*k2*(h2**2 + k2**2) + 46*h2**2*k2**2
        + sqrt(4*(h2**2 + k2**2) - 7*h2*k2)*(-8*(h2**3 + k2**3) +
        11*h2*k2*(h2 + k2)))
        return 16*pi/13125*k2*h2*res

    def G31(h2, k2):
        res = (16*(h2**4 + k2**4) - 36*h2*k2*(h2**2 + k2**2) + 46*h2**2*k2**2
        + sqrt(4*(h2**2 + k2**2) - 7*h2*k2)*(8*(h2**3 + k2**3) -
        11*h2*k2*(h2 + k2)))
        return 16*pi/13125*h2*k2*res

    def G34(h2, k2):
        res = (6*h2**4 + 16*k2**4 - 12*h2**3*k2 - 28*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(h2**2 + 4*k2**2 - h2*k2)*(-6*h2**3 - 8*k2**3 + 9*h2**2*k2 +
                                            13*h2*k2**2))
        return 16*pi/13125*h2*(k2 - h2)*res

    def G33(h2, k2):
        res = (6*h2**4 + 16*k2**4 - 12*h2**3*k2 - 28*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(h2**2 + 4*k2**2 - h2*k2)*(6*h2**3 + 8*k2**3 - 9*h2**2*k2 -
        13*h2*k2**2))
        return 16*pi/13125*h2*(k2 - h2)*res

    def G36(h2, k2):
        res = (16*h2**4 + 6*k2**4 - 28*h2**3*k2 - 12*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(4*h2**2 + k2**2 - h2*k2)*(-8*h2**3 - 6*k2**3 + 13*h2**2*k2 +
        9*h2*k2**2))
        return 16*pi/13125*k2*(k2 - h2)*res

    def G35(h2, k2):
        res = (16*h2**4 + 6*k2**4 - 28*h2**3*k2 - 12*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(4*h2**2 + k2**2 - h2*k2)*(8*h2**3 + 6*k2**3 - 13*h2**2*k2 -
        9*h2*k2**2))
        return 16*pi/13125*k2*(k2 - h2)*res

    def G37(h2, k2):
        return 4*pi*h2**2*k2**2*(k2 - h2)**2/105

    known_funcs = {(0, 1): G01, (1, 1): G11, (1, 2): G12, (1, 3): G13,
                   (2, 1): G21, (2, 2): G22, (2, 3): G23, (2, 4): G24,
                   (2, 5): G25, (3, 1): G31, (3, 2): G32, (3, 3): G33,
                   (3, 4): G34, (3, 5): G35, (3, 6): G36, (3, 7): G37}

    def _ellip_norm(n, p, h2, k2):
        func = known_funcs[n, p]
        return func(h2, k2)
    _ellip_norm = np.vectorize(_ellip_norm)

    def ellip_normal_known(h2, k2, n, p):
        return _ellip_norm(n, p, h2, k2)

    # generate both large and small h2 < k2 pairs
    np.random.seed(1234)
    h2 = np.random.pareto(0.5, size=1)
    k2 = h2 * (1 + np.random.pareto(0.5, size=h2.size))

    points = []
    for n in range(4):
        for p in range(1, 2*n+2):
            points.append((h2, k2, np.full(h2.size, n), np.full(h2.size, p)))
    points = np.array(points)
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        assert_func_equal(ellip_normal, ellip_normal_known, points, rtol=1e-12)


def test_ellip_harm_2():

    def I1(h2, k2, s):
        res = (ellip_harm_2(h2, k2, 1, 1, s)/(3 * ellip_harm(h2, k2, 1, 1, s))
        + ellip_harm_2(h2, k2, 1, 2, s)/(3 * ellip_harm(h2, k2, 1, 2, s)) +
        ellip_harm_2(h2, k2, 1, 3, s)/(3 * ellip_harm(h2, k2, 1, 3, s)))
        return res

    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        assert_almost_equal(I1(5, 8, 10), 1/(10*sqrt((100-5)*(100-8))))

        # Values produced by code from arXiv:1204.0267
        assert_almost_equal(ellip_harm_2(5, 8, 2, 1, 10), 0.00108056853382)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 2, 10), 0.00105820513809)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 3, 10), 0.00106058384743)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 4, 10), 0.00106774492306)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 5, 10), 0.00107976356454)


def test_ellip_harm():

    def E01(h2, k2, s):
        return 1

    def E11(h2, k2, s):
        return s

    def E12(h2, k2, s):
        return sqrt(abs(s*s - h2))

    def E13(h2, k2, s):
        return sqrt(abs(s*s - k2))

    def E21(h2, k2, s):
        return s*s - 1/3*((h2 + k2) + sqrt(abs((h2 + k2)*(h2 + k2)-3*h2*k2)))

    def E22(h2, k2, s):
        return s*s - 1/3*((h2 + k2) - sqrt(abs((h2 + k2)*(h2 + k2)-3*h2*k2)))

    def E23(h2, k2, s):
        return s * sqrt(abs(s*s - h2))

    def E24(h2, k2, s):
        return s * sqrt(abs(s*s - k2))

    def E25(h2, k2, s):
        return sqrt(abs((s*s - h2)*(s*s - k2)))

    def E31(h2, k2, s):
        return s*s*s - (s/5)*(2*(h2 + k2) + sqrt(4*(h2 + k2)*(h2 + k2) -
        15*h2*k2))

    def E32(h2, k2, s):
        return s*s*s - (s/5)*(2*(h2 + k2) - sqrt(4*(h2 + k2)*(h2 + k2) -
        15*h2*k2))

    def E33(h2, k2, s):
        return sqrt(abs(s*s - h2))*(s*s - 1/5*((h2 + 2*k2) + sqrt(abs((h2 +
        2*k2)*(h2 + 2*k2) - 5*h2*k2))))

    def E34(h2, k2, s):
        return sqrt(abs(s*s - h2))*(s*s - 1/5*((h2 + 2*k2) - sqrt(abs((h2 +
        2*k2)*(h2 + 2*k2) - 5*h2*k2))))

    def E35(h2, k2, s):
        return sqrt(abs(s*s - k2))*(s*s - 1/5*((2*h2 + k2) + sqrt(abs((2*h2
        + k2)*(2*h2 + k2) - 5*h2*k2))))

    def E36(h2, k2, s):
        return sqrt(abs(s*s - k2))*(s*s - 1/5*((2*h2 + k2) - sqrt(abs((2*h2
        + k2)*(2*h2 + k2) - 5*h2*k2))))

    def E37(h2, k2, s):
        return s * sqrt(abs((s*s - h2)*(s*s - k2)))

    assert_equal(ellip_harm(5, 8, 1, 2, 2.5, 1, 1),
    ellip_harm(5, 8, 1, 2, 2.5))

    known_funcs = {(0, 1): E01, (1, 1): E11, (1, 2): E12, (1, 3): E13,
                   (2, 1): E21, (2, 2): E22, (2, 3): E23, (2, 4): E24,
                   (2, 5): E25, (3, 1): E31, (3, 2): E32, (3, 3): E33,
                   (3, 4): E34, (3, 5): E35, (3, 6): E36, (3, 7): E37}

    point_ref = []

    def ellip_harm_known(h2, k2, n, p, s):
        for i in range(h2.size):
            func = known_funcs[(int(n[i]), int(p[i]))]
            point_ref.append(func(h2[i], k2[i], s[i]))
        return point_ref

    np.random.seed(1234)
    h2 = np.random.pareto(0.5, size=30)
    k2 = h2*(1 + np.random.pareto(0.5, size=h2.size))
    s = np.random.pareto(0.5, size=h2.size)
    points = []
    for i in range(h2.size):
        for n in range(4):
            for p in range(1, 2*n+2):
                points.append((h2[i], k2[i], n, p, s[i]))
    points = np.array(points)
    assert_func_equal(ellip_harm, ellip_harm_known, points, rtol=1e-12)


def test_ellip_harm_invalid_p():
    # Regression test. This should return nan.
    n = 4
    # Make p > 2*n + 1.
    p = 2*n + 2
    result = ellip_harm(0.5, 2.0, n, p, 0.2)
    assert np.isnan(result)
