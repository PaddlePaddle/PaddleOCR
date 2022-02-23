"""
Test SciPy functions versus mpmath, if available.

"""
import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools

from distutils.version import LooseVersion

import scipy.special as sc
from scipy.special._testutils import (
    MissingModule, check_version, FuncData,
    assert_func_equal)
from scipy.special._mptestutils import (
    Arg, FixedArg, ComplexArg, IntArg, assert_mpmath_equal,
    nonfunctional_tooslow, trace_args, time_limited, exception_to_nan,
    inf_to_nan)
from scipy.special._ufuncs import (
    _sinpi, _cospi, _lgam1p, _lanczos_sum_expg_scaled, _log1pmx,
    _igam_fac)

try:
    import mpmath
except ImportError:
    mpmath = MissingModule('mpmath')


# ------------------------------------------------------------------------------
# expi
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.10')
def test_expi_complex():
    dataset = []
    for r in np.logspace(-99, 2, 10):
        for p in np.linspace(0, 2*np.pi, 30):
            z = r*np.exp(1j*p)
            dataset.append((z, complex(mpmath.ei(z))))
    dataset = np.array(dataset, dtype=np.complex_)

    FuncData(sc.expi, dataset, 0, 1).check()


# ------------------------------------------------------------------------------
# expn
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
def test_expn_large_n():
    # Test the transition to the asymptotic regime of n.
    dataset = []
    for n in [50, 51]:
        for x in np.logspace(0, 4, 200):
            with mpmath.workdps(100):
                dataset.append((n, x, float(mpmath.expint(n, x))))
    dataset = np.asarray(dataset)

    FuncData(sc.expn, dataset, (0, 1), 2, rtol=1e-13).check()

# ------------------------------------------------------------------------------
# hyp0f1
# ------------------------------------------------------------------------------


@check_version(mpmath, '0.19')
def test_hyp0f1_gh5764():
    # Do a small and somewhat systematic test that runs quickly
    dataset = []
    axis = [-99.5, -9.5, -0.5, 0.5, 9.5, 99.5]
    for v in axis:
        for x in axis:
            for y in axis:
                z = x + 1j*y
                # mpmath computes the answer correctly at dps ~ 17 but
                # fails for 20 < dps < 120 (uses a different method);
                # set the dps high enough that this isn't an issue
                with mpmath.workdps(120):
                    res = complex(mpmath.hyp0f1(v, z))
                dataset.append((v, z, res))
    dataset = np.array(dataset)

    FuncData(lambda v, z: sc.hyp0f1(v.real, z), dataset, (0, 1), 2,
             rtol=1e-13).check()


@check_version(mpmath, '0.19')
def test_hyp0f1_gh_1609():
    # this is a regression test for gh-1609
    vv = np.linspace(150, 180, 21)
    af = sc.hyp0f1(vv, 0.5)
    mf = np.array([mpmath.hyp0f1(v, 0.5) for v in vv])
    assert_allclose(af, mf.astype(float), rtol=1e-12)


# ------------------------------------------------------------------------------
# hyperu
# ------------------------------------------------------------------------------

@check_version(mpmath, '1.1.0')
def test_hyperu_around_0():
    dataset = []
    # DLMF 13.2.14-15 test points.
    for n in np.arange(-5, 5):
        for b in np.linspace(-5, 5, 20):
            a = -n
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
            a = -n + b - 1
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    # DLMF 13.2.16-22 test points.
    for a in [-10.5, -1.5, -0.5, 0, 0.5, 1, 10]:
        for b in [-1.0, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
            dataset.append((a, b, 0, float(mpmath.hyperu(a, b, 0))))
    dataset = np.array(dataset)

    FuncData(sc.hyperu, dataset, (0, 1, 2), 3, rtol=1e-15, atol=5e-13).check()

# ------------------------------------------------------------------------------
# hyp2f1
# ------------------------------------------------------------------------------

@check_version(mpmath, '1.0.0')
def test_hyp2f1_strange_points():
    pts = [
        (2, -1, -1, 0.7),  # expected: 2.4
        (2, -2, -2, 0.7),  # expected: 3.87
    ]
    pts += list(itertools.product([2, 1, -0.7, -1000], repeat=4))
    pts = [
        (a, b, c, x) for a, b, c, x in pts
        if b == c and round(b) == b and b < 0 and b != -1000
    ]
    kw = dict(eliminate=True)
    dataset = [p + (float(mpmath.hyp2f1(*p, **kw)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-10).check()


@check_version(mpmath, '0.13')
def test_hyp2f1_real_some_points():
    pts = [
        (1, 2, 3, 0),
        (1./3, 2./3, 5./6, 27./32),
        (1./4, 1./2, 3./4, 80./81),
        (2,-2, -3, 3),
        (2, -3, -2, 3),
        (2, -1.5, -1.5, 3),
        (1, 2, 3, 0),
        (0.7235, -1, -5, 0.3),
        (0.25, 1./3, 2, 0.999),
        (0.25, 1./3, 2, -1),
        (2, 3, 5, 0.99),
        (3./2, -0.5, 3, 0.99),
        (2, 2.5, -3.25, 0.999),
        (-8, 18.016500331508873, 10.805295997850628, 0.90875647507000001),
        (-10, 900, -10.5, 0.99),
        (-10, 900, 10.5, 0.99),
        (-1, 2, 1, 1.0),
        (-1, 2, 1, -1.0),
        (-3, 13, 5, 1.0),
        (-3, 13, 5, -1.0),
        (0.5, 1 - 270.5, 1.5, 0.999**2),  # from issue 1561
    ]
    dataset = [p + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-10).check()


@check_version(mpmath, '0.14')
def test_hyp2f1_some_points_2():
    # Taken from mpmath unit tests -- this point failed for mpmath 0.13 but
    # was fixed in their SVN since then
    pts = [
        (112, (51,10), (-9,10), -0.99999),
        (10,-900,10.5,0.99),
        (10,-900,-10.5,0.99),
    ]

    def fev(x):
        if isinstance(x, tuple):
            return float(x[0]) / x[1]
        else:
            return x

    dataset = [tuple(map(fev, p)) + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-10).check()


@check_version(mpmath, '0.13')
def test_hyp2f1_real_some():
    dataset = []
    for a in [-10, -5, -1.8, 1.8, 5, 10]:
        for b in [-2.5, -1, 1, 7.4]:
            for c in [-9, -1.8, 5, 20.4]:
                for z in [-10, -1.01, -0.99, 0, 0.6, 0.95, 1.5, 10]:
                    try:
                        v = float(mpmath.hyp2f1(a, b, c, z))
                    except Exception:
                        continue
                    dataset.append((a, b, c, z, v))
    dataset = np.array(dataset, dtype=np.float_)

    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0,1,2,3), 4, rtol=1e-9,
                 ignore_inf_sign=True).check()


@check_version(mpmath, '0.12')
@pytest.mark.slow
def test_hyp2f1_real_random():
    npoints = 500
    dataset = np.zeros((npoints, 5), np.float_)

    np.random.seed(1234)
    dataset[:, 0] = np.random.pareto(1.5, npoints)
    dataset[:, 1] = np.random.pareto(1.5, npoints)
    dataset[:, 2] = np.random.pareto(1.5, npoints)
    dataset[:, 3] = 2*np.random.rand(npoints) - 1

    dataset[:, 0] *= (-1)**np.random.randint(2, npoints)
    dataset[:, 1] *= (-1)**np.random.randint(2, npoints)
    dataset[:, 2] *= (-1)**np.random.randint(2, npoints)

    for ds in dataset:
        if mpmath.__version__ < '0.14':
            # mpmath < 0.14 fails for c too much smaller than a, b
            if abs(ds[:2]).max() > abs(ds[2]):
                ds[2] = abs(ds[:2]).max()
        ds[4] = float(mpmath.hyp2f1(*tuple(ds[:4])))

    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-9).check()

# ------------------------------------------------------------------------------
# erf (complex)
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.14')
def test_erf_complex():
    # need to increase mpmath precision for this test
    old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
    try:
        mpmath.mp.dps = 70
        x1, y1 = np.meshgrid(np.linspace(-10, 1, 31), np.linspace(-10, 1, 11))
        x2, y2 = np.meshgrid(np.logspace(-80, .8, 31), np.logspace(-80, .8, 11))
        points = np.r_[x1.ravel(),x2.ravel()] + 1j*np.r_[y1.ravel(), y2.ravel()]

        assert_func_equal(sc.erf, lambda x: complex(mpmath.erf(x)), points,
                          vectorized=False, rtol=1e-13)
        assert_func_equal(sc.erfc, lambda x: complex(mpmath.erfc(x)), points,
                          vectorized=False, rtol=1e-13)
    finally:
        mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec


# ------------------------------------------------------------------------------
# lpmv
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.15')
def test_lpmv():
    pts = []
    for x in [-0.99, -0.557, 1e-6, 0.132, 1]:
        pts.extend([
            (1, 1, x),
            (1, -1, x),
            (-1, 1, x),
            (-1, -2, x),
            (1, 1.7, x),
            (1, -1.7, x),
            (-1, 1.7, x),
            (-1, -2.7, x),
            (1, 10, x),
            (1, 11, x),
            (3, 8, x),
            (5, 11, x),
            (-3, 8, x),
            (-5, 11, x),
            (3, -8, x),
            (5, -11, x),
            (-3, -8, x),
            (-5, -11, x),
            (3, 8.3, x),
            (5, 11.3, x),
            (-3, 8.3, x),
            (-5, 11.3, x),
            (3, -8.3, x),
            (5, -11.3, x),
            (-3, -8.3, x),
            (-5, -11.3, x),
        ])

    def mplegenp(nu, mu, x):
        if mu == int(mu) and x == 1:
            # mpmath 0.17 gets this wrong
            if mu == 0:
                return 1
            else:
                return 0
        return mpmath.legenp(nu, mu, x)

    dataset = [p + (mplegenp(p[1], p[0], p[2]),) for p in pts]
    dataset = np.array(dataset, dtype=np.float_)

    def evf(mu, nu, x):
        return sc.lpmv(mu.astype(int), nu, x)

    with np.errstate(invalid='ignore'):
        FuncData(evf, dataset, (0,1,2), 3, rtol=1e-10, atol=1e-14).check()


# ------------------------------------------------------------------------------
# beta
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.15')
def test_beta():
    np.random.seed(1234)

    b = np.r_[np.logspace(-200, 200, 4),
              np.logspace(-10, 10, 4),
              np.logspace(-1, 1, 4),
              np.arange(-10, 11, 1),
              np.arange(-10, 11, 1) + 0.5,
              -1, -2.3, -3, -100.3, -10003.4]
    a = b

    ab = np.array(np.broadcast_arrays(a[:,None], b[None,:])).reshape(2, -1).T

    old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
    try:
        mpmath.mp.dps = 400

        assert_func_equal(sc.beta,
                          lambda a, b: float(mpmath.beta(a, b)),
                          ab,
                          vectorized=False,
                          rtol=1e-10,
                          ignore_inf_sign=True)

        assert_func_equal(
            sc.betaln,
            lambda a, b: float(mpmath.log(abs(mpmath.beta(a, b)))),
            ab,
            vectorized=False,
            rtol=1e-10)
    finally:
        mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec


# ------------------------------------------------------------------------------
# loggamma
# ------------------------------------------------------------------------------

LOGGAMMA_TAYLOR_RADIUS = 0.2


@check_version(mpmath, '0.19')
def test_loggamma_taylor_transition():
    # Make sure there isn't a big jump in accuracy when we move from
    # using the Taylor series to using the recurrence relation.

    r = LOGGAMMA_TAYLOR_RADIUS + np.array([-0.1, -0.01, 0, 0.01, 0.1])
    theta = np.linspace(0, 2*np.pi, 20)
    r, theta = np.meshgrid(r, theta)
    dz = r*np.exp(1j*theta)
    z = np.r_[1 + dz, 2 + dz].flatten()

    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)

    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()


@check_version(mpmath, '0.19')
def test_loggamma_taylor():
    # Test around the zeros at z = 1, 2.

    r = np.logspace(-16, np.log10(LOGGAMMA_TAYLOR_RADIUS), 10)
    theta = np.linspace(0, 2*np.pi, 20)
    r, theta = np.meshgrid(r, theta)
    dz = r*np.exp(1j*theta)
    z = np.r_[1 + dz, 2 + dz].flatten()

    dataset = [(z0, complex(mpmath.loggamma(z0))) for z0 in z]
    dataset = np.array(dataset)

    FuncData(sc.loggamma, dataset, 0, 1, rtol=5e-14).check()


# ------------------------------------------------------------------------------
# rgamma
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_rgamma_zeros():
    # Test around the zeros at z = 0, -1, -2, ...,  -169. (After -169 we
    # get values that are out of floating point range even when we're
    # within 0.1 of the zero.)

    # Can't use too many points here or the test takes forever.
    dx = np.r_[-np.logspace(-1, -13, 3), 0, np.logspace(-13, -1, 3)]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    zeros = np.arange(0, -170, -1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,)*zeros.size)).flatten()
    with mpmath.workdps(100):
        dataset = [(z0, complex(mpmath.rgamma(z0))) for z0 in z]

    dataset = np.array(dataset)
    FuncData(sc.rgamma, dataset, 0, 1, rtol=1e-12).check()


# ------------------------------------------------------------------------------
# digamma
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_digamma_roots():
    # Test the special-cased roots for digamma.
    root = mpmath.findroot(mpmath.digamma, 1.5)
    roots = [float(root)]
    root = mpmath.findroot(mpmath.digamma, -0.5)
    roots.append(float(root))
    roots = np.array(roots)

    # If we test beyond a radius of 0.24 mpmath will take forever.
    dx = np.r_[-0.24, -np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10), 0.24]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    z = (roots + np.dstack((dz,)*roots.size)).flatten()
    with mpmath.workdps(30):
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]

    dataset = np.array(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()


@check_version(mpmath, '0.19')
def test_digamma_negreal():
    # Test digamma around the negative real axis. Don't do this in
    # TestSystematic because the points need some jiggering so that
    # mpmath doesn't take forever.

    digamma = exception_to_nan(mpmath.digamma)

    x = -np.logspace(300, -30, 100)
    y = np.r_[-np.logspace(0, -3, 5), 0, np.logspace(-3, 0, 5)]
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    with mpmath.workdps(40):
        dataset = [(z0, complex(digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)

    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()


@check_version(mpmath, '0.19')
def test_digamma_boundary():
    # Check that there isn't a jump in accuracy when we switch from
    # using the asymptotic series to the reflection formula.

    x = -np.logspace(300, -30, 100)
    y = np.array([-6.1, -5.9, 5.9, 6.1])
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    with mpmath.workdps(30):
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]
    dataset = np.asarray(dataset)

    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-13).check()


# ------------------------------------------------------------------------------
# gammainc
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_gammainc_boundary():
    # Test the transition to the asymptotic series.
    small = 20
    a = np.linspace(0.5*small, 2*small, 50)
    x = a.copy()
    a, x = np.meshgrid(a, x)
    a, x = a.flatten(), x.flatten()
    with mpmath.workdps(100):
        dataset = [(a0, x0, float(mpmath.gammainc(a0, b=x0, regularized=True)))
                   for a0, x0 in zip(a, x)]
    dataset = np.array(dataset)

    FuncData(sc.gammainc, dataset, (0, 1), 2, rtol=1e-12).check()


# ------------------------------------------------------------------------------
# spence
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
@pytest.mark.slow
def test_spence_circle():
    # The trickiest region for spence is around the circle |z - 1| = 1,
    # so test that region carefully.

    def spence(z):
        return complex(mpmath.polylog(2, 1 - z))

    r = np.linspace(0.5, 1.5)
    theta = np.linspace(0, 2*pi)
    z = (1 + np.outer(r, np.exp(1j*theta))).flatten()
    dataset = np.asarray([(z0, spence(z0)) for z0 in z])

    FuncData(sc.spence, dataset, 0, 1, rtol=1e-14).check()


# ------------------------------------------------------------------------------
# sinpi and cospi
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
def test_sinpi_zeros():
    eps = np.finfo(float).eps
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    zeros = np.arange(-100, 100, 1).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,)*zeros.size)).flatten()
    dataset = np.asarray([(z0, complex(mpmath.sinpi(z0)))
                          for z0 in z])
    FuncData(_sinpi, dataset, 0, 1, rtol=2*eps).check()


@check_version(mpmath, '0.19')
def test_cospi_zeros():
    eps = np.finfo(float).eps
    dx = np.r_[-np.logspace(0, -13, 3), 0, np.logspace(-13, 0, 3)]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j*dy
    zeros = (np.arange(-100, 100, 1) + 0.5).reshape(1, 1, -1)
    z = (zeros + np.dstack((dz,)*zeros.size)).flatten()
    dataset = np.asarray([(z0, complex(mpmath.cospi(z0)))
                          for z0 in z])

    FuncData(_cospi, dataset, 0, 1, rtol=2*eps).check()


# ------------------------------------------------------------------------------
# ellipj
# ------------------------------------------------------------------------------

@check_version(mpmath, '0.19')
def test_dn_quarter_period():
    def dn(u, m):
        return sc.ellipj(u, m)[2]

    def mpmath_dn(u, m):
        return float(mpmath.ellipfun("dn", u=u, m=m))

    m = np.linspace(0, 1, 20)
    du = np.r_[-np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10)]
    dataset = []
    for m0 in m:
        u0 = float(mpmath.ellipk(m0))
        for du0 in du:
            p = u0 + du0
            dataset.append((p, m0, mpmath_dn(p, m0)))
    dataset = np.asarray(dataset)

    FuncData(dn, dataset, (0, 1), 2, rtol=1e-10).check()


# ------------------------------------------------------------------------------
# Wright Omega
# ------------------------------------------------------------------------------

def _mpmath_wrightomega(z, dps):
    with mpmath.workdps(dps):
        z = mpmath.mpc(z)
        unwind = mpmath.ceil((z.imag - mpmath.pi)/(2*mpmath.pi))
        res = mpmath.lambertw(mpmath.exp(z), unwind)
    return res


@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_branch():
    x = -np.logspace(10, 0, 25)
    picut_above = [np.nextafter(np.pi, np.inf)]
    picut_below = [np.nextafter(np.pi, -np.inf)]
    npicut_above = [np.nextafter(-np.pi, np.inf)]
    npicut_below = [np.nextafter(-np.pi, -np.inf)]
    for i in range(50):
        picut_above.append(np.nextafter(picut_above[-1], np.inf))
        picut_below.append(np.nextafter(picut_below[-1], -np.inf))
        npicut_above.append(np.nextafter(npicut_above[-1], np.inf))
        npicut_below.append(np.nextafter(npicut_below[-1], -np.inf))
    y = np.hstack((picut_above, picut_below, npicut_above, npicut_below))
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25)))
                          for z0 in z])

    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-8).check()


@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region1():
    # This region gets less coverage in the TestSystematic test
    x = np.linspace(-2, 1)
    y = np.linspace(1, 2*np.pi)
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25)))
                          for z0 in z])

    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()


@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_wrightomega_region2():
    # This region gets less coverage in the TestSystematic test
    x = np.linspace(-2, 1)
    y = np.linspace(-2*np.pi, -1)
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    dataset = np.asarray([(z0, complex(_mpmath_wrightomega(z0, 25)))
                          for z0 in z])

    FuncData(sc.wrightomega, dataset, 0, 1, rtol=1e-15).check()


# ------------------------------------------------------------------------------
# lambertw
# ------------------------------------------------------------------------------

@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_lambertw_smallz():
    x, y = np.linspace(-1, 1, 25), np.linspace(-1, 1, 25)
    x, y = np.meshgrid(x, y)
    z = (x + 1j*y).flatten()

    dataset = np.asarray([(z0, complex(mpmath.lambertw(z0)))
                          for z0 in z])

    FuncData(sc.lambertw, dataset, 0, 1, rtol=1e-13).check()


# ------------------------------------------------------------------------------
# Systematic tests
# ------------------------------------------------------------------------------

HYPERKW = dict(maxprec=200, maxterms=200)


@pytest.mark.slow
@check_version(mpmath, '0.17')
class TestSystematic:

    def test_airyai(self):
        # oscillating function, limit range
        assert_mpmath_equal(lambda z: sc.airy(z)[0],
                            mpmath.airyai,
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        assert_mpmath_equal(lambda z: sc.airy(z)[0],
                            mpmath.airyai,
                            [Arg(-1e3, 1e3)])

    def test_airyai_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[0],
                            mpmath.airyai,
                            [ComplexArg()])

    def test_airyai_prime(self):
        # oscillating function, limit range
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                            mpmath.airyai(z, derivative=1),
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                            mpmath.airyai(z, derivative=1),
                            [Arg(-1e3, 1e3)])

    def test_airyai_prime_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[1], lambda z:
                            mpmath.airyai(z, derivative=1),
                            [ComplexArg()])

    def test_airybi(self):
        # oscillating function, limit range
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                            mpmath.airybi(z),
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                            mpmath.airybi(z),
                            [Arg(-1e3, 1e3)])

    def test_airybi_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[2], lambda z:
                            mpmath.airybi(z),
                            [ComplexArg()])

    def test_airybi_prime(self):
        # oscillating function, limit range
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                            mpmath.airybi(z, derivative=1),
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                            mpmath.airybi(z, derivative=1),
                            [Arg(-1e3, 1e3)])

    def test_airybi_prime_complex(self):
        assert_mpmath_equal(lambda z: sc.airy(z)[3], lambda z:
                            mpmath.airybi(z, derivative=1),
                            [ComplexArg()])

    def test_bei(self):
        assert_mpmath_equal(sc.bei,
                            exception_to_nan(lambda z: mpmath.bei(0, z, **HYPERKW)),
                            [Arg(-1e3, 1e3)])

    def test_ber(self):
        assert_mpmath_equal(sc.ber,
                            exception_to_nan(lambda z: mpmath.ber(0, z, **HYPERKW)),
                            [Arg(-1e3, 1e3)])

    def test_bernoulli(self):
        assert_mpmath_equal(lambda n: sc.bernoulli(int(n))[int(n)],
                            lambda n: float(mpmath.bernoulli(int(n))),
                            [IntArg(0, 13000)],
                            rtol=1e-9, n=13000)

    def test_besseli(self):
        assert_mpmath_equal(sc.iv,
                            exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)),
                            [Arg(-1e100, 1e100), Arg()],
                            atol=1e-270)

    def test_besseli_complex(self):
        assert_mpmath_equal(lambda v, z: sc.iv(v.real, z),
                            exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)),
                            [Arg(-1e100, 1e100), ComplexArg()])

    def test_besselj(self):
        assert_mpmath_equal(sc.jv,
                            exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)),
                            [Arg(-1e100, 1e100), Arg(-1e3, 1e3)],
                            ignore_inf_sign=True)

        # loss of precision at large arguments due to oscillation
        assert_mpmath_equal(sc.jv,
                            exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)),
                            [Arg(-1e100, 1e100), Arg(-1e8, 1e8)],
                            ignore_inf_sign=True,
                            rtol=1e-5)

    def test_besselj_complex(self):
        assert_mpmath_equal(lambda v, z: sc.jv(v.real, z),
                            exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)),
                            [Arg(), ComplexArg()])

    def test_besselk(self):
        assert_mpmath_equal(sc.kv,
                            mpmath.besselk,
                            [Arg(-200, 200), Arg(0, np.inf)],
                            nan_ok=False, rtol=1e-12)

    def test_besselk_int(self):
        assert_mpmath_equal(sc.kn,
                            mpmath.besselk,
                            [IntArg(-200, 200), Arg(0, np.inf)],
                            nan_ok=False, rtol=1e-12)

    def test_besselk_complex(self):
        assert_mpmath_equal(lambda v, z: sc.kv(v.real, z),
                            exception_to_nan(lambda v, z: mpmath.besselk(v, z, **HYPERKW)),
                            [Arg(-1e100, 1e100), ComplexArg()])

    def test_bessely(self):
        def mpbessely(v, x):
            r = float(mpmath.bessely(v, x, **HYPERKW))
            if abs(r) > 1e305:
                # overflowing to inf a bit earlier is OK
                r = np.inf * np.sign(r)
            if abs(r) == 0 and x == 0:
                # invalid result from mpmath, point x=0 is a divergence
                return np.nan
            return r
        assert_mpmath_equal(sc.yv,
                            exception_to_nan(mpbessely),
                            [Arg(-1e100, 1e100), Arg(-1e8, 1e8)],
                            n=5000)

    def test_bessely_complex(self):
        def mpbessely(v, x):
            r = complex(mpmath.bessely(v, x, **HYPERKW))
            if abs(r) > 1e305:
                # overflowing to inf a bit earlier is OK
                with np.errstate(invalid='ignore'):
                    r = np.inf * np.sign(r)
            return r
        assert_mpmath_equal(lambda v, z: sc.yv(v.real, z),
                            exception_to_nan(mpbessely),
                            [Arg(), ComplexArg()],
                            n=15000)

    def test_bessely_int(self):
        def mpbessely(v, x):
            r = float(mpmath.bessely(v, x))
            if abs(r) == 0 and x == 0:
                # invalid result from mpmath, point x=0 is a divergence
                return np.nan
            return r
        assert_mpmath_equal(lambda v, z: sc.yn(int(v), z),
                            exception_to_nan(mpbessely),
                            [IntArg(-1000, 1000), Arg(-1e8, 1e8)])

    def test_beta(self):
        bad_points = []

        def beta(a, b, nonzero=False):
            if a < -1e12 or b < -1e12:
                # Function is defined here only at integers, but due
                # to loss of precision this is numerically
                # ill-defined. Don't compare values here.
                return np.nan
            if (a < 0 or b < 0) and (abs(float(a + b)) % 1) == 0:
                # close to a zero of the function: mpmath and scipy
                # will not round here the same, so the test needs to be
                # run with an absolute tolerance
                if nonzero:
                    bad_points.append((float(a), float(b)))
                    return np.nan
            return mpmath.beta(a, b)

        assert_mpmath_equal(sc.beta,
                            lambda a, b: beta(a, b, nonzero=True),
                            [Arg(), Arg()],
                            dps=400,
                            ignore_inf_sign=True)

        assert_mpmath_equal(sc.beta,
                            beta,
                            np.array(bad_points),
                            dps=400,
                            ignore_inf_sign=True,
                            atol=1e-11)

    def test_betainc(self):
        assert_mpmath_equal(sc.betainc,
                            time_limited()(exception_to_nan(lambda a, b, x: mpmath.betainc(a, b, 0, x, regularized=True))),
                            [Arg(), Arg(), Arg()])

    def test_binom(self):
        bad_points = []

        def binomial(n, k, nonzero=False):
            if abs(k) > 1e8*(abs(n) + 1):
                # The binomial is rapidly oscillating in this region,
                # and the function is numerically ill-defined. Don't
                # compare values here.
                return np.nan
            if n < k and abs(float(n-k) - np.round(float(n-k))) < 1e-15:
                # close to a zero of the function: mpmath and scipy
                # will not round here the same, so the test needs to be
                # run with an absolute tolerance
                if nonzero:
                    bad_points.append((float(n), float(k)))
                    return np.nan
            return mpmath.binomial(n, k)

        assert_mpmath_equal(sc.binom,
                            lambda n, k: binomial(n, k, nonzero=True),
                            [Arg(), Arg()],
                            dps=400)

        assert_mpmath_equal(sc.binom,
                            binomial,
                            np.array(bad_points),
                            dps=400,
                            atol=1e-14)

    def test_chebyt_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_chebyt(int(n), x),
                            exception_to_nan(lambda n, x: mpmath.chebyt(n, x, **HYPERKW)),
                            [IntArg(), Arg()], dps=50)

    @pytest.mark.xfail(run=False, reason="some cases in hyp2f1 not fully accurate")
    def test_chebyt(self):
        assert_mpmath_equal(sc.eval_chebyt,
                            lambda n, x: time_limited()(exception_to_nan(mpmath.chebyt))(n, x, **HYPERKW),
                            [Arg(-101, 101), Arg()], n=10000)

    def test_chebyu_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_chebyu(int(n), x),
                            exception_to_nan(lambda n, x: mpmath.chebyu(n, x, **HYPERKW)),
                            [IntArg(), Arg()], dps=50)

    @pytest.mark.xfail(run=False, reason="some cases in hyp2f1 not fully accurate")
    def test_chebyu(self):
        assert_mpmath_equal(sc.eval_chebyu,
                            lambda n, x: time_limited()(exception_to_nan(mpmath.chebyu))(n, x, **HYPERKW),
                            [Arg(-101, 101), Arg()])

    def test_chi(self):
        def chi(x):
            return sc.shichi(x)[1]
        assert_mpmath_equal(chi, mpmath.chi, [Arg()])
        # check asymptotic series cross-over
        assert_mpmath_equal(chi, mpmath.chi, [FixedArg([88 - 1e-9, 88, 88 + 1e-9])])

    def test_chi_complex(self):
        def chi(z):
            return sc.shichi(z)[1]
        # chi oscillates as Im[z] -> +- inf, so limit range
        assert_mpmath_equal(chi,
                            mpmath.chi,
                            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
                            rtol=1e-12)

    def test_ci(self):
        def ci(x):
            return sc.sici(x)[1]
        # oscillating function: limit range
        assert_mpmath_equal(ci,
                            mpmath.ci,
                            [Arg(-1e8, 1e8)])

    def test_ci_complex(self):
        def ci(z):
            return sc.sici(z)[1]
        # ci oscillates as Re[z] -> +- inf, so limit range
        assert_mpmath_equal(ci,
                            mpmath.ci,
                            [ComplexArg(complex(-1e8, -np.inf), complex(1e8, np.inf))],
                            rtol=1e-8)

    def test_cospi(self):
        eps = np.finfo(float).eps
        assert_mpmath_equal(_cospi,
                            mpmath.cospi,
                            [Arg()], nan_ok=False, rtol=eps)

    def test_cospi_complex(self):
        assert_mpmath_equal(_cospi,
                            mpmath.cospi,
                            [ComplexArg()], nan_ok=False, rtol=1e-13)

    def test_digamma(self):
        assert_mpmath_equal(sc.digamma,
                            exception_to_nan(mpmath.digamma),
                            [Arg()], rtol=1e-12, dps=50)

    def test_digamma_complex(self):
        # Test on a cut plane because mpmath will hang. See
        # test_digamma_negreal for tests on the negative real axis.
        def param_filter(z):
            return np.where((z.real < 0) & (np.abs(z.imag) < 1.12), False, True)

        assert_mpmath_equal(sc.digamma,
                            exception_to_nan(mpmath.digamma),
                            [ComplexArg()], rtol=1e-13, dps=40,
                            param_filter=param_filter)

    def test_e1(self):
        assert_mpmath_equal(sc.exp1,
                            mpmath.e1,
                            [Arg()], rtol=1e-14)

    def test_e1_complex(self):
        # E_1 oscillates as Im[z] -> +- inf, so limit range
        assert_mpmath_equal(sc.exp1,
                            mpmath.e1,
                            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
                            rtol=1e-11)

        # Check cross-over region
        assert_mpmath_equal(sc.exp1,
                            mpmath.e1,
                            (np.linspace(-50, 50, 171)[:, None] +
                             np.r_[0, np.logspace(-3, 2, 61),
                                   -np.logspace(-3, 2, 11)]*1j).ravel(),
                            rtol=1e-11)
        assert_mpmath_equal(sc.exp1,
                            mpmath.e1,
                            (np.linspace(-50, -35, 10000) + 0j),
                            rtol=1e-11)

    def test_exprel(self):
        assert_mpmath_equal(sc.exprel,
                            lambda x: mpmath.expm1(x)/x if x != 0 else mpmath.mpf('1.0'),
                            [Arg(a=-np.log(np.finfo(np.double).max), b=np.log(np.finfo(np.double).max))])
        assert_mpmath_equal(sc.exprel,
                            lambda x: mpmath.expm1(x)/x if x != 0 else mpmath.mpf('1.0'),
                            np.array([1e-12, 1e-24, 0, 1e12, 1e24, np.inf]), rtol=1e-11)
        assert_(np.isinf(sc.exprel(np.inf)))
        assert_(sc.exprel(-np.inf) == 0)

    def test_expm1_complex(self):
        # Oscillates as a function of Im[z], so limit range to avoid loss of precision
        assert_mpmath_equal(sc.expm1,
                            mpmath.expm1,
                            [ComplexArg(complex(-np.inf, -1e7), complex(np.inf, 1e7))])

    def test_log1p_complex(self):
        assert_mpmath_equal(sc.log1p,
                            lambda x: mpmath.log(x+1),
                            [ComplexArg()], dps=60)

    def test_log1pmx(self):
        assert_mpmath_equal(_log1pmx,
                            lambda x: mpmath.log(x + 1) - x,
                            [Arg()], dps=60, rtol=1e-14)

    def test_ei(self):
        assert_mpmath_equal(sc.expi,
                            mpmath.ei,
                            [Arg()],
                            rtol=1e-11)

    def test_ei_complex(self):
        # Ei oscillates as Im[z] -> +- inf, so limit range
        assert_mpmath_equal(sc.expi,
                            mpmath.ei,
                            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
                            rtol=1e-9)

    def test_ellipe(self):
        assert_mpmath_equal(sc.ellipe,
                            mpmath.ellipe,
                            [Arg(b=1.0)])

    def test_ellipeinc(self):
        assert_mpmath_equal(sc.ellipeinc,
                            mpmath.ellipe,
                            [Arg(-1e3, 1e3), Arg(b=1.0)])

    def test_ellipeinc_largephi(self):
        assert_mpmath_equal(sc.ellipeinc,
                            mpmath.ellipe,
                            [Arg(), Arg()])

    def test_ellipf(self):
        assert_mpmath_equal(sc.ellipkinc,
                            mpmath.ellipf,
                            [Arg(-1e3, 1e3), Arg()])

    def test_ellipf_largephi(self):
        assert_mpmath_equal(sc.ellipkinc,
                            mpmath.ellipf,
                            [Arg(), Arg()])

    def test_ellipk(self):
        assert_mpmath_equal(sc.ellipk,
                            mpmath.ellipk,
                            [Arg(b=1.0)])
        assert_mpmath_equal(sc.ellipkm1,
                            lambda m: mpmath.ellipk(1 - m),
                            [Arg(a=0.0)],
                            dps=400)

    def test_ellipkinc(self):
        def ellipkinc(phi, m):
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(sc.ellipkinc,
                            ellipkinc,
                            [Arg(-1e3, 1e3), Arg(b=1.0)],
                            ignore_inf_sign=True)

    def test_ellipkinc_largephi(self):
        def ellipkinc(phi, m):
            return mpmath.ellippi(0, phi, m)
        assert_mpmath_equal(sc.ellipkinc,
                            ellipkinc,
                            [Arg(), Arg(b=1.0)],
                            ignore_inf_sign=True)

    def test_ellipfun_sn(self):
        def sn(u, m):
            # mpmath doesn't get the zero at u = 0--fix that
            if u == 0:
                return 0
            else:
                return mpmath.ellipfun("sn", u=u, m=m)

        # Oscillating function --- limit range of first argument; the
        # loss of precision there is an expected numerical feature
        # rather than an actual bug
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[0],
                            sn,
                            [Arg(-1e6, 1e6), Arg(a=0, b=1)],
                            rtol=1e-8)

    def test_ellipfun_cn(self):
        # see comment in ellipfun_sn
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[1],
                            lambda u, m: mpmath.ellipfun("cn", u=u, m=m),
                            [Arg(-1e6, 1e6), Arg(a=0, b=1)],
                            rtol=1e-8)

    def test_ellipfun_dn(self):
        # see comment in ellipfun_sn
        assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[2],
                            lambda u, m: mpmath.ellipfun("dn", u=u, m=m),
                            [Arg(-1e6, 1e6), Arg(a=0, b=1)],
                            rtol=1e-8)

    def test_erf(self):
        assert_mpmath_equal(sc.erf,
                            lambda z: mpmath.erf(z),
                            [Arg()])

    def test_erf_complex(self):
        assert_mpmath_equal(sc.erf,
                            lambda z: mpmath.erf(z),
                            [ComplexArg()], n=200)

    def test_erfc(self):
        assert_mpmath_equal(sc.erfc,
                            exception_to_nan(lambda z: mpmath.erfc(z)),
                            [Arg()], rtol=1e-13)

    def test_erfc_complex(self):
        assert_mpmath_equal(sc.erfc,
                            exception_to_nan(lambda z: mpmath.erfc(z)),
                            [ComplexArg()], n=200)

    def test_erfi(self):
        assert_mpmath_equal(sc.erfi,
                            mpmath.erfi,
                            [Arg()], n=200)

    def test_erfi_complex(self):
        assert_mpmath_equal(sc.erfi,
                            mpmath.erfi,
                            [ComplexArg()], n=200)

    def test_ndtr(self):
        assert_mpmath_equal(sc.ndtr,
                            exception_to_nan(lambda z: mpmath.ncdf(z)),
                            [Arg()], n=200)

    def test_ndtr_complex(self):
        assert_mpmath_equal(sc.ndtr,
                            lambda z: mpmath.erfc(-z/np.sqrt(2.))/2.,
                            [ComplexArg(a=complex(-10000, -10000), b=complex(10000, 10000))], n=400)

    def test_log_ndtr(self):
        assert_mpmath_equal(sc.log_ndtr,
                            exception_to_nan(lambda z: mpmath.log(mpmath.ncdf(z))),
                            [Arg()], n=600, dps=300)

    def test_log_ndtr_complex(self):
        assert_mpmath_equal(sc.log_ndtr,
                            exception_to_nan(lambda z: mpmath.log(mpmath.erfc(-z/np.sqrt(2.))/2.)),
                            [ComplexArg(a=complex(-10000, -100),
                                        b=complex(10000, 100))], n=200, dps=300)

    def test_eulernum(self):
        assert_mpmath_equal(lambda n: sc.euler(n)[-1],
                            mpmath.eulernum,
                            [IntArg(1, 10000)], n=10000)

    def test_expint(self):
        assert_mpmath_equal(sc.expn,
                            mpmath.expint,
                            [IntArg(0, 200), Arg(0, np.inf)],
                            rtol=1e-13, dps=160)

    def test_fresnels(self):
        def fresnels(x):
            return sc.fresnel(x)[0]
        assert_mpmath_equal(fresnels,
                            mpmath.fresnels,
                            [Arg()])

    def test_fresnelc(self):
        def fresnelc(x):
            return sc.fresnel(x)[1]
        assert_mpmath_equal(fresnelc,
                            mpmath.fresnelc,
                            [Arg()])

    def test_gamma(self):
        assert_mpmath_equal(sc.gamma,
                            exception_to_nan(mpmath.gamma),
                            [Arg()])

    def test_gamma_complex(self):
        assert_mpmath_equal(sc.gamma,
                            exception_to_nan(mpmath.gamma),
                            [ComplexArg()], rtol=5e-13)

    def test_gammainc(self):
        # Larger arguments are tested in test_data.py:test_local
        assert_mpmath_equal(sc.gammainc,
                            lambda z, b: mpmath.gammainc(z, b=b, regularized=True),
                            [Arg(0, 1e4, inclusive_a=False), Arg(0, 1e4)],
                            nan_ok=False, rtol=1e-11)

    def test_gammaincc(self):
        # Larger arguments are tested in test_data.py:test_local
        assert_mpmath_equal(sc.gammaincc,
                            lambda z, a: mpmath.gammainc(z, a=a, regularized=True),
                            [Arg(0, 1e4, inclusive_a=False), Arg(0, 1e4)],
                            nan_ok=False, rtol=1e-11)

    def test_gammaln(self):
        # The real part of loggamma is log(|gamma(z)|).
        def f(z):
            return mpmath.loggamma(z).real

        assert_mpmath_equal(sc.gammaln, exception_to_nan(f), [Arg()])

    @pytest.mark.xfail(run=False)
    def test_gegenbauer(self):
        assert_mpmath_equal(sc.eval_gegenbauer,
                            exception_to_nan(mpmath.gegenbauer),
                            [Arg(-1e3, 1e3), Arg(), Arg()])

    def test_gegenbauer_int(self):
        # Redefine functions to deal with numerical + mpmath issues
        def gegenbauer(n, a, x):
            # Avoid overflow at large `a` (mpmath would need an even larger
            # dps to handle this correctly, so just skip this region)
            if abs(a) > 1e100:
                return np.nan

            # Deal with n=0, n=1 correctly; mpmath 0.17 doesn't do these
            # always correctly
            if n == 0:
                r = 1.0
            elif n == 1:
                r = 2*a*x
            else:
                r = mpmath.gegenbauer(n, a, x)

            # Mpmath 0.17 gives wrong results (spurious zero) in some cases, so
            # compute the value by perturbing the result
            if float(r) == 0 and a < -1 and float(a) == int(float(a)):
                r = mpmath.gegenbauer(n, a + mpmath.mpf('1e-50'), x)
                if abs(r) < mpmath.mpf('1e-50'):
                    r = mpmath.mpf('0.0')

            # Differing overflow thresholds in scipy vs. mpmath
            if abs(r) > 1e270:
                return np.inf
            return r

        def sc_gegenbauer(n, a, x):
            r = sc.eval_gegenbauer(int(n), a, x)
            # Differing overflow thresholds in scipy vs. mpmath
            if abs(r) > 1e270:
                return np.inf
            return r
        assert_mpmath_equal(sc_gegenbauer,
                            exception_to_nan(gegenbauer),
                            [IntArg(0, 100), Arg(-1e9, 1e9), Arg()],
                            n=40000, dps=100,
                            ignore_inf_sign=True, rtol=1e-6)

        # Check the small-x expansion
        assert_mpmath_equal(sc_gegenbauer,
                            exception_to_nan(gegenbauer),
                            [IntArg(0, 100), Arg(), FixedArg(np.logspace(-30, -4, 30))],
                            dps=100,
                            ignore_inf_sign=True)

    @pytest.mark.xfail(run=False)
    def test_gegenbauer_complex(self):
        assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(int(n), a.real, x),
                            exception_to_nan(mpmath.gegenbauer),
                            [IntArg(0, 100), Arg(), ComplexArg()])

    @nonfunctional_tooslow
    def test_gegenbauer_complex_general(self):
        assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(n.real, a.real, x),
                            exception_to_nan(mpmath.gegenbauer),
                            [Arg(-1e3, 1e3), Arg(), ComplexArg()])

    def test_hankel1(self):
        assert_mpmath_equal(sc.hankel1,
                            exception_to_nan(lambda v, x: mpmath.hankel1(v, x,
                                                                          **HYPERKW)),
                            [Arg(-1e20, 1e20), Arg()])

    def test_hankel2(self):
        assert_mpmath_equal(sc.hankel2,
                            exception_to_nan(lambda v, x: mpmath.hankel2(v, x, **HYPERKW)),
                            [Arg(-1e20, 1e20), Arg()])

    @pytest.mark.xfail(run=False, reason="issues at intermediately large orders")
    def test_hermite(self):
        assert_mpmath_equal(lambda n, x: sc.eval_hermite(int(n), x),
                            exception_to_nan(mpmath.hermite),
                            [IntArg(0, 10000), Arg()])

    # hurwitz: same as zeta

    def test_hyp0f1(self):
        # mpmath reports no convergence unless maxterms is large enough
        KW = dict(maxprec=400, maxterms=1500)
        # n=500 (non-xslow default) fails for one bad point
        assert_mpmath_equal(sc.hyp0f1,
                            lambda a, x: mpmath.hyp0f1(a, x, **KW),
                            [Arg(-1e7, 1e7), Arg(0, 1e5)],
                            n=5000)
        # NB: The range of the second parameter ("z") is limited from below
        # because of an overflow in the intermediate calculations. The way
        # for fix it is to implement an asymptotic expansion for Bessel J
        # (similar to what is implemented for Bessel I here).

    def test_hyp0f1_complex(self):
        assert_mpmath_equal(lambda a, z: sc.hyp0f1(a.real, z),
                            exception_to_nan(lambda a, x: mpmath.hyp0f1(a, x, **HYPERKW)),
                            [Arg(-10, 10), ComplexArg(complex(-120, -120), complex(120, 120))])
        # NB: The range of the first parameter ("v") are limited by an overflow
        # in the intermediate calculations. Can be fixed by implementing an
        # asymptotic expansion for Bessel functions for large order.

    def test_hyp1f1(self):
        def mpmath_hyp1f1(a, b, x):
            try:
                return mpmath.hyp1f1(a, b, x)
            except ZeroDivisionError:
                return np.inf

        assert_mpmath_equal(
            sc.hyp1f1,
            mpmath_hyp1f1,
            [Arg(-50, 50), Arg(1, 50, inclusive_a=False), Arg(-50, 50)],
            n=500,
            nan_ok=False
        )

    @pytest.mark.xfail(run=False)
    def test_hyp1f1_complex(self):
        assert_mpmath_equal(inf_to_nan(lambda a, b, x: sc.hyp1f1(a.real, b.real, x)),
                            exception_to_nan(lambda a, b, x: mpmath.hyp1f1(a, b, x, **HYPERKW)),
                            [Arg(-1e3, 1e3), Arg(-1e3, 1e3), ComplexArg()],
                            n=2000)

    @nonfunctional_tooslow
    def test_hyp2f1_complex(self):
        # SciPy's hyp2f1 seems to have performance and accuracy problems
        assert_mpmath_equal(lambda a, b, c, x: sc.hyp2f1(a.real, b.real, c.real, x),
                            exception_to_nan(lambda a, b, c, x: mpmath.hyp2f1(a, b, c, x, **HYPERKW)),
                            [Arg(-1e2, 1e2), Arg(-1e2, 1e2), Arg(-1e2, 1e2), ComplexArg()],
                            n=10)

    @pytest.mark.xfail(run=False)
    def test_hyperu(self):
        assert_mpmath_equal(sc.hyperu,
                            exception_to_nan(lambda a, b, x: mpmath.hyperu(a, b, x, **HYPERKW)),
                            [Arg(), Arg(), Arg()])

    @pytest.mark.xfail_on_32bit("mpmath issue gh-342: unsupported operand mpz, long for pow")
    def test_igam_fac(self):
        def mp_igam_fac(a, x):
            return mpmath.power(x, a)*mpmath.exp(-x)/mpmath.gamma(a)

        assert_mpmath_equal(_igam_fac,
                            mp_igam_fac,
                            [Arg(0, 1e14, inclusive_a=False), Arg(0, 1e14)],
                            rtol=1e-10)

    def test_j0(self):
        # The Bessel function at large arguments is j0(x) ~ cos(x + phi)/sqrt(x)
        # and at large arguments the phase of the cosine loses precision.
        #
        # This is numerically expected behavior, so we compare only up to
        # 1e8 = 1e15 * 1e-7
        assert_mpmath_equal(sc.j0,
                            mpmath.j0,
                            [Arg(-1e3, 1e3)])
        assert_mpmath_equal(sc.j0,
                            mpmath.j0,
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)

    def test_j1(self):
        # See comment in test_j0
        assert_mpmath_equal(sc.j1,
                            mpmath.j1,
                            [Arg(-1e3, 1e3)])
        assert_mpmath_equal(sc.j1,
                            mpmath.j1,
                            [Arg(-1e8, 1e8)],
                            rtol=1e-5)

    @pytest.mark.xfail(run=False)
    def test_jacobi(self):
        assert_mpmath_equal(sc.eval_jacobi,
                            exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)),
                            [Arg(), Arg(), Arg(), Arg()])
        assert_mpmath_equal(lambda n, b, c, x: sc.eval_jacobi(int(n), b, c, x),
                            exception_to_nan(lambda a, b, c, x: mpmath.jacobi(a, b, c, x, **HYPERKW)),
                            [IntArg(), Arg(), Arg(), Arg()])

    @pytest.mark.xslow
    def test_jacobi_int(self):
        # Redefine functions to deal with numerical + mpmath issues
        def jacobi(n, a, b, x):
            # Mpmath does not handle n=0 case always correctly
            if n == 0:
                return 1.0
            return mpmath.jacobi(n, a, b, x)
        assert_mpmath_equal(lambda n, a, b, x: sc.eval_jacobi(int(n), a, b, x),
                            lambda n, a, b, x: exception_to_nan(jacobi)(n, a, b, x, **HYPERKW),
                            [IntArg(), Arg(), Arg(), Arg()],
                            n=20000, dps=50)

    def test_kei(self):
        def kei(x):
            if x == 0:
                # work around mpmath issue at x=0
                return -pi/4
            return exception_to_nan(mpmath.kei)(0, x, **HYPERKW)
        assert_mpmath_equal(sc.kei,
                            kei,
                            [Arg(-1e30, 1e30)], n=1000)

    def test_ker(self):
        assert_mpmath_equal(sc.ker,
                            exception_to_nan(lambda x: mpmath.ker(0, x, **HYPERKW)),
                            [Arg(-1e30, 1e30)], n=1000)

    @nonfunctional_tooslow
    def test_laguerre(self):
        assert_mpmath_equal(trace_args(sc.eval_laguerre),
                            lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW),
                            [Arg(), Arg()])

    def test_laguerre_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_laguerre(int(n), x),
                            lambda n, x: exception_to_nan(mpmath.laguerre)(n, x, **HYPERKW),
                            [IntArg(), Arg()], n=20000)

    @pytest.mark.xfail_on_32bit("see gh-3551 for bad points")
    def test_lambertw_real(self):
        assert_mpmath_equal(lambda x, k: sc.lambertw(x, int(k.real)),
                            lambda x, k: mpmath.lambertw(x, int(k.real)),
                            [ComplexArg(-np.inf, np.inf), IntArg(0, 10)],
                            rtol=1e-13, nan_ok=False)

    def test_lanczos_sum_expg_scaled(self):
        maxgamma = 171.624376956302725
        e = np.exp(1)
        g = 6.024680040776729583740234375

        def gamma(x):
            with np.errstate(over='ignore'):
                fac = ((x + g - 0.5)/e)**(x - 0.5)
                if fac != np.inf:
                    res = fac*_lanczos_sum_expg_scaled(x)
                else:
                    fac = ((x + g - 0.5)/e)**(0.5*(x - 0.5))
                    res = fac*_lanczos_sum_expg_scaled(x)
                    res *= fac
            return res

        assert_mpmath_equal(gamma,
                            mpmath.gamma,
                            [Arg(0, maxgamma, inclusive_a=False)],
                            rtol=1e-13)

    @nonfunctional_tooslow
    def test_legendre(self):
        assert_mpmath_equal(sc.eval_legendre,
                            mpmath.legendre,
                            [Arg(), Arg()])

    def test_legendre_int(self):
        assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x),
                            lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW),
                            [IntArg(), Arg()],
                            n=20000)

        # Check the small-x expansion
        assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x),
                            lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW),
                            [IntArg(), FixedArg(np.logspace(-30, -4, 20))])

    def test_legenp(self):
        def lpnm(n, m, z):
            try:
                v = sc.lpmn(m, n, z)[0][-1,-1]
            except ValueError:
                return np.nan
            if abs(v) > 1e306:
                # harmonize overflow to inf
                v = np.inf * np.sign(v.real)
            return v

        def lpnm_2(n, m, z):
            v = sc.lpmv(m, n, z)
            if abs(v) > 1e306:
                # harmonize overflow to inf
                v = np.inf * np.sign(v.real)
            return v

        def legenp(n, m, z):
            if (z == 1 or z == -1) and int(n) == n:
                # Special case (mpmath may give inf, we take the limit by
                # continuity)
                if m == 0:
                    if n < 0:
                        n = -n - 1
                    return mpmath.power(mpmath.sign(z), n)
                else:
                    return 0

            if abs(z) < 1e-15:
                # mpmath has bad performance here
                return np.nan

            typ = 2 if abs(z) < 1 else 3
            v = exception_to_nan(mpmath.legenp)(n, m, z, type=typ)

            if abs(v) > 1e306:
                # harmonize overflow to inf
                v = mpmath.inf * mpmath.sign(v.real)

            return v

        assert_mpmath_equal(lpnm,
                            legenp,
                            [IntArg(-100, 100), IntArg(-100, 100), Arg()])

        assert_mpmath_equal(lpnm_2,
                            legenp,
                            [IntArg(-100, 100), Arg(-100, 100), Arg(-1, 1)],
                            atol=1e-10)

    def test_legenp_complex_2(self):
        def clpnm(n, m, z):
            try:
                return sc.clpmn(m.real, n.real, z, type=2)[0][-1,-1]
            except ValueError:
                return np.nan

        def legenp(n, m, z):
            if abs(z) < 1e-15:
                # mpmath has bad performance here
                return np.nan
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=2)

        # mpmath is quite slow here
        x = np.array([-2, -0.99, -0.5, 0, 1e-5, 0.5, 0.99, 20, 2e3])
        y = np.array([-1e3, -0.5, 0.5, 1.3])
        z = (x[:,None] + 1j*y[None,:]).ravel()

        assert_mpmath_equal(clpnm,
                            legenp,
                            [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)],
                            rtol=1e-6,
                            n=500)

    def test_legenp_complex_3(self):
        def clpnm(n, m, z):
            try:
                return sc.clpmn(m.real, n.real, z, type=3)[0][-1,-1]
            except ValueError:
                return np.nan

        def legenp(n, m, z):
            if abs(z) < 1e-15:
                # mpmath has bad performance here
                return np.nan
            return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=3)

        # mpmath is quite slow here
        x = np.array([-2, -0.99, -0.5, 0, 1e-5, 0.5, 0.99, 20, 2e3])
        y = np.array([-1e3, -0.5, 0.5, 1.3])
        z = (x[:,None] + 1j*y[None,:]).ravel()

        assert_mpmath_equal(clpnm,
                            legenp,
                            [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)],
                            rtol=1e-6,
                            n=500)

    @pytest.mark.xfail(run=False, reason="apparently picks wrong function at |z| > 1")
    def test_legenq(self):
        def lqnm(n, m, z):
            return sc.lqmn(m, n, z)[0][-1,-1]

        def legenq(n, m, z):
            if abs(z) < 1e-15:
                # mpmath has bad performance here
                return np.nan
            return exception_to_nan(mpmath.legenq)(n, m, z, type=2)

        assert_mpmath_equal(lqnm,
                            legenq,
                            [IntArg(0, 100), IntArg(0, 100), Arg()])

    @nonfunctional_tooslow
    def test_legenq_complex(self):
        def lqnm(n, m, z):
            return sc.lqmn(int(m.real), int(n.real), z)[0][-1,-1]

        def legenq(n, m, z):
            if abs(z) < 1e-15:
                # mpmath has bad performance here
                return np.nan
            return exception_to_nan(mpmath.legenq)(int(n.real), int(m.real), z, type=2)

        assert_mpmath_equal(lqnm,
                            legenq,
                            [IntArg(0, 100), IntArg(0, 100), ComplexArg()],
                            n=100)

    def test_lgam1p(self):
        def param_filter(x):
            # Filter the poles
            return np.where((np.floor(x) == x) & (x <= 0), False, True)

        def mp_lgam1p(z):
            # The real part of loggamma is log(|gamma(z)|)
            return mpmath.loggamma(1 + z).real

        assert_mpmath_equal(_lgam1p,
                            mp_lgam1p,
                            [Arg()], rtol=1e-13, dps=100,
                            param_filter=param_filter)

    def test_loggamma(self):
        def mpmath_loggamma(z):
            try:
                res = mpmath.loggamma(z)
            except ValueError:
                res = complex(np.nan, np.nan)
            return res

        assert_mpmath_equal(sc.loggamma,
                            mpmath_loggamma,
                            [ComplexArg()], nan_ok=False,
                            distinguish_nan_and_inf=False, rtol=5e-14)

    @pytest.mark.xfail(run=False)
    def test_pcfd(self):
        def pcfd(v, x):
            return sc.pbdv(v, x)[0]
        assert_mpmath_equal(pcfd,
                            exception_to_nan(lambda v, x: mpmath.pcfd(v, x, **HYPERKW)),
                            [Arg(), Arg()])

    @pytest.mark.xfail(run=False, reason="it's not the same as the mpmath function --- maybe different definition?")
    def test_pcfv(self):
        def pcfv(v, x):
            return sc.pbvv(v, x)[0]
        assert_mpmath_equal(pcfv,
                            lambda v, x: time_limited()(exception_to_nan(mpmath.pcfv))(v, x, **HYPERKW),
                            [Arg(), Arg()], n=1000)

    def test_pcfw(self):
        def pcfw(a, x):
            return sc.pbwa(a, x)[0]

        def dpcfw(a, x):
            return sc.pbwa(a, x)[1]

        def mpmath_dpcfw(a, x):
            return mpmath.diff(mpmath.pcfw, (a, x), (0, 1))

        # The Zhang and Jin implementation only uses Taylor series and
        # is thus accurate in only a very small range.
        assert_mpmath_equal(pcfw,
                            mpmath.pcfw,
                            [Arg(-5, 5), Arg(-5, 5)], rtol=2e-8, n=100)

        assert_mpmath_equal(dpcfw,
                            mpmath_dpcfw,
                            [Arg(-5, 5), Arg(-5, 5)], rtol=2e-9, n=100)

    @pytest.mark.xfail(run=False, reason="issues at large arguments (atol OK, rtol not) and <eps-close to z=0")
    def test_polygamma(self):
        assert_mpmath_equal(sc.polygamma,
                            time_limited()(exception_to_nan(mpmath.polygamma)),
                            [IntArg(0, 1000), Arg()])

    def test_rgamma(self):
        assert_mpmath_equal(
            sc.rgamma,
            mpmath.rgamma,
            [Arg(-8000, np.inf)],
            n=5000,
            nan_ok=False,
            ignore_inf_sign=True,
        )

    def test_rgamma_complex(self):
        assert_mpmath_equal(sc.rgamma,
                            exception_to_nan(mpmath.rgamma),
                            [ComplexArg()], rtol=5e-13)

    @pytest.mark.xfail(reason=("see gh-3551 for bad points on 32 bit "
                               "systems and gh-8095 for another bad "
                               "point"))
    def test_rf(self):
        if LooseVersion(mpmath.__version__) >= LooseVersion("1.0.0"):
            # no workarounds needed
            mppoch = mpmath.rf
        else:
            def mppoch(a, m):
                # deal with cases where the result in double precision
                # hits exactly a non-positive integer, but the
                # corresponding extended-precision mpf floats don't
                if float(a + m) == int(a + m) and float(a + m) <= 0:
                    a = mpmath.mpf(a)
                    m = int(a + m) - a
                return mpmath.rf(a, m)

        assert_mpmath_equal(sc.poch,
                            mppoch,
                            [Arg(), Arg()],
                            dps=400)

    def test_sinpi(self):
        eps = np.finfo(float).eps
        assert_mpmath_equal(_sinpi, mpmath.sinpi,
                            [Arg()], nan_ok=False, rtol=eps)

    def test_sinpi_complex(self):
        assert_mpmath_equal(_sinpi, mpmath.sinpi,
                            [ComplexArg()], nan_ok=False, rtol=2e-14)

    def test_shi(self):
        def shi(x):
            return sc.shichi(x)[0]
        assert_mpmath_equal(shi, mpmath.shi, [Arg()])
        # check asymptotic series cross-over
        assert_mpmath_equal(shi, mpmath.shi, [FixedArg([88 - 1e-9, 88, 88 + 1e-9])])

    def test_shi_complex(self):
        def shi(z):
            return sc.shichi(z)[0]
        # shi oscillates as Im[z] -> +- inf, so limit range
        assert_mpmath_equal(shi,
                            mpmath.shi,
                            [ComplexArg(complex(-np.inf, -1e8), complex(np.inf, 1e8))],
                            rtol=1e-12)

    def test_si(self):
        def si(x):
            return sc.sici(x)[0]
        assert_mpmath_equal(si, mpmath.si, [Arg()])

    def test_si_complex(self):
        def si(z):
            return sc.sici(z)[0]
        # si oscillates as Re[z] -> +- inf, so limit range
        assert_mpmath_equal(si,
                            mpmath.si,
                            [ComplexArg(complex(-1e8, -np.inf), complex(1e8, np.inf))],
                            rtol=1e-12)

    def test_spence(self):
        # mpmath uses a different convention for the dilogarithm
        def dilog(x):
            return mpmath.polylog(2, 1 - x)
        # Spence has a branch cut on the negative real axis
        assert_mpmath_equal(sc.spence,
                            exception_to_nan(dilog),
                            [Arg(0, np.inf)], rtol=1e-14)

    def test_spence_complex(self):
        def dilog(z):
            return mpmath.polylog(2, 1 - z)
        assert_mpmath_equal(sc.spence,
                            exception_to_nan(dilog),
                            [ComplexArg()], rtol=1e-14)

    def test_spherharm(self):
        def spherharm(l, m, theta, phi):
            if m > l:
                return np.nan
            return sc.sph_harm(m, l, phi, theta)
        assert_mpmath_equal(spherharm,
                            mpmath.spherharm,
                            [IntArg(0, 100), IntArg(0, 100),
                             Arg(a=0, b=pi), Arg(a=0, b=2*pi)],
                            atol=1e-8, n=6000,
                            dps=150)

    def test_struveh(self):
        assert_mpmath_equal(sc.struve,
                            exception_to_nan(mpmath.struveh),
                            [Arg(-1e4, 1e4), Arg(0, 1e4)],
                            rtol=5e-10)

    def test_struvel(self):
        def mp_struvel(v, z):
            if v < 0 and z < -v and abs(v) > 1000:
                # larger DPS needed for correct results
                old_dps = mpmath.mp.dps
                try:
                    mpmath.mp.dps = 300
                    return mpmath.struvel(v, z)
                finally:
                    mpmath.mp.dps = old_dps
            return mpmath.struvel(v, z)

        assert_mpmath_equal(sc.modstruve,
                            exception_to_nan(mp_struvel),
                            [Arg(-1e4, 1e4), Arg(0, 1e4)],
                            rtol=5e-10,
                            ignore_inf_sign=True)

    def test_wrightomega_real(self):
        def mpmath_wrightomega_real(x):
            return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))

        # For x < -1000 the Wright Omega function is just 0 to double
        # precision, and for x > 1e21 it is just x to double
        # precision.
        assert_mpmath_equal(
            sc.wrightomega,
            mpmath_wrightomega_real,
            [Arg(-1000, 1e21)],
            rtol=5e-15,
            atol=0,
            nan_ok=False,
        )

    def test_wrightomega(self):
        assert_mpmath_equal(sc.wrightomega,
                            lambda z: _mpmath_wrightomega(z, 25),
                            [ComplexArg()], rtol=1e-14, nan_ok=False)

    def test_hurwitz_zeta(self):
        assert_mpmath_equal(sc.zeta,
                            exception_to_nan(mpmath.zeta),
                            [Arg(a=1, b=1e10, inclusive_a=False),
                             Arg(a=0, inclusive_a=False)])

    def test_riemann_zeta(self):
        assert_mpmath_equal(
            sc.zeta,
            mpmath.zeta,
            [Arg(-100, 100)],
            nan_ok=False,
            rtol=1e-13,
        )

    def test_zetac(self):
        assert_mpmath_equal(sc.zetac,
                            lambda x: mpmath.zeta(x) - 1,
                            [Arg(-100, 100)],
                            nan_ok=False, dps=45, rtol=1e-13)

    def test_boxcox(self):

        def mp_boxcox(x, lmbda):
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            if lmbda == 0:
                return mpmath.mp.log(x)
            else:
                return mpmath.mp.powm1(x, lmbda) / lmbda

        assert_mpmath_equal(sc.boxcox,
                            exception_to_nan(mp_boxcox),
                            [Arg(a=0, inclusive_a=False), Arg()],
                            n=200,
                            dps=60,
                            rtol=1e-13)

    def test_boxcox1p(self):

        def mp_boxcox1p(x, lmbda):
            x = mpmath.mp.mpf(x)
            lmbda = mpmath.mp.mpf(lmbda)
            one = mpmath.mp.mpf(1)
            if lmbda == 0:
                return mpmath.mp.log(one + x)
            else:
                return mpmath.mp.powm1(one + x, lmbda) / lmbda

        assert_mpmath_equal(sc.boxcox1p,
                            exception_to_nan(mp_boxcox1p),
                            [Arg(a=-1, inclusive_a=False), Arg()],
                            n=200,
                            dps=60,
                            rtol=1e-13)

    def test_spherical_jn(self):
        def mp_spherical_jn(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.besselj(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n), z),
                            exception_to_nan(mp_spherical_jn),
                            [IntArg(0, 200), Arg(-1e8, 1e8)],
                            dps=300)

    def test_spherical_jn_complex(self):
        def mp_spherical_jn(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.besselj(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n.real), z),
                            exception_to_nan(mp_spherical_jn),
                            [IntArg(0, 200), ComplexArg()])

    def test_spherical_yn(self):
        def mp_spherical_yn(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.bessely(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_yn(int(n), z),
                            exception_to_nan(mp_spherical_yn),
                            [IntArg(0, 200), Arg(-1e10, 1e10)],
                            dps=100)

    def test_spherical_yn_complex(self):
        def mp_spherical_yn(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.bessely(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_yn(int(n.real), z),
                            exception_to_nan(mp_spherical_yn),
                            [IntArg(0, 200), ComplexArg()])

    def test_spherical_in(self):
        def mp_spherical_in(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.besseli(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n), z),
                            exception_to_nan(mp_spherical_in),
                            [IntArg(0, 200), Arg()],
                            dps=200, atol=10**(-278))

    def test_spherical_in_complex(self):
        def mp_spherical_in(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.besseli(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n.real), z),
                            exception_to_nan(mp_spherical_in),
                            [IntArg(0, 200), ComplexArg()])

    def test_spherical_kn(self):
        def mp_spherical_kn(n, z):
            out = (mpmath.besselk(n + mpmath.mpf(1)/2, z) *
                   mpmath.sqrt(mpmath.pi/(2*mpmath.mpmathify(z))))
            if mpmath.mpmathify(z).imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n), z),
                            exception_to_nan(mp_spherical_kn),
                            [IntArg(0, 150), Arg()],
                            dps=100)

    @pytest.mark.xfail(run=False, reason="Accuracy issues near z = -1 inherited from kv.")
    def test_spherical_kn_complex(self):
        def mp_spherical_kn(n, z):
            arg = mpmath.mpmathify(z)
            out = (mpmath.besselk(n + mpmath.mpf(1)/2, arg) /
                   mpmath.sqrt(2*arg/mpmath.pi))
            if arg.imag == 0:
                return out.real
            else:
                return out

        assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n.real), z),
                            exception_to_nan(mp_spherical_kn),
                            [IntArg(0, 200), ComplexArg()],
                            dps=200)
