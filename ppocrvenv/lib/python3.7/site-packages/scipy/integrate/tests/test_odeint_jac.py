
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.integrate import odeint
import scipy.integrate._test_odeint_banded as banded5x5


def rhs(y, t):
    dydt = np.zeros_like(y)
    banded5x5.banded5x5(t, y, dydt)
    return dydt


def jac(y, t):
    n = len(y)
    jac = np.zeros((n, n), order='F')
    banded5x5.banded5x5_jac(t, y, 1, 1, jac)
    return jac


def bjac(y, t):
    n = len(y)
    bjac = np.zeros((4, n), order='F')
    banded5x5.banded5x5_bjac(t, y, 1, 1, bjac)
    return bjac


JACTYPE_FULL = 1
JACTYPE_BANDED = 4


def check_odeint(jactype):
    if jactype == JACTYPE_FULL:
        ml = None
        mu = None
        jacobian = jac
    elif jactype == JACTYPE_BANDED:
        ml = 2
        mu = 1
        jacobian = bjac
    else:
        raise ValueError("invalid jactype: %r" % (jactype,))

    y0 = np.arange(1.0, 6.0)
    # These tolerances must match the tolerances used in banded5x5.f.
    rtol = 1e-11
    atol = 1e-13
    dt = 0.125
    nsteps = 64
    t = dt * np.arange(nsteps+1)

    sol, info = odeint(rhs, y0, t,
                       Dfun=jacobian, ml=ml, mu=mu,
                       atol=atol, rtol=rtol, full_output=True)
    yfinal = sol[-1]
    odeint_nst = info['nst'][-1]
    odeint_nfe = info['nfe'][-1]
    odeint_nje = info['nje'][-1]

    y1 = y0.copy()
    # Pure Fortran solution. y1 is modified in-place.
    nst, nfe, nje = banded5x5.banded5x5_solve(y1, nsteps, dt, jactype)

    # It is likely that yfinal and y1 are *exactly* the same, but
    # we'll be cautious and use assert_allclose.
    assert_allclose(yfinal, y1, rtol=1e-12)
    assert_equal((odeint_nst, odeint_nfe, odeint_nje), (nst, nfe, nje))


def test_odeint_full_jac():
    check_odeint(JACTYPE_FULL)


def test_odeint_banded_jac():
    check_odeint(JACTYPE_BANDED)
