import numpy as np
from scipy.optimize import _lbfgsb


def objfun(x):
    """simplified objective func to test lbfgsb bound violation"""
    x0 = [0.8750000000000278,
          0.7500000000000153,
          0.9499999999999722,
          0.8214285714285992,
          0.6363636363636085]
    x1 = [1.0, 0.0, 1.0, 0.0, 0.0]
    x2 = [1.0,
          0.0,
          0.9889733043149325,
          0.0,
          0.026353554421041155]
    x3 = [1.0,
          0.0,
          0.9889917442915558,
          0.0,
          0.020341986743231205]

    f0 = 5163.647901211178
    f1 = 5149.8181642072905
    f2 = 5149.379332309634
    f3 = 5149.374490771297

    g0 = np.array([-0.5934820547965749,
                   1.6251549718258351,
                   -71.99168459202559,
                   5.346636965797545,
                   37.10732723092604])
    g1 = np.array([-0.43295349282641515,
                   1.008607936794592,
                   18.223666726602975,
                   31.927010036981997,
                   -19.667512518739386])
    g2 = np.array([-0.4699874455100256,
                   0.9466285353668347,
                   -0.016874360242016825,
                   48.44999161133457,
                   5.819631620590712])
    g3 = np.array([-0.46970678696829116,
                   0.9612719312174818,
                   0.006129809488833699,
                   48.43557729419473,
                   6.005481418498221])

    if np.allclose(x, x0):
        f = f0
        g = g0
    elif np.allclose(x, x1):
        f = f1
        g = g1
    elif np.allclose(x, x2):
        f = f2
        g = g2
    elif np.allclose(x, x3):
        f = f3
        g = g3
    else:
        raise ValueError(
            'Simplified objective function not defined '
            'at requested point')
    return (np.copy(f), np.copy(g))


def test_setulb_floatround():
    """test if setulb() violates bounds

    checks for violation due to floating point rounding error
    """

    n = 5
    m = 10
    factr = 1e7
    pgtol = 1e-5
    maxls = 20
    iprint = -1
    nbd = np.full((n,), 2)
    low_bnd = np.zeros(n, np.float64)
    upper_bnd = np.ones(n, np.float64)

    x0 = np.array(
        [0.8750000000000278,
         0.7500000000000153,
         0.9499999999999722,
         0.8214285714285992,
         0.6363636363636085])
    x = np.copy(x0)

    f = np.array(0.0, np.float64)
    g = np.zeros(n, np.float64)

    fortran_int = _lbfgsb.types.intvar.dtype

    wa = np.zeros(2*m*n + 5*n + 11*m*m + 8*m, np.float64)
    iwa = np.zeros(3*n, fortran_int)
    task = np.zeros(1, 'S60')
    csave = np.zeros(1, 'S60')
    lsave = np.zeros(4, fortran_int)
    isave = np.zeros(44, fortran_int)
    dsave = np.zeros(29, np.float64)

    task[:] = b'START'

    for n_iter in range(7):  # 7 steps required to reproduce error
        f, g = objfun(x)

        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, iprint, csave, lsave,
                       isave, dsave, maxls)

        assert (x <= upper_bnd).all() and (x >= low_bnd).all(), (
            "_lbfgsb.setulb() stepped to a point outside of the bounds")
