import sys

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
                           assert_equal)
from pytest import raises as assert_raises

from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
                                  estimate_bc_jac, compute_jac_indices,
                                  construct_global_jac, solve_bvp)


def exp_fun(x, y):
    return np.vstack((y[1], y[0]))


def exp_fun_jac(x, y):
    df_dy = np.empty((2, 2, x.shape[0]))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = 1
    df_dy[1, 1] = 0
    return df_dy


def exp_bc(ya, yb):
    return np.hstack((ya[0] - 1, yb[0]))


def exp_bc_complex(ya, yb):
    return np.hstack((ya[0] - 1 - 1j, yb[0]))


def exp_bc_jac(ya, yb):
    dbc_dya = np.array([
        [1, 0],
        [0, 0]
    ])
    dbc_dyb = np.array([
        [0, 0],
        [1, 0]
    ])
    return dbc_dya, dbc_dyb


def exp_sol(x):
    return (np.exp(-x) - np.exp(x - 2)) / (1 - np.exp(-2))


def sl_fun(x, y, p):
    return np.vstack((y[1], -p[0]**2 * y[0]))


def sl_fun_jac(x, y, p):
    n, m = y.shape
    df_dy = np.empty((n, 2, m))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = -p[0]**2
    df_dy[1, 1] = 0

    df_dp = np.empty((n, 1, m))
    df_dp[0, 0] = 0
    df_dp[1, 0] = -2 * p[0] * y[0]

    return df_dy, df_dp


def sl_bc(ya, yb, p):
    return np.hstack((ya[0], yb[0], ya[1] - p[0]))


def sl_bc_jac(ya, yb, p):
    dbc_dya = np.zeros((3, 2))
    dbc_dya[0, 0] = 1
    dbc_dya[2, 1] = 1

    dbc_dyb = np.zeros((3, 2))
    dbc_dyb[1, 0] = 1

    dbc_dp = np.zeros((3, 1))
    dbc_dp[2, 0] = -1

    return dbc_dya, dbc_dyb, dbc_dp


def sl_sol(x, p):
    return np.sin(p[0] * x)


def emden_fun(x, y):
    return np.vstack((y[1], -y[0]**5))


def emden_fun_jac(x, y):
    df_dy = np.empty((2, 2, x.shape[0]))
    df_dy[0, 0] = 0
    df_dy[0, 1] = 1
    df_dy[1, 0] = -5 * y[0]**4
    df_dy[1, 1] = 0
    return df_dy


def emden_bc(ya, yb):
    return np.array([ya[1], yb[0] - (3/4)**0.5])


def emden_bc_jac(ya, yb):
    dbc_dya = np.array([
        [0, 1],
        [0, 0]
    ])
    dbc_dyb = np.array([
        [0, 0],
        [1, 0]
    ])
    return dbc_dya, dbc_dyb


def emden_sol(x):
    return (1 + x**2/3)**-0.5


def undefined_fun(x, y):
    return np.zeros_like(y)


def undefined_bc(ya, yb):
    return np.array([ya[0], yb[0] - 1])


def big_fun(x, y):
    f = np.zeros_like(y)
    f[::2] = y[1::2]
    return f


def big_bc(ya, yb):
    return np.hstack((ya[::2], yb[::2] - 1))


def big_sol(x, n):
    y = np.ones((2 * n, x.size))
    y[::2] = x
    return x


def big_fun_with_parameters(x, y, p):
    """ Big version of sl_fun, with two parameters.

    The two differential equations represented by sl_fun are broadcast to the
    number of rows of y, rotating between the parameters p[0] and p[1].
    Here are the differential equations:

        dy[0]/dt = y[1]
        dy[1]/dt = -p[0]**2 * y[0]
        dy[2]/dt = y[3]
        dy[3]/dt = -p[1]**2 * y[2]
        dy[4]/dt = y[5]
        dy[5]/dt = -p[0]**2 * y[4]
        dy[6]/dt = y[7]
        dy[7]/dt = -p[1]**2 * y[6]
        .
        .
        .

    """
    f = np.zeros_like(y)
    f[::2] = y[1::2]
    f[1::4] = -p[0]**2 * y[::4]
    f[3::4] = -p[1]**2 * y[2::4]
    return f


def big_fun_with_parameters_jac(x, y, p):
    # big version of sl_fun_jac, with two parameters
    n, m = y.shape
    df_dy = np.zeros((n, n, m))
    df_dy[range(0, n, 2), range(1, n, 2)] = 1
    df_dy[range(1, n, 4), range(0, n, 4)] = -p[0]**2
    df_dy[range(3, n, 4), range(2, n, 4)] = -p[1]**2

    df_dp = np.zeros((n, 2, m))
    df_dp[range(1, n, 4), 0] = -2 * p[0] * y[range(0, n, 4)]
    df_dp[range(3, n, 4), 1] = -2 * p[1] * y[range(2, n, 4)]

    return df_dy, df_dp


def big_bc_with_parameters(ya, yb, p):
    # big version of sl_bc, with two parameters
    return np.hstack((ya[::2], yb[::2], ya[1] - p[0], ya[3] - p[1]))


def big_bc_with_parameters_jac(ya, yb, p):
    # big version of sl_bc_jac, with two parameters
    n = ya.shape[0]
    dbc_dya = np.zeros((n + 2, n))
    dbc_dyb = np.zeros((n + 2, n))

    dbc_dya[range(n // 2), range(0, n, 2)] = 1
    dbc_dyb[range(n // 2, n), range(0, n, 2)] = 1

    dbc_dp = np.zeros((n + 2, 2))
    dbc_dp[n, 0] = -1
    dbc_dya[n, 1] = 1
    dbc_dp[n + 1, 1] = -1
    dbc_dya[n + 1, 3] = 1

    return dbc_dya, dbc_dyb, dbc_dp


def big_sol_with_parameters(x, p):
    # big version of sl_sol, with two parameters
    return np.vstack((np.sin(p[0] * x), np.sin(p[1] * x)))


def shock_fun(x, y):
    eps = 1e-3
    return np.vstack((
        y[1],
        -(x * y[1] + eps * np.pi**2 * np.cos(np.pi * x) +
          np.pi * x * np.sin(np.pi * x)) / eps
    ))


def shock_bc(ya, yb):
    return np.array([ya[0] + 2, yb[0]])


def shock_sol(x):
    eps = 1e-3
    k = np.sqrt(2 * eps)
    return np.cos(np.pi * x) + erf(x / k) / erf(1 / k)


def nonlin_bc_fun(x, y):
    # laplace eq.
    return np.stack([y[1], np.zeros_like(x)])


def nonlin_bc_bc(ya, yb):
    phiA, phipA = ya
    phiC, phipC = yb

    kappa, ioA, ioC, V, f = 1.64, 0.01, 1.0e-4, 0.5, 38.9

    # Butler-Volmer Kinetics at Anode
    hA = 0.0-phiA-0.0
    iA = ioA * (np.exp(f*hA) - np.exp(-f*hA))
    res0 = iA + kappa * phipA

    # Butler-Volmer Kinetics at Cathode
    hC = V - phiC - 1.0
    iC = ioC * (np.exp(f*hC) - np.exp(-f*hC))
    res1 = iC - kappa*phipC

    return np.array([res0, res1])


def nonlin_bc_sol(x):
    return -0.13426436116763119 - 1.1308709 * x


def test_modify_mesh():
    x = np.array([0, 1, 3, 9], dtype=float)
    x_new = modify_mesh(x, np.array([0]), np.array([2]))
    assert_array_equal(x_new, np.array([0, 0.5, 1, 3, 5, 7, 9]))

    x = np.array([-6, -3, 0, 3, 6], dtype=float)
    x_new = modify_mesh(x, np.array([1], dtype=int), np.array([0, 2, 3]))
    assert_array_equal(x_new, [-6, -5, -4, -3, -1.5, 0, 1, 2, 3, 4, 5, 6])


def test_compute_fun_jac():
    x = np.linspace(0, 1, 5)
    y = np.empty((2, x.shape[0]))
    y[0] = 0.01
    y[1] = 0.02
    p = np.array([])
    df_dy, df_dp = estimate_fun_jac(lambda x, y, p: exp_fun(x, y), x, y, p)
    df_dy_an = exp_fun_jac(x, y)
    assert_allclose(df_dy, df_dy_an)
    assert_(df_dp is None)

    x = np.linspace(0, np.pi, 5)
    y = np.empty((2, x.shape[0]))
    y[0] = np.sin(x)
    y[1] = np.cos(x)
    p = np.array([1.0])
    df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
    df_dy_an, df_dp_an = sl_fun_jac(x, y, p)
    assert_allclose(df_dy, df_dy_an)
    assert_allclose(df_dp, df_dp_an)

    x = np.linspace(0, 1, 10)
    y = np.empty((2, x.shape[0]))
    y[0] = (3/4)**0.5
    y[1] = 1e-4
    p = np.array([])
    df_dy, df_dp = estimate_fun_jac(lambda x, y, p: emden_fun(x, y), x, y, p)
    df_dy_an = emden_fun_jac(x, y)
    assert_allclose(df_dy, df_dy_an)
    assert_(df_dp is None)


def test_compute_bc_jac():
    ya = np.array([-1.0, 2])
    yb = np.array([0.5, 3])
    p = np.array([])
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(
        lambda ya, yb, p: exp_bc(ya, yb), ya, yb, p)
    dbc_dya_an, dbc_dyb_an = exp_bc_jac(ya, yb)
    assert_allclose(dbc_dya, dbc_dya_an)
    assert_allclose(dbc_dyb, dbc_dyb_an)
    assert_(dbc_dp is None)

    ya = np.array([0.0, 1])
    yb = np.array([0.0, -1])
    p = np.array([0.5])
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, ya, yb, p)
    dbc_dya_an, dbc_dyb_an, dbc_dp_an = sl_bc_jac(ya, yb, p)
    assert_allclose(dbc_dya, dbc_dya_an)
    assert_allclose(dbc_dyb, dbc_dyb_an)
    assert_allclose(dbc_dp, dbc_dp_an)

    ya = np.array([0.5, 100])
    yb = np.array([-1000, 10.5])
    p = np.array([])
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(
        lambda ya, yb, p: emden_bc(ya, yb), ya, yb, p)
    dbc_dya_an, dbc_dyb_an = emden_bc_jac(ya, yb)
    assert_allclose(dbc_dya, dbc_dya_an)
    assert_allclose(dbc_dyb, dbc_dyb_an)
    assert_(dbc_dp is None)


def test_compute_jac_indices():
    n = 2
    m = 4
    k = 2
    i, j = compute_jac_indices(n, m, k)
    s = coo_matrix((np.ones_like(i), (i, j))).toarray()
    s_true = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    ])
    assert_array_equal(s, s_true)


def test_compute_global_jac():
    n = 2
    m = 5
    k = 1
    i_jac, j_jac = compute_jac_indices(2, 5, 1)
    x = np.linspace(0, 1, 5)
    h = np.diff(x)
    y = np.vstack((np.sin(np.pi * x), np.pi * np.cos(np.pi * x)))
    p = np.array([3.0])

    f = sl_fun(x, y, p)

    x_middle = x[:-1] + 0.5 * h
    y_middle = 0.5 * (y[:, :-1] + y[:, 1:]) - h/8 * (f[:, 1:] - f[:, :-1])

    df_dy, df_dp = sl_fun_jac(x, y, p)
    df_dy_middle, df_dp_middle = sl_fun_jac(x_middle, y_middle, p)
    dbc_dya, dbc_dyb, dbc_dp = sl_bc_jac(y[:, 0], y[:, -1], p)

    J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle,
                             df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    J = J.toarray()

    def J_block(h, p):
        return np.array([
            [h**2*p**2/12 - 1, -0.5*h, -h**2*p**2/12 + 1, -0.5*h],
            [0.5*h*p**2, h**2*p**2/12 - 1, 0.5*h*p**2, 1 - h**2*p**2/12]
        ])

    J_true = np.zeros((m * n + k, m * n + k))
    for i in range(m - 1):
        J_true[i * n: (i + 1) * n, i * n: (i + 2) * n] = J_block(h[i], p[0])

    J_true[:(m - 1) * n:2, -1] = p * h**2/6 * (y[0, :-1] - y[0, 1:])
    J_true[1:(m - 1) * n:2, -1] = p * (h * (y[0, :-1] + y[0, 1:]) +
                                       h**2/6 * (y[1, :-1] - y[1, 1:]))

    J_true[8, 0] = 1
    J_true[9, 8] = 1
    J_true[10, 1] = 1
    J_true[10, 10] = -1

    assert_allclose(J, J_true, rtol=1e-10)

    df_dy, df_dp = estimate_fun_jac(sl_fun, x, y, p)
    df_dy_middle, df_dp_middle = estimate_fun_jac(sl_fun, x_middle, y_middle, p)
    dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(sl_bc, y[:, 0], y[:, -1], p)
    J = construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle,
                             df_dp, df_dp_middle, dbc_dya, dbc_dyb, dbc_dp)
    J = J.toarray()
    assert_allclose(J, J_true, rtol=2e-8, atol=2e-8)


def test_parameter_validation():
    x = [0, 1, 0.5]
    y = np.zeros((2, 3))
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)

    x = np.linspace(0, 1, 5)
    y = np.zeros((2, 4))
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)

    fun = lambda x, y, p: exp_fun(x, y)
    bc = lambda ya, yb, p: exp_bc(ya, yb)

    y = np.zeros((2, x.shape[0]))
    assert_raises(ValueError, solve_bvp, fun, bc, x, y, p=[1])

    def wrong_shape_fun(x, y):
        return np.zeros(3)

    assert_raises(ValueError, solve_bvp, wrong_shape_fun, bc, x, y)

    S = np.array([[0, 0]])
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y, S=S)


def test_no_params():
    x = np.linspace(0, 1, 5)
    x_test = np.linspace(0, 1, 100)
    y = np.zeros((2, x.shape[0]))
    for fun_jac in [None, exp_fun_jac]:
        for bc_jac in [None, exp_bc_jac]:
            sol = solve_bvp(exp_fun, exp_bc, x, y, fun_jac=fun_jac,
                            bc_jac=bc_jac)

            assert_equal(sol.status, 0)
            assert_(sol.success)

            assert_equal(sol.x.size, 5)

            sol_test = sol.sol(x_test)

            assert_allclose(sol_test[0], exp_sol(x_test), atol=1e-5)

            f_test = exp_fun(x_test, sol_test)
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res**2, axis=0)**0.5
            assert_(np.all(norm_res < 1e-3))

            assert_(np.all(sol.rms_residuals < 1e-3))
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_with_params():
    x = np.linspace(0, np.pi, 5)
    x_test = np.linspace(0, np.pi, 100)
    y = np.ones((2, x.shape[0]))

    for fun_jac in [None, sl_fun_jac]:
        for bc_jac in [None, sl_bc_jac]:
            sol = solve_bvp(sl_fun, sl_bc, x, y, p=[0.5], fun_jac=fun_jac,
                            bc_jac=bc_jac)

            assert_equal(sol.status, 0)
            assert_(sol.success)

            assert_(sol.x.size < 10)

            assert_allclose(sol.p, [1], rtol=1e-4)

            sol_test = sol.sol(x_test)

            assert_allclose(sol_test[0], sl_sol(x_test, [1]),
                            rtol=1e-4, atol=1e-4)

            f_test = sl_fun(x_test, sol_test, [1])
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
            assert_(np.all(norm_res < 1e-3))

            assert_(np.all(sol.rms_residuals < 1e-3))
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_singular_term():
    x = np.linspace(0, 1, 10)
    x_test = np.linspace(0.05, 1, 100)
    y = np.empty((2, 10))
    y[0] = (3/4)**0.5
    y[1] = 1e-4
    S = np.array([[0, 0], [0, -2]])

    for fun_jac in [None, emden_fun_jac]:
        for bc_jac in [None, emden_bc_jac]:
            sol = solve_bvp(emden_fun, emden_bc, x, y, S=S, fun_jac=fun_jac,
                            bc_jac=bc_jac)

            assert_equal(sol.status, 0)
            assert_(sol.success)

            assert_equal(sol.x.size, 10)

            sol_test = sol.sol(x_test)
            assert_allclose(sol_test[0], emden_sol(x_test), atol=1e-5)

            f_test = emden_fun(x_test, sol_test) + S.dot(sol_test) / x_test
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5

            assert_(np.all(norm_res < 1e-3))
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_complex():
    # The test is essentially the same as test_no_params, but boundary
    # conditions are turned into complex.
    x = np.linspace(0, 1, 5)
    x_test = np.linspace(0, 1, 100)
    y = np.zeros((2, x.shape[0]), dtype=complex)
    for fun_jac in [None, exp_fun_jac]:
        for bc_jac in [None, exp_bc_jac]:
            sol = solve_bvp(exp_fun, exp_bc_complex, x, y, fun_jac=fun_jac,
                            bc_jac=bc_jac)

            assert_equal(sol.status, 0)
            assert_(sol.success)

            sol_test = sol.sol(x_test)

            assert_allclose(sol_test[0].real, exp_sol(x_test), atol=1e-5)
            assert_allclose(sol_test[0].imag, exp_sol(x_test), atol=1e-5)

            f_test = exp_fun(x_test, sol_test)
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(np.real(rel_res * np.conj(rel_res)),
                              axis=0) ** 0.5
            assert_(np.all(norm_res < 1e-3))

            assert_(np.all(sol.rms_residuals < 1e-3))
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_failures():
    x = np.linspace(0, 1, 2)
    y = np.zeros((2, x.size))
    res = solve_bvp(exp_fun, exp_bc, x, y, tol=1e-5, max_nodes=5)
    assert_equal(res.status, 1)
    assert_(not res.success)

    x = np.linspace(0, 1, 5)
    y = np.zeros((2, x.size))
    res = solve_bvp(undefined_fun, undefined_bc, x, y)
    assert_equal(res.status, 2)
    assert_(not res.success)


def test_big_problem():
    n = 30
    x = np.linspace(0, 1, 5)
    y = np.zeros((2 * n, x.size))
    sol = solve_bvp(big_fun, big_bc, x, y)

    assert_equal(sol.status, 0)
    assert_(sol.success)

    sol_test = sol.sol(x)

    assert_allclose(sol_test[0], big_sol(x, n))

    f_test = big_fun(x, sol_test)
    r = sol.sol(x, 1) - f_test
    rel_res = r / (1 + np.abs(f_test))
    norm_res = np.sum(np.real(rel_res * np.conj(rel_res)), axis=0) ** 0.5
    assert_(np.all(norm_res < 1e-3))

    assert_(np.all(sol.rms_residuals < 1e-3))
    assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_big_problem_with_parameters():
    n = 30
    x = np.linspace(0, np.pi, 5)
    x_test = np.linspace(0, np.pi, 100)
    y = np.ones((2 * n, x.size))

    for fun_jac in [None, big_fun_with_parameters_jac]:
        for bc_jac in [None, big_bc_with_parameters_jac]:
            sol = solve_bvp(big_fun_with_parameters, big_bc_with_parameters, x,
                            y, p=[0.5, 0.5], fun_jac=fun_jac, bc_jac=bc_jac)

            assert_equal(sol.status, 0)
            assert_(sol.success)

            assert_allclose(sol.p, [1, 1], rtol=1e-4)

            sol_test = sol.sol(x_test)

            for isol in range(0, n, 4):
                assert_allclose(sol_test[isol],
                                big_sol_with_parameters(x_test, [1, 1])[0],
                                rtol=1e-4, atol=1e-4)
                assert_allclose(sol_test[isol + 2],
                                big_sol_with_parameters(x_test, [1, 1])[1],
                                rtol=1e-4, atol=1e-4)

            f_test = big_fun_with_parameters(x_test, sol_test, [1, 1])
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
            assert_(np.all(norm_res < 1e-3))

            assert_(np.all(sol.rms_residuals < 1e-3))
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_shock_layer():
    x = np.linspace(-1, 1, 5)
    x_test = np.linspace(-1, 1, 100)
    y = np.zeros((2, x.size))
    sol = solve_bvp(shock_fun, shock_bc, x, y)

    assert_equal(sol.status, 0)
    assert_(sol.success)

    assert_(sol.x.size < 110)

    sol_test = sol.sol(x_test)
    assert_allclose(sol_test[0], shock_sol(x_test), rtol=1e-5, atol=1e-5)

    f_test = shock_fun(x_test, sol_test)
    r = sol.sol(x_test, 1) - f_test
    rel_res = r / (1 + np.abs(f_test))
    norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5

    assert_(np.all(norm_res < 1e-3))
    assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_nonlin_bc():
    x = np.linspace(0, 0.1, 5)
    x_test = x
    y = np.zeros([2, x.size])
    sol = solve_bvp(nonlin_bc_fun, nonlin_bc_bc, x, y)

    assert_equal(sol.status, 0)
    assert_(sol.success)

    assert_(sol.x.size < 8)

    sol_test = sol.sol(x_test)
    assert_allclose(sol_test[0], nonlin_bc_sol(x_test), rtol=1e-5, atol=1e-5)

    f_test = nonlin_bc_fun(x_test, sol_test)
    r = sol.sol(x_test, 1) - f_test
    rel_res = r / (1 + np.abs(f_test))
    norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5

    assert_(np.all(norm_res < 1e-3))
    assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
    assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)


def test_verbose():
    # Smoke test that checks the printing does something and does not crash
    x = np.linspace(0, 1, 5)
    y = np.zeros((2, x.shape[0]))
    for verbose in [0, 1, 2]:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            sol = solve_bvp(exp_fun, exp_bc, x, y, verbose=verbose)
            text = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert_(sol.success)
        if verbose == 0:
            assert_(not text, text)
        if verbose >= 1:
            assert_("Solved in" in text, text)
        if verbose >= 2:
            assert_("Max residual" in text, text)
