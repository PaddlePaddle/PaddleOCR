from itertools import product
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix


def fun_zero(t, y):
    return np.zeros_like(y)


def fun_linear(t, y):
    return np.array([-y[0] - 5 * y[1], y[0] + y[1]])


def jac_linear():
    return np.array([[-1, -5], [1, 1]])


def sol_linear(t):
    return np.vstack((-5 * np.sin(2 * t),
                      2 * np.cos(2 * t) + np.sin(2 * t)))


def fun_rational(t, y):
    return np.array([y[1] / t,
                     y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))])


def fun_rational_vectorized(t, y):
    return np.vstack((y[1] / t,
                      y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))))


def jac_rational(t, y):
    return np.array([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def jac_rational_sparse(t, y):
    return csc_matrix([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def sol_rational(t):
    return np.asarray((t / (t + 10), 10 * t / (t + 10) ** 2))


def fun_medazko(t, y):
    n = y.shape[0] // 2
    k = 100
    c = 4

    phi = 2 if t <= 5 else 0
    y = np.hstack((phi, 0, y, y[-2]))

    d = 1 / n
    j = np.arange(n) + 1
    alpha = 2 * (j * d - 1) ** 3 / c ** 2
    beta = (j * d - 1) ** 4 / c ** 2

    j_2_p1 = 2 * j + 2
    j_2_m3 = 2 * j - 2
    j_2_m1 = 2 * j
    j_2 = 2 * j + 1

    f = np.empty(2 * n)
    f[::2] = (alpha * (y[j_2_p1] - y[j_2_m3]) / (2 * d) +
              beta * (y[j_2_m3] - 2 * y[j_2_m1] + y[j_2_p1]) / d ** 2 -
              k * y[j_2_m1] * y[j_2])
    f[1::2] = -k * y[j_2] * y[j_2_m1]

    return f


def medazko_sparsity(n):
    cols = []
    rows = []

    i = np.arange(n) * 2

    cols.append(i[1:])
    rows.append(i[1:] - 2)

    cols.append(i)
    rows.append(i)

    cols.append(i)
    rows.append(i + 1)

    cols.append(i[:-1])
    rows.append(i[:-1] + 2)

    i = np.arange(n) * 2 + 1

    cols.append(i)
    rows.append(i)

    cols.append(i)
    rows.append(i - 1)

    cols = np.hstack(cols)
    rows = np.hstack(rows)

    return coo_matrix((np.ones_like(cols), (cols, rows)))


def fun_complex(t, y):
    return -y


def jac_complex(t, y):
    return -np.eye(y.shape[0])


def jac_complex_sparse(t, y):
    return csc_matrix(jac_complex(t, y))


def sol_complex(t):
    y = (0.5 + 1j) * np.exp(-t)
    return y.reshape((1, -1))


def compute_error(y, y_true, rtol, atol):
    e = (y - y_true) / (atol + rtol * np.abs(y_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])


def test_integration():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]

    for vectorized, method, t_span, jac in product(
            [False, True],
            ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'],
            [[5, 9], [5, 1]],
            [None, jac_rational, jac_rational_sparse]):

        if vectorized:
            fun = fun_rational_vectorized
        else:
            fun = fun_rational

        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "The following arguments have no effect for a chosen "
                       "solver: `jac`")
            res = solve_ivp(fun, t_span, y0, rtol=rtol,
                            atol=atol, method=method, dense_output=True,
                            jac=jac, vectorized=vectorized)
        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        if method == 'DOP853':
            # DOP853 spends more functions evaluation because it doesn't
            # have enough time to develop big enough step size.
            assert_(res.nfev < 50)
        else:
            assert_(res.nfev < 40)

        if method in ['RK23', 'RK45', 'DOP853', 'LSODA']:
            assert_equal(res.njev, 0)
            assert_equal(res.nlu, 0)
        else:
            assert_(0 < res.njev < 3)
            assert_(0 < res.nlu < 10)

        y_true = sol_rational(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 5))

        tc = np.linspace(*t_span)
        yc_true = sol_rational(tc)
        yc = res.sol(tc)

        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 5))

        tc = (t_span[0] + t_span[-1]) / 2
        yc_true = sol_rational(tc)
        yc = res.sol(tc)

        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 5))

        # LSODA for some reasons doesn't pass the polynomial through the
        # previous points exactly after the order change. It might be some
        # bug in LSOSA implementation or maybe we missing something.
        if method != 'LSODA':
            assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)


def test_integration_complex():
    rtol = 1e-3
    atol = 1e-6
    y0 = [0.5 + 1j]
    t_span = [0, 1]
    tc = np.linspace(t_span[0], t_span[1])
    for method, jac in product(['RK23', 'RK45', 'DOP853', 'BDF'],
                               [None, jac_complex, jac_complex_sparse]):
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "The following arguments have no effect for a chosen "
                       "solver: `jac`")
            res = solve_ivp(fun_complex, t_span, y0, method=method,
                            dense_output=True, rtol=rtol, atol=atol, jac=jac)

        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        if method == 'DOP853':
            assert res.nfev < 35
        else:
            assert res.nfev < 25

        if method == 'BDF':
            assert_equal(res.njev, 1)
            assert res.nlu < 6
        else:
            assert res.njev == 0
            assert res.nlu == 0

        y_true = sol_complex(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert np.all(e < 5)

        yc_true = sol_complex(tc)
        yc = res.sol(tc)
        e = compute_error(yc, yc_true, rtol, atol)

        assert np.all(e < 5)


def test_integration_sparse_difference():
    n = 200
    t_span = [0, 20]
    y0 = np.zeros(2 * n)
    y0[1::2] = 1
    sparsity = medazko_sparsity(n)

    for method in ['BDF', 'Radau']:
        res = solve_ivp(fun_medazko, t_span, y0, method=method,
                        jac_sparsity=sparsity)

        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        assert_allclose(res.y[78, -1], 0.233994e-3, rtol=1e-2)
        assert_allclose(res.y[79, -1], 0, atol=1e-3)
        assert_allclose(res.y[148, -1], 0.359561e-3, rtol=1e-2)
        assert_allclose(res.y[149, -1], 0, atol=1e-3)
        assert_allclose(res.y[198, -1], 0.117374129e-3, rtol=1e-2)
        assert_allclose(res.y[199, -1], 0.6190807e-5, atol=1e-3)
        assert_allclose(res.y[238, -1], 0, atol=1e-3)
        assert_allclose(res.y[239, -1], 0.9999997, rtol=1e-2)


def test_integration_const_jac():
    rtol = 1e-3
    atol = 1e-6
    y0 = [0, 2]
    t_span = [0, 2]
    J = jac_linear()
    J_sparse = csc_matrix(J)

    for method, jac in product(['Radau', 'BDF'], [J, J_sparse]):
        res = solve_ivp(fun_linear, t_span, y0, rtol=rtol, atol=atol,
                        method=method, dense_output=True, jac=jac)
        assert_equal(res.t[0], t_span[0])
        assert_(res.t_events is None)
        assert_(res.y_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        assert_(res.nfev < 100)
        assert_equal(res.njev, 0)
        assert_(0 < res.nlu < 15)

        y_true = sol_linear(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 10))

        tc = np.linspace(*t_span)
        yc_true = sol_linear(tc)
        yc = res.sol(tc)

        e = compute_error(yc, yc_true, rtol, atol)
        assert_(np.all(e < 15))

        assert_allclose(res.sol(res.t), res.y, rtol=1e-14, atol=1e-14)


@pytest.mark.slow
@pytest.mark.parametrize('method', ['Radau', 'BDF', 'LSODA'])
def test_integration_stiff(method):
    rtol = 1e-6
    atol = 1e-6
    y0 = [1e4, 0, 0]
    tspan = [0, 1e8]

    def fun_robertson(t, state):
        x, y, z = state
        return [
            -0.04 * x + 1e4 * y * z,
            0.04 * x - 1e4 * y * z - 3e7 * y * y,
            3e7 * y * y,
        ]

    res = solve_ivp(fun_robertson, tspan, y0, rtol=rtol,
                    atol=atol, method=method)

    # If the stiff mode is not activated correctly, these numbers will be much bigger
    assert res.nfev < 5000
    assert res.njev < 200


def test_events():
    def event_rational_1(t, y):
        return y[0] - y[1] ** 0.7

    def event_rational_2(t, y):
        return y[1] ** 0.6 - y[0]

    def event_rational_3(t, y):
        return t - 7.4

    event_rational_3.terminal = True

    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        res = solve_ivp(fun_rational, [5, 8], [1/3, 2/9], method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 1)
        assert_equal(res.t_events[1].size, 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        assert_equal(res.y_events[0].shape, (1, 2))
        assert_equal(res.y_events[1].shape, (1, 2))
        assert np.isclose(
            event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)
        assert np.isclose(
            event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

        event_rational_1.direction = 1
        event_rational_2.direction = 1
        res = solve_ivp(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 1)
        assert_equal(res.t_events[1].size, 0)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_equal(res.y_events[0].shape, (1, 2))
        assert_equal(res.y_events[1].shape, (0,))
        assert np.isclose(
            event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)

        event_rational_1.direction = -1
        event_rational_2.direction = -1
        res = solve_ivp(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 0)
        assert_equal(res.t_events[1].size, 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)
        assert_equal(res.y_events[0].shape, (0,))
        assert_equal(res.y_events[1].shape, (1, 2))
        assert np.isclose(
            event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

        event_rational_1.direction = 0
        event_rational_2.direction = 0

        res = solve_ivp(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                        events=(event_rational_1, event_rational_2,
                                event_rational_3), dense_output=True)
        assert_equal(res.status, 1)
        assert_equal(res.t_events[0].size, 1)
        assert_equal(res.t_events[1].size, 0)
        assert_equal(res.t_events[2].size, 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[2][0] < 7.5)
        assert_equal(res.y_events[0].shape, (1, 2))
        assert_equal(res.y_events[1].shape, (0,))
        assert_equal(res.y_events[2].shape, (1, 2))
        assert np.isclose(
            event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)
        assert np.isclose(
            event_rational_3(res.t_events[2][0], res.y_events[2][0]), 0)

        res = solve_ivp(fun_rational, [5, 8], [1 / 3, 2 / 9], method=method,
                        events=event_rational_1, dense_output=True)
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)

        assert_equal(res.y_events[0].shape, (1, 2))
        assert np.isclose(
            event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)

        # Also test that termination by event doesn't break interpolants.
        tc = np.linspace(res.t[0], res.t[-1])
        yc_true = sol_rational(tc)
        yc = res.sol(tc)
        e = compute_error(yc, yc_true, 1e-3, 1e-6)
        assert_(np.all(e < 5))

        # Test that the y_event matches solution
        assert np.allclose(sol_rational(res.t_events[0][0]), res.y_events[0][0], rtol=1e-3, atol=1e-6)

    # Test in backward direction.
    event_rational_1.direction = 0
    event_rational_2.direction = 0
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        res = solve_ivp(fun_rational, [8, 5], [4/9, 20/81], method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 1)
        assert_equal(res.t_events[1].size, 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        assert_equal(res.y_events[0].shape, (1, 2))
        assert_equal(res.y_events[1].shape, (1, 2))
        assert np.isclose(
            event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)
        assert np.isclose(
            event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

        event_rational_1.direction = -1
        event_rational_2.direction = -1
        res = solve_ivp(fun_rational, [8, 5], [4/9, 20/81], method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 1)
        assert_equal(res.t_events[1].size, 0)
        assert_(5.3 < res.t_events[0][0] < 5.7)

        assert_equal(res.y_events[0].shape, (1, 2))
        assert_equal(res.y_events[1].shape, (0,))
        assert np.isclose(
            event_rational_1(res.t_events[0][0], res.y_events[0][0]), 0)

        event_rational_1.direction = 1
        event_rational_2.direction = 1
        res = solve_ivp(fun_rational, [8, 5], [4/9, 20/81], method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.status, 0)
        assert_equal(res.t_events[0].size, 0)
        assert_equal(res.t_events[1].size, 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        assert_equal(res.y_events[0].shape, (0,))
        assert_equal(res.y_events[1].shape, (1, 2))
        assert np.isclose(
            event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)

        event_rational_1.direction = 0
        event_rational_2.direction = 0

        res = solve_ivp(fun_rational, [8, 5], [4/9, 20/81], method=method,
                        events=(event_rational_1, event_rational_2,
                                event_rational_3), dense_output=True)
        assert_equal(res.status, 1)
        assert_equal(res.t_events[0].size, 0)
        assert_equal(res.t_events[1].size, 1)
        assert_equal(res.t_events[2].size, 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)
        assert_(7.3 < res.t_events[2][0] < 7.5)

        assert_equal(res.y_events[0].shape, (0,))
        assert_equal(res.y_events[1].shape, (1, 2))
        assert_equal(res.y_events[2].shape, (1, 2))
        assert np.isclose(
            event_rational_2(res.t_events[1][0], res.y_events[1][0]), 0)
        assert np.isclose(
            event_rational_3(res.t_events[2][0], res.y_events[2][0]), 0)

        # Also test that termination by event doesn't break interpolants.
        tc = np.linspace(res.t[-1], res.t[0])
        yc_true = sol_rational(tc)
        yc = res.sol(tc)
        e = compute_error(yc, yc_true, 1e-3, 1e-6)
        assert_(np.all(e < 5))

        assert np.allclose(sol_rational(res.t_events[1][0]), res.y_events[1][0], rtol=1e-3, atol=1e-6)
        assert np.allclose(sol_rational(res.t_events[2][0]), res.y_events[2][0], rtol=1e-3, atol=1e-6)


def test_max_step():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        for t_span in ([5, 9], [5, 1]):
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                            max_step=0.5, atol=atol, method=method,
                            dense_output=True)
            assert_equal(res.t[0], t_span[0])
            assert_equal(res.t[-1], t_span[-1])
            assert_(np.all(np.abs(np.diff(res.t)) <= 0.5 + 1e-15))
            assert_(res.t_events is None)
            assert_(res.success)
            assert_equal(res.status, 0)

            y_true = sol_rational(res.t)
            e = compute_error(res.y, y_true, rtol, atol)
            assert_(np.all(e < 5))

            tc = np.linspace(*t_span)
            yc_true = sol_rational(tc)
            yc = res.sol(tc)

            e = compute_error(yc, yc_true, rtol, atol)
            assert_(np.all(e < 5))

            # See comment in test_integration.
            if method is not LSODA:
                assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)

            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
                          t_span[1], max_step=-1)

            if method is not LSODA:
                solver = method(fun_rational, t_span[0], y0, t_span[1],
                                rtol=rtol, atol=atol, max_step=1e-20)
                message = solver.step()

                assert_equal(solver.status, 'failed')
                assert_("step size is less" in message)
                assert_raises(RuntimeError, solver.step)


def test_first_step():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    first_step = 0.1
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        for t_span in ([5, 9], [5, 1]):
            res = solve_ivp(fun_rational, t_span, y0, rtol=rtol,
                            max_step=0.5, atol=atol, method=method,
                            dense_output=True, first_step=first_step)

            assert_equal(res.t[0], t_span[0])
            assert_equal(res.t[-1], t_span[-1])
            assert_allclose(first_step, np.abs(res.t[1] - 5))
            assert_(res.t_events is None)
            assert_(res.success)
            assert_equal(res.status, 0)

            y_true = sol_rational(res.t)
            e = compute_error(res.y, y_true, rtol, atol)
            assert_(np.all(e < 5))

            tc = np.linspace(*t_span)
            yc_true = sol_rational(tc)
            yc = res.sol(tc)

            e = compute_error(yc, yc_true, rtol, atol)
            assert_(np.all(e < 5))

            # See comment in test_integration.
            if method is not LSODA:
                assert_allclose(res.sol(res.t), res.y, rtol=1e-15, atol=1e-15)

            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
                          t_span[1], first_step=-1)
            assert_raises(ValueError, method, fun_rational, t_span[0], y0,
                          t_span[1], first_step=5)


def test_t_eval():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    for t_span in ([5, 9], [5, 1]):
        t_eval = np.linspace(t_span[0], t_span[1], 10)
        res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                        t_eval=t_eval)
        assert_equal(res.t, t_eval)
        assert_(res.t_events is None)
        assert_(res.success)
        assert_equal(res.status, 0)

        y_true = sol_rational(res.t)
        e = compute_error(res.y, y_true, rtol, atol)
        assert_(np.all(e < 5))

    t_eval = [5, 5.01, 7, 8, 8.01, 9]
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))

    t_eval = [5, 4.99, 3, 1.5, 1.1, 1.01, 1]
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    t_eval = [5.01, 7, 8, 8.01]
    res = solve_ivp(fun_rational, [5, 9], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))

    t_eval = [4.99, 3, 1.5, 1.1, 1.01]
    res = solve_ivp(fun_rational, [5, 1], y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    t_eval = [4, 6]
    assert_raises(ValueError, solve_ivp, fun_rational, [5, 9], y0,
                  rtol=rtol, atol=atol, t_eval=t_eval)


def test_t_eval_dense_output():
    rtol = 1e-3
    atol = 1e-6
    y0 = [1/3, 2/9]
    t_span = [5, 9]
    t_eval = np.linspace(t_span[0], t_span[1], 10)
    res = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                    t_eval=t_eval)
    res_d = solve_ivp(fun_rational, t_span, y0, rtol=rtol, atol=atol,
                      t_eval=t_eval, dense_output=True)
    assert_equal(res.t, t_eval)
    assert_(res.t_events is None)
    assert_(res.success)
    assert_equal(res.status, 0)

    assert_equal(res.t, res_d.t)
    assert_equal(res.y, res_d.y)
    assert_(res_d.t_events is None)
    assert_(res_d.success)
    assert_equal(res_d.status, 0)

    # if t and y are equal only test values for one case
    y_true = sol_rational(res.t)
    e = compute_error(res.y, y_true, rtol, atol)
    assert_(np.all(e < 5))


def test_no_integration():
    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        sol = solve_ivp(lambda t, y: -y, [4, 4], [2, 3],
                        method=method, dense_output=True)
        assert_equal(sol.sol(4), [2, 3])
        assert_equal(sol.sol([4, 5, 6]), [[2, 2, 2], [3, 3, 3]])


def test_no_integration_class():
    for method in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        solver = method(lambda t, y: -y, 0.0, [10.0, 0.0], 0.0)
        solver.step()
        assert_equal(solver.status, 'finished')
        sol = solver.dense_output()
        assert_equal(sol(0.0), [10.0, 0.0])
        assert_equal(sol([0, 1, 2]), [[10, 10, 10], [0, 0, 0]])

        solver = method(lambda t, y: -y, 0.0, [], np.inf)
        solver.step()
        assert_equal(solver.status, 'finished')
        sol = solver.dense_output()
        assert_equal(sol(100.0), [])
        assert_equal(sol([0, 1, 2]), np.empty((0, 3)))


def test_empty():
    def fun(t, y):
        return np.zeros((0,))

    y0 = np.zeros((0,))

    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        sol = assert_no_warnings(solve_ivp, fun, [0, 10], y0,
                                 method=method, dense_output=True)
        assert_equal(sol.sol(10), np.zeros((0,)))
        assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))

    for method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        sol = assert_no_warnings(solve_ivp, fun, [0, np.inf], y0,
                                 method=method, dense_output=True)
        assert_equal(sol.sol(10), np.zeros((0,)))
        assert_equal(sol.sol([1, 2, 3]), np.zeros((0, 3)))


def test_ConstantDenseOutput():
    sol = ConstantDenseOutput(0, 1, np.array([1, 2]))
    assert_allclose(sol(1.5), [1, 2])
    assert_allclose(sol([1, 1.5, 2]), [[1, 1, 1], [2, 2, 2]])

    sol = ConstantDenseOutput(0, 1, np.array([]))
    assert_allclose(sol(1.5), np.empty(0))
    assert_allclose(sol([1, 1.5, 2]), np.empty((0, 3)))


def test_classes():
    y0 = [1 / 3, 2 / 9]
    for cls in [RK23, RK45, DOP853, Radau, BDF, LSODA]:
        solver = cls(fun_rational, 5, y0, np.inf)
        assert_equal(solver.n, 2)
        assert_equal(solver.status, 'running')
        assert_equal(solver.t_bound, np.inf)
        assert_equal(solver.direction, 1)
        assert_equal(solver.t, 5)
        assert_equal(solver.y, y0)
        assert_(solver.step_size is None)
        if cls is not LSODA:
            assert_(solver.nfev > 0)
            assert_(solver.njev >= 0)
            assert_equal(solver.nlu, 0)
        else:
            assert_equal(solver.nfev, 0)
            assert_equal(solver.njev, 0)
            assert_equal(solver.nlu, 0)

        assert_raises(RuntimeError, solver.dense_output)

        message = solver.step()
        assert_equal(solver.status, 'running')
        assert_equal(message, None)
        assert_equal(solver.n, 2)
        assert_equal(solver.t_bound, np.inf)
        assert_equal(solver.direction, 1)
        assert_(solver.t > 5)
        assert_(not np.all(np.equal(solver.y, y0)))
        assert_(solver.step_size > 0)
        assert_(solver.nfev > 0)
        assert_(solver.njev >= 0)
        assert_(solver.nlu >= 0)
        sol = solver.dense_output()
        assert_allclose(sol(5), y0, rtol=1e-15, atol=0)


def test_OdeSolution():
    ts = np.array([0, 2, 5], dtype=float)
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))

    sol = OdeSolution(ts, [s1, s2])

    assert_equal(sol(-1), [-1])
    assert_equal(sol(1), [-1])
    assert_equal(sol(2), [-1])
    assert_equal(sol(3), [1])
    assert_equal(sol(5), [1])
    assert_equal(sol(6), [1])

    assert_equal(sol([0, 6, -2, 1.5, 4.5, 2.5, 5, 5.5, 2]),
                 np.array([[-1, 1, -1, -1, 1, 1, 1, 1, -1]]))

    ts = np.array([10, 4, -3])
    s1 = ConstantDenseOutput(ts[0], ts[1], np.array([-1]))
    s2 = ConstantDenseOutput(ts[1], ts[2], np.array([1]))

    sol = OdeSolution(ts, [s1, s2])
    assert_equal(sol(11), [-1])
    assert_equal(sol(10), [-1])
    assert_equal(sol(5), [-1])
    assert_equal(sol(4), [-1])
    assert_equal(sol(0), [1])
    assert_equal(sol(-3), [1])
    assert_equal(sol(-4), [1])

    assert_equal(sol([12, -5, 10, -3, 6, 1, 4]),
                 np.array([[-1, 1, -1, 1, -1, 1, -1]]))

    ts = np.array([1, 1])
    s = ConstantDenseOutput(1, 1, np.array([10]))
    sol = OdeSolution(ts, [s])
    assert_equal(sol(0), [10])
    assert_equal(sol(1), [10])
    assert_equal(sol(2), [10])

    assert_equal(sol([2, 1, 0]), np.array([[10, 10, 10]]))


def test_num_jac():
    def fun(t, y):
        return np.vstack([
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2
        ])

    def jac(t, y):
        return np.array([
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0, 6e7 * y[1], 0]
        ])

    t = 1
    y = np.array([1, 0, 0])
    J_true = jac(t, y)
    threshold = 1e-5
    f = fun(t, y).ravel()

    J_num, factor = num_jac(fun, t, y, f, threshold, None)
    assert_allclose(J_num, J_true, rtol=1e-5, atol=1e-5)

    J_num, factor = num_jac(fun, t, y, f, threshold, factor)
    assert_allclose(J_num, J_true, rtol=1e-5, atol=1e-5)


def test_num_jac_sparse():
    def fun(t, y):
        e = y[1:]**3 - y[:-1]**2
        z = np.zeros(y.shape[1])
        return np.vstack((z, 3 * e)) + np.vstack((2 * e, z))

    def structure(n):
        A = np.zeros((n, n), dtype=int)
        A[0, 0] = 1
        A[0, 1] = 1
        for i in range(1, n - 1):
            A[i, i - 1: i + 2] = 1
        A[-1, -1] = 1
        A[-1, -2] = 1

        return A

    np.random.seed(0)
    n = 20
    y = np.random.randn(n)
    A = structure(n)
    groups = group_columns(A)

    f = fun(0, y[:, None]).ravel()

    # Compare dense and sparse results, assuming that dense implementation
    # is correct (as it is straightforward).
    J_num_sparse, factor_sparse = num_jac(fun, 0, y.ravel(), f, 1e-8, None,
                                          sparsity=(A, groups))
    J_num_dense, factor_dense = num_jac(fun, 0, y.ravel(), f, 1e-8, None)
    assert_allclose(J_num_dense, J_num_sparse.toarray(),
                    rtol=1e-12, atol=1e-14)
    assert_allclose(factor_dense, factor_sparse, rtol=1e-12, atol=1e-14)

    # Take small factors to trigger their recomputing inside.
    factor = np.random.uniform(0, 1e-12, size=n)
    J_num_sparse, factor_sparse = num_jac(fun, 0, y.ravel(), f, 1e-8, factor,
                                          sparsity=(A, groups))
    J_num_dense, factor_dense = num_jac(fun, 0, y.ravel(), f, 1e-8, factor)

    assert_allclose(J_num_dense, J_num_sparse.toarray(),
                    rtol=1e-12, atol=1e-14)
    assert_allclose(factor_dense, factor_sparse, rtol=1e-12, atol=1e-14)


def test_args():

    # sys3 is actually two decoupled systems. (x, y) form a
    # linear oscillator, while z is a nonlinear first order
    # system with equilibria at z=0 and z=1. If k > 0, z=1
    # is stable and z=0 is unstable.

    def sys3(t, w, omega, k, zfinal):
        x, y, z = w
        return [-omega*y, omega*x, k*z*(1 - z)]

    def sys3_jac(t, w, omega, k, zfinal):
        x, y, z = w
        J = np.array([[0, -omega, 0],
                      [omega, 0, 0],
                      [0, 0, k*(1 - 2*z)]])
        return J

    def sys3_x0decreasing(t, w, omega, k, zfinal):
        x, y, z = w
        return x

    def sys3_y0increasing(t, w, omega, k, zfinal):
        x, y, z = w
        return y

    def sys3_zfinal(t, w, omega, k, zfinal):
        x, y, z = w
        return z - zfinal

    # Set the event flags for the event functions.
    sys3_x0decreasing.direction = -1
    sys3_y0increasing.direction = 1
    sys3_zfinal.terminal = True

    omega = 2
    k = 4

    tfinal = 5
    zfinal = 0.99
    # Find z0 such that when z(0) = z0, z(tfinal) = zfinal.
    # The condition z(tfinal) = zfinal is the terminal event.
    z0 = np.exp(-k*tfinal)/((1 - zfinal)/zfinal + np.exp(-k*tfinal))

    w0 = [0, -1, z0]

    # Provide the jac argument and use the Radau method to ensure that the use
    # of the Jacobian function is exercised.
    # If event handling is working, the solution will stop at tfinal, not tend.
    tend = 2*tfinal
    sol = solve_ivp(sys3, [0, tend], w0,
                    events=[sys3_x0decreasing, sys3_y0increasing, sys3_zfinal],
                    dense_output=True, args=(omega, k, zfinal),
                    method='Radau', jac=sys3_jac,
                    rtol=1e-10, atol=1e-13)

    # Check that we got the expected events at the expected times.
    x0events_t = sol.t_events[0]
    y0events_t = sol.t_events[1]
    zfinalevents_t = sol.t_events[2]
    assert_allclose(x0events_t, [0.5*np.pi, 1.5*np.pi])
    assert_allclose(y0events_t, [0.25*np.pi, 1.25*np.pi])
    assert_allclose(zfinalevents_t, [tfinal])

    # Check that the solution agrees with the known exact solution.
    t = np.linspace(0, zfinalevents_t[0], 250)
    w = sol.sol(t)
    assert_allclose(w[0], np.sin(omega*t), rtol=1e-9, atol=1e-12)
    assert_allclose(w[1], -np.cos(omega*t), rtol=1e-9, atol=1e-12)
    assert_allclose(w[2], 1/(((1 - z0)/z0)*np.exp(-k*t) + 1),
                    rtol=1e-9, atol=1e-12)

    # Check that the state variables have the expected values at the events.
    x0events = sol.sol(x0events_t)
    y0events = sol.sol(y0events_t)
    zfinalevents = sol.sol(zfinalevents_t)
    assert_allclose(x0events[0], np.zeros_like(x0events[0]), atol=5e-14)
    assert_allclose(x0events[1], np.ones_like(x0events[1]))
    assert_allclose(y0events[0], np.ones_like(y0events[0]))
    assert_allclose(y0events[1], np.zeros_like(y0events[1]), atol=5e-14)
    assert_allclose(zfinalevents[2], [zfinal])


@pytest.mark.parametrize('method', ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'])
def test_integration_zero_rhs(method):
    result = solve_ivp(fun_zero, [0, 10], np.ones(3), method=method)
    assert_(result.success)
    assert_equal(result.status, 0)
    assert_allclose(result.y, 1.0, rtol=1e-15)
