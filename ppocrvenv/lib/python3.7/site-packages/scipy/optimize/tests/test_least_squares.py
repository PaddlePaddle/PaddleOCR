from itertools import product

import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose,
                           assert_equal, suppress_warnings)
from pytest import raises as assert_raises
from scipy.sparse import issparse, lil_matrix
from scipy.sparse.linalg import aslinearoperator

from scipy.optimize import least_squares
from scipy.optimize._lsq.least_squares import IMPLEMENTED_LOSSES
from scipy.optimize._lsq.common import EPS, make_strictly_feasible


def fun_trivial(x, a=0):
    return (x - a)**2 + 5.0


def jac_trivial(x, a=0.0):
    return 2 * (x - a)


def fun_2d_trivial(x):
    return np.array([x[0], x[1]])


def jac_2d_trivial(x):
    return np.identity(2)


def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])


def jac_rosenbrock(x):
    return np.array([
        [-20 * x[0], 10],
        [-1, 0]
    ])


def jac_rosenbrock_bad_dim(x):
    return np.array([
        [-20 * x[0], 10],
        [-1, 0],
        [0.0, 0.0]
    ])


def fun_rosenbrock_cropped(x):
    return fun_rosenbrock(x)[0]


def jac_rosenbrock_cropped(x):
    return jac_rosenbrock(x)[0]


# When x is 1-D array, return is 2-D array.
def fun_wrong_dimensions(x):
    return np.array([x, x**2, x**3])


def jac_wrong_dimensions(x, a=0.0):
    return np.atleast_3d(jac_trivial(x, a=a))


def fun_bvp(x):
    n = int(np.sqrt(x.shape[0]))
    u = np.zeros((n + 2, n + 2))
    x = x.reshape((n, n))
    u[1:-1, 1:-1] = x
    y = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * x + x**3
    return y.ravel()


class BroydenTridiagonal:
    def __init__(self, n=100, mode='sparse'):
        np.random.seed(0)

        self.n = n

        self.x0 = -np.ones(n)
        self.lb = np.linspace(-2, -1.5, n)
        self.ub = np.linspace(-0.8, 0.0, n)

        self.lb += 0.1 * np.random.randn(n)
        self.ub += 0.1 * np.random.randn(n)

        self.x0 += 0.1 * np.random.randn(n)
        self.x0 = make_strictly_feasible(self.x0, self.lb, self.ub)

        if mode == 'sparse':
            self.sparsity = lil_matrix((n, n), dtype=int)
            i = np.arange(n)
            self.sparsity[i, i] = 1
            i = np.arange(1, n)
            self.sparsity[i, i - 1] = 1
            i = np.arange(n - 1)
            self.sparsity[i, i + 1] = 1

            self.jac = self._jac
        elif mode == 'operator':
            self.jac = lambda x: aslinearoperator(self._jac(x))
        elif mode == 'dense':
            self.sparsity = None
            self.jac = lambda x: self._jac(x).toarray()
        else:
            assert_(False)

    def fun(self, x):
        f = (3 - x) * x + 1
        f[1:] -= x[:-1]
        f[:-1] -= 2 * x[1:]
        return f

    def _jac(self, x):
        J = lil_matrix((self.n, self.n))
        i = np.arange(self.n)
        J[i, i] = 3 - 2 * x
        i = np.arange(1, self.n)
        J[i, i - 1] = -1
        i = np.arange(self.n - 1)
        J[i, i + 1] = -2
        return J


class ExponentialFittingProblem:
    """Provide data and function for exponential fitting in the form
    y = a + exp(b * x) + noise."""

    def __init__(self, a, b, noise, n_outliers=1, x_range=(-1, 1),
                 n_points=11, random_seed=None):
        np.random.seed(random_seed)
        self.m = n_points
        self.n = 2

        self.p0 = np.zeros(2)
        self.x = np.linspace(x_range[0], x_range[1], n_points)

        self.y = a + np.exp(b * self.x)
        self.y += noise * np.random.randn(self.m)

        outliers = np.random.randint(0, self.m, n_outliers)
        self.y[outliers] += 50 * noise * np.random.rand(n_outliers)

        self.p_opt = np.array([a, b])

    def fun(self, p):
        return p[0] + np.exp(p[1] * self.x) - self.y

    def jac(self, p):
        J = np.empty((self.m, self.n))
        J[:, 0] = 1
        J[:, 1] = self.x * np.exp(p[1] * self.x)
        return J


def cubic_soft_l1(z):
    rho = np.empty((3, z.size))

    t = 1 + z
    rho[0] = 3 * (t**(1/3) - 1)
    rho[1] = t ** (-2/3)
    rho[2] = -2/3 * t**(-5/3)

    return rho


LOSSES = list(IMPLEMENTED_LOSSES.keys()) + [cubic_soft_l1]


class BaseMixin:
    def test_basic(self):
        # Test that the basic calling sequence works.
        res = least_squares(fun_trivial, 2., method=self.method)
        assert_allclose(res.x, 0, atol=1e-4)
        assert_allclose(res.fun, fun_trivial(res.x))

    def test_args_kwargs(self):
        # Test that args and kwargs are passed correctly to the functions.
        a = 3.0
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                           "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
                res = least_squares(fun_trivial, 2.0, jac, args=(a,),
                                    method=self.method)
                res1 = least_squares(fun_trivial, 2.0, jac, kwargs={'a': a},
                                    method=self.method)

            assert_allclose(res.x, a, rtol=1e-4)
            assert_allclose(res1.x, a, rtol=1e-4)

            assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                          args=(3, 4,), method=self.method)
            assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                          kwargs={'kaboom': 3}, method=self.method)

    def test_jac_options(self):
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                           "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
                res = least_squares(fun_trivial, 2.0, jac, method=self.method)
            assert_allclose(res.x, 0, atol=1e-4)

        assert_raises(ValueError, least_squares, fun_trivial, 2.0, jac='oops',
                      method=self.method)

    def test_nfev_options(self):
        for max_nfev in [None, 20]:
            res = least_squares(fun_trivial, 2.0, max_nfev=max_nfev,
                                method=self.method)
            assert_allclose(res.x, 0, atol=1e-4)

    def test_x_scale_options(self):
        for x_scale in [1.0, np.array([0.5]), 'jac']:
            res = least_squares(fun_trivial, 2.0, x_scale=x_scale)
            assert_allclose(res.x, 0)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale='auto', method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale=-1.0, method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale=None, method=self.method)
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, x_scale=1.0+2.0j, method=self.method)

    def test_diff_step(self):
        # res1 and res2 should be equivalent.
        # res2 and res3 should be different.
        res1 = least_squares(fun_trivial, 2.0, diff_step=1e-1,
                             method=self.method)
        res2 = least_squares(fun_trivial, 2.0, diff_step=-1e-1,
                             method=self.method)
        res3 = least_squares(fun_trivial, 2.0,
                             diff_step=None, method=self.method)
        assert_allclose(res1.x, 0, atol=1e-4)
        assert_allclose(res2.x, 0, atol=1e-4)
        assert_allclose(res3.x, 0, atol=1e-4)
        assert_equal(res1.x, res2.x)
        assert_equal(res1.nfev, res2.nfev)
        assert_(res2.nfev != res3.nfev)

    def test_incorrect_options_usage(self):
        assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                      method=self.method, options={'no_such_option': 100})
        assert_raises(TypeError, least_squares, fun_trivial, 2.0,
                      method=self.method, options={'max_nfev': 100})

    def test_full_result(self):
        # MINPACK doesn't work very well with factor=100 on this problem,
        # thus using low 'atol'.
        res = least_squares(fun_trivial, 2.0, method=self.method)
        assert_allclose(res.x, 0, atol=1e-4)
        assert_allclose(res.cost, 12.5)
        assert_allclose(res.fun, 5)
        assert_allclose(res.jac, 0, atol=1e-4)
        assert_allclose(res.grad, 0, atol=1e-2)
        assert_allclose(res.optimality, 0, atol=1e-2)
        assert_equal(res.active_mask, 0)
        if self.method == 'lm':
            assert_(res.nfev < 30)
            assert_(res.njev is None)
        else:
            assert_(res.nfev < 10)
            assert_(res.njev < 10)
        assert_(res.status > 0)
        assert_(res.success)

    def test_full_result_single_fev(self):
        # MINPACK checks the number of nfev after the iteration,
        # so it's hard to tell what he is going to compute.
        if self.method == 'lm':
            return

        res = least_squares(fun_trivial, 2.0, method=self.method,
                            max_nfev=1)
        assert_equal(res.x, np.array([2]))
        assert_equal(res.cost, 40.5)
        assert_equal(res.fun, np.array([9]))
        assert_equal(res.jac, np.array([[4]]))
        assert_equal(res.grad, np.array([36]))
        assert_equal(res.optimality, 36)
        assert_equal(res.active_mask, np.array([0]))
        assert_equal(res.nfev, 1)
        assert_equal(res.njev, 1)
        assert_equal(res.status, 0)
        assert_equal(res.success, 0)

    def test_rosenbrock(self):
        x0 = [-2, 1]
        x_opt = [1, 1]
        for jac, x_scale, tr_solver in product(
                ['2-point', '3-point', 'cs', jac_rosenbrock],
                [1.0, np.array([1.0, 0.2]), 'jac'],
                ['exact', 'lsmr']):
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                           "jac='(3-point|cs)' works equivalently to '2-point' for method='lm'")
                res = least_squares(fun_rosenbrock, x0, jac, x_scale=x_scale,
                                    tr_solver=tr_solver, method=self.method)
            assert_allclose(res.x, x_opt)

    def test_rosenbrock_cropped(self):
        x0 = [-2, 1]
        if self.method == 'lm':
            assert_raises(ValueError, least_squares, fun_rosenbrock_cropped,
                          x0, method='lm')
        else:
            for jac, x_scale, tr_solver in product(
                    ['2-point', '3-point', 'cs', jac_rosenbrock_cropped],
                    [1.0, np.array([1.0, 0.2]), 'jac'],
                    ['exact', 'lsmr']):
                res = least_squares(
                    fun_rosenbrock_cropped, x0, jac, x_scale=x_scale,
                    tr_solver=tr_solver, method=self.method)
                assert_allclose(res.cost, 0, atol=1e-14)

    def test_fun_wrong_dimensions(self):
        assert_raises(ValueError, least_squares, fun_wrong_dimensions,
                      2.0, method=self.method)

    def test_jac_wrong_dimensions(self):
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, jac_wrong_dimensions, method=self.method)

    def test_fun_and_jac_inconsistent_dimensions(self):
        x0 = [1, 2]
        assert_raises(ValueError, least_squares, fun_rosenbrock, x0,
                      jac_rosenbrock_bad_dim, method=self.method)

    def test_x0_multidimensional(self):
        x0 = np.ones(4).reshape(2, 2)
        assert_raises(ValueError, least_squares, fun_trivial, x0,
                      method=self.method)

    def test_x0_complex_scalar(self):
        x0 = 2.0 + 0.0*1j
        assert_raises(ValueError, least_squares, fun_trivial, x0,
                      method=self.method)

    def test_x0_complex_array(self):
        x0 = [1.0, 2.0 + 0.0*1j]
        assert_raises(ValueError, least_squares, fun_trivial, x0,
                      method=self.method)

    def test_bvp(self):
        # This test was introduced with fix #5556. It turned out that
        # dogbox solver had a bug with trust-region radius update, which
        # could block its progress and create an infinite loop. And this
        # discrete boundary value problem is the one which triggers it.
        n = 10
        x0 = np.ones(n**2)
        if self.method == 'lm':
            max_nfev = 5000  # To account for Jacobian estimation.
        else:
            max_nfev = 100
        res = least_squares(fun_bvp, x0, ftol=1e-2, method=self.method,
                            max_nfev=max_nfev)

        assert_(res.nfev < max_nfev)
        assert_(res.cost < 0.5)

    def test_error_raised_when_all_tolerances_below_eps(self):
        # Test that all 0 tolerances are not allowed.
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      method=self.method, ftol=None, xtol=None, gtol=None)

    def test_convergence_with_only_one_tolerance_enabled(self):
        if self.method == 'lm':
            return  # should not do test
        x0 = [-2, 1]
        x_opt = [1, 1]
        for ftol, xtol, gtol in [(1e-8, None, None),
                                  (None, 1e-8, None),
                                  (None, None, 1e-8)]:
            res = least_squares(fun_rosenbrock, x0, jac=jac_rosenbrock,
                                ftol=ftol, gtol=gtol, xtol=xtol,
                                method=self.method)
            assert_allclose(res.x, x_opt)


class BoundsMixin:
    def test_inconsistent(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      bounds=(10.0, 0.0), method=self.method)

    def test_infeasible(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      bounds=(3., 4), method=self.method)

    def test_wrong_number(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.,
                      bounds=(1., 2, 3), method=self.method)

    def test_inconsistent_shape(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      bounds=(1.0, [2.0, 3.0]), method=self.method)
        # 1-D array wont't be broadcasted
        assert_raises(ValueError, least_squares, fun_rosenbrock, [1.0, 2.0],
                      bounds=([0.0], [3.0, 4.0]), method=self.method)

    def test_in_bounds(self):
        for jac in ['2-point', '3-point', 'cs', jac_trivial]:
            res = least_squares(fun_trivial, 2.0, jac=jac,
                                bounds=(-1.0, 3.0), method=self.method)
            assert_allclose(res.x, 0.0, atol=1e-4)
            assert_equal(res.active_mask, [0])
            assert_(-1 <= res.x <= 3)
            res = least_squares(fun_trivial, 2.0, jac=jac,
                                bounds=(0.5, 3.0), method=self.method)
            assert_allclose(res.x, 0.5, atol=1e-4)
            assert_equal(res.active_mask, [-1])
            assert_(0.5 <= res.x <= 3)

    def test_bounds_shape(self):
        for jac in ['2-point', '3-point', 'cs', jac_2d_trivial]:
            x0 = [1.0, 1.0]
            res = least_squares(fun_2d_trivial, x0, jac=jac)
            assert_allclose(res.x, [0.0, 0.0])
            res = least_squares(fun_2d_trivial, x0, jac=jac,
                                bounds=(0.5, [2.0, 2.0]), method=self.method)
            assert_allclose(res.x, [0.5, 0.5])
            res = least_squares(fun_2d_trivial, x0, jac=jac,
                                bounds=([0.3, 0.2], 3.0), method=self.method)
            assert_allclose(res.x, [0.3, 0.2])
            res = least_squares(
                fun_2d_trivial, x0, jac=jac, bounds=([-1, 0.5], [1.0, 3.0]),
                method=self.method)
            assert_allclose(res.x, [0.0, 0.5], atol=1e-5)

    def test_rosenbrock_bounds(self):
        x0_1 = np.array([-2.0, 1.0])
        x0_2 = np.array([2.0, 2.0])
        x0_3 = np.array([-2.0, 2.0])
        x0_4 = np.array([0.0, 2.0])
        x0_5 = np.array([-1.2, 1.0])
        problems = [
            (x0_1, ([-np.inf, -1.5], np.inf)),
            (x0_2, ([-np.inf, 1.5], np.inf)),
            (x0_3, ([-np.inf, 1.5], np.inf)),
            (x0_4, ([-np.inf, 1.5], [1.0, np.inf])),
            (x0_2, ([1.0, 1.5], [3.0, 3.0])),
            (x0_5, ([-50.0, 0.0], [0.5, 100]))
        ]
        for x0, bounds in problems:
            for jac, x_scale, tr_solver in product(
                    ['2-point', '3-point', 'cs', jac_rosenbrock],
                    [1.0, [1.0, 0.5], 'jac'],
                    ['exact', 'lsmr']):
                res = least_squares(fun_rosenbrock, x0, jac, bounds,
                                    x_scale=x_scale, tr_solver=tr_solver,
                                    method=self.method)
                assert_allclose(res.optimality, 0.0, atol=1e-5)


class SparseMixin:
    def test_exact_tr_solver(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      tr_solver='exact', method=self.method)
        assert_raises(ValueError, least_squares, p.fun, p.x0,
                      tr_solver='exact', jac_sparsity=p.sparsity,
                      method=self.method)

    def test_equivalence(self):
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        res_sparse = least_squares(
            sparse.fun, sparse.x0, jac=sparse.jac,
            method=self.method)
        res_dense = least_squares(
            dense.fun, dense.x0, jac=sparse.jac,
            method=self.method)
        assert_equal(res_sparse.nfev, res_dense.nfev)
        assert_allclose(res_sparse.x, res_dense.x, atol=1e-20)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)

    def test_tr_options(self):
        p = BroydenTridiagonal()
        res = least_squares(p.fun, p.x0, p.jac, method=self.method,
                            tr_options={'btol': 1e-10})
        assert_allclose(res.cost, 0, atol=1e-20)

    def test_wrong_parameters(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      tr_solver='best', method=self.method)
        assert_raises(TypeError, least_squares, p.fun, p.x0, p.jac,
                      tr_solver='lsmr', tr_options={'tol': 1e-10})

    def test_solver_selection(self):
        sparse = BroydenTridiagonal(mode='sparse')
        dense = BroydenTridiagonal(mode='dense')
        res_sparse = least_squares(sparse.fun, sparse.x0, jac=sparse.jac,
                                   method=self.method)
        res_dense = least_squares(dense.fun, dense.x0, jac=dense.jac,
                                  method=self.method)
        assert_allclose(res_sparse.cost, 0, atol=1e-20)
        assert_allclose(res_dense.cost, 0, atol=1e-20)
        assert_(issparse(res_sparse.jac))
        assert_(isinstance(res_dense.jac, np.ndarray))

    def test_numerical_jac(self):
        p = BroydenTridiagonal()
        for jac in ['2-point', '3-point', 'cs']:
            res_dense = least_squares(p.fun, p.x0, jac, method=self.method)
            res_sparse = least_squares(
                p.fun, p.x0, jac,method=self.method,
                jac_sparsity=p.sparsity)
            assert_equal(res_dense.nfev, res_sparse.nfev)
            assert_allclose(res_dense.x, res_sparse.x, atol=1e-20)
            assert_allclose(res_dense.cost, 0, atol=1e-20)
            assert_allclose(res_sparse.cost, 0, atol=1e-20)

    def test_with_bounds(self):
        p = BroydenTridiagonal()
        for jac, jac_sparsity in product(
                [p.jac, '2-point', '3-point', 'cs'], [None, p.sparsity]):
            res_1 = least_squares(
                p.fun, p.x0, jac, bounds=(p.lb, np.inf),
                method=self.method,jac_sparsity=jac_sparsity)
            res_2 = least_squares(
                p.fun, p.x0, jac, bounds=(-np.inf, p.ub),
                method=self.method, jac_sparsity=jac_sparsity)
            res_3 = least_squares(
                p.fun, p.x0, jac, bounds=(p.lb, p.ub),
                method=self.method, jac_sparsity=jac_sparsity)
            assert_allclose(res_1.optimality, 0, atol=1e-10)
            assert_allclose(res_2.optimality, 0, atol=1e-10)
            assert_allclose(res_3.optimality, 0, atol=1e-10)

    def test_wrong_jac_sparsity(self):
        p = BroydenTridiagonal()
        sparsity = p.sparsity[:-1]
        assert_raises(ValueError, least_squares, p.fun, p.x0,
                      jac_sparsity=sparsity, method=self.method)

    def test_linear_operator(self):
        p = BroydenTridiagonal(mode='operator')
        res = least_squares(p.fun, p.x0, p.jac, method=self.method)
        assert_allclose(res.cost, 0.0, atol=1e-20)
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method=self.method, tr_solver='exact')

    def test_x_scale_jac_scale(self):
        p = BroydenTridiagonal()
        res = least_squares(p.fun, p.x0, p.jac, method=self.method,
                            x_scale='jac')
        assert_allclose(res.cost, 0.0, atol=1e-20)

        p = BroydenTridiagonal(mode='operator')
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method=self.method, x_scale='jac')


class LossFunctionMixin:
    def test_options(self):
        for loss in LOSSES:
            res = least_squares(fun_trivial, 2.0, loss=loss,
                                method=self.method)
            assert_allclose(res.x, 0, atol=1e-15)

        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      loss='hinge', method=self.method)

    def test_fun(self):
        # Test that res.fun is actual residuals, and not modified by loss
        # function stuff.
        for loss in LOSSES:
            res = least_squares(fun_trivial, 2.0, loss=loss,
                                method=self.method)
            assert_equal(res.fun, fun_trivial(res.x))

    def test_grad(self):
        # Test that res.grad is true gradient of loss function at the
        # solution. Use max_nfev = 1, to avoid reaching minimum.
        x = np.array([2.0])  # res.x will be this.

        res = least_squares(fun_trivial, x, jac_trivial, loss='linear',
                            max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x * (x**2 + 5))

        res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
                            max_nfev=1, method=self.method)
        assert_equal(res.grad, 2 * x)

        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1',
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad,
                        2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2)**0.5)

        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2))

        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad, 2 * x * (x**2 + 5) / (1 + (x**2 + 5)**4))

        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1,
                            max_nfev=1, method=self.method)
        assert_allclose(res.grad,
                        2 * x * (x**2 + 5) / (1 + (x**2 + 5)**2)**(2/3))

    def test_jac(self):
        # Test that res.jac.T.dot(res.jac) gives Gauss-Newton approximation
        # of Hessian. This approximation is computed by doubly differentiating
        # the cost function and dropping the part containing second derivative
        # of f. For a scalar function it is computed as
        # H = (rho' + 2 * rho'' * f**2) * f'**2, if the expression inside the
        # brackets is less than EPS it is replaced by EPS. Here, we check
        # against the root of H.

        x = 2.0  # res.x will be this.
        f = x**2 + 5  # res.fun will be this.

        res = least_squares(fun_trivial, x, jac_trivial, loss='linear',
                            max_nfev=1, method=self.method)
        assert_equal(res.jac, 2 * x)

        # For `huber` loss the Jacobian correction is identically zero
        # in outlier region, in such cases it is modified to be equal EPS**0.5.
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
                            max_nfev=1, method=self.method)
        assert_equal(res.jac, 2 * x * EPS**0.5)

        # Now, let's apply `loss_scale` to turn the residual into an inlier.
        # The loss function becomes linear.
        res = least_squares(fun_trivial, x, jac_trivial, loss='huber',
                            f_scale=10, max_nfev=1)
        assert_equal(res.jac, 2 * x)

        # 'soft_l1' always gives a positive scaling.
        res = least_squares(fun_trivial, x, jac_trivial, loss='soft_l1',
                            max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * (1 + f**2)**-0.75)

        # For 'cauchy' the correction term turns out to be negative, and it
        # replaced by EPS**0.5.
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
                            max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * EPS**0.5)

        # Now use scaling to turn the residual to inlier.
        res = least_squares(fun_trivial, x, jac_trivial, loss='cauchy',
                            f_scale=10, max_nfev=1, method=self.method)
        fs = f / 10
        assert_allclose(res.jac, 2 * x * (1 - fs**2)**0.5 / (1 + fs**2))

        # 'arctan' gives an outlier.
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
                            max_nfev=1, method=self.method)
        assert_allclose(res.jac, 2 * x * EPS**0.5)

        # Turn to inlier.
        res = least_squares(fun_trivial, x, jac_trivial, loss='arctan',
                            f_scale=20.0, max_nfev=1, method=self.method)
        fs = f / 20
        assert_allclose(res.jac, 2 * x * (1 - 3 * fs**4)**0.5 / (1 + fs**4))

        # cubic_soft_l1 will give an outlier.
        res = least_squares(fun_trivial, x, jac_trivial, loss=cubic_soft_l1,
                            max_nfev=1)
        assert_allclose(res.jac, 2 * x * EPS**0.5)

        # Turn to inlier.
        res = least_squares(fun_trivial, x, jac_trivial,
                            loss=cubic_soft_l1, f_scale=6, max_nfev=1)
        fs = f / 6
        assert_allclose(res.jac,
                        2 * x * (1 - fs**2 / 3)**0.5 * (1 + fs**2)**(-5/6))

    def test_robustness(self):
        for noise in [0.1, 1.0]:
            p = ExponentialFittingProblem(1, 0.1, noise, random_seed=0)

            for jac in ['2-point', '3-point', 'cs', p.jac]:
                res_lsq = least_squares(p.fun, p.p0, jac=jac,
                                        method=self.method)
                assert_allclose(res_lsq.optimality, 0, atol=1e-2)
                for loss in LOSSES:
                    if loss == 'linear':
                        continue
                    res_robust = least_squares(
                        p.fun, p.p0, jac=jac, loss=loss, f_scale=noise,
                        method=self.method)
                    assert_allclose(res_robust.optimality, 0, atol=1e-2)
                    assert_(norm(res_robust.x - p.p_opt) <
                            norm(res_lsq.x - p.p_opt))


class TestDogbox(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
    method = 'dogbox'


class TestTRF(BaseMixin, BoundsMixin, SparseMixin, LossFunctionMixin):
    method = 'trf'

    def test_lsmr_regularization(self):
        p = BroydenTridiagonal()
        for regularize in [True, False]:
            res = least_squares(p.fun, p.x0, p.jac, method='trf',
                                tr_options={'regularize': regularize})
            assert_allclose(res.cost, 0, atol=1e-20)


class TestLM(BaseMixin):
    method = 'lm'

    def test_bounds_not_supported(self):
        assert_raises(ValueError, least_squares, fun_trivial,
                      2.0, bounds=(-3.0, 3.0), method='lm')

    def test_m_less_n_not_supported(self):
        x0 = [-2, 1]
        assert_raises(ValueError, least_squares, fun_rosenbrock_cropped, x0,
                      method='lm')

    def test_sparse_not_supported(self):
        p = BroydenTridiagonal()
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method='lm')

    def test_jac_sparsity_not_supported(self):
        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      jac_sparsity=[1], method='lm')

    def test_LinearOperator_not_supported(self):
        p = BroydenTridiagonal(mode="operator")
        assert_raises(ValueError, least_squares, p.fun, p.x0, p.jac,
                      method='lm')

    def test_loss(self):
        res = least_squares(fun_trivial, 2.0, loss='linear', method='lm')
        assert_allclose(res.x, 0.0, atol=1e-4)

        assert_raises(ValueError, least_squares, fun_trivial, 2.0,
                      method='lm', loss='huber')


def test_basic():
    # test that 'method' arg is really optional
    res = least_squares(fun_trivial, 2.0)
    assert_allclose(res.x, 0, atol=1e-10)


def test_small_tolerances_for_lm():
    for ftol, xtol, gtol in [(None, 1e-13, 1e-13),
                             (1e-13, None, 1e-13),
                             (1e-13, 1e-13, None)]:
        assert_raises(ValueError, least_squares, fun_trivial, 2.0, xtol=xtol,
                      ftol=ftol, gtol=gtol, method='lm')


def test_fp32_gh12991():
    # checks that smaller FP sizes can be used in least_squares
    # this is the minimum working example reported for gh12991
    np.random.seed(1)

    x = np.linspace(0, 1, 100).astype("float32")
    y = np.random.random(100).astype("float32")

    def func(p, x):
        return p[0] + p[1] * x

    def err(p, x, y):
        return func(p, x) - y

    res = least_squares(err, [-1.0, -1.0], args=(x, y))
    # previously the initial jacobian calculated for this would be all 0
    # and the minimize would terminate immediately, with nfev=1, would
    # report a successful minimization (it shouldn't have done), but be
    # unchanged from the initial solution.
    # It was terminating early because the underlying approx_derivative
    # used a step size for FP64 when the working space was FP32.
    assert res.nfev > 3
    assert_allclose(res.x, np.array([0.4082241, 0.15530563]), atol=5e-5)
