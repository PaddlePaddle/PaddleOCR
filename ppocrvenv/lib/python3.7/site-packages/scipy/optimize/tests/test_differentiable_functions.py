import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_, assert_allclose,
                           assert_equal)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
                                                      VectorFunction,
                                                      LinearVectorFunction,
                                                      IdentityVectorFunction)
from scipy.optimize._hessian_update_strategy import BFGS


class ExScalarFunction:

    def __init__(self):
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0

    def fun(self, x):
        self.nfev += 1
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    def grad(self, x):
        self.ngev += 1
        return np.array([4*x[0]-1, 4*x[1]])

    def hess(self, x):
        self.nhev += 1
        return 4*np.eye(2)


class TestScalarFunction(TestCase):

    def test_finite_difference_grad(self):
        ex = ExScalarFunction()
        nfev = 0
        ngev = 0

        x0 = [1.0, 0.0]
        analit = ScalarFunction(ex.fun, x0, (), ex.grad,
                                ex.hess, None, (-np.inf, np.inf))
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev, nfev)
        approx = ScalarFunction(ex.fun, x0, (), '2-point',
                                ex.hess, None, (-np.inf, np.inf))
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(analit.f, approx.f)
        assert_array_almost_equal(analit.g, approx.g)

        x = [10, 0.3]
        f_analit = analit.fun(x)
        g_analit = analit.grad(x)
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        f_approx = approx.fun(x)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(g_analit, g_approx)

        x = [2.0, 1.0]
        g_analit = analit.grad(x)
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)

        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_almost_equal(g_analit, g_approx)

        x = [2.5, 0.3]
        f_analit = analit.fun(x)
        g_analit = analit.grad(x)
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        f_approx = approx.fun(x)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(g_analit, g_approx)

        x = [2, 0.3]
        f_analit = analit.fun(x)
        g_analit = analit.grad(x)
        nfev += 1
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        f_approx = approx.fun(x)
        g_approx = approx.grad(x)
        nfev += 3
        ngev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(g_analit, g_approx)

    def test_fun_and_grad(self):
        ex = ExScalarFunction()

        def fg_allclose(x, y):
            assert_allclose(x[0], y[0])
            assert_allclose(x[1], y[1])

        # with analytic gradient
        x0 = [2.0, 0.3]
        analit = ScalarFunction(ex.fun, x0, (), ex.grad,
                                ex.hess, None, (-np.inf, np.inf))

        fg = ex.fun(x0), ex.grad(x0)
        fg_allclose(analit.fun_and_grad(x0), fg)
        assert(analit.ngev == 1)

        x0[1] = 1.
        fg = ex.fun(x0), ex.grad(x0)
        fg_allclose(analit.fun_and_grad(x0), fg)

        # with finite difference gradient
        x0 = [2.0, 0.3]
        sf = ScalarFunction(ex.fun, x0, (), '3-point',
                                ex.hess, None, (-np.inf, np.inf))
        assert(sf.ngev == 1)
        fg = ex.fun(x0), ex.grad(x0)
        fg_allclose(sf.fun_and_grad(x0), fg)
        assert(sf.ngev == 1)

        x0[1] = 1.
        fg = ex.fun(x0), ex.grad(x0)
        fg_allclose(sf.fun_and_grad(x0), fg)

    def test_finite_difference_hess_linear_operator(self):
        ex = ExScalarFunction()
        nfev = 0
        ngev = 0
        nhev = 0

        x0 = [1.0, 0.0]
        analit = ScalarFunction(ex.fun, x0, (), ex.grad,
                                ex.hess, None, (-np.inf, np.inf))
        nfev += 1
        ngev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev, nhev)
        approx = ScalarFunction(ex.fun, x0, (), ex.grad,
                                '2-point', None, (-np.inf, np.inf))
        assert_(isinstance(approx.H, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_equal(analit.f, approx.f)
            assert_array_almost_equal(analit.g, approx.g)
            assert_array_almost_equal(analit.H.dot(v), approx.H.dot(v))
        nfev += 1
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [2.0, 1.0]
        H_analit = analit.hess(x)
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [2.1, 1.2]
        H_analit = analit.hess(x)
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [2.5, 0.3]
        _ = analit.grad(x)
        H_analit = analit.hess(x)
        ngev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        _ = approx.grad(x)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [5.2, 2.3]
        _ = analit.grad(x)
        H_analit = analit.hess(x)
        ngev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        _ = approx.grad(x)
        H_approx = approx.hess(x)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        ngev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.ngev, ngev)
        assert_array_equal(analit.ngev+approx.ngev, ngev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

    def test_x_storage_overlap(self):
        # Scalar_Function should not store references to arrays, it should
        # store copies - this checks that updating an array in-place causes
        # Scalar_Function.x to be updated.

        def f(x):
            return np.sum(np.asarray(x) ** 2)

        x = np.array([1., 2., 3.])
        sf = ScalarFunction(f, x, (), '3-point', lambda x: x, None, (-np.inf, np.inf))

        assert x is not sf.x
        assert_equal(sf.fun(x), 14.0)
        assert x is not sf.x

        x[0] = 0.
        f1 = sf.fun(x)
        assert_equal(f1, 13.0)

        x[0] = 1
        f2 = sf.fun(x)
        assert_equal(f2, 14.0)
        assert x is not sf.x

        # now test with a HessianUpdate strategy specified
        hess = BFGS()
        x = np.array([1., 2., 3.])
        sf = ScalarFunction(f, x, (), '3-point', hess, None, (-np.inf, np.inf))

        assert x is not sf.x
        assert_equal(sf.fun(x), 14.0)
        assert x is not sf.x

        x[0] = 0.
        f1 = sf.fun(x)
        assert_equal(f1, 13.0)

        x[0] = 1
        f2 = sf.fun(x)
        assert_equal(f2, 14.0)
        assert x is not sf.x

        # gh13740 x is changed in user function
        def ff(x):
            x *= x    # overwrite x
            return np.sum(x)

        x = np.array([1., 2., 3.])
        sf = ScalarFunction(
            ff, x, (), '3-point', lambda x: x, None, (-np.inf, np.inf)
        )
        assert x is not sf.x
        assert_equal(sf.fun(x), 14.0)
        assert_equal(sf.x, np.array([1., 2., 3.]))
        assert x is not sf.x


class ExVectorialFunction:

    def __init__(self):
        self.nfev = 0
        self.njev = 0
        self.nhev = 0

    def fun(self, x):
        self.nfev += 1
        return np.array([2*(x[0]**2 + x[1]**2 - 1) - x[0],
                         4*(x[0]**3 + x[1]**2 - 4) - 3*x[0]])

    def jac(self, x):
        self.njev += 1
        return np.array([[4*x[0]-1, 4*x[1]],
                         [12*x[0]**2-3, 8*x[1]]])

    def hess(self, x, v):
        self.nhev += 1
        return v[0]*4*np.eye(2) + v[1]*np.array([[24*x[0], 0],
                                                 [0, 8]])


class TestVectorialFunction(TestCase):

    def test_finite_difference_jac(self):
        ex = ExVectorialFunction()
        nfev = 0
        njev = 0

        x0 = [1.0, 0.0]
        analit = VectorFunction(ex.fun, x0, ex.jac, ex.hess, None, None,
                                (-np.inf, np.inf), None)
        nfev += 1
        njev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev, njev)
        approx = VectorFunction(ex.fun, x0, '2-point', ex.hess, None, None,
                                (-np.inf, np.inf), None)
        nfev += 3
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(analit.f, approx.f)
        assert_array_almost_equal(analit.J, approx.J)

        x = [10, 0.3]
        f_analit = analit.fun(x)
        J_analit = analit.jac(x)
        nfev += 1
        njev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        f_approx = approx.fun(x)
        J_approx = approx.jac(x)
        nfev += 3
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(J_analit, J_approx, decimal=4)

        x = [2.0, 1.0]
        J_analit = analit.jac(x)
        njev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        J_approx = approx.jac(x)
        nfev += 3
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_almost_equal(J_analit, J_approx)

        x = [2.5, 0.3]
        f_analit = analit.fun(x)
        J_analit = analit.jac(x)
        nfev += 1
        njev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        f_approx = approx.fun(x)
        J_approx = approx.jac(x)
        nfev += 3
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(J_analit, J_approx)

        x = [2, 0.3]
        f_analit = analit.fun(x)
        J_analit = analit.jac(x)
        nfev += 1
        njev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        f_approx = approx.fun(x)
        J_approx = approx.jac(x)
        nfev += 3
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_almost_equal(f_analit, f_approx)
        assert_array_almost_equal(J_analit, J_approx)

    def test_finite_difference_hess_linear_operator(self):
        ex = ExVectorialFunction()
        nfev = 0
        njev = 0
        nhev = 0

        x0 = [1.0, 0.0]
        v0 = [1.0, 2.0]
        analit = VectorFunction(ex.fun, x0, ex.jac, ex.hess, None, None,
                                (-np.inf, np.inf), None)
        nfev += 1
        njev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev, nhev)
        approx = VectorFunction(ex.fun, x0, ex.jac, '2-point', None, None,
                                (-np.inf, np.inf), None)
        assert_(isinstance(approx.H, LinearOperator))
        for p in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_equal(analit.f, approx.f)
            assert_array_almost_equal(analit.J, approx.J)
            assert_array_almost_equal(analit.H.dot(p), approx.H.dot(p))
        nfev += 1
        njev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [2.0, 1.0]
        H_analit = analit.hess(x, v0)
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        H_approx = approx.hess(x, v0)
        assert_(isinstance(H_approx, LinearOperator))
        for p in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(p), H_approx.dot(p),
                                      decimal=5)
        njev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [2.1, 1.2]
        v = [1.0, 1.0]
        H_analit = analit.hess(x, v)
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        H_approx = approx.hess(x, v)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v))
        njev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [2.5, 0.3]
        _ = analit.jac(x)
        H_analit = analit.hess(x, v0)
        njev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        _ = approx.jac(x)
        H_approx = approx.hess(x, v0)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v), decimal=4)
        njev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

        x = [5.2, 2.3]
        v = [2.3, 5.2]
        _ = analit.jac(x)
        H_analit = analit.hess(x, v)
        njev += 1
        nhev += 1
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)
        _ = approx.jac(x)
        H_approx = approx.hess(x, v)
        assert_(isinstance(H_approx, LinearOperator))
        for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 2.0]):
            assert_array_almost_equal(H_analit.dot(v), H_approx.dot(v), decimal=4)
        njev += 4
        assert_array_equal(ex.nfev, nfev)
        assert_array_equal(analit.nfev+approx.nfev, nfev)
        assert_array_equal(ex.njev, njev)
        assert_array_equal(analit.njev+approx.njev, njev)
        assert_array_equal(ex.nhev, nhev)
        assert_array_equal(analit.nhev+approx.nhev, nhev)

    def test_x_storage_overlap(self):
        # VectorFunction should not store references to arrays, it should
        # store copies - this checks that updating an array in-place causes
        # Scalar_Function.x to be updated.
        ex = ExVectorialFunction()
        x0 = np.array([1.0, 0.0])

        vf = VectorFunction(ex.fun, x0, '3-point', ex.hess, None, None,
                            (-np.inf, np.inf), None)

        assert x0 is not vf.x
        assert_equal(vf.fun(x0), ex.fun(x0))
        assert x0 is not vf.x

        x0[0] = 2.
        assert_equal(vf.fun(x0), ex.fun(x0))
        assert x0 is not vf.x

        x0[0] = 1.
        assert_equal(vf.fun(x0), ex.fun(x0))
        assert x0 is not vf.x

        # now test with a HessianUpdate strategy specified
        hess = BFGS()
        x0 = np.array([1.0, 0.0])
        vf = VectorFunction(ex.fun, x0, '3-point', hess, None, None,
                            (-np.inf, np.inf), None)

        with pytest.warns(UserWarning):
            # filter UserWarning because ExVectorialFunction is linear and
            # a quasi-Newton approximation is used for the Hessian.
            assert x0 is not vf.x
            assert_equal(vf.fun(x0), ex.fun(x0))
            assert x0 is not vf.x

            x0[0] = 2.
            assert_equal(vf.fun(x0), ex.fun(x0))
            assert x0 is not vf.x

            x0[0] = 1.
            assert_equal(vf.fun(x0), ex.fun(x0))
            assert x0 is not vf.x


def test_LinearVectorFunction():
    A_dense = np.array([
        [-1, 2, 0],
        [0, 4, 2]
    ])
    x0 = np.zeros(3)
    A_sparse = csr_matrix(A_dense)
    x = np.array([1, -1, 0])
    v = np.array([-1, 1])
    Ax = np.array([-3, -4])

    f1 = LinearVectorFunction(A_dense, x0, None)
    assert_(not f1.sparse_jacobian)

    f2 = LinearVectorFunction(A_dense, x0, True)
    assert_(f2.sparse_jacobian)

    f3 = LinearVectorFunction(A_dense, x0, False)
    assert_(not f3.sparse_jacobian)

    f4 = LinearVectorFunction(A_sparse, x0, None)
    assert_(f4.sparse_jacobian)

    f5 = LinearVectorFunction(A_sparse, x0, True)
    assert_(f5.sparse_jacobian)

    f6 = LinearVectorFunction(A_sparse, x0, False)
    assert_(not f6.sparse_jacobian)

    assert_array_equal(f1.fun(x), Ax)
    assert_array_equal(f2.fun(x), Ax)
    assert_array_equal(f1.jac(x), A_dense)
    assert_array_equal(f2.jac(x).toarray(), A_sparse.toarray())
    assert_array_equal(f1.hess(x, v).toarray(), np.zeros((3, 3)))


def test_LinearVectorFunction_memoization():
    A = np.array([[-1, 2, 0], [0, 4, 2]])
    x0 = np.array([1, 2, -1])
    fun = LinearVectorFunction(A, x0, False)

    assert_array_equal(x0, fun.x)
    assert_array_equal(A.dot(x0), fun.f)

    x1 = np.array([-1, 3, 10])
    assert_array_equal(A, fun.jac(x1))
    assert_array_equal(x1, fun.x)
    assert_array_equal(A.dot(x0), fun.f)
    assert_array_equal(A.dot(x1), fun.fun(x1))
    assert_array_equal(A.dot(x1), fun.f)


def test_IdentityVectorFunction():
    x0 = np.zeros(3)

    f1 = IdentityVectorFunction(x0, None)
    f2 = IdentityVectorFunction(x0, False)
    f3 = IdentityVectorFunction(x0, True)

    assert_(f1.sparse_jacobian)
    assert_(not f2.sparse_jacobian)
    assert_(f3.sparse_jacobian)

    x = np.array([-1, 2, 1])
    v = np.array([-2, 3, 0])

    assert_array_equal(f1.fun(x), x)
    assert_array_equal(f2.fun(x), x)

    assert_array_equal(f1.jac(x).toarray(), np.eye(3))
    assert_array_equal(f2.jac(x), np.eye(3))

    assert_array_equal(f1.hess(x, v).toarray(), np.zeros((3, 3)))
