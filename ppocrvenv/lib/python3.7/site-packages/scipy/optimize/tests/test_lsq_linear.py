import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_

from scipy.sparse import rand
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear


A = np.array([
    [0.171, -0.057],
    [-0.049, -0.248],
    [-0.166, 0.054],
])
b = np.array([0.074, 1.014, -0.383])


class BaseMixin:
    def setup_method(self):
        self.rnd = np.random.RandomState(0)

    def test_dense_no_bounds(self):
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, method=self.method, lsq_solver=lsq_solver)
            assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])

    def test_dense_bounds(self):
        # Solutions for comparison are taken from MATLAB.
        lb = np.array([-1, -10])
        ub = np.array([1, 0])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, lstsq(A, b, rcond=-1)[0])

        lb = np.array([0.0, -np.inf])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, np.inf), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([0.0, -4.084174437334673]),
                            atol=1e-6)

        lb = np.array([-1, 0])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, np.inf), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([0.448427311733504, 0]),
                            atol=1e-15)

        ub = np.array([np.inf, -5])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([-0.105560998682388, -5]))

        ub = np.array([-1, np.inf])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (-np.inf, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([-1, -4.181102129483254]))

        lb = np.array([0, -4])
        ub = np.array([1, 0])
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, np.array([0.005236663400791, -4]))

    def test_np_matrix(self):
        # gh-10711
        with np.testing.suppress_warnings() as sup:
            sup.filter(PendingDeprecationWarning)
            A = np.matrix([[20, -4, 0, 2, 3], [10, -2, 1, 0, -1]])
        k = np.array([20, 15])
        s_t = lsq_linear(A, k)

    def test_dense_rank_deficient(self):
        A = np.array([[-0.307, -0.184]])
        b = np.array([0.773])
        lb = [-0.1, -0.1]
        ub = [0.1, 0.1]
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.x, [-0.1, -0.1])

        A = np.array([
            [0.334, 0.668],
            [-0.516, -1.032],
            [0.192, 0.384],
        ])
        b = np.array([-1.436, 0.135, 0.909])
        lb = [0, -1]
        ub = [1, -0.5]
        for lsq_solver in self.lsq_solvers:
            res = lsq_linear(A, b, (lb, ub), method=self.method,
                             lsq_solver=lsq_solver)
            assert_allclose(res.optimality, 0, atol=1e-11)

    def test_full_result(self):
        lb = np.array([0, -4])
        ub = np.array([1, 0])
        res = lsq_linear(A, b, (lb, ub), method=self.method)

        assert_allclose(res.x, [0.005236663400791, -4])

        r = A.dot(res.x) - b
        assert_allclose(res.cost, 0.5 * np.dot(r, r))
        assert_allclose(res.fun, r)

        assert_allclose(res.optimality, 0.0, atol=1e-12)
        assert_equal(res.active_mask, [0, -1])
        assert_(res.nit < 15)
        assert_(res.status == 1 or res.status == 3)
        assert_(isinstance(res.message, str))
        assert_(res.success)

    # This is a test for issue #9982.
    def test_almost_singular(self):
        A = np.array(
            [[0.8854232310355122, 0.0365312146937765, 0.0365312146836789],
             [0.3742460132129041, 0.0130523214078376, 0.0130523214077873],
             [0.9680633871281361, 0.0319366128718639, 0.0319366128718388]])

        b = np.array(
            [0.0055029366538097, 0.0026677442422208, 0.0066612514782381])

        result = lsq_linear(A, b, method=self.method)
        assert_(result.cost < 1.1e-8)

    def test_large_rank_deficient(self):
        np.random.seed(0)
        n, m = np.sort(np.random.randint(2, 1000, size=2))
        m *= 2   # make m >> n
        A = 1.0 * np.random.randint(-99, 99, size=[m, n])
        b = 1.0 * np.random.randint(-99, 99, size=[m])
        bounds = 1.0 * np.sort(np.random.randint(-99, 99, size=(2, n)), axis=0)
        bounds[1, :] += 1.0  # ensure up > lb

        # Make the A matrix strongly rank deficient by replicating some columns
        w = np.random.choice(n, n)  # Select random columns with duplicates
        A = A[:, w]

        x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
        x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x

        cost_bvls = np.sum((A @ x_bvls - b)**2)
        cost_trf = np.sum((A @ x_trf - b)**2)

        assert_(abs(cost_bvls - cost_trf) < cost_trf*1e-10)

    def test_convergence_small_matrix(self):
        A = np.array([[49.0, 41.0, -32.0],
                      [-19.0, -32.0, -8.0],
                      [-13.0, 10.0, 69.0]])
        b = np.array([-41.0, -90.0, 47.0])
        bounds = np.array([[31.0, -44.0, 26.0],
                           [54.0, -32.0, 28.0]])

        x_bvls = lsq_linear(A, b, bounds=bounds, method='bvls').x
        x_trf = lsq_linear(A, b, bounds=bounds, method='trf').x

        cost_bvls = np.sum((A @ x_bvls - b)**2)
        cost_trf = np.sum((A @ x_trf - b)**2)

        assert_(abs(cost_bvls - cost_trf) < cost_trf*1e-10)


class SparseMixin:
    def test_sparse_and_LinearOperator(self):
        m = 5000
        n = 1000
        A = rand(m, n, random_state=0)
        b = self.rnd.randn(m)
        res = lsq_linear(A, b)
        assert_allclose(res.optimality, 0, atol=1e-6)

        A = aslinearoperator(A)
        res = lsq_linear(A, b)
        assert_allclose(res.optimality, 0, atol=1e-6)

    def test_sparse_bounds(self):
        m = 5000
        n = 1000
        A = rand(m, n, random_state=0)
        b = self.rnd.randn(m)
        lb = self.rnd.randn(n)
        ub = lb + 1
        res = lsq_linear(A, b, (lb, ub))
        assert_allclose(res.optimality, 0.0, atol=1e-6)

        res = lsq_linear(A, b, (lb, ub), lsmr_tol=1e-13)
        assert_allclose(res.optimality, 0.0, atol=1e-6)

        res = lsq_linear(A, b, (lb, ub), lsmr_tol='auto')
        assert_allclose(res.optimality, 0.0, atol=1e-6)


class TestTRF(BaseMixin, SparseMixin):
    method = 'trf'
    lsq_solvers = ['exact', 'lsmr']


class TestBVLS(BaseMixin):
    method = 'bvls'
    lsq_solvers = ['exact']
