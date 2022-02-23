"""
Unit test for Linear Programming via Simplex Algorithm.
"""

# TODO: add tests for:
# https://github.com/scipy/scipy/issues/5400
# https://github.com/scipy/scipy/issues/6690

import numpy as np
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_equal)

from .test_linprog import magic_square
from scipy.optimize._remove_redundancy import _remove_redundancy_svd
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_dense
from scipy.optimize._remove_redundancy import _remove_redundancy_pivot_sparse
from scipy.optimize._remove_redundancy import _remove_redundancy_id

from scipy.sparse import csc_matrix


def setup_module():
    np.random.seed(2017)


def _assert_success(
        res,
        desired_fun=None,
        desired_x=None,
        rtol=1e-7,
        atol=1e-7):
    # res: linprog result object
    # desired_fun: desired objective function value or None
    # desired_x: desired solution or None
    assert_(res.success)
    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(
            res.fun,
            desired_fun,
            err_msg="converged to an unexpected objective value",
            rtol=rtol,
            atol=atol)
    if desired_x is not None:
        assert_allclose(
            res.x,
            desired_x,
            err_msg="converged to an unexpected solution",
            rtol=rtol,
            atol=atol)


def redundancy_removed(A, B):
    """Checks whether a matrix contains only independent rows of another"""
    for rowA in A:
        # `rowA in B` is not a reliable check
        for rowB in B:
            if np.all(rowA == rowB):
                break
        else:
            return False
    return A.shape[0] == np.linalg.matrix_rank(A) == np.linalg.matrix_rank(B)


class RRCommonTests:
    def test_no_redundancy(self):
        m, n = 10, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A1, b1, status, message = self.rr(A0, b0)
        assert_allclose(A0, A1)
        assert_allclose(b0, b1)
        assert_equal(status, 0)

    def test_infeasible_zero_row(self):
        A = np.eye(3)
        A[1, :] = 0
        b = np.random.rand(3)
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 2)

    def test_remove_zero_row(self):
        A = np.eye(3)
        A[1, :] = 0
        b = np.random.rand(3)
        b[1] = 0
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_allclose(A1, A[[0, 2], :])
        assert_allclose(b1, b[[0, 2]])

    def test_infeasible_m_gt_n(self):
        m, n = 20, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 2)

    def test_infeasible_m_eq_n(self):
        m, n = 10, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = 2 * A0[-2, :]
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 2)

    def test_infeasible_m_lt_n(self):
        m, n = 9, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 2)

    def test_m_gt_n(self):
        np.random.seed(2032)
        m, n = 20, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        x = np.linalg.solve(A0[:n, :], b0[:n])
        b0[n:] = A0[n:, :].dot(x)
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], n)
        assert_equal(np.linalg.matrix_rank(A1), n)

    def test_m_gt_n_rank_deficient(self):
        m, n = 20, 10
        A0 = np.zeros((m, n))
        A0[:, 0] = 1
        b0 = np.ones(m)
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_allclose(A1, A0[0:1, :])
        assert_allclose(b1, b0[0])

    def test_m_lt_n_rank_deficient(self):
        m, n = 9, 10
        A0 = np.random.rand(m, n)
        b0 = np.random.rand(m)
        A0[-1, :] = np.arange(m - 1).dot(A0[:-1])
        b0[-1] = np.arange(m - 1).dot(b0[:-1])
        A1, b1, status, message = self.rr(A0, b0)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], 8)
        assert_equal(np.linalg.matrix_rank(A1), 8)

    def test_dense1(self):
        A = np.ones((6, 6))
        A[0, :3] = 0
        A[1, 3:] = 0
        A[3:, ::2] = -1
        A[3, :2] = 0
        A[4, 2:] = 0
        b = np.zeros(A.shape[0])

        A1, b1, status, message = self.rr(A, b)
        assert_(redundancy_removed(A1, A))
        assert_equal(status, 0)

    def test_dense2(self):
        A = np.eye(6)
        A[-2, -1] = 1
        A[-1, :] = 1
        b = np.zeros(A.shape[0])
        A1, b1, status, message = self.rr(A, b)
        assert_(redundancy_removed(A1, A))
        assert_equal(status, 0)

    def test_dense3(self):
        A = np.eye(6)
        A[-2, -1] = 1
        A[-1, :] = 1
        b = np.random.rand(A.shape[0])
        b[-1] = np.sum(b[:-1])
        A1, b1, status, message = self.rr(A, b)
        assert_(redundancy_removed(A1, A))
        assert_equal(status, 0)

    def test_m_gt_n_sparse(self):
        np.random.seed(2013)
        m, n = 20, 5
        p = 0.1
        A = np.random.rand(m, n)
        A[np.random.rand(m, n) > p] = 0
        rank = np.linalg.matrix_rank(A)
        b = np.zeros(A.shape[0])
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], rank)
        assert_equal(np.linalg.matrix_rank(A1), rank)

    def test_m_lt_n_sparse(self):
        np.random.seed(2017)
        m, n = 20, 50
        p = 0.05
        A = np.random.rand(m, n)
        A[np.random.rand(m, n) > p] = 0
        rank = np.linalg.matrix_rank(A)
        b = np.zeros(A.shape[0])
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], rank)
        assert_equal(np.linalg.matrix_rank(A1), rank)

    def test_m_eq_n_sparse(self):
        np.random.seed(2017)
        m, n = 100, 100
        p = 0.01
        A = np.random.rand(m, n)
        A[np.random.rand(m, n) > p] = 0
        rank = np.linalg.matrix_rank(A)
        b = np.zeros(A.shape[0])
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], rank)
        assert_equal(np.linalg.matrix_rank(A1), rank)

    def test_magic_square(self):
        A, b, c, numbers = magic_square(3)
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], 23)
        assert_equal(np.linalg.matrix_rank(A1), 23)

    def test_magic_square2(self):
        A, b, c, numbers = magic_square(4)
        A1, b1, status, message = self.rr(A, b)
        assert_equal(status, 0)
        assert_equal(A1.shape[0], 39)
        assert_equal(np.linalg.matrix_rank(A1), 39)


class TestRRSVD(RRCommonTests):
    def rr(self, A, b):
        return _remove_redundancy_svd(A, b)


class TestRRPivotDense(RRCommonTests):
    def rr(self, A, b):
        return _remove_redundancy_pivot_dense(A, b)


class TestRRID(RRCommonTests):
    def rr(self, A, b):
        return _remove_redundancy_id(A, b)


class TestRRPivotSparse(RRCommonTests):
    def rr(self, A, b):
        rr_res = _remove_redundancy_pivot_sparse(csc_matrix(A), b)
        A1, b1, status, message = rr_res
        return A1.toarray(), b1, status, message
