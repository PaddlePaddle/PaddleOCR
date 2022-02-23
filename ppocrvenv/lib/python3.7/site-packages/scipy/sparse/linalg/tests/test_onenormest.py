"""Test functions for the sparse.linalg._onenormest module
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2


class MatrixProductOperator(scipy.sparse.linalg.LinearOperator):
    """
    This is purely for onenormest testing.
    """

    def __init__(self, A, B):
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError('expected ndarrays representing matrices')
        if A.shape[1] != B.shape[0]:
            raise ValueError('incompatible shapes')
        self.A = A
        self.B = B
        self.ndim = 2
        self.shape = (A.shape[0], B.shape[1])

    def _matvec(self, x):
        return np.dot(self.A, np.dot(self.B, x))

    def _rmatvec(self, x):
        return np.dot(np.dot(x, self.A), self.B)

    def _matmat(self, X):
        return np.dot(self.A, np.dot(self.B, X))

    @property
    def T(self):
        return MatrixProductOperator(self.B.T, self.A.T)


class TestOnenormest:

    @pytest.mark.xslow
    def test_onenormest_table_3_t_2(self):
        # This will take multiple seconds if your computer is slow like mine.
        # It is stochastic, so the tolerance could be too strict.
        np.random.seed(1234)
        t = 2
        n = 100
        itmax = 5
        nsamples = 5000
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
            observed.append(est)
            expected.append(scipy.linalg.norm(A, 1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        relative_errors = np.abs(observed - expected) / expected

        # check the mean underestimation ratio
        underestimation_ratio = observed / expected
        assert_(0.99 < np.mean(underestimation_ratio) < 1.0)

        # check the max and mean required column resamples
        assert_equal(np.max(nresample_list), 2)
        assert_(0.05 < np.mean(nresample_list) < 0.2)

        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.9 < proportion_exact < 0.95)

        # check the average number of matrix*vector multiplications
        assert_(3.5 < np.mean(nmult_list) < 4.5)

    @pytest.mark.xslow
    def test_onenormest_table_4_t_7(self):
        # This will take multiple seconds if your computer is slow like mine.
        # It is stochastic, so the tolerance could be too strict.
        np.random.seed(1234)
        t = 7
        n = 100
        itmax = 5
        nsamples = 5000
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []
        for i in range(nsamples):
            A = np.random.randint(-1, 2, size=(n, n))
            est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
            observed.append(est)
            expected.append(scipy.linalg.norm(A, 1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        relative_errors = np.abs(observed - expected) / expected

        # check the mean underestimation ratio
        underestimation_ratio = observed / expected
        assert_(0.90 < np.mean(underestimation_ratio) < 0.99)

        # check the required column resamples
        assert_equal(np.max(nresample_list), 0)

        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.15 < proportion_exact < 0.25)

        # check the average number of matrix*vector multiplications
        assert_(3.5 < np.mean(nmult_list) < 4.5)

    def test_onenormest_table_5_t_1(self):
        # "note that there is no randomness and hence only one estimate for t=1"
        t = 1
        n = 100
        itmax = 5
        alpha = 1 - 1e-6
        A = -scipy.linalg.inv(np.identity(n) + alpha*np.eye(n, k=1))
        first_col = np.array([1] + [0]*(n-1))
        first_row = np.array([(-alpha)**i for i in range(n)])
        B = -scipy.linalg.toeplitz(first_col, first_row)
        assert_allclose(A, B)
        est, v, w, nmults, nresamples = _onenormest_core(B, B.T, t, itmax)
        exact_value = scipy.linalg.norm(B, 1)
        underest_ratio = est / exact_value
        assert_allclose(underest_ratio, 0.05, rtol=1e-4)
        assert_equal(nmults, 11)
        assert_equal(nresamples, 0)
        # check the non-underscored version of onenormest
        est_plain = scipy.sparse.linalg.onenormest(B, t=t, itmax=itmax)
        assert_allclose(est, est_plain)

    @pytest.mark.xslow
    def test_onenormest_table_6_t_1(self):
        #TODO this test seems to give estimates that match the table,
        #TODO even though no attempt has been made to deal with
        #TODO complex numbers in the one-norm estimation.
        # This will take multiple seconds if your computer is slow like mine.
        # It is stochastic, so the tolerance could be too strict.
        np.random.seed(1234)
        t = 1
        n = 100
        itmax = 5
        nsamples = 5000
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []
        for i in range(nsamples):
            A_inv = np.random.rand(n, n) + 1j * np.random.rand(n, n)
            A = scipy.linalg.inv(A_inv)
            est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
            observed.append(est)
            expected.append(scipy.linalg.norm(A, 1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        relative_errors = np.abs(observed - expected) / expected

        # check the mean underestimation ratio
        underestimation_ratio = observed / expected
        underestimation_ratio_mean = np.mean(underestimation_ratio)
        assert_(0.90 < underestimation_ratio_mean < 0.99)

        # check the required column resamples
        max_nresamples = np.max(nresample_list)
        assert_equal(max_nresamples, 0)

        # check the proportion of norms computed exactly correctly
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.7 < proportion_exact < 0.8)

        # check the average number of matrix*vector multiplications
        mean_nmult = np.mean(nmult_list)
        assert_(4 < mean_nmult < 5)

    def _help_product_norm_slow(self, A, B):
        # for profiling
        C = np.dot(A, B)
        return scipy.linalg.norm(C, 1)

    def _help_product_norm_fast(self, A, B):
        # for profiling
        t = 2
        itmax = 5
        D = MatrixProductOperator(A, B)
        est, v, w, nmults, nresamples = _onenormest_core(D, D.T, t, itmax)
        return est

    @pytest.mark.slow
    def test_onenormest_linear_operator(self):
        # Define a matrix through its product A B.
        # Depending on the shapes of A and B,
        # it could be easy to multiply this product by a small matrix,
        # but it could be annoying to look at all of
        # the entries of the product explicitly.
        np.random.seed(1234)
        n = 6000
        k = 3
        A = np.random.randn(n, k)
        B = np.random.randn(k, n)
        fast_estimate = self._help_product_norm_fast(A, B)
        exact_value = self._help_product_norm_slow(A, B)
        assert_(fast_estimate <= exact_value <= 3*fast_estimate,
                'fast: %g\nexact:%g' % (fast_estimate, exact_value))

    def test_returns(self):
        np.random.seed(1234)
        A = scipy.sparse.rand(50, 50, 0.1)

        s0 = scipy.linalg.norm(A.todense(), 1)
        s1, v = scipy.sparse.linalg.onenormest(A, compute_v=True)
        s2, w = scipy.sparse.linalg.onenormest(A, compute_w=True)
        s3, v2, w2 = scipy.sparse.linalg.onenormest(A, compute_w=True, compute_v=True)

        assert_allclose(s1, s0, rtol=1e-9)
        assert_allclose(np.linalg.norm(A.dot(v), 1), s0*np.linalg.norm(v, 1), rtol=1e-9)
        assert_allclose(A.dot(v), w, rtol=1e-9)


class TestAlgorithm_2_2:

    def test_randn_inv(self):
        np.random.seed(1234)
        n = 20
        nsamples = 100
        for i in range(nsamples):

            # Choose integer t uniformly between 1 and 3 inclusive.
            t = np.random.randint(1, 4)

            # Choose n uniformly between 10 and 40 inclusive.
            n = np.random.randint(10, 41)

            # Sample the inverse of a matrix with random normal entries.
            A = scipy.linalg.inv(np.random.randn(n, n))

            # Compute the 1-norm bounds.
            g, ind = _algorithm_2_2(A, A.T, t)

