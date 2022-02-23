"""Test functions for the sparse.linalg._expm_multiply module
"""

import numpy as np
from numpy.testing import (assert_allclose, assert_, assert_equal,
                           suppress_warnings)
from scipy.sparse import SparseEfficiencyWarning
import scipy.linalg
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
        _onenormest_matrix_power, expm_multiply, _expm_multiply_simple,
        _expm_multiply_interval)


def less_than_or_close(a, b):
    return np.allclose(a, b) or (a < b)


class TestExpmActionSimple:
    """
    These tests do not consider the case of multiple time steps in one call.
    """

    def test_theta_monotonicity(self):
        pairs = sorted(_theta.items())
        for (m_a, theta_a), (m_b, theta_b) in zip(pairs[:-1], pairs[1:]):
            assert_(theta_a < theta_b)

    def test_p_max_default(self):
        m_max = 55
        expected_p_max = 8
        observed_p_max = _compute_p_max(m_max)
        assert_equal(observed_p_max, expected_p_max)

    def test_p_max_range(self):
        for m_max in range(1, 55+1):
            p_max = _compute_p_max(m_max)
            assert_(p_max*(p_max - 1) <= m_max + 1)
            p_too_big = p_max + 1
            assert_(p_too_big*(p_too_big - 1) > m_max + 1)

    def test_onenormest_matrix_power(self):
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            for p in range(4):
                if not p:
                    M = np.identity(n)
                else:
                    M = np.dot(M, A)
                estimated = _onenormest_matrix_power(A, p)
                exact = np.linalg.norm(M, 1)
                assert_(less_than_or_close(estimated, exact))
                assert_(less_than_or_close(exact, 3*estimated))

    def test_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            B = np.random.randn(n, k)
            observed = expm_multiply(A, B)
            expected = np.dot(scipy.linalg.expm(A), B)
            assert_allclose(observed, expected)

    def test_matrix_vector_multiply(self):
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            v = np.random.randn(n)
            observed = expm_multiply(A, v)
            expected = np.dot(scipy.linalg.expm(A), v)
            assert_allclose(observed, expected)

    def test_scaled_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            for t in (0.2, 1.0, 1.5):
                with np.errstate(invalid='ignore'):
                    A = scipy.linalg.inv(np.random.randn(n, n))
                    B = np.random.randn(n, k)
                    observed = _expm_multiply_simple(A, B, t=t)
                    expected = np.dot(scipy.linalg.expm(t*A), B)
                    assert_allclose(observed, expected)

    def test_scaled_expm_multiply_single_timepoint(self):
        np.random.seed(1234)
        t = 0.1
        n = 5
        k = 2
        A = np.random.randn(n, n)
        B = np.random.randn(n, k)
        observed = _expm_multiply_simple(A, B, t=t)
        expected = scipy.linalg.expm(t*A).dot(B)
        assert_allclose(observed, expected)

    def test_sparse_expm_multiply(self):
        np.random.seed(1234)
        n = 40
        k = 3
        nsamples = 10
        for i in range(nsamples):
            A = scipy.sparse.rand(n, n, density=0.05)
            B = np.random.randn(n, k)
            observed = expm_multiply(A, B)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning,
                           "splu requires CSC matrix format")
                sup.filter(SparseEfficiencyWarning,
                           "spsolve is more efficient when sparse b is in the CSC matrix format")
                expected = scipy.linalg.expm(A).dot(B)
            assert_allclose(observed, expected)

    def test_complex(self):
        A = np.array([
            [1j, 1j],
            [0, 1j]], dtype=complex)
        B = np.array([1j, 1j])
        observed = expm_multiply(A, B)
        expected = np.array([
            1j * np.exp(1j) + 1j * (1j*np.cos(1) - np.sin(1)),
            1j * np.exp(1j)], dtype=complex)
        assert_allclose(observed, expected)


class TestExpmActionInterval:

    def test_sparse_expm_multiply_interval(self):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        n = 40
        k = 3
        endpoint = True
        for num in (14, 13, 2):
            A = scipy.sparse.rand(n, n, density=0.05)
            B = np.random.randn(n, k)
            v = np.random.randn(n)
            for target in (B, v):
                X = expm_multiply(A, target,
                        start=start, stop=stop, num=num, endpoint=endpoint)
                samples = np.linspace(start=start, stop=stop,
                        num=num, endpoint=endpoint)
                with suppress_warnings() as sup:
                    sup.filter(SparseEfficiencyWarning,
                               "splu requires CSC matrix format")
                    sup.filter(SparseEfficiencyWarning,
                               "spsolve is more efficient when sparse b is in the CSC matrix format")
                    for solution, t in zip(X, samples):
                        assert_allclose(solution,
                                scipy.linalg.expm(t*A).dot(target))

    def test_expm_multiply_interval_vector(self):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        endpoint = True
        for num in (14, 13, 2):
            for n in (1, 2, 5, 20, 40):
                A = scipy.linalg.inv(np.random.randn(n, n))
                v = np.random.randn(n)
                X = expm_multiply(A, v,
                        start=start, stop=stop, num=num, endpoint=endpoint)
                samples = np.linspace(start=start, stop=stop,
                        num=num, endpoint=endpoint)
                for solution, t in zip(X, samples):
                    assert_allclose(solution, scipy.linalg.expm(t*A).dot(v))

    def test_expm_multiply_interval_matrix(self):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        endpoint = True
        for num in (14, 13, 2):
            for n in (1, 2, 5, 20, 40):
                for k in (1, 2):
                    A = scipy.linalg.inv(np.random.randn(n, n))
                    B = np.random.randn(n, k)
                    X = expm_multiply(A, B,
                            start=start, stop=stop, num=num, endpoint=endpoint)
                    samples = np.linspace(start=start, stop=stop,
                            num=num, endpoint=endpoint)
                    for solution, t in zip(X, samples):
                        assert_allclose(solution, scipy.linalg.expm(t*A).dot(B))

    def test_sparse_expm_multiply_interval_dtypes(self):
        # Test A & B int
        A = scipy.sparse.diags(np.arange(5),format='csr', dtype=int)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)),format='csr')
        assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))
    
        # Test A complex, B int
        A = scipy.sparse.diags(-1j*np.arange(5),format='csr', dtype=complex)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(-1j*np.arange(5)),format='csr')
        assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))
    
        # Test A int, B complex
        A = scipy.sparse.diags(np.arange(5),format='csr', dtype=int)
        B = np.full(5, 1j, dtype=complex)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)),format='csr')
        assert_allclose(expm_multiply(A,B,0,1)[-1], Aexpm.dot(B))

    def test_expm_multiply_interval_status_0(self):
        self._help_test_specific_expm_interval_status(0)

    def test_expm_multiply_interval_status_1(self):
        self._help_test_specific_expm_interval_status(1)

    def test_expm_multiply_interval_status_2(self):
        self._help_test_specific_expm_interval_status(2)

    def _help_test_specific_expm_interval_status(self, target_status):
        np.random.seed(1234)
        start = 0.1
        stop = 3.2
        num = 13
        endpoint = True
        n = 5
        k = 2
        nrepeats = 10
        nsuccesses = 0
        for num in [14, 13, 2] * nrepeats:
            A = np.random.randn(n, n)
            B = np.random.randn(n, k)
            status = _expm_multiply_interval(A, B,
                    start=start, stop=stop, num=num, endpoint=endpoint,
                    status_only=True)
            if status == target_status:
                X, status = _expm_multiply_interval(A, B,
                        start=start, stop=stop, num=num, endpoint=endpoint,
                        status_only=False)
                assert_equal(X.shape, (num, n, k))
                samples = np.linspace(start=start, stop=stop,
                        num=num, endpoint=endpoint)
                for solution, t in zip(X, samples):
                    assert_allclose(solution, scipy.linalg.expm(t*A).dot(B))
                nsuccesses += 1
        if not nsuccesses:
            msg = 'failed to find a status-' + str(target_status) + ' interval'
            raise Exception(msg)

