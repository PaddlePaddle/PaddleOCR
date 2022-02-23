"""Test functions for the sparse.linalg.norm module
"""

import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose
from pytest import raises as assert_raises

import scipy.sparse
from scipy.sparse.linalg import norm as spnorm


class TestNorm:
    def setup_method(self):
        a = np.arange(9) - 4
        b = a.reshape((3, 3))
        self.b = scipy.sparse.csr_matrix(b)

    def test_matrix_norm(self):

        # Frobenius norm is the default
        assert_allclose(spnorm(self.b), 7.745966692414834)        
        assert_allclose(spnorm(self.b, 'fro'), 7.745966692414834)

        assert_allclose(spnorm(self.b, np.inf), 9)
        assert_allclose(spnorm(self.b, -np.inf), 2)
        assert_allclose(spnorm(self.b, 1), 7)
        assert_allclose(spnorm(self.b, -1), 6)

        # _multi_svd_norm is not implemented for sparse matrix
        assert_raises(NotImplementedError, spnorm, self.b, 2)
        assert_raises(NotImplementedError, spnorm, self.b, -2)

    def test_matrix_norm_axis(self):
        for m, axis in ((self.b, None), (self.b, (0, 1)), (self.b.T, (1, 0))):
            assert_allclose(spnorm(m, axis=axis), 7.745966692414834)        
            assert_allclose(spnorm(m, 'fro', axis=axis), 7.745966692414834)
            assert_allclose(spnorm(m, np.inf, axis=axis), 9)
            assert_allclose(spnorm(m, -np.inf, axis=axis), 2)
            assert_allclose(spnorm(m, 1, axis=axis), 7)
            assert_allclose(spnorm(m, -1, axis=axis), 6)

    def test_vector_norm(self):
        v = [4.5825756949558398, 4.2426406871192848, 4.5825756949558398]
        for m, a in (self.b, 0), (self.b.T, 1):
            for axis in a, (a, ), a-2, (a-2, ):
                assert_allclose(spnorm(m, 1, axis=axis), [7, 6, 7])
                assert_allclose(spnorm(m, np.inf, axis=axis), [4, 3, 4])
                assert_allclose(spnorm(m, axis=axis), v)
                assert_allclose(spnorm(m, ord=2, axis=axis), v)
                assert_allclose(spnorm(m, ord=None, axis=axis), v)

    def test_norm_exceptions(self):
        m = self.b
        assert_raises(TypeError, spnorm, m, None, 1.5)
        assert_raises(TypeError, spnorm, m, None, [2])
        assert_raises(ValueError, spnorm, m, None, ())
        assert_raises(ValueError, spnorm, m, None, (0, 1, 2))
        assert_raises(ValueError, spnorm, m, None, (0, 0))
        assert_raises(ValueError, spnorm, m, None, (0, 2))
        assert_raises(ValueError, spnorm, m, None, (-3, 0))
        assert_raises(ValueError, spnorm, m, None, 2)
        assert_raises(ValueError, spnorm, m, None, -3)
        assert_raises(ValueError, spnorm, m, 'plate_of_shrimp', 0)
        assert_raises(ValueError, spnorm, m, 'plate_of_shrimp', (0, 1))


class TestVsNumpyNorm:
    _sparse_types = (
            scipy.sparse.bsr_matrix,
            scipy.sparse.coo_matrix,
            scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix,
            scipy.sparse.dia_matrix,
            scipy.sparse.dok_matrix,
            scipy.sparse.lil_matrix,
            )
    _test_matrices = (
            (np.arange(9) - 4).reshape((3, 3)),
            [
                [1, 2, 3],
                [-1, 1, 4]],
            [
                [1, 0, 3],
                [-1, 1, 4j]],
            )

    def test_sparse_matrix_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                assert_allclose(spnorm(S), npnorm(M))
                assert_allclose(spnorm(S, 'fro'), npnorm(M, 'fro'))
                assert_allclose(spnorm(S, np.inf), npnorm(M, np.inf))
                assert_allclose(spnorm(S, -np.inf), npnorm(M, -np.inf))
                assert_allclose(spnorm(S, 1), npnorm(M, 1))
                assert_allclose(spnorm(S, -1), npnorm(M, -1))

    def test_sparse_matrix_norms_with_axis(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in None, (0, 1), (1, 0):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    for ord in 'fro', np.inf, -np.inf, 1, -1:
                        assert_allclose(spnorm(S, ord, axis=axis),
                                        npnorm(M, ord, axis=axis))
                # Some numpy matrix norms are allergic to negative axes.
                for axis in (-2, -1), (-1, -2), (1, -2):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    assert_allclose(spnorm(S, 'f', axis=axis),
                                    npnorm(M, 'f', axis=axis))
                    assert_allclose(spnorm(S, 'fro', axis=axis),
                                    npnorm(M, 'fro', axis=axis))

    def test_sparse_vector_norms(self):
        for sparse_type in self._sparse_types:
            for M in self._test_matrices:
                S = sparse_type(M)
                for axis in (0, 1, -1, -2, (0, ), (1, ), (-1, ), (-2, )):
                    assert_allclose(spnorm(S, axis=axis), npnorm(M, axis=axis))
                    for ord in None, 2, np.inf, -np.inf, 1, 0.5, 0.42:
                        assert_allclose(spnorm(S, ord, axis=axis),
                                        npnorm(M, ord, axis=axis))
