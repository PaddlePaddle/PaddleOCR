import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
    import projections, orthogonality
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_equal, assert_allclose)

try:
    from sksparse.cholmod import cholesky_AAt
    sksparse_available = True
    available_sparse_methods = ("NormalEquation", "AugmentedSystem")
except ImportError:
    sksparse_available = False
    available_sparse_methods = ("AugmentedSystem",)
available_dense_methods = ('QRFactorization', 'SVDFactorization')


class TestProjections(TestCase):

    def test_nullspace_and_least_squares_sparse(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        At_dense = A_dense.T
        A = csc_matrix(A_dense)
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])

        for method in available_sparse_methods:
            Z, LS, _ = projections(A, method)
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_array_almost_equal(A.dot(x), 0)
                # Test orthogonality
                assert_array_almost_equal(orthogonality(A, x), 0)
                # Test if x is the least square solution
                x = LS.matvec(z)
                x2 = scipy.linalg.lstsq(At_dense, z)[0]
                assert_array_almost_equal(x, x2)

    def test_iterative_refinements_sparse(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        A = csc_matrix(A_dense)
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8],
                       [1, 0, 0, 0, 0, 1, 2, 3+1e-10])

        for method in available_sparse_methods:
            Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=100)
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                atol = 1e-13 * abs(x).max()
                assert_allclose(A.dot(x), 0, atol=atol)
                # Test orthogonality
                assert_allclose(orthogonality(A, x), 0, atol=1e-13)

    def test_rowspace_sparse(self):
        A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        A = csc_matrix(A_dense)
        test_points = ([1, 2, 3],
                       [1, 10, 3],
                       [1.12, 10, 0])

        for method in available_sparse_methods:
            _, _, Y = projections(A, method)
            for z in test_points:
                # Test if x is solution of A x = z
                x = Y.matvec(z)
                assert_array_almost_equal(A.dot(x), z)
                # Test if x is in the return row space of A
                A_ext = np.vstack((A_dense, x))
                assert_equal(np.linalg.matrix_rank(A_dense),
                             np.linalg.matrix_rank(A_ext))

    def test_nullspace_and_least_squares_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        At = A.T
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1.12, 10, 0, 0, 100000, 6, 0.7, 8])

        for method in available_dense_methods:
            Z, LS, _ = projections(A, method)
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_array_almost_equal(A.dot(x), 0)
                # Test orthogonality
                assert_array_almost_equal(orthogonality(A, x), 0)
                # Test if x is the least square solution
                x = LS.matvec(z)
                x2 = scipy.linalg.lstsq(At, z)[0]
                assert_array_almost_equal(x, x2)

    def test_compare_dense_and_sparse(self):
        D = np.diag(range(1, 101))
        A = np.hstack([D, D, D, D])
        A_sparse = csc_matrix(A)
        np.random.seed(0)

        Z, LS, Y = projections(A)
        Z_sparse, LS_sparse, Y_sparse = projections(A_sparse)
        for k in range(20):
            z = np.random.normal(size=(400,))
            assert_array_almost_equal(Z.dot(z), Z_sparse.dot(z))
            assert_array_almost_equal(LS.dot(z), LS_sparse.dot(z))
            x = np.random.normal(size=(100,))
            assert_array_almost_equal(Y.dot(x), Y_sparse.dot(x))

    def test_compare_dense_and_sparse2(self):
        D1 = np.diag([-1.7, 1, 0.5])
        D2 = np.diag([1, -0.6, -0.3])
        D3 = np.diag([-0.3, -1.5, 2])
        A = np.hstack([D1, D2, D3])
        A_sparse = csc_matrix(A)
        np.random.seed(0)

        Z, LS, Y = projections(A)
        Z_sparse, LS_sparse, Y_sparse = projections(A_sparse)
        for k in range(1):
            z = np.random.normal(size=(9,))
            assert_array_almost_equal(Z.dot(z), Z_sparse.dot(z))
            assert_array_almost_equal(LS.dot(z), LS_sparse.dot(z))
            x = np.random.normal(size=(3,))
            assert_array_almost_equal(Y.dot(x), Y_sparse.dot(x))

    def test_iterative_refinements_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                            [0, 8, 7, 0, 1, 5, 9, 0],
                            [1, 0, 0, 0, 0, 1, 2, 3]])
        test_points = ([1, 2, 3, 4, 5, 6, 7, 8],
                       [1, 10, 3, 0, 1, 6, 7, 8],
                       [1, 0, 0, 0, 0, 1, 2, 3+1e-10])

        for method in available_dense_methods:
            Z, LS, _ = projections(A, method, orth_tol=1e-18, max_refin=10)
            for z in test_points:
                # Test if x is in the null_space
                x = Z.matvec(z)
                assert_allclose(A.dot(x), 0, rtol=0, atol=2.5e-14)
                # Test orthogonality
                assert_allclose(orthogonality(A, x), 0, rtol=0, atol=5e-16)

    def test_rowspace_dense(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        test_points = ([1, 2, 3],
                       [1, 10, 3],
                       [1.12, 10, 0])

        for method in available_dense_methods:
            _, _, Y = projections(A, method)
            for z in test_points:
                # Test if x is solution of A x = z
                x = Y.matvec(z)
                assert_array_almost_equal(A.dot(x), z)
                # Test if x is in the return row space of A
                A_ext = np.vstack((A, x))
                assert_equal(np.linalg.matrix_rank(A),
                             np.linalg.matrix_rank(A_ext))


class TestOrthogonality(TestCase):

    def test_dense_matrix(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806, 2935.29289083])
        test_expected_orth = (0, 0)

        for i in range(len(test_vectors)):
            x = test_vectors[i]
            orth = test_expected_orth[i]
            assert_array_almost_equal(orthogonality(A, x), orth)

    def test_sparse_matrix(self):
        A = np.array([[1, 2, 3, 4, 0, 5, 0, 7],
                      [0, 8, 7, 0, 1, 5, 9, 0],
                      [1, 0, 0, 0, 0, 1, 2, 3]])
        A = csc_matrix(A)
        test_vectors = ([-1.98931144, -1.56363389,
                         -0.84115584, 2.2864762,
                         5.599141, 0.09286976,
                         1.37040802, -0.28145812],
                        [697.92794044, -4091.65114008,
                         -3327.42316335, 836.86906951,
                         99434.98929065, -1285.37653682,
                         -4109.21503806, 2935.29289083])
        test_expected_orth = (0, 0)

        for i in range(len(test_vectors)):
            x = test_vectors[i]
            orth = test_expected_orth[i]
            assert_array_almost_equal(orthogonality(A, x), orth)
