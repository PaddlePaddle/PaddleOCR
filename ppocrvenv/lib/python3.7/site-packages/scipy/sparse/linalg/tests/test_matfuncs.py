#
# Created by: Pearu Peterson, March 2002
#
""" Test functions for scipy.linalg.matfuncs module

"""
import math

import numpy as np
from numpy import array, eye, exp, random
from numpy.linalg import matrix_power
from numpy.testing import (
        assert_allclose, assert_, assert_array_almost_equal, assert_equal,
        assert_array_almost_equal_nulp, suppress_warnings)

from scipy.sparse import csc_matrix, SparseEfficiencyWarning
from scipy.sparse.construct import eye as speye
from scipy.sparse.linalg.matfuncs import (expm, _expm,
        ProductOperator, MatrixPowerOperator,
        _onenorm_matrix_power_nnm)
from scipy.sparse.sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg


def _burkardt_13_power(n, p):
    """
    A helper function for testing matrix functions.

    Parameters
    ----------
    n : integer greater than 1
        Order of the square matrix to be returned.
    p : non-negative integer
        Power of the matrix.

    Returns
    -------
    out : ndarray representing a square matrix
        A Forsythe matrix of order n, raised to the power p.

    """
    # Input validation.
    if n != int(n) or n < 2:
        raise ValueError('n must be an integer greater than 1')
    n = int(n)
    if p != int(p) or p < 0:
        raise ValueError('p must be a non-negative integer')
    p = int(p)

    # Construct the matrix explicitly.
    a, b = divmod(p, n)
    large = np.power(10.0, -n*a)
    small = large * np.power(10.0, -n)
    return np.diag([large]*(n-b), b) + np.diag([small]*b, b-n)


def test_onenorm_matrix_power_nnm():
    np.random.seed(1234)
    for n in range(1, 5):
        for p in range(5):
            M = np.random.random((n, n))
            Mp = np.linalg.matrix_power(M, p)
            observed = _onenorm_matrix_power_nnm(M, p)
            expected = np.linalg.norm(Mp, 1)
            assert_allclose(observed, expected)


class TestExpM:
    def test_zero_ndarray(self):
        a = array([[0.,0],[0,0]])
        assert_array_almost_equal(expm(a),[[1,0],[0,1]])

    def test_zero_sparse(self):
        a = csc_matrix([[0.,0],[0,0]])
        assert_array_almost_equal(expm(a).toarray(),[[1,0],[0,1]])

    def test_zero_matrix(self):
        a = matrix([[0.,0],[0,0]])
        assert_array_almost_equal(expm(a),[[1,0],[0,1]])

    def test_misc_types(self):
        A = expm(np.array([[1]]))
        assert_allclose(expm(((1,),)), A)
        assert_allclose(expm([[1]]), A)
        assert_allclose(expm(matrix([[1]])), A)
        assert_allclose(expm(np.array([[1]])), A)
        assert_allclose(expm(csc_matrix([[1]])).A, A)
        B = expm(np.array([[1j]]))
        assert_allclose(expm(((1j,),)), B)
        assert_allclose(expm([[1j]]), B)
        assert_allclose(expm(matrix([[1j]])), B)
        assert_allclose(expm(csc_matrix([[1j]])).A, B)

    def test_bidiagonal_sparse(self):
        A = csc_matrix([
            [1, 3, 0],
            [0, 1, 5],
            [0, 0, 2]], dtype=float)
        e1 = math.exp(1)
        e2 = math.exp(2)
        expected = np.array([
            [e1, 3*e1, 15*(e2 - 2*e1)],
            [0, e1, 5*(e2 - e1)],
            [0, 0, e2]], dtype=float)
        observed = expm(A).toarray()
        assert_array_almost_equal(observed, expected)

    def test_padecases_dtype_float(self):
        for dtype in [np.float32, np.float64]:
            for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
                A = scale * eye(3, dtype=dtype)
                observed = expm(A)
                expected = exp(scale) * eye(3, dtype=dtype)
                assert_array_almost_equal_nulp(observed, expected, nulp=100)

    def test_padecases_dtype_complex(self):
        for dtype in [np.complex64, np.complex128]:
            for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
                A = scale * eye(3, dtype=dtype)
                observed = expm(A)
                expected = exp(scale) * eye(3, dtype=dtype)
                assert_array_almost_equal_nulp(observed, expected, nulp=100)

    def test_padecases_dtype_sparse_float(self):
        # float32 and complex64 lead to errors in spsolve/UMFpack
        dtype = np.float64
        for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
            a = scale * speye(3, 3, dtype=dtype, format='csc')
            e = exp(scale) * eye(3, dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning,
                           "Changing the sparsity structure of a csc_matrix is expensive.")
                exact_onenorm = _expm(a, use_exact_onenorm=True).toarray()
                inexact_onenorm = _expm(a, use_exact_onenorm=False).toarray()
            assert_array_almost_equal_nulp(exact_onenorm, e, nulp=100)
            assert_array_almost_equal_nulp(inexact_onenorm, e, nulp=100)

    def test_padecases_dtype_sparse_complex(self):
        # float32 and complex64 lead to errors in spsolve/UMFpack
        dtype = np.complex128
        for scale in [1e-2, 1e-1, 5e-1, 1, 10]:
            a = scale * speye(3, 3, dtype=dtype, format='csc')
            e = exp(scale) * eye(3, dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning,
                           "Changing the sparsity structure of a csc_matrix is expensive.")
                assert_array_almost_equal_nulp(expm(a).toarray(), e, nulp=100)

    def test_logm_consistency(self):
        random.seed(1234)
        for dtype in [np.float64, np.complex128]:
            for n in range(1, 10):
                for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]:
                    # make logm(A) be of a given scale
                    A = (eye(n) + random.rand(n, n) * scale).astype(dtype)
                    if np.iscomplexobj(A):
                        A = A + 1j * random.rand(n, n) * scale
                    assert_array_almost_equal(expm(logm(A)), A)

    def test_integer_matrix(self):
        Q = np.array([
            [-3, 1, 1, 1],
            [1, -3, 1, 1],
            [1, 1, -3, 1],
            [1, 1, 1, -3]])
        assert_allclose(expm(Q), expm(1.0 * Q))

    def test_integer_matrix_2(self):
        # Check for integer overflows
        Q = np.array([[-500, 500, 0, 0],
                      [0, -550, 360, 190],
                      [0, 630, -630, 0],
                      [0, 0, 0, 0]], dtype=np.int16)
        assert_allclose(expm(Q), expm(1.0 * Q))

        Q = csc_matrix(Q)
        assert_allclose(expm(Q).A, expm(1.0 * Q).A)

    def test_triangularity_perturbation(self):
        # Experiment (1) of
        # Awad H. Al-Mohy and Nicholas J. Higham (2012)
        # Improved Inverse Scaling and Squaring Algorithms
        # for the Matrix Logarithm.
        A = np.array([
            [3.2346e-1, 3e4, 3e4, 3e4],
            [0, 3.0089e-1, 3e4, 3e4],
            [0, 0, 3.221e-1, 3e4],
            [0, 0, 0, 3.0744e-1]],
            dtype=float)
        A_logm = np.array([
            [-1.12867982029050462e+00, 9.61418377142025565e+04,
             -4.52485573953179264e+09, 2.92496941103871812e+14],
            [0.00000000000000000e+00, -1.20101052953082288e+00,
             9.63469687211303099e+04, -4.68104828911105442e+09],
            [0.00000000000000000e+00, 0.00000000000000000e+00,
             -1.13289322264498393e+00, 9.53249183094775653e+04],
            [0.00000000000000000e+00, 0.00000000000000000e+00,
             0.00000000000000000e+00, -1.17947533272554850e+00]],
            dtype=float)
        assert_allclose(expm(A_logm), A, rtol=1e-4)

        # Perturb the upper triangular matrix by tiny amounts,
        # so that it becomes technically not upper triangular.
        random.seed(1234)
        tiny = 1e-17
        A_logm_perturbed = A_logm.copy()
        A_logm_perturbed[1, 0] = tiny
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "Ill-conditioned.*")
            A_expm_logm_perturbed = expm(A_logm_perturbed)
        rtol = 1e-4
        atol = 100 * tiny
        assert_(not np.allclose(A_expm_logm_perturbed, A, rtol=rtol, atol=atol))

    def test_burkardt_1(self):
        # This matrix is diagonal.
        # The calculation of the matrix exponential is simple.
        #
        # This is the first of a series of matrix exponential tests
        # collected by John Burkardt from the following sources.
        #
        # Alan Laub,
        # Review of "Linear System Theory" by Joao Hespanha,
        # SIAM Review,
        # Volume 52, Number 4, December 2010, pages 779--781.
        #
        # Cleve Moler and Charles Van Loan,
        # Nineteen Dubious Ways to Compute the Exponential of a Matrix,
        # Twenty-Five Years Later,
        # SIAM Review,
        # Volume 45, Number 1, March 2003, pages 3--49.
        #
        # Cleve Moler,
        # Cleve's Corner: A Balancing Act for the Matrix Exponential,
        # 23 July 2012.
        #
        # Robert Ward,
        # Numerical computation of the matrix exponential
        # with accuracy estimate,
        # SIAM Journal on Numerical Analysis,
        # Volume 14, Number 4, September 1977, pages 600--610.
        exp1 = np.exp(1)
        exp2 = np.exp(2)
        A = np.array([
            [1, 0],
            [0, 2],
            ], dtype=float)
        desired = np.array([
            [exp1, 0],
            [0, exp2],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_2(self):
        # This matrix is symmetric.
        # The calculation of the matrix exponential is straightforward.
        A = np.array([
            [1, 3],
            [3, 2],
            ], dtype=float)
        desired = np.array([
            [39.322809708033859, 46.166301438885753],
            [46.166301438885768, 54.711576854329110],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_3(self):
        # This example is due to Laub.
        # This matrix is ill-suited for the Taylor series approach.
        # As powers of A are computed, the entries blow up too quickly.
        exp1 = np.exp(1)
        exp39 = np.exp(39)
        A = np.array([
            [0, 1],
            [-39, -40],
            ], dtype=float)
        desired = np.array([
            [
                39/(38*exp1) - 1/(38*exp39),
                -np.expm1(-38) / (38*exp1)],
            [
                39*np.expm1(-38) / (38*exp1),
                -1/(38*exp1) + 39/(38*exp39)],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_4(self):
        # This example is due to Moler and Van Loan.
        # The example will cause problems for the series summation approach,
        # as well as for diagonal Pade approximations.
        A = np.array([
            [-49, 24],
            [-64, 31],
            ], dtype=float)
        U = np.array([[3, 1], [4, 2]], dtype=float)
        V = np.array([[1, -1/2], [-2, 3/2]], dtype=float)
        w = np.array([-17, -1], dtype=float)
        desired = np.dot(U * np.exp(w), V)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_5(self):
        # This example is due to Moler and Van Loan.
        # This matrix is strictly upper triangular
        # All powers of A are zero beyond some (low) limit.
        # This example will cause problems for Pade approximations.
        A = np.array([
            [0, 6, 0, 0],
            [0, 0, 6, 0],
            [0, 0, 0, 6],
            [0, 0, 0, 0],
            ], dtype=float)
        desired = np.array([
            [1, 6, 18, 36],
            [0, 1, 6, 18],
            [0, 0, 1, 6],
            [0, 0, 0, 1],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_6(self):
        # This example is due to Moler and Van Loan.
        # This matrix does not have a complete set of eigenvectors.
        # That means the eigenvector approach will fail.
        exp1 = np.exp(1)
        A = np.array([
            [1, 1],
            [0, 1],
            ], dtype=float)
        desired = np.array([
            [exp1, exp1],
            [0, exp1],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_7(self):
        # This example is due to Moler and Van Loan.
        # This matrix is very close to example 5.
        # Mathematically, it has a complete set of eigenvectors.
        # Numerically, however, the calculation will be suspect.
        exp1 = np.exp(1)
        eps = np.spacing(1)
        A = np.array([
            [1 + eps, 1],
            [0, 1 - eps],
            ], dtype=float)
        desired = np.array([
            [exp1, exp1],
            [0, exp1],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_8(self):
        # This matrix was an example in Wikipedia.
        exp4 = np.exp(4)
        exp16 = np.exp(16)
        A = np.array([
            [21, 17, 6],
            [-5, -1, -6],
            [4, 4, 16],
            ], dtype=float)
        desired = np.array([
            [13*exp16 - exp4, 13*exp16 - 5*exp4, 2*exp16 - 2*exp4],
            [-9*exp16 + exp4, -9*exp16 + 5*exp4, -2*exp16 + 2*exp4],
            [16*exp16, 16*exp16, 4*exp16],
            ], dtype=float) * 0.25
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_9(self):
        # This matrix is due to the NAG Library.
        # It is an example for function F01ECF.
        A = np.array([
            [1, 2, 2, 2],
            [3, 1, 1, 2],
            [3, 2, 1, 2],
            [3, 3, 3, 1],
            ], dtype=float)
        desired = np.array([
            [740.7038, 610.8500, 542.2743, 549.1753],
            [731.2510, 603.5524, 535.0884, 542.2743],
            [823.7630, 679.4257, 603.5524, 610.8500],
            [998.4355, 823.7630, 731.2510, 740.7038],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_10(self):
        # This is Ward's example #1.
        # It is defective and nonderogatory.
        A = np.array([
            [4, 2, 0],
            [1, 4, 1],
            [1, 1, 4],
            ], dtype=float)
        assert_allclose(sorted(scipy.linalg.eigvals(A)), (3, 3, 6))
        desired = np.array([
            [147.8666224463699, 183.7651386463682, 71.79703239999647],
            [127.7810855231823, 183.7651386463682, 91.88256932318415],
            [127.7810855231824, 163.6796017231806, 111.9681062463718],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_11(self):
        # This is Ward's example #2.
        # It is a symmetric matrix.
        A = np.array([
            [29.87942128909879, 0.7815750847907159, -2.289519314033932],
            [0.7815750847907159, 25.72656945571064, 8.680737820540137],
            [-2.289519314033932, 8.680737820540137, 34.39400925519054],
            ], dtype=float)
        assert_allclose(scipy.linalg.eigvalsh(A), (20, 30, 40))
        desired = np.array([
             [
                 5.496313853692378E+15,
                 -1.823188097200898E+16,
                 -3.047577080858001E+16],
             [
                -1.823188097200899E+16,
                6.060522870222108E+16,
                1.012918429302482E+17],
             [
                -3.047577080858001E+16,
                1.012918429302482E+17,
                1.692944112408493E+17],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_12(self):
        # This is Ward's example #3.
        # Ward's algorithm has difficulty estimating the accuracy
        # of its results.
        A = np.array([
            [-131, 19, 18],
            [-390, 56, 54],
            [-387, 57, 52],
            ], dtype=float)
        assert_allclose(sorted(scipy.linalg.eigvals(A)), (-20, -2, -1))
        desired = np.array([
            [-1.509644158793135, 0.3678794391096522, 0.1353352811751005],
            [-5.632570799891469, 1.471517758499875, 0.4060058435250609],
            [-4.934938326088363, 1.103638317328798, 0.5413411267617766],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_13(self):
        # This is Ward's example #4.
        # This is a version of the Forsythe matrix.
        # The eigenvector problem is badly conditioned.
        # Ward's algorithm has difficulty esimating the accuracy
        # of its results for this problem.
        #
        # Check the construction of one instance of this family of matrices.
        A4_actual = _burkardt_13_power(4, 1)
        A4_desired = [[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1e-4, 0, 0, 0]]
        assert_allclose(A4_actual, A4_desired)
        # Check the expm for a few instances.
        for n in (2, 3, 4, 10):
            # Approximate expm using Taylor series.
            # This works well for this matrix family
            # because each matrix in the summation,
            # even before dividing by the factorial,
            # is entrywise positive with max entry 10**(-floor(p/n)*n).
            k = max(1, int(np.ceil(16/n)))
            desired = np.zeros((n, n), dtype=float)
            for p in range(n*k):
                Ap = _burkardt_13_power(n, p)
                assert_equal(np.min(Ap), 0)
                assert_allclose(np.max(Ap), np.power(10, -np.floor(p/n)*n))
                desired += Ap / factorial(p)
            actual = expm(_burkardt_13_power(n, 1))
            assert_allclose(actual, desired)

    def test_burkardt_14(self):
        # This is Moler's example.
        # This badly scaled matrix caused problems for MATLAB's expm().
        A = np.array([
            [0, 1e-8, 0],
            [-(2e10 + 4e8/6.), -3, 2e10],
            [200./3., 0, -200./3.],
            ], dtype=float)
        desired = np.array([
            [0.446849468283175, 1.54044157383952e-09, 0.462811453558774],
            [-5743067.77947947, -0.0152830038686819, -4526542.71278401],
            [0.447722977849494, 1.54270484519591e-09, 0.463480648837651],
            ], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_pascal(self):
        # Test pascal triangle.
        # Nilpotent exponential, used to trigger a failure (gh-8029)

        for scale in [1.0, 1e-3, 1e-6]:
            for n in range(0, 80, 3):
                sc = scale ** np.arange(n, -1, -1)
                if np.any(sc < 1e-300):
                    break

                A = np.diag(np.arange(1, n + 1), -1) * scale
                B = expm(A)

                got = B
                expected = binom(np.arange(n + 1)[:,None],
                                 np.arange(n + 1)[None,:]) * sc[None,:] / sc[:,None]
                atol = 1e-13 * abs(expected).max()
                assert_allclose(got, expected, atol=atol)

    def test_matrix_input(self):
        # Large np.matrix inputs should work, gh-5546
        A = np.zeros((200, 200))
        A[-1,0] = 1
        B0 = expm(A)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "the matrix subclass.*")
            sup.filter(PendingDeprecationWarning, "the matrix subclass.*")
            B = expm(np.matrix(A))
        assert_allclose(B, B0)

    def test_exp_sinch_overflow(self):
        # Check overflow in intermediate steps is fixed (gh-11839)
        L = np.array([[1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -0.5, -0.5, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, -0.5, -0.5],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        E0 = expm(-L)
        E1 = expm(-2**11 * L)
        E2 = E0
        for j in range(11):
            E2 = E2 @ E2

        assert_allclose(E1, E2)


class TestOperators:

    def test_product_operator(self):
        random.seed(1234)
        n = 5
        k = 2
        nsamples = 10
        for i in range(nsamples):
            A = np.random.randn(n, n)
            B = np.random.randn(n, n)
            C = np.random.randn(n, n)
            D = np.random.randn(n, k)
            op = ProductOperator(A, B, C)
            assert_allclose(op.matmat(D), A.dot(B).dot(C).dot(D))
            assert_allclose(op.T.matmat(D), (A.dot(B).dot(C)).T.dot(D))

    def test_matrix_power_operator(self):
        random.seed(1234)
        n = 5
        k = 2
        p = 3
        nsamples = 10
        for i in range(nsamples):
            A = np.random.randn(n, n)
            B = np.random.randn(n, k)
            op = MatrixPowerOperator(A, p)
            assert_allclose(op.matmat(B), matrix_power(A, p).dot(B))
            assert_allclose(op.T.matmat(B), matrix_power(A, p).T.dot(B))

