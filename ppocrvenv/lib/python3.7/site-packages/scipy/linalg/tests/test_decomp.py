""" Test functions for linalg.decomp module

"""
__usage__ = """
Build linalg:
  python setup_linalg.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.linalg.test()'
"""

import itertools
import platform
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_, assert_allclose)

import pytest
from pytest import raises as assert_raises

from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
                          schur, rsf2csf, lu_solve, lu_factor, solve, diagsvd,
                          hessenberg, rq, eig_banded, eigvals_banded, eigh,
                          eigvalsh, qr_multiply, qz, orth, ordqz,
                          subspace_angles, hadamard, eigvalsh_tridiagonal,
                          eigh_tridiagonal, null_space, cdf2rdf, LinAlgError)

from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
                                 dsbevd, dsbevx, zhbevd, zhbevx)

from scipy.linalg.misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group

from numpy import (array, diag, ones, full, linalg, argsort, zeros, arange,
                   float32, complex64, ravel, sqrt, iscomplex, shape, sort,
                   sign, asarray, isfinite, ndarray, eye, dtype, triu, tril)

from numpy.random import seed, random

from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse.sputils import matrix

from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64


def _random_hermitian_matrix(n, posdef=False, dtype=float):
    "Generate random sym/hermitian array of the given size n"
    if dtype in COMPLEX_DTYPES:
        A = np.random.rand(n, n) + np.random.rand(n, n)*1.0j
        A = (A + A.conj().T)/2
    else:
        A = np.random.rand(n, n)
        A = (A + A.T)/2

    if posdef:
        A += sqrt(2*n)*np.eye(n)

    return A.astype(dtype)


REAL_DTYPES = [np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES


def clear_fuss(ar, fuss_binary_bits=7):
    """Clears trailing `fuss_binary_bits` of mantissa of a floating number"""
    x = np.asanyarray(ar)
    if np.iscomplexobj(x):
        return clear_fuss(x.real) + 1j * clear_fuss(x.imag)

    significant_binary_bits = np.finfo(x.dtype).nmant
    x_mant, x_exp = np.frexp(x)
    f = 2.0**(significant_binary_bits - fuss_binary_bits)
    x_mant *= f
    np.rint(x_mant, out=x_mant)
    x_mant /= f

    return np.ldexp(x_mant, x_exp)


# XXX: This function should be available through numpy.testing
def assert_dtype_equal(act, des):
    if isinstance(act, ndarray):
        act = act.dtype
    else:
        act = dtype(act)

    if isinstance(des, ndarray):
        des = des.dtype
    else:
        des = dtype(des)

    assert_(act == des,
            'dtype mismatch: "{}" (should be "{}")'.format(act, des))


# XXX: This function should not be defined here, but somewhere in
#      scipy.linalg namespace
def symrand(dim_or_eigv):
    """Return a random symmetric (Hermitian) matrix.

    If 'dim_or_eigv' is an integer N, return a NxN matrix, with eigenvalues
        uniformly distributed on (-1,1).

    If 'dim_or_eigv' is  1-D real array 'a', return a matrix whose
                      eigenvalues are 'a'.
    """
    if isinstance(dim_or_eigv, int):
        dim = dim_or_eigv
        d = random(dim)*2 - 1
    elif (isinstance(dim_or_eigv, ndarray) and
          len(dim_or_eigv.shape) == 1):
        dim = dim_or_eigv.shape[0]
        d = dim_or_eigv
    else:
        raise TypeError("input type not supported.")

    v = ortho_group.rvs(dim)
    h = v.T.conj() @ diag(d) @ v
    # to avoid roundoff errors, symmetrize the matrix (again)
    h = 0.5*(h.T+h)
    return h


def _complex_symrand(dim, dtype):
    a1, a2 = symrand(dim), symrand(dim)
    # add antisymmetric matrix as imag part
    a = a1 + 1j*(triu(a2)-tril(a2))
    return a.astype(dtype)


class TestEigVals:

    def test_simple(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        w = eigvals(a)
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        assert_array_almost_equal(w, exact_w)

    def test_simple_tr(self):
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]], 'd').T
        a = a.copy()
        a = a.T
        w = eigvals(a)
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        assert_array_almost_equal(w, exact_w)

    def test_simple_complex(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]]
        w = eigvals(a)
        exact_w = [(9+1j+sqrt(92+6j))/2,
                   0,
                   (9+1j-sqrt(92+6j))/2]
        assert_array_almost_equal(w, exact_w)

    def test_finite(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        w = eigvals(a, check_finite=False)
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        assert_array_almost_equal(w, exact_w)


class TestEig:

    def test_simple(self):
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        w, v = eig(a)
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        v0 = array([1, 1, (1+sqrt(93)/3)/2])
        v1 = array([3., 0, -1])
        v2 = array([1, 1, (1-sqrt(93)/3)/2])
        v0 = v0 / norm(v0)
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)
        assert_array_almost_equal(w, exact_w)
        assert_array_almost_equal(v0, v[:, 0]*sign(v[0, 0]))
        assert_array_almost_equal(v1, v[:, 1]*sign(v[0, 1]))
        assert_array_almost_equal(v2, v[:, 2]*sign(v[0, 2]))
        for i in range(3):
            assert_array_almost_equal(a @ v[:, i], w[i]*v[:, i])
        w, v = eig(a, left=1, right=0)
        for i in range(3):
            assert_array_almost_equal(a.T @ v[:, i], w[i]*v[:, i])

    def test_simple_complex_eig(self):
        a = array([[1, 2], [-2, 1]])
        w, vl, vr = eig(a, left=1, right=1)
        assert_array_almost_equal(w, array([1+2j, 1-2j]))
        for i in range(2):
            assert_array_almost_equal(a @ vr[:, i], w[i]*vr[:, i])
        for i in range(2):
            assert_array_almost_equal(a.conj().T @ vl[:, i],
                                      w[i].conj()*vl[:, i])

    def test_simple_complex(self):
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]])
        w, vl, vr = eig(a, left=1, right=1)
        for i in range(3):
            assert_array_almost_equal(a @ vr[:, i], w[i]*vr[:, i])
        for i in range(3):
            assert_array_almost_equal(a.conj().T @ vl[:, i],
                                      w[i].conj()*vl[:, i])

    def test_gh_3054(self):
        a = [[1]]
        b = [[0]]
        w, vr = eig(a, b, homogeneous_eigvals=True)
        assert_allclose(w[1, 0], 0)
        assert_(w[0, 0] != 0)
        assert_allclose(vr, 1)

        w, vr = eig(a, b)
        assert_equal(w, np.inf)
        assert_allclose(vr, 1)

    def _check_gen_eig(self, A, B):
        if B is not None:
            A, B = asarray(A), asarray(B)
            B0 = B
        else:
            A = asarray(A)
            B0 = B
            B = np.eye(*A.shape)
        msg = "\n%r\n%r" % (A, B)

        # Eigenvalues in homogeneous coordinates
        w, vr = eig(A, B0, homogeneous_eigvals=True)
        wt = eigvals(A, B0, homogeneous_eigvals=True)
        val1 = A @ vr * w[1, :]
        val2 = B @ vr * w[0, :]
        for i in range(val1.shape[1]):
            assert_allclose(val1[:, i], val2[:, i],
                            rtol=1e-13, atol=1e-13, err_msg=msg)

        if B0 is None:
            assert_allclose(w[1, :], 1)
            assert_allclose(wt[1, :], 1)

        perm = np.lexsort(w)
        permt = np.lexsort(wt)
        assert_allclose(w[:, perm], wt[:, permt], atol=1e-7, rtol=1e-7,
                        err_msg=msg)

        length = np.empty(len(vr))

        for i in range(len(vr)):
            length[i] = norm(vr[:, i])

        assert_allclose(length, np.ones(length.size), err_msg=msg,
                        atol=1e-7, rtol=1e-7)

        # Convert homogeneous coordinates
        beta_nonzero = (w[1, :] != 0)
        wh = w[0, beta_nonzero] / w[1, beta_nonzero]

        # Eigenvalues in standard coordinates
        w, vr = eig(A, B0)
        wt = eigvals(A, B0)
        val1 = A @ vr
        val2 = B @ vr * w
        res = val1 - val2
        for i in range(res.shape[1]):
            if np.all(isfinite(res[:, i])):
                assert_allclose(res[:, i], 0,
                                rtol=1e-13, atol=1e-13, err_msg=msg)

        w_fin = w[isfinite(w)]
        wt_fin = wt[isfinite(wt)]
        perm = argsort(clear_fuss(w_fin))
        permt = argsort(clear_fuss(wt_fin))
        assert_allclose(w[perm], wt[permt],
                        atol=1e-7, rtol=1e-7, err_msg=msg)

        length = np.empty(len(vr))
        for i in range(len(vr)):
            length[i] = norm(vr[:, i])
        assert_allclose(length, np.ones(length.size), err_msg=msg)

        # Compare homogeneous and nonhomogeneous versions
        assert_allclose(sort(wh), sort(w[np.isfinite(w)]))

    @pytest.mark.xfail(reason="See gh-2254")
    def test_singular(self):
        # Example taken from
        # https://web.archive.org/web/20040903121217/http://www.cs.umu.se/research/nla/singular_pairs/guptri/matlab.html
        A = array([[22, 34, 31, 31, 17],
                   [45, 45, 42, 19, 29],
                   [39, 47, 49, 26, 34],
                   [27, 31, 26, 21, 15],
                   [38, 44, 44, 24, 30]])
        B = array([[13, 26, 25, 17, 24],
                   [31, 46, 40, 26, 37],
                   [26, 40, 19, 25, 25],
                   [16, 25, 27, 14, 23],
                   [24, 35, 18, 21, 22]])

        with np.errstate(all='ignore'):
            self._check_gen_eig(A, B)

    def test_falker(self):
        # Test matrices giving some Nan generalized eigenvalues.
        M = diag(array(([1, 0, 3])))
        K = array(([2, -1, -1], [-1, 2, -1], [-1, -1, 2]))
        D = array(([1, -1, 0], [-1, 1, 0], [0, 0, 0]))
        Z = zeros((3, 3))
        I3 = eye(3)
        A = np.block([[I3, Z], [Z, -K]])
        B = np.block([[Z, I3], [M, D]])

        with np.errstate(all='ignore'):
            self._check_gen_eig(A, B)

    def test_bad_geneig(self):
        # Ticket #709 (strange return values from DGGEV)

        def matrices(omega):
            c1 = -9 + omega**2
            c2 = 2*omega
            A = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, c1, 0],
                 [0, 0, 0, c1]]
            B = [[0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [1, 0, 0, -c2],
                 [0, 1, c2, 0]]
            return A, B

        # With a buggy LAPACK, this can fail for different omega on different
        # machines -- so we need to test several values
        with np.errstate(all='ignore'):
            for k in range(100):
                A, B = matrices(omega=k*5./100)
                self._check_gen_eig(A, B)

    def test_make_eigvals(self):
        # Step through all paths in _make_eigvals
        seed(1234)
        # Real eigenvalues
        A = symrand(3)
        self._check_gen_eig(A, None)
        B = symrand(3)
        self._check_gen_eig(A, B)
        # Complex eigenvalues
        A = random((3, 3)) + 1j*random((3, 3))
        self._check_gen_eig(A, None)
        B = random((3, 3)) + 1j*random((3, 3))
        self._check_gen_eig(A, B)

    def test_check_finite(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        w, v = eig(a, check_finite=False)
        exact_w = [(9+sqrt(93))/2, 0, (9-sqrt(93))/2]
        v0 = array([1, 1, (1+sqrt(93)/3)/2])
        v1 = array([3., 0, -1])
        v2 = array([1, 1, (1-sqrt(93)/3)/2])
        v0 = v0 / norm(v0)
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)
        assert_array_almost_equal(w, exact_w)
        assert_array_almost_equal(v0, v[:, 0]*sign(v[0, 0]))
        assert_array_almost_equal(v1, v[:, 1]*sign(v[0, 1]))
        assert_array_almost_equal(v2, v[:, 2]*sign(v[0, 2]))
        for i in range(3):
            assert_array_almost_equal(a @ v[:, i], w[i]*v[:, i])

    def test_not_square_error(self):
        """Check that passing a non-square array raises a ValueError."""
        A = np.arange(6).reshape(3, 2)
        assert_raises(ValueError, eig, A)

    def test_shape_mismatch(self):
        """Check that passing arrays of with different shapes
        raises a ValueError."""
        A = eye(2)
        B = np.arange(9.0).reshape(3, 3)
        assert_raises(ValueError, eig, A, B)
        assert_raises(ValueError, eig, B, A)


class TestEigBanded:
    def setup_method(self):
        self.create_bandmat()

    def create_bandmat(self):
        """Create the full matrix `self.fullmat` and
           the corresponding band matrix `self.bandmat`."""
        N = 10
        self.KL = 2   # number of subdiagonals (below the diagonal)
        self.KU = 2   # number of superdiagonals (above the diagonal)

        # symmetric band matrix
        self.sym_mat = (diag(full(N, 1.0))
                        + diag(full(N-1, -1.0), -1) + diag(full(N-1, -1.0), 1)
                        + diag(full(N-2, -2.0), -2) + diag(full(N-2, -2.0), 2))

        # hermitian band matrix
        self.herm_mat = (diag(full(N, -1.0))
                         + 1j*diag(full(N-1, 1.0), -1)
                         - 1j*diag(full(N-1, 1.0), 1)
                         + diag(full(N-2, -2.0), -2)
                         + diag(full(N-2, -2.0), 2))

        # general real band matrix
        self.real_mat = (diag(full(N, 1.0))
                         + diag(full(N-1, -1.0), -1) + diag(full(N-1, -3.0), 1)
                         + diag(full(N-2, 2.0), -2) + diag(full(N-2, -2.0), 2))

        # general complex band matrix
        self.comp_mat = (1j*diag(full(N, 1.0))
                         + diag(full(N-1, -1.0), -1)
                         + 1j*diag(full(N-1, -3.0), 1)
                         + diag(full(N-2, 2.0), -2)
                         + diag(full(N-2, -2.0), 2))

        # Eigenvalues and -vectors from linalg.eig
        ew, ev = linalg.eig(self.sym_mat)
        ew = ew.real
        args = argsort(ew)
        self.w_sym_lin = ew[args]
        self.evec_sym_lin = ev[:, args]

        ew, ev = linalg.eig(self.herm_mat)
        ew = ew.real
        args = argsort(ew)
        self.w_herm_lin = ew[args]
        self.evec_herm_lin = ev[:, args]

        # Extract upper bands from symmetric and hermitian band matrices
        # (for use in dsbevd, dsbevx, zhbevd, zhbevx
        #  and their single precision versions)
        LDAB = self.KU + 1
        self.bandmat_sym = zeros((LDAB, N), dtype=float)
        self.bandmat_herm = zeros((LDAB, N), dtype=complex)
        for i in range(LDAB):
            self.bandmat_sym[LDAB-i-1, i:N] = diag(self.sym_mat, i)
            self.bandmat_herm[LDAB-i-1, i:N] = diag(self.herm_mat, i)

        # Extract bands from general real and complex band matrix
        # (for use in dgbtrf, dgbtrs and their single precision versions)
        LDAB = 2*self.KL + self.KU + 1
        self.bandmat_real = zeros((LDAB, N), dtype=float)
        self.bandmat_real[2*self.KL, :] = diag(self.real_mat)  # diagonal
        for i in range(self.KL):
            # superdiagonals
            self.bandmat_real[2*self.KL-1-i, i+1:N] = diag(self.real_mat, i+1)
            # subdiagonals
            self.bandmat_real[2*self.KL+1+i, 0:N-1-i] = diag(self.real_mat,
                                                             -i-1)

        self.bandmat_comp = zeros((LDAB, N), dtype=complex)
        self.bandmat_comp[2*self.KL, :] = diag(self.comp_mat)  # diagonal
        for i in range(self.KL):
            # superdiagonals
            self.bandmat_comp[2*self.KL-1-i, i+1:N] = diag(self.comp_mat, i+1)
            # subdiagonals
            self.bandmat_comp[2*self.KL+1+i, 0:N-1-i] = diag(self.comp_mat,
                                                             -i-1)

        # absolute value for linear equation system A*x = b
        self.b = 1.0*arange(N)
        self.bc = self.b * (1 + 1j)

    #####################################################################

    def test_dsbev(self):
        """Compare dsbev eigenvalues and eigenvectors with
           the result of linalg.eig."""
        w, evec, info = dsbev(self.bandmat_sym, compute_v=1)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_dsbevd(self):
        """Compare dsbevd eigenvalues and eigenvectors with
           the result of linalg.eig."""
        w, evec, info = dsbevd(self.bandmat_sym, compute_v=1)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_dsbevx(self):
        """Compare dsbevx eigenvalues and eigenvectors
           with the result of linalg.eig."""
        N, N = shape(self.sym_mat)
        # Achtung: Argumente 0.0,0.0,range?
        w, evec, num, ifail, info = dsbevx(self.bandmat_sym, 0.0, 0.0, 1, N,
                                           compute_v=1, range=2)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_sym_lin))

    def test_zhbevd(self):
        """Compare zhbevd eigenvalues and eigenvectors
           with the result of linalg.eig."""
        w, evec, info = zhbevd(self.bandmat_herm, compute_v=1)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_herm_lin))

    def test_zhbevx(self):
        """Compare zhbevx eigenvalues and eigenvectors
           with the result of linalg.eig."""
        N, N = shape(self.herm_mat)
        # Achtung: Argumente 0.0,0.0,range?
        w, evec, num, ifail, info = zhbevx(self.bandmat_herm, 0.0, 0.0, 1, N,
                                           compute_v=1, range=2)
        evec_ = evec[:, argsort(w)]
        assert_array_almost_equal(sort(w), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_), abs(self.evec_herm_lin))

    def test_eigvals_banded(self):
        """Compare eigenvalues of eigvals_banded with those of linalg.eig."""
        w_sym = eigvals_banded(self.bandmat_sym)
        w_sym = w_sym.real
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)

        w_herm = eigvals_banded(self.bandmat_herm)
        w_herm = w_herm.real
        assert_array_almost_equal(sort(w_herm), self.w_herm_lin)

        # extracting eigenvalues with respect to an index range
        ind1 = 2
        ind2 = np.longlong(6)
        w_sym_ind = eigvals_banded(self.bandmat_sym,
                                   select='i', select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_sym_ind),
                                  self.w_sym_lin[ind1:ind2+1])
        w_herm_ind = eigvals_banded(self.bandmat_herm,
                                    select='i', select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_herm_ind),
                                  self.w_herm_lin[ind1:ind2+1])

        # extracting eigenvalues with respect to a value range
        v_lower = self.w_sym_lin[ind1] - 1.0e-5
        v_upper = self.w_sym_lin[ind2] + 1.0e-5
        w_sym_val = eigvals_banded(self.bandmat_sym,
                                   select='v', select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_sym_val),
                                  self.w_sym_lin[ind1:ind2+1])

        v_lower = self.w_herm_lin[ind1] - 1.0e-5
        v_upper = self.w_herm_lin[ind2] + 1.0e-5
        w_herm_val = eigvals_banded(self.bandmat_herm,
                                    select='v',
                                    select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_herm_val),
                                  self.w_herm_lin[ind1:ind2+1])

        w_sym = eigvals_banded(self.bandmat_sym, check_finite=False)
        w_sym = w_sym.real
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)

    def test_eig_banded(self):
        """Compare eigenvalues and eigenvectors of eig_banded
           with those of linalg.eig. """
        w_sym, evec_sym = eig_banded(self.bandmat_sym)
        evec_sym_ = evec_sym[:, argsort(w_sym.real)]
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_sym_), abs(self.evec_sym_lin))

        w_herm, evec_herm = eig_banded(self.bandmat_herm)
        evec_herm_ = evec_herm[:, argsort(w_herm.real)]
        assert_array_almost_equal(sort(w_herm), self.w_herm_lin)
        assert_array_almost_equal(abs(evec_herm_), abs(self.evec_herm_lin))

        # extracting eigenvalues with respect to an index range
        ind1 = 2
        ind2 = 6
        w_sym_ind, evec_sym_ind = eig_banded(self.bandmat_sym,
                                             select='i',
                                             select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_sym_ind),
                                  self.w_sym_lin[ind1:ind2+1])
        assert_array_almost_equal(abs(evec_sym_ind),
                                  abs(self.evec_sym_lin[:, ind1:ind2+1]))

        w_herm_ind, evec_herm_ind = eig_banded(self.bandmat_herm,
                                               select='i',
                                               select_range=(ind1, ind2))
        assert_array_almost_equal(sort(w_herm_ind),
                                  self.w_herm_lin[ind1:ind2+1])
        assert_array_almost_equal(abs(evec_herm_ind),
                                  abs(self.evec_herm_lin[:, ind1:ind2+1]))

        # extracting eigenvalues with respect to a value range
        v_lower = self.w_sym_lin[ind1] - 1.0e-5
        v_upper = self.w_sym_lin[ind2] + 1.0e-5
        w_sym_val, evec_sym_val = eig_banded(self.bandmat_sym,
                                             select='v',
                                             select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_sym_val),
                                  self.w_sym_lin[ind1:ind2+1])
        assert_array_almost_equal(abs(evec_sym_val),
                                  abs(self.evec_sym_lin[:, ind1:ind2+1]))

        v_lower = self.w_herm_lin[ind1] - 1.0e-5
        v_upper = self.w_herm_lin[ind2] + 1.0e-5
        w_herm_val, evec_herm_val = eig_banded(self.bandmat_herm,
                                               select='v',
                                               select_range=(v_lower, v_upper))
        assert_array_almost_equal(sort(w_herm_val),
                                  self.w_herm_lin[ind1:ind2+1])
        assert_array_almost_equal(abs(evec_herm_val),
                                  abs(self.evec_herm_lin[:, ind1:ind2+1]))

        w_sym, evec_sym = eig_banded(self.bandmat_sym, check_finite=False)
        evec_sym_ = evec_sym[:, argsort(w_sym.real)]
        assert_array_almost_equal(sort(w_sym), self.w_sym_lin)
        assert_array_almost_equal(abs(evec_sym_), abs(self.evec_sym_lin))

    def test_dgbtrf(self):
        """Compare dgbtrf  LU factorisation with the LU factorisation result
           of linalg.lu."""
        M, N = shape(self.real_mat)
        lu_symm_band, ipiv, info = dgbtrf(self.bandmat_real, self.KL, self.KU)

        # extract matrix u from lu_symm_band
        u = diag(lu_symm_band[2*self.KL, :])
        for i in range(self.KL + self.KU):
            u += diag(lu_symm_band[2*self.KL-1-i, i+1:N], i+1)

        p_lin, l_lin, u_lin = lu(self.real_mat, permute_l=0)
        assert_array_almost_equal(u, u_lin)

    def test_zgbtrf(self):
        """Compare zgbtrf  LU factorisation with the LU factorisation result
           of linalg.lu."""
        M, N = shape(self.comp_mat)
        lu_symm_band, ipiv, info = zgbtrf(self.bandmat_comp, self.KL, self.KU)

        # extract matrix u from lu_symm_band
        u = diag(lu_symm_band[2*self.KL, :])
        for i in range(self.KL + self.KU):
            u += diag(lu_symm_band[2*self.KL-1-i, i+1:N], i+1)

        p_lin, l_lin, u_lin = lu(self.comp_mat, permute_l=0)
        assert_array_almost_equal(u, u_lin)

    def test_dgbtrs(self):
        """Compare dgbtrs  solutions for linear equation system  A*x = b
           with solutions of linalg.solve."""

        lu_symm_band, ipiv, info = dgbtrf(self.bandmat_real, self.KL, self.KU)
        y, info = dgbtrs(lu_symm_band, self.KL, self.KU, self.b, ipiv)

        y_lin = linalg.solve(self.real_mat, self.b)
        assert_array_almost_equal(y, y_lin)

    def test_zgbtrs(self):
        """Compare zgbtrs  solutions for linear equation system  A*x = b
           with solutions of linalg.solve."""

        lu_symm_band, ipiv, info = zgbtrf(self.bandmat_comp, self.KL, self.KU)
        y, info = zgbtrs(lu_symm_band, self.KL, self.KU, self.bc, ipiv)

        y_lin = linalg.solve(self.comp_mat, self.bc)
        assert_array_almost_equal(y, y_lin)


class TestEigTridiagonal:
    def setup_method(self):
        self.create_trimat()

    def create_trimat(self):
        """Create the full matrix `self.fullmat`, `self.d`, and `self.e`."""
        N = 10

        # symmetric band matrix
        self.d = full(N, 1.0)
        self.e = full(N-1, -1.0)
        self.full_mat = (diag(self.d) + diag(self.e, -1) + diag(self.e, 1))

        ew, ev = linalg.eig(self.full_mat)
        ew = ew.real
        args = argsort(ew)
        self.w = ew[args]
        self.evec = ev[:, args]

    def test_degenerate(self):
        """Test error conditions."""
        # Wrong sizes
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e[:-1])
        # Must be real
        assert_raises(TypeError, eigvalsh_tridiagonal, self.d, self.e * 1j)
        # Bad driver
        assert_raises(TypeError, eigvalsh_tridiagonal, self.d, self.e,
                      lapack_driver=1.)
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e,
                      lapack_driver='foo')
        # Bad bounds
        assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e,
                      select='i', select_range=(0, -1))

    def test_eigvalsh_tridiagonal(self):
        """Compare eigenvalues of eigvalsh_tridiagonal with those of eig."""
        # can't use ?STERF with subselection
        for driver in ('sterf', 'stev', 'stebz', 'stemr', 'auto'):
            w = eigvalsh_tridiagonal(self.d, self.e, lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w)

        for driver in ('sterf', 'stev'):
            assert_raises(ValueError, eigvalsh_tridiagonal, self.d, self.e,
                          lapack_driver='stev', select='i',
                          select_range=(0, 1))
        for driver in ('stebz', 'stemr', 'auto'):
            # extracting eigenvalues with respect to the full index range
            w_ind = eigvalsh_tridiagonal(
                self.d, self.e, select='i', select_range=(0, len(self.d)-1),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w_ind), self.w)

            # extracting eigenvalues with respect to an index range
            ind1 = 2
            ind2 = 6
            w_ind = eigvalsh_tridiagonal(
                self.d, self.e, select='i', select_range=(ind1, ind2),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w_ind), self.w[ind1:ind2+1])

            # extracting eigenvalues with respect to a value range
            v_lower = self.w[ind1] - 1.0e-5
            v_upper = self.w[ind2] + 1.0e-5
            w_val = eigvalsh_tridiagonal(
                self.d, self.e, select='v', select_range=(v_lower, v_upper),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w_val), self.w[ind1:ind2+1])

    def test_eigh_tridiagonal(self):
        """Compare eigenvalues and eigenvectors of eigh_tridiagonal
           with those of eig. """
        # can't use ?STERF when eigenvectors are requested
        assert_raises(ValueError, eigh_tridiagonal, self.d, self.e,
                      lapack_driver='sterf')
        for driver in ('stebz', 'stev', 'stemr', 'auto'):
            w, evec = eigh_tridiagonal(self.d, self.e, lapack_driver=driver)
            evec_ = evec[:, argsort(w)]
            assert_array_almost_equal(sort(w), self.w)
            assert_array_almost_equal(abs(evec_), abs(self.evec))

        assert_raises(ValueError, eigh_tridiagonal, self.d, self.e,
                      lapack_driver='stev', select='i', select_range=(0, 1))
        for driver in ('stebz', 'stemr', 'auto'):
            # extracting eigenvalues with respect to an index range
            ind1 = 0
            ind2 = len(self.d)-1
            w, evec = eigh_tridiagonal(
                self.d, self.e, select='i', select_range=(ind1, ind2),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w)
            assert_array_almost_equal(abs(evec), abs(self.evec))
            ind1 = 2
            ind2 = 6
            w, evec = eigh_tridiagonal(
                self.d, self.e, select='i', select_range=(ind1, ind2),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w[ind1:ind2+1])
            assert_array_almost_equal(abs(evec),
                                      abs(self.evec[:, ind1:ind2+1]))

            # extracting eigenvalues with respect to a value range
            v_lower = self.w[ind1] - 1.0e-5
            v_upper = self.w[ind2] + 1.0e-5
            w, evec = eigh_tridiagonal(
                self.d, self.e, select='v', select_range=(v_lower, v_upper),
                lapack_driver=driver)
            assert_array_almost_equal(sort(w), self.w[ind1:ind2+1])
            assert_array_almost_equal(abs(evec),
                                      abs(self.evec[:, ind1:ind2+1]))


class TestEigh:
    def setup_class(self):
        seed(1234)

    def test_wrong_inputs(self):
        # Nonsquare a
        assert_raises(ValueError, eigh, np.ones([1, 2]))
        # Nonsquare b
        assert_raises(ValueError, eigh, np.ones([2, 2]), np.ones([2, 1]))
        # Incompatible a, b sizes
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([2, 2]))
        # Wrong type parameter for generalized problem
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      type=4)
        # Both value and index subsets requested
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_value=[1, 2], eigvals=[2, 4])
        # Invalid upper index spec
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      eigvals=[0, 4])
        # Invalid lower index
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      eigvals=[-2, 2])
        # Invalid index spec #2
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      eigvals=[2, 0])
        # Invalid value spec
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      subset_by_value=[2, 0])
        # Invalid driver name
        assert_raises(ValueError, eigh, np.ones([2, 2]), driver='wrong')
        # Generalized driver selection without b
        assert_raises(ValueError, eigh, np.ones([3, 3]), None, driver='gvx')
        # Standard driver with b
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      driver='evr', turbo=False)
        # Subset request from invalid driver
        assert_raises(ValueError, eigh, np.ones([3, 3]), np.ones([3, 3]),
                      driver='gvd', eigvals=[1, 2], turbo=False)

    def test_nonpositive_b(self):
        assert_raises(LinAlgError, eigh, np.ones([3, 3]), np.ones([3, 3]))

    # index based subsets are done in the legacy test_eigh()
    def test_value_subsets(self):
        for ind, dt in enumerate(DTYPES):

            a = _random_hermitian_matrix(20, dtype=dt)
            w, v = eigh(a, subset_by_value=[-2, 2])
            assert_equal(v.shape[1], len(w))
            assert all((w > -2) & (w < 2))

            b = _random_hermitian_matrix(20, posdef=True, dtype=dt)
            w, v = eigh(a, b, subset_by_value=[-2, 2])
            assert_equal(v.shape[1], len(w))
            assert all((w > -2) & (w < 2))

    def test_eigh_integer(self):
        a = array([[1, 2], [2, 7]])
        b = array([[3, 1], [1, 5]])
        w, z = eigh(a)
        w, z = eigh(a, b)

    def test_eigh_of_sparse(self):
        # This tests the rejection of inputs that eigh cannot currently handle.
        import scipy.sparse
        a = scipy.sparse.identity(2).tocsc()
        b = np.atleast_2d(a)
        assert_raises(ValueError, eigh, a)
        assert_raises(ValueError, eigh, b)

    @pytest.mark.parametrize('dtype_', DTYPES)
    @pytest.mark.parametrize('driver', ("ev", "evd", "evr", "evx"))
    def test_various_drivers_standard(self, driver, dtype_):
        a = _random_hermitian_matrix(n=20, dtype=dtype_)
        w, v = eigh(a, driver=driver)
        assert_allclose(a @ v - (v * w), 0.,
                        atol=1000*np.finfo(dtype_).eps,
                        rtol=0.)

    @pytest.mark.parametrize('type', (1, 2, 3))
    @pytest.mark.parametrize('driver', ("gv", "gvd", "gvx"))
    def test_various_drivers_generalized(self, driver, type):
        atol = np.spacing(5000.)
        a = _random_hermitian_matrix(20)
        b = _random_hermitian_matrix(20, posdef=True)
        w, v = eigh(a=a, b=b, driver=driver, type=type)
        if type == 1:
            assert_allclose(a @ v - w*(b @ v), 0., atol=atol, rtol=0.)
        elif type == 2:
            assert_allclose(a @ b @ v - v * w, 0., atol=atol, rtol=0.)
        else:
            assert_allclose(b @ a @ v - v * w, 0., atol=atol, rtol=0.)

    # Old eigh tests kept for backwards compatibility
    @pytest.mark.parametrize('eigvals', (None, (2, 4)))
    @pytest.mark.parametrize('turbo', (True, False))
    @pytest.mark.parametrize('lower', (True, False))
    @pytest.mark.parametrize('overwrite', (True, False))
    @pytest.mark.parametrize('dtype_', ('f', 'd', 'F', 'D'))
    @pytest.mark.parametrize('dim', (6,))
    def test_eigh(self, dim, dtype_, overwrite, lower, turbo, eigvals):
        atol = 1e-11 if dtype_ in ('dD') else 1e-4
        a = _random_hermitian_matrix(n=dim, dtype=dtype_)
        w, z = eigh(a, overwrite_a=overwrite, lower=lower, eigvals=eigvals)
        assert_dtype_equal(z.dtype, dtype_)
        w = w.astype(dtype_)
        diag_ = diag(z.T.conj() @ a @ z).real
        assert_allclose(diag_, w, rtol=0., atol=atol)

        a = _random_hermitian_matrix(n=dim, dtype=dtype_)
        b = _random_hermitian_matrix(n=dim, dtype=dtype_, posdef=True)
        w, z = eigh(a, b, overwrite_a=overwrite, lower=lower,
                    overwrite_b=overwrite, turbo=turbo, eigvals=eigvals)
        assert_dtype_equal(z.dtype, dtype_)
        w = w.astype(dtype_)
        diag1_ = diag(z.T.conj() @ a @ z).real
        assert_allclose(diag1_, w, rtol=0., atol=atol)
        diag2_ = diag(z.T.conj() @ b @ z).real
        assert_allclose(diag2_, ones(diag2_.shape[0]), rtol=0., atol=atol)

    def test_eigvalsh_new_args(self):
        a = _random_hermitian_matrix(5)
        w = eigvalsh(a, eigvals=[1, 2])
        assert_equal(len(w), 2)

        w2 = eigvalsh(a, subset_by_index=[1, 2])
        assert_equal(len(w2), 2)
        assert_allclose(w, w2)

        b = np.diag([1, 1.2, 1.3, 1.5, 2])
        w3 = eigvalsh(b, subset_by_value=[1, 1.4])
        assert_equal(len(w3), 2)
        assert_allclose(w3, np.array([1.2, 1.3]))


class TestLU:
    def setup_method(self):
        self.a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        self.ca = array([[1, 2, 3], [1, 2, 3], [2, 5j, 6]])
        # Those matrices are more robust to detect problems in permutation
        # matrices than the ones above
        self.b = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.cb = array([[1j, 2j, 3j], [4j, 5j, 6j], [7j, 8j, 9j]])

        # Reectangular matrices
        self.hrect = array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]])
        self.chrect = 1.j * array([[1, 2, 3, 4],
                                   [5, 6, 7, 8],
                                   [9, 10, 12, 12]])

        self.vrect = array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 12]])
        self.cvrect = 1.j * array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 12, 12]])

        # Medium sizes matrices
        self.med = random((30, 40))
        self.cmed = random((30, 40)) + 1.j * random((30, 40))

    def _test_common(self, data):
        p, l, u = lu(data)
        assert_array_almost_equal(p @ l @ u, data)
        pl, u = lu(data, permute_l=1)
        assert_array_almost_equal(pl @ u, data)

    # Simple tests
    def test_simple(self):
        self._test_common(self.a)

    def test_simple_complex(self):
        self._test_common(self.ca)

    def test_simple2(self):
        self._test_common(self.b)

    def test_simple2_complex(self):
        self._test_common(self.cb)

    # rectangular matrices tests
    def test_hrectangular(self):
        self._test_common(self.hrect)

    def test_vrectangular(self):
        self._test_common(self.vrect)

    def test_hrectangular_complex(self):
        self._test_common(self.chrect)

    def test_vrectangular_complex(self):
        self._test_common(self.cvrect)

    # Bigger matrices
    def test_medium1(self):
        """Check lu decomposition on medium size, rectangular matrix."""
        self._test_common(self.med)

    def test_medium1_complex(self):
        """Check lu decomposition on medium size, rectangular matrix."""
        self._test_common(self.cmed)

    def test_check_finite(self):
        p, l, u = lu(self.a, check_finite=False)
        assert_array_almost_equal(p @ l @ u, self.a)

    def test_simple_known(self):
        # Ticket #1458
        for order in ['C', 'F']:
            A = np.array([[2, 1], [0, 1.]], order=order)
            LU, P = lu_factor(A)
            assert_array_almost_equal(LU, np.array([[2, 1], [0, 1]]))
            assert_array_equal(P, np.array([0, 1]))


class TestLUSingle(TestLU):
    """LU testers for single precision, real and double"""

    def setup_method(self):
        TestLU.setup_method(self)

        self.a = self.a.astype(float32)
        self.ca = self.ca.astype(complex64)
        self.b = self.b.astype(float32)
        self.cb = self.cb.astype(complex64)

        self.hrect = self.hrect.astype(float32)
        self.chrect = self.hrect.astype(complex64)

        self.vrect = self.vrect.astype(float32)
        self.cvrect = self.vrect.astype(complex64)

        self.med = self.vrect.astype(float32)
        self.cmed = self.vrect.astype(complex64)


class TestLUSolve:
    def setup_method(self):
        seed(1234)

    def test_lu(self):
        a0 = random((10, 10))
        b = random((10,))

        for order in ['C', 'F']:
            a = np.array(a0, order=order)
            x1 = solve(a, b)
            lu_a = lu_factor(a)
            x2 = lu_solve(lu_a, b)
            assert_array_almost_equal(x1, x2)

    def test_check_finite(self):
        a = random((10, 10))
        b = random((10,))
        x1 = solve(a, b)
        lu_a = lu_factor(a, check_finite=False)
        x2 = lu_solve(lu_a, b, check_finite=False)
        assert_array_almost_equal(x1, x2)


class TestSVD_GESDD:
    def setup_method(self):
        self.lapack_driver = 'gesdd'
        seed(1234)

    def test_degenerate(self):
        assert_raises(TypeError, svd, [[1.]], lapack_driver=1.)
        assert_raises(ValueError, svd, [[1.]], lapack_driver='foo')

    def test_simple(self):
        a = [[1, 2, 3], [1, 20, 3], [2, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices,
                           lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(3))
            assert_array_almost_equal(vh.T @ vh, eye(3))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_singular(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices,
                           lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(3))
            assert_array_almost_equal(vh.T @ vh, eye(3))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_underdet(self):
        a = [[1, 2, 3], [4, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices,
                           lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(u.shape[0]))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_overdet(self):
        a = [[1, 2], [4, 5], [3, 4]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices,
                           lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(u.shape[1]))
            assert_array_almost_equal(vh.T @ vh, eye(2))
            sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_random(self):
        n = 20
        m = 15
        for i in range(3):
            for a in [random([n, m]), random([m, n])]:
                for full_matrices in (True, False):
                    u, s, vh = svd(a, full_matrices=full_matrices,
                                   lapack_driver=self.lapack_driver)
                    assert_array_almost_equal(u.T @ u, eye(u.shape[1]))
                    assert_array_almost_equal(vh @ vh.T, eye(vh.shape[0]))
                    sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
                    for i in range(len(s)):
                        sigma[i, i] = s[i]
                    assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_complex(self):
        a = [[1, 2, 3], [1, 2j, 3], [2, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices,
                           lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.conj().T @ u, eye(u.shape[1]))
            assert_array_almost_equal(vh.conj().T @ vh, eye(vh.shape[0]))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_random_complex(self):
        n = 20
        m = 15
        for i in range(3):
            for full_matrices in (True, False):
                for a in [random([n, m]), random([m, n])]:
                    a = a + 1j*random(list(a.shape))
                    u, s, vh = svd(a, full_matrices=full_matrices,
                                   lapack_driver=self.lapack_driver)
                    assert_array_almost_equal(u.conj().T @ u,
                                              eye(u.shape[1]))
                    # This fails when [m,n]
                    # assert_array_almost_equal(vh.conj().T @ vh,
                    #                        eye(len(vh),dtype=vh.dtype.char))
                    sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
                    for i in range(len(s)):
                        sigma[i, i] = s[i]
                    assert_array_almost_equal(u @ sigma @ vh, a)

    def test_crash_1580(self):
        sizes = [(13, 23), (30, 50), (60, 100)]
        np.random.seed(1234)
        for sz in sizes:
            for dt in [np.float32, np.float64, np.complex64, np.complex128]:
                a = np.random.rand(*sz).astype(dt)
                # should not crash
                svd(a, lapack_driver=self.lapack_driver)

    def test_check_finite(self):
        a = [[1, 2, 3], [1, 20, 3], [2, 5, 6]]
        u, s, vh = svd(a, check_finite=False, lapack_driver=self.lapack_driver)
        assert_array_almost_equal(u.T @ u, eye(3))
        assert_array_almost_equal(vh.T @ vh, eye(3))
        sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
        for i in range(len(s)):
            sigma[i, i] = s[i]
        assert_array_almost_equal(u @ sigma @ vh, a)

    def test_gh_5039(self):
        # This is a smoke test for https://github.com/scipy/scipy/issues/5039
        #
        # The following is reported to raise "ValueError: On entry to DGESDD
        # parameter number 12 had an illegal value".
        # `interp1d([1,2,3,4], [1,2,3,4], kind='cubic')`
        # This is reported to only show up on LAPACK 3.0.3.
        #
        # The matrix below is taken from the call to
        # `B = _fitpack._bsplmat(order, xk)` in interpolate._find_smoothest
        b = np.array(
            [[0.16666667, 0.66666667, 0.16666667, 0., 0., 0.],
             [0., 0.16666667, 0.66666667, 0.16666667, 0., 0.],
             [0., 0., 0.16666667, 0.66666667, 0.16666667, 0.],
             [0., 0., 0., 0.16666667, 0.66666667, 0.16666667]])
        svd(b, lapack_driver=self.lapack_driver)

    @pytest.mark.skipif(not HAS_ILP64, reason="64-bit LAPACK required")
    @pytest.mark.slow
    def test_large_matrix(self):
        check_free_memory(free_mb=17000)
        A = np.zeros([1, 2**31], dtype=np.float32)
        A[0, -1] = 1
        u, s, vh = svd(A, full_matrices=False)
        assert_allclose(s[0], 1.0)
        assert_allclose(u[0, 0] * vh[0, -1], 1.0)


class TestSVD_GESVD(TestSVD_GESDD):
    def setup_method(self):
        self.lapack_driver = 'gesvd'
        seed(1234)


class TestSVDVals:

    def test_empty(self):
        for a in [[]], np.empty((2, 0)), np.ones((0, 3)):
            s = svdvals(a)
            assert_equal(s, np.empty(0))

    def test_simple(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        s = svdvals(a)
        assert_(len(s) == 3)
        assert_(s[0] >= s[1] >= s[2])

    def test_simple_underdet(self):
        a = [[1, 2, 3], [4, 5, 6]]
        s = svdvals(a)
        assert_(len(s) == 2)
        assert_(s[0] >= s[1])

    def test_simple_overdet(self):
        a = [[1, 2], [4, 5], [3, 4]]
        s = svdvals(a)
        assert_(len(s) == 2)
        assert_(s[0] >= s[1])

    def test_simple_complex(self):
        a = [[1, 2, 3], [1, 20, 3j], [2, 5, 6]]
        s = svdvals(a)
        assert_(len(s) == 3)
        assert_(s[0] >= s[1] >= s[2])

    def test_simple_underdet_complex(self):
        a = [[1, 2, 3], [4, 5j, 6]]
        s = svdvals(a)
        assert_(len(s) == 2)
        assert_(s[0] >= s[1])

    def test_simple_overdet_complex(self):
        a = [[1, 2], [4, 5], [3j, 4]]
        s = svdvals(a)
        assert_(len(s) == 2)
        assert_(s[0] >= s[1])

    def test_check_finite(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        s = svdvals(a, check_finite=False)
        assert_(len(s) == 3)
        assert_(s[0] >= s[1] >= s[2])

    @pytest.mark.slow
    def test_crash_2609(self):
        np.random.seed(1234)
        a = np.random.rand(1500, 2800)
        # Shouldn't crash:
        svdvals(a)


class TestDiagSVD:

    def test_simple(self):
        assert_array_almost_equal(diagsvd([1, 0, 0], 3, 3),
                                  [[1, 0, 0], [0, 0, 0], [0, 0, 0]])


class TestQR:

    def setup_method(self):
        seed(1234)

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_left(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        c = [1, 2, 3]
        qc, r2 = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        qc, r2 = qr_multiply(a, eye(3), "left")
        assert_array_almost_equal(q, qc)

    def test_simple_right(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a)
        c = [1, 2, 3]
        qc, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, qc)
        assert_array_almost_equal(r, r2)
        qc, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(q, qc)

    def test_simple_pivoting(self):
        a = np.asarray([[8, 2, 3], [2, 9, 3], [5, 3, 6]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3]
        qc, r, jpvt = qr_multiply(a, c, "left", True)
        assert_array_almost_equal(q @ c, qc)

    def test_simple_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3]
        qc, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, qc)

    def test_simple_trap(self):
        a = [[8, 2, 3], [2, 9, 3]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)

    def test_simple_trap_pivoting(self):
        a = np.asarray([[8, 2, 3], [2, 9, 3]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall(self):
        # full version
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_tall_pivoting(self):
        # full version pivoting
        a = np.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_e(self):
        # economy version
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a, mode='economic')
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (3, 2))
        assert_equal(r.shape, (2, 2))

    def test_simple_tall_e_pivoting(self):
        # economy version pivoting
        a = np.asarray([[8, 2], [2, 9], [5, 3]])
        q, r, p = qr(a, pivoting=True, mode='economic')
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p], mode='economic')
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_tall_left(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a, mode="economic")
        c = [1, 2]
        qc, r2 = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        c = array([1, 2, 0])
        qc, r2 = qr_multiply(a, c, "left", overwrite_c=True)
        assert_array_almost_equal(q @ c[:2], qc)
        qc, r = qr_multiply(a, eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_tall_left_pivoting(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, jpvt = qr(a, mode="economic", pivoting=True)
        c = [1, 2]
        qc, r, kpvt = qr_multiply(a, c, "left", True)
        assert_array_equal(jpvt, kpvt)
        assert_array_almost_equal(q @ c, qc)
        qc, r, jpvt = qr_multiply(a, eye(2), "left", True)
        assert_array_almost_equal(qc, q)

    def test_simple_tall_right(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r = qr(a, mode="economic")
        c = [1, 2, 3]
        cq, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(cq, q)

    def test_simple_tall_right_pivoting(self):
        a = [[8, 2], [2, 9], [5, 3]]
        q, r, jpvt = qr(a, pivoting=True, mode="economic")
        c = [1, 2, 3]
        cq, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt = qr_multiply(a, eye(3), pivoting=True)
        assert_array_almost_equal(cq, q)

    def test_simple_fat(self):
        # full version
        a = [[8, 2, 5], [2, 9, 3]]
        q, r = qr(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_pivoting(self):
        # full version pivoting
        a = np.asarray([[8, 2, 5], [2, 9, 3]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_e(self):
        # economy version
        a = [[8, 2, 3], [2, 9, 5]]
        q, r = qr(a, mode='economic')
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a)
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))

    def test_simple_fat_e_pivoting(self):
        # economy version pivoting
        a = np.asarray([[8, 2, 3], [2, 9, 5]])
        q, r, p = qr(a, pivoting=True, mode='economic')
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(q @ r, a[:, p])
        assert_equal(q.shape, (2, 2))
        assert_equal(r.shape, (2, 3))
        q2, r2 = qr(a[:, p], mode='economic')
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_fat_left(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r = qr(a, mode="economic")
        c = [1, 2]
        qc, r2 = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        qc, r = qr_multiply(a, eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_fat_left_pivoting(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, jpvt = qr(a, mode="economic", pivoting=True)
        c = [1, 2]
        qc, r, jpvt = qr_multiply(a, c, "left", True)
        assert_array_almost_equal(q @ c, qc)
        qc, r, jpvt = qr_multiply(a, eye(2), "left", True)
        assert_array_almost_equal(qc, q)

    def test_simple_fat_right(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r = qr(a, mode="economic")
        c = [1, 2]
        cq, r2 = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, cq)
        assert_array_almost_equal(r, r2)
        cq, r = qr_multiply(a, eye(2))
        assert_array_almost_equal(cq, q)

    def test_simple_fat_right_pivoting(self):
        a = [[8, 2, 3], [2, 9, 5]]
        q, r, jpvt = qr(a, pivoting=True, mode="economic")
        c = [1, 2]
        cq, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, cq)
        cq, r, jpvt = qr_multiply(a, eye(2), pivoting=True)
        assert_array_almost_equal(cq, q)

    def test_simple_complex(self):
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        q, r = qr(a)
        assert_array_almost_equal(q.conj().T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_simple_complex_left(self):
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        q, r = qr(a)
        c = [1, 2, 3+4j]
        qc, r = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        qc, r = qr_multiply(a, eye(3), "left")
        assert_array_almost_equal(q, qc)

    def test_simple_complex_right(self):
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        q, r = qr(a)
        c = [1, 2, 3+4j]
        qc, r = qr_multiply(a, c)
        assert_array_almost_equal(c @ q, qc)
        qc, r = qr_multiply(a, eye(3))
        assert_array_almost_equal(q, qc)

    def test_simple_tall_complex_left(self):
        a = [[8, 2+3j], [2, 9], [5+7j, 3]]
        q, r = qr(a, mode="economic")
        c = [1, 2+2j]
        qc, r2 = qr_multiply(a, c, "left")
        assert_array_almost_equal(q @ c, qc)
        assert_array_almost_equal(r, r2)
        c = array([1, 2, 0])
        qc, r2 = qr_multiply(a, c, "left", overwrite_c=True)
        assert_array_almost_equal(q @ c[:2], qc)
        qc, r = qr_multiply(a, eye(2), "left")
        assert_array_almost_equal(qc, q)

    def test_simple_complex_left_conjugate(self):
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        q, r = qr(a)
        c = [1, 2, 3+4j]
        qc, r = qr_multiply(a, c, "left", conjugate=True)
        assert_array_almost_equal(q.conj() @ c, qc)

    def test_simple_complex_tall_left_conjugate(self):
        a = [[3, 3+4j], [5, 2+2j], [3, 2]]
        q, r = qr(a, mode='economic')
        c = [1, 3+4j]
        qc, r = qr_multiply(a, c, "left", conjugate=True)
        assert_array_almost_equal(q.conj() @ c, qc)

    def test_simple_complex_right_conjugate(self):
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        q, r = qr(a)
        c = np.array([1, 2, 3+4j])
        qc, r = qr_multiply(a, c, conjugate=True)
        assert_array_almost_equal(c @ q.conj(), qc)

    def test_simple_complex_pivoting(self):
        a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
        q, r, p = qr(a, pivoting=True)
        d = abs(diag(r))
        assert_(np.all(d[1:] <= d[:-1]))
        assert_array_almost_equal(q.conj().T @ q, eye(3))
        assert_array_almost_equal(q @ r, a[:, p])
        q2, r2 = qr(a[:, p])
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(r, r2)

    def test_simple_complex_left_pivoting(self):
        a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3+4j]
        qc, r, jpvt = qr_multiply(a, c, "left", True)
        assert_array_almost_equal(q @ c, qc)

    def test_simple_complex_right_pivoting(self):
        a = array([[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]])
        q, r, jpvt = qr(a, pivoting=True)
        c = [1, 2, 3+4j]
        qc, r, jpvt = qr_multiply(a, c, pivoting=True)
        assert_array_almost_equal(c @ q, qc)

    def test_random(self):
        n = 20
        for k in range(2):
            a = random([n, n])
            q, r = qr(a)
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a)

    def test_random_left(self):
        n = 20
        for k in range(2):
            a = random([n, n])
            q, r = qr(a)
            c = random([n])
            qc, r = qr_multiply(a, c, "left")
            assert_array_almost_equal(q @ c, qc)
            qc, r = qr_multiply(a, eye(n), "left")
            assert_array_almost_equal(q, qc)

    def test_random_right(self):
        n = 20
        for k in range(2):
            a = random([n, n])
            q, r = qr(a)
            c = random([n])
            cq, r = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r = qr_multiply(a, eye(n))
            assert_array_almost_equal(q, cq)

    def test_random_pivoting(self):
        n = 20
        for k in range(2):
            a = random([n, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall(self):
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            q, r = qr(a)
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_tall_left(self):
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            q, r = qr(a, mode="economic")
            c = random([n])
            qc, r = qr_multiply(a, c, "left")
            assert_array_almost_equal(q @ c, qc)
            qc, r = qr_multiply(a, eye(n), "left")
            assert_array_almost_equal(qc, q)

    def test_random_tall_right(self):
        # full version
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            q, r = qr(a, mode="economic")
            c = random([m])
            cq, r = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r = qr_multiply(a, eye(m))
            assert_array_almost_equal(cq, q)

    def test_random_tall_pivoting(self):
        # full version pivoting
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_tall_e(self):
        # economy version
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            q, r = qr(a, mode='economic')
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))

    def test_random_tall_e_pivoting(self):
        # economy version pivoting
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            q, r, p = qr(a, pivoting=True, mode='economic')
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (n, n))
            q2, r2 = qr(a[:, p], mode='economic')
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_trap(self):
        m = 100
        n = 200
        for k in range(2):
            a = random([m, n])
            q, r = qr(a)
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a)

    def test_random_trap_pivoting(self):
        m = 100
        n = 200
        for k in range(2):
            a = random([m, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.T @ q, eye(m))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_random_complex(self):
        n = 20
        for k in range(2):
            a = random([n, n])+1j*random([n, n])
            q, r = qr(a)
            assert_array_almost_equal(q.conj().T @ q, eye(n))
            assert_array_almost_equal(q @ r, a)

    def test_random_complex_left(self):
        n = 20
        for k in range(2):
            a = random([n, n])+1j*random([n, n])
            q, r = qr(a)
            c = random([n])+1j*random([n])
            qc, r = qr_multiply(a, c, "left")
            assert_array_almost_equal(q @ c, qc)
            qc, r = qr_multiply(a, eye(n), "left")
            assert_array_almost_equal(q, qc)

    def test_random_complex_right(self):
        n = 20
        for k in range(2):
            a = random([n, n])+1j*random([n, n])
            q, r = qr(a)
            c = random([n])+1j*random([n])
            cq, r = qr_multiply(a, c)
            assert_array_almost_equal(c @ q, cq)
            cq, r = qr_multiply(a, eye(n))
            assert_array_almost_equal(q, cq)

    def test_random_complex_pivoting(self):
        n = 20
        for k in range(2):
            a = random([n, n])+1j*random([n, n])
            q, r, p = qr(a, pivoting=True)
            d = abs(diag(r))
            assert_(np.all(d[1:] <= d[:-1]))
            assert_array_almost_equal(q.conj().T @ q, eye(n))
            assert_array_almost_equal(q @ r, a[:, p])
            q2, r2 = qr(a[:, p])
            assert_array_almost_equal(q, q2)
            assert_array_almost_equal(r, r2)

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        q, r = qr(a, check_finite=False)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(q @ r, a)

    def test_lwork(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        # Get comparison values
        q, r = qr(a, lwork=None)

        # Test against minimum valid lwork
        q2, r2 = qr(a, lwork=3)
        assert_array_almost_equal(q2, q)
        assert_array_almost_equal(r2, r)

        # Test against larger lwork
        q3, r3 = qr(a, lwork=10)
        assert_array_almost_equal(q3, q)
        assert_array_almost_equal(r3, r)

        # Test against explicit lwork=-1
        q4, r4 = qr(a, lwork=-1)
        assert_array_almost_equal(q4, q)
        assert_array_almost_equal(r4, r)

        # Test against invalid lwork
        assert_raises(Exception, qr, (a,), {'lwork': 0})
        assert_raises(Exception, qr, (a,), {'lwork': 2})


class TestRQ:

    def setup_method(self):
        seed(1234)

    def test_simple(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        r, q = rq(a)
        assert_array_almost_equal(q @ q.T, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_r(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        r, q = rq(a)
        r2 = rq(a, mode='r')
        assert_array_almost_equal(r, r2)

    def test_random(self):
        n = 20
        for k in range(2):
            a = random([n, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_simple_trap(self):
        a = [[8, 2, 3], [2, 9, 3]]
        r, q = rq(a)
        assert_array_almost_equal(q.T @ q, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_simple_tall(self):
        a = [[8, 2], [2, 9], [5, 3]]
        r, q = rq(a)
        assert_array_almost_equal(q.T @ q, eye(2))
        assert_array_almost_equal(r @ q, a)

    def test_simple_fat(self):
        a = [[8, 2, 5], [2, 9, 3]]
        r, q = rq(a)
        assert_array_almost_equal(q @ q.T, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_simple_complex(self):
        a = [[3, 3+4j, 5], [5, 2, 2+7j], [3, 2, 7]]
        r, q = rq(a)
        assert_array_almost_equal(q @ q.conj().T, eye(3))
        assert_array_almost_equal(r @ q, a)

    def test_random_tall(self):
        m = 200
        n = 100
        for k in range(2):
            a = random([m, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_random_trap(self):
        m = 100
        n = 200
        for k in range(2):
            a = random([m, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_random_trap_economic(self):
        m = 100
        n = 200
        for k in range(2):
            a = random([m, n])
            r, q = rq(a, mode='economic')
            assert_array_almost_equal(q @ q.T, eye(m))
            assert_array_almost_equal(r @ q, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (m, m))

    def test_random_complex(self):
        n = 20
        for k in range(2):
            a = random([n, n])+1j*random([n, n])
            r, q = rq(a)
            assert_array_almost_equal(q @ q.conj().T, eye(n))
            assert_array_almost_equal(r @ q, a)

    def test_random_complex_economic(self):
        m = 100
        n = 200
        for k in range(2):
            a = random([m, n])+1j*random([m, n])
            r, q = rq(a, mode='economic')
            assert_array_almost_equal(q @ q.conj().T, eye(m))
            assert_array_almost_equal(r @ q, a)
            assert_equal(q.shape, (m, n))
            assert_equal(r.shape, (m, m))

    def test_check_finite(self):
        a = [[8, 2, 3], [2, 9, 3], [5, 3, 6]]
        r, q = rq(a, check_finite=False)
        assert_array_almost_equal(q @ q.T, eye(3))
        assert_array_almost_equal(r @ q, a)


class TestSchur:

    def test_simple(self):
        a = [[8, 12, 3], [2, 9, 3], [10, 3, 6]]
        t, z = schur(a)
        assert_array_almost_equal(z @ t @ z.conj().T, a)
        tc, zc = schur(a, 'complex')
        assert_(np.any(ravel(iscomplex(zc))) and np.any(ravel(iscomplex(tc))))
        assert_array_almost_equal(zc @ tc @ zc.conj().T, a)
        tc2, zc2 = rsf2csf(tc, zc)
        assert_array_almost_equal(zc2 @ tc2 @ zc2.conj().T, a)

    def test_sort(self):
        a = [[4., 3., 1., -1.],
             [-4.5, -3.5, -1., 1.],
             [9., 6., -4., 4.5],
             [6., 4., -3., 3.5]]
        s, u, sdim = schur(a, sort='lhp')
        assert_array_almost_equal([[0.1134, 0.5436, 0.8316, 0.],
                                   [-0.1134, -0.8245, 0.5544, 0.],
                                   [-0.8213, 0.1308, 0.0265, -0.5547],
                                   [-0.5475, 0.0872, 0.0177, 0.8321]],
                                  u, 3)
        assert_array_almost_equal([[-1.4142, 0.1456, -11.5816, -7.7174],
                                   [0., -0.5000, 9.4472, -0.7184],
                                   [0., 0., 1.4142, -0.1456],
                                   [0., 0., 0., 0.5]],
                                  s, 3)
        assert_equal(2, sdim)

        s, u, sdim = schur(a, sort='rhp')
        assert_array_almost_equal([[0.4862, -0.4930, 0.1434, -0.7071],
                                   [-0.4862, 0.4930, -0.1434, -0.7071],
                                   [0.6042, 0.3944, -0.6924, 0.],
                                   [0.4028, 0.5986, 0.6924, 0.]],
                                  u, 3)
        assert_array_almost_equal([[1.4142, -0.9270, 4.5368, -14.4130],
                                   [0., 0.5, 6.5809, -3.1870],
                                   [0., 0., -1.4142, 0.9270],
                                   [0., 0., 0., -0.5]],
                                  s, 3)
        assert_equal(2, sdim)

        s, u, sdim = schur(a, sort='iuc')
        assert_array_almost_equal([[0.5547, 0., -0.5721, -0.6042],
                                   [-0.8321, 0., -0.3814, -0.4028],
                                   [0., 0.7071, -0.5134, 0.4862],
                                   [0., 0.7071, 0.5134, -0.4862]],
                                  u, 3)
        assert_array_almost_equal([[-0.5000, 0.0000, -6.5809, -4.0974],
                                   [0., 0.5000, -3.3191, -14.4130],
                                   [0., 0., 1.4142, 2.1573],
                                   [0., 0., 0., -1.4142]],
                                  s, 3)
        assert_equal(2, sdim)

        s, u, sdim = schur(a, sort='ouc')
        assert_array_almost_equal([[0.4862, -0.5134, 0.7071, 0.],
                                   [-0.4862, 0.5134, 0.7071, 0.],
                                   [0.6042, 0.5721, 0., -0.5547],
                                   [0.4028, 0.3814, 0., 0.8321]],
                                  u, 3)
        assert_array_almost_equal([[1.4142, -2.1573, 14.4130, 4.0974],
                                   [0., -1.4142, 3.3191, 6.5809],
                                   [0., 0., -0.5000, 0.],
                                   [0., 0., 0., 0.5000]],
                                  s, 3)
        assert_equal(2, sdim)

        s, u, sdim = schur(a, sort=lambda x: x >= 0.0)
        assert_array_almost_equal([[0.4862, -0.4930, 0.1434, -0.7071],
                                   [-0.4862, 0.4930, -0.1434, -0.7071],
                                   [0.6042, 0.3944, -0.6924, 0.],
                                   [0.4028, 0.5986, 0.6924, 0.]],
                                  u, 3)
        assert_array_almost_equal([[1.4142, -0.9270, 4.5368, -14.4130],
                                   [0., 0.5, 6.5809, -3.1870],
                                   [0., 0., -1.4142, 0.9270],
                                   [0., 0., 0., -0.5]],
                                  s, 3)
        assert_equal(2, sdim)

    def test_sort_errors(self):
        a = [[4., 3., 1., -1.],
             [-4.5, -3.5, -1., 1.],
             [9., 6., -4., 4.5],
             [6., 4., -3., 3.5]]
        assert_raises(ValueError, schur, a, sort='unsupported')
        assert_raises(ValueError, schur, a, sort=1)

    def test_check_finite(self):
        a = [[8, 12, 3], [2, 9, 3], [10, 3, 6]]
        t, z = schur(a, check_finite=False)
        assert_array_almost_equal(z @ t @ z.conj().T, a)


class TestHessenberg:

    def test_simple(self):
        a = [[-149, -50, -154],
             [537, 180, 546],
             [-27, -9, -25]]
        h1 = [[-149.0000, 42.2037, -156.3165],
              [-537.6783, 152.5511, -554.9272],
              [0, 0.0728, 2.4489]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.T @ a @ q, h)
        assert_array_almost_equal(h, h1, decimal=4)

    def test_simple_complex(self):
        a = [[-149, -50, -154],
             [537, 180j, 546],
             [-27j, -9, -25]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.conj().T @ a @ q, h)

    def test_simple2(self):
        a = [[1, 2, 3, 4, 5, 6, 7],
             [0, 2, 3, 4, 6, 7, 2],
             [0, 2, 2, 3, 0, 3, 2],
             [0, 0, 2, 8, 0, 0, 2],
             [0, 3, 1, 2, 0, 1, 2],
             [0, 1, 2, 3, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 2]]
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.T @ a @ q, h)

    def test_simple3(self):
        a = np.eye(3)
        a[-1, 0] = 2
        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q.T @ a @ q, h)

    def test_random(self):
        n = 20
        for k in range(2):
            a = random([n, n])
            h, q = hessenberg(a, calc_q=1)
            assert_array_almost_equal(q.T @ a @ q, h)

    def test_random_complex(self):
        n = 20
        for k in range(2):
            a = random([n, n])+1j*random([n, n])
            h, q = hessenberg(a, calc_q=1)
            assert_array_almost_equal(q.conj().T @ a @ q, h)

    def test_check_finite(self):
        a = [[-149, -50, -154],
             [537, 180, 546],
             [-27, -9, -25]]
        h1 = [[-149.0000, 42.2037, -156.3165],
              [-537.6783, 152.5511, -554.9272],
              [0, 0.0728, 2.4489]]
        h, q = hessenberg(a, calc_q=1, check_finite=False)
        assert_array_almost_equal(q.T @ a @ q, h)
        assert_array_almost_equal(h, h1, decimal=4)

    def test_2x2(self):
        a = [[2, 1], [7, 12]]

        h, q = hessenberg(a, calc_q=1)
        assert_array_almost_equal(q, np.eye(2))
        assert_array_almost_equal(h, a)

        b = [[2-7j, 1+2j], [7+3j, 12-2j]]
        h2, q2 = hessenberg(b, calc_q=1)
        assert_array_almost_equal(q2, np.eye(2))
        assert_array_almost_equal(h2, b)


class TestQZ:
    def setup_method(self):
        seed(12345)

    def test_qz_single(self):
        n = 5
        A = random([n, n]).astype(float32)
        B = random([n, n]).astype(float32)
        AA, BB, Q, Z = qz(A, B)
        assert_array_almost_equal(Q @ AA @ Z.T, A, decimal=5)
        assert_array_almost_equal(Q @ BB @ Z.T, B, decimal=5)
        assert_array_almost_equal(Q @ Q.T, eye(n), decimal=5)
        assert_array_almost_equal(Z @ Z.T, eye(n), decimal=5)
        assert_(np.all(diag(BB) >= 0))

    def test_qz_double(self):
        n = 5
        A = random([n, n])
        B = random([n, n])
        AA, BB, Q, Z = qz(A, B)
        assert_array_almost_equal(Q @ AA @ Z.T, A)
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        assert_array_almost_equal(Q @ Q.T, eye(n))
        assert_array_almost_equal(Z @ Z.T, eye(n))
        assert_(np.all(diag(BB) >= 0))

    def test_qz_complex(self):
        n = 5
        A = random([n, n]) + 1j*random([n, n])
        B = random([n, n]) + 1j*random([n, n])
        AA, BB, Q, Z = qz(A, B)
        assert_array_almost_equal(Q @ AA @ Z.conj().T, A)
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B)
        assert_array_almost_equal(Q @ Q.conj().T, eye(n))
        assert_array_almost_equal(Z @ Z.conj().T, eye(n))
        assert_(np.all(diag(BB) >= 0))
        assert_(np.all(diag(BB).imag == 0))

    def test_qz_complex64(self):
        n = 5
        A = (random([n, n]) + 1j*random([n, n])).astype(complex64)
        B = (random([n, n]) + 1j*random([n, n])).astype(complex64)
        AA, BB, Q, Z = qz(A, B)
        assert_array_almost_equal(Q @ AA @ Z.conj().T, A, decimal=5)
        assert_array_almost_equal(Q @ BB @ Z.conj().T, B, decimal=5)
        assert_array_almost_equal(Q @ Q.conj().T, eye(n), decimal=5)
        assert_array_almost_equal(Z @ Z.conj().T, eye(n), decimal=5)
        assert_(np.all(diag(BB) >= 0))
        assert_(np.all(diag(BB).imag == 0))

    def test_qz_double_complex(self):
        n = 5
        A = random([n, n])
        B = random([n, n])
        AA, BB, Q, Z = qz(A, B, output='complex')
        aa = Q @ AA @ Z.conj().T
        assert_array_almost_equal(aa.real, A)
        assert_array_almost_equal(aa.imag, 0)
        bb = Q @ BB @ Z.conj().T
        assert_array_almost_equal(bb.real, B)
        assert_array_almost_equal(bb.imag, 0)
        assert_array_almost_equal(Q @ Q.conj().T, eye(n))
        assert_array_almost_equal(Z @ Z.conj().T, eye(n))
        assert_(np.all(diag(BB) >= 0))

    def test_qz_double_sort(self):
        # from https://www.nag.com/lapack-ex/node119.html
        # NOTE: These matrices may be ill-conditioned and lead to a
        # seg fault on certain python versions when compiled with
        # sse2 or sse3 older ATLAS/LAPACK binaries for windows
        # A =   np.array([[3.9,  12.5, -34.5,  -0.5],
        #                [ 4.3,  21.5, -47.5,   7.5],
        #                [ 4.3,  21.5, -43.5,   3.5],
        #                [ 4.4,  26.0, -46.0,   6.0 ]])

        # B = np.array([[ 1.0,   2.0,  -3.0,   1.0],
        #              [1.0,   3.0,  -5.0,   4.0],
        #              [1.0,   3.0,  -4.0,   3.0],
        #              [1.0,   3.0,  -4.0,   4.0]])
        A = np.array([[3.9, 12.5, -34.5, 2.5],
                      [4.3, 21.5, -47.5, 7.5],
                      [4.3, 1.5, -43.5, 3.5],
                      [4.4, 6.0, -46.0, 6.0]])

        B = np.array([[1.0, 1.0, -3.0, 1.0],
                      [1.0, 3.0, -5.0, 4.4],
                      [1.0, 2.0, -4.0, 1.0],
                      [1.2, 3.0, -4.0, 4.0]])

        assert_raises(ValueError, qz, A, B, sort=lambda ar, ai, beta: ai == 0)
        if False:
            AA, BB, Q, Z, sdim = qz(A, B, sort=lambda ar, ai, beta: ai == 0)
            # assert_(sdim == 2)
            assert_(sdim == 4)
            assert_array_almost_equal(Q @ AA @ Z.T, A)
            assert_array_almost_equal(Q @ BB @ Z.T, B)

            # test absolute values bc the sign is ambiguous and
            # might be platform dependent
            assert_array_almost_equal(np.abs(AA), np.abs(np.array(
                            [[35.7864, -80.9061, -12.0629, -9.498],
                             [0., 2.7638, -2.3505, 7.3256],
                             [0., 0., 0.6258, -0.0398],
                             [0., 0., 0., -12.8217]])), 4)
            assert_array_almost_equal(np.abs(BB), np.abs(np.array(
                            [[4.5324, -8.7878, 3.2357, -3.5526],
                             [0., 1.4314, -2.1894, 0.9709],
                             [0., 0., 1.3126, -0.3468],
                             [0., 0., 0., 0.559]])), 4)
            assert_array_almost_equal(np.abs(Q), np.abs(np.array(
                            [[-0.4193, -0.605, -0.1894, -0.6498],
                             [-0.5495, 0.6987, 0.2654, -0.3734],
                             [-0.4973, -0.3682, 0.6194, 0.4832],
                             [-0.5243, 0.1008, -0.7142, 0.4526]])), 4)
            assert_array_almost_equal(np.abs(Z), np.abs(np.array(
                            [[-0.9471, -0.2971, -0.1217, 0.0055],
                             [-0.0367, 0.1209, 0.0358, 0.9913],
                             [0.3171, -0.9041, -0.2547, 0.1312],
                             [0.0346, 0.2824, -0.9587, 0.0014]])), 4)

        # test absolute values bc the sign is ambiguous and might be platform
        # dependent
        # assert_array_almost_equal(abs(AA), abs(np.array([
        #                [3.8009, -69.4505, 50.3135, -43.2884],
        #                [0.0000, 9.2033, -0.2001, 5.9881],
        #                [0.0000, 0.0000, 1.4279, 4.4453],
        #                [0.0000, 0.0000, 0.9019, -1.1962]])), 4)
        # assert_array_almost_equal(abs(BB), abs(np.array([
        #                [1.9005, -10.2285, 0.8658, -5.2134],
        #                [0.0000,   2.3008, 0.7915,  0.4262],
        #                [0.0000,   0.0000, 0.8101,  0.0000],
        #                [0.0000,   0.0000, 0.0000, -0.2823]])), 4)
        # assert_array_almost_equal(abs(Q), abs(np.array([
        #                [0.4642,  0.7886,  0.2915, -0.2786],
        #                [0.5002, -0.5986,  0.5638, -0.2713],
        #                [0.5002,  0.0154, -0.0107,  0.8657],
        #                [0.5331, -0.1395, -0.7727, -0.3151]])), 4)
        # assert_array_almost_equal(dot(Q,Q.T), eye(4))
        # assert_array_almost_equal(abs(Z), abs(np.array([
        #                [0.9961, -0.0014,  0.0887, -0.0026],
        #                [0.0057, -0.0404, -0.0938, -0.9948],
        #                [0.0626,  0.7194, -0.6908,  0.0363],
        #                [0.0626, -0.6934, -0.7114,  0.0956]])), 4)
        # assert_array_almost_equal(dot(Z,Z.T), eye(4))

    # def test_qz_complex_sort(self):
    #    cA = np.array([
    #   [-21.10+22.50*1j, 53.50+-50.50*1j, -34.50+127.50*1j, 7.50+  0.50*1j],
    #   [-0.46+ -7.78*1j, -3.50+-37.50*1j, -15.50+ 58.50*1j,-10.50+ -1.50*1j],
    #   [ 4.30+ -5.50*1j, 39.70+-17.10*1j, -68.50+ 12.50*1j, -7.50+ -3.50*1j],
    #   [ 5.50+  4.40*1j, 14.40+ 43.30*1j, -32.50+-46.00*1j,-19.00+-32.50*1j]])

    #    cB =  np.array([
    #   [1.00+ -5.00*1j, 1.60+  1.20*1j,-3.00+  0.00*1j, 0.00+ -1.00*1j],
    #   [0.80+ -0.60*1j, 3.00+ -5.00*1j,-4.00+  3.00*1j,-2.40+ -3.20*1j],
    #   [1.00+  0.00*1j, 2.40+  1.80*1j,-4.00+ -5.00*1j, 0.00+ -3.00*1j],
    #   [0.00+  1.00*1j,-1.80+  2.40*1j, 0.00+ -4.00*1j, 4.00+ -5.00*1j]])

    #    AAS,BBS,QS,ZS,sdim = qz(cA,cB,sort='lhp')

    #    eigenvalues = diag(AAS)/diag(BBS)
    #    assert_(np.all(np.real(eigenvalues[:sdim] < 0)))
    #    assert_(np.all(np.real(eigenvalues[sdim:] > 0)))

    def test_check_finite(self):
        n = 5
        A = random([n, n])
        B = random([n, n])
        AA, BB, Q, Z = qz(A, B, check_finite=False)
        assert_array_almost_equal(Q @ AA @ Z.T, A)
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        assert_array_almost_equal(Q @ Q.T, eye(n))
        assert_array_almost_equal(Z @ Z.T, eye(n))
        assert_(np.all(diag(BB) >= 0))


def _make_pos(X):
    # the decompositions can have different signs than verified results
    return np.sign(X)*X


class TestOrdQZ:
    @classmethod
    def setup_class(cls):
        # https://www.nag.com/lapack-ex/node119.html
        A1 = np.array([[-21.10 - 22.50j, 53.5 - 50.5j, -34.5 + 127.5j,
                        7.5 + 0.5j],
                       [-0.46 - 7.78j, -3.5 - 37.5j, -15.5 + 58.5j,
                        -10.5 - 1.5j],
                       [4.30 - 5.50j, 39.7 - 17.1j, -68.5 + 12.5j,
                        -7.5 - 3.5j],
                       [5.50 + 4.40j, 14.4 + 43.3j, -32.5 - 46.0j,
                        -19.0 - 32.5j]])

        B1 = np.array([[1.0 - 5.0j, 1.6 + 1.2j, -3 + 0j, 0.0 - 1.0j],
                       [0.8 - 0.6j, .0 - 5.0j, -4 + 3j, -2.4 - 3.2j],
                       [1.0 + 0.0j, 2.4 + 1.8j, -4 - 5j, 0.0 - 3.0j],
                       [0.0 + 1.0j, -1.8 + 2.4j, 0 - 4j, 4.0 - 5.0j]])

        # https://www.nag.com/numeric/fl/nagdoc_fl23/xhtml/F08/f08yuf.xml
        A2 = np.array([[3.9, 12.5, -34.5, -0.5],
                       [4.3, 21.5, -47.5, 7.5],
                       [4.3, 21.5, -43.5, 3.5],
                       [4.4, 26.0, -46.0, 6.0]])

        B2 = np.array([[1, 2, -3, 1],
                       [1, 3, -5, 4],
                       [1, 3, -4, 3],
                       [1, 3, -4, 4]])

        # example with the eigenvalues
        # -0.33891648, 1.61217396+0.74013521j, 1.61217396-0.74013521j,
        # 0.61244091
        # thus featuring:
        #  * one complex conjugate eigenvalue pair,
        #  * one eigenvalue in the lhp
        #  * 2 eigenvalues in the unit circle
        #  * 2 non-real eigenvalues
        A3 = np.array([[5., 1., 3., 3.],
                       [4., 4., 2., 7.],
                       [7., 4., 1., 3.],
                       [0., 4., 8., 7.]])
        B3 = np.array([[8., 10., 6., 10.],
                       [7., 7., 2., 9.],
                       [9., 1., 6., 6.],
                       [5., 1., 4., 7.]])

        # example with infinite eigenvalues
        A4 = np.eye(2)
        B4 = np.diag([0, 1])

        # example with (alpha, beta) = (0, 0)
        A5 = np.diag([1, 0])

        cls.A = [A1, A2, A3, A4, A5]
        cls.B = [B1, B2, B3, B4, A5]

    def qz_decomp(self, sort):
        with np.errstate(all='raise'):
            ret = [ordqz(Ai, Bi, sort=sort) for Ai, Bi in zip(self.A, self.B)]
        return tuple(ret)

    def check(self, A, B, sort, AA, BB, alpha, beta, Q, Z):
        Id = np.eye(*A.shape)
        # make sure Q and Z are orthogonal
        assert_array_almost_equal(Q @ Q.T.conj(), Id)
        assert_array_almost_equal(Z @ Z.T.conj(), Id)
        # check factorization
        assert_array_almost_equal(Q @ AA, A @ Z)
        assert_array_almost_equal(Q @ BB, B @ Z)
        # check shape of AA and BB
        assert_array_equal(np.tril(AA, -2), np.zeros(AA.shape))
        assert_array_equal(np.tril(BB, -1), np.zeros(BB.shape))
        # check eigenvalues
        for i in range(A.shape[0]):
            # does the current diagonal element belong to a 2-by-2 block
            # that was already checked?
            if i > 0 and A[i, i - 1] != 0:
                continue
            # take care of 2-by-2 blocks
            if i < AA.shape[0] - 1 and AA[i + 1, i] != 0:
                evals, _ = eig(AA[i:i + 2, i:i + 2], BB[i:i + 2, i:i + 2])
                # make sure the pair of complex conjugate eigenvalues
                # is ordered consistently (positive imaginary part first)
                if evals[0].imag < 0:
                    evals = evals[[1, 0]]
                tmp = alpha[i:i + 2]/beta[i:i + 2]
                if tmp[0].imag < 0:
                    tmp = tmp[[1, 0]]
                assert_array_almost_equal(evals, tmp)
            else:
                if alpha[i] == 0 and beta[i] == 0:
                    assert_equal(AA[i, i], 0)
                    assert_equal(BB[i, i], 0)
                elif beta[i] == 0:
                    assert_equal(BB[i, i], 0)
                else:
                    assert_almost_equal(AA[i, i]/BB[i, i], alpha[i]/beta[i])
        sortfun = _select_function(sort)
        lastsort = True
        for i in range(A.shape[0]):
            cursort = sortfun(np.array([alpha[i]]), np.array([beta[i]]))
            # once the sorting criterion was not matched all subsequent
            # eigenvalues also shouldn't match
            if not lastsort:
                assert(not cursort)
            lastsort = cursort

    def check_all(self, sort):
        ret = self.qz_decomp(sort)

        for reti, Ai, Bi in zip(ret, self.A, self.B):
            self.check(Ai, Bi, sort, *reti)

    def test_lhp(self):
        self.check_all('lhp')

    def test_rhp(self):
        self.check_all('rhp')

    def test_iuc(self):
        self.check_all('iuc')

    def test_ouc(self):
        self.check_all('ouc')

    def test_ref(self):
        # real eigenvalues first (top-left corner)
        def sort(x, y):
            out = np.empty_like(x, dtype=bool)
            nonzero = (y != 0)
            out[~nonzero] = False
            out[nonzero] = (x[nonzero]/y[nonzero]).imag == 0
            return out

        self.check_all(sort)

    def test_cef(self):
        # complex eigenvalues first (top-left corner)
        def sort(x, y):
            out = np.empty_like(x, dtype=bool)
            nonzero = (y != 0)
            out[~nonzero] = False
            out[nonzero] = (x[nonzero]/y[nonzero]).imag != 0
            return out

        self.check_all(sort)

    def test_diff_input_types(self):
        ret = ordqz(self.A[1], self.B[2], sort='lhp')
        self.check(self.A[1], self.B[2], 'lhp', *ret)

        ret = ordqz(self.B[2], self.A[1], sort='lhp')
        self.check(self.B[2], self.A[1], 'lhp', *ret)

    def test_sort_explicit(self):
        # Test order of the eigenvalues in the 2 x 2 case where we can
        # explicitly compute the solution
        A1 = np.eye(2)
        B1 = np.diag([-2, 0.5])
        expected1 = [('lhp', [-0.5, 2]),
                     ('rhp', [2, -0.5]),
                     ('iuc', [-0.5, 2]),
                     ('ouc', [2, -0.5])]
        A2 = np.eye(2)
        B2 = np.diag([-2 + 1j, 0.5 + 0.5j])
        expected2 = [('lhp', [1/(-2 + 1j), 1/(0.5 + 0.5j)]),
                     ('rhp', [1/(0.5 + 0.5j), 1/(-2 + 1j)]),
                     ('iuc', [1/(-2 + 1j), 1/(0.5 + 0.5j)]),
                     ('ouc', [1/(0.5 + 0.5j), 1/(-2 + 1j)])]
        # 'lhp' is ambiguous so don't test it
        A3 = np.eye(2)
        B3 = np.diag([2, 0])
        expected3 = [('rhp', [0.5, np.inf]),
                     ('iuc', [0.5, np.inf]),
                     ('ouc', [np.inf, 0.5])]
        # 'rhp' is ambiguous so don't test it
        A4 = np.eye(2)
        B4 = np.diag([-2, 0])
        expected4 = [('lhp', [-0.5, np.inf]),
                     ('iuc', [-0.5, np.inf]),
                     ('ouc', [np.inf, -0.5])]
        A5 = np.diag([0, 1])
        B5 = np.diag([0, 0.5])
        # 'lhp' and 'iuc' are ambiguous so don't test them
        expected5 = [('rhp', [2, np.nan]),
                     ('ouc', [2, np.nan])]

        A = [A1, A2, A3, A4, A5]
        B = [B1, B2, B3, B4, B5]
        expected = [expected1, expected2, expected3, expected4, expected5]
        for Ai, Bi, expectedi in zip(A, B, expected):
            for sortstr, expected_eigvals in expectedi:
                _, _, alpha, beta, _, _ = ordqz(Ai, Bi, sort=sortstr)
                azero = (alpha == 0)
                bzero = (beta == 0)
                x = np.empty_like(alpha)
                x[azero & bzero] = np.nan
                x[~azero & bzero] = np.inf
                x[~bzero] = alpha[~bzero]/beta[~bzero]
                assert_allclose(expected_eigvals, x)


class TestOrdQZWorkspaceSize:

    def setup_method(self):
        seed(12345)

    def test_decompose(self):

        N = 202

        # raises error if lwork parameter to dtrsen is too small
        for ddtype in [np.float32, np.float64]:
            A = random((N, N)).astype(ddtype)
            B = random((N, N)).astype(ddtype)
            # sort = lambda ar, ai, b: ar**2 + ai**2 < b**2
            _ = ordqz(A, B, sort=lambda alpha, beta: alpha < beta,
                      output='real')

        for ddtype in [np.complex128, np.complex64]:
            A = random((N, N)).astype(ddtype)
            B = random((N, N)).astype(ddtype)
            _ = ordqz(A, B, sort=lambda alpha, beta: alpha < beta,
                      output='complex')

    @pytest.mark.slow
    def test_decompose_ouc(self):

        N = 202

        # segfaults if lwork parameter to dtrsen is too small
        for ddtype in [np.float32, np.float64, np.complex128, np.complex64]:
            A = random((N, N)).astype(ddtype)
            B = random((N, N)).astype(ddtype)
            S, T, alpha, beta, U, V = ordqz(A, B, sort='ouc')


class TestDatacopied:

    def test_datacopied(self):
        from scipy.linalg.decomp import _datacopied

        M = matrix([[0, 1], [2, 3]])
        A = asarray(M)
        L = M.tolist()
        M2 = M.copy()

        class Fake1:
            def __array__(self):
                return A

        class Fake2:
            __array_interface__ = A.__array_interface__

        F1 = Fake1()
        F2 = Fake2()

        for item, status in [(M, False), (A, False), (L, True),
                             (M2, False), (F1, False), (F2, False)]:
            arr = asarray(item)
            assert_equal(_datacopied(arr, item), status,
                         err_msg=repr(item))


def test_aligned_mem_float():
    """Check linalg works with non-aligned memory (float32)"""
    # Allocate 402 bytes of memory (allocated on boundary)
    a = arange(402, dtype=np.uint8)

    # Create an array with boundary offset 4
    z = np.frombuffer(a.data, offset=2, count=100, dtype=float32)
    z.shape = 10, 10

    eig(z, overwrite_a=True)
    eig(z.T, overwrite_a=True)


@pytest.mark.skip(platform.machine() == 'ppc64le',
                  reason="crashes on ppc64le")
def test_aligned_mem():
    """Check linalg works with non-aligned memory (float64)"""
    # Allocate 804 bytes of memory (allocated on boundary)
    a = arange(804, dtype=np.uint8)

    # Create an array with boundary offset 4
    z = np.frombuffer(a.data, offset=4, count=100, dtype=float)
    z.shape = 10, 10

    eig(z, overwrite_a=True)
    eig(z.T, overwrite_a=True)


def test_aligned_mem_complex():
    """Check that complex objects don't need to be completely aligned"""
    # Allocate 1608 bytes of memory (allocated on boundary)
    a = zeros(1608, dtype=np.uint8)

    # Create an array with boundary offset 8
    z = np.frombuffer(a.data, offset=8, count=100, dtype=complex)
    z.shape = 10, 10

    eig(z, overwrite_a=True)
    # This does not need special handling
    eig(z.T, overwrite_a=True)


def check_lapack_misaligned(func, args, kwargs):
    args = list(args)
    for i in range(len(args)):
        a = args[:]
        if isinstance(a[i], np.ndarray):
            # Try misaligning a[i]
            aa = np.zeros(a[i].size*a[i].dtype.itemsize+8, dtype=np.uint8)
            aa = np.frombuffer(aa.data, offset=4, count=a[i].size,
                               dtype=a[i].dtype)
            aa.shape = a[i].shape
            aa[...] = a[i]
            a[i] = aa
            func(*a, **kwargs)
            if len(a[i].shape) > 1:
                a[i] = a[i].T
                func(*a, **kwargs)


@pytest.mark.xfail(run=False,
                   reason="Ticket #1152, triggers a segfault in rare cases.")
def test_lapack_misaligned():
    M = np.eye(10, dtype=float)
    R = np.arange(100)
    R.shape = 10, 10
    S = np.arange(20000, dtype=np.uint8)
    S = np.frombuffer(S.data, offset=4, count=100, dtype=float)
    S.shape = 10, 10
    b = np.ones(10)
    LU, piv = lu_factor(S)
    for (func, args, kwargs) in [
            (eig, (S,), dict(overwrite_a=True)),  # crash
            (eigvals, (S,), dict(overwrite_a=True)),  # no crash
            (lu, (S,), dict(overwrite_a=True)),  # no crash
            (lu_factor, (S,), dict(overwrite_a=True)),  # no crash
            (lu_solve, ((LU, piv), b), dict(overwrite_b=True)),
            (solve, (S, b), dict(overwrite_a=True, overwrite_b=True)),
            (svd, (M,), dict(overwrite_a=True)),  # no crash
            (svd, (R,), dict(overwrite_a=True)),  # no crash
            (svd, (S,), dict(overwrite_a=True)),  # crash
            (svdvals, (S,), dict()),  # no crash
            (svdvals, (S,), dict(overwrite_a=True)),  # crash
            (cholesky, (M,), dict(overwrite_a=True)),  # no crash
            (qr, (S,), dict(overwrite_a=True)),  # crash
            (rq, (S,), dict(overwrite_a=True)),  # crash
            (hessenberg, (S,), dict(overwrite_a=True)),  # crash
            (schur, (S,), dict(overwrite_a=True)),  # crash
            ]:
        check_lapack_misaligned(func, args, kwargs)
# not properly tested
# cholesky, rsf2csf, lu_solve, solve, eig_banded, eigvals_banded, eigh, diagsvd


class TestOverwrite:
    def test_eig(self):
        assert_no_overwrite(eig, [(3, 3)])
        assert_no_overwrite(eig, [(3, 3), (3, 3)])

    def test_eigh(self):
        assert_no_overwrite(eigh, [(3, 3)])
        assert_no_overwrite(eigh, [(3, 3), (3, 3)])

    def test_eig_banded(self):
        assert_no_overwrite(eig_banded, [(3, 2)])

    def test_eigvals(self):
        assert_no_overwrite(eigvals, [(3, 3)])

    def test_eigvalsh(self):
        assert_no_overwrite(eigvalsh, [(3, 3)])

    def test_eigvals_banded(self):
        assert_no_overwrite(eigvals_banded, [(3, 2)])

    def test_hessenberg(self):
        assert_no_overwrite(hessenberg, [(3, 3)])

    def test_lu_factor(self):
        assert_no_overwrite(lu_factor, [(3, 3)])

    def test_lu_solve(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
        xlu = lu_factor(x)
        assert_no_overwrite(lambda b: lu_solve(xlu, b), [(3,)])

    def test_lu(self):
        assert_no_overwrite(lu, [(3, 3)])

    def test_qr(self):
        assert_no_overwrite(qr, [(3, 3)])

    def test_rq(self):
        assert_no_overwrite(rq, [(3, 3)])

    def test_schur(self):
        assert_no_overwrite(schur, [(3, 3)])

    def test_schur_complex(self):
        assert_no_overwrite(lambda a: schur(a, 'complex'), [(3, 3)],
                            dtypes=[np.float32, np.float64])

    def test_svd(self):
        assert_no_overwrite(svd, [(3, 3)])
        assert_no_overwrite(lambda a: svd(a, lapack_driver='gesvd'), [(3, 3)])

    def test_svdvals(self):
        assert_no_overwrite(svdvals, [(3, 3)])


def _check_orth(n, dtype, skip_big=False):
    X = np.ones((n, 2), dtype=float).astype(dtype)

    eps = np.finfo(dtype).eps
    tol = 1000 * eps

    Y = orth(X)
    assert_equal(Y.shape, (n, 1))
    assert_allclose(Y, Y.mean(), atol=tol)

    Y = orth(X.T)
    assert_equal(Y.shape, (2, 1))
    assert_allclose(Y, Y.mean(), atol=tol)

    if n > 5 and not skip_big:
        np.random.seed(1)
        X = np.random.rand(n, 5) @ np.random.rand(5, n)
        X = X + 1e-4 * np.random.rand(n, 1) @ np.random.rand(1, n)
        X = X.astype(dtype)

        Y = orth(X, rcond=1e-3)
        assert_equal(Y.shape, (n, 5))

        Y = orth(X, rcond=1e-6)
        assert_equal(Y.shape, (n, 5 + 1))


@pytest.mark.slow
@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8,
                    reason="test only on 64-bit, else too slow")
def test_orth_memory_efficiency():
    # Pick n so that 16*n bytes is reasonable but 8*n*n bytes is unreasonable.
    # Keep in mind that @pytest.mark.slow tests are likely to be running
    # under configurations that support 4Gb+ memory for tests related to
    # 32 bit overflow.
    n = 10*1000*1000
    try:
        _check_orth(n, np.float64, skip_big=True)
    except MemoryError as e:
        raise AssertionError(
            'memory error perhaps caused by orth regression'
        ) from e


def test_orth():
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    sizes = [1, 2, 3, 10, 100]
    for dt, n in itertools.product(dtypes, sizes):
        _check_orth(n, dt)


def test_null_space():
    np.random.seed(1)

    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    sizes = [1, 2, 3, 10, 100]

    for dt, n in itertools.product(dtypes, sizes):
        X = np.ones((2, n), dtype=dt)

        eps = np.finfo(dt).eps
        tol = 1000 * eps

        Y = null_space(X)
        assert_equal(Y.shape, (n, n-1))
        assert_allclose(X @ Y, 0, atol=tol)

        Y = null_space(X.T)
        assert_equal(Y.shape, (2, 1))
        assert_allclose(X.T @ Y, 0, atol=tol)

        X = np.random.randn(1 + n//2, n)
        Y = null_space(X)
        assert_equal(Y.shape, (n, n - 1 - n//2))
        assert_allclose(X @ Y, 0, atol=tol)

        if n > 5:
            np.random.seed(1)
            X = np.random.rand(n, 5) @ np.random.rand(5, n)
            X = X + 1e-4 * np.random.rand(n, 1) @ np.random.rand(1, n)
            X = X.astype(dt)

            Y = null_space(X, rcond=1e-3)
            assert_equal(Y.shape, (n, n - 5))

            Y = null_space(X, rcond=1e-6)
            assert_equal(Y.shape, (n, n - 6))


def test_subspace_angles():
    H = hadamard(8, float)
    A = H[:, :3]
    B = H[:, 3:]
    assert_allclose(subspace_angles(A, B), [np.pi / 2.] * 3, atol=1e-14)
    assert_allclose(subspace_angles(B, A), [np.pi / 2.] * 3, atol=1e-14)
    for x in (A, B):
        assert_allclose(subspace_angles(x, x), np.zeros(x.shape[1]),
                        atol=1e-14)
    # From MATLAB function "subspace", which effectively only returns the
    # last value that we calculate
    x = np.array(
        [[0.537667139546100, 0.318765239858981, 3.578396939725760, 0.725404224946106],  # noqa: E501
         [1.833885014595086, -1.307688296305273, 2.769437029884877, -0.063054873189656],  # noqa: E501
         [-2.258846861003648, -0.433592022305684, -1.349886940156521, 0.714742903826096],  # noqa: E501
         [0.862173320368121, 0.342624466538650, 3.034923466331855, -0.204966058299775]])  # noqa: E501
    expected = 1.481454682101605
    assert_allclose(subspace_angles(x[:, :2], x[:, 2:])[0], expected,
                    rtol=1e-12)
    assert_allclose(subspace_angles(x[:, 2:], x[:, :2])[0], expected,
                    rtol=1e-12)
    expected = 0.746361174247302
    assert_allclose(subspace_angles(x[:, :2], x[:, [2]]), expected, rtol=1e-12)
    assert_allclose(subspace_angles(x[:, [2]], x[:, :2]), expected, rtol=1e-12)
    expected = 0.487163718534313
    assert_allclose(subspace_angles(x[:, :3], x[:, [3]]), expected, rtol=1e-12)
    assert_allclose(subspace_angles(x[:, [3]], x[:, :3]), expected, rtol=1e-12)
    expected = 0.328950515907756
    assert_allclose(subspace_angles(x[:, :2], x[:, 1:]), [expected, 0],
                    atol=1e-12)
    # Degenerate conditions
    assert_raises(ValueError, subspace_angles, x[0], x)
    assert_raises(ValueError, subspace_angles, x, x[0])
    assert_raises(ValueError, subspace_angles, x[:-1], x)

    # Test branch if mask.any is True:
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0]])
    B = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 1]])
    expected = np.array([np.pi/2, 0, 0])
    assert_allclose(subspace_angles(A, B), expected, rtol=1e-12)

    # Complex
    # second column in "b" does not affect result, just there so that
    # b can have more cols than a, and vice-versa (both conditional code paths)
    a = [[1 + 1j], [0]]
    b = [[1 - 1j, 0], [0, 1]]
    assert_allclose(subspace_angles(a, b), 0., atol=1e-14)
    assert_allclose(subspace_angles(b, a), 0., atol=1e-14)


class TestCDF2RDF:

    def matmul(self, a, b):
        return np.einsum('...ij,...jk->...ik', a, b)

    def assert_eig_valid(self, w, v, x):
        assert_array_almost_equal(
            self.matmul(v, w),
            self.matmul(x, v)
        )

    def test_single_array0x0real(self):
        # eig doesn't support 0x0 in old versions of numpy
        X = np.empty((0, 0))
        w, v = np.empty(0), np.empty((0, 0))
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array2x2_real(self):
        X = np.array([[1, 2], [3, -1]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array2x2_complex(self):
        X = np.array([[1, 2], [-2, 1]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array3x3_real(self):
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_single_array3x3_complex(self):
        X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
        w, v = np.linalg.eig(X)
        wr, vr = cdf2rdf(w, v)
        self.assert_eig_valid(wr, vr, X)

    def test_random_1d_stacked_arrays(self):
        # cannot test M == 0 due to bug in old numpy
        for M in range(1, 7):
            np.random.seed(999999999)
            X = np.random.rand(100, M, M)
            w, v = np.linalg.eig(X)
            wr, vr = cdf2rdf(w, v)
            self.assert_eig_valid(wr, vr, X)

    def test_random_2d_stacked_arrays(self):
        # cannot test M == 0 due to bug in old numpy
        for M in range(1, 7):
            X = np.random.rand(10, 10, M, M)
            w, v = np.linalg.eig(X)
            wr, vr = cdf2rdf(w, v)
            self.assert_eig_valid(wr, vr, X)

    def test_low_dimensionality_error(self):
        w, v = np.empty(()), np.array((2,))
        assert_raises(ValueError, cdf2rdf, w, v)

    def test_not_square_error(self):
        # Check that passing a non-square array raises a ValueError.
        w, v = np.arange(3), np.arange(6).reshape(3, 2)
        assert_raises(ValueError, cdf2rdf, w, v)

    def test_swapped_v_w_error(self):
        # Check that exchanging places of w and v raises ValueError.
        X = np.array([[1, 2, 3], [0, 4, 5], [0, -5, 4]])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, v, w)

    def test_non_associated_error(self):
        # Check that passing non-associated eigenvectors raises a ValueError.
        w, v = np.arange(3), np.arange(16).reshape(4, 4)
        assert_raises(ValueError, cdf2rdf, w, v)

    def test_not_conjugate_pairs(self):
        # Check that passing non-conjugate pairs raises a ValueError.
        X = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, w, v)

        # different arrays in the stack, so not conjugate
        X = np.array([
            [[1, 2, 3], [1, 2, 3], [2, 5, 6+1j]],
            [[1, 2, 3], [1, 2, 3], [2, 5, 6-1j]],
        ])
        w, v = np.linalg.eig(X)
        assert_raises(ValueError, cdf2rdf, w, v)
