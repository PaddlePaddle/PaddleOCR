#******************************************************************************
#   Copyright (C) 2013 Kenneth L. Ho
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer. Redistributions in binary
#   form must reproduce the above copyright notice, this list of conditions and
#   the following disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
#   None of the names of the copyright holders may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#******************************************************************************

import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
import itertools

from numpy.testing import (assert_, assert_allclose, assert_equal,
                           assert_array_equal)
import pytest
from pytest import raises as assert_raises
import sys
_IS_32BIT = (sys.maxsize < 2**32)


@pytest.fixture()
def eps():
    yield 1e-12


@pytest.fixture(params=[np.float64, np.complex128])
def A(request):
    # construct Hilbert matrix
    # set parameters
    n = 300
    yield hilbert(n).astype(request.param)


@pytest.fixture()
def L(A):
    yield aslinearoperator(A)


@pytest.fixture()
def rank(A, eps):
    S = np.linalg.svd(A, compute_uv=False)
    try:
        rank = np.nonzero(S < eps)[0][0]
    except IndexError:
        rank = A.shape[0]
    return rank


class TestInterpolativeDecomposition:

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_real_id_fixed_precision(self, A, L, eps, rand, lin_op):
        if _IS_32BIT and A.dtype == np.complex_ and rand:
            pytest.xfail("bug in external fortran code")
        # Test ID routines on a Hilbert matrix.
        A_or_L = A if not lin_op else L

        k, idx, proj = pymatrixid.interp_decomp(A_or_L, eps, rand=rand)
        B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_real_id_fixed_rank(self, A, L, eps, rank, rand, lin_op):
        if _IS_32BIT and A.dtype == np.complex_ and rand:
            pytest.xfail("bug in external fortran code")
        k = rank
        A_or_L = A if not lin_op else L

        idx, proj = pymatrixid.interp_decomp(A_or_L, k, rand=rand)
        B = pymatrixid.reconstruct_matrix_from_id(A[:, idx[:k]], idx, proj)
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize("rand,lin_op", [(False, False)])
    def test_real_id_skel_and_interp_matrices(
            self, A, L, eps, rank, rand, lin_op):
        k = rank
        A_or_L = A if not lin_op else L

        idx, proj = pymatrixid.interp_decomp(A_or_L, k, rand=rand)
        P = pymatrixid.reconstruct_interp_matrix(idx, proj)
        B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
        assert_allclose(B, A[:, idx[:k]], rtol=eps, atol=1e-08)
        assert_allclose(B @ P, A, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_svd_fixed_precison(self, A, L, eps, rand, lin_op):
        if _IS_32BIT and A.dtype == np.complex_ and rand:
            pytest.xfail("bug in external fortran code")
        A_or_L = A if not lin_op else L

        U, S, V = pymatrixid.svd(A_or_L, eps, rand=rand)
        B = U * S @ V.T.conj()
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    @pytest.mark.parametrize(
        "rand,lin_op",
        [(False, False), (True, False), (True, True)])
    def test_svd_fixed_rank(self, A, L, eps, rank, rand, lin_op):
        if _IS_32BIT and A.dtype == np.complex_ and rand:
            pytest.xfail("bug in external fortran code")
        k = rank
        A_or_L = A if not lin_op else L

        U, S, V = pymatrixid.svd(A_or_L, k, rand=rand)
        B = U * S @ V.T.conj()
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    def test_id_to_svd(self, A, eps, rank):
        k = rank

        idx, proj = pymatrixid.interp_decomp(A, k, rand=False)
        U, S, V = pymatrixid.id_to_svd(A[:, idx[:k]], idx, proj)
        B = U * S @ V.T.conj()
        assert_allclose(A, B, rtol=eps, atol=1e-08)

    def test_estimate_spectral_norm(self, A):
        s = svdvals(A)
        norm_2_est = pymatrixid.estimate_spectral_norm(A)
        assert_(np.allclose(norm_2_est, s[0], 1e-6))

    def test_estimate_spectral_norm_diff(self, A):
        B = A.copy()
        B[:, 0] *= 1.2
        s = svdvals(A - B)
        norm_2_est = pymatrixid.estimate_spectral_norm_diff(A, B)
        assert_(np.allclose(norm_2_est, s[0], 1e-6))

    def test_rank_estimates_array(self, A):
        B = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=A.dtype)

        for M in [A, B]:
            rank_tol = 1e-9
            rank_np = np.linalg.matrix_rank(M, norm(M, 2) * rank_tol)
            rank_est = pymatrixid.estimate_rank(M, rank_tol)
            assert_(rank_est >= rank_np)
            assert_(rank_est <= rank_np + 10)

    def test_rank_estimates_lin_op(self, A):
        B = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=A.dtype)

        for M in [A, B]:
            ML = aslinearoperator(M)
            rank_tol = 1e-9
            rank_np = np.linalg.matrix_rank(M, norm(M, 2) * rank_tol)
            rank_est = pymatrixid.estimate_rank(ML, rank_tol)
            assert_(rank_est >= rank_np - 4)
            assert_(rank_est <= rank_np + 4)

    def test_rand(self):
        pymatrixid.seed('default')
        assert_(np.allclose(pymatrixid.rand(2), [0.8932059, 0.64500803], 1e-4))

        pymatrixid.seed(1234)
        x1 = pymatrixid.rand(2)
        assert_(np.allclose(x1, [0.7513823, 0.06861718], 1e-4))

        np.random.seed(1234)
        pymatrixid.seed()
        x2 = pymatrixid.rand(2)

        np.random.seed(1234)
        pymatrixid.seed(np.random.rand(55))
        x3 = pymatrixid.rand(2)

        assert_allclose(x1, x2)
        assert_allclose(x1, x3)

    def test_badcall(self):
        A = hilbert(5).astype(np.float32)
        assert_raises(ValueError, pymatrixid.interp_decomp, A, 1e-6, rand=False)

    def test_rank_too_large(self):
        # svd(array, k) should not segfault
        a = np.ones((4, 3))
        with assert_raises(ValueError):
            pymatrixid.svd(a, 4)

    def test_full_rank(self):
        eps = 1.0e-12

        # fixed precision
        A = np.random.rand(16, 8)
        k, idx, proj = pymatrixid.interp_decomp(A, eps)
        assert_(k == A.shape[1])

        P = pymatrixid.reconstruct_interp_matrix(idx, proj)
        B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
        assert_allclose(A, B.dot(P))

        # fixed rank
        idx, proj = pymatrixid.interp_decomp(A, k)

        P = pymatrixid.reconstruct_interp_matrix(idx, proj)
        B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
        assert_allclose(A, B.dot(P))

    @pytest.mark.parametrize("dtype", [np.float_, np.complex_])
    @pytest.mark.parametrize("rand", [True, False])
    @pytest.mark.parametrize("eps", [1, 0.1])
    def test_bug_9793(self, dtype, rand, eps):
        if _IS_32BIT and dtype == np.complex_ and rand:
            pytest.xfail("bug in external fortran code")
        A = np.array([[-1, -1, -1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1],
                      [1, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 1]],
                     dtype=dtype, order="C")
        B = A.copy()
        interp_decomp(A.T, eps, rand=rand)
        assert_array_equal(A, B)
