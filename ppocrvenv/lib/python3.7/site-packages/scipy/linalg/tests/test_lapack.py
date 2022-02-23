#
# Created by: Pearu Peterson, September 2002
#

import sys
import subprocess
import time
from functools import reduce

from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
                           assert_allclose, assert_almost_equal,
                           assert_array_equal)
import pytest
from pytest import raises as assert_raises

import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
                   triu_indices)

from numpy.random import rand, randint, seed

from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
                          solve, ldl, norm, block_diag, qr, eigh)

from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group


import scipy.sparse as sps

try:
    from scipy.linalg import _clapack as clapack
except ImportError:
    clapack = None
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs

REAL_DTYPES = [np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
DTYPES = REAL_DTYPES + COMPLEX_DTYPES


def generate_random_dtype_array(shape, dtype):
    # generates a random matrix of desired data type of shape
    if dtype in COMPLEX_DTYPES:
        return (np.random.rand(*shape)
                + np.random.rand(*shape)*1.0j).astype(dtype)
    return np.random.rand(*shape).astype(dtype)


def test_lapack_documented():
    """Test that all entries are in the doc."""
    if lapack.__doc__ is None:  # just in case there is a python -OO
        pytest.skip('lapack.__doc__ is None')
    names = set(lapack.__doc__.split())
    ignore_list = set([
        'absolute_import', 'clapack', 'division', 'find_best_lapack_type',
        'flapack', 'print_function', 'HAS_ILP64',
    ])
    missing = list()
    for name in dir(lapack):
        if (not name.startswith('_') and name not in ignore_list and
                name not in names):
            missing.append(name)
    assert missing == [], 'Name(s) missing from lapack.__doc__ or ignore_list'


class TestFlapackSimple:

    def test_gebal(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a1 = [[1, 0, 0, 3e-4],
              [4, 0, 0, 2e-3],
              [7, 1, 0, 0],
              [0, 1, 0, 0]]
        for p in 'sdzc':
            f = getattr(flapack, p+'gebal', None)
            if f is None:
                continue
            ba, lo, hi, pivscale, info = f(a)
            assert_(not info, repr(info))
            assert_array_almost_equal(ba, a)
            assert_equal((lo, hi), (0, len(a[0])-1))
            assert_array_almost_equal(pivscale, np.ones(len(a)))

            ba, lo, hi, pivscale, info = f(a1, permute=1, scale=1)
            assert_(not info, repr(info))
            # print(a1)
            # print(ba, lo, hi, pivscale)

    def test_gehrd(self):
        a = [[-149, -50, -154],
             [537, 180, 546],
             [-27, -9, -25]]
        for p in 'd':
            f = getattr(flapack, p+'gehrd', None)
            if f is None:
                continue
            ht, tau, info = f(a)
            assert_(not info, repr(info))

    def test_trsyl(self):
        a = np.array([[1, 2], [0, 4]])
        b = np.array([[5, 6], [0, 8]])
        c = np.array([[9, 10], [11, 12]])
        trans = 'T'

        # Test single and double implementations, including most
        # of the options
        for dtype in 'fdFD':
            a1, b1, c1 = a.astype(dtype), b.astype(dtype), c.astype(dtype)
            trsyl, = get_lapack_funcs(('trsyl',), (a1,))
            if dtype.isupper():  # is complex dtype
                a1[0] += 1j
                trans = 'C'

            x, scale, info = trsyl(a1, b1, c1)
            assert_array_almost_equal(np.dot(a1, x) + np.dot(x, b1),
                                      scale * c1)

            x, scale, info = trsyl(a1, b1, c1, trana=trans, tranb=trans)
            assert_array_almost_equal(
                    np.dot(a1.conjugate().T, x) + np.dot(x, b1.conjugate().T),
                    scale * c1, decimal=4)

            x, scale, info = trsyl(a1, b1, c1, isgn=-1)
            assert_array_almost_equal(np.dot(a1, x) - np.dot(x, b1),
                                      scale * c1, decimal=4)

    def test_lange(self):
        a = np.array([
            [-149, -50, -154],
            [537, 180, 546],
            [-27, -9, -25]])

        for dtype in 'fdFD':
            for norm_str in 'Mm1OoIiFfEe':
                a1 = a.astype(dtype)
                if dtype.isupper():
                    # is complex dtype
                    a1[0, 0] += 1j

                lange, = get_lapack_funcs(('lange',), (a1,))
                value = lange(norm_str, a1)

                if norm_str in 'FfEe':
                    if dtype in 'Ff':
                        decimal = 3
                    else:
                        decimal = 7
                    ref = np.sqrt(np.sum(np.square(np.abs(a1))))
                    assert_almost_equal(value, ref, decimal)
                else:
                    if norm_str in 'Mm':
                        ref = np.max(np.abs(a1))
                    elif norm_str in '1Oo':
                        ref = np.max(np.sum(np.abs(a1), axis=0))
                    elif norm_str in 'Ii':
                        ref = np.max(np.sum(np.abs(a1), axis=1))

                    assert_equal(value, ref)


class TestLapack:

    def test_flapack(self):
        if hasattr(flapack, 'empty_module'):
            # flapack module is empty
            pass

    def test_clapack(self):
        if hasattr(clapack, 'empty_module'):
            # clapack module is empty
            pass


class TestLeastSquaresSolvers:

    def test_gels(self):
        seed(1234)
        # Test fat/tall matrix argument handling - gh-issue #8329
        for ind, dtype in enumerate(DTYPES):
            m = 10
            n = 20
            nrhs = 1
            a1 = rand(m, n).astype(dtype)
            b1 = rand(n).astype(dtype)
            gls, glslw = get_lapack_funcs(('gels', 'gels_lwork'), dtype=dtype)

            # Request of sizes
            lwork = _compute_lwork(glslw, m, n, nrhs)
            _, _, info = gls(a1, b1, lwork=lwork)
            assert_(info >= 0)
            _, _, info = gls(a1, b1, trans='TTCC'[ind], lwork=lwork)
            assert_(info >= 0)

        for dtype in REAL_DTYPES:
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            gels, gels_lwork, geqrf = get_lapack_funcs(
                    ('gels', 'gels_lwork', 'geqrf'), (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            lwork = _compute_lwork(gels_lwork, m, n, nrhs)

            lqr, x, info = gels(a1, b1, lwork=lwork)
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            lqr_truth, _, _, _ = geqrf(a1)
            assert_array_equal(lqr, lqr_truth)

        for dtype in COMPLEX_DTYPES:
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            gels, gels_lwork, geqrf = get_lapack_funcs(
                    ('gels', 'gels_lwork', 'geqrf'), (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            lwork = _compute_lwork(gels_lwork, m, n, nrhs)

            lqr, x, info = gels(a1, b1, lwork=lwork)
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype), rtol=25*np.finfo(dtype).eps)
            lqr_truth, _, _, _ = geqrf(a1)
            assert_array_equal(lqr, lqr_truth)

    def test_gelsd(self):
        for dtype in REAL_DTYPES:
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            gelsd, gelsd_lwork = get_lapack_funcs(('gelsd', 'gelsd_lwork'),
                                                  (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            work, iwork, info = gelsd_lwork(m, n, nrhs, -1)
            lwork = int(np.real(work))
            iwork_size = iwork

            x, s, rank, info = gelsd(a1, b1, lwork, iwork_size,
                                     -1, False, False)
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            assert_allclose(s, np.array([12.596017180511966,
                                         0.583396253199685], dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

        for dtype in COMPLEX_DTYPES:
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            gelsd, gelsd_lwork = get_lapack_funcs(('gelsd', 'gelsd_lwork'),
                                                  (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            work, rwork, iwork, info = gelsd_lwork(m, n, nrhs, -1)
            lwork = int(np.real(work))
            rwork_size = int(rwork)
            iwork_size = iwork

            x, s, rank, info = gelsd(a1, b1, lwork, rwork_size, iwork_size,
                                     -1, False, False)
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype), rtol=25*np.finfo(dtype).eps)
            assert_allclose(s,
                            np.array([13.035514762572043, 4.337666985231382],
                                     dtype=dtype), rtol=25*np.finfo(dtype).eps)

    def test_gelss(self):

        for dtype in REAL_DTYPES:
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            gelss, gelss_lwork = get_lapack_funcs(('gelss', 'gelss_lwork'),
                                                  (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            work, info = gelss_lwork(m, n, nrhs, -1)
            lwork = int(np.real(work))

            v, x, s, rank, work, info = gelss(a1, b1, -1, lwork, False, False)
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            assert_allclose(s, np.array([12.596017180511966,
                                         0.583396253199685], dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

        for dtype in COMPLEX_DTYPES:
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            gelss, gelss_lwork = get_lapack_funcs(('gelss', 'gelss_lwork'),
                                                  (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            work, info = gelss_lwork(m, n, nrhs, -1)
            lwork = int(np.real(work))

            v, x, s, rank, work, info = gelss(a1, b1, -1, lwork, False, False)
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)
            assert_allclose(s, np.array([13.035514762572043,
                                         4.337666985231382], dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

    def test_gelsy(self):

        for dtype in REAL_DTYPES:
            a1 = np.array([[1.0, 2.0],
                           [4.0, 5.0],
                           [7.0, 8.0]], dtype=dtype)
            b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
            gelsy, gelsy_lwork = get_lapack_funcs(('gelsy', 'gelss_lwork'),
                                                  (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            work, info = gelsy_lwork(m, n, nrhs, 10*np.finfo(dtype).eps)
            lwork = int(np.real(work))

            jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
            v, x, j, rank, info = gelsy(a1, b1, jptv, np.finfo(dtype).eps,
                                        lwork, False, False)
            assert_allclose(x[:-1], np.array([-14.333333333333323,
                                              14.999999999999991],
                                             dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)

        for dtype in COMPLEX_DTYPES:
            a1 = np.array([[1.0+4.0j, 2.0],
                           [4.0+0.5j, 5.0-3.0j],
                           [7.0-2.0j, 8.0+0.7j]], dtype=dtype)
            b1 = np.array([16.0, 17.0+2.0j, 20.0-4.0j], dtype=dtype)
            gelsy, gelsy_lwork = get_lapack_funcs(('gelsy', 'gelss_lwork'),
                                                  (a1, b1))

            m, n = a1.shape
            if len(b1.shape) == 2:
                nrhs = b1.shape[1]
            else:
                nrhs = 1

            # Request of sizes
            work, info = gelsy_lwork(m, n, nrhs, 10*np.finfo(dtype).eps)
            lwork = int(np.real(work))

            jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
            v, x, j, rank, info = gelsy(a1, b1, jptv, np.finfo(dtype).eps,
                                        lwork, False, False)
            assert_allclose(x[:-1],
                            np.array([1.161753632288328-1.901075709391912j,
                                      1.735882340522193+1.521240901196909j],
                                     dtype=dtype),
                            rtol=25*np.finfo(dtype).eps)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(3, 4), (5, 2), (2**18, 2**18)])
def test_geqrf_lwork(dtype, shape):
    geqrf_lwork = get_lapack_funcs(('geqrf_lwork'), dtype=dtype)
    m, n = shape
    lwork, info = geqrf_lwork(m=m, n=n)
    assert_equal(info, 0)


class TestRegression:

    def test_ticket_1645(self):
        # Check that RQ routines have correct lwork
        for dtype in DTYPES:
            a = np.zeros((300, 2), dtype=dtype)

            gerqf, = get_lapack_funcs(['gerqf'], [a])
            assert_raises(Exception, gerqf, a, lwork=2)
            rq, tau, work, info = gerqf(a)

            if dtype in REAL_DTYPES:
                orgrq, = get_lapack_funcs(['orgrq'], [a])
                assert_raises(Exception, orgrq, rq[-2:], tau, lwork=1)
                orgrq(rq[-2:], tau, lwork=2)
            elif dtype in COMPLEX_DTYPES:
                ungrq, = get_lapack_funcs(['ungrq'], [a])
                assert_raises(Exception, ungrq, rq[-2:], tau, lwork=1)
                ungrq(rq[-2:], tau, lwork=2)


class TestDpotr:
    def test_gh_2691(self):
        # 'lower' argument of dportf/dpotri
        for lower in [True, False]:
            for clean in [True, False]:
                np.random.seed(42)
                x = np.random.normal(size=(3, 3))
                a = x.dot(x.T)

                dpotrf, dpotri = get_lapack_funcs(("potrf", "potri"), (a, ))

                c, info = dpotrf(a, lower, clean=clean)
                dpt = dpotri(c, lower)[0]

                if lower:
                    assert_allclose(np.tril(dpt), np.tril(inv(a)))
                else:
                    assert_allclose(np.triu(dpt), np.triu(inv(a)))


class TestDlasd4:
    def test_sing_val_update(self):

        sigmas = np.array([4., 3., 2., 0])
        m_vec = np.array([3.12, 5.7, -4.8, -2.2])

        M = np.hstack((np.vstack((np.diag(sigmas[0:-1]),
                                  np.zeros((1, len(m_vec) - 1)))),
                       m_vec[:, np.newaxis]))
        SM = svd(M, full_matrices=False, compute_uv=False, overwrite_a=False,
                 check_finite=False)

        it_len = len(sigmas)
        sgm = np.concatenate((sigmas[::-1], [sigmas[0] + it_len*norm(m_vec)]))
        mvc = np.concatenate((m_vec[::-1], (0,)))

        lasd4 = get_lapack_funcs('lasd4', (sigmas,))

        roots = []
        for i in range(0, it_len):
            res = lasd4(i, sgm, mvc)
            roots.append(res[1])

            assert_((res[3] <= 0), "LAPACK root finding dlasd4 failed to find \
                                    the singular value %i" % i)
        roots = np.array(roots)[::-1]

        assert_((not np.any(np.isnan(roots)), "There are NaN roots"))
        assert_allclose(SM, roots, atol=100*np.finfo(np.float64).eps,
                        rtol=100*np.finfo(np.float64).eps)


class TestTbtrs:

    @pytest.mark.parametrize('dtype', DTYPES)
    def test_nag_example_f07vef_f07vsf(self, dtype):
        """Test real (f07vef) and complex (f07vsf) examples from NAG

        Examples available from:
        * https://www.nag.com/numeric/fl/nagdoc_latest/html/f07/f07vef.html
        * https://www.nag.com/numeric/fl/nagdoc_latest/html/f07/f07vsf.html

        """
        if dtype in REAL_DTYPES:
            ab = np.array([[-4.16, 4.78, 6.32, 0.16],
                           [-2.25, 5.86, -4.82, 0]],
                          dtype=dtype)
            b = np.array([[-16.64, -4.16],
                          [-13.78, -16.59],
                          [13.10, -4.94],
                          [-14.14, -9.96]],
                         dtype=dtype)
            x_out = np.array([[4, 1],
                              [-1, -3],
                              [3, 2],
                              [2, -2]],
                             dtype=dtype)
        elif dtype in COMPLEX_DTYPES:
            ab = np.array([[-1.94+4.43j, 4.12-4.27j, 0.43-2.66j, 0.44+0.1j],
                           [-3.39+3.44j, -1.84+5.52j, 1.74 - 0.04j, 0],
                           [1.62+3.68j, -2.77-1.93j, 0, 0]],
                          dtype=dtype)
            b = np.array([[-8.86 - 3.88j, -24.09 - 5.27j],
                          [-15.57 - 23.41j, -57.97 + 8.14j],
                          [-7.63 + 22.78j, 19.09 - 29.51j],
                          [-14.74 - 2.40j, 19.17 + 21.33j]],
                         dtype=dtype)
            x_out = np.array([[2j, 1 + 5j],
                              [1 - 3j, -7 - 2j],
                              [-4.001887 - 4.988417j, 3.026830 + 4.003182j],
                              [1.996158 - 1.045105j, -6.103357 - 8.986653j]],
                             dtype=dtype)
        else:
            raise ValueError(f"Datatype {dtype} not understood.")

        tbtrs = get_lapack_funcs(('tbtrs'), dtype=dtype)
        x, info = tbtrs(ab=ab, b=b, uplo='L')
        assert_equal(info, 0)
        assert_allclose(x, x_out, rtol=0, atol=1e-5)

    @pytest.mark.parametrize('dtype,trans',
                             [(dtype, trans)
                              for dtype in DTYPES for trans in ['N', 'T', 'C']
                              if not (trans == 'C' and dtype in REAL_DTYPES)])
    @pytest.mark.parametrize('uplo', ['U', 'L'])
    @pytest.mark.parametrize('diag', ['N', 'U'])
    def test_random_matrices(self, dtype, trans, uplo, diag):
        seed(1724)
        # n, nrhs, kd are used to specify A and b.
        # A is of shape n x n with kd super/sub-diagonals
        # b is of shape n x nrhs matrix
        n, nrhs, kd = 4, 3, 2
        tbtrs = get_lapack_funcs('tbtrs', dtype=dtype)

        is_upper = (uplo == 'U')
        ku = kd * is_upper
        kl = kd - ku

        # Construct the diagonal and kd super/sub diagonals of A with
        # the corresponding offsets.
        band_offsets = range(ku, -kl - 1, -1)
        band_widths = [n - abs(x) for x in band_offsets]
        bands = [generate_random_dtype_array((width,), dtype)
                 for width in band_widths]

        if diag == 'U':  # A must be unit triangular
            bands[ku] = np.ones(n, dtype=dtype)

        # Construct the diagonal banded matrix A from the bands and offsets.
        a = sps.diags(bands, band_offsets, format='dia')

        # Convert A into banded storage form
        ab = np.zeros((kd + 1, n), dtype)
        for row, k in enumerate(band_offsets):
            ab[row, max(k, 0):min(n+k, n)] = a.diagonal(k)

        # The RHS values.
        b = generate_random_dtype_array((n, nrhs), dtype)

        x, info = tbtrs(ab=ab, b=b, uplo=uplo, trans=trans, diag=diag)
        assert_equal(info, 0)

        if trans == 'N':
            assert_allclose(a @ x, b, rtol=5e-5)
        elif trans == 'T':
            assert_allclose(a.T @ x, b, rtol=5e-5)
        elif trans == 'C':
            assert_allclose(a.H @ x, b, rtol=5e-5)
        else:
            raise ValueError('Invalid trans argument')

    @pytest.mark.parametrize('uplo,trans,diag',
                             [['U', 'N', 'Invalid'],
                              ['U', 'Invalid', 'N'],
                              ['Invalid', 'N', 'N']])
    def test_invalid_argument_raises_exception(self, uplo, trans, diag):
        """Test if invalid values of uplo, trans and diag raise exceptions"""
        # Argument checks occur independently of used datatype.
        # This mean we must not parameterize all available datatypes.
        tbtrs = get_lapack_funcs('tbtrs', dtype=np.float64)
        ab = rand(4, 2)
        b = rand(2, 4)
        assert_raises(Exception, tbtrs, ab, b, uplo, trans, diag)

    def test_zero_element_in_diagonal(self):
        """Test if a matrix with a zero diagonal element is singular

        If the i-th diagonal of A is zero, ?tbtrs should return `i` in `info`
        indicating the provided matrix is singular.

        Note that ?tbtrs requires the matrix A to be stored in banded form.
        In this form the diagonal corresponds to the last row."""
        ab = np.ones((3, 4), dtype=float)
        b = np.ones(4, dtype=float)
        tbtrs = get_lapack_funcs('tbtrs', dtype=float)

        ab[-1, 3] = 0
        _, info = tbtrs(ab=ab, b=b, uplo='U')
        assert_equal(info, 4)

    @pytest.mark.parametrize('ldab,n,ldb,nrhs', [
                              (5, 5, 0, 5),
                              (5, 5, 3, 5)
    ])
    def test_invalid_matrix_shapes(self, ldab, n, ldb, nrhs):
        """Test ?tbtrs fails correctly if shapes are invalid."""
        ab = np.ones((ldab, n), dtype=float)
        b = np.ones((ldb, nrhs), dtype=float)
        tbtrs = get_lapack_funcs('tbtrs', dtype=float)
        assert_raises(Exception, tbtrs, ab, b)


def test_lartg():
    for dtype in 'fdFD':
        lartg = get_lapack_funcs('lartg', dtype=dtype)

        f = np.array(3, dtype)
        g = np.array(4, dtype)

        if np.iscomplexobj(g):
            g *= 1j

        cs, sn, r = lartg(f, g)

        assert_allclose(cs, 3.0/5.0)
        assert_allclose(r, 5.0)

        if np.iscomplexobj(g):
            assert_allclose(sn, -4.0j/5.0)
            assert_(type(r) == complex)
            assert_(type(cs) == float)
        else:
            assert_allclose(sn, 4.0/5.0)


def test_rot():
    # srot, drot from blas and crot and zrot from lapack.

    for dtype in 'fdFD':
        c = 0.6
        s = 0.8

        u = np.full(4, 3, dtype)
        v = np.full(4, 4, dtype)
        atol = 10**-(np.finfo(dtype).precision-1)

        if dtype in 'fd':
            rot = get_blas_funcs('rot', dtype=dtype)
            f = 4
        else:
            rot = get_lapack_funcs('rot', dtype=dtype)
            s *= -1j
            v *= 1j
            f = 4j

        assert_allclose(rot(u, v, c, s), [[5, 5, 5, 5],
                                          [0, 0, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, n=2), [[5, 5, 3, 3],
                                               [0, 0, f, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, offy=2),
                        [[3, 3, 5, 5], [f, f, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, incx=2, offy=2, n=2),
                        [[5, 3, 5, 3], [f, f, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, incy=2, n=2),
                        [[3, 3, 5, 5], [0, f, 0, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, incx=2, offy=2, incy=2, n=1),
                        [[3, 3, 5, 3], [f, f, 0, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, incx=-2, incy=-2, n=2),
                        [[5, 3, 5, 3], [0, f, 0, f]], atol=atol)

        a, b = rot(u, v, c, s, overwrite_x=1, overwrite_y=1)
        assert_(a is u)
        assert_(b is v)
        assert_allclose(a, [5, 5, 5, 5], atol=atol)
        assert_allclose(b, [0, 0, 0, 0], atol=atol)


def test_larfg_larf():
    np.random.seed(1234)
    a0 = np.random.random((4, 4))
    a0 = a0.T.dot(a0)

    a0j = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
    a0j = a0j.T.conj().dot(a0j)

    # our test here will be to do one step of reducing a hermetian matrix to
    # tridiagonal form using householder transforms.

    for dtype in 'fdFD':
        larfg, larf = get_lapack_funcs(['larfg', 'larf'], dtype=dtype)

        if dtype in 'FD':
            a = a0j.copy()
        else:
            a = a0.copy()

        # generate a householder transform to clear a[2:,0]
        alpha, x, tau = larfg(a.shape[0]-1, a[1, 0], a[2:, 0])

        # create expected output
        expected = np.zeros_like(a[:, 0])
        expected[0] = a[0, 0]
        expected[1] = alpha

        # assemble householder vector
        v = np.zeros_like(a[1:, 0])
        v[0] = 1.0
        v[1:] = x

        # apply transform from the left
        a[1:, :] = larf(v, tau.conjugate(), a[1:, :], np.zeros(a.shape[1]))

        # apply transform from the right
        a[:, 1:] = larf(v, tau, a[:, 1:], np.zeros(a.shape[0]), side='R')

        assert_allclose(a[:, 0], expected, atol=1e-5)
        assert_allclose(a[0, :], expected, atol=1e-5)


@pytest.mark.xslow
def test_sgesdd_lwork_bug_workaround():
    # Test that SGESDD lwork is sufficiently large for LAPACK.
    #
    # This checks that workaround around an apparent LAPACK bug
    # actually works. cf. gh-5401
    #
    # xslow: requires 1GB+ of memory

    p = subprocess.Popen([sys.executable, '-c',
                          'import numpy as np; '
                          'from scipy.linalg import svd; '
                          'a = np.zeros([9537, 9537], dtype=np.float32); '
                          'svd(a)'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    # Check if it an error occurred within 5 sec; the computation can
    # take substantially longer, and we will not wait for it to finish
    for j in range(50):
        time.sleep(0.1)
        if p.poll() is not None:
            returncode = p.returncode
            break
    else:
        # Didn't exit in time -- probably entered computation.  The
        # error is raised before entering computation, so things are
        # probably OK.
        returncode = 0
        p.terminate()

    assert_equal(returncode, 0,
                 "Code apparently failed: " + p.stdout.read().decode())


class TestSytrd:
    @pytest.mark.parametrize('dtype', REAL_DTYPES)
    def test_sytrd_with_zero_dim_array(self, dtype):
        # Assert that a 0x0 matrix raises an error
        A = np.zeros((0, 0), dtype=dtype)
        sytrd = get_lapack_funcs('sytrd', (A,))
        assert_raises(ValueError, sytrd, A)

    @pytest.mark.parametrize('dtype', REAL_DTYPES)
    @pytest.mark.parametrize('n', (1, 3))
    def test_sytrd(self, dtype, n):
        A = np.zeros((n, n), dtype=dtype)

        sytrd, sytrd_lwork = \
            get_lapack_funcs(('sytrd', 'sytrd_lwork'), (A,))

        # some upper triangular array
        A[np.triu_indices_from(A)] = \
            np.arange(1, n*(n+1)//2+1, dtype=dtype)

        # query lwork
        lwork, info = sytrd_lwork(n)
        assert_equal(info, 0)

        # check lower=1 behavior (shouldn't do much since the matrix is
        # upper triangular)
        data, d, e, tau, info = sytrd(A, lower=1, lwork=lwork)
        assert_equal(info, 0)

        assert_allclose(data, A, atol=5*np.finfo(dtype).eps, rtol=1.0)
        assert_allclose(d, np.diag(A))
        assert_allclose(e, 0.0)
        assert_allclose(tau, 0.0)

        # and now for the proper test (lower=0 is the default)
        data, d, e, tau, info = sytrd(A, lwork=lwork)
        assert_equal(info, 0)

        # assert Q^T*A*Q = tridiag(e, d, e)

        # build tridiagonal matrix
        T = np.zeros_like(A, dtype=dtype)
        k = np.arange(A.shape[0])
        T[k, k] = d
        k2 = np.arange(A.shape[0]-1)
        T[k2+1, k2] = e
        T[k2, k2+1] = e

        # build Q
        Q = np.eye(n, n, dtype=dtype)
        for i in range(n-1):
            v = np.zeros(n, dtype=dtype)
            v[:i] = data[:i, i+1]
            v[i] = 1.0
            H = np.eye(n, n, dtype=dtype) - tau[i] * np.outer(v, v)
            Q = np.dot(H, Q)

        # Make matrix fully symmetric
        i_lower = np.tril_indices(n, -1)
        A[i_lower] = A.T[i_lower]

        QTAQ = np.dot(Q.T, np.dot(A, Q))

        # disable rtol here since some values in QTAQ and T are very close
        # to 0.
        assert_allclose(QTAQ, T, atol=5*np.finfo(dtype).eps, rtol=1.0)


class TestHetrd:
    @pytest.mark.parametrize('complex_dtype', COMPLEX_DTYPES)
    def test_hetrd_with_zero_dim_array(self, complex_dtype):
        # Assert that a 0x0 matrix raises an error
        A = np.zeros((0, 0), dtype=complex_dtype)
        hetrd = get_lapack_funcs('hetrd', (A,))
        assert_raises(ValueError, hetrd, A)

    @pytest.mark.parametrize('real_dtype,complex_dtype',
                             zip(REAL_DTYPES, COMPLEX_DTYPES))
    @pytest.mark.parametrize('n', (1, 3))
    def test_hetrd(self, n, real_dtype, complex_dtype):
        A = np.zeros((n, n), dtype=complex_dtype)
        hetrd, hetrd_lwork = \
            get_lapack_funcs(('hetrd', 'hetrd_lwork'), (A,))

        # some upper triangular array
        A[np.triu_indices_from(A)] = (
            np.arange(1, n*(n+1)//2+1, dtype=real_dtype)
            + 1j * np.arange(1, n*(n+1)//2+1, dtype=real_dtype)
            )
        np.fill_diagonal(A, np.real(np.diag(A)))

        # test query lwork
        for x in [0, 1]:
            _, info = hetrd_lwork(n, lower=x)
            assert_equal(info, 0)
        # lwork returns complex which segfaults hetrd call (gh-10388)
        # use the safe and recommended option
        lwork = _compute_lwork(hetrd_lwork, n)

        # check lower=1 behavior (shouldn't do much since the matrix is
        # upper triangular)
        data, d, e, tau, info = hetrd(A, lower=1, lwork=lwork)
        assert_equal(info, 0)

        assert_allclose(data, A, atol=5*np.finfo(real_dtype).eps, rtol=1.0)

        assert_allclose(d, np.real(np.diag(A)))
        assert_allclose(e, 0.0)
        assert_allclose(tau, 0.0)

        # and now for the proper test (lower=0 is the default)
        data, d, e, tau, info = hetrd(A, lwork=lwork)
        assert_equal(info, 0)

        # assert Q^T*A*Q = tridiag(e, d, e)

        # build tridiagonal matrix
        T = np.zeros_like(A, dtype=real_dtype)
        k = np.arange(A.shape[0], dtype=int)
        T[k, k] = d
        k2 = np.arange(A.shape[0]-1, dtype=int)
        T[k2+1, k2] = e
        T[k2, k2+1] = e

        # build Q
        Q = np.eye(n, n, dtype=complex_dtype)
        for i in range(n-1):
            v = np.zeros(n, dtype=complex_dtype)
            v[:i] = data[:i, i+1]
            v[i] = 1.0
            H = np.eye(n, n, dtype=complex_dtype) \
                - tau[i] * np.outer(v, np.conj(v))
            Q = np.dot(H, Q)

        # Make matrix fully Hermitian
        i_lower = np.tril_indices(n, -1)
        A[i_lower] = np.conj(A.T[i_lower])

        QHAQ = np.dot(np.conj(Q.T), np.dot(A, Q))

        # disable rtol here since some values in QTAQ and T are very close
        # to 0.
        assert_allclose(
            QHAQ, T, atol=10*np.finfo(real_dtype).eps, rtol=1.0
            )


def test_gglse():
    # Example data taken from NAG manual
    for ind, dtype in enumerate(DTYPES):
        # DTYPES = <s,d,c,z> gglse
        func, func_lwork = get_lapack_funcs(('gglse', 'gglse_lwork'),
                                            dtype=dtype)
        lwork = _compute_lwork(func_lwork, m=6, n=4, p=2)
        # For <s,d>gglse
        if ind < 2:
            a = np.array([[-0.57, -1.28, -0.39, 0.25],
                          [-1.93, 1.08, -0.31, -2.14],
                          [2.30, 0.24, 0.40, -0.35],
                          [-1.93, 0.64, -0.66, 0.08],
                          [0.15, 0.30, 0.15, -2.13],
                          [-0.02, 1.03, -1.43, 0.50]], dtype=dtype)
            c = np.array([-1.50, -2.14, 1.23, -0.54, -1.68, 0.82], dtype=dtype)
            d = np.array([0., 0.], dtype=dtype)
        # For <s,d>gglse
        else:
            a = np.array([[0.96-0.81j, -0.03+0.96j, -0.91+2.06j, -0.05+0.41j],
                          [-0.98+1.98j, -1.20+0.19j, -0.66+0.42j, -0.81+0.56j],
                          [0.62-0.46j, 1.01+0.02j, 0.63-0.17j, -1.11+0.60j],
                          [0.37+0.38j, 0.19-0.54j, -0.98-0.36j, 0.22-0.20j],
                          [0.83+0.51j, 0.20+0.01j, -0.17-0.46j, 1.47+1.59j],
                          [1.08-0.28j, 0.20-0.12j, -0.07+1.23j, 0.26+0.26j]])
            c = np.array([[-2.54+0.09j],
                          [1.65-2.26j],
                          [-2.11-3.96j],
                          [1.82+3.30j],
                          [-6.41+3.77j],
                          [2.07+0.66j]])
            d = np.zeros(2, dtype=dtype)

        b = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.]], dtype=dtype)

        _, _, _, result, _ = func(a, b, c, d, lwork=lwork)
        if ind < 2:
            expected = np.array([0.48904455,
                                 0.99754786,
                                 0.48904455,
                                 0.99754786])
        else:
            expected = np.array([1.08742917-1.96205783j,
                                 -0.74093902+3.72973919j,
                                 1.08742917-1.96205759j,
                                 -0.74093896+3.72973895j])
        assert_array_almost_equal(result, expected, decimal=4)


def test_sycon_hecon():
    seed(1234)
    for ind, dtype in enumerate(DTYPES+COMPLEX_DTYPES):
        # DTYPES + COMPLEX DTYPES = <s,d,c,z> sycon + <c,z>hecon
        n = 10
        # For <s,d,c,z>sycon
        if ind < 4:
            func_lwork = get_lapack_funcs('sytrf_lwork', dtype=dtype)
            funcon, functrf = get_lapack_funcs(('sycon', 'sytrf'), dtype=dtype)
            A = (rand(n, n)).astype(dtype)
        # For <c,z>hecon
        else:
            func_lwork = get_lapack_funcs('hetrf_lwork', dtype=dtype)
            funcon, functrf = get_lapack_funcs(('hecon', 'hetrf'), dtype=dtype)
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)

        # Since sycon only refers to upper/lower part, conj() is safe here.
        A = (A + A.conj().T)/2 + 2*np.eye(n, dtype=dtype)

        anorm = norm(A, 1)
        lwork = _compute_lwork(func_lwork, n)
        ldu, ipiv, _ = functrf(A, lwork=lwork, lower=1)
        rcond, _ = funcon(a=ldu, ipiv=ipiv, anorm=anorm, lower=1)
        # The error is at most 1-fold
        assert_(abs(1/rcond - np.linalg.cond(A, p=1))*rcond < 1)


def test_sygst():
    seed(1234)
    for ind, dtype in enumerate(REAL_DTYPES):
        # DTYPES = <s,d> sygst
        n = 10

        potrf, sygst, syevd, sygvd = get_lapack_funcs(('potrf', 'sygst',
                                                       'syevd', 'sygvd'),
                                                      dtype=dtype)

        A = rand(n, n).astype(dtype)
        A = (A + A.T)/2
        # B must be positive definite
        B = rand(n, n).astype(dtype)
        B = (B + B.T)/2 + 2 * np.eye(n, dtype=dtype)

        # Perform eig (sygvd)
        eig_gvd, _, info = sygvd(A, B)
        assert_(info == 0)

        # Convert to std problem potrf
        b, info = potrf(B)
        assert_(info == 0)
        a, info = sygst(A, b)
        assert_(info == 0)

        eig, _, info = syevd(a)
        assert_(info == 0)
        assert_allclose(eig, eig_gvd, rtol=1e-4)


def test_hegst():
    seed(1234)
    for ind, dtype in enumerate(COMPLEX_DTYPES):
        # DTYPES = <c,z> hegst
        n = 10

        potrf, hegst, heevd, hegvd = get_lapack_funcs(('potrf', 'hegst',
                                                       'heevd', 'hegvd'),
                                                      dtype=dtype)

        A = rand(n, n).astype(dtype) + 1j * rand(n, n).astype(dtype)
        A = (A + A.conj().T)/2
        # B must be positive definite
        B = rand(n, n).astype(dtype) + 1j * rand(n, n).astype(dtype)
        B = (B + B.conj().T)/2 + 2 * np.eye(n, dtype=dtype)

        # Perform eig (hegvd)
        eig_gvd, _, info = hegvd(A, B)
        assert_(info == 0)

        # Convert to std problem potrf
        b, info = potrf(B)
        assert_(info == 0)
        a, info = hegst(A, b)
        assert_(info == 0)

        eig, _, info = heevd(a)
        assert_(info == 0)
        assert_allclose(eig, eig_gvd, rtol=1e-4)


def test_tzrzf():
    """
    This test performs an RZ decomposition in which an m x n upper trapezoidal
    array M (m <= n) is factorized as M = [R 0] * Z where R is upper triangular
    and Z is unitary.
    """
    seed(1234)
    m, n = 10, 15
    for ind, dtype in enumerate(DTYPES):
        tzrzf, tzrzf_lw = get_lapack_funcs(('tzrzf', 'tzrzf_lwork'),
                                           dtype=dtype)
        lwork = _compute_lwork(tzrzf_lw, m, n)

        if ind < 2:
            A = triu(rand(m, n).astype(dtype))
        else:
            A = triu((rand(m, n) + rand(m, n)*1j).astype(dtype))

        # assert wrong shape arg, f2py returns generic error
        assert_raises(Exception, tzrzf, A.T)
        rz, tau, info = tzrzf(A, lwork=lwork)
        # Check success
        assert_(info == 0)

        # Get Z manually for comparison
        R = np.hstack((rz[:, :m], np.zeros((m, n-m), dtype=dtype)))
        V = np.hstack((np.eye(m, dtype=dtype), rz[:, m:]))
        Id = np.eye(n, dtype=dtype)
        ref = [Id-tau[x]*V[[x], :].T.dot(V[[x], :].conj()) for x in range(m)]
        Z = reduce(np.dot, ref)
        assert_allclose(R.dot(Z) - A, zeros_like(A, dtype=dtype),
                        atol=10*np.spacing(dtype(1.0).real), rtol=0.)


def test_tfsm():
    """
    Test for solving a linear system with the coefficient matrix is a
    triangular array stored in Full Packed (RFP) format.
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = triu(rand(n, n) + rand(n, n)*1j + eye(n)).astype(dtype)
            trans = 'C'
        else:
            A = triu(rand(n, n) + eye(n)).astype(dtype)
            trans = 'T'

        trttf, tfttr, tfsm = get_lapack_funcs(('trttf', 'tfttr', 'tfsm'),
                                              dtype=dtype)

        Afp, _ = trttf(A)
        B = rand(n, 2).astype(dtype)
        soln = tfsm(-1, Afp, B)
        assert_array_almost_equal(soln, solve(-A, B),
                                  decimal=4 if ind % 2 == 0 else 6)

        soln = tfsm(-1, Afp, B, trans=trans)
        assert_array_almost_equal(soln, solve(-A.conj().T, B),
                                  decimal=4 if ind % 2 == 0 else 6)

        # Make A, unit diagonal
        A[np.arange(n), np.arange(n)] = dtype(1.)
        soln = tfsm(-1, Afp, B, trans=trans, diag='U')
        assert_array_almost_equal(soln, solve(-A.conj().T, B),
                                  decimal=4 if ind % 2 == 0 else 6)

        # Change side
        B2 = rand(3, n).astype(dtype)
        soln = tfsm(-1, Afp, B2, trans=trans, diag='U', side='R')
        assert_array_almost_equal(soln, solve(-A, B2.T).conj().T,
                                  decimal=4 if ind % 2 == 0 else 6)


def test_ormrz_unmrz():
    """
    This test performs a matrix multiplication with an arbitrary m x n matric C
    and a unitary matrix Q without explicitly forming the array. The array data
    is encoded in the rectangular part of A which is obtained from ?TZRZF. Q
    size is inferred by m, n, side keywords.
    """
    seed(1234)
    qm, qn, cn = 10, 15, 15
    for ind, dtype in enumerate(DTYPES):
        tzrzf, tzrzf_lw = get_lapack_funcs(('tzrzf', 'tzrzf_lwork'),
                                           dtype=dtype)
        lwork_rz = _compute_lwork(tzrzf_lw, qm, qn)

        if ind < 2:
            A = triu(rand(qm, qn).astype(dtype))
            C = rand(cn, cn).astype(dtype)
            orun_mrz, orun_mrz_lw = get_lapack_funcs(('ormrz', 'ormrz_lwork'),
                                                     dtype=dtype)
        else:
            A = triu((rand(qm, qn) + rand(qm, qn)*1j).astype(dtype))
            C = (rand(cn, cn) + rand(cn, cn)*1j).astype(dtype)
            orun_mrz, orun_mrz_lw = get_lapack_funcs(('unmrz', 'unmrz_lwork'),
                                                     dtype=dtype)

        lwork_mrz = _compute_lwork(orun_mrz_lw, cn, cn)
        rz, tau, info = tzrzf(A, lwork=lwork_rz)

        # Get Q manually for comparison
        V = np.hstack((np.eye(qm, dtype=dtype), rz[:, qm:]))
        Id = np.eye(qn, dtype=dtype)
        ref = [Id-tau[x]*V[[x], :].T.dot(V[[x], :].conj()) for x in range(qm)]
        Q = reduce(np.dot, ref)

        # Now that we have Q, we can test whether lapack results agree with
        # each case of CQ, CQ^H, QC, and QC^H
        trans = 'T' if ind < 2 else 'C'
        tol = 10*np.spacing(dtype(1.0).real)

        cq, info = orun_mrz(rz, tau, C, lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - Q.dot(C), zeros_like(C), atol=tol, rtol=0.)

        cq, info = orun_mrz(rz, tau, C, trans=trans, lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - Q.conj().T.dot(C), zeros_like(C), atol=tol,
                        rtol=0.)

        cq, info = orun_mrz(rz, tau, C, side='R', lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - C.dot(Q), zeros_like(C), atol=tol, rtol=0.)

        cq, info = orun_mrz(rz, tau, C, side='R', trans=trans, lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - C.dot(Q.conj().T), zeros_like(C), atol=tol,
                        rtol=0.)


def test_tfttr_trttf():
    """
    Test conversion routines between the Rectengular Full Packed (RFP) format
    and Standard Triangular Array (TR)
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A_full = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            transr = 'C'
        else:
            A_full = (rand(n, n)).astype(dtype)
            transr = 'T'

        trttf, tfttr = get_lapack_funcs(('trttf', 'tfttr'), dtype=dtype)
        A_tf_U, info = trttf(A_full)
        assert_(info == 0)
        A_tf_L, info = trttf(A_full, uplo='L')
        assert_(info == 0)
        A_tf_U_T, info = trttf(A_full, transr=transr, uplo='U')
        assert_(info == 0)
        A_tf_L_T, info = trttf(A_full, transr=transr, uplo='L')
        assert_(info == 0)

        # Create the RFP array manually (n is even!)
        A_tf_U_m = zeros((n+1, n//2), dtype=dtype)
        A_tf_U_m[:-1, :] = triu(A_full)[:, n//2:]
        A_tf_U_m[n//2+1:, :] += triu(A_full)[:n//2, :n//2].conj().T

        A_tf_L_m = zeros((n+1, n//2), dtype=dtype)
        A_tf_L_m[1:, :] = tril(A_full)[:, :n//2]
        A_tf_L_m[:n//2, :] += tril(A_full)[n//2:, n//2:].conj().T

        assert_array_almost_equal(A_tf_U, A_tf_U_m.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_U_T,
                                  A_tf_U_m.conj().T.reshape(-1, order='F'))

        assert_array_almost_equal(A_tf_L, A_tf_L_m.reshape(-1, order='F'))
        assert_array_almost_equal(A_tf_L_T,
                                  A_tf_L_m.conj().T.reshape(-1, order='F'))

        # Get the original array from RFP
        A_tr_U, info = tfttr(n, A_tf_U)
        assert_(info == 0)
        A_tr_L, info = tfttr(n, A_tf_L, uplo='L')
        assert_(info == 0)
        A_tr_U_T, info = tfttr(n, A_tf_U_T, transr=transr, uplo='U')
        assert_(info == 0)
        A_tr_L_T, info = tfttr(n, A_tf_L_T, transr=transr, uplo='L')
        assert_(info == 0)

        assert_array_almost_equal(A_tr_U, triu(A_full))
        assert_array_almost_equal(A_tr_U_T, triu(A_full))
        assert_array_almost_equal(A_tr_L, tril(A_full))
        assert_array_almost_equal(A_tr_L_T, tril(A_full))


def test_tpttr_trttp():
    """
    Test conversion routines between the Rectengular Full Packed (RFP) format
    and Standard Triangular Array (TR)
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A_full = (rand(n, n) + rand(n, n)*1j).astype(dtype)
        else:
            A_full = (rand(n, n)).astype(dtype)

        trttp, tpttr = get_lapack_funcs(('trttp', 'tpttr'), dtype=dtype)
        A_tp_U, info = trttp(A_full)
        assert_(info == 0)
        A_tp_L, info = trttp(A_full, uplo='L')
        assert_(info == 0)

        # Create the TP array manually
        inds = tril_indices(n)
        A_tp_U_m = zeros(n*(n+1)//2, dtype=dtype)
        A_tp_U_m[:] = (triu(A_full).T)[inds]

        inds = triu_indices(n)
        A_tp_L_m = zeros(n*(n+1)//2, dtype=dtype)
        A_tp_L_m[:] = (tril(A_full).T)[inds]

        assert_array_almost_equal(A_tp_U, A_tp_U_m)
        assert_array_almost_equal(A_tp_L, A_tp_L_m)

        # Get the original array from TP
        A_tr_U, info = tpttr(n, A_tp_U)
        assert_(info == 0)
        A_tr_L, info = tpttr(n, A_tp_L, uplo='L')
        assert_(info == 0)

        assert_array_almost_equal(A_tr_U, triu(A_full))
        assert_array_almost_equal(A_tr_L, tril(A_full))


def test_pftrf():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n*eye(n)
        else:
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        pftrf, trttf, tfttr = get_lapack_funcs(('pftrf', 'trttf', 'tfttr'),
                                               dtype=dtype)

        # Get the original array from TP
        Afp, info = trttf(A)
        Achol_rfp, info = pftrf(n, Afp)
        assert_(info == 0)
        A_chol_r, _ = tfttr(n, Achol_rfp)
        Achol = cholesky(A)
        assert_array_almost_equal(A_chol_r, Achol)


def test_pftri():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array to find its inverse
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n*eye(n)
        else:
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        pftri, pftrf, trttf, tfttr = get_lapack_funcs(('pftri',
                                                       'pftrf',
                                                       'trttf',
                                                       'tfttr'),
                                                      dtype=dtype)

        # Get the original array from TP
        Afp, info = trttf(A)
        A_chol_rfp, info = pftrf(n, Afp)
        A_inv_rfp, info = pftri(n, A_chol_rfp)
        assert_(info == 0)
        A_inv_r, _ = tfttr(n, A_inv_rfp)
        Ainv = inv(A)
        assert_array_almost_equal(A_inv_r, triu(Ainv),
                                  decimal=4 if ind % 2 == 0 else 6)


def test_pftrs():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array and solve a linear system
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n*eye(n)
        else:
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        B = ones((n, 3), dtype=dtype)
        Bf1 = ones((n+2, 3), dtype=dtype)
        Bf2 = ones((n-2, 3), dtype=dtype)
        pftrs, pftrf, trttf, tfttr = get_lapack_funcs(('pftrs',
                                                       'pftrf',
                                                       'trttf',
                                                       'tfttr'),
                                                      dtype=dtype)

        # Get the original array from TP
        Afp, info = trttf(A)
        A_chol_rfp, info = pftrf(n, Afp)
        # larger B arrays shouldn't segfault
        soln, info = pftrs(n, A_chol_rfp, Bf1)
        assert_(info == 0)
        assert_raises(Exception, pftrs, n, A_chol_rfp, Bf2)
        soln, info = pftrs(n, A_chol_rfp, B)
        assert_(info == 0)
        assert_array_almost_equal(solve(A, B), soln,
                                  decimal=4 if ind % 2 == 0 else 6)


def test_sfrk_hfrk():
    """
    Test for performing a symmetric rank-k operation for matrix in RFP format.
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            A = A + A.conj().T + n*eye(n)
        else:
            A = (rand(n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        prefix = 's'if ind < 2 else 'h'
        trttf, tfttr, shfrk = get_lapack_funcs(('trttf', 'tfttr', '{}frk'
                                                ''.format(prefix)),
                                               dtype=dtype)

        Afp, _ = trttf(A)
        C = np.random.rand(n, 2).astype(dtype)
        Afp_out = shfrk(n, 2, -1, C, 2, Afp)
        A_out, _ = tfttr(n, Afp_out)
        assert_array_almost_equal(A_out, triu(-C.dot(C.conj().T) + 2*A),
                                  decimal=4 if ind % 2 == 0 else 6)


def test_syconv():
    """
    Test for going back and forth between the returned format of he/sytrf to
    L and D factors/permutations.
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 10

        if ind > 1:
            A = (randint(-30, 30, (n, n)) +
                 randint(-30, 30, (n, n))*1j).astype(dtype)

            A = A + A.conj().T
        else:
            A = randint(-30, 30, (n, n)).astype(dtype)
            A = A + A.T + n*eye(n)

        tol = 100*np.spacing(dtype(1.0).real)
        syconv, trf, trf_lwork = get_lapack_funcs(('syconv', 'sytrf',
                                                   'sytrf_lwork'), dtype=dtype)
        lw = _compute_lwork(trf_lwork, n, lower=1)
        L, D, perm = ldl(A, lower=1, hermitian=False)
        lw = _compute_lwork(trf_lwork, n, lower=1)
        ldu, ipiv, info = trf(A, lower=1, lwork=lw)
        a, e, info = syconv(ldu, ipiv, lower=1)
        assert_allclose(tril(a, -1,), tril(L[perm, :], -1), atol=tol, rtol=0.)

        # Test also upper
        U, D, perm = ldl(A, lower=0, hermitian=False)
        ldu, ipiv, info = trf(A, lower=0)
        a, e, info = syconv(ldu, ipiv, lower=0)
        assert_allclose(triu(a, 1), triu(U[perm, :], 1), atol=tol, rtol=0.)


class TestBlockedQR:
    """
    Tests for the blocked QR factorization, namely through geqrt, gemqrt, tpqrt
    and tpmqr.
    """

    def test_geqrt_gemqrt(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 20

            if ind > 1:
                A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            else:
                A = (rand(n, n)).astype(dtype)

            tol = 100*np.spacing(dtype(1.0).real)
            geqrt, gemqrt = get_lapack_funcs(('geqrt', 'gemqrt'), dtype=dtype)

            a, t, info = geqrt(n, A)
            assert(info == 0)

            # Extract elementary reflectors from lower triangle, adding the
            # main diagonal of ones.
            v = np.tril(a, -1) + np.eye(n, dtype=dtype)
            # Generate the block Householder transform I - VTV^H
            Q = np.eye(n, dtype=dtype) - v @ t @ v.T.conj()
            R = np.triu(a)

            # Test columns of Q are orthogonal
            assert_allclose(Q.T.conj() @ Q, np.eye(n, dtype=dtype), atol=tol,
                            rtol=0.)
            assert_allclose(Q @ R, A, atol=tol, rtol=0.)

            if ind > 1:
                C = (rand(n, n) + rand(n, n)*1j).astype(dtype)
                transpose = 'C'
            else:
                C = (rand(n, n)).astype(dtype)
                transpose = 'T'

            for side in ('L', 'R'):
                for trans in ('N', transpose):
                    c, info = gemqrt(a, t, C, side=side, trans=trans)
                    assert(info == 0)

                    if trans == transpose:
                        q = Q.T.conj()
                    else:
                        q = Q

                    if side == 'L':
                        qC = q @ C
                    else:
                        qC = C @ q

                    assert_allclose(c, qC, atol=tol, rtol=0.)

                    # Test default arguments
                    if (side, trans) == ('L', 'N'):
                        c_default, info = gemqrt(a, t, C)
                        assert(info == 0)
                        assert_equal(c_default, c)

            # Test invalid side/trans
            assert_raises(Exception, gemqrt, a, t, C, side='A')
            assert_raises(Exception, gemqrt, a, t, C, trans='A')

    def test_tpqrt_tpmqrt(self):
        seed(1234)
        for ind, dtype in enumerate(DTYPES):
            n = 20

            if ind > 1:
                A = (rand(n, n) + rand(n, n)*1j).astype(dtype)
                B = (rand(n, n) + rand(n, n)*1j).astype(dtype)
            else:
                A = (rand(n, n)).astype(dtype)
                B = (rand(n, n)).astype(dtype)

            tol = 100*np.spacing(dtype(1.0).real)
            tpqrt, tpmqrt = get_lapack_funcs(('tpqrt', 'tpmqrt'), dtype=dtype)

            # Test for the range of pentagonal B, from square to upper
            # triangular
            for l in (0, n // 2, n):
                a, b, t, info = tpqrt(l, n, A, B)
                assert(info == 0)

                # Check that lower triangular part of A has not been modified
                assert_equal(np.tril(a, -1), np.tril(A, -1))
                # Check that elements not part of the pentagonal portion of B
                # have not been modified.
                assert_equal(np.tril(b, l - n - 1), np.tril(B, l - n - 1))

                # Extract pentagonal portion of B
                B_pent, b_pent = np.triu(B, l - n), np.triu(b, l - n)

                # Generate elementary reflectors
                v = np.concatenate((np.eye(n, dtype=dtype), b_pent))
                # Generate the block Householder transform I - VTV^H
                Q = np.eye(2 * n, dtype=dtype) - v @ t @ v.T.conj()
                R = np.concatenate((np.triu(a), np.zeros_like(a)))

                # Test columns of Q are orthogonal
                assert_allclose(Q.T.conj() @ Q, np.eye(2 * n, dtype=dtype),
                                atol=tol, rtol=0.)
                assert_allclose(Q @ R, np.concatenate((np.triu(A), B_pent)),
                                atol=tol, rtol=0.)

                if ind > 1:
                    C = (rand(n, n) + rand(n, n)*1j).astype(dtype)
                    D = (rand(n, n) + rand(n, n)*1j).astype(dtype)
                    transpose = 'C'
                else:
                    C = (rand(n, n)).astype(dtype)
                    D = (rand(n, n)).astype(dtype)
                    transpose = 'T'

                for side in ('L', 'R'):
                    for trans in ('N', transpose):
                        c, d, info = tpmqrt(l, b, t, C, D, side=side,
                                            trans=trans)
                        assert(info == 0)

                        if trans == transpose:
                            q = Q.T.conj()
                        else:
                            q = Q

                        if side == 'L':
                            cd = np.concatenate((c, d), axis=0)
                            CD = np.concatenate((C, D), axis=0)
                            qCD = q @ CD
                        else:
                            cd = np.concatenate((c, d), axis=1)
                            CD = np.concatenate((C, D), axis=1)
                            qCD = CD @ q

                        assert_allclose(cd, qCD, atol=tol, rtol=0.)

                        if (side, trans) == ('L', 'N'):
                            c_default, d_default, info = tpmqrt(l, b, t, C, D)
                            assert(info == 0)
                            assert_equal(c_default, c)
                            assert_equal(d_default, d)

                # Test invalid side/trans
                assert_raises(Exception, tpmqrt, l, b, t, C, D, side='A')
                assert_raises(Exception, tpmqrt, l, b, t, C, D, trans='A')


def test_pstrf():
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        # DTYPES = <s, d, c, z> pstrf
        n = 10
        r = 2
        pstrf = get_lapack_funcs('pstrf', dtype=dtype)

        # Create positive semidefinite A
        if ind > 1:
            A = rand(n, n-r).astype(dtype) + 1j * rand(n, n-r).astype(dtype)
            A = A @ A.conj().T
        else:
            A = rand(n, n-r).astype(dtype)
            A = A @ A.T

        c, piv, r_c, info = pstrf(A)
        U = triu(c)
        U[r_c - n:, r_c - n:] = 0.

        assert_equal(info, 1)
        # python-dbg 3.5.2 runs cause trouble with the following assertion.
        # assert_equal(r_c, n - r)
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        assert_allclose(A[piv-1][:, piv-1], U.conj().T @ U, rtol=0., atol=atol)

        c, piv, r_c, info = pstrf(A, lower=1)
        L = tril(c)
        L[r_c - n:, r_c - n:] = 0.

        assert_equal(info, 1)
        # assert_equal(r_c, n - r)
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        assert_allclose(A[piv-1][:, piv-1], L @ L.conj().T, rtol=0., atol=atol)


def test_pstf2():
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        # DTYPES = <s, d, c, z> pstf2
        n = 10
        r = 2
        pstf2 = get_lapack_funcs('pstf2', dtype=dtype)

        # Create positive semidefinite A
        if ind > 1:
            A = rand(n, n-r).astype(dtype) + 1j * rand(n, n-r).astype(dtype)
            A = A @ A.conj().T
        else:
            A = rand(n, n-r).astype(dtype)
            A = A @ A.T

        c, piv, r_c, info = pstf2(A)
        U = triu(c)
        U[r_c - n:, r_c - n:] = 0.

        assert_equal(info, 1)
        # python-dbg 3.5.2 runs cause trouble with the commented assertions.
        # assert_equal(r_c, n - r)
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        assert_allclose(A[piv-1][:, piv-1], U.conj().T @ U, rtol=0., atol=atol)

        c, piv, r_c, info = pstf2(A, lower=1)
        L = tril(c)
        L[r_c - n:, r_c - n:] = 0.

        assert_equal(info, 1)
        # assert_equal(r_c, n - r)
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        assert_allclose(A[piv-1][:, piv-1], L @ L.conj().T, rtol=0., atol=atol)


def test_geequ():
    desired_real = np.array([[0.6250, 1.0000, 0.0393, -0.4269],
                             [1.0000, -0.5619, -1.0000, -1.0000],
                             [0.5874, -1.0000, -0.0596, -0.5341],
                             [-1.0000, -0.5946, -0.0294, 0.9957]])

    desired_cplx = np.array([[-0.2816+0.5359*1j,
                              0.0812+0.9188*1j,
                              -0.7439-0.2561*1j],
                             [-0.3562-0.2954*1j,
                              0.9566-0.0434*1j,
                              -0.0174+0.1555*1j],
                             [0.8607+0.1393*1j,
                              -0.2759+0.7241*1j,
                              -0.1642-0.1365*1j]])

    for ind, dtype in enumerate(DTYPES):
        if ind < 2:
            # Use examples from the NAG documentation
            A = np.array([[1.80e+10, 2.88e+10, 2.05e+00, -8.90e+09],
                          [5.25e+00, -2.95e+00, -9.50e-09, -3.80e+00],
                          [1.58e+00, -2.69e+00, -2.90e-10, -1.04e+00],
                          [-1.11e+00, -6.60e-01, -5.90e-11, 8.00e-01]])
            A = A.astype(dtype)
        else:
            A = np.array([[-1.34e+00, 0.28e+10, -6.39e+00],
                          [-1.70e+00, 3.31e+10, -0.15e+00],
                          [2.41e-10, -0.56e+00, -0.83e-10]], dtype=dtype)
            A += np.array([[2.55e+00, 3.17e+10, -2.20e+00],
                           [-1.41e+00, -0.15e+10, 1.34e+00],
                           [0.39e-10, 1.47e+00, -0.69e-10]])*1j

            A = A.astype(dtype)

        geequ = get_lapack_funcs('geequ', dtype=dtype)
        r, c, rowcnd, colcnd, amax, info = geequ(A)

        if ind < 2:
            assert_allclose(desired_real.astype(dtype), r[:, None]*A*c,
                            rtol=0, atol=1e-4)
        else:
            assert_allclose(desired_cplx.astype(dtype), r[:, None]*A*c,
                            rtol=0, atol=1e-4)


def test_syequb():
    desired_log2s = np.array([0, 0, 0, 0, 0, 0, -1, -1, -2, -3])

    for ind, dtype in enumerate(DTYPES):
        A = np.eye(10, dtype=dtype)
        alpha = dtype(1. if ind < 2 else 1.j)
        d = np.array([alpha * 2.**x for x in range(-5, 5)], dtype=dtype)
        A += np.rot90(np.diag(d))

        syequb = get_lapack_funcs('syequb', dtype=dtype)
        s, scond, amax, info = syequb(A)

        assert_equal(np.log2(s).astype(int), desired_log2s)


@pytest.mark.skipif(True,
                    reason="Failing on some OpenBLAS version, see gh-12276")
def test_heequb():
    # zheequb has a bug for versions =< LAPACK 3.9.0
    # See Reference-LAPACK gh-61 and gh-408
    # Hence the zheequb test is customized accordingly to avoid
    # work scaling.
    A = np.diag([2]*5 + [1002]*5) + np.diag(np.ones(9), k=1)*1j
    s, scond, amax, info = lapack.zheequb(A)
    assert_equal(info, 0)
    assert_allclose(np.log2(s), [0., -1.]*2 + [0.] + [-4]*5)

    A = np.diag(2**np.abs(np.arange(-5, 6)) + 0j)
    A[5, 5] = 1024
    A[5, 0] = 16j
    s, scond, amax, info = lapack.cheequb(A.astype(np.complex64), lower=1)
    assert_equal(info, 0)
    assert_allclose(np.log2(s), [-2, -1, -1, 0, 0, -5, 0, -1, -1, -2, -2])


def test_getc2_gesc2():
    np.random.seed(42)
    n = 10
    desired_real = np.random.rand(n)
    desired_cplx = np.random.rand(n) + np.random.rand(n)*1j

    for ind, dtype in enumerate(DTYPES):
        if ind < 2:
            A = np.random.rand(n, n)
            A = A.astype(dtype)
            b = A @ desired_real
            b = b.astype(dtype)
        else:
            A = np.random.rand(n, n) + np.random.rand(n, n)*1j
            A = A.astype(dtype)
            b = A @ desired_cplx
            b = b.astype(dtype)

        getc2 = get_lapack_funcs('getc2', dtype=dtype)
        gesc2 = get_lapack_funcs('gesc2', dtype=dtype)
        lu, ipiv, jpiv, info = getc2(A, overwrite_a=0)
        x, scale = gesc2(lu, b, ipiv, jpiv, overwrite_rhs=0)

        if ind < 2:
            assert_array_almost_equal(desired_real.astype(dtype),
                                      x/scale, decimal=4)
        else:
            assert_array_almost_equal(desired_cplx.astype(dtype),
                                      x/scale, decimal=4)


@pytest.mark.parametrize('size', [(6, 5), (5, 5)])
@pytest.mark.parametrize('dtype', REAL_DTYPES)
@pytest.mark.parametrize('joba', range(6))  # 'C', 'E', 'F', 'G', 'A', 'R'
@pytest.mark.parametrize('jobu', range(4))  # 'U', 'F', 'W', 'N'
@pytest.mark.parametrize('jobv', range(4))  # 'V', 'J', 'W', 'N'
@pytest.mark.parametrize('jobr', [0, 1])
@pytest.mark.parametrize('jobp', [0, 1])
def test_gejsv_general(size, dtype, joba, jobu, jobv, jobr, jobp, jobt=0):
    """Test the lapack routine ?gejsv.

    This function tests that a singular value decomposition can be performed
    on the random M-by-N matrix A. The test performs the SVD using ?gejsv
    then performs the following checks:

    * ?gejsv exist successfully (info == 0)
    * The returned singular values are correct
    * `A` can be reconstructed from `u`, `SIGMA`, `v`
    * Ensure that u.T @ u is the identity matrix
    * Ensure that v.T @ v is the identity matrix
    * The reported matrix rank
    * The reported number of singular values
    * If denormalized floats are required

    Notes
    -----
    joba specifies several choices effecting the calculation's accuracy
    Although all arguments are tested, the tests only check that the correct
    solution is returned - NOT that the prescribed actions are performed
    internally.

    jobt is, as of v3.9.0, still experimental and removed to cut down number of
    test cases. However keyword itself is tested externally.
    """
    seed(42)

    # Define some constants for later use:
    m, n = size
    atol = 100 * np.finfo(dtype).eps
    A = generate_random_dtype_array(size, dtype)
    gejsv = get_lapack_funcs('gejsv', dtype=dtype)

    # Set up checks for invalid job? combinations
    # if an invalid combination occurs we set the appropriate
    # exit status.
    lsvec = jobu < 2  # Calculate left singular vectors
    rsvec = jobv < 2  # Calculate right singular vectors
    l2tran = (jobt == 1) and (m == n)
    is_complex = np.iscomplexobj(A)

    invalid_real_jobv = (jobv == 1) and (not lsvec) and (not is_complex)
    invalid_cplx_jobu = (jobu == 2) and not (rsvec and l2tran) and is_complex
    invalid_cplx_jobv = (jobv == 2) and not (lsvec and l2tran) and is_complex

    # Set the exit status to the expected value.
    # Here we only check for invalid combinations, not individual
    # parameters.
    if invalid_cplx_jobu:
        exit_status = -2
    elif invalid_real_jobv or invalid_cplx_jobv:
        exit_status = -3
    else:
        exit_status = 0

    if (jobu > 1) and (jobv == 1):
        assert_raises(Exception, gejsv, A, joba, jobu, jobv, jobr, jobt, jobp)
    else:
        sva, u, v, work, iwork, info = gejsv(A,
                                             joba=joba,
                                             jobu=jobu,
                                             jobv=jobv,
                                             jobr=jobr,
                                             jobt=jobt,
                                             jobp=jobp)

        # Check that ?gejsv exited successfully/as expected
        assert_equal(info, exit_status)

        # If exit_status is non-zero the combination of jobs is invalid.
        # We test this above but no calculations are performed.
        if not exit_status:

            # Check the returned singular values
            sigma = (work[0] / work[1]) * sva[:n]
            assert_allclose(sigma, svd(A, compute_uv=False), atol=atol)

            if jobu == 1:
                # If JOBU = 'F', then u contains the M-by-M matrix of
                # the left singular vectors, including an ONB of the orthogonal
                # complement of the Range(A)
                # However, to recalculate A we are concerned about the
                # first n singular values and so can ignore the latter.
                # TODO: Add a test for ONB?
                u = u[:, :n]

            if lsvec and rsvec:
                assert_allclose(u @ np.diag(sigma) @ v.conj().T, A, atol=atol)
            if lsvec:
                assert_allclose(u.conj().T @ u, np.identity(n), atol=atol)
            if rsvec:
                assert_allclose(v.conj().T @ v, np.identity(n), atol=atol)

            assert_equal(iwork[0], np.linalg.matrix_rank(A))
            assert_equal(iwork[1], np.count_nonzero(sigma))
            # iwork[2] is non-zero if requested accuracy is not warranted for
            # the data. This should never occur for these tests.
            assert_equal(iwork[2], 0)


@pytest.mark.parametrize('dtype', REAL_DTYPES)
def test_gejsv_edge_arguments(dtype):
    """Test edge arguments return expected status"""
    gejsv = get_lapack_funcs('gejsv', dtype=dtype)

    # scalar A
    sva, u, v, work, iwork, info = gejsv(1.)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 1))
    assert_equal(v.shape, (1, 1))
    assert_equal(sva, np.array([1.], dtype=dtype))

    # 1d A
    A = np.ones((1,), dtype=dtype)
    sva, u, v, work, iwork, info = gejsv(A)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 1))
    assert_equal(v.shape, (1, 1))
    assert_equal(sva, np.array([1.], dtype=dtype))

    # 2d empty A
    A = np.ones((1, 0), dtype=dtype)
    sva, u, v, work, iwork, info = gejsv(A)
    assert_equal(info, 0)
    assert_equal(u.shape, (1, 0))
    assert_equal(v.shape, (1, 0))
    assert_equal(sva, np.array([], dtype=dtype))

    # make sure "overwrite_a" is respected - user reported in gh-13191
    A = np.sin(np.arange(100).reshape(10, 10)).astype(dtype)
    A = np.asfortranarray(A + A.T)  # make it symmetric and column major
    Ac = A.copy('A')
    _ = gejsv(A)
    assert_allclose(A, Ac)


@pytest.mark.parametrize(('kwargs'),
                         ({'joba': 9},
                          {'jobu': 9},
                          {'jobv': 9},
                          {'jobr': 9},
                          {'jobt': 9},
                          {'jobp': 9})
                         )
def test_gejsv_invalid_job_arguments(kwargs):
    """Test invalid job arguments raise an Exception"""
    A = np.ones((2, 2), dtype=float)
    gejsv = get_lapack_funcs('gejsv', dtype=float)
    assert_raises(Exception, gejsv, A, **kwargs)


@pytest.mark.parametrize("A,sva_expect,u_expect,v_expect",
                         [(np.array([[2.27, -1.54, 1.15, -1.94],
                                     [0.28, -1.67, 0.94, -0.78],
                                     [-0.48, -3.09, 0.99, -0.21],
                                     [1.07, 1.22, 0.79, 0.63],
                                     [-2.35, 2.93, -1.45, 2.30],
                                     [0.62, -7.39, 1.03, -2.57]]),
                           np.array([9.9966, 3.6831, 1.3569, 0.5000]),
                           np.array([[0.2774, -0.6003, -0.1277, 0.1323],
                                     [0.2020, -0.0301, 0.2805, 0.7034],
                                     [0.2918, 0.3348, 0.6453, 0.1906],
                                     [-0.0938, -0.3699, 0.6781, -0.5399],
                                     [-0.4213, 0.5266, 0.0413, -0.0575],
                                     [0.7816, 0.3353, -0.1645, -0.3957]]),
                           np.array([[0.1921, -0.8030, 0.0041, -0.5642],
                                     [-0.8794, -0.3926, -0.0752, 0.2587],
                                     [0.2140, -0.2980, 0.7827, 0.5027],
                                     [-0.3795, 0.3351, 0.6178, -0.6017]]))])
def test_gejsv_NAG(A, sva_expect, u_expect, v_expect):
    """
    This test implements the example found in the NAG manual, f08khf.
    An example was not found for the complex case.
    """
    # NAG manual provides accuracy up to 4 decimals
    atol = 1e-4
    gejsv = get_lapack_funcs('gejsv', dtype=A.dtype)

    sva, u, v, work, iwork, info = gejsv(A)

    assert_allclose(sva_expect, sva, atol=atol)
    assert_allclose(u_expect, u, atol=atol)
    assert_allclose(v_expect, v, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gttrf_gttrs(dtype):
    # The test uses ?gttrf and ?gttrs to solve a random system for each dtype,
    # tests that the output of ?gttrf define LU matricies, that input
    # parameters are unmodified, transposal options function correctly, that
    # incompatible matrix shapes raise an error, and singular matrices return
    # non zero info.

    seed(42)
    n = 10
    atol = 100 * np.finfo(dtype).eps

    # create the matrix in accordance with the data type
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    dl = generate_random_dtype_array((n-1,), dtype=dtype)

    diag_cpy = [dl.copy(), d.copy(), du.copy()]

    A = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    x = np.random.rand(n)
    b = A @ x

    gttrf, gttrs = get_lapack_funcs(('gttrf', 'gttrs'), dtype=dtype)

    _dl, _d, _du, du2, ipiv, info = gttrf(dl, d, du)
    # test to assure that the inputs of ?gttrf are unmodified
    assert_array_equal(dl, diag_cpy[0])
    assert_array_equal(d, diag_cpy[1])
    assert_array_equal(du, diag_cpy[2])

    # generate L and U factors from ?gttrf return values
    # L/U are lower/upper triangular by construction (initially and at end)
    U = np.diag(_d, 0) + np.diag(_du, 1) + np.diag(du2, 2)
    L = np.eye(n, dtype=dtype)

    for i, m in enumerate(_dl):
        # L is given in a factored form.
        # See
        # www.hpcavf.uclan.ac.uk/softwaredoc/sgi_scsl_html/sgi_html/ch03.html
        piv = ipiv[i] - 1
        # right multiply by permutation matrix
        L[:, [i, piv]] = L[:, [piv, i]]
        # right multiply by Li, rank-one modification of identity
        L[:, i] += L[:, i+1]*m

    # one last permutation
    i, piv = -1, ipiv[-1] - 1
    # right multiply by final permutation matrix
    L[:, [i, piv]] = L[:, [piv, i]]

    # check that the outputs of ?gttrf define an LU decomposition of A
    assert_allclose(A, L @ U, atol=atol)

    b_cpy = b.copy()
    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b)
    # test that the inputs of ?gttrs are unmodified
    assert_array_equal(b, b_cpy)
    # test that the result of ?gttrs matches the expected input
    assert_allclose(x, x_gttrs, atol=atol)

    # test that ?gttrf and ?gttrs work with transposal options
    if dtype in REAL_DTYPES:
        trans = "T"
        b_trans = A.T @ x
    else:
        trans = "C"
        b_trans = A.conj().T @ x

    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b_trans, trans=trans)
    assert_allclose(x, x_gttrs, atol=atol)

    # test that ValueError is raised with incompatible matrix shapes
    with assert_raises(ValueError):
        gttrf(dl[:-1], d, du)
    with assert_raises(ValueError):
        gttrf(dl, d[:-1], du)
    with assert_raises(ValueError):
        gttrf(dl, d, du[:-1])

    # test that matrix of size n=2 raises exception
    with assert_raises(Exception):
        gttrf(dl[0], d[:1], du[0])

    # test that singular (row of all zeroes) matrix fails via info
    du[0] = 0
    d[0] = 0
    __dl, __d, __du, _du2, _ipiv, _info = gttrf(dl, d, du)
    np.testing.assert_(__d[info - 1] == 0,
                       "?gttrf: _d[info-1] is {}, not the illegal value :0."
                       .format(__d[info - 1]))


@pytest.mark.parametrize("du, d, dl, du_exp, d_exp, du2_exp, ipiv_exp, b, x",
                         [(np.array([2.1, -1.0, 1.9, 8.0]),
                             np.array([3.0, 2.3, -5.0, -.9, 7.1]),
                             np.array([3.4, 3.6, 7.0, -6.0]),
                             np.array([2.3, -5, -.9, 7.1]),
                             np.array([3.4, 3.6, 7, -6, -1.015373]),
                             np.array([-1, 1.9, 8]),
                             np.array([2, 3, 4, 5, 5]),
                             np.array([[2.7, 6.6],
                                       [-0.5, 10.8],
                                       [2.6, -3.2],
                                       [0.6, -11.2],
                                       [2.7, 19.1]
                                       ]),
                             np.array([[-4, 5],
                                       [7, -4],
                                       [3, -3],
                                       [-4, -2],
                                       [-3, 1]])),
                          (
                             np.array([2 - 1j, 2 + 1j, -1 + 1j, 1 - 1j]),
                             np.array([-1.3 + 1.3j, -1.3 + 1.3j,
                                       -1.3 + 3.3j, - .3 + 4.3j,
                                       -3.3 + 1.3j]),
                             np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j]),
                             # du exp
                             np.array([-1.3 + 1.3j, -1.3 + 3.3j,
                                       -0.3 + 4.3j, -3.3 + 1.3j]),
                             np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j,
                                       -1.3399 + 0.2875j]),
                             np.array([2 + 1j, -1 + 1j, 1 - 1j]),
                             np.array([2, 3, 4, 5, 5]),
                             np.array([[2.4 - 5j, 2.7 + 6.9j],
                                       [3.4 + 18.2j, - 6.9 - 5.3j],
                                       [-14.7 + 9.7j, - 6 - .6j],
                                       [31.9 - 7.7j, -3.9 + 9.3j],
                                       [-1 + 1.6j, -3 + 12.2j]]),
                             np.array([[1 + 1j, 2 - 1j],
                                       [3 - 1j, 1 + 2j],
                                       [4 + 5j, -1 + 1j],
                                       [-1 - 2j, 2 + 1j],
                                       [1 - 1j, 2 - 2j]])
                            )])
def test_gttrf_gttrs_NAG_f07cdf_f07cef_f07crf_f07csf(du, d, dl, du_exp, d_exp,
                                                     du2_exp, ipiv_exp, b, x):
    # test to assure that wrapper is consistent with NAG Library Manual Mark 26
    # example problems: f07cdf and f07cef (real)
    # examples: f07crf and f07csf (complex)
    # (Links may expire, so search for "NAG Library Manual Mark 26" online)

    gttrf, gttrs = get_lapack_funcs(('gttrf', "gttrs"), (du[0], du[0]))

    _dl, _d, _du, du2, ipiv, info = gttrf(dl, d, du)
    assert_allclose(du2, du2_exp)
    assert_allclose(_du, du_exp)
    assert_allclose(_d, d_exp, atol=1e-4)  # NAG examples provide 4 decimals.
    assert_allclose(ipiv, ipiv_exp)

    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b)

    assert_allclose(x_gttrs, x)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(3, 7), (7, 3), (2**18, 2**18)])
def test_geqrfp_lwork(dtype, shape):
    geqrfp_lwork = get_lapack_funcs(('geqrfp_lwork'), dtype=dtype)
    m, n = shape
    lwork, info = geqrfp_lwork(m=m, n=n)
    assert_equal(info, 0)


@pytest.mark.parametrize("ddtype,dtype",
                         zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
def test_pttrf_pttrs(ddtype, dtype):
    seed(42)
    # set test tolerance appropriate for dtype
    atol = 100*np.finfo(dtype).eps
    # n is the length diagonal of A
    n = 10
    # create diagonals according to size and dtype

    # diagonal d should always be real.
    # add 4 to d so it will be dominant for all dtypes
    d = generate_random_dtype_array((n,), ddtype) + 4
    # diagonal e may be real or complex.
    e = generate_random_dtype_array((n-1,), dtype)

    # assemble diagonals together into matrix
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    # store a copy of diagonals to later verify
    diag_cpy = [d.copy(), e.copy()]

    pttrf = get_lapack_funcs('pttrf', dtype=dtype)

    _d, _e, info = pttrf(d, e)
    # test to assure that the inputs of ?pttrf are unmodified
    assert_array_equal(d, diag_cpy[0])
    assert_array_equal(e, diag_cpy[1])
    assert_equal(info, 0, err_msg="pttrf: info = {}, should be 0".format(info))

    # test that the factors from pttrf can be recombined to make A
    L = np.diag(_e, -1) + np.diag(np.ones(n))
    D = np.diag(_d)

    assert_allclose(A, L@D@L.conjugate().T, atol=atol)

    # generate random solution x
    x = generate_random_dtype_array((n,), dtype)
    # determine accompanying b to get soln x
    b = A@x

    # determine _x from pttrs
    pttrs = get_lapack_funcs('pttrs', dtype=dtype)
    _x, info = pttrs(_d, _e.conj(), b)
    assert_equal(info, 0, err_msg="pttrs: info = {}, should be 0".format(info))

    # test that _x from pttrs matches the expected x
    assert_allclose(x, _x, atol=atol)


@pytest.mark.parametrize("ddtype,dtype",
                         zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
def test_pttrf_pttrs_errors_incompatible_shape(ddtype, dtype):
    n = 10
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)
    d = generate_random_dtype_array((n,), ddtype) + 2
    e = generate_random_dtype_array((n-1,), dtype)
    # test that ValueError is raised with incompatible matrix shapes
    assert_raises(ValueError, pttrf, d[:-1], e)
    assert_raises(ValueError, pttrf, d, e[:-1])


@pytest.mark.parametrize("ddtype,dtype",
                         zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
def test_pttrf_pttrs_errors_singular_nonSPD(ddtype, dtype):
    n = 10
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)
    d = generate_random_dtype_array((n,), ddtype) + 2
    e = generate_random_dtype_array((n-1,), dtype)
    # test that singular (row of all zeroes) matrix fails via info
    d[0] = 0
    e[0] = 0
    _d, _e, info = pttrf(d, e)
    assert_equal(_d[info - 1], 0,
                 "?pttrf: _d[info-1] is {}, not the illegal value :0."
                 .format(_d[info - 1]))

    # test with non-spd matrix
    d = generate_random_dtype_array((n,), ddtype)
    _d, _e, info = pttrf(d, e)
    assert_(info != 0, "?pttrf should fail with non-spd matrix, but didn't")


@pytest.mark.parametrize(("d, e, d_expect, e_expect, b, x_expect"), [
                         (np.array([4, 10, 29, 25, 5]),
                          np.array([-2, -6, 15, 8]),
                          np.array([4, 9, 25, 16, 1]),
                          np.array([-.5, -.6667, .6, .5]),
                          np.array([[6, 10], [9, 4], [2, 9], [14, 65],
                                    [7, 23]]),
                          np.array([[2.5, 2], [2, -1], [1, -3], [-1, 6],
                                    [3, -5]])
                          ), (
                          np.array([16, 41, 46, 21]),
                          np.array([16 + 16j, 18 - 9j, 1 - 4j]),
                          np.array([16, 9, 1, 4]),
                          np.array([1+1j, 2-1j, 1-4j]),
                          np.array([[64+16j, -16-32j], [93+62j, 61-66j],
                                    [78-80j, 71-74j], [14-27j, 35+15j]]),
                          np.array([[2+1j, -3-2j], [1+1j, 1+1j], [1-2j, 1-2j],
                                    [1-1j, 2+1j]])
                         )])
def test_pttrf_pttrs_NAG(d, e, d_expect, e_expect, b, x_expect):
    # test to assure that wrapper is consistent with NAG Manual Mark 26
    # example problems: f07jdf and f07jef (real)
    # examples: f07jrf and f07csf (complex)
    # NAG examples provide 4 decimals.
    # (Links expire, so please search for "NAG Library Manual Mark 26" online)

    atol = 1e-4
    pttrf = get_lapack_funcs('pttrf', dtype=e[0])
    _d, _e, info = pttrf(d, e)
    assert_allclose(_d, d_expect, atol=atol)
    assert_allclose(_e, e_expect, atol=atol)

    pttrs = get_lapack_funcs('pttrs', dtype=e[0])
    _x, info = pttrs(_d, _e.conj(), b)
    assert_allclose(_x, x_expect, atol=atol)

    # also test option `lower`
    if e.dtype in COMPLEX_DTYPES:
        _x, info = pttrs(_d, _e, b, lower=1)
        assert_allclose(_x, x_expect, atol=atol)


def pteqr_get_d_e_A_z(dtype, realtype, n, compute_z):
    # used by ?pteqr tests to build parameters
    # returns tuple of (d, e, A, z)
    if compute_z == 1:
        # build Hermitian A from Q**T * tri * Q = A by creating Q and tri
        A_eig = generate_random_dtype_array((n, n), dtype)
        A_eig = A_eig + np.diag(np.zeros(n) + 4*n)
        A_eig = (A_eig + A_eig.conj().T) / 2
        # obtain right eigenvectors (orthogonal)
        vr = eigh(A_eig)[1]
        # create tridiagonal matrix
        d = generate_random_dtype_array((n,), realtype) + 4
        e = generate_random_dtype_array((n-1,), realtype)
        tri = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        # Build A using these factors that sytrd would: (Q**T * tri * Q = A)
        A = vr @ tri @ vr.conj().T
        # vr is orthogonal
        z = vr

    else:
        # d and e are always real per lapack docs.
        d = generate_random_dtype_array((n,), realtype)
        e = generate_random_dtype_array((n-1,), realtype)

        # make SPD
        d = d + 4
        A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        z = np.diag(d) + np.diag(e, -1) + np.diag(e, 1)
    return (d, e, A, z)


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr(dtype, realtype, compute_z):
    '''
    Tests the ?pteqr lapack routine for all dtypes and compute_z parameters.
    It generates random SPD matrix diagonals d and e, and then confirms
    correct eigenvalues with scipy.linalg.eig. With applicable compute_z=2 it
    tests that z can reform A.
    '''
    seed(42)
    atol = 1000*np.finfo(dtype).eps
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)

    n = 10

    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)

    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d=d, e=e, z=z, compute_z=compute_z)
    assert_equal(info, 0, "info = {}, should be 0.".format(info))

    # compare the routine's eigenvalues with scipy.linalg.eig's.
    assert_allclose(np.sort(eigh(A)[0]), np.sort(d_pteqr), atol=atol)

    if compute_z:
        # verify z_pteqr as orthogonal
        assert_allclose(z_pteqr @ np.conj(z_pteqr).T, np.identity(n),
                        atol=atol)
        # verify that z_pteqr recombines to A
        assert_allclose(z_pteqr @ np.diag(d_pteqr) @ np.conj(z_pteqr).T,
                        A, atol=atol)


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr_error_non_spd(dtype, realtype, compute_z):
    seed(42)
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)

    n = 10
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)

    # test with non-spd matrix
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d - 4, e, z=z, compute_z=compute_z)
    assert info > 0


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr_raise_error_wrong_shape(dtype, realtype, compute_z):
    seed(42)
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)
    n = 10
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)
    # test with incorrect/incompatible array sizes
    assert_raises(ValueError, pteqr, d[:-1], e, z=z, compute_z=compute_z)
    assert_raises(ValueError, pteqr, d, e[:-1], z=z, compute_z=compute_z)
    if compute_z:
        assert_raises(ValueError, pteqr, d, e, z=z[:-1], compute_z=compute_z)


@pytest.mark.parametrize("dtype,realtype",
                         zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize("compute_z", range(3))
def test_pteqr_error_singular(dtype, realtype, compute_z):
    seed(42)
    pteqr = get_lapack_funcs(('pteqr'), dtype=dtype)
    n = 10
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)
    # test with singular matrix
    d[0] = 0
    e[0] = 0
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d, e, z=z, compute_z=compute_z)
    assert info > 0


@pytest.mark.parametrize("compute_z,d,e,d_expect,z_expect",
                         [(2,  # "I"
                           np.array([4.16, 5.25, 1.09, .62]),
                           np.array([3.17, -.97, .55]),
                           np.array([8.0023, 1.9926, 1.0014, 0.1237]),
                           np.array([[0.6326, 0.6245, -0.4191, 0.1847],
                                     [0.7668, -0.4270, 0.4176, -0.2352],
                                     [-0.1082, 0.6071, 0.4594, -0.6393],
                                     [-0.0081, 0.2432, 0.6625, 0.7084]])),
                          ])
def test_pteqr_NAG_f08jgf(compute_z, d, e, d_expect, z_expect):
    '''
    Implements real (f08jgf) example from NAG Manual Mark 26.
    Tests for correct outputs.
    '''
    # the NAG manual has 4 decimals accuracy
    atol = 1e-4
    pteqr = get_lapack_funcs(('pteqr'), dtype=d.dtype)

    z = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    _d, _e, _z, info = pteqr(d=d, e=e, z=z, compute_z=compute_z)
    assert_allclose(_d, d_expect, atol=atol)
    assert_allclose(np.abs(_z), np.abs(z_expect), atol=atol)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('matrix_size', [(3, 4), (7, 6), (6, 6)])
def test_geqrfp(dtype, matrix_size):
    # Tests for all dytpes, tall, wide, and square matrices.
    # Using the routine with random matrix A, Q and R are obtained and then
    # tested such that R is upper triangular and non-negative on the diagonal,
    # and Q is an orthagonal matrix. Verifies that A=Q@R. It also
    # tests against a matrix that for which the  linalg.qr method returns
    # negative diagonals, and for error messaging.

    # set test tolerance appropriate for dtype
    np.random.seed(42)
    rtol = 250*np.finfo(dtype).eps
    atol = 100*np.finfo(dtype).eps
    # get appropriate ?geqrfp for dtype
    geqrfp = get_lapack_funcs(('geqrfp'), dtype=dtype)
    gqr = get_lapack_funcs(("orgqr"), dtype=dtype)

    m, n = matrix_size

    # create random matrix of dimentions m x n
    A = generate_random_dtype_array((m, n), dtype=dtype)
    # create qr matrix using geqrfp
    qr_A, tau, info = geqrfp(A)

    # obtain r from the upper triangular area
    r = np.triu(qr_A)

    # obtain q from the orgqr lapack routine
    # based on linalg.qr's extraction strategy of q with orgqr

    if m > n:
        # this adds an extra column to the end of qr_A
        # let qqr be an empty m x m matrix
        qqr = np.zeros((m, m), dtype=dtype)
        # set first n columns of qqr to qr_A
        qqr[:, :n] = qr_A
        # determine q from this qqr
        # note that m is a sufficient for lwork based on LAPACK documentation
        q = gqr(qqr, tau=tau, lwork=m)[0]
    else:
        q = gqr(qr_A[:, :m], tau=tau, lwork=m)[0]

    # test that q and r still make A
    assert_allclose(q@r, A, rtol=rtol)
    # ensure that q is orthogonal (that q @ transposed q is the identity)
    assert_allclose(np.eye(q.shape[0]), q@(q.conj().T), rtol=rtol,
                    atol=atol)
    # ensure r is upper tri by comparing original r to r as upper triangular
    assert_allclose(r, np.triu(r), rtol=rtol)
    # make sure diagonals of r are positive for this random solution
    assert_(np.all(np.diag(r) > np.zeros(len(np.diag(r)))))
    # ensure that info is zero for this success
    assert_(info == 0)

    # test that this routine gives r diagonals that are positive for a
    # matrix that returns negatives in the diagonal with scipy.linalg.rq
    A_negative = generate_random_dtype_array((n, m), dtype=dtype) * -1
    r_rq_neg, q_rq_neg = qr(A_negative)
    rq_A_neg, tau_neg, info_neg = geqrfp(A_negative)
    # assert that any of the entries on the diagonal from linalg.qr
    #   are negative and that all of geqrfp are positive.
    assert_(np.any(np.diag(r_rq_neg) < 0) and
            np.all(np.diag(r) > 0))


def test_geqrfp_errors_with_empty_array():
    # check that empty array raises good error message
    A_empty = np.array([])
    geqrfp = get_lapack_funcs('geqrfp', dtype=A_empty.dtype)
    assert_raises(Exception, geqrfp, A_empty)


@pytest.mark.parametrize("driver", ['ev', 'evd', 'evr', 'evx'])
@pytest.mark.parametrize("pfx", ['sy', 'he'])
def test_standard_eigh_lworks(pfx, driver):
    n = 1200  # Some sufficiently big arbitrary number
    dtype = REAL_DTYPES if pfx == 'sy' else COMPLEX_DTYPES
    sc_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[0])
    dz_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[1])
    try:
        _compute_lwork(sc_dlw, n, lower=1)
        _compute_lwork(dz_dlw, n, lower=1)
    except Exception as e:
        pytest.fail("{}_lwork raised unexpected exception: {}"
                    "".format(pfx+driver, e))


@pytest.mark.parametrize("driver", ['gv', 'gvx'])
@pytest.mark.parametrize("pfx", ['sy', 'he'])
def test_generalized_eigh_lworks(pfx, driver):
    n = 1200  # Some sufficiently big arbitrary number
    dtype = REAL_DTYPES if pfx == 'sy' else COMPLEX_DTYPES
    sc_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[0])
    dz_dlw = get_lapack_funcs(pfx+driver+'_lwork', dtype=dtype[1])
    # Shouldn't raise any exceptions
    try:
        _compute_lwork(sc_dlw, n, uplo="L")
        _compute_lwork(dz_dlw, n, uplo="L")
    except Exception as e:
        pytest.fail("{}_lwork raised unexpected exception: {}"
                    "".format(pfx+driver, e))


@pytest.mark.parametrize("dtype_", DTYPES)
@pytest.mark.parametrize("m", [1, 10, 100, 1000])
def test_orcsd_uncsd_lwork(dtype_, m):
    seed(1234)
    p = randint(0, m)
    q = m - p
    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'
    dlw = pfx + 'csd_lwork'
    lw = get_lapack_funcs(dlw, dtype=dtype_)
    lwval = _compute_lwork(lw, m, p, q)
    lwval = lwval if pfx == 'un' else (lwval,)
    assert all([x > 0 for x in lwval])


@pytest.mark.parametrize("dtype_", DTYPES)
def test_orcsd_uncsd(dtype_):
    m, p, q = 250, 80, 170

    pfx = 'or' if dtype_ in REAL_DTYPES else 'un'
    X = ortho_group.rvs(m) if pfx == 'or' else unitary_group.rvs(m)

    drv, dlw = get_lapack_funcs((pfx + 'csd', pfx + 'csd_lwork'), dtype=dtype_)
    lwval = _compute_lwork(dlw, m, p, q)
    lwvals = {'lwork': lwval} if pfx == 'or' else dict(zip(['lwork',
                                                            'lrwork'], lwval))

    cs11, cs12, cs21, cs22, theta, u1, u2, v1t, v2t, info =\
        drv(X[:p, :q], X[:p, q:], X[p:, :q], X[p:, q:], **lwvals)

    assert info == 0

    U = block_diag(u1, u2)
    VH = block_diag(v1t, v2t)
    r = min(min(p, q), min(m-p, m-q))
    n11 = min(p, q) - r
    n12 = min(p, m-q) - r
    n21 = min(m-p, q) - r
    n22 = min(m-p, m-q) - r

    S = np.zeros((m, m), dtype=dtype_)
    one = dtype_(1.)
    for i in range(n11):
        S[i, i] = one
    for i in range(n22):
        S[p+i, q+i] = one
    for i in range(n12):
        S[i+n11+r, i+n11+r+n21+n22+r] = -one
    for i in range(n21):
        S[p+n22+r+i, n11+r+i] = one

    for i in range(r):
        S[i+n11, i+n11] = np.cos(theta[i])
        S[p+n22+i, i+r+n21+n22] = np.cos(theta[i])

        S[i+n11, i+n11+n21+n22+r] = -np.sin(theta[i])
        S[p+n22+i, i+n11] = np.sin(theta[i])

    Xc = U @ S @ VH
    assert_allclose(X, Xc, rtol=0., atol=1e4*np.finfo(dtype_).eps)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans_bool", [False, True])
@pytest.mark.parametrize("fact", ["F", "N"])
def test_gtsvx(dtype, trans_bool, fact):
    """
    These tests uses ?gtsvx to solve a random Ax=b system for each dtype.
    It tests that the outputs define an LU matrix, that inputs are unmodified,
    transposal options, incompatible shapes, singular matrices, and
    singular factorizations. It parametrizes DTYPES and the 'fact' value along
    with the fact related inputs.
    """
    seed(42)
    # set test tolerance appropriate for dtype
    atol = 100 * np.finfo(dtype).eps
    # obtain routine
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    # Generate random tridiagonal matrix A
    n = 10
    dl = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    # generate random solution x
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    # create b from x for equation Ax=b
    trans = ("T" if dtype in REAL_DTYPES else "C") if trans_bool else "N"
    b = (A.conj().T if trans_bool else A) @ x

    # store a copy of the inputs to check they haven't been modified later
    inputs_cpy = [dl.copy(), d.copy(), du.copy(), b.copy()]

    # set these to None if fact = 'N', or the output of gttrf is fact = 'F'
    dlf_, df_, duf_, du2f_, ipiv_, info_ = \
        gttrf(dl, d, du) if fact == 'F' else [None]*6

    gtsvx_out = gtsvx(dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
    assert_(info == 0, "?gtsvx info = {}, should be zero".format(info))

    # assure that inputs are unmodified
    assert_array_equal(dl, inputs_cpy[0])
    assert_array_equal(d, inputs_cpy[1])
    assert_array_equal(du, inputs_cpy[2])
    assert_array_equal(b, inputs_cpy[3])

    # test that x_soln matches the expected x
    assert_allclose(x, x_soln, atol=atol)

    # assert that the outputs are of correct type or shape
    # rcond should be a scalar
    assert_(hasattr(rcond, "__len__") is not True,
            "rcond should be scalar but is {}".format(rcond))
    # ferr should be length of # of cols in x
    assert_(ferr.shape[0] == b.shape[1], "ferr.shape is {} but shoud be {},"
            .format(ferr.shape[0], b.shape[1]))
    # berr should be length of # of cols in x
    assert_(berr.shape[0] == b.shape[1], "berr.shape is {} but shoud be {},"
            .format(berr.shape[0], b.shape[1]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("trans_bool", [0, 1])
@pytest.mark.parametrize("fact", ["F", "N"])
def test_gtsvx_error_singular(dtype, trans_bool, fact):
    seed(42)
    # obtain routine
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    # Generate random tridiagonal matrix A
    n = 10
    dl = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    # generate random solution x
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    # create b from x for equation Ax=b
    trans = "T" if dtype in REAL_DTYPES else "C"
    b = (A.conj().T if trans_bool else A) @ x

    # set these to None if fact = 'N', or the output of gttrf is fact = 'F'
    dlf_, df_, duf_, du2f_, ipiv_, info_ = \
        gttrf(dl, d, du) if fact == 'F' else [None]*6

    gtsvx_out = gtsvx(dl, d, du, b, fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
    # test with singular matrix
    # no need to test inputs with fact "F" since ?gttrf already does.
    if fact == "N":
        # Construct a singular example manually
        d[-1] = 0
        dl[-1] = 0
        # solve using routine
        gtsvx_out = gtsvx(dl, d, du, b)
        dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
        # test for the singular matrix.
        assert info > 0, "info should be > 0 for singular matrix"

    elif fact == 'F':
        # assuming that a singular factorization is input
        df_[-1] = 0
        duf_[-1] = 0
        du2f_[-1] = 0

        gtsvx_out = gtsvx(dl, d, du, b, fact=fact, dlf=dlf_, df=df_, duf=duf_,
                          du2=du2f_, ipiv=ipiv_)
        dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out
        # info should not be zero and should provide index of illegal value
        assert info > 0, "info should be > 0 for singular matrix"


@pytest.mark.parametrize("dtype", DTYPES*2)
@pytest.mark.parametrize("trans_bool", [False, True])
@pytest.mark.parametrize("fact", ["F", "N"])
def test_gtsvx_error_incompatible_size(dtype, trans_bool, fact):
    seed(42)
    # obtain routine
    gtsvx, gttrf = get_lapack_funcs(('gtsvx', 'gttrf'), dtype=dtype)
    # Generate random tridiagonal matrix A
    n = 10
    dl = generate_random_dtype_array((n-1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    du = generate_random_dtype_array((n-1,), dtype=dtype)
    A = np.diag(dl, -1) + np.diag(d) + np.diag(du, 1)
    # generate random solution x
    x = generate_random_dtype_array((n, 2), dtype=dtype)
    # create b from x for equation Ax=b
    trans = "T" if dtype in REAL_DTYPES else "C"
    b = (A.conj().T if trans_bool else A) @ x

    # set these to None if fact = 'N', or the output of gttrf is fact = 'F'
    dlf_, df_, duf_, du2f_, ipiv_, info_ = \
        gttrf(dl, d, du) if fact == 'F' else [None]*6

    if fact == "N":
        assert_raises(ValueError, gtsvx, dl[:-1], d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d[:-1], du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du[:-1], b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(Exception, gtsvx, dl, d, du, b[:-1],
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
    else:
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_[:-1], df=df_,
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_[:-1],
                      duf=duf_, du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_[:-1], du2=du2f_, ipiv=ipiv_)
        assert_raises(ValueError, gtsvx, dl, d, du, b,
                      fact=fact, trans=trans, dlf=dlf_, df=df_,
                      duf=duf_, du2=du2f_[:-1], ipiv=ipiv_)


@pytest.mark.parametrize("du,d,dl,b,x",
                         [(np.array([2.1, -1.0, 1.9, 8.0]),
                           np.array([3.0, 2.3, -5.0, -0.9, 7.1]),
                           np.array([3.4, 3.6, 7.0, -6.0]),
                           np.array([[2.7, 6.6], [-.5, 10.8], [2.6, -3.2],
                                     [.6, -11.2], [2.7, 19.1]]),
                           np.array([[-4, 5], [7, -4], [3, -3], [-4, -2],
                                     [-3, 1]])),
                          (np.array([2 - 1j, 2 + 1j, -1 + 1j, 1 - 1j]),
                           np.array([-1.3 + 1.3j, -1.3 + 1.3j, -1.3 + 3.3j,
                                     -.3 + 4.3j, -3.3 + 1.3j]),
                           np.array([1 - 2j, 1 + 1j, 2 - 3j, 1 + 1j]),
                           np.array([[2.4 - 5j, 2.7 + 6.9j],
                                     [3.4 + 18.2j, -6.9 - 5.3j],
                                     [-14.7 + 9.7j, -6 - .6j],
                                     [31.9 - 7.7j, -3.9 + 9.3j],
                                     [-1 + 1.6j, -3 + 12.2j]]),
                           np.array([[1 + 1j, 2 - 1j], [3 - 1j, 1 + 2j],
                                     [4 + 5j, -1 + 1j], [-1 - 2j, 2 + 1j],
                                     [1 - 1j, 2 - 2j]]))])
def test_gtsvx_NAG(du, d, dl, b, x):
    # Test to ensure wrapper is consistent with NAG Manual Mark 26
    # example problems: real (f07cbf) and complex (f07cpf)
    gtsvx = get_lapack_funcs('gtsvx', dtype=d.dtype)

    gtsvx_out = gtsvx(dl, d, du, b)
    dlf, df, duf, du2f, ipiv, x_soln, rcond, ferr, berr, info = gtsvx_out

    assert_array_almost_equal(x, x_soln)


@pytest.mark.parametrize("dtype,realtype", zip(DTYPES, REAL_DTYPES
                                               + REAL_DTYPES))
@pytest.mark.parametrize("fact,df_de_lambda",
                         [("F",
                           lambda d, e:get_lapack_funcs('pttrf',
                                                        dtype=e.dtype)(d, e)),
                          ("N", lambda d, e: (None, None, None))])
def test_ptsvx(dtype, realtype, fact, df_de_lambda):
    '''
    This tests the ?ptsvx lapack routine wrapper to solve a random system
    Ax = b for all dtypes and input variations. Tests for: unmodified
    input parameters, fact options, incompatible matrix shapes raise an error,
    and singular matrices return info of illegal value.
    '''
    seed(42)
    # set test tolerance appropriate for dtype
    atol = 100 * np.finfo(dtype).eps
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    # create diagonals according to size and dtype
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n-1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln

    # use lambda to determine what df, ef are
    df, ef, info = df_de_lambda(d, e)

    # create copy to later test that they are unmodified
    diag_cpy = [d.copy(), e.copy(), b.copy()]

    # solve using routine
    df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b, fact=fact,
                                               df=df, ef=ef)
    # d, e, and b should be unmodified
    assert_array_equal(d, diag_cpy[0])
    assert_array_equal(e, diag_cpy[1])
    assert_array_equal(b, diag_cpy[2])
    assert_(info == 0, "info should be 0 but is {}.".format(info))
    assert_array_almost_equal(x_soln, x)

    # test that the factors from ptsvx can be recombined to make A
    L = np.diag(ef, -1) + np.diag(np.ones(n))
    D = np.diag(df)
    assert_allclose(A, L@D@(np.conj(L).T), atol=atol)

    # assert that the outputs are of correct type or shape
    # rcond should be a scalar
    assert not hasattr(rcond, "__len__"), \
        "rcond should be scalar but is {}".format(rcond)
    # ferr should be length of # of cols in x
    assert_(ferr.shape == (2,), "ferr.shape is {} but shoud be ({},)"
            .format(ferr.shape, x_soln.shape[1]))
    # berr should be length of # of cols in x
    assert_(berr.shape == (2,), "berr.shape is {} but shoud be ({},)"
            .format(berr.shape, x_soln.shape[1]))

@pytest.mark.parametrize("dtype,realtype", zip(DTYPES, REAL_DTYPES
                                               + REAL_DTYPES))
@pytest.mark.parametrize("fact,df_de_lambda",
                         [("F",
                           lambda d, e:get_lapack_funcs('pttrf',
                                                        dtype=e.dtype)(d, e)),
                          ("N", lambda d, e: (None, None, None))])
def test_ptsvx_error_raise_errors(dtype, realtype, fact, df_de_lambda):
    seed(42)
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    # create diagonals according to size and dtype
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n-1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln

    # use lambda to determine what df, ef are
    df, ef, info = df_de_lambda(d, e)

    # test with malformatted array sizes
    assert_raises(ValueError, ptsvx, d[:-1], e, b, fact=fact, df=df, ef=ef)
    assert_raises(ValueError, ptsvx, d, e[:-1], b, fact=fact, df=df, ef=ef)
    assert_raises(Exception, ptsvx, d, e, b[:-1], fact=fact, df=df, ef=ef)


@pytest.mark.parametrize("dtype,realtype", zip(DTYPES, REAL_DTYPES
                                               + REAL_DTYPES))
@pytest.mark.parametrize("fact,df_de_lambda",
                         [("F",
                           lambda d, e:get_lapack_funcs('pttrf',
                                                        dtype=e.dtype)(d, e)),
                          ("N", lambda d, e: (None, None, None))])
def test_ptsvx_non_SPD_singular(dtype, realtype, fact, df_de_lambda):
    seed(42)
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    # create diagonals according to size and dtype
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n-1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln

    # use lambda to determine what df, ef are
    df, ef, info = df_de_lambda(d, e)

    if fact == "N":
        d[3] = 0
        # obtain new df, ef
        df, ef, info = df_de_lambda(d, e)
        # solve using routine
        df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b)
        # test for the singular matrix.
        assert info > 0 and info <= n

        # non SPD matrix
        d = generate_random_dtype_array((n,), realtype)
        df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b)
        assert info > 0 and info <= n
    else:
        # assuming that someone is using a singular factorization
        df, ef, info = df_de_lambda(d, e)
        df[0] = 0
        ef[0] = 0
        df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b, fact=fact,
                                                   df=df, ef=ef)
        assert info > 0


@pytest.mark.parametrize('d,e,b,x',
                         [(np.array([4, 10, 29, 25, 5]),
                           np.array([-2, -6, 15, 8]),
                           np.array([[6, 10], [9, 4], [2, 9], [14, 65],
                                     [7, 23]]),
                           np.array([[2.5, 2], [2, -1], [1, -3],
                                     [-1, 6], [3, -5]])),
                          (np.array([16, 41, 46, 21]),
                           np.array([16 + 16j, 18 - 9j, 1 - 4j]),
                           np.array([[64 + 16j, -16 - 32j],
                                     [93 + 62j, 61 - 66j],
                                     [78 - 80j, 71 - 74j],
                                     [14 - 27j, 35 + 15j]]),
                           np.array([[2 + 1j, -3 - 2j],
                                     [1 + 1j, 1 + 1j],
                                     [1 - 2j, 1 - 2j],
                                     [1 - 1j, 2 + 1j]]))])
def test_ptsvx_NAG(d, e, b, x):
    # test to assure that wrapper is consistent with NAG Manual Mark 26
    # example problemss: f07jbf, f07jpf
    # (Links expire, so please search for "NAG Library Manual Mark 26" online)

    # obtain routine with correct type based on e.dtype
    ptsvx = get_lapack_funcs('ptsvx', dtype=e.dtype)
    # solve using routine
    df, ef, x_ptsvx, rcond, ferr, berr, info = ptsvx(d, e, b)
    # determine ptsvx's solution and x are the same.
    assert_array_almost_equal(x, x_ptsvx)


@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('dtype', DTYPES)
def test_pptrs_pptri_pptrf_ppsv_ppcon(dtype, lower):
    seed(1234)
    atol = np.finfo(dtype).eps*100
    # Manual conversion to/from packed format is feasible here.
    n, nrhs = 10, 4
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, nrhs], dtype=dtype)

    a = a.conj().T + a + np.eye(n, dtype=dtype) * dtype(5.)
    if lower:
        inds = ([x for y in range(n) for x in range(y, n)],
                [y for y in range(n) for x in range(y, n)])
    else:
        inds = ([x for y in range(1, n+1) for x in range(y)],
                [y-1 for y in range(1, n+1) for x in range(y)])
    ap = a[inds]
    ppsv, pptrf, pptrs, pptri, ppcon = get_lapack_funcs(
        ('ppsv', 'pptrf', 'pptrs', 'pptri', 'ppcon'),
        dtype=dtype,
        ilp64="preferred")

    ul, info = pptrf(n, ap, lower=lower)
    assert_equal(info, 0)
    aul = cholesky(a, lower=lower)[inds]
    assert_allclose(ul, aul, rtol=0, atol=atol)

    uli, info = pptri(n, ul, lower=lower)
    assert_equal(info, 0)
    auli = inv(a)[inds]
    assert_allclose(uli, auli, rtol=0, atol=atol)

    x, info = pptrs(n, ul, b, lower=lower)
    assert_equal(info, 0)
    bx = solve(a, b)
    assert_allclose(x, bx, rtol=0, atol=atol)

    xv, info = ppsv(n, ap, b, lower=lower)
    assert_equal(info, 0)
    assert_allclose(xv, bx, rtol=0, atol=atol)

    anorm = np.linalg.norm(a, 1)
    rcond, info = ppcon(n, ap, anorm=anorm, lower=lower)
    assert_equal(info, 0)
    assert_(abs(1/rcond - np.linalg.cond(a, p=1))*rcond < 1)


@pytest.mark.parametrize('dtype', DTYPES)
def test_gges_tgexc(dtype):
    seed(1234)
    atol = np.finfo(dtype).eps*100

    n = 10
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, n], dtype=dtype)

    gges, tgexc = get_lapack_funcs(('gges', 'tgexc'), dtype=dtype)

    result = gges(lambda x: None, a, b, overwrite_a=False, overwrite_b=False)
    assert_equal(result[-1], 0)

    s = result[0]
    t = result[1]
    q = result[-4]
    z = result[-3]

    d1 = s[0, 0] / t[0, 0]
    d2 = s[6, 6] / t[6, 6]

    if dtype in COMPLEX_DTYPES:
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

    result = tgexc(s, t, q, z, 6, 0)
    assert_equal(result[-1], 0)

    s = result[0]
    t = result[1]
    q = result[2]
    z = result[3]

    if dtype in COMPLEX_DTYPES:
        assert_allclose(s, np.triu(s), rtol=0, atol=atol)
        assert_allclose(t, np.triu(t), rtol=0, atol=atol)

    assert_allclose(q @ s @ z.conj().T, a, rtol=0, atol=atol)
    assert_allclose(q @ t @ z.conj().T, b, rtol=0, atol=atol)

    assert_allclose(s[0, 0] / t[0, 0], d2, rtol=0, atol=atol)
    assert_allclose(s[1, 1] / t[1, 1], d1, rtol=0, atol=atol)
