__usage__ = """
To run tests locally:
  python tests/test_arpack.py [-l<int>] [-v<int>]

"""

import threading
import itertools
import sys
import platform

import numpy as np

from numpy.testing import (assert_allclose, assert_array_almost_equal_nulp,
                           assert_equal, assert_array_equal, suppress_warnings)
from pytest import raises as assert_raises
import pytest

from numpy import dot, conj, random
from scipy.linalg import eig, eigh, hilbert, svd
from scipy.sparse import csc_matrix, csr_matrix, isspmatrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg.eigen.arpack import eigs, eigsh, svds, \
     ArpackNoConvergence, arpack

from scipy._lib._gcutils import assert_deallocated, IS_PYPY

IS_MACOS_ARM64 = sys.platform == 'darwin' and platform.machine() == 'arm64'


# precision for tests
_ndigits = {'f': 3, 'd': 11, 'F': 3, 'D': 11}


def _get_test_tolerance(type_char, mattype=None):
    """
    Return tolerance values suitable for a given test:

    Parameters
    ----------
    type_char : {'f', 'd', 'F', 'D'}
        Data type in ARPACK eigenvalue problem
    mattype : {csr_matrix, aslinearoperator, asarray}, optional
        Linear operator type

    Returns
    -------
    tol
        Tolerance to pass to the ARPACK routine
    rtol
        Relative tolerance for outputs
    atol
        Absolute tolerance for outputs

    """

    rtol = {'f': 3000 * np.finfo(np.float32).eps,
            'F': 3000 * np.finfo(np.float32).eps,
            'd': 2000 * np.finfo(np.float64).eps,
            'D': 2000 * np.finfo(np.float64).eps}[type_char]
    atol = rtol
    tol = 0

    if mattype is aslinearoperator and type_char in ('f', 'F'):
        # iterative methods in single precision: worse errors
        # also: bump ARPACK tolerance so that the iterative method converges
        tol = 30 * np.finfo(np.float32).eps
        rtol *= 5

    if mattype is csr_matrix and type_char in ('f', 'F'):
        # sparse in single precision: worse errors
        rtol *= 5

    return tol, rtol, atol


def generate_matrix(N, complex_=False, hermitian=False,
                    pos_definite=False, sparse=False):
    M = np.random.random((N, N))
    if complex_:
        M = M + 1j * np.random.random((N, N))

    if hermitian:
        if pos_definite:
            if sparse:
                i = np.arange(N)
                j = np.random.randint(N, size=N-2)
                i, j = np.meshgrid(i, j)
                M[i, j] = 0
            M = np.dot(M.conj(), M.T)
        else:
            M = np.dot(M.conj(), M.T)
            if sparse:
                i = np.random.randint(N, size=N * N // 4)
                j = np.random.randint(N, size=N * N // 4)
                ind = np.nonzero(i == j)
                j[ind] = (j[ind] + 1) % N
                M[i, j] = 0
                M[j, i] = 0
    else:
        if sparse:
            i = np.random.randint(N, size=N * N // 2)
            j = np.random.randint(N, size=N * N // 2)
            M[i, j] = 0
    return M


def generate_matrix_symmetric(N, pos_definite=False, sparse=False):
    M = np.random.random((N, N))

    M = 0.5 * (M + M.T)  # Make M symmetric

    if pos_definite:
        Id = N * np.eye(N)
        if sparse:
            M = csr_matrix(M)
        M += Id
    else:
        if sparse:
            M = csr_matrix(M)

    return M


def _aslinearoperator_with_dtype(m):
    m = aslinearoperator(m)
    if not hasattr(m, 'dtype'):
        x = np.zeros(m.shape[1])
        m.dtype = (m * x).dtype
    return m


def assert_allclose_cc(actual, desired, **kw):
    """Almost equal or complex conjugates almost equal"""
    try:
        assert_allclose(actual, desired, **kw)
    except AssertionError:
        assert_allclose(actual, conj(desired), **kw)


def argsort_which(eigenvalues, typ, k, which,
                  sigma=None, OPpart=None, mode=None):
    """Return sorted indices of eigenvalues using the "which" keyword
    from eigs and eigsh"""
    if sigma is None:
        reval = np.round(eigenvalues, decimals=_ndigits[typ])
    else:
        if mode is None or mode == 'normal':
            if OPpart is None:
                reval = 1. / (eigenvalues - sigma)
            elif OPpart == 'r':
                reval = 0.5 * (1. / (eigenvalues - sigma)
                               + 1. / (eigenvalues - np.conj(sigma)))
            elif OPpart == 'i':
                reval = -0.5j * (1. / (eigenvalues - sigma)
                                 - 1. / (eigenvalues - np.conj(sigma)))
        elif mode == 'cayley':
            reval = (eigenvalues + sigma) / (eigenvalues - sigma)
        elif mode == 'buckling':
            reval = eigenvalues / (eigenvalues - sigma)
        else:
            raise ValueError("mode='%s' not recognized" % mode)

        reval = np.round(reval, decimals=_ndigits[typ])

    if which in ['LM', 'SM']:
        ind = np.argsort(abs(reval))
    elif which in ['LR', 'SR', 'LA', 'SA', 'BE']:
        ind = np.argsort(np.real(reval))
    elif which in ['LI', 'SI']:
        # for LI,SI ARPACK returns largest,smallest abs(imaginary) why?
        if typ.islower():
            ind = np.argsort(abs(np.imag(reval)))
        else:
            ind = np.argsort(np.imag(reval))
    else:
        raise ValueError("which='%s' is unrecognized" % which)

    if which in ['LM', 'LA', 'LR', 'LI']:
        return ind[-k:]
    elif which in ['SM', 'SA', 'SR', 'SI']:
        return ind[:k]
    elif which == 'BE':
        return np.concatenate((ind[:k//2], ind[k//2-k:]))


def eval_evec(symmetric, d, typ, k, which, v0=None, sigma=None,
              mattype=np.asarray, OPpart=None, mode='normal'):
    general = ('bmat' in d)

    if symmetric:
        eigs_func = eigsh
    else:
        eigs_func = eigs

    if general:
        err = ("error for %s:general, typ=%s, which=%s, sigma=%s, "
               "mattype=%s, OPpart=%s, mode=%s" % (eigs_func.__name__,
                                                   typ, which, sigma,
                                                   mattype.__name__,
                                                   OPpart, mode))
    else:
        err = ("error for %s:standard, typ=%s, which=%s, sigma=%s, "
               "mattype=%s, OPpart=%s, mode=%s" % (eigs_func.__name__,
                                                   typ, which, sigma,
                                                   mattype.__name__,
                                                   OPpart, mode))

    a = d['mat'].astype(typ)
    ac = mattype(a)

    if general:
        b = d['bmat'].astype(typ)
        bc = mattype(b)

    # get exact eigenvalues
    exact_eval = d['eval'].astype(typ.upper())
    ind = argsort_which(exact_eval, typ, k, which,
                        sigma, OPpart, mode)
    exact_eval = exact_eval[ind]

    # compute arpack eigenvalues
    kwargs = dict(which=which, v0=v0, sigma=sigma)
    if eigs_func is eigsh:
        kwargs['mode'] = mode
    else:
        kwargs['OPpart'] = OPpart

    # compute suitable tolerances
    kwargs['tol'], rtol, atol = _get_test_tolerance(typ, mattype)

    # on rare occasions, ARPACK routines return results that are proper
    # eigenvalues and -vectors, but not necessarily the ones requested in
    # the parameter which. This is inherent to the Krylov methods, and
    # should not be treated as a failure. If such a rare situation
    # occurs, the calculation is tried again (but at most a few times).
    ntries = 0
    while ntries < 5:
        # solve
        if general:
            try:
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
            except ArpackNoConvergence:
                kwargs['maxiter'] = 20*a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
        else:
            try:
                eigenvalues, evec = eigs_func(ac, k, **kwargs)
            except ArpackNoConvergence:
                kwargs['maxiter'] = 20*a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, **kwargs)

        ind = argsort_which(eigenvalues, typ, k, which,
                            sigma, OPpart, mode)
        eigenvalues = eigenvalues[ind]
        evec = evec[:, ind]

        # check eigenvectors
        LHS = np.dot(a, evec)
        if general:
            RHS = eigenvalues * np.dot(b, evec)
        else:
            RHS = eigenvalues * evec

            assert_allclose(LHS, RHS, rtol=rtol, atol=atol, err_msg=err)

        try:
            # check eigenvalues
            assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol,
                               err_msg=err)
            break
        except AssertionError:
            ntries += 1

    # check eigenvalues
    assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol, err_msg=err)


class DictWithRepr(dict):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<%s>" % self.name


class SymmetricParams:
    def __init__(self):
        self.eigs = eigsh
        self.which = ['LM', 'SM', 'LA', 'SA', 'BE']
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        self.sigmas_modes = {None: ['normal'],
                             0.5: ['normal', 'buckling', 'cayley']}

        # generate matrices
        # these should all be float32 so that the eigenvalues
        # are the same in float32 and float64
        N = 6
        np.random.seed(2300)
        Ar = generate_matrix(N, hermitian=True,
                             pos_definite=True).astype('f').astype('d')
        M = generate_matrix(N, hermitian=True,
                            pos_definite=True).astype('f').astype('d')
        Ac = generate_matrix(N, hermitian=True, pos_definite=True,
                             complex_=True).astype('F').astype('D')
        Mc = generate_matrix(N, hermitian=True, pos_definite=True,
                             complex_=True).astype('F').astype('D')
        v0 = np.random.random(N)

        # standard symmetric problem
        SS = DictWithRepr("std-symmetric")
        SS['mat'] = Ar
        SS['v0'] = v0
        SS['eval'] = eigh(SS['mat'], eigvals_only=True)

        # general symmetric problem
        GS = DictWithRepr("gen-symmetric")
        GS['mat'] = Ar
        GS['bmat'] = M
        GS['v0'] = v0
        GS['eval'] = eigh(GS['mat'], GS['bmat'], eigvals_only=True)

        # standard hermitian problem
        SH = DictWithRepr("std-hermitian")
        SH['mat'] = Ac
        SH['v0'] = v0
        SH['eval'] = eigh(SH['mat'], eigvals_only=True)

        # general hermitian problem
        GH = DictWithRepr("gen-hermitian")
        GH['mat'] = Ac
        GH['bmat'] = M
        GH['v0'] = v0
        GH['eval'] = eigh(GH['mat'], GH['bmat'], eigvals_only=True)

        # general hermitian problem with hermitian M
        GHc = DictWithRepr("gen-hermitian-Mc")
        GHc['mat'] = Ac
        GHc['bmat'] = Mc
        GHc['v0'] = v0
        GHc['eval'] = eigh(GHc['mat'], GHc['bmat'], eigvals_only=True)

        self.real_test_cases = [SS, GS]
        self.complex_test_cases = [SH, GH, GHc]


class NonSymmetricParams:
    def __init__(self):
        self.eigs = eigs
        self.which = ['LM', 'LR', 'LI']  # , 'SM', 'LR', 'SR', 'LI', 'SI']
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        self.sigmas_OPparts = {None: [None],
                               0.1: ['r'],
                               0.1 + 0.1j: ['r', 'i']}

        # generate matrices
        # these should all be float32 so that the eigenvalues
        # are the same in float32 and float64
        N = 6
        np.random.seed(2300)
        Ar = generate_matrix(N).astype('f').astype('d')
        M = generate_matrix(N, hermitian=True,
                            pos_definite=True).astype('f').astype('d')
        Ac = generate_matrix(N, complex_=True).astype('F').astype('D')
        v0 = np.random.random(N)

        # standard real nonsymmetric problem
        SNR = DictWithRepr("std-real-nonsym")
        SNR['mat'] = Ar
        SNR['v0'] = v0
        SNR['eval'] = eig(SNR['mat'], left=False, right=False)

        # general real nonsymmetric problem
        GNR = DictWithRepr("gen-real-nonsym")
        GNR['mat'] = Ar
        GNR['bmat'] = M
        GNR['v0'] = v0
        GNR['eval'] = eig(GNR['mat'], GNR['bmat'], left=False, right=False)

        # standard complex nonsymmetric problem
        SNC = DictWithRepr("std-cmplx-nonsym")
        SNC['mat'] = Ac
        SNC['v0'] = v0
        SNC['eval'] = eig(SNC['mat'], left=False, right=False)

        # general complex nonsymmetric problem
        GNC = DictWithRepr("gen-cmplx-nonsym")
        GNC['mat'] = Ac
        GNC['bmat'] = M
        GNC['v0'] = v0
        GNC['eval'] = eig(GNC['mat'], GNC['bmat'], left=False, right=False)

        self.real_test_cases = [SNR, GNR]
        self.complex_test_cases = [SNC, GNC]


def test_symmetric_modes():
    params = SymmetricParams()
    k = 2
    symmetric = True
    for D in params.real_test_cases:
        for typ in 'fd':
            for which in params.which:
                for mattype in params.mattypes:
                    for (sigma, modes) in params.sigmas_modes.items():
                        for mode in modes:
                            eval_evec(symmetric, D, typ, k, which,
                                      None, sigma, mattype, None, mode)


def test_hermitian_modes():
    params = SymmetricParams()
    k = 2
    symmetric = True
    for D in params.complex_test_cases:
        for typ in 'FD':
            for which in params.which:
                if which == 'BE':
                    continue  # BE invalid for complex
                for mattype in params.mattypes:
                    for sigma in params.sigmas_modes:
                        eval_evec(symmetric, D, typ, k, which,
                                  None, sigma, mattype)


def test_symmetric_starting_vector():
    params = SymmetricParams()
    symmetric = True
    for k in [1, 2, 3, 4, 5]:
        for D in params.real_test_cases:
            for typ in 'fd':
                v0 = random.rand(len(D['v0'])).astype(typ)
                eval_evec(symmetric, D, typ, k, 'LM', v0)


def test_symmetric_no_convergence():
    np.random.seed(1234)
    m = generate_matrix(30, hermitian=True, pos_definite=True)
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        w, v = eigsh(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol, ncv=9)
        raise AssertionError("Spurious no-error exit")
    except ArpackNoConvergence as err:
        k = len(err.eigenvalues)
        if k <= 0:
            raise AssertionError("Spurious no-eigenvalues-found case") from err
        w, v = err.eigenvalues, err.eigenvectors
        assert_allclose(dot(m, v), w * v, rtol=rtol, atol=atol)


def test_real_nonsymmetric_modes():
    params = NonSymmetricParams()
    k = 2
    symmetric = False
    for D in params.real_test_cases:
        for typ in 'fd':
            for which in params.which:
                for mattype in params.mattypes:
                    for sigma, OPparts in params.sigmas_OPparts.items():
                        for OPpart in OPparts:
                            eval_evec(symmetric, D, typ, k, which,
                                      None, sigma, mattype, OPpart)


def test_complex_nonsymmetric_modes():
    params = NonSymmetricParams()
    k = 2
    symmetric = False
    for D in params.complex_test_cases:
        for typ in 'DF':
            for which in params.which:
                for mattype in params.mattypes:
                    for sigma in params.sigmas_OPparts:
                        eval_evec(symmetric, D, typ, k, which,
                                  None, sigma, mattype)


def test_standard_nonsymmetric_starting_vector():
    params = NonSymmetricParams()
    sigma = None
    symmetric = False
    for k in [1, 2, 3, 4]:
        for d in params.complex_test_cases:
            for typ in 'FD':
                A = d['mat']
                n = A.shape[0]
                v0 = random.rand(n).astype(typ)
                eval_evec(symmetric, d, typ, k, "LM", v0, sigma)


def test_general_nonsymmetric_starting_vector():
    params = NonSymmetricParams()
    sigma = None
    symmetric = False
    for k in [1, 2, 3, 4]:
        for d in params.complex_test_cases:
            for typ in 'FD':
                A = d['mat']
                n = A.shape[0]
                v0 = random.rand(n).astype(typ)
                eval_evec(symmetric, d, typ, k, "LM", v0, sigma)


@pytest.mark.skipif(IS_MACOS_ARM64, reason='failing on arm64')
def test_standard_nonsymmetric_no_convergence():
    np.random.seed(1234)
    m = generate_matrix(30, complex_=True)
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        w, v = eigs(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol)
        raise AssertionError("Spurious no-error exit")
    except ArpackNoConvergence as err:
        k = len(err.eigenvalues)
        if k <= 0:
            raise AssertionError("Spurious no-eigenvalues-found case") from err
        w, v = err.eigenvalues, err.eigenvectors
        for ww, vv in zip(w, v.T):
            assert_allclose(dot(m, vv), ww * vv, rtol=rtol, atol=atol)


def test_eigen_bad_shapes():
    # A is not square.
    A = csc_matrix(np.zeros((2, 3)))
    assert_raises(ValueError, eigs, A)


def test_eigen_bad_kwargs():
    # Test eigen on wrong keyword argument
    A = csc_matrix(np.zeros((8, 8)))
    assert_raises(ValueError, eigs, A, which='XX')


def test_ticket_1459_arpack_crash():
    for dtype in [np.float32, np.float64]:
        # This test does not seem to catch the issue for float32,
        # but we made the same fix there, just to be sure

        N = 6
        k = 2

        np.random.seed(2301)
        A = np.random.random((N, N)).astype(dtype)
        v0 = np.array([-0.71063568258907849895, -0.83185111795729227424,
                       -0.34365925382227402451, 0.46122533684552280420,
                       -0.58001341115969040629, -0.78844877570084292984e-01],
                      dtype=dtype)

        # Should not crash:
        evals, evecs = eigs(A, k, v0=v0)


#----------------------------------------------------------------------
# sparse SVD tests

def sorted_svd(m, k, which='LM'):
    # Compute svd of a dense matrix m, and return singular vectors/values
    # sorted.
    if isspmatrix(m):
        m = m.todense()
    u, s, vh = svd(m)
    if which == 'LM':
        ii = np.argsort(s)[-k:]
    elif which == 'SM':
        ii = np.argsort(s)[:k]
    else:
        raise ValueError("unknown which=%r" % (which,))

    return u[:, ii], s[ii], vh[ii]


def svd_estimate(u, s, vh):
    return np.dot(u, np.dot(np.diag(s), vh))


def svd_test_input_check():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1, 0, 2],
                  [0, 0, 1]], float)

    assert_raises(ValueError, svds, x, k=-1)
    assert_raises(ValueError, svds, x, k=0)
    assert_raises(ValueError, svds, x, k=10)
    assert_raises(ValueError, svds, x, k=x.shape[0])
    assert_raises(ValueError, svds, x, k=x.shape[1])
    assert_raises(ValueError, svds, x.T, k=x.shape[0])
    assert_raises(ValueError, svds, x.T, k=x.shape[1])


def test_svd_simple_real():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1, 0, 2],
                  [0, 0, 1]], float)
    y = np.array([[1, 2, 3, 8],
                  [3, 4, 3, 5],
                  [1, 0, 2, 3],
                  [0, 0, 1, 0]], float)
    z = csc_matrix(x)

    for solver in [None, 'arpack', 'lobpcg']:
        for m in [x.T, x, y, z, z.T]:
            for k in range(1, min(m.shape)):
                u, s, vh = sorted_svd(m, k)
                su, ss, svh = svds(m, k, solver=solver)

                m_hat = svd_estimate(u, s, vh)
                sm_hat = svd_estimate(su, ss, svh)

                assert_array_almost_equal_nulp(m_hat, sm_hat, nulp=1000)


def test_svd_simple_complex():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1 + 1j, 0, 2],
                  [0, 0, 1]], complex)
    y = np.array([[1, 2, 3, 8 + 5j],
                  [3 - 2j, 4, 3, 5],
                  [1, 0, 2, 3],
                  [0, 0, 1, 0]], complex)
    z = csc_matrix(x)

    for solver in [None, 'arpack', 'lobpcg']:
        for m in [x, x.T.conjugate(), x.T, y, y.conjugate(), z, z.T]:
            for k in range(1, min(m.shape) - 1):
                u, s, vh = sorted_svd(m, k)
                su, ss, svh = svds(m, k, solver=solver)

                m_hat = svd_estimate(u, s, vh)
                sm_hat = svd_estimate(su, ss, svh)

                assert_array_almost_equal_nulp(m_hat, sm_hat, nulp=1000)


def test_svd_maxiter():
    # check that maxiter works as expected
    x = hilbert(6)
    # ARPACK shouldn't converge on such an ill-conditioned matrix with just
    # one iteration
    assert_raises(ArpackNoConvergence, svds, x, 1, maxiter=1, ncv=3)
    # but 100 iterations should be more than enough
    u, s, vt = svds(x, 1, maxiter=100, ncv=3)
    assert_allclose(s, [1.7], atol=0.5)


def test_svd_return():
    # check that the return_singular_vectors parameter works as expected
    x = hilbert(6)
    _, s, _ = sorted_svd(x, 2)
    ss = svds(x, 2, return_singular_vectors=False)
    assert_allclose(s, ss)


def test_svd_which():
    # check that the which parameter works as expected
    x = hilbert(6)
    for which in ['LM', 'SM']:
        _, s, _ = sorted_svd(x, 2, which=which)
        for solver in [None, 'arpack', 'lobpcg']:
            ss = svds(x, 2, which=which, return_singular_vectors=False,
                      solver=solver)
            ss.sort()
            assert_allclose(s, ss, atol=np.sqrt(1e-15))


def test_svd_v0():
    # check that the v0 parameter works as expected
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)

    for solver in [None, 'arpack', 'lobpcg']:
        u, s, vh = svds(x, 1, solver=solver)
        u2, s2, vh2 = svds(x, 1, v0=u[:, 0], solver=solver)

        assert_allclose(s, s2, atol=np.sqrt(1e-15))


def _check_svds(A, k, U, s, VH):
    n, m = A.shape

    # Check shapes.
    assert_equal(U.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(VH.shape, (k, m))

    # Check that the original matrix can be reconstituted.
    A_rebuilt = (U*s).dot(VH)
    assert_equal(A_rebuilt.shape, A.shape)
    assert_allclose(A_rebuilt, A)

    # Check that U is a semi-orthogonal matrix.
    UH_U = np.dot(U.T.conj(), U)
    assert_equal(UH_U.shape, (k, k))
    assert_allclose(UH_U, np.identity(k), atol=1e-12)

    # Check that V is a semi-orthogonal matrix.
    VH_V = np.dot(VH, VH.T.conj())
    assert_equal(VH_V.shape, (k, k))
    assert_allclose(VH_V, np.identity(k), atol=1e-12)


def test_svd_LM_ones_matrix():
    # Check that svds can deal with matrix_rank less than k in LM mode.
    k = 3
    for n, m in (6, 5), (5, 5), (5, 6):
        for t in float, complex:
            A = np.ones((n, m), dtype=t)
            for solver in [None, 'arpack', 'lobpcg']:
                U, s, VH = svds(A, k, solver=solver)

                # Check some generic properties of svd.
                _check_svds(A, k, U, s, VH)

                # Check that the largest singular value is near sqrt(n*m)
                # and the other singular values have been forced to zero.
                assert_allclose(np.max(s), np.sqrt(n*m))
                assert_array_equal(sorted(s)[:-1], 0)


def test_svd_LM_zeros_matrix():
    # Check that svds can deal with matrices containing only zeros.
    k = 1
    for n, m in (3, 4), (4, 4), (4, 3):
        for t in float, complex:
            A = np.zeros((n, m), dtype=t)
            for solver in [None, 'arpack', 'lobpcg']:
                U, s, VH = svds(A, k, solver=solver)

                # Check some generic properties of svd.
                _check_svds(A, k, U, s, VH)

                # Check that the singular values are zero.
                assert_array_equal(s, 0)


def test_svd_LM_zeros_matrix_gh_3452():
    # Regression test for a github issue.
    # https://github.com/scipy/scipy/issues/3452
    # Note that for complex dype the size of this matrix is too small for k=1.
    n, m, k = 4, 2, 1
    A = np.zeros((n, m))
    for solver in [None, 'arpack', 'lobpcg']:
        U, s, VH = svds(A, k, solver=solver)

        # Check some generic properties of svd.
        _check_svds(A, k, U, s, VH)

        # Check that the singular values are zero.
        assert_array_equal(s, 0)


class CheckingLinearOperator(LinearOperator):
    def __init__(self, A):
        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape

    def _matvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.dot(x)

    def _rmatvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.T.conjugate().dot(x)


def test_svd_linop():
    nmks = [(6, 7, 3),
            (9, 5, 4),
            (10, 8, 5)]

    def reorder(args):
        U, s, VH = args
        j = np.argsort(s)
        return U[:, j], s[j], VH[j, :]

    for n, m, k in nmks:
        # Test svds on a LinearOperator.
        A = np.random.RandomState(52).randn(n, m)
        L = CheckingLinearOperator(A)

        v0 = np.ones(min(A.shape))

        for solver in [None, 'arpack', 'lobpcg']:
            U1, s1, VH1 = reorder(svds(A, k, v0=v0, solver=solver))
            U2, s2, VH2 = reorder(svds(L, k, v0=v0, solver=solver))

            assert_allclose(np.abs(U1), np.abs(U2))
            assert_allclose(s1, s2)
            assert_allclose(np.abs(VH1), np.abs(VH2))
            assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
                            np.dot(U2, np.dot(np.diag(s2), VH2)))

        # Try again with which="SM".
        A = np.random.RandomState(1909).randn(n, m)
        L = CheckingLinearOperator(A)

        for solver in [None, 'arpack', 'lobpcg']:
            U1, s1, VH1 = reorder(svds(A, k, which="SM", solver=solver))
            U2, s2, VH2 = reorder(svds(L, k, which="SM", solver=solver))

            assert_allclose(np.abs(U1), np.abs(U2))
            assert_allclose(s1, s2)
            assert_allclose(np.abs(VH1), np.abs(VH2))
            assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
                            np.dot(U2, np.dot(np.diag(s2), VH2)))

        if k < min(n, m) - 1:
            # Complex input and explicit which="LM".
            for (dt, eps) in [(complex, 1e-7), (np.complex64, 1e-3)]:
                rng = np.random.RandomState(1648)
                A = (rng.randn(n, m) + 1j * rng.randn(n, m)).astype(dt)
                L = CheckingLinearOperator(A)

                for solver in [None, 'arpack', 'lobpcg']:
                    U1, s1, VH1 = reorder(svds(A, k, which="LM", solver=solver))
                    U2, s2, VH2 = reorder(svds(L, k, which="LM", solver=solver))

                    assert_allclose(np.abs(U1), np.abs(U2), rtol=eps)
                    assert_allclose(s1, s2, rtol=eps)
                    assert_allclose(np.abs(VH1), np.abs(VH2), rtol=eps)
                    assert_allclose(np.dot(U1, np.dot(np.diag(s1), VH1)),
                                    np.dot(U2, np.dot(np.diag(s2), VH2)),
                                    rtol=eps)


@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_linearoperator_deallocation():
    # Check that the linear operators used by the Arpack wrappers are
    # deallocatable by reference counting -- they are big objects, so
    # Python's cyclic GC may not collect them fast enough before
    # running out of memory if eigs/eigsh are called in a tight loop.

    M_d = np.eye(10)
    M_s = csc_matrix(M_d)
    M_o = aslinearoperator(M_d)

    with assert_deallocated(lambda: arpack.SpLuInv(M_s)):
        pass
    with assert_deallocated(lambda: arpack.LuInv(M_d)):
        pass
    with assert_deallocated(lambda: arpack.IterInv(M_s)):
        pass
    with assert_deallocated(lambda: arpack.IterOpInv(M_o, None, 0.3)):
        pass
    with assert_deallocated(lambda: arpack.IterOpInv(M_o, M_o, 0.3)):
        pass


def test_svds_partial_return():
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1, 0, 2],
                  [0, 0, 1]], float)
    # test vertical matrix
    z = csr_matrix(x)
    vh_full = svds(z, 2)[-1]
    vh_partial = svds(z, 2, return_singular_vectors='vh')[-1]
    dvh = np.linalg.norm(np.abs(vh_full) - np.abs(vh_partial))
    if dvh > 1e-10:
        raise AssertionError('right eigenvector matrices differ when using return_singular_vectors parameter')
    if svds(z, 2, return_singular_vectors='vh')[0] is not None:
        raise AssertionError('left eigenvector matrix was computed when it should not have been')
    # test horizontal matrix
    z = csr_matrix(x.T)
    u_full = svds(z, 2)[0]
    u_partial = svds(z, 2, return_singular_vectors='vh')[0]
    du = np.linalg.norm(np.abs(u_full) - np.abs(u_partial))
    if du > 1e-10:
        raise AssertionError('left eigenvector matrices differ when using return_singular_vectors parameter')
    if svds(z, 2, return_singular_vectors='u')[-1] is not None:
        raise AssertionError('right eigenvector matrix was computed when it should not have been')

def test_svds_wrong_eigen_type():
    # Regression test for a github issue.
    # https://github.com/scipy/scipy/issues/4590
    # Function was not checking for eigenvalue type and unintended
    # values could be returned.
    x = np.array([[1, 2, 3],
                  [3, 4, 3],
                  [1, 0, 2],
                  [0, 0, 1]], float)
    assert_raises(ValueError, svds, x, 1, which='LA')


def test_parallel_threads():
    results = []
    v0 = np.random.rand(50)

    def worker():
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        w, v = eigs(x, k=3, v0=v0)
        results.append(w)

        w, v = eigsh(x, k=3, v0=v0)
        results.append(w)

    threads = [threading.Thread(target=worker) for k in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    worker()

    for r in results:
        assert_allclose(r, results[-1])


def test_reentering():
    # Just some linear operator that calls eigs recursively
    def A_matvec(x):
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        w, v = eigs(x, k=1)
        return v / w[0]
    A = LinearOperator(matvec=A_matvec, dtype=float, shape=(50, 50))

    # The Fortran code is not reentrant, so this fails (gracefully, not crashing)
    assert_raises(RuntimeError, eigs, A, k=1)
    assert_raises(RuntimeError, eigsh, A, k=1)


def test_regression_arpackng_1315():
    # Check that issue arpack-ng/#1315 is not present.
    # Adapted from arpack-ng/TESTS/bug_1315_single.c
    # If this fails, then the installed ARPACK library is faulty.

    for dtype in [np.float32, np.float64]:
        np.random.seed(1234)

        w0 = np.arange(1, 1000+1).astype(dtype)
        A = diags([w0], [0], shape=(1000, 1000))

        v0 = np.random.rand(1000).astype(dtype)
        w, v = eigs(A, k=9, ncv=2*9+1, which="LM", v0=v0)

        assert_allclose(np.sort(w), np.sort(w0[-9:]),
                        rtol=1e-4)


def test_eigs_for_k_greater():
    # Test eigs() for k beyond limits.
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))  # sparse
    A = generate_matrix(4, sparse=False)
    M_dense = np.random.random((4, 4))
    M_sparse = generate_matrix(4, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eig(A, b=M_dense)
    eig_tuple2 = eig(A, b=M_sparse)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)

        assert_equal(eigs(A, M=M_dense, k=3), eig_tuple1)
        assert_equal(eigs(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigs(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigs(A, M=M_sparse, k=5), eig_tuple2)

        # M as LinearOperator
        assert_raises(TypeError, eigs, A, M=M_linop, k=3)

        # Test 'A' for different types
        assert_raises(TypeError, eigs, aslinearoperator(A), k=3)
        assert_raises(TypeError, eigs, A_sparse, k=3)


def test_eigsh_for_k_greater():
    # Test eigsh() for k beyond limits.
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))  # sparse
    A = generate_matrix(4, sparse=False)
    M_dense = generate_matrix_symmetric(4, pos_definite=True)
    M_sparse = generate_matrix_symmetric(4, pos_definite=True, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eigh(A, b=M_dense)
    eig_tuple2 = eigh(A, b=M_sparse)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)

        assert_equal(eigsh(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigsh(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigsh(A, M=M_sparse, k=5), eig_tuple2)

        # M as LinearOperator
        assert_raises(TypeError, eigsh, A, M=M_linop, k=4)

        # Test 'A' for different types
        assert_raises(TypeError, eigsh, aslinearoperator(A), k=4)
        assert_raises(TypeError, eigsh, A_sparse, M=M_dense, k=4)


def test_real_eigs_real_k_subset():
    np.random.seed(1)

    n = 10
    A = rand(n, n, density=0.5)
    A.data *= 2
    A.data -= 1

    v0 = np.ones(n)

    whichs = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']
    dtypes = [np.float32, np.float64]

    for which, sigma, dtype in itertools.product(whichs, [None, 0, 5], dtypes):
        prev_w = np.array([], dtype=dtype)
        eps = np.finfo(dtype).eps
        for k in range(1, 9):
            w, z = eigs(A.astype(dtype), k=k, which=which, sigma=sigma,
                        v0=v0.astype(dtype), tol=0)
            assert_allclose(np.linalg.norm(A.dot(z) - z * w), 0, atol=np.sqrt(eps))

            # Check that the set of eigenvalues for `k` is a subset of that for `k+1`
            dist = abs(prev_w[:,None] - w).min(axis=1)
            assert_allclose(dist, 0, atol=np.sqrt(eps))

            prev_w = w

            # Check sort order
            if sigma is None:
                d = w
            else:
                d = 1 / (w - sigma)

            if which == 'LM':
                # ARPACK is systematic for 'LM', but sort order
                # appears not well defined for other modes
                assert np.all(np.diff(abs(d)) <= 1e-6)
