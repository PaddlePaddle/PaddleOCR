"""
Copyright (C) 2010 David Fong and Michael Saunders
Distributed under the same license as SciPy

Testing Code for LSMR.

03 Jun 2010: First version release with lsmr.py

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""

from numpy import array, arange, eye, zeros, ones, sqrt, transpose, hstack
from numpy.linalg import norm
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_almost_equal)

from scipy.sparse import coo_matrix
from scipy.sparse.linalg.interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b


class TestLSMR:
    def setup_method(self):
        self.n = 10
        self.m = 10

    def assertCompatibleSystem(self, A, xtrue):
        Afun = aslinearoperator(A)
        b = Afun.matvec(xtrue)
        x = lsmr(A, b)[0]
        assert_almost_equal(norm(x - xtrue), 0, decimal=5)

    def testIdentityACase1(self):
        A = eye(self.n)
        xtrue = zeros((self.n, 1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase2(self):
        A = eye(self.n)
        xtrue = ones((self.n,1))
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase3(self):
        A = eye(self.n)
        xtrue = transpose(arange(self.n,0,-1))
        self.assertCompatibleSystem(A, xtrue)

    def testBidiagonalA(self):
        A = lowerBidiagonalMatrix(20,self.n)
        xtrue = transpose(arange(self.n,0,-1))
        self.assertCompatibleSystem(A,xtrue)

    def testScalarB(self):
        A = array([[1.0, 2.0]])
        b = 3.0
        x = lsmr(A, b)[0]
        assert_almost_equal(norm(A.dot(x) - b), 0)

    def testComplexX(self):
        A = eye(self.n)
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexX0(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1))
        b = aslinearoperator(A).matvec(xtrue)
        x0 = zeros(self.n, dtype=complex)
        x = lsmr(A, b, x0=x0)[0]
        assert_almost_equal(norm(x - xtrue), 0, decimal=5)

    def testComplexA(self):
        A = 4 * eye(self.n) + 1j * ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1).astype(complex))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexB(self):
        A = 4 * eye(self.n) + ones((self.n, self.n))
        xtrue = transpose(arange(self.n, 0, -1) * (1 + 1j))
        b = aslinearoperator(A).matvec(xtrue)
        x = lsmr(A, b)[0]
        assert_almost_equal(norm(x - xtrue), 0, decimal=5)

    def testColumnB(self):
        A = eye(self.n)
        b = ones((self.n, 1))
        x = lsmr(A, b)[0]
        assert_almost_equal(norm(A.dot(x) - b.ravel()), 0)

    def testInitialization(self):
        # Test that the default setting is not modified
        x_ref, _, itn_ref, normr_ref, *_ = lsmr(G, b)
        assert_allclose(norm(b - G@x_ref), normr_ref, atol=1e-6)

        # Test passing zeros yields similiar result
        x0 = zeros(b.shape)
        x = lsmr(G, b, x0=x0)[0]
        assert_array_almost_equal(x, x_ref)

        # Test warm-start with single iteration
        x0 = lsmr(G, b, maxiter=1)[0]

        x, _, itn, normr, *_ = lsmr(G, b, x0=x0)
        assert_allclose(norm(b - G@x), normr, atol=1e-6)

        # NOTE(gh-12139): This doesn't always converge to the same value as
        # ref because error estimates will be slightly different when calculated
        # from zeros vs x0 as a result only compare norm and itn (not x).

        # x generally converges 1 iteration faster because it started at x0.
        # itn == itn_ref means that lsmr(x0) took an extra iteration see above.
        # -1 is technically possible but is rare (1 in 100000) so it's more
        # likely to be an error elsewhere.
        assert itn - itn_ref in (0, 1)

        # If an extra iteration is performed normr may be 0, while normr_ref
        # may be much larger.
        assert normr < normr_ref * (1 + 1e-6)


class TestLSMRReturns:
    def setup_method(self):
        self.n = 10
        self.A = lowerBidiagonalMatrix(20,self.n)
        self.xtrue = transpose(arange(self.n,0,-1))
        self.Afun = aslinearoperator(self.A)
        self.b = self.Afun.matvec(self.xtrue)
        self.returnValues = lsmr(self.A,self.b)

    def testNormr(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert_almost_equal(normr, norm(self.b - self.Afun.matvec(x)))

    def testNormar(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert_almost_equal(normar,
                norm(self.Afun.rmatvec(self.b - self.Afun.matvec(x))))

    def testNormx(self):
        x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        assert_almost_equal(normx, norm(x))


def lowerBidiagonalMatrix(m, n):
    # This is a simple example for testing LSMR.
    # It uses the leading m*n submatrix from
    # A = [ 1
    #       1 2
    #         2 3
    #           3 4
    #             ...
    #               n ]
    # suitably padded by zeros.
    #
    # 04 Jun 2010: First version for distribution with lsmr.py
    if m <= n:
        row = hstack((arange(m, dtype=int),
                      arange(1, m, dtype=int)))
        col = hstack((arange(m, dtype=int),
                      arange(m-1, dtype=int)))
        data = hstack((arange(1, m+1, dtype=float),
                       arange(1,m, dtype=float)))
        return coo_matrix((data, (row, col)), shape=(m,n))
    else:
        row = hstack((arange(n, dtype=int),
                      arange(1, n+1, dtype=int)))
        col = hstack((arange(n, dtype=int),
                      arange(n, dtype=int)))
        data = hstack((arange(1, n+1, dtype=float),
                       arange(1,n+1, dtype=float)))
        return coo_matrix((data,(row, col)), shape=(m,n))


def lsmrtest(m, n, damp):
    """Verbose testing of lsmr"""

    A = lowerBidiagonalMatrix(m,n)
    xtrue = arange(n,0,-1, dtype=float)
    Afun = aslinearoperator(A)

    b = Afun.matvec(xtrue)

    atol = 1.0e-7
    btol = 1.0e-7
    conlim = 1.0e+10
    itnlim = 10*n
    show = 1

    x, istop, itn, normr, normar, norma, conda, normx \
      = lsmr(A, b, damp, atol, btol, conlim, itnlim, show)

    j1 = min(n,5)
    j2 = max(n-4,1)
    print(' ')
    print('First elements of x:')
    str = ['%10.4f' % (xi) for xi in x[0:j1]]
    print(''.join(str))
    print(' ')
    print('Last  elements of x:')
    str = ['%10.4f' % (xi) for xi in x[j2-1:]]
    print(''.join(str))

    r = b - Afun.matvec(x)
    r2 = sqrt(norm(r)**2 + (damp*norm(x))**2)
    print(' ')
    str = 'normr (est.)  %17.10e' % (normr)
    str2 = 'normr (true)  %17.10e' % (r2)
    print(str)
    print(str2)
    print(' ')


if __name__ == "__main__":
    lsmrtest(20,10,0)
