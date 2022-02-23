"""Test functions for the sparse.linalg.interface module
"""

from functools import partial
from itertools import product
import operator
import pytest
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal

import numpy as np
import scipy.sparse as sparse

from scipy.sparse.linalg import interface
from scipy.sparse.sputils import matrix


class TestLinearOperator:
    def setup_method(self):
        self.A = np.array([[1,2,3],
                           [4,5,6]])
        self.B = np.array([[1,2],
                           [3,4],
                           [5,6]])
        self.C = np.array([[1,2],
                           [3,4]])

    def test_matvec(self):
        def get_matvecs(A):
            return [{
                        'shape': A.shape,
                        'matvec': lambda x: np.dot(A, x).reshape(A.shape[0]),
                        'rmatvec': lambda x: np.dot(A.T.conj(),
                                                    x).reshape(A.shape[1])
                    },
                    {
                        'shape': A.shape,
                        'matvec': lambda x: np.dot(A, x),
                        'rmatvec': lambda x: np.dot(A.T.conj(), x),
                        'rmatmat': lambda x: np.dot(A.T.conj(), x),
                        'matmat': lambda x: np.dot(A, x)
                    }]

        for matvecs in get_matvecs(self.A):
            A = interface.LinearOperator(**matvecs)

            assert_(A.args == ())

            assert_equal(A.matvec(np.array([1,2,3])), [14,32])
            assert_equal(A.matvec(np.array([[1],[2],[3]])), [[14],[32]])
            assert_equal(A * np.array([1,2,3]), [14,32])
            assert_equal(A * np.array([[1],[2],[3]]), [[14],[32]])
            assert_equal(A.dot(np.array([1,2,3])), [14,32])
            assert_equal(A.dot(np.array([[1],[2],[3]])), [[14],[32]])

            assert_equal(A.matvec(matrix([[1],[2],[3]])), [[14],[32]])
            assert_equal(A * matrix([[1],[2],[3]]), [[14],[32]])
            assert_equal(A.dot(matrix([[1],[2],[3]])), [[14],[32]])

            assert_equal((2*A)*[1,1,1], [12,30])
            assert_equal((2 * A).rmatvec([1, 1]), [10, 14, 18])
            assert_equal((2*A).H.matvec([1,1]), [10, 14, 18])
            assert_equal((2*A)*[[1],[1],[1]], [[12],[30]])
            assert_equal((2 * A).matmat([[1], [1], [1]]), [[12], [30]])
            assert_equal((A*2)*[1,1,1], [12,30])
            assert_equal((A*2)*[[1],[1],[1]], [[12],[30]])
            assert_equal((2j*A)*[1,1,1], [12j,30j])
            assert_equal((A+A)*[1,1,1], [12, 30])
            assert_equal((A + A).rmatvec([1, 1]), [10, 14, 18])
            assert_equal((A+A).H.matvec([1,1]), [10, 14, 18])
            assert_equal((A+A)*[[1],[1],[1]], [[12], [30]])
            assert_equal((A+A).matmat([[1],[1],[1]]), [[12], [30]])
            assert_equal((-A)*[1,1,1], [-6,-15])
            assert_equal((-A)*[[1],[1],[1]], [[-6],[-15]])
            assert_equal((A-A)*[1,1,1], [0,0])
            assert_equal((A - A) * [[1], [1], [1]], [[0], [0]])

            X = np.array([[1, 2], [3, 4]])
            # A_asarray = np.array([[1, 2, 3], [4, 5, 6]])
            assert_equal((2 * A).rmatmat(X), np.dot((2 * self.A).T, X))
            assert_equal((A * 2).rmatmat(X), np.dot((self.A * 2).T, X))
            assert_equal((2j * A).rmatmat(X),
                         np.dot((2j * self.A).T.conj(), X))
            assert_equal((A * 2j).rmatmat(X),
                         np.dot((self.A * 2j).T.conj(), X))
            assert_equal((A + A).rmatmat(X),
                         np.dot((self.A + self.A).T, X))
            assert_equal((A + 2j * A).rmatmat(X),
                         np.dot((self.A + 2j * self.A).T.conj(), X))
            assert_equal((-A).rmatmat(X), np.dot((-self.A).T, X))
            assert_equal((A - A).rmatmat(X),
                         np.dot((self.A - self.A).T, X))
            assert_equal((2j * A).rmatmat(2j * X),
                         np.dot((2j * self.A).T.conj(), 2j * X))

            z = A+A
            assert_(len(z.args) == 2 and z.args[0] is A and z.args[1] is A)
            z = 2*A
            assert_(len(z.args) == 2 and z.args[0] is A and z.args[1] == 2)

            assert_(isinstance(A.matvec([1, 2, 3]), np.ndarray))
            assert_(isinstance(A.matvec(np.array([[1],[2],[3]])), np.ndarray))
            assert_(isinstance(A * np.array([1,2,3]), np.ndarray))
            assert_(isinstance(A * np.array([[1],[2],[3]]), np.ndarray))
            assert_(isinstance(A.dot(np.array([1,2,3])), np.ndarray))
            assert_(isinstance(A.dot(np.array([[1],[2],[3]])), np.ndarray))

            assert_(isinstance(A.matvec(matrix([[1],[2],[3]])), np.ndarray))
            assert_(isinstance(A * matrix([[1],[2],[3]]), np.ndarray))
            assert_(isinstance(A.dot(matrix([[1],[2],[3]])), np.ndarray))

            assert_(isinstance(2*A, interface._ScaledLinearOperator))
            assert_(isinstance(2j*A, interface._ScaledLinearOperator))
            assert_(isinstance(A+A, interface._SumLinearOperator))
            assert_(isinstance(-A, interface._ScaledLinearOperator))
            assert_(isinstance(A-A, interface._SumLinearOperator))

            assert_((2j*A).dtype == np.complex_)

            assert_raises(ValueError, A.matvec, np.array([1,2]))
            assert_raises(ValueError, A.matvec, np.array([1,2,3,4]))
            assert_raises(ValueError, A.matvec, np.array([[1],[2]]))
            assert_raises(ValueError, A.matvec, np.array([[1],[2],[3],[4]]))

            assert_raises(ValueError, lambda: A*A)
            assert_raises(ValueError, lambda: A**2)

        for matvecsA, matvecsB in product(get_matvecs(self.A),
                                          get_matvecs(self.B)):
            A = interface.LinearOperator(**matvecsA)
            B = interface.LinearOperator(**matvecsB)
            # AtimesB = np.array([[22, 28], [49, 64]])
            AtimesB = self.A.dot(self.B)
            X = np.array([[1, 2], [3, 4]])

            assert_equal((A * B).rmatmat(X), np.dot((AtimesB).T, X))
            assert_equal((2j * A * B).rmatmat(X),
                         np.dot((2j * AtimesB).T.conj(), X))

            assert_equal((A*B)*[1,1], [50,113])
            assert_equal((A*B)*[[1],[1]], [[50],[113]])
            assert_equal((A*B).matmat([[1],[1]]), [[50],[113]])

            assert_equal((A * B).rmatvec([1, 1]), [71, 92])
            assert_equal((A * B).H.matvec([1, 1]), [71, 92])

            assert_(isinstance(A*B, interface._ProductLinearOperator))

            assert_raises(ValueError, lambda: A+B)
            assert_raises(ValueError, lambda: A**2)

            z = A*B
            assert_(len(z.args) == 2 and z.args[0] is A and z.args[1] is B)

        for matvecsC in get_matvecs(self.C):
            C = interface.LinearOperator(**matvecsC)
            X = np.array([[1, 2], [3, 4]])

            assert_equal(C.rmatmat(X), np.dot((self.C).T, X))
            assert_equal((C**2).rmatmat(X),
                         np.dot((np.dot(self.C, self.C)).T, X))

            assert_equal((C**2)*[1,1], [17,37])
            assert_equal((C**2).rmatvec([1, 1]), [22, 32])
            assert_equal((C**2).H.matvec([1, 1]), [22, 32])
            assert_equal((C**2).matmat([[1],[1]]), [[17],[37]])

            assert_(isinstance(C**2, interface._PowerLinearOperator))

    def test_matmul(self):
        D = {'shape': self.A.shape,
             'matvec': lambda x: np.dot(self.A, x).reshape(self.A.shape[0]),
             'rmatvec': lambda x: np.dot(self.A.T.conj(),
                                         x).reshape(self.A.shape[1]),
             'rmatmat': lambda x: np.dot(self.A.T.conj(), x),
             'matmat': lambda x: np.dot(self.A, x)}
        A = interface.LinearOperator(**D)
        B = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = B[0]

        assert_equal(operator.matmul(A, b), A * b)
        assert_equal(operator.matmul(A, B), A * B)
        assert_raises(ValueError, operator.matmul, A, 2)
        assert_raises(ValueError, operator.matmul, 2, A)


class TestAsLinearOperator:
    def setup_method(self):
        self.cases = []

        def make_cases(original, dtype):
            cases = []

            cases.append((matrix(original, dtype=dtype), original))
            cases.append((np.array(original, dtype=dtype), original))
            cases.append((sparse.csr_matrix(original, dtype=dtype), original))

            # Test default implementations of _adjoint and _rmatvec, which
            # refer to each other.
            def mv(x, dtype):
                y = original.dot(x)
                if len(x.shape) == 2:
                    y = y.reshape(-1, 1)
                return y

            def rmv(x, dtype):
                return original.T.conj().dot(x)

            class BaseMatlike(interface.LinearOperator):
                args = ()

                def __init__(self, dtype):
                    self.dtype = np.dtype(dtype)
                    self.shape = original.shape

                def _matvec(self, x):
                    return mv(x, self.dtype)

            class HasRmatvec(BaseMatlike):
                args = ()

                def _rmatvec(self,x):
                    return rmv(x, self.dtype)

            class HasAdjoint(BaseMatlike):
                args = ()

                def _adjoint(self):
                    shape = self.shape[1], self.shape[0]
                    matvec = partial(rmv, dtype=self.dtype)
                    rmatvec = partial(mv, dtype=self.dtype)
                    return interface.LinearOperator(matvec=matvec,
                                                    rmatvec=rmatvec,
                                                    dtype=self.dtype,
                                                    shape=shape)

            class HasRmatmat(HasRmatvec):
                def _matmat(self, x):
                    return original.dot(x)

                def _rmatmat(self, x):
                    return original.T.conj().dot(x)

            cases.append((HasRmatvec(dtype), original))
            cases.append((HasAdjoint(dtype), original))
            cases.append((HasRmatmat(dtype), original))
            return cases

        original = np.array([[1,2,3], [4,5,6]])
        self.cases += make_cases(original, np.int32)
        self.cases += make_cases(original, np.float32)
        self.cases += make_cases(original, np.float64)
        self.cases += [(interface.aslinearoperator(M).T, A.T)
                       for M, A in make_cases(original.T, np.float64)]
        self.cases += [(interface.aslinearoperator(M).H, A.T.conj())
                       for M, A in make_cases(original.T, np.float64)]

        original = np.array([[1, 2j, 3j], [4j, 5j, 6]])
        self.cases += make_cases(original, np.complex_)
        self.cases += [(interface.aslinearoperator(M).T, A.T)
                       for M, A in make_cases(original.T, np.complex_)]
        self.cases += [(interface.aslinearoperator(M).H, A.T.conj())
                       for M, A in make_cases(original.T, np.complex_)]

    def test_basic(self):

        for M, A_array in self.cases:
            A = interface.aslinearoperator(M)
            M,N = A.shape

            xs = [np.array([1, 2, 3]),
                  np.array([[1], [2], [3]])]
            ys = [np.array([1, 2]), np.array([[1], [2]])]

            if A.dtype == np.complex_:
                xs += [np.array([1, 2j, 3j]),
                       np.array([[1], [2j], [3j]])]
                ys += [np.array([1, 2j]), np.array([[1], [2j]])]

            x2 = np.array([[1, 4], [2, 5], [3, 6]])

            for x in xs:
                assert_equal(A.matvec(x), A_array.dot(x))
                assert_equal(A * x, A_array.dot(x))

            assert_equal(A.matmat(x2), A_array.dot(x2))
            assert_equal(A * x2, A_array.dot(x2))

            for y in ys:
                assert_equal(A.rmatvec(y), A_array.T.conj().dot(y))
                assert_equal(A.T.matvec(y), A_array.T.dot(y))
                assert_equal(A.H.matvec(y), A_array.T.conj().dot(y))

            for y in ys:
                if y.ndim < 2:
                    continue
                assert_equal(A.rmatmat(y), A_array.T.conj().dot(y))
                assert_equal(A.T.matmat(y), A_array.T.dot(y))
                assert_equal(A.H.matmat(y), A_array.T.conj().dot(y))

            if hasattr(M,'dtype'):
                assert_equal(A.dtype, M.dtype)

            assert_(hasattr(A, 'args'))

    def test_dot(self):

        for M, A_array in self.cases:
            A = interface.aslinearoperator(M)
            M,N = A.shape

            x0 = np.array([1, 2, 3])
            x1 = np.array([[1], [2], [3]])
            x2 = np.array([[1, 4], [2, 5], [3, 6]])

            assert_equal(A.dot(x0), A_array.dot(x0))
            assert_equal(A.dot(x1), A_array.dot(x1))
            assert_equal(A.dot(x2), A_array.dot(x2))


def test_repr():
    A = interface.LinearOperator(shape=(1, 1), matvec=lambda x: 1)
    repr_A = repr(A)
    assert_('unspecified dtype' not in repr_A, repr_A)


def test_identity():
    ident = interface.IdentityOperator((3, 3))
    assert_equal(ident * [1, 2, 3], [1, 2, 3])
    assert_equal(ident.dot(np.arange(9).reshape(3, 3)).ravel(), np.arange(9))

    assert_raises(ValueError, ident.matvec, [1, 2, 3, 4])


def test_attributes():
    A = interface.aslinearoperator(np.arange(16).reshape(4, 4))

    def always_four_ones(x):
        x = np.asarray(x)
        assert_(x.shape == (3,) or x.shape == (3, 1))
        return np.ones(4)

    B = interface.LinearOperator(shape=(4, 3), matvec=always_four_ones)

    for op in [A, B, A * B, A.H, A + A, B + B, A**4]:
        assert_(hasattr(op, "dtype"))
        assert_(hasattr(op, "shape"))
        assert_(hasattr(op, "_matvec"))

def matvec(x):
    """ Needed for test_pickle as local functions are not pickleable """
    return np.zeros(3)

def test_pickle():
    import pickle

    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        A = interface.LinearOperator((3, 3), matvec)
        s = pickle.dumps(A, protocol=protocol)
        B = pickle.loads(s)

        for k in A.__dict__:
            assert_equal(getattr(A, k), getattr(B, k))

def test_inheritance():
    class Empty(interface.LinearOperator):
        pass

    with warns(RuntimeWarning, match="should implement at least"):
        assert_raises(TypeError, Empty)

    class Identity(interface.LinearOperator):
        def __init__(self, n):
            super().__init__(dtype=None, shape=(n, n))

        def _matvec(self, x):
            return x

    id3 = Identity(3)
    assert_equal(id3.matvec([1, 2, 3]), [1, 2, 3])
    assert_raises(NotImplementedError, id3.rmatvec, [4, 5, 6])

    class MatmatOnly(interface.LinearOperator):
        def __init__(self, A):
            super().__init__(A.dtype, A.shape)
            self.A = A

        def _matmat(self, x):
            return self.A.dot(x)

    mm = MatmatOnly(np.random.randn(5, 3))
    assert_equal(mm.matvec(np.random.randn(3)).shape, (5,))

def test_dtypes_of_operator_sum():
    # gh-6078

    mat_complex = np.random.rand(2,2) + 1j * np.random.rand(2,2)
    mat_real = np.random.rand(2,2)

    complex_operator = interface.aslinearoperator(mat_complex)
    real_operator = interface.aslinearoperator(mat_real)

    sum_complex = complex_operator + complex_operator
    sum_real = real_operator + real_operator

    assert_equal(sum_real.dtype, np.float64)
    assert_equal(sum_complex.dtype, np.complex128)

def test_no_double_init():
    call_count = [0]

    def matvec(v):
        call_count[0] += 1
        return v

    # It should call matvec exactly once (in order to determine the
    # operator dtype)
    interface.LinearOperator((2, 2), matvec=matvec)
    assert_equal(call_count[0], 1)

def test_adjoint_conjugate():
    X = np.array([[1j]])
    A = interface.aslinearoperator(X)

    B = 1j * A
    Y = 1j * X

    v = np.array([1])

    assert_equal(B.dot(v), Y.dot(v))
    assert_equal(B.H.dot(v), Y.T.conj().dot(v))

def test_ndim():
    X = np.array([[1]])
    A = interface.aslinearoperator(X)
    assert_equal(A.ndim, 2)

def test_transpose_noconjugate():
    X = np.array([[1j]])
    A = interface.aslinearoperator(X)

    B = 1j * A
    Y = 1j * X

    v = np.array([1])

    assert_equal(B.dot(v), Y.dot(v))
    assert_equal(B.T.dot(v), Y.T.dot(v))
