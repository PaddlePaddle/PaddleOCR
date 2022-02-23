import itertools
from numpy.testing import assert_array_almost_equal, assert_allclose, assert_
from numpy import (array, eye, zeros, empty_like, empty, tril_indices_from,
                   tril, triu_indices_from, spacing, float32, float64,
                   complex64, complex128)
from numpy.random import rand, randint, seed
from scipy.linalg import ldl
from pytest import raises as assert_raises, warns
from numpy import ComplexWarning


def test_args():
    A = eye(3)
    # Nonsquare array
    assert_raises(ValueError, ldl, A[:, :2])
    # Complex matrix with imaginary diagonal entries with "hermitian=True"
    with warns(ComplexWarning):
        ldl(A*1j)


def test_empty_array():
    a = empty((0, 0), dtype=complex)
    l, d, p = ldl(empty((0, 0)))
    assert_array_almost_equal(l, empty_like(a))
    assert_array_almost_equal(d, empty_like(a))
    assert_array_almost_equal(p, array([], dtype=int))


def test_simple():
    a = array([[-0.39-0.71j, 5.14-0.64j, -7.86-2.96j, 3.80+0.92j],
               [5.14-0.64j, 8.86+1.81j, -3.52+0.58j, 5.32-1.59j],
               [-7.86-2.96j, -3.52+0.58j, -2.83-0.03j, -1.54-2.86j],
               [3.80+0.92j, 5.32-1.59j, -1.54-2.86j, -0.56+0.12j]])
    b = array([[5., 10, 1, 18],
               [10., 2, 11, 1],
               [1., 11, 19, 9],
               [18., 1, 9, 0]])
    c = array([[52., 97, 112, 107, 50],
               [97., 114, 89, 98, 13],
               [112., 89, 64, 33, 6],
               [107., 98, 33, 60, 73],
               [50., 13, 6, 73, 77]])

    d = array([[2., 2, -4, 0, 4],
               [2., -2, -2, 10, -8],
               [-4., -2, 6, -8, -4],
               [0., 10, -8, 6, -6],
               [4., -8, -4, -6, 10]])
    e = array([[-1.36+0.00j, 0+0j, 0+0j, 0+0j],
               [1.58-0.90j, -8.87+0j, 0+0j, 0+0j],
               [2.21+0.21j, -1.84+0.03j, -4.63+0j, 0+0j],
               [3.91-1.50j, -1.78-1.18j, 0.11-0.11j, -1.84+0.00j]])
    for x in (b, c, d):
        l, d, p = ldl(x)
        assert_allclose(l.dot(d).dot(l.T), x, atol=spacing(1000.), rtol=0)

        u, d, p = ldl(x, lower=False)
        assert_allclose(u.dot(d).dot(u.T), x, atol=spacing(1000.), rtol=0)

    l, d, p = ldl(a, hermitian=False)
    assert_allclose(l.dot(d).dot(l.T), a, atol=spacing(1000.), rtol=0)

    u, d, p = ldl(a, lower=False, hermitian=False)
    assert_allclose(u.dot(d).dot(u.T), a, atol=spacing(1000.), rtol=0)

    # Use upper part for the computation and use the lower part for comparison
    l, d, p = ldl(e.conj().T, lower=0)
    assert_allclose(tril(l.dot(d).dot(l.conj().T)-e), zeros((4, 4)),
                    atol=spacing(1000.), rtol=0)


def test_permutations():
    seed(1234)
    for _ in range(10):
        n = randint(1, 100)
        # Random real/complex array
        x = rand(n, n) if randint(2) else rand(n, n) + rand(n, n)*1j
        x = x + x.conj().T
        x += eye(n)*randint(5, 1e6)
        l_ind = tril_indices_from(x, k=-1)
        u_ind = triu_indices_from(x, k=1)

        # Test whether permutations lead to a triangular array
        u, d, p = ldl(x, lower=0)
        # lower part should be zero
        assert_(not any(u[p, :][l_ind]), 'Spin {} failed'.format(_))

        l, d, p = ldl(x, lower=1)
        # upper part should be zero
        assert_(not any(l[p, :][u_ind]), 'Spin {} failed'.format(_))


def test_ldl_type_size_combinations():
    seed(1234)
    sizes = [30, 750]
    real_dtypes = [float32, float64]
    complex_dtypes = [complex64, complex128]

    for n, dtype in itertools.product(sizes, real_dtypes):
        msg = ("Failed for size: {}, dtype: {}".format(n, dtype))

        x = rand(n, n).astype(dtype)
        x = x + x.T
        x += eye(n, dtype=dtype)*dtype(randint(5, 1e6))

        l, d1, p = ldl(x)
        u, d2, p = ldl(x, lower=0)
        rtol = 1e-4 if dtype is float32 else 1e-10
        assert_allclose(l.dot(d1).dot(l.T), x, rtol=rtol, err_msg=msg)
        assert_allclose(u.dot(d2).dot(u.T), x, rtol=rtol, err_msg=msg)

    for n, dtype in itertools.product(sizes, complex_dtypes):
        msg1 = ("Her failed for size: {}, dtype: {}".format(n, dtype))
        msg2 = ("Sym failed for size: {}, dtype: {}".format(n, dtype))

        # Complex hermitian upper/lower
        x = (rand(n, n)+1j*rand(n, n)).astype(dtype)
        x = x+x.conj().T
        x += eye(n, dtype=dtype)*dtype(randint(5, 1e6))

        l, d1, p = ldl(x)
        u, d2, p = ldl(x, lower=0)
        rtol = 1e-4 if dtype is complex64 else 1e-10
        assert_allclose(l.dot(d1).dot(l.conj().T), x, rtol=rtol, err_msg=msg1)
        assert_allclose(u.dot(d2).dot(u.conj().T), x, rtol=rtol, err_msg=msg1)

        # Complex symmetric upper/lower
        x = (rand(n, n)+1j*rand(n, n)).astype(dtype)
        x = x+x.T
        x += eye(n, dtype=dtype)*dtype(randint(5, 1e6))

        l, d1, p = ldl(x, hermitian=0)
        u, d2, p = ldl(x, lower=0, hermitian=0)
        assert_allclose(l.dot(d1).dot(l.T), x, rtol=rtol, err_msg=msg2)
        assert_allclose(u.dot(d2).dot(u.T), x, rtol=rtol, err_msg=msg2)
