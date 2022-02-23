# Test interfaces to fortran blas.
#
# The tests are more of interface than they are of the underlying blas.
# Only very small matrices checked -- N=3 or so.
#
# !! Complex calculations really aren't checked that carefully.
# !! Only real valued complex numbers are used in tests.

from numpy import float32, float64, complex64, complex128, arange, array, \
                  zeros, shape, transpose, newaxis, common_type, conjugate

from scipy.linalg import _fblas as fblas

from numpy.testing import assert_array_equal, \
    assert_allclose, assert_array_almost_equal, assert_

import pytest

# decimal accuracy to require between Python and LAPACK/BLAS calculations
accuracy = 5

# Since numpy.dot likely uses the same blas, use this routine
# to check.


def matrixmultiply(a, b):
    if len(b.shape) == 1:
        b_is_vector = True
        b = b[:, newaxis]
    else:
        b_is_vector = False
    assert_(a.shape[1] == b.shape[0])
    c = zeros((a.shape[0], b.shape[1]), common_type(a, b))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            s = 0
            for k in range(a.shape[1]):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    if b_is_vector:
        c = c.reshape((a.shape[0],))
    return c

##################################################
# Test blas ?axpy


class BaseAxpy:
    ''' Mixin class for axpy tests '''

    def test_default_a(self):
        x = arange(3., dtype=self.dtype)
        y = arange(3., dtype=x.dtype)
        real_y = x*1.+y
        y = self.blas_func(x, y)
        assert_array_equal(real_y, y)

    def test_simple(self):
        x = arange(3., dtype=self.dtype)
        y = arange(3., dtype=x.dtype)
        real_y = x*3.+y
        y = self.blas_func(x, y, a=3.)
        assert_array_equal(real_y, y)

    def test_x_stride(self):
        x = arange(6., dtype=self.dtype)
        y = zeros(3, x.dtype)
        y = arange(3., dtype=x.dtype)
        real_y = x[::2]*3.+y
        y = self.blas_func(x, y, a=3., n=3, incx=2)
        assert_array_equal(real_y, y)

    def test_y_stride(self):
        x = arange(3., dtype=self.dtype)
        y = zeros(6, x.dtype)
        real_y = x*3.+y[::2]
        y = self.blas_func(x, y, a=3., n=3, incy=2)
        assert_array_equal(real_y, y[::2])

    def test_x_and_y_stride(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        real_y = x[::4]*3.+y[::2]
        y = self.blas_func(x, y, a=3., n=3, incx=4, incy=2)
        assert_array_equal(real_y, y[::2])

    def test_x_bad_size(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=4, incx=5)

    def test_y_bad_size(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=3, incy=5)


try:
    class TestSaxpy(BaseAxpy):
        blas_func = fblas.saxpy
        dtype = float32
except AttributeError:
    class TestSaxpy:
        pass


class TestDaxpy(BaseAxpy):
    blas_func = fblas.daxpy
    dtype = float64


try:
    class TestCaxpy(BaseAxpy):
        blas_func = fblas.caxpy
        dtype = complex64
except AttributeError:
    class TestCaxpy:
        pass


class TestZaxpy(BaseAxpy):
    blas_func = fblas.zaxpy
    dtype = complex128


##################################################
# Test blas ?scal

class BaseScal:
    ''' Mixin class for scal testing '''

    def test_simple(self):
        x = arange(3., dtype=self.dtype)
        real_x = x*3.
        x = self.blas_func(3., x)
        assert_array_equal(real_x, x)

    def test_x_stride(self):
        x = arange(6., dtype=self.dtype)
        real_x = x.copy()
        real_x[::2] = x[::2]*array(3., self.dtype)
        x = self.blas_func(3., x, n=3, incx=2)
        assert_array_equal(real_x, x)

    def test_x_bad_size(self):
        x = arange(12., dtype=self.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(2., x, n=4, incx=5)


try:
    class TestSscal(BaseScal):
        blas_func = fblas.sscal
        dtype = float32
except AttributeError:
    class TestSscal:
        pass


class TestDscal(BaseScal):
    blas_func = fblas.dscal
    dtype = float64


try:
    class TestCscal(BaseScal):
        blas_func = fblas.cscal
        dtype = complex64
except AttributeError:
    class TestCscal:
        pass


class TestZscal(BaseScal):
    blas_func = fblas.zscal
    dtype = complex128


##################################################
# Test blas ?copy

class BaseCopy:
    ''' Mixin class for copy testing '''

    def test_simple(self):
        x = arange(3., dtype=self.dtype)
        y = zeros(shape(x), x.dtype)
        y = self.blas_func(x, y)
        assert_array_equal(x, y)

    def test_x_stride(self):
        x = arange(6., dtype=self.dtype)
        y = zeros(3, x.dtype)
        y = self.blas_func(x, y, n=3, incx=2)
        assert_array_equal(x[::2], y)

    def test_y_stride(self):
        x = arange(3., dtype=self.dtype)
        y = zeros(6, x.dtype)
        y = self.blas_func(x, y, n=3, incy=2)
        assert_array_equal(x, y[::2])

    def test_x_and_y_stride(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        y = self.blas_func(x, y, n=3, incx=4, incy=2)
        assert_array_equal(x[::4], y[::2])

    def test_x_bad_size(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=4, incx=5)

    def test_y_bad_size(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=3, incy=5)

    # def test_y_bad_type(self):
    ##   Hmmm. Should this work?  What should be the output.
    #    x = arange(3.,dtype=self.dtype)
    #    y = zeros(shape(x))
    #    self.blas_func(x,y)
    #    assert_array_equal(x,y)


try:
    class TestScopy(BaseCopy):
        blas_func = fblas.scopy
        dtype = float32
except AttributeError:
    class TestScopy:
        pass


class TestDcopy(BaseCopy):
    blas_func = fblas.dcopy
    dtype = float64


try:
    class TestCcopy(BaseCopy):
        blas_func = fblas.ccopy
        dtype = complex64
except AttributeError:
    class TestCcopy:
        pass


class TestZcopy(BaseCopy):
    blas_func = fblas.zcopy
    dtype = complex128


##################################################
# Test blas ?swap

class BaseSwap:
    ''' Mixin class for swap tests '''

    def test_simple(self):
        x = arange(3., dtype=self.dtype)
        y = zeros(shape(x), x.dtype)
        desired_x = y.copy()
        desired_y = x.copy()
        x, y = self.blas_func(x, y)
        assert_array_equal(desired_x, x)
        assert_array_equal(desired_y, y)

    def test_x_stride(self):
        x = arange(6., dtype=self.dtype)
        y = zeros(3, x.dtype)
        desired_x = y.copy()
        desired_y = x.copy()[::2]
        x, y = self.blas_func(x, y, n=3, incx=2)
        assert_array_equal(desired_x, x[::2])
        assert_array_equal(desired_y, y)

    def test_y_stride(self):
        x = arange(3., dtype=self.dtype)
        y = zeros(6, x.dtype)
        desired_x = y.copy()[::2]
        desired_y = x.copy()
        x, y = self.blas_func(x, y, n=3, incy=2)
        assert_array_equal(desired_x, x)
        assert_array_equal(desired_y, y[::2])

    def test_x_and_y_stride(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        desired_x = y.copy()[::2]
        desired_y = x.copy()[::4]
        x, y = self.blas_func(x, y, n=3, incx=4, incy=2)
        assert_array_equal(desired_x, x[::4])
        assert_array_equal(desired_y, y[::2])

    def test_x_bad_size(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=4, incx=5)

    def test_y_bad_size(self):
        x = arange(12., dtype=self.dtype)
        y = zeros(6, x.dtype)
        with pytest.raises(Exception, match='failed for 1st keyword'):
            self.blas_func(x, y, n=3, incy=5)


try:
    class TestSswap(BaseSwap):
        blas_func = fblas.sswap
        dtype = float32
except AttributeError:
    class TestSswap:
        pass


class TestDswap(BaseSwap):
    blas_func = fblas.dswap
    dtype = float64


try:
    class TestCswap(BaseSwap):
        blas_func = fblas.cswap
        dtype = complex64
except AttributeError:
    class TestCswap:
        pass


class TestZswap(BaseSwap):
    blas_func = fblas.zswap
    dtype = complex128

##################################################
# Test blas ?gemv
# This will be a mess to test all cases.


class BaseGemv:
    ''' Mixin class for gemv tests '''

    def get_data(self, x_stride=1, y_stride=1):
        mult = array(1, dtype=self.dtype)
        if self.dtype in [complex64, complex128]:
            mult = array(1+1j, dtype=self.dtype)
        from numpy.random import normal, seed
        seed(1234)
        alpha = array(1., dtype=self.dtype) * mult
        beta = array(1., dtype=self.dtype) * mult
        a = normal(0., 1., (3, 3)).astype(self.dtype) * mult
        x = arange(shape(a)[0]*x_stride, dtype=self.dtype) * mult
        y = arange(shape(a)[1]*y_stride, dtype=self.dtype) * mult
        return alpha, beta, a, x, y

    def test_simple(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = alpha*matrixmultiply(a, x)+beta*y
        y = self.blas_func(alpha, a, x, beta, y)
        assert_array_almost_equal(desired_y, y)

    def test_default_beta_y(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = matrixmultiply(a, x)
        y = self.blas_func(1, a, x)
        assert_array_almost_equal(desired_y, y)

    def test_simple_transpose(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = alpha*matrixmultiply(transpose(a), x)+beta*y
        y = self.blas_func(alpha, a, x, beta, y, trans=1)
        assert_array_almost_equal(desired_y, y)

    def test_simple_transpose_conj(self):
        alpha, beta, a, x, y = self.get_data()
        desired_y = alpha*matrixmultiply(transpose(conjugate(a)), x)+beta*y
        y = self.blas_func(alpha, a, x, beta, y, trans=2)
        assert_array_almost_equal(desired_y, y)

    def test_x_stride(self):
        alpha, beta, a, x, y = self.get_data(x_stride=2)
        desired_y = alpha*matrixmultiply(a, x[::2])+beta*y
        y = self.blas_func(alpha, a, x, beta, y, incx=2)
        assert_array_almost_equal(desired_y, y)

    def test_x_stride_transpose(self):
        alpha, beta, a, x, y = self.get_data(x_stride=2)
        desired_y = alpha*matrixmultiply(transpose(a), x[::2])+beta*y
        y = self.blas_func(alpha, a, x, beta, y, trans=1, incx=2)
        assert_array_almost_equal(desired_y, y)

    def test_x_stride_assert(self):
        # What is the use of this test?
        alpha, beta, a, x, y = self.get_data(x_stride=2)
        with pytest.raises(Exception, match='failed for 3rd argument'):
            y = self.blas_func(1, a, x, 1, y, trans=0, incx=3)
        with pytest.raises(Exception, match='failed for 3rd argument'):
            y = self.blas_func(1, a, x, 1, y, trans=1, incx=3)

    def test_y_stride(self):
        alpha, beta, a, x, y = self.get_data(y_stride=2)
        desired_y = y.copy()
        desired_y[::2] = alpha*matrixmultiply(a, x)+beta*y[::2]
        y = self.blas_func(alpha, a, x, beta, y, incy=2)
        assert_array_almost_equal(desired_y, y)

    def test_y_stride_transpose(self):
        alpha, beta, a, x, y = self.get_data(y_stride=2)
        desired_y = y.copy()
        desired_y[::2] = alpha*matrixmultiply(transpose(a), x)+beta*y[::2]
        y = self.blas_func(alpha, a, x, beta, y, trans=1, incy=2)
        assert_array_almost_equal(desired_y, y)

    def test_y_stride_assert(self):
        # What is the use of this test?
        alpha, beta, a, x, y = self.get_data(y_stride=2)
        with pytest.raises(Exception, match='failed for 2nd keyword'):
            y = self.blas_func(1, a, x, 1, y, trans=0, incy=3)
        with pytest.raises(Exception, match='failed for 2nd keyword'):
            y = self.blas_func(1, a, x, 1, y, trans=1, incy=3)


try:
    class TestSgemv(BaseGemv):
        blas_func = fblas.sgemv
        dtype = float32

        def test_sgemv_on_osx(self):
            from itertools import product
            import sys
            import numpy as np

            if sys.platform != 'darwin':
                return

            def aligned_array(shape, align, dtype, order='C'):
                # Make array shape `shape` with aligned at `align` bytes
                d = dtype()
                # Make array of correct size with `align` extra bytes
                N = np.prod(shape)
                tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
                address = tmp.__array_interface__["data"][0]
                # Find offset into array giving desired alignment
                for offset in range(align):
                    if (address + offset) % align == 0:
                        break
                tmp = tmp[offset:offset+N*d.nbytes].view(dtype=dtype)
                return tmp.reshape(shape, order=order)

            def as_aligned(arr, align, dtype, order='C'):
                # Copy `arr` into an aligned array with same shape
                aligned = aligned_array(arr.shape, align, dtype, order)
                aligned[:] = arr[:]
                return aligned

            def assert_dot_close(A, X, desired):
                assert_allclose(self.blas_func(1.0, A, X), desired,
                                rtol=1e-5, atol=1e-7)

            testdata = product((15, 32), (10000,), (200, 89), ('C', 'F'))
            for align, m, n, a_order in testdata:
                A_d = np.random.rand(m, n)
                X_d = np.random.rand(n)
                desired = np.dot(A_d, X_d)
                # Calculation with aligned single precision
                A_f = as_aligned(A_d, align, np.float32, order=a_order)
                X_f = as_aligned(X_d, align, np.float32, order=a_order)
                assert_dot_close(A_f, X_f, desired)

except AttributeError:
    class TestSgemv:
        pass


class TestDgemv(BaseGemv):
    blas_func = fblas.dgemv
    dtype = float64


try:
    class TestCgemv(BaseGemv):
        blas_func = fblas.cgemv
        dtype = complex64
except AttributeError:
    class TestCgemv:
        pass


class TestZgemv(BaseGemv):
    blas_func = fblas.zgemv
    dtype = complex128


"""
##################################################
### Test blas ?ger
### This will be a mess to test all cases.

class BaseGer:
    def get_data(self,x_stride=1,y_stride=1):
        from numpy.random import normal, seed
        seed(1234)
        alpha = array(1., dtype = self.dtype)
        a = normal(0.,1.,(3,3)).astype(self.dtype)
        x = arange(shape(a)[0]*x_stride,dtype=self.dtype)
        y = arange(shape(a)[1]*y_stride,dtype=self.dtype)
        return alpha,a,x,y
    def test_simple(self):
        alpha,a,x,y = self.get_data()
        # tranpose takes care of Fortran vs. C(and Python) memory layout
        desired_a = alpha*transpose(x[:,newaxis]*y) + a
        self.blas_func(x,y,a)
        assert_array_almost_equal(desired_a,a)
    def test_x_stride(self):
        alpha,a,x,y = self.get_data(x_stride=2)
        desired_a = alpha*transpose(x[::2,newaxis]*y) + a
        self.blas_func(x,y,a,incx=2)
        assert_array_almost_equal(desired_a,a)
    def test_x_stride_assert(self):
        alpha,a,x,y = self.get_data(x_stride=2)
        with pytest.raises(ValueError, match='foo'):
            self.blas_func(x,y,a,incx=3)
    def test_y_stride(self):
        alpha,a,x,y = self.get_data(y_stride=2)
        desired_a = alpha*transpose(x[:,newaxis]*y[::2]) + a
        self.blas_func(x,y,a,incy=2)
        assert_array_almost_equal(desired_a,a)

    def test_y_stride_assert(self):
        alpha,a,x,y = self.get_data(y_stride=2)
        with pytest.raises(ValueError, match='foo'):
            self.blas_func(a,x,y,incy=3)

class TestSger(BaseGer):
    blas_func = fblas.sger
    dtype = float32
class TestDger(BaseGer):
    blas_func = fblas.dger
    dtype = float64
"""
##################################################
# Test blas ?gerc
# This will be a mess to test all cases.

"""
class BaseGerComplex(BaseGer):
    def get_data(self,x_stride=1,y_stride=1):
        from numpy.random import normal, seed
        seed(1234)
        alpha = array(1+1j, dtype = self.dtype)
        a = normal(0.,1.,(3,3)).astype(self.dtype)
        a = a + normal(0.,1.,(3,3)) * array(1j, dtype = self.dtype)
        x = normal(0.,1.,shape(a)[0]*x_stride).astype(self.dtype)
        x = x + x * array(1j, dtype = self.dtype)
        y = normal(0.,1.,shape(a)[1]*y_stride).astype(self.dtype)
        y = y + y * array(1j, dtype = self.dtype)
        return alpha,a,x,y
    def test_simple(self):
        alpha,a,x,y = self.get_data()
        # tranpose takes care of Fortran vs. C(and Python) memory layout
        a = a * array(0.,dtype = self.dtype)
        #desired_a = alpha*transpose(x[:,newaxis]*self.transform(y)) + a
        desired_a = alpha*transpose(x[:,newaxis]*y) + a
        #self.blas_func(x,y,a,alpha = alpha)
        fblas.cgeru(x,y,a,alpha = alpha)
        assert_array_almost_equal(desired_a,a)

    #def test_x_stride(self):
    #    alpha,a,x,y = self.get_data(x_stride=2)
    #    desired_a = alpha*transpose(x[::2,newaxis]*self.transform(y)) + a
    #    self.blas_func(x,y,a,incx=2)
    #    assert_array_almost_equal(desired_a,a)
    #def test_y_stride(self):
    #    alpha,a,x,y = self.get_data(y_stride=2)
    #    desired_a = alpha*transpose(x[:,newaxis]*self.transform(y[::2])) + a
    #    self.blas_func(x,y,a,incy=2)
    #    assert_array_almost_equal(desired_a,a)

class TestCgeru(BaseGerComplex):
    blas_func = fblas.cgeru
    dtype = complex64
    def transform(self,x):
        return x
class TestZgeru(BaseGerComplex):
    blas_func = fblas.zgeru
    dtype = complex128
    def transform(self,x):
        return x

class TestCgerc(BaseGerComplex):
    blas_func = fblas.cgerc
    dtype = complex64
    def transform(self,x):
        return conjugate(x)

class TestZgerc(BaseGerComplex):
    blas_func = fblas.zgerc
    dtype = complex128
    def transform(self,x):
        return conjugate(x)
"""
