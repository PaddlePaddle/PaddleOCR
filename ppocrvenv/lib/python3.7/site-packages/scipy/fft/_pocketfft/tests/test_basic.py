# Created by Pearu Peterson, September 2002

from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
                           assert_array_almost_equal_nulp, assert_array_less,
                           assert_allclose)
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
                                  rfft, irfft, rfftn, irfftn, fft2,
                                  hfft, ihfft, hfftn, ihfftn)

from numpy import (arange, add, array, asarray, zeros, dot, exp, pi,
                   swapaxes, cdouble)
import numpy as np
import numpy.fft
from numpy.random import rand

# "large" composite numbers supported by FFT._PYPOCKETFFT
LARGE_COMPOSITE_SIZES = [
    2**13,
    2**5 * 3**5,
    2**3 * 3**3 * 5**2,
]
SMALL_COMPOSITE_SIZES = [
    2,
    2*3*5,
    2*2*3*3,
]
# prime
LARGE_PRIME_SIZES = [
    2011
]
SMALL_PRIME_SIZES = [
    29
]


def _assert_close_in_norm(x, y, rtol, size, rdt):
    # helper function for testing
    err_msg = "size: %s  rdt: %s" % (size, rdt)
    assert_array_less(np.linalg.norm(x - y), rtol*np.linalg.norm(x), err_msg)


def random(size):
    return rand(*size)

def swap_byteorder(arr):
    """Returns the same array with swapped byteorder"""
    dtype = arr.dtype.newbyteorder('S')
    return arr.astype(dtype)

def get_mat(n):
    data = arange(n)
    data = add.outer(data, data)
    return data


def direct_dft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = -arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w), x)
    return y


def direct_idft(x):
    x = asarray(x)
    n = len(x)
    y = zeros(n, dtype=cdouble)
    w = arange(n)*(2j*pi/n)
    for i in range(n):
        y[i] = dot(exp(i*w), x)/n
    return y


def direct_dftn(x):
    x = asarray(x)
    for axis in range(len(x.shape)):
        x = fft(x, axis=axis)
    return x


def direct_idftn(x):
    x = asarray(x)
    for axis in range(len(x.shape)):
        x = ifft(x, axis=axis)
    return x


def direct_rdft(x):
    x = asarray(x)
    n = len(x)
    w = -arange(n)*(2j*pi/n)
    y = zeros(n//2+1, dtype=cdouble)
    for i in range(n//2+1):
        y[i] = dot(exp(i*w), x)
    return y


def direct_irdft(x, n):
    x = asarray(x)
    x1 = zeros(n, dtype=cdouble)
    for i in range(n//2+1):
        x1[i] = x[i]
        if i > 0 and 2*i < n:
            x1[n-i] = np.conj(x[i])
    return direct_idft(x1).real


def direct_rdftn(x):
    return fftn(rfft(x), axes=range(x.ndim - 1))


class _TestFFTBase:
    def setup_method(self):
        self.cdt = None
        self.rdt = None
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([1,2,3,4+1j,1,2,3,4+2j], dtype=self.cdt)
        y = fft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_dft(x)
        assert_array_almost_equal(y,y1)
        x = np.array([1,2,3,4+0j,5], dtype=self.cdt)
        assert_array_almost_equal(fft(x),direct_dft(x))

    def test_n_argument_real(self):
        x1 = np.array([1,2,3,4], dtype=self.rdt)
        x2 = np.array([1,2,3,4], dtype=self.rdt)
        y = fft([x1,x2],n=4)
        assert_equal(y.dtype, self.cdt)
        assert_equal(y.shape,(2,4))
        assert_array_almost_equal(y[0],direct_dft(x1))
        assert_array_almost_equal(y[1],direct_dft(x2))

    def _test_n_argument_complex(self):
        x1 = np.array([1,2,3,4+1j], dtype=self.cdt)
        x2 = np.array([1,2,3,4+1j], dtype=self.cdt)
        y = fft([x1,x2],n=4)
        assert_equal(y.dtype, self.cdt)
        assert_equal(y.shape,(2,4))
        assert_array_almost_equal(y[0],direct_dft(x1))
        assert_array_almost_equal(y[1],direct_dft(x2))

    def test_djbfft(self):
        for i in range(2,14):
            n = 2**i
            x = np.arange(n)
            y = fft(x.astype(complex))
            y2 = numpy.fft.fft(x)
            assert_array_almost_equal(y,y2)
            y = fft(x)
            assert_array_almost_equal(y,y2)

    def test_invalid_sizes(self):
        assert_raises(ValueError, fft, [])
        assert_raises(ValueError, fft, [[1,1],[2,2]], -5)


class TestLongDoubleFFT(_TestFFTBase):
    def setup_method(self):
        self.cdt = np.longcomplex
        self.rdt = np.longdouble


class TestDoubleFFT(_TestFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double


class TestSingleFFT(_TestFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32


class TestFloat16FFT:

    def test_1_argument_real(self):
        x1 = np.array([1, 2, 3, 4], dtype=np.float16)
        y = fft(x1, n=4)
        assert_equal(y.dtype, np.complex64)
        assert_equal(y.shape, (4, ))
        assert_array_almost_equal(y, direct_dft(x1.astype(np.float32)))

    def test_n_argument_real(self):
        x1 = np.array([1, 2, 3, 4], dtype=np.float16)
        x2 = np.array([1, 2, 3, 4], dtype=np.float16)
        y = fft([x1, x2], n=4)
        assert_equal(y.dtype, np.complex64)
        assert_equal(y.shape, (2, 4))
        assert_array_almost_equal(y[0], direct_dft(x1.astype(np.float32)))
        assert_array_almost_equal(y[1], direct_dft(x2.astype(np.float32)))


class _TestIFFTBase:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = np.array([1,2,3,4+1j,1,2,3,4+2j], self.cdt)
        y = ifft(x)
        y1 = direct_idft(x)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(y,y1)

        x = np.array([1,2,3,4+0j,5], self.cdt)
        assert_array_almost_equal(ifft(x),direct_idft(x))

    def test_definition_real(self):
        x = np.array([1,2,3,4,1,2,3,4], self.rdt)
        y = ifft(x)
        assert_equal(y.dtype, self.cdt)
        y1 = direct_idft(x)
        assert_array_almost_equal(y,y1)

        x = np.array([1,2,3,4,5], dtype=self.rdt)
        assert_equal(y.dtype, self.cdt)
        assert_array_almost_equal(ifft(x),direct_idft(x))

    def test_djbfft(self):
        for i in range(2,14):
            n = 2**i
            x = np.arange(n)
            y = ifft(x.astype(self.cdt))
            y2 = numpy.fft.ifft(x)
            assert_allclose(y,y2, rtol=self.rtol, atol=self.atol)
            y = ifft(x)
            assert_allclose(y,y2, rtol=self.rtol, atol=self.atol)

    def test_random_complex(self):
        for size in [1,51,111,100,200,64,128,256,1024]:
            x = random([size]).astype(self.cdt)
            x = random([size]).astype(self.cdt) + 1j*x
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_random_real(self):
        for size in [1,51,111,100,200,64,128,256,1024]:
            x = random([size]).astype(self.rdt)
            y1 = ifft(fft(x))
            y2 = fft(ifft(x))
            assert_equal(y1.dtype, self.cdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x)
            assert_array_almost_equal(y2, x)

    def test_size_accuracy(self):
        # Sanity check for the accuracy for prime and non-prime sized inputs
        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            np.random.seed(1234)
            x = np.random.rand(size).astype(self.rdt)
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

            x = (x + 1j*np.random.rand(size)).astype(self.cdt)
            y = ifft(fft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)
            y = fft(ifft(x))
            _assert_close_in_norm(x, y, self.rtol, size, self.rdt)

    def test_invalid_sizes(self):
        assert_raises(ValueError, ifft, [])
        assert_raises(ValueError, ifft, [[1,1],[2,2]], -5)


@pytest.mark.skipif(np.longdouble is np.float64,
                    reason="Long double is aliased to double")
class TestLongDoubleIFFT(_TestIFFTBase):
    def setup_method(self):
        self.cdt = np.longcomplex
        self.rdt = np.longdouble
        self.rtol = 1e-10
        self.atol = 1e-10


class TestDoubleIFFT(_TestIFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double
        self.rtol = 1e-10
        self.atol = 1e-10


class TestSingleIFFT(_TestIFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32
        self.rtol = 1e-5
        self.atol = 1e-4


class _TestRFFTBase:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        for t in [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4, 5]]:
            x = np.array(t, dtype=self.rdt)
            y = rfft(x)
            y1 = direct_rdft(x)
            assert_array_almost_equal(y,y1)
            assert_equal(y.dtype, self.cdt)

    def test_djbfft(self):
        for i in range(2,14):
            n = 2**i
            x = np.arange(n)
            y1 = np.fft.rfft(x)
            y = rfft(x)
            assert_array_almost_equal(y,y1)

    def test_invalid_sizes(self):
        assert_raises(ValueError, rfft, [])
        assert_raises(ValueError, rfft, [[1,1],[2,2]], -5)

    def test_complex_input(self):
        x = np.zeros(10, dtype=self.cdt)
        with assert_raises(TypeError, match="x must be a real sequence"):
            rfft(x)

    # See gh-5790
    class MockSeries:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getattr__(self, item):
            try:
                return getattr(self.data, item)
            except AttributeError as e:
                raise AttributeError(("'MockSeries' object "
                                      "has no attribute '{attr}'".
                                      format(attr=item))) from e

    def test_non_ndarray_with_dtype(self):
        x = np.array([1., 2., 3., 4., 5.])
        xs = _TestRFFTBase.MockSeries(x)

        expected = [1, 2, 3, 4, 5]
        rfft(xs)

        # Data should not have been overwritten
        assert_equal(x, expected)
        assert_equal(xs.data, expected)

@pytest.mark.skipif(np.longfloat is np.float64,
                    reason="Long double is aliased to double")
class TestRFFTLongDouble(_TestRFFTBase):
    def setup_method(self):
        self.cdt = np.longcomplex
        self.rdt = np.longfloat


class TestRFFTDouble(_TestRFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double


class TestRFFTSingle(_TestRFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32


class _TestIRFFTBase:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x1 = [1,2+3j,4+1j,1+2j,3+4j]
        x1_1 = [1,2+3j,4+1j,2+3j,4,2-3j,4-1j,2-3j]
        x1 = x1_1[:5]
        x2_1 = [1,2+3j,4+1j,2+3j,4+5j,4-5j,2-3j,4-1j,2-3j]
        x2 = x2_1[:5]

        def _test(x, xr):
            y = irfft(np.array(x, dtype=self.cdt), n=len(xr))
            y1 = direct_irdft(x, len(xr))
            assert_equal(y.dtype, self.rdt)
            assert_array_almost_equal(y,y1, decimal=self.ndec)
            assert_array_almost_equal(y,ifft(xr), decimal=self.ndec)

        _test(x1, x1_1)
        _test(x2, x2_1)

    def test_djbfft(self):
        for i in range(2,14):
            n = 2**i
            x = np.arange(-1, n, 2) + 1j * np.arange(0, n+1, 2)
            x[0] = 0
            if n % 2 == 0:
                x[-1] = np.real(x[-1])
            y1 = np.fft.irfft(x)
            y = irfft(x)
            assert_array_almost_equal(y,y1)

    def test_random_real(self):
        for size in [1,51,111,100,200,64,128,256,1024]:
            x = random([size]).astype(self.rdt)
            y1 = irfft(rfft(x), n=size)
            y2 = rfft(irfft(x, n=(size*2-1)))
            assert_equal(y1.dtype, self.rdt)
            assert_equal(y2.dtype, self.cdt)
            assert_array_almost_equal(y1, x, decimal=self.ndec,
                                       err_msg="size=%d" % size)
            assert_array_almost_equal(y2, x, decimal=self.ndec,
                                       err_msg="size=%d" % size)

    def test_size_accuracy(self):
        # Sanity check for the accuracy for prime and non-prime sized inputs
        if self.rdt == np.float32:
            rtol = 1e-5
        elif self.rdt == np.float64:
            rtol = 1e-10

        for size in LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES:
            np.random.seed(1234)
            x = np.random.rand(size).astype(self.rdt)
            y = irfft(rfft(x), len(x))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)
            y = rfft(irfft(x, 2 * len(x) - 1))
            _assert_close_in_norm(x, y, rtol, size, self.rdt)

    def test_invalid_sizes(self):
        assert_raises(ValueError, irfft, [])
        assert_raises(ValueError, irfft, [[1,1],[2,2]], -5)


# self.ndec is bogus; we should have a assert_array_approx_equal for number of
# significant digits

@pytest.mark.skipif(np.longfloat is np.float64,
                    reason="Long double is aliased to double")
class TestIRFFTLongDouble(_TestIRFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double
        self.ndec = 14


class TestIRFFTDouble(_TestIRFFTBase):
    def setup_method(self):
        self.cdt = np.cdouble
        self.rdt = np.double
        self.ndec = 14


class TestIRFFTSingle(_TestIRFFTBase):
    def setup_method(self):
        self.cdt = np.complex64
        self.rdt = np.float32
        self.ndec = 5


class Testfft2:
    def setup_method(self):
        np.random.seed(1234)

    def test_regression_244(self):
        """FFT returns wrong result with axes parameter."""
        # fftn (and hence fft2) used to break when both axes and shape were
        # used
        x = numpy.ones((4, 4, 2))
        y = fft2(x, s=(8, 8), axes=(-3, -2))
        y_r = numpy.fft.fftn(x, s=(8, 8), axes=(-3, -2))
        assert_array_almost_equal(y, y_r)

    def test_invalid_sizes(self):
        assert_raises(ValueError, fft2, [[]])
        assert_raises(ValueError, fft2, [[1, 1], [2, 2]], (4, -3))


class TestFftnSingle:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(np.array(x, np.float32))
        assert_(y.dtype == np.complex64,
                msg="double precision output with single precision")

        y_r = np.array(fftn(x), np.complex64)
        assert_array_almost_equal_nulp(y, y_r)

    @pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_size_accuracy_small(self, size):
        x = np.random.rand(size, size) + 1j*np.random.rand(size, size)
        y1 = fftn(x.real.astype(np.float32))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000)

    @pytest.mark.parametrize('size', LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    def test_size_accuracy_large(self, size):
        x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)
        y1 = fftn(x.real.astype(np.float32))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2000)

    def test_definition_float16(self):
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(np.array(x, np.float16))
        assert_equal(y.dtype, np.complex64)
        y_r = np.array(fftn(x), np.complex64)
        assert_array_almost_equal_nulp(y, y_r)

    @pytest.mark.parametrize('size', SMALL_COMPOSITE_SIZES + SMALL_PRIME_SIZES)
    def test_float16_input_small(self, size):
        x = np.random.rand(size, size) + 1j*np.random.rand(size, size)
        y1 = fftn(x.real.astype(np.float16))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 5e5)

    @pytest.mark.parametrize('size', LARGE_COMPOSITE_SIZES + LARGE_PRIME_SIZES)
    def test_float16_input_large(self, size):
        x = np.random.rand(size, 3) + 1j*np.random.rand(size, 3)
        y1 = fftn(x.real.astype(np.float16))
        y2 = fftn(x.real.astype(np.float64)).astype(np.complex64)

        assert_equal(y1.dtype, np.complex64)
        assert_array_almost_equal_nulp(y1, y2, 2e6)


class TestFftn:
    def setup_method(self):
        np.random.seed(1234)

    def test_definition(self):
        x = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        y = fftn(x)
        assert_array_almost_equal(y, direct_dftn(x))

        x = random((20, 26))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

        x = random((5, 4, 3, 20))
        assert_array_almost_equal(fftn(x), direct_dftn(x))

    def test_axes_argument(self):
        # plane == ji_plane, x== kji_space
        plane1 = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
        plane2 = [[10, 11, 12],
                  [13, 14, 15],
                  [16, 17, 18]]
        plane3 = [[19, 20, 21],
                  [22, 23, 24],
                  [25, 26, 27]]
        ki_plane1 = [[1, 2, 3],
                     [10, 11, 12],
                     [19, 20, 21]]
        ki_plane2 = [[4, 5, 6],
                     [13, 14, 15],
                     [22, 23, 24]]
        ki_plane3 = [[7, 8, 9],
                     [16, 17, 18],
                     [25, 26, 27]]
        jk_plane1 = [[1, 10, 19],
                     [4, 13, 22],
                     [7, 16, 25]]
        jk_plane2 = [[2, 11, 20],
                     [5, 14, 23],
                     [8, 17, 26]]
        jk_plane3 = [[3, 12, 21],
                     [6, 15, 24],
                     [9, 18, 27]]
        kj_plane1 = [[1, 4, 7],
                     [10, 13, 16], [19, 22, 25]]
        kj_plane2 = [[2, 5, 8],
                     [11, 14, 17], [20, 23, 26]]
        kj_plane3 = [[3, 6, 9],
                     [12, 15, 18], [21, 24, 27]]
        ij_plane1 = [[1, 4, 7],
                     [2, 5, 8],
                     [3, 6, 9]]
        ij_plane2 = [[10, 13, 16],
                     [11, 14, 17],
                     [12, 15, 18]]
        ij_plane3 = [[19, 22, 25],
                     [20, 23, 26],
                     [21, 24, 27]]
        ik_plane1 = [[1, 10, 19],
                     [2, 11, 20],
                     [3, 12, 21]]
        ik_plane2 = [[4, 13, 22],
                     [5, 14, 23],
                     [6, 15, 24]]
        ik_plane3 = [[7, 16, 25],
                     [8, 17, 26],
                     [9, 18, 27]]
        ijk_space = [jk_plane1, jk_plane2, jk_plane3]
        ikj_space = [kj_plane1, kj_plane2, kj_plane3]
        jik_space = [ik_plane1, ik_plane2, ik_plane3]
        jki_space = [ki_plane1, ki_plane2, ki_plane3]
        kij_space = [ij_plane1, ij_plane2, ij_plane3]
        x = array([plane1, plane2, plane3])

        assert_array_almost_equal(fftn(x),
                                  fftn(x, axes=(-3, -2, -1)))  # kji_space
        assert_array_almost_equal(fftn(x), fftn(x, axes=(0, 1, 2)))
        assert_array_almost_equal(fftn(x, axes=(0, 2)), fftn(x, axes=(0, -1)))
        y = fftn(x, axes=(2, 1, 0))  # ijk_space
        assert_array_almost_equal(swapaxes(y, -1, -3), fftn(ijk_space))
        y = fftn(x, axes=(2, 0, 1))  # ikj_space
        assert_array_almost_equal(swapaxes(swapaxes(y, -1, -3), -1, -2),
                                  fftn(ikj_space))
        y = fftn(x, axes=(1, 2, 0))  # jik_space
        assert_array_almost_equal(swapaxes(swapaxes(y, -1, -3), -3, -2),
                                  fftn(jik_space))
        y = fftn(x, axes=(1, 0, 2))  # jki_space
        assert_array_almost_equal(swapaxes(y, -2, -3), fftn(jki_space))
        y = fftn(x, axes=(0, 2, 1))  # kij_space
        assert_array_almost_equal(swapaxes(y, -2, -1), fftn(kij_space))

        y = fftn(x, axes=(-2, -1))  # ji_plane
        assert_array_almost_equal(fftn(plane1), y[0])
        assert_array_almost_equal(fftn(plane2), y[1])
        assert_array_almost_equal(fftn(plane3), y[2])

        y = fftn(x, axes=(1, 2))  # ji_plane
        assert_array_almost_equal(fftn(plane1), y[0])
        assert_array_almost_equal(fftn(plane2), y[1])
        assert_array_almost_equal(fftn(plane3), y[2])

        y = fftn(x, axes=(-3, -2))  # kj_plane
        assert_array_almost_equal(fftn(x[:, :, 0]), y[:, :, 0])
        assert_array_almost_equal(fftn(x[:, :, 1]), y[:, :, 1])
        assert_array_almost_equal(fftn(x[:, :, 2]), y[:, :, 2])

        y = fftn(x, axes=(-3, -1))  # ki_plane
        assert_array_almost_equal(fftn(x[:, 0, :]), y[:, 0, :])
        assert_array_almost_equal(fftn(x[:, 1, :]), y[:, 1, :])
        assert_array_almost_equal(fftn(x[:, 2, :]), y[:, 2, :])

        y = fftn(x, axes=(-1, -2))  # ij_plane
        assert_array_almost_equal(fftn(ij_plane1), swapaxes(y[0], -2, -1))
        assert_array_almost_equal(fftn(ij_plane2), swapaxes(y[1], -2, -1))
        assert_array_almost_equal(fftn(ij_plane3), swapaxes(y[2], -2, -1))

        y = fftn(x, axes=(-1, -3))  # ik_plane
        assert_array_almost_equal(fftn(ik_plane1),
                                  swapaxes(y[:, 0, :], -1, -2))
        assert_array_almost_equal(fftn(ik_plane2),
                                  swapaxes(y[:, 1, :], -1, -2))
        assert_array_almost_equal(fftn(ik_plane3),
                                  swapaxes(y[:, 2, :], -1, -2))

        y = fftn(x, axes=(-2, -3))  # jk_plane
        assert_array_almost_equal(fftn(jk_plane1),
                                  swapaxes(y[:, :, 0], -1, -2))
        assert_array_almost_equal(fftn(jk_plane2),
                                  swapaxes(y[:, :, 1], -1, -2))
        assert_array_almost_equal(fftn(jk_plane3),
                                  swapaxes(y[:, :, 2], -1, -2))

        y = fftn(x, axes=(-1,))  # i_line
        for i in range(3):
            for j in range(3):
                assert_array_almost_equal(fft(x[i, j, :]), y[i, j, :])
        y = fftn(x, axes=(-2,))  # j_line
        for i in range(3):
            for j in range(3):
                assert_array_almost_equal(fft(x[i, :, j]), y[i, :, j])
        y = fftn(x, axes=(0,))  # k_line
        for i in range(3):
            for j in range(3):
                assert_array_almost_equal(fft(x[:, i, j]), y[:, i, j])

        y = fftn(x, axes=())  # point
        assert_array_almost_equal(y, x)

    def test_shape_argument(self):
        small_x = [[1, 2, 3],
                   [4, 5, 6]]
        large_x1 = [[1, 2, 3, 0],
                    [4, 5, 6, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]

        y = fftn(small_x, s=(4, 4))
        assert_array_almost_equal(y, fftn(large_x1))

        y = fftn(small_x, s=(3, 4))
        assert_array_almost_equal(y, fftn(large_x1[:-1]))

    def test_shape_axes_argument(self):
        small_x = [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]
        large_x1 = array([[1, 2, 3, 0],
                          [4, 5, 6, 0],
                          [7, 8, 9, 0],
                          [0, 0, 0, 0]])
        y = fftn(small_x, s=(4, 4), axes=(-2, -1))
        assert_array_almost_equal(y, fftn(large_x1))
        y = fftn(small_x, s=(4, 4), axes=(-1, -2))

        assert_array_almost_equal(y, swapaxes(
            fftn(swapaxes(large_x1, -1, -2)), -1, -2))

    def test_shape_axes_argument2(self):
        # Change shape of the last axis
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-1,), s=(8,))
        assert_array_almost_equal(y, fft(x, axis=-1, n=8))

        # Change shape of an arbitrary axis which is not the last one
        x = numpy.random.random((10, 5, 3, 7))
        y = fftn(x, axes=(-2,), s=(8,))
        assert_array_almost_equal(y, fft(x, axis=-2, n=8))

        # Change shape of axes: cf #244, where shape and axes were mixed up
        x = numpy.random.random((4, 4, 2))
        y = fftn(x, axes=(-3, -2), s=(8, 8))
        assert_array_almost_equal(y,
                                  numpy.fft.fftn(x, axes=(-3, -2), s=(8, 8)))

    def test_shape_argument_more(self):
        x = zeros((4, 4, 2))
        with assert_raises(ValueError,
                           match="shape requires more axes than are present"):
            fftn(x, s=(8, 8, 2, 1))

    def test_invalid_sizes(self):
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            fftn([[]])

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            fftn([[1, 1], [2, 2]], (4, -3))

    def test_no_axes(self):
        x = numpy.random.random((2,2,2))
        assert_allclose(fftn(x, axes=[]), x, atol=1e-7)


class TestIfftn:
    dtype = None
    cdtype = None

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize('dtype,cdtype,maxnlp',
                             [(np.float64, np.complex128, 2000),
                              (np.float32, np.complex64, 3500)])
    def test_definition(self, dtype, cdtype, maxnlp):
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=dtype)
        y = ifftn(x)
        assert_equal(y.dtype, cdtype)
        assert_array_almost_equal_nulp(y, direct_idftn(x), maxnlp)

        x = random((20, 26))
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

        x = random((5, 4, 3, 20))
        assert_array_almost_equal_nulp(ifftn(x), direct_idftn(x), maxnlp)

    @pytest.mark.parametrize('maxnlp', [2000, 3500])
    @pytest.mark.parametrize('size', [1, 2, 51, 32, 64, 92])
    def test_random_complex(self, maxnlp, size):
        x = random([size, size]) + 1j*random([size, size])
        assert_array_almost_equal_nulp(ifftn(fftn(x)), x, maxnlp)
        assert_array_almost_equal_nulp(fftn(ifftn(x)), x, maxnlp)

    def test_invalid_sizes(self):
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            ifftn([[]])

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            ifftn([[1, 1], [2, 2]], (4, -3))

    def test_no_axes(self):
        x = numpy.random.random((2,2,2))
        assert_allclose(ifftn(x, axes=[]), x, atol=1e-7)

class TestRfftn:
    dtype = None
    cdtype = None

    def setup_method(self):
        np.random.seed(1234)

    @pytest.mark.parametrize('dtype,cdtype,maxnlp',
                             [(np.float64, np.complex128, 2000),
                              (np.float32, np.complex64, 3500)])
    def test_definition(self, dtype, cdtype, maxnlp):
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=dtype)
        y = rfftn(x)
        assert_equal(y.dtype, cdtype)
        assert_array_almost_equal_nulp(y, direct_rdftn(x), maxnlp)

        x = random((20, 26))
        assert_array_almost_equal_nulp(rfftn(x), direct_rdftn(x), maxnlp)

        x = random((5, 4, 3, 20))
        assert_array_almost_equal_nulp(rfftn(x), direct_rdftn(x), maxnlp)

    @pytest.mark.parametrize('size', [1, 2, 51, 32, 64, 92])
    def test_random(self, size):
        x = random([size, size])
        assert_allclose(irfftn(rfftn(x), x.shape), x, atol=1e-10)

    @pytest.mark.parametrize('func', [rfftn, irfftn])
    def test_invalid_sizes(self, func):
        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[1, 0\]\) specified"):
            func([[]])

        with assert_raises(ValueError,
                           match="invalid number of data points"
                           r" \(\[4, -3\]\) specified"):
            func([[1, 1], [2, 2]], (4, -3))

    @pytest.mark.parametrize('func', [rfftn, irfftn])
    def test_no_axes(self, func):
        with assert_raises(ValueError,
                           match="at least 1 axis must be transformed"):
            func([], axes=[])

    def test_complex_input(self):
        with assert_raises(TypeError, match="x must be a real sequence"):
            rfftn(np.zeros(10, dtype=np.complex64))


class FakeArray:
    def __init__(self, data):
        self._data = data
        self.__array_interface__ = data.__array_interface__


class FakeArray2:
    def __init__(self, data):
        self._data = data

    def __array__(self):
        return self._data

# TODO: Is this test actually valuable? The behavior it's testing shouldn't be
# relied upon by users except for overwrite_x = False
class TestOverwrite:
    """Check input overwrite behavior of the FFT functions."""

    real_dtypes = [np.float32, np.float64, np.longfloat]
    dtypes = real_dtypes + [np.complex64, np.complex128, np.longcomplex]
    fftsizes = [8, 16, 32]

    def _check(self, x, routine, fftsize, axis, overwrite_x, should_overwrite):
        x2 = x.copy()
        for fake in [lambda x: x, FakeArray, FakeArray2]:
            routine(fake(x2), fftsize, axis, overwrite_x=overwrite_x)

            sig = "%s(%s%r, %r, axis=%r, overwrite_x=%r)" % (
                routine.__name__, x.dtype, x.shape, fftsize, axis, overwrite_x)
            if not should_overwrite:
                assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)

    def _check_1d(self, routine, dtype, shape, axis, overwritable_dtypes,
                  fftsize, overwrite_x):
        np.random.seed(1234)
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            data = np.random.randn(*shape)
        data = data.astype(dtype)

        should_overwrite = (overwrite_x
                            and dtype in overwritable_dtypes
                            and fftsize <= shape[axis])
        self._check(data, routine, fftsize, axis,
                    overwrite_x=overwrite_x,
                    should_overwrite=should_overwrite)

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('fftsize', fftsizes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), -1),
                                            ((16, 2), 0),
                                            ((2, 16), 1)])
    def test_fft_ifft(self, dtype, fftsize, overwrite_x, shape, axes):
        overwritable = (np.longcomplex, np.complex128, np.complex64)
        self._check_1d(fft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
        self._check_1d(ifft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)

    @pytest.mark.parametrize('dtype', real_dtypes)
    @pytest.mark.parametrize('fftsize', fftsizes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), -1),
                                            ((16, 2), 0),
                                            ((2, 16), 1)])
    def test_rfft_irfft(self, dtype, fftsize, overwrite_x, shape, axes):
        overwritable = self.real_dtypes
        self._check_1d(irfft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)
        self._check_1d(rfft, dtype, shape, axes, overwritable,
                       fftsize, overwrite_x)

    def _check_nd_one(self, routine, dtype, shape, axes, overwritable_dtypes,
                      overwrite_x):
        np.random.seed(1234)
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            data = np.random.randn(*shape)
        data = data.astype(dtype)

        def fftshape_iter(shp):
            if len(shp) <= 0:
                yield ()
            else:
                for j in (shp[0]//2, shp[0], shp[0]*2):
                    for rest in fftshape_iter(shp[1:]):
                        yield (j,) + rest

        def part_shape(shape, axes):
            if axes is None:
                return shape
            else:
                return tuple(np.take(shape, axes))

        def should_overwrite(data, shape, axes):
            s = part_shape(data.shape, axes)
            return (overwrite_x and
                    np.prod(shape) <= np.prod(s)
                    and dtype in overwritable_dtypes)

        for fftshape in fftshape_iter(part_shape(shape, axes)):
            self._check(data, routine, fftshape, axes,
                        overwrite_x=overwrite_x,
                        should_overwrite=should_overwrite(data, fftshape, axes))
            if data.ndim > 1:
                # check fortran order
                self._check(data.T, routine, fftshape, axes,
                            overwrite_x=overwrite_x,
                            should_overwrite=should_overwrite(
                                data.T, fftshape, axes))

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('overwrite_x', [True, False])
    @pytest.mark.parametrize('shape,axes', [((16,), None),
                                            ((16,), (0,)),
                                            ((16, 2), (0,)),
                                            ((2, 16), (1,)),
                                            ((8, 16), None),
                                            ((8, 16), (0, 1)),
                                            ((8, 16, 2), (0, 1)),
                                            ((8, 16, 2), (1, 2)),
                                            ((8, 16, 2), (0,)),
                                            ((8, 16, 2), (1,)),
                                            ((8, 16, 2), (2,)),
                                            ((8, 16, 2), None),
                                            ((8, 16, 2), (0, 1, 2))])
    def test_fftn_ifftn(self, dtype, overwrite_x, shape, axes):
        overwritable = (np.longcomplex, np.complex128, np.complex64)
        self._check_nd_one(fftn, dtype, shape, axes, overwritable,
                           overwrite_x)
        self._check_nd_one(ifftn, dtype, shape, axes, overwritable,
                           overwrite_x)


@pytest.mark.parametrize('func', [fft, ifft, fftn, ifftn,
                                 rfft, irfft, rfftn, irfftn])
def test_invalid_norm(func):
    x = np.arange(10, dtype=float)
    with assert_raises(ValueError,
                       match='Invalid norm value \'o\', should be'
                             ' "backward", "ortho" or "forward"'):
        func(x, norm='o')


@pytest.mark.parametrize('func', [fft, ifft, fftn, ifftn,
                                   irfft, irfftn, hfft, hfftn])
def test_swapped_byte_order_complex(func):
    rng = np.random.RandomState(1234)
    x = rng.rand(10) + 1j * rng.rand(10)
    assert_allclose(func(swap_byteorder(x)), func(x))


@pytest.mark.parametrize('func', [ihfft, ihfftn, rfft, rfftn])
def test_swapped_byte_order_real(func):
    rng = np.random.RandomState(1234)
    x = rng.rand(10)
    assert_allclose(func(swap_byteorder(x)), func(x))
