''' Some tests for filters '''
import functools
import math
import numpy

from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_almost_equal,
                           assert_array_equal, assert_almost_equal,
                           suppress_warnings, assert_)
import pytest
from pytest import raises as assert_raises

from scipy import ndimage
from scipy.ndimage.filters import _gaussian_kernel1d, rank_filter

from . import types, float_types, complex_types


def sumsq(a, b):
    return math.sqrt(((a - b)**2).sum())


def _complex_correlate(array, kernel, real_dtype, convolve=False,
                       mode="reflect", cval=0, ):
    """Utility to perform a reference complex-valued convolutions.

    When convolve==False, correlation is performed instead
    """
    array = numpy.asarray(array)
    kernel = numpy.asarray(kernel)
    complex_array = array.dtype.kind == 'c'
    complex_kernel = kernel.dtype.kind == 'c'
    if array.ndim == 1:
        func = ndimage.convolve1d if convolve else ndimage.correlate1d
    else:
        func = ndimage.convolve if convolve else ndimage.correlate
    if not convolve:
        kernel = kernel.conj()
    if complex_array and complex_kernel:
        # use: real(cval) for array.real component
        #      imag(cval) for array.imag component
        output = (
            func(array.real, kernel.real, output=real_dtype,
                 mode=mode, cval=numpy.real(cval)) -
            func(array.imag, kernel.imag, output=real_dtype,
                 mode=mode, cval=numpy.imag(cval)) +
            1j * func(array.imag, kernel.real, output=real_dtype,
                      mode=mode, cval=numpy.imag(cval)) +
            1j * func(array.real, kernel.imag, output=real_dtype,
                      mode=mode, cval=numpy.real(cval))
        )
    elif complex_array:
        output = (
            func(array.real, kernel, output=real_dtype, mode=mode,
                 cval=numpy.real(cval)) +
            1j * func(array.imag, kernel, output=real_dtype, mode=mode,
                      cval=numpy.imag(cval))
        )
    elif complex_kernel:
        # real array so cval is real too
        output = (
            func(array, kernel.real, output=real_dtype, mode=mode, cval=cval) +
            1j * func(array, kernel.imag, output=real_dtype, mode=mode,
                      cval=cval)
        )
    return output


class TestNdimageFilters:

    def _validate_complex(self, array, kernel, type2, mode='reflect', cval=0):
        # utility for validating complex-valued correlations
        real_dtype = numpy.asarray([], dtype=type2).real.dtype
        expected = _complex_correlate(
            array, kernel, real_dtype, convolve=False, mode=mode, cval=cval
        )

        if array.ndim == 1:
            correlate = functools.partial(ndimage.correlate1d, axis=-1,
                                          mode=mode, cval=cval)
            convolve = functools.partial(ndimage.convolve1d, axis=-1,
                                         mode=mode, cval=cval)
        else:
            correlate = functools.partial(ndimage.correlate, mode=mode,
                                          cval=cval)
            convolve = functools.partial(ndimage.convolve, mode=mode,
                                          cval=cval)

        # test correlate output dtype
        output = correlate(array, kernel, output=type2)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)

        # test correlate with pre-allocated output
        output = numpy.zeros_like(array, dtype=type2)
        correlate(array, kernel, output=output)
        assert_array_almost_equal(expected, output)

        # test convolve output dtype
        output = convolve(array, kernel, output=type2)
        expected = _complex_correlate(
            array, kernel, real_dtype, convolve=True, mode=mode, cval=cval,
        )
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)

        # convolve with pre-allocated output
        convolve(array, kernel, output=output)
        assert_array_almost_equal(expected, output)
        assert_equal(output.dtype.type, type2)

        # warns if the output is not a complex dtype
        with pytest.warns(UserWarning,
                          match="promoting specified output dtype to complex"):
            correlate(array, kernel, output=real_dtype)

        with pytest.warns(UserWarning,
                          match="promoting specified output dtype to complex"):
            convolve(array, kernel, output=real_dtype)

        # raises if output array is provided, but is not complex-valued
        output_real = numpy.zeros_like(array, dtype=real_dtype)
        with assert_raises(RuntimeError):
            correlate(array, kernel, output=output_real)

        with assert_raises(RuntimeError):
            convolve(array, kernel, output=output_real)

    def test_correlate01(self):
        array = numpy.array([1, 2])
        weights = numpy.array([2])
        expected = [2, 4]

        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate01_overlap(self):
        array = numpy.arange(256).reshape(16, 16)
        weights = numpy.array([2])
        expected = 2 * array

        ndimage.correlate1d(array, weights, output=array)
        assert_array_almost_equal(array, expected)

    def test_correlate02(self):
        array = numpy.array([1, 2, 3])
        kernel = numpy.array([1])

        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate03(self):
        array = numpy.array([1])
        weights = numpy.array([1, 1])
        expected = [2]

        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate04(self):
        array = numpy.array([1, 2])
        tcor = [2, 3]
        tcov = [3, 4]
        weights = numpy.array([1, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate05(self):
        array = numpy.array([1, 2, 3])
        tcor = [2, 3, 5]
        tcov = [3, 5, 6]
        kernel = numpy.array([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(tcov, output)

    def test_correlate06(self):
        array = numpy.array([1, 2, 3])
        tcor = [9, 14, 17]
        tcov = [7, 10, 15]
        weights = numpy.array([1, 2, 3])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate07(self):
        array = numpy.array([1, 2, 3])
        expected = [5, 8, 11]
        weights = numpy.array([1, 2, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate08(self):
        array = numpy.array([1, 2, 3])
        tcor = [1, 2, 5]
        tcov = [3, 6, 7]
        weights = numpy.array([1, 2, -1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate09(self):
        array = []
        kernel = numpy.array([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate10(self):
        array = [[]]
        kernel = numpy.array([[1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate11(self):
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]])
        kernel = numpy.array([[1, 1],
                              [1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal([[12, 16, 18], [18, 22, 24]], output)

    def test_correlate12(self):
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]])
        kernel = numpy.array([[1, 0],
                              [0, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_kernel', types)
    def test_correlate13(self, dtype_array, dtype_kernel):
        kernel = numpy.array([[1, 0],
                              [0, 1]])
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=dtype_kernel)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, dtype_kernel)

        output = ndimage.convolve(array, kernel,
                                  output=dtype_kernel)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, dtype_kernel)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate14(self, dtype_array, dtype_output):
        kernel = numpy.array([[1, 0],
                              [0, 1]])
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_array)
        output = numpy.zeros(array.shape, dtype_output)
        ndimage.correlate(array, kernel, output=output)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, dtype_output)

        ndimage.convolve(array, kernel, output=output)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, dtype_output)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate15(self, dtype_array):
        kernel = numpy.array([[1, 0],
                              [0, 1]])
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, numpy.float32)

        output = ndimage.convolve(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, numpy.float32)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate16(self, dtype_array):
        kernel = numpy.array([[0.5, 0],
                              [0, 0.5]])
        array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[1, 1.5, 2.5], [2.5, 3, 4]], output)
        assert_equal(output.dtype.type, numpy.float32)

        output = ndimage.convolve(array, kernel, output=numpy.float32)
        assert_array_almost_equal([[3, 4, 4.5], [4.5, 5.5, 6]], output)
        assert_equal(output.dtype.type, numpy.float32)

    def test_correlate17(self):
        array = numpy.array([1, 2, 3])
        tcor = [3, 5, 6]
        tcov = [2, 3, 5]
        kernel = numpy.array([1, 1])
        output = ndimage.correlate(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate18(self, dtype_array):
        kernel = numpy.array([[1, 0],
                              [0, 1]])
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel,
                                   output=numpy.float32,
                                   mode='nearest', origin=-1)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
        assert_equal(output.dtype.type, numpy.float32)

        output = ndimage.convolve(array, kernel,
                                  output=numpy.float32,
                                  mode='nearest', origin=-1)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        assert_equal(output.dtype.type, numpy.float32)

    def test_correlate_mode_sequence(self):
        kernel = numpy.ones((2, 2))
        array = numpy.ones((3, 3), float)
        with assert_raises(RuntimeError):
            ndimage.correlate(array, kernel, mode=['nearest', 'reflect'])
        with assert_raises(RuntimeError):
            ndimage.convolve(array, kernel, mode=['nearest', 'reflect'])

    @pytest.mark.parametrize('dtype_array', types)
    def test_correlate19(self, dtype_array):
        kernel = numpy.array([[1, 0],
                              [0, 1]])
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_array)
        output = ndimage.correlate(array, kernel,
                                   output=numpy.float32,
                                   mode='nearest', origin=[-1, 0])
        assert_array_almost_equal([[5, 6, 8], [8, 9, 11]], output)
        assert_equal(output.dtype.type, numpy.float32)

        output = ndimage.convolve(array, kernel,
                                  output=numpy.float32,
                                  mode='nearest', origin=[-1, 0])
        assert_array_almost_equal([[3, 5, 6], [6, 8, 9]], output)
        assert_equal(output.dtype.type, numpy.float32)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate20(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        expected = [[5, 10, 15], [7, 14, 21]]
        array = numpy.array([[1, 2, 3],
                             [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0, output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0, output=output)
        assert_array_almost_equal(output, expected)

    def test_correlate21(self):
        array = numpy.array([[1, 2, 3],
                             [2, 4, 6]])
        expected = [[5, 10, 15], [7, 14, 21]]
        weights = numpy.array([1, 2, 1])
        output = ndimage.correlate1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate22(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        expected = [[6, 12, 18], [6, 12, 18]]
        array = numpy.array([[1, 2, 3],
                             [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='wrap', output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='wrap', output=output)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate23(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        expected = [[5, 10, 15], [7, 14, 21]]
        array = numpy.array([[1, 2, 3],
                             [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output)
        assert_array_almost_equal(output, expected)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output)
        assert_array_almost_equal(output, expected)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate24(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        tcor = [[7, 14, 21], [8, 16, 24]]
        tcov = [[4, 8, 12], [5, 10, 15]]
        array = numpy.array([[1, 2, 3],
                             [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output, origin=-1)
        assert_array_almost_equal(output, tcor)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output, origin=-1)
        assert_array_almost_equal(output, tcov)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_correlate25(self, dtype_array, dtype_output):
        weights = numpy.array([1, 2, 1])
        tcor = [[4, 8, 12], [5, 10, 15]]
        tcov = [[7, 14, 21], [8, 16, 24]]
        array = numpy.array([[1, 2, 3],
                             [2, 4, 6]], dtype_array)
        output = numpy.zeros((2, 3), dtype_output)
        ndimage.correlate1d(array, weights, axis=0,
                            mode='nearest', output=output, origin=1)
        assert_array_almost_equal(output, tcor)
        ndimage.convolve1d(array, weights, axis=0,
                           mode='nearest', output=output, origin=1)
        assert_array_almost_equal(output, tcov)

    def test_correlate26(self):
        # test fix for gh-11661 (mirror extension of a length 1 signal)
        y = ndimage.convolve1d(numpy.ones(1), numpy.ones(5), mode='mirror')
        assert_array_equal(y, numpy.array(5.))

        y = ndimage.correlate1d(numpy.ones(1), numpy.ones(5), mode='mirror')
        assert_array_equal(y, numpy.array(5.))

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_kernel(self, dtype_input, dtype_kernel,
                                      dtype_output):
        kernel = numpy.array([[1, 0],
                              [0, 1 + 1j]], dtype_kernel)
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    @pytest.mark.parametrize('mode', ['grid-constant', 'constant'])
    def test_correlate_complex_kernel_cval(self, dtype_input, dtype_kernel,
                                           dtype_output, mode):
        # test use of non-zero cval with complex inputs
        # also verifies that mode 'grid-constant' does not segfault
        kernel = numpy.array([[1, 0],
                              [0, 1 + 1j]], dtype_kernel)
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_input)
        self._validate_complex(array, kernel, dtype_output, mode=mode,
                               cval=5.0)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    def test_correlate_complex_kernel_invalid_cval(self, dtype_input,
                                                   dtype_kernel):
        # cannot give complex cval with a real image
        kernel = numpy.array([[1, 0],
                              [0, 1 + 1j]], dtype_kernel)
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype_input)
        for func in [ndimage.convolve, ndimage.correlate, ndimage.convolve1d,
                     ndimage.correlate1d]:
            with pytest.raises(ValueError):
                func(array, kernel, mode='constant', cval=5.0 + 1.0j,
                     output=numpy.complex64)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel(self, dtype_input, dtype_kernel,
                                        dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype_kernel)
        array = numpy.array([1, 2, 3, 4, 5, 6], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', complex_types)
    @pytest.mark.parametrize('dtype_input', types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_kernel_cval(self, dtype_input, dtype_kernel,
                                             dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype_kernel)
        array = numpy.array([1, 2, 3, 4, 5, 6], dtype_input)
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5.0)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input(self, dtype_input, dtype_kernel,
                                     dtype_output):
        kernel = numpy.array([[1, 0],
                              [0, 1]], dtype_kernel)
        array = numpy.array([[1, 2j, 3],
                             [1 + 4j, 5, 6j]], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input(self, dtype_input, dtype_kernel,
                                       dtype_output):
        kernel = numpy.array([1, 0, 1], dtype_kernel)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype_kernel', types)
    @pytest.mark.parametrize('dtype_input', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_cval(self, dtype_input, dtype_kernel,
                                            dtype_output):
        kernel = numpy.array([1, 0, 1], dtype_kernel)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype_input)
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5 - 3j)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel(self, dtype, dtype_output):
        kernel = numpy.array([[1, 0],
                              [0, 1 + 1j]], dtype)
        array = numpy.array([[1, 2j, 3],
                             [1 + 4j, 5, 6j]], dtype)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate_complex_input_and_kernel_cval(self, dtype,
                                                     dtype_output):
        kernel = numpy.array([[1, 0],
                              [0, 1 + 1j]], dtype)
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6]], dtype)
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5.0 + 2.0j)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel(self, dtype, dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        self._validate_complex(array, kernel, dtype_output)

    @pytest.mark.parametrize('dtype', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_correlate1d_complex_input_and_kernel_cval(self, dtype,
                                                       dtype_output):
        kernel = numpy.array([1, 1 + 1j], dtype)
        array = numpy.array([1, 2j, 3, 1 + 4j, 5, 6j], dtype)
        self._validate_complex(array, kernel, dtype_output, mode='constant',
                               cval=5.0 + 2.0j)

    def test_gauss01(self):
        input = numpy.array([[1, 2, 3],
                             [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 0)
        assert_array_almost_equal(output, input)

    def test_gauss02(self):
        input = numpy.array([[1, 2, 3],
                             [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 1.0)
        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)

    def test_gauss03(self):
        # single precision data
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        output = ndimage.gaussian_filter(input, [1.0, 1.0])

        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)

        # input.sum() is 49995000.0.  With single precision floats, we can't
        # expect more than 8 digits of accuracy, so use decimal=0 in this test.
        assert_almost_equal(output.sum(dtype='d'), input.sum(dtype='d'),
                            decimal=0)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss04(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss05(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0],
                                         order=1, output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss06(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output1 = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        output2 = ndimage.gaussian_filter(input, 1.0, output=otype)
        assert_array_almost_equal(output1, output2)

    def test_gauss_memory_overlap(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        output1 = ndimage.gaussian_filter(input, 1.0)
        ndimage.gaussian_filter(input, 1.0, output=input)
        assert_array_almost_equal(output1, input)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
        output = ndimage.prewitt(array, 0)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
        output = numpy.zeros(array.shape, dtype)
        ndimage.prewitt(array, 0, output)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt03(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
        t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 0)
        output = ndimage.prewitt(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_prewitt04(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.prewitt(array, -1)
        output = ndimage.prewitt(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel01(sel, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
        output = ndimage.sobel(array, 0)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
        output = numpy.zeros(array.shape, dtype)
        ndimage.sobel(array, 0, output)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel03(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
        t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 0)
        output = numpy.zeros(array.shape, dtype)
        output = ndimage.sobel(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_sobel04(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        t = ndimage.sobel(array, -1)
        output = ndimage.sobel(array, 1)
        assert_array_almost_equal(t, output)

    @pytest.mark.parametrize('dtype',
                             [numpy.int32, numpy.float32, numpy.float64,
                              numpy.complex64, numpy.complex128])
    def test_laplace01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
        tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
        output = ndimage.laplace(array)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype',
                             [numpy.int32, numpy.float32, numpy.float64,
                              numpy.complex64, numpy.complex128])
    def test_laplace02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
        tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
        output = numpy.zeros(array.shape, dtype)
        ndimage.laplace(array, output=output)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype',
                             [numpy.int32, numpy.float32, numpy.float64,
                              numpy.complex64, numpy.complex128])
    def test_gaussian_laplace01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        output = ndimage.gaussian_laplace(array, 1.0)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype',
                             [numpy.int32, numpy.float32, numpy.float64,
                              numpy.complex64, numpy.complex128])
    def test_gaussian_laplace02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
        output = numpy.zeros(array.shape, dtype)
        ndimage.gaussian_laplace(array, 1.0, output)
        assert_array_almost_equal(tmp1 + tmp2, output)

    @pytest.mark.parametrize('dtype', types + complex_types)
    def test_generic_laplace01(self, dtype):
        def derivative2(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = numpy.asarray(input)
            order = [0] * input.ndim
            order[axis] = 2
            return ndimage.gaussian_filter(input, sigma, order,
                                           output, mode, cval)
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        output = numpy.zeros(array.shape, dtype)
        tmp = ndimage.generic_laplace(array, derivative2,
                                      extra_arguments=(1.0,),
                                      extra_keywords={'b': 2.0})
        ndimage.gaussian_laplace(array, 1.0, output)
        assert_array_almost_equal(tmp, output)

    @pytest.mark.parametrize('dtype',
                             [numpy.int32, numpy.float32, numpy.float64,
                              numpy.complex64, numpy.complex128])
    def test_gaussian_gradient_magnitude01(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        output = ndimage.gaussian_gradient_magnitude(array, 1.0)
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = numpy.sqrt(expected).astype(dtype)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype',
                             [numpy.int32, numpy.float32, numpy.float64,
                              numpy.complex64, numpy.complex128])
    def test_gaussian_gradient_magnitude02(self, dtype):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype) * 100
        tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
        tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
        output = numpy.zeros(array.shape, dtype)
        ndimage.gaussian_gradient_magnitude(array, 1.0, output)
        expected = tmp1 * tmp1 + tmp2 * tmp2
        expected = numpy.sqrt(expected).astype(dtype)
        assert_array_almost_equal(expected, output)

    def test_generic_gradient_magnitude01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], numpy.float64)

        def derivative(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = numpy.asarray(input)
            order = [0] * input.ndim
            order[axis] = 1
            return ndimage.gaussian_filter(input, sigma, order,
                                           output, mode, cval)
        tmp1 = ndimage.gaussian_gradient_magnitude(array, 1.0)
        tmp2 = ndimage.generic_gradient_magnitude(
            array, derivative, extra_arguments=(1.0,),
            extra_keywords={'b': 2.0})
        assert_array_almost_equal(tmp1, tmp2)

    def test_uniform01(self):
        array = numpy.array([2, 4, 6])
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal([3, 5, 6], output)

    def test_uniform01_complex(self):
        array = numpy.array([2 + 1j, 4 + 2j, 6 + 3j], dtype=numpy.complex128)
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal([3, 5, 6], output.real)
        assert_array_almost_equal([1.5, 2.5, 3], output.imag)

    def test_uniform02(self):
        array = numpy.array([1, 2, 3])
        filter_shape = [0]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform03(self):
        array = numpy.array([1, 2, 3])
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform04(self):
        array = numpy.array([2, 4, 6])
        filter_shape = [2]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal([2, 3, 5], output)

    def test_uniform05(self):
        array = []
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal([], output)

    @pytest.mark.parametrize('dtype_array', types)
    @pytest.mark.parametrize('dtype_output', types)
    def test_uniform06(self, dtype_array, dtype_output):
        filter_shape = [2, 2]
        array = numpy.array([[4, 8, 12],
                             [16, 20, 24]], dtype_array)
        output = ndimage.uniform_filter(
            array, filter_shape, output=dtype_output)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        assert_equal(output.dtype.type, dtype_output)

    @pytest.mark.parametrize('dtype_array', complex_types)
    @pytest.mark.parametrize('dtype_output', complex_types)
    def test_uniform06_complex(self, dtype_array, dtype_output):
        filter_shape = [2, 2]
        array = numpy.array([[4, 8 + 5j, 12],
                             [16, 20, 24]], dtype_array)
        output = ndimage.uniform_filter(
            array, filter_shape, output=dtype_output)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output.real)
        assert_equal(output.dtype.type, dtype_output)

    def test_minimum_filter01(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([2])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    def test_minimum_filter02(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    def test_minimum_filter03(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([2])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([3, 2, 2, 1, 1], output)

    def test_minimum_filter04(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([2, 2, 1, 1, 1], output)

    def test_minimum_filter05(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        filter_shape = numpy.array([2, 3])
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 2, 1, 1, 1],
                                   [5, 3, 3, 1, 1]], output)

    def test_minimum_filter05_overlap(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        filter_shape = numpy.array([2, 3])
        ndimage.minimum_filter(array, filter_shape, output=array)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 2, 1, 1, 1],
                                   [5, 3, 3, 1, 1]], array)

    def test_minimum_filter06(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 2, 1, 1, 1],
                                   [5, 3, 3, 1, 1]], output)
        # separable footprint should allow mode sequence
        output2 = ndimage.minimum_filter(array, footprint=footprint,
                                         mode=['reflect', 'reflect'])
        assert_array_almost_equal(output2, output)

    def test_minimum_filter07(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], output)
        with assert_raises(RuntimeError):
            ndimage.minimum_filter(array, footprint=footprint,
                                   mode=['reflect', 'constant'])

    def test_minimum_filter08(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.minimum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal([[3, 1, 3, 1, 1],
                                   [5, 3, 3, 1, 1],
                                   [3, 3, 1, 1, 1]], output)

    def test_minimum_filter09(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.minimum_filter(array, footprint=footprint,
                                        origin=[-1, 0])
        assert_array_almost_equal([[2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1],
                                   [5, 3, 3, 1, 1]], output)

    def test_maximum_filter01(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([2])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([1, 2, 3, 4, 5], output)

    def test_maximum_filter02(self):
        array = numpy.array([1, 2, 3, 4, 5])
        filter_shape = numpy.array([3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([2, 3, 4, 5, 5], output)

    def test_maximum_filter03(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([2])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    def test_maximum_filter04(self):
        array = numpy.array([3, 2, 5, 1, 4])
        filter_shape = numpy.array([3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([3, 5, 5, 5, 4], output)

    def test_maximum_filter05(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        filter_shape = numpy.array([2, 3])
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([[3, 5, 5, 5, 4],
                                   [7, 9, 9, 9, 5],
                                   [8, 9, 9, 9, 7]], output)

    def test_maximum_filter06(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal([[3, 5, 5, 5, 4],
                                   [7, 9, 9, 9, 5],
                                   [8, 9, 9, 9, 7]], output)
        # separable footprint should allow mode sequence
        output2 = ndimage.maximum_filter(array, footprint=footprint,
                                         mode=['reflect', 'reflect'])
        assert_array_almost_equal(output2, output)

    def test_maximum_filter07(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal([[3, 5, 5, 5, 4],
                                   [7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7]], output)
        # non-separable footprint should not allow mode sequence
        with assert_raises(RuntimeError):
            ndimage.maximum_filter(array, footprint=footprint,
                                   mode=['reflect', 'reflect'])

    def test_maximum_filter08(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.maximum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal([[7, 9, 9, 5, 5],
                                   [9, 8, 9, 7, 5],
                                   [8, 8, 7, 7, 7]], output)

    def test_maximum_filter09(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.maximum_filter(array, footprint=footprint,
                                        origin=[-1, 0])
        assert_array_almost_equal([[7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7],
                                   [8, 8, 8, 7, 7]], output)

    def test_rank01(self):
        array = numpy.array([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, 2)
        assert_array_almost_equal(array, output)

    def test_rank02(self):
        array = numpy.array([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=[3])
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, (3,))
        assert_array_almost_equal(array, output)

    def test_rank03(self):
        array = numpy.array([3, 2, 5, 1, 4])
        output = ndimage.rank_filter(array, 1, size=[2])
        assert_array_almost_equal([3, 3, 5, 5, 4], output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    def test_rank04(self):
        array = numpy.array([3, 2, 5, 1, 4])
        expected = [3, 3, 2, 4, 4]
        output = ndimage.rank_filter(array, 1, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank05(self):
        array = numpy.array([3, 2, 5, 1, 4])
        expected = [3, 3, 2, 4, 4]
        output = ndimage.rank_filter(array, -2, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank06(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]])
        expected = [[2, 2, 1, 1, 1],
                    [3, 3, 2, 1, 1],
                    [5, 5, 3, 3, 1]]
        output = ndimage.rank_filter(array, 1, size=[2, 3])
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 17, size=(2, 3))
        assert_array_almost_equal(expected, output)

    def test_rank06_overlap(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]])
        array_copy = array.copy()
        expected = [[2, 2, 1, 1, 1],
                    [3, 3, 2, 1, 1],
                    [5, 5, 3, 3, 1]]
        ndimage.rank_filter(array, 1, size=[2, 3], output=array)
        assert_array_almost_equal(expected, array)

        ndimage.percentile_filter(array_copy, 17, size=(2, 3),
                                  output=array_copy)
        assert_array_almost_equal(expected, array_copy)

    def test_rank07(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]])
        expected = [[3, 5, 5, 5, 4],
                    [5, 5, 7, 5, 4],
                    [6, 8, 8, 7, 5]]
        output = ndimage.rank_filter(array, -2, size=[2, 3])
        assert_array_almost_equal(expected, output)

    def test_rank08(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]])
        expected = [[3, 3, 2, 4, 4],
                    [5, 5, 5, 4, 4],
                    [5, 6, 7, 5, 5]]
        output = ndimage.percentile_filter(array, 50.0, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.rank_filter(array, 3, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=(2, 3))
        assert_array_almost_equal(expected, output)

        # non-separable: does not allow mode sequence
        with assert_raises(RuntimeError):
            ndimage.percentile_filter(array, 50.0, size=(2, 3),
                                      mode=['reflect', 'constant'])
        with assert_raises(RuntimeError):
            ndimage.rank_filter(array, 3, size=(2, 3), mode=['reflect']*2)
        with assert_raises(RuntimeError):
            ndimage.median_filter(array, size=(2, 3), mode=['reflect']*2)

    @pytest.mark.parametrize('dtype', types)
    def test_rank09(self, dtype):
        expected = [[3, 3, 2, 4, 4],
                    [3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 35, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank10(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        expected = [[2, 2, 1, 1, 1],
                    [2, 3, 1, 3, 1],
                    [5, 5, 3, 3, 1]]
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.rank_filter(array, 0, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 0.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank11(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        expected = [[3, 5, 5, 5, 4],
                    [7, 7, 9, 9, 5],
                    [7, 9, 8, 9, 7]]
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.rank_filter(array, -1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 100.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank12(self, dtype):
        expected = [[3, 3, 2, 4, 4],
                    [3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50.0,
                                           footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank13(self, dtype):
        expected = [[5, 2, 5, 1, 1],
                    [5, 8, 3, 5, 5],
                    [6, 6, 5, 5, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint,
                                     origin=-1)
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank14(self, dtype):
        expected = [[3, 5, 2, 5, 1],
                    [5, 5, 8, 3, 5],
                    [5, 6, 6, 5, 5]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 1, footprint=footprint,
                                     origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_rank15(self, dtype):
        expected = [[2, 3, 1, 4, 1],
                    [5, 3, 7, 1, 1],
                    [5, 5, 3, 3, 3]]
        footprint = [[1, 0, 1], [0, 1, 0]]
        array = numpy.array([[3, 2, 5, 1, 4],
                             [5, 8, 3, 7, 1],
                             [5, 6, 9, 3, 5]], dtype)
        output = ndimage.rank_filter(array, 0, footprint=footprint,
                                     origin=[-1, 0])
        assert_array_almost_equal(expected, output)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter1d01(self, dtype):
        weights = numpy.array([1.1, 2.2, 3.3])

        def _filter_func(input, output, fltr, total):
            fltr = fltr / total
            for ii in range(input.shape[0] - 2):
                output[ii] = input[ii] * fltr[0]
                output[ii] += input[ii + 1] * fltr[1]
                output[ii] += input[ii + 2] * fltr[2]
        a = numpy.arange(12, dtype=dtype)
        a.shape = (3, 4)
        r1 = ndimage.correlate1d(a, weights / weights.sum(), 0, origin=-1)
        r2 = ndimage.generic_filter1d(
            a, _filter_func, 3, axis=0, origin=-1,
            extra_arguments=(weights,),
            extra_keywords={'total': weights.sum()})
        assert_array_almost_equal(r1, r2)

    @pytest.mark.parametrize('dtype', types)
    def test_generic_filter01(self, dtype):
        filter_ = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        footprint = numpy.array([[1, 0], [0, 1]])
        cf = numpy.array([1., 4.])

        def _filter_func(buffer, weights, total=1.0):
            weights = cf / total
            return (buffer * weights).sum()

        a = numpy.arange(12, dtype=dtype)
        a.shape = (3, 4)
        r1 = ndimage.correlate(a, filter_ * footprint)
        if dtype in float_types:
            r1 /= 5
        else:
            r1 //= 5
        r2 = ndimage.generic_filter(
            a, _filter_func, footprint=footprint, extra_arguments=(cf,),
            extra_keywords={'total': cf.sum()})
        assert_array_almost_equal(r1, r2)

        # generic_filter doesn't allow mode sequence
        with assert_raises(RuntimeError):
            r2 = ndimage.generic_filter(
                a, _filter_func, mode=['reflect', 'reflect'],
                footprint=footprint, extra_arguments=(cf,),
                extra_keywords={'total': cf.sum()})

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 2]),
         ('wrap', [3, 1, 2]),
         ('reflect', [1, 1, 2]),
         ('mirror', [2, 1, 2]),
         ('constant', [0, 1, 2])]
    )
    def test_extend01(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([1, 0])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 1]),
         ('wrap', [3, 1, 2]),
         ('reflect', [3, 3, 2]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend02(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([1, 0, 0, 0, 0, 0, 0, 0])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [2, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 3, 3]),
         ('mirror', [2, 3, 2]),
         ('constant', [2, 3, 0])]
    )
    def test_extend03(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 1])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend04(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]),
         ('wrap', [[9, 7, 8], [3, 1, 2], [6, 4, 5]]),
         ('reflect', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]),
         ('mirror', [[5, 4, 5], [2, 1, 2], [5, 4, 5]]),
         ('constant', [[0, 0, 0], [0, 1, 2], [0, 4, 5]])]
    )
    def test_extend05(self, mode, expected_value):
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        weights = numpy.array([[1, 0], [0, 0]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]),
         ('wrap', [[5, 6, 4], [8, 9, 7], [2, 3, 1]]),
         ('reflect', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]),
         ('mirror', [[5, 6, 5], [8, 9, 8], [5, 6, 5]]),
         ('constant', [[5, 6, 0], [8, 9, 0], [0, 0, 0]])]
    )
    def test_extend06(self, mode, expected_value):
        array = numpy.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        weights = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend07(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[3], [3], [3]]),
         ('wrap', [[2], [3], [1]]),
         ('reflect', [[2], [1], [1]]),
         ('mirror', [[1], [2], [3]]),
         ('constant', [[0], [0], [0]])]
    )
    def test_extend08(self, mode, expected_value):
        array = numpy.array([[1], [2], [3]])
        weights = numpy.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [3, 3, 3]),
         ('wrap', [2, 3, 1]),
         ('reflect', [2, 1, 1]),
         ('mirror', [1, 2, 3]),
         ('constant', [0, 0, 0])]
    )
    def test_extend09(self, mode, expected_value):
        array = numpy.array([1, 2, 3])
        weights = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [[3], [3], [3]]),
         ('wrap', [[2], [3], [1]]),
         ('reflect', [[2], [1], [1]]),
         ('mirror', [[1], [2], [3]]),
         ('constant', [[0], [0], [0]])]
    )
    def test_extend10(self, mode, expected_value):
        array = numpy.array([[1], [2], [3]])
        weights = numpy.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        output = ndimage.correlate(array, weights, mode=mode, cval=0)
        assert_array_equal(output, expected_value)


def test_ticket_701():
    # Test generic filter sizes
    arr = numpy.arange(4).reshape((2, 2))
    func = lambda x: numpy.min(x)
    res = ndimage.generic_filter(arr, func, size=(1, 1))
    # The following raises an error unless ticket 701 is fixed
    res2 = ndimage.generic_filter(arr, func, size=1)
    assert_equal(res, res2)


def test_gh_5430():
    # At least one of these raises an error unless gh-5430 is
    # fixed. In py2k an int is implemented using a C long, so
    # which one fails depends on your system. In py3k there is only
    # one arbitrary precision integer type, so both should fail.
    sigma = numpy.int32(1)
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert_equal(out, [sigma])
    sigma = numpy.int64(1)
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert_equal(out, [sigma])
    # This worked before; make sure it still works
    sigma = 1
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert_equal(out, [sigma])
    # This worked before; make sure it still works
    sigma = [1, 1]
    out = ndimage._ni_support._normalize_sequence(sigma, 2)
    assert_equal(out, sigma)
    # Also include the OPs original example to make sure we fixed the issue
    x = numpy.random.normal(size=(256, 256))
    perlin = numpy.zeros_like(x)
    for i in 2**numpy.arange(6):
        perlin += ndimage.filters.gaussian_filter(x, i, mode="wrap") * i**2
    # This also fixes gh-4106, show that the OPs example now runs.
    x = numpy.int64(21)
    ndimage._ni_support._normalize_sequence(x, 0)


def test_gaussian_kernel1d():
    radius = 10
    sigma = 2
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1, dtype=numpy.double)
    phi_x = numpy.exp(-0.5 * x * x / sigma2)
    phi_x /= phi_x.sum()
    assert_allclose(phi_x, _gaussian_kernel1d(sigma, 0, radius))
    assert_allclose(-phi_x * x / sigma2, _gaussian_kernel1d(sigma, 1, radius))
    assert_allclose(phi_x * (x * x / sigma2 - 1) / sigma2,
                    _gaussian_kernel1d(sigma, 2, radius))
    assert_allclose(phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2),
                    _gaussian_kernel1d(sigma, 3, radius))


def test_orders_gauss():
    # Check order inputs to Gaussians
    arr = numpy.zeros((1,))
    assert_equal(0, ndimage.gaussian_filter(arr, 1, order=0))
    assert_equal(0, ndimage.gaussian_filter(arr, 1, order=3))
    assert_raises(ValueError, ndimage.gaussian_filter, arr, 1, -1)
    assert_equal(0, ndimage.gaussian_filter1d(arr, 1, axis=-1, order=0))
    assert_equal(0, ndimage.gaussian_filter1d(arr, 1, axis=-1, order=3))
    assert_raises(ValueError, ndimage.gaussian_filter1d, arr, 1, -1, -1)


def test_valid_origins():
    """Regression test for #1311."""
    func = lambda x: numpy.mean(x)
    data = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float64)
    assert_raises(ValueError, ndimage.generic_filter, data, func, size=3,
                  origin=2)
    assert_raises(ValueError, ndimage.generic_filter1d, data, func,
                  filter_size=3, origin=2)
    assert_raises(ValueError, ndimage.percentile_filter, data, 0.2, size=3,
                  origin=2)

    for filter in [ndimage.uniform_filter, ndimage.minimum_filter,
                   ndimage.maximum_filter, ndimage.maximum_filter1d,
                   ndimage.median_filter, ndimage.minimum_filter1d]:
        # This should work, since for size == 3, the valid range for origin is
        # -1 to 1.
        list(filter(data, 3, origin=-1))
        list(filter(data, 3, origin=1))
        # Just check this raises an error instead of silently accepting or
        # segfaulting.
        assert_raises(ValueError, filter, data, 3, origin=2)


def test_bad_convolve_and_correlate_origins():
    """Regression test for gh-822."""
    # Before gh-822 was fixed, these would generate seg. faults or
    # other crashes on many system.
    assert_raises(ValueError, ndimage.correlate1d,
                  [0, 1, 2, 3, 4, 5], [1, 1, 2, 0], origin=2)
    assert_raises(ValueError, ndimage.correlate,
                  [0, 1, 2, 3, 4, 5], [0, 1, 2], origin=[2])
    assert_raises(ValueError, ndimage.correlate,
                  numpy.ones((3, 5)), numpy.ones((2, 2)), origin=[0, 1])

    assert_raises(ValueError, ndimage.convolve1d,
                  numpy.arange(10), numpy.ones(3), origin=-2)
    assert_raises(ValueError, ndimage.convolve,
                  numpy.arange(10), numpy.ones(3), origin=[-2])
    assert_raises(ValueError, ndimage.convolve,
                  numpy.ones((3, 5)), numpy.ones((2, 2)), origin=[0, -2])


def test_multiple_modes():
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying a single mode.
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    mode1 = 'reflect'
    mode2 = ['reflect', 'reflect']

    assert_equal(ndimage.gaussian_filter(arr, 1, mode=mode1),
                 ndimage.gaussian_filter(arr, 1, mode=mode2))
    assert_equal(ndimage.prewitt(arr, mode=mode1),
                 ndimage.prewitt(arr, mode=mode2))
    assert_equal(ndimage.sobel(arr, mode=mode1),
                 ndimage.sobel(arr, mode=mode2))
    assert_equal(ndimage.laplace(arr, mode=mode1),
                 ndimage.laplace(arr, mode=mode2))
    assert_equal(ndimage.gaussian_laplace(arr, 1, mode=mode1),
                 ndimage.gaussian_laplace(arr, 1, mode=mode2))
    assert_equal(ndimage.maximum_filter(arr, size=5, mode=mode1),
                 ndimage.maximum_filter(arr, size=5, mode=mode2))
    assert_equal(ndimage.minimum_filter(arr, size=5, mode=mode1),
                 ndimage.minimum_filter(arr, size=5, mode=mode2))
    assert_equal(ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode1),
                 ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode2))
    assert_equal(ndimage.uniform_filter(arr, 5, mode=mode1),
                 ndimage.uniform_filter(arr, 5, mode=mode2))


def test_multiple_modes_sequentially():
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying the filters with
    # different modes sequentially
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    modes = ['reflect', 'wrap']

    expected = ndimage.gaussian_filter1d(arr, 1, axis=0, mode=modes[0])
    expected = ndimage.gaussian_filter1d(expected, 1, axis=1, mode=modes[1])
    assert_equal(expected,
                 ndimage.gaussian_filter(arr, 1, mode=modes))

    expected = ndimage.uniform_filter1d(arr, 5, axis=0, mode=modes[0])
    expected = ndimage.uniform_filter1d(expected, 5, axis=1, mode=modes[1])
    assert_equal(expected,
                 ndimage.uniform_filter(arr, 5, mode=modes))

    expected = ndimage.maximum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = ndimage.maximum_filter1d(expected, size=5, axis=1,
                                        mode=modes[1])
    assert_equal(expected,
                 ndimage.maximum_filter(arr, size=5, mode=modes))

    expected = ndimage.minimum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = ndimage.minimum_filter1d(expected, size=5, axis=1,
                                        mode=modes[1])
    assert_equal(expected,
                 ndimage.minimum_filter(arr, size=5, mode=modes))


def test_multiple_modes_prewitt():
    # Test prewitt filter for multiple extrapolation modes
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    expected = numpy.array([[1., -3., 2.],
                            [1., -2., 1.],
                            [1., -1., 0.]])

    modes = ['reflect', 'wrap']

    assert_equal(expected,
                 ndimage.prewitt(arr, mode=modes))


def test_multiple_modes_sobel():
    # Test sobel filter for multiple extrapolation modes
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    expected = numpy.array([[1., -4., 3.],
                            [2., -3., 1.],
                            [1., -1., 0.]])

    modes = ['reflect', 'wrap']

    assert_equal(expected,
                 ndimage.sobel(arr, mode=modes))


def test_multiple_modes_laplace():
    # Test laplace filter for multiple extrapolation modes
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    expected = numpy.array([[-2., 2., 1.],
                            [-2., -3., 2.],
                            [1., 1., 0.]])

    modes = ['reflect', 'wrap']

    assert_equal(expected,
                 ndimage.laplace(arr, mode=modes))


def test_multiple_modes_gaussian_laplace():
    # Test gaussian_laplace filter for multiple extrapolation modes
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    expected = numpy.array([[-0.28438687, 0.01559809, 0.19773499],
                            [-0.36630503, -0.20069774, 0.07483620],
                            [0.15849176, 0.18495566, 0.21934094]])

    modes = ['reflect', 'wrap']

    assert_almost_equal(expected,
                        ndimage.gaussian_laplace(arr, 1, mode=modes))


def test_multiple_modes_gaussian_gradient_magnitude():
    # Test gaussian_gradient_magnitude filter for multiple
    # extrapolation modes
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    expected = numpy.array([[0.04928965, 0.09745625, 0.06405368],
                            [0.23056905, 0.14025305, 0.04550846],
                            [0.19894369, 0.14950060, 0.06796850]])

    modes = ['reflect', 'wrap']

    calculated = ndimage.gaussian_gradient_magnitude(arr, 1, mode=modes)

    assert_almost_equal(expected, calculated)


def test_multiple_modes_uniform():
    # Test uniform filter for multiple extrapolation modes
    arr = numpy.array([[1., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.]])

    expected = numpy.array([[0.32, 0.40, 0.48],
                            [0.20, 0.28, 0.32],
                            [0.28, 0.32, 0.40]])

    modes = ['reflect', 'wrap']

    assert_almost_equal(expected,
                        ndimage.uniform_filter(arr, 5, mode=modes))


def test_gaussian_truncate():
    # Test that Gaussian filters can be truncated at different widths.
    # These tests only check that the result has the expected number
    # of nonzero elements.
    arr = numpy.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (ndimage.gaussian_filter(arr, 5, truncate=2) > 0).sum()
    assert_equal(num_nonzeros_2, 21**2)
    num_nonzeros_5 = (ndimage.gaussian_filter(arr, 5, truncate=5) > 0).sum()
    assert_equal(num_nonzeros_5, 51**2)

    # Test truncate when sigma is a sequence.
    f = ndimage.gaussian_filter(arr, [0.5, 2.5], truncate=3.5)
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    # n0 should be 2*int(2.5*3.5 + 0.5) + 1
    assert_equal(n0, 19)
    n1 = fpos.any(axis=1).sum()
    # n1 should be 2*int(0.5*3.5 + 0.5) + 1
    assert_equal(n1, 5)

    # Test gaussian_filter1d.
    x = numpy.zeros(51)
    x[25] = 1
    f = ndimage.gaussian_filter1d(x, sigma=2, truncate=3.5)
    n = (f > 0).sum()
    assert_equal(n, 15)

    # Test gaussian_laplace
    y = ndimage.gaussian_laplace(x, sigma=2, truncate=3.5)
    nonzero_indices = numpy.nonzero(y != 0)[0]
    n = nonzero_indices.ptp() + 1
    assert_equal(n, 15)

    # Test gaussian_gradient_magnitude
    y = ndimage.gaussian_gradient_magnitude(x, sigma=2, truncate=3.5)
    nonzero_indices = numpy.nonzero(y != 0)[0]
    n = nonzero_indices.ptp() + 1
    assert_equal(n, 15)


class TestThreading:
    def check_func_thread(self, n, fun, args, out):
        from threading import Thread
        thrds = [Thread(target=fun, args=args, kwargs={'output': out[x]})
                 for x in range(n)]
        [t.start() for t in thrds]
        [t.join() for t in thrds]

    def check_func_serial(self, n, fun, args, out):
        for i in range(n):
            fun(*args, output=out[i])

    def test_correlate1d(self):
        d = numpy.random.randn(5000)
        os = numpy.empty((4, d.size))
        ot = numpy.empty_like(os)
        k = numpy.arange(5)
        self.check_func_serial(4, ndimage.correlate1d, (d, k), os)
        self.check_func_thread(4, ndimage.correlate1d, (d, k), ot)
        assert_array_equal(os, ot)

    def test_correlate(self):
        d = numpy.random.randn(500, 500)
        k = numpy.random.randn(10, 10)
        os = numpy.empty([4] + list(d.shape))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.correlate, (d, k), os)
        self.check_func_thread(4, ndimage.correlate, (d, k), ot)
        assert_array_equal(os, ot)

    def test_median_filter(self):
        d = numpy.random.randn(500, 500)
        os = numpy.empty([4] + list(d.shape))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.median_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.median_filter, (d, 3), ot)
        assert_array_equal(os, ot)

    def test_uniform_filter1d(self):
        d = numpy.random.randn(5000)
        os = numpy.empty((4, d.size))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.uniform_filter1d, (d, 5), os)
        self.check_func_thread(4, ndimage.uniform_filter1d, (d, 5), ot)
        assert_array_equal(os, ot)

    def test_minmax_filter(self):
        d = numpy.random.randn(500, 500)
        os = numpy.empty([4] + list(d.shape))
        ot = numpy.empty_like(os)
        self.check_func_serial(4, ndimage.maximum_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.maximum_filter, (d, 3), ot)
        assert_array_equal(os, ot)
        self.check_func_serial(4, ndimage.minimum_filter, (d, 3), os)
        self.check_func_thread(4, ndimage.minimum_filter, (d, 3), ot)
        assert_array_equal(os, ot)


def test_minmaximum_filter1d():
    # Regression gh-3898
    in_ = numpy.arange(10)
    out = ndimage.minimum_filter1d(in_, 1)
    assert_equal(in_, out)
    out = ndimage.maximum_filter1d(in_, 1)
    assert_equal(in_, out)
    # Test reflect
    out = ndimage.minimum_filter1d(in_, 5, mode='reflect')
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='reflect')
    assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    # Test constant
    out = ndimage.minimum_filter1d(in_, 5, mode='constant', cval=-1)
    assert_equal([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='constant', cval=10)
    assert_equal([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], out)
    # Test nearest
    out = ndimage.minimum_filter1d(in_, 5, mode='nearest')
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='nearest')
    assert_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    # Test wrap
    out = ndimage.minimum_filter1d(in_, 5, mode='wrap')
    assert_equal([0, 0, 0, 1, 2, 3, 4, 5, 0, 0], out)
    out = ndimage.maximum_filter1d(in_, 5, mode='wrap')
    assert_equal([9, 9, 4, 5, 6, 7, 8, 9, 9, 9], out)


def test_uniform_filter1d_roundoff_errors():
    # gh-6930
    in_ = numpy.repeat([0, 1, 0], [9, 9, 9])
    for filter_size in range(3, 10):
        out = ndimage.uniform_filter1d(in_, filter_size)
        assert_equal(out.sum(), 10 - filter_size)


def test_footprint_all_zeros():
    # regression test for gh-6876: footprint of all zeros segfaults
    arr = numpy.random.randint(0, 100, (100, 100))
    kernel = numpy.zeros((3, 3), bool)
    with assert_raises(ValueError):
        ndimage.maximum_filter(arr, footprint=kernel)


def test_gaussian_filter():
    # Test gaussian filter with numpy.float16
    # gh-8207
    data = numpy.array([1], dtype=numpy.float16)
    sigma = 1.0
    with assert_raises(RuntimeError):
        ndimage.gaussian_filter(data, sigma)


def test_rank_filter_noninteger_rank():
    # regression test for issue 9388: ValueError for
    # non integer rank when performing rank_filter
    arr = numpy.random.random((10, 20, 30))
    assert_raises(TypeError, rank_filter, arr, 0.5,
                  footprint=numpy.ones((1, 1, 10), dtype=bool))


def test_size_footprint_both_set():
    # test for input validation, expect user warning when
    # size and footprint is set
    with suppress_warnings() as sup:
        sup.filter(UserWarning,
                   "ignoring size because footprint is set")
        arr = numpy.random.random((10, 20, 30))
        rank_filter(arr, 5, size=2, footprint=numpy.ones((1, 1, 10),
                    dtype=bool))


def test_byte_order_median():
    """Regression test for #413: median_filter does not handle bytes orders."""
    a = numpy.arange(9, dtype='<f4').reshape(3, 3)
    ref = ndimage.filters.median_filter(a, (3, 3))
    b = numpy.arange(9, dtype='>f4').reshape(3, 3)
    t = ndimage.filters.median_filter(b, (3, 3))
    assert_array_almost_equal(ref, t)
