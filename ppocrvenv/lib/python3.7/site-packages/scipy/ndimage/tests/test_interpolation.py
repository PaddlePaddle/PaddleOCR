import sys

import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose,
                           suppress_warnings)
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage

from . import types

eps = 1e-12

ndimage_to_numpy_mode = {
    'mirror': 'reflect',
    'reflect': 'symmetric',
    'grid-mirror': 'symmetric',
    'grid-wrap': 'wrap',
    'nearest': 'edge',
    'grid-constant': 'constant',
}


class TestNdimageInterpolation:

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1.5, 2.5, 3.5, 4, 4, 4, 4]),
         ('wrap', [1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1.5]),
         ('grid-wrap', [1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5]),
         ('mirror', [1.5, 2.5, 3.5, 3.5, 2.5, 1.5, 1.5]),
         ('reflect', [1.5, 2.5, 3.5, 4, 3.5, 2.5, 1.5]),
         ('constant', [1.5, 2.5, 3.5, -1, -1, -1, -1]),
         ('grid-constant', [1.5, 2.5, 3.5, 1.5, -1, -1, -1])]
    )
    def test_boundaries(self, mode, expected_value):
        def shift(x):
            return (x[0] + 0.5,)

        data = numpy.array([1, 2, 3, 4.])
        assert_array_equal(
            expected_value,
            ndimage.geometric_transform(data, shift, cval=-1, mode=mode,
                                        output_shape=(7,), order=1))

    @pytest.mark.parametrize(
        'mode, expected_value',
        [('nearest', [1, 1, 2, 3]),
         ('wrap', [3, 1, 2, 3]),
         ('grid-wrap', [4, 1, 2, 3]),
         ('mirror', [2, 1, 2, 3]),
         ('reflect', [1, 1, 2, 3]),
         ('constant', [-1, 1, 2, 3]),
         ('grid-constant', [-1, 1, 2, 3])]
    )
    def test_boundaries2(self, mode, expected_value):
        def shift(x):
            return (x[0] - 0.9,)

        data = numpy.array([1, 2, 3, 4])
        assert_array_equal(
            expected_value,
            ndimage.geometric_transform(data, shift, cval=-1, mode=mode,
                                        output_shape=(4,)))

    @pytest.mark.parametrize('mode', ['mirror', 'reflect', 'grid-mirror',
                                      'grid-wrap', 'grid-constant',
                                      'nearest'])
    @pytest.mark.parametrize('order', range(6))
    def test_boundary_spline_accuracy(self, mode, order):
        """Tests based on examples from gh-2640"""
        data = numpy.arange(-6, 7, dtype=float)
        x = numpy.linspace(-8, 15, num=1000)
        y = ndimage.map_coordinates(data, [x], order=order, mode=mode)

        # compute expected value using explicit padding via numpy.pad
        npad = 32
        pad_mode = ndimage_to_numpy_mode.get(mode)
        padded = numpy.pad(data, npad, mode=pad_mode)
        expected = ndimage.map_coordinates(padded, [npad + x], order=order,
                                           mode=mode)

        atol = 1e-5 if mode == 'grid-constant' else 1e-12
        assert_allclose(y, expected, rtol=1e-7, atol=atol)

    @pytest.mark.parametrize('order', range(2, 6))
    @pytest.mark.parametrize('dtype', types)
    def test_spline01(self, dtype, order):
        data = numpy.ones([], dtype)
        out = ndimage.spline_filter(data, order=order)
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('order', range(2, 6))
    @pytest.mark.parametrize('dtype', types)
    def test_spline02(self, dtype, order):
        data = numpy.array([1], dtype)
        out = ndimage.spline_filter(data, order=order)
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('order', range(2, 6))
    @pytest.mark.parametrize('dtype', types)
    def test_spline03(self, dtype, order):
        data = numpy.ones([], dtype)
        out = ndimage.spline_filter(data, order, output=dtype)
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('order', range(2, 6))
    @pytest.mark.parametrize('dtype', types)
    def test_spline04(self, dtype, order):
        data = numpy.ones([4], dtype)
        out = ndimage.spline_filter(data, order)
        assert_array_almost_equal(out, [1, 1, 1, 1])

    @pytest.mark.parametrize('order', range(2, 6))
    @pytest.mark.parametrize('dtype', types)
    def test_spline05(self, dtype, order):
        data = numpy.ones([4, 4], dtype)
        out = ndimage.spline_filter(data, order=order)
        assert_array_almost_equal(out, [[1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform01(self, order):
        data = numpy.array([1])

        def mapping(x):
            return x

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform02(self, order):
        data = numpy.ones([4])

        def mapping(x):
            return x

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [1, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform03(self, order):
        data = numpy.ones([4])

        def mapping(x):
            return (x[0] - 1,)

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [0, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform04(self, order):
        data = numpy.array([4, 1, 3, 2])

        def mapping(x):
            return (x[0] - 1,)

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [0, 4, 1, 3])

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_geometric_transform05(self, order, dtype):
        data = numpy.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]], dtype=dtype)
        expected = numpy.array([[0, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]], dtype=dtype)
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected

        def mapping(x):
            return (x[0], x[1] - 1)

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform06(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])

        def mapping(x):
            return (x[0], x[1] - 1)

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [[0, 4, 1, 3],
                                        [0, 7, 6, 8],
                                        [0, 3, 5, 3]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform07(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])

        def mapping(x):
            return (x[0] - 1, x[1])

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [4, 1, 3, 2],
                                        [7, 6, 8, 5]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform08(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])

        def mapping(x):
            return (x[0] - 1, x[1] - 1)

        out = ndimage.geometric_transform(data, mapping, data.shape,
                                          order=order)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform10(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])

        def mapping(x):
            return (x[0] - 1, x[1] - 1)

        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        out = ndimage.geometric_transform(filtered, mapping, data.shape,
                                          order=order, prefilter=False)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform13(self, order):
        data = numpy.ones([2], numpy.float64)

        def mapping(x):
            return (x[0] // 2,)

        out = ndimage.geometric_transform(data, mapping, [4], order=order)
        assert_array_almost_equal(out, [1, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform14(self, order):
        data = [1, 5, 2, 6, 3, 7, 4, 4]

        def mapping(x):
            return (2 * x[0],)

        out = ndimage.geometric_transform(data, mapping, [4], order=order)
        assert_array_almost_equal(out, [1, 2, 3, 4])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform15(self, order):
        data = [1, 2, 3, 4]

        def mapping(x):
            return (x[0] / 2,)

        out = ndimage.geometric_transform(data, mapping, [8], order=order)
        assert_array_almost_equal(out[::2], [1, 2, 3, 4])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform16(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9.0, 10, 11, 12]]

        def mapping(x):
            return (x[0], x[1] * 2)

        out = ndimage.geometric_transform(data, mapping, (3, 2),
                                          order=order)
        assert_array_almost_equal(out, [[1, 3], [5, 7], [9, 11]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform17(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x):
            return (x[0] * 2, x[1])

        out = ndimage.geometric_transform(data, mapping, (1, 4),
                                          order=order)
        assert_array_almost_equal(out, [[1, 2, 3, 4]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform18(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x):
            return (x[0] * 2, x[1] * 2)

        out = ndimage.geometric_transform(data, mapping, (1, 2),
                                          order=order)
        assert_array_almost_equal(out, [[1, 3]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform19(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x):
            return (x[0], x[1] / 2)

        out = ndimage.geometric_transform(data, mapping, (3, 8),
                                          order=order)
        assert_array_almost_equal(out[..., ::2], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform20(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x):
            return (x[0] / 2, x[1])

        out = ndimage.geometric_transform(data, mapping, (6, 4),
                                          order=order)
        assert_array_almost_equal(out[::2, ...], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform21(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x):
            return (x[0] / 2, x[1] / 2)

        out = ndimage.geometric_transform(data, mapping, (6, 8),
                                          order=order)
        assert_array_almost_equal(out[::2, ::2], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform22(self, order):
        data = numpy.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]], numpy.float64)

        def mapping1(x):
            return (x[0] / 2, x[1] / 2)

        def mapping2(x):
            return (x[0] * 2, x[1] * 2)

        out = ndimage.geometric_transform(data, mapping1,
                                          (6, 8), order=order)
        out = ndimage.geometric_transform(out, mapping2,
                                          (3, 4), order=order)
        assert_array_almost_equal(out, data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform23(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x):
            return (1, x[0] * 2)

        out = ndimage.geometric_transform(data, mapping, (2,), order=order)
        out = out.astype(numpy.int32)
        assert_array_almost_equal(out, [5, 7])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_geometric_transform24(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

        def mapping(x, a, b):
            return (a, x[0] * b)

        out = ndimage.geometric_transform(
            data, mapping, (2,), order=order, extra_arguments=(1,),
            extra_keywords={'b': 2})
        assert_array_almost_equal(out, [5, 7])

    def test_geometric_transform_grid_constant_order1(self):
        # verify interpolation outside the original bounds
        x = numpy.array([[1, 2, 3],
                         [4, 5, 6]], dtype=float)

        def mapping(x):
            return (x[0] - 0.5), (x[1] - 0.5)

        expected_result = numpy.array([[0.25, 0.75, 1.25],
                                       [1.25, 3.00, 4.00]])
        assert_array_almost_equal(
            ndimage.geometric_transform(x, mapping, mode='grid-constant',
                                        order=1),
            expected_result,
        )

    @pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest',
                                      'mirror', 'reflect'])
    @pytest.mark.parametrize('order', range(6))
    def test_geometric_transform_vs_padded(self, order, mode):
        x = numpy.arange(144, dtype=float).reshape(12, 12)

        def mapping(x):
            return (x[0] - 0.4), (x[1] + 2.3)

        # Manually pad and then extract center after the transform to get the
        # expected result.
        npad = 24
        pad_mode = ndimage_to_numpy_mode.get(mode)
        xp = numpy.pad(x, npad, mode=pad_mode)
        center_slice = tuple([slice(npad, -npad)] * x.ndim)
        expected_result = ndimage.geometric_transform(
            xp, mapping, mode=mode, order=order)[center_slice]

        assert_allclose(
            ndimage.geometric_transform(x, mapping, mode=mode,
                                        order=order),
            expected_result,
            rtol=1e-7,
        )

    def test_geometric_transform_endianness_with_output_parameter(self):
        # geometric transform given output ndarray or dtype with
        # non-native endianness. see issue #4127
        data = numpy.array([1])

        def mapping(x):
            return x

        for out in [data.dtype, data.dtype.newbyteorder(),
                    numpy.empty_like(data),
                    numpy.empty_like(data).astype(data.dtype.newbyteorder())]:
            returned = ndimage.geometric_transform(data, mapping, data.shape,
                                                   output=out)
            result = out if returned is None else returned
            assert_array_almost_equal(result, [1])

    def test_geometric_transform_with_string_output(self):
        data = numpy.array([1])

        def mapping(x):
            return x

        out = ndimage.geometric_transform(data, mapping, output='f')
        assert_(out.dtype is numpy.dtype('f'))
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_map_coordinates01(self, order, dtype):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        expected = numpy.array([[0, 0, 0, 0],
                                [0, 4, 1, 3],
                                [0, 7, 6, 8]])
        if data.dtype.kind == 'c':
            data = data - 1j * data
            expected = expected - 1j * expected

        idx = numpy.indices(data.shape)
        idx -= 1

        out = ndimage.map_coordinates(data, idx, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_map_coordinates02(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        idx = numpy.indices(data.shape, numpy.float64)
        idx -= 0.5

        out1 = ndimage.shift(data, 0.5, order=order)
        out2 = ndimage.map_coordinates(data, idx, order=order)
        assert_array_almost_equal(out1, out2)

    def test_map_coordinates03(self):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]], order='F')
        idx = numpy.indices(data.shape) - 1
        out = ndimage.map_coordinates(data, idx)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])
        assert_array_almost_equal(out, ndimage.shift(data, (1, 1)))
        idx = numpy.indices(data[::2].shape) - 1
        out = ndimage.map_coordinates(data[::2], idx)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3]])
        assert_array_almost_equal(out, ndimage.shift(data[::2], (1, 1)))
        idx = numpy.indices(data[:, ::2].shape) - 1
        out = ndimage.map_coordinates(data[:, ::2], idx)
        assert_array_almost_equal(out, [[0, 0], [0, 4], [0, 7]])
        assert_array_almost_equal(out, ndimage.shift(data[:, ::2], (1, 1)))

    def test_map_coordinates_endianness_with_output_parameter(self):
        # output parameter given as array or dtype with either endianness
        # see issue #4127
        data = numpy.array([[1, 2], [7, 6]])
        expected = numpy.array([[0, 0], [0, 1]])
        idx = numpy.indices(data.shape)
        idx -= 1
        for out in [
            data.dtype,
            data.dtype.newbyteorder(),
            numpy.empty_like(expected),
            numpy.empty_like(expected).astype(expected.dtype.newbyteorder())
        ]:
            returned = ndimage.map_coordinates(data, idx, output=out)
            result = out if returned is None else returned
            assert_array_almost_equal(result, expected)

    def test_map_coordinates_with_string_output(self):
        data = numpy.array([[1]])
        idx = numpy.indices(data.shape)
        out = ndimage.map_coordinates(data, idx, output='f')
        assert_(out.dtype is numpy.dtype('f'))
        assert_array_almost_equal(out, [[1]])

    @pytest.mark.skipif('win32' in sys.platform or numpy.intp(0).itemsize < 8,
                        reason='do not run on 32 bit or windows '
                               '(no sparse memory)')
    def test_map_coordinates_large_data(self):
        # check crash on large data
        try:
            n = 30000
            a = numpy.empty(n**2, dtype=numpy.float32).reshape(n, n)
            # fill the part we might read
            a[n - 3:, n - 3:] = 0
            ndimage.map_coordinates(a, [[n - 1.5], [n - 1.5]], order=1)
        except MemoryError as e:
            raise pytest.skip('Not enough memory available') from e

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform01(self, order):
        data = numpy.array([1])
        out = ndimage.affine_transform(data, [[1]], order=order)
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform02(self, order):
        data = numpy.ones([4])
        out = ndimage.affine_transform(data, [[1]], order=order)
        assert_array_almost_equal(out, [1, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform03(self, order):
        data = numpy.ones([4])
        out = ndimage.affine_transform(data, [[1]], -1, order=order)
        assert_array_almost_equal(out, [0, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform04(self, order):
        data = numpy.array([4, 1, 3, 2])
        out = ndimage.affine_transform(data, [[1]], -1, order=order)
        assert_array_almost_equal(out, [0, 4, 1, 3])

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_affine_transform05(self, order, dtype):
        data = numpy.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]], dtype=dtype)
        expected = numpy.array([[0, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]], dtype=dtype)
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [0, -1], order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform06(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [0, -1], order=order)
        assert_array_almost_equal(out, [[0, 4, 1, 3],
                                        [0, 7, 6, 8],
                                        [0, 3, 5, 3]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform07(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [-1, 0], order=order)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [4, 1, 3, 2],
                                        [7, 6, 8, 5]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform08(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        out = ndimage.affine_transform(data, [[1, 0], [0, 1]],
                                       [-1, -1], order=order)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform09(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        out = ndimage.affine_transform(filtered, [[1, 0], [0, 1]],
                                       [-1, -1], order=order,
                                       prefilter=False)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform10(self, order):
        data = numpy.ones([2], numpy.float64)
        out = ndimage.affine_transform(data, [[0.5]], output_shape=(4,),
                                       order=order)
        assert_array_almost_equal(out, [1, 1, 1, 0])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform11(self, order):
        data = [1, 5, 2, 6, 3, 7, 4, 4]
        out = ndimage.affine_transform(data, [[2]], 0, (4,), order=order)
        assert_array_almost_equal(out, [1, 2, 3, 4])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform12(self, order):
        data = [1, 2, 3, 4]
        out = ndimage.affine_transform(data, [[0.5]], 0, (8,), order=order)
        assert_array_almost_equal(out[::2], [1, 2, 3, 4])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform13(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9.0, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[1, 0], [0, 2]], 0, (3, 2),
                                       order=order)
        assert_array_almost_equal(out, [[1, 3], [5, 7], [9, 11]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform14(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[2, 0], [0, 1]], 0, (1, 4),
                                       order=order)
        assert_array_almost_equal(out, [[1, 2, 3, 4]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform15(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[2, 0], [0, 2]], 0, (1, 2),
                                       order=order)
        assert_array_almost_equal(out, [[1, 3]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform16(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[1, 0.0], [0, 0.5]], 0,
                                       (3, 8), order=order)
        assert_array_almost_equal(out[..., ::2], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform17(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[0.5, 0], [0, 1]], 0,
                                       (6, 4), order=order)
        assert_array_almost_equal(out[::2, ...], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform18(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[0.5, 0], [0, 0.5]], 0,
                                       (6, 8), order=order)
        assert_array_almost_equal(out[::2, ::2], data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform19(self, order):
        data = numpy.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]], numpy.float64)
        out = ndimage.affine_transform(data, [[0.5, 0], [0, 0.5]], 0,
                                       (6, 8), order=order)
        out = ndimage.affine_transform(out, [[2.0, 0], [0, 2.0]], 0,
                                       (3, 4), order=order)
        assert_array_almost_equal(out, data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform20(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[0], [2]], 0, (2,),
                                       order=order)
        assert_array_almost_equal(out, [1, 3])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform21(self, order):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        out = ndimage.affine_transform(data, [[2], [0]], 0, (2,),
                                       order=order)
        assert_array_almost_equal(out, [1, 9])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform22(self, order):
        # shift and offset interaction; see issue #1547
        data = numpy.array([4, 1, 3, 2])
        out = ndimage.affine_transform(data, [[2]], [-1], (3,),
                                       order=order)
        assert_array_almost_equal(out, [0, 1, 2])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform23(self, order):
        # shift and offset interaction; see issue #1547
        data = numpy.array([4, 1, 3, 2])
        out = ndimage.affine_transform(data, [[0.5]], [-1], (8,),
                                       order=order)
        assert_array_almost_equal(out[::2], [0, 4, 1, 3])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform24(self, order):
        # consistency between diagonal and non-diagonal case; see issue #1547
        data = numpy.array([4, 1, 3, 2])
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed')
            out1 = ndimage.affine_transform(data, [2], -1, order=order)
        out2 = ndimage.affine_transform(data, [[2]], -1, order=order)
        assert_array_almost_equal(out1, out2)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform25(self, order):
        # consistency between diagonal and non-diagonal case; see issue #1547
        data = numpy.array([4, 1, 3, 2])
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed')
            out1 = ndimage.affine_transform(data, [0.5], -1, order=order)
        out2 = ndimage.affine_transform(data, [[0.5]], -1, order=order)
        assert_array_almost_equal(out1, out2)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform26(self, order):
        # test homogeneous coordinates
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        tform_original = numpy.eye(2)
        offset_original = -numpy.ones((2, 1))
        tform_h1 = numpy.hstack((tform_original, offset_original))
        tform_h2 = numpy.vstack((tform_h1, [[0, 0, 1]]))
        out1 = ndimage.affine_transform(filtered, tform_original,
                                        offset_original.ravel(),
                                        order=order, prefilter=False)
        out2 = ndimage.affine_transform(filtered, tform_h1, order=order,
                                        prefilter=False)
        out3 = ndimage.affine_transform(filtered, tform_h2, order=order,
                                        prefilter=False)
        for out in [out1, out2, out3]:
            assert_array_almost_equal(out, [[0, 0, 0, 0],
                                            [0, 4, 1, 3],
                                            [0, 7, 6, 8]])

    def test_affine_transform27(self):
        # test valid homogeneous transformation matrix
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        tform_h1 = numpy.hstack((numpy.eye(2), -numpy.ones((2, 1))))
        tform_h2 = numpy.vstack((tform_h1, [[5, 2, 1]]))
        assert_raises(ValueError, ndimage.affine_transform, data, tform_h2)

    def test_affine_transform_1d_endianness_with_output_parameter(self):
        # 1d affine transform given output ndarray or dtype with
        # either endianness. see issue #7388
        data = numpy.ones((2, 2))
        for out in [numpy.empty_like(data),
                    numpy.empty_like(data).astype(data.dtype.newbyteorder()),
                    data.dtype, data.dtype.newbyteorder()]:
            with suppress_warnings() as sup:
                sup.filter(UserWarning,
                           'The behavior of affine_transform with a 1-D array '
                           '.* has changed')
                returned = ndimage.affine_transform(data, [1, 1], output=out)
            result = out if returned is None else returned
            assert_array_almost_equal(result, [[1, 1], [1, 1]])

    def test_affine_transform_multi_d_endianness_with_output_parameter(self):
        # affine transform given output ndarray or dtype with either endianness
        # see issue #4127
        data = numpy.array([1])
        for out in [data.dtype, data.dtype.newbyteorder(),
                    numpy.empty_like(data),
                    numpy.empty_like(data).astype(data.dtype.newbyteorder())]:
            returned = ndimage.affine_transform(data, [[1]], output=out)
            result = out if returned is None else returned
            assert_array_almost_equal(result, [1])


    def test_affine_transform_output_shape(self):
        # don't require output_shape when out of a different size is given
        data = numpy.arange(8, dtype=numpy.float64)
        out = numpy.ones((16,))
        oshape = out.shape

        ndimage.affine_transform(data, [[1]], output=out)
        assert_array_almost_equal(out[:8], data)

        # mismatched output shape raises an error
        with pytest.raises(RuntimeError):
            ndimage.affine_transform(
                data, [[1]], output=out, output_shape=(12,))


    def test_affine_transform_with_string_output(self):
        data = numpy.array([1])
        out = ndimage.affine_transform(data, [[1]], output='f')
        assert_(out.dtype is numpy.dtype('f'))
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform_shift_via_grid_wrap(self, shift, order):
        # For mode 'grid-wrap', integer shifts should match numpy.roll
        x = numpy.array([[0, 1],
                         [2, 3]])
        affine = numpy.zeros((2, 3))
        affine[:2, :2] = numpy.eye(2)
        affine[:, 2] = shift
        assert_array_almost_equal(
            ndimage.affine_transform(x, affine, mode='grid-wrap', order=order),
            numpy.roll(x, shift, axis=(0, 1)),
        )

    @pytest.mark.parametrize('order', range(0, 6))
    def test_affine_transform_shift_reflect(self, order):
        # shift by x.shape results in reflection
        x = numpy.array([[0, 1, 2],
                         [3, 4, 5]])
        affine = numpy.zeros((2, 3))
        affine[:2, :2] = numpy.eye(2)
        affine[:, 2] = x.shape
        assert_array_almost_equal(
            ndimage.affine_transform(x, affine, mode='reflect', order=order),
            x[::-1, ::-1],
        )

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift01(self, order):
        data = numpy.array([1])
        out = ndimage.shift(data, [1], order=order)
        assert_array_almost_equal(out, [0])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift02(self, order):
        data = numpy.ones([4])
        out = ndimage.shift(data, [1], order=order)
        assert_array_almost_equal(out, [0, 1, 1, 1])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift03(self, order):
        data = numpy.ones([4])
        out = ndimage.shift(data, -1, order=order)
        assert_array_almost_equal(out, [1, 1, 1, 0])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift04(self, order):
        data = numpy.array([4, 1, 3, 2])
        out = ndimage.shift(data, 1, order=order)
        assert_array_almost_equal(out, [0, 4, 1, 3])

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_shift05(self, order, dtype):
        data = numpy.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]], dtype=dtype)
        expected = numpy.array([[0, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]], dtype=dtype)
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        out = ndimage.shift(data, [0, 1], order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('mode', ['constant', 'grid-constant'])
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_shift_with_nonzero_cval(self, order, mode, dtype):
        data = numpy.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]], dtype=dtype)

        expected = numpy.array([[0, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]], dtype=dtype)

        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        cval = 5.0
        expected[:, 0] = cval  # specific to shift of [0, 1] used below
        out = ndimage.shift(data, [0, 1], order=order, mode=mode, cval=cval)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift06(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        out = ndimage.shift(data, [0, 1], order=order)
        assert_array_almost_equal(out, [[0, 4, 1, 3],
                                        [0, 7, 6, 8],
                                        [0, 3, 5, 3]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift07(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        out = ndimage.shift(data, [1, 0], order=order)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [4, 1, 3, 2],
                                        [7, 6, 8, 5]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift08(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        out = ndimage.shift(data, [1, 1], order=order)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift09(self, order):
        data = numpy.array([[4, 1, 3, 2],
                            [7, 6, 8, 5],
                            [3, 5, 3, 6]])
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        out = ndimage.shift(filtered, [1, 1], order=order, prefilter=False)
        assert_array_almost_equal(out, [[0, 0, 0, 0],
                                        [0, 4, 1, 3],
                                        [0, 7, 6, 8]])

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift_grid_wrap(self, shift, order):
        # For mode 'grid-wrap', integer shifts should match numpy.roll
        x = numpy.array([[0, 1],
                         [2, 3]])
        assert_array_almost_equal(
            ndimage.shift(x, shift, mode='grid-wrap', order=order),
            numpy.roll(x, shift, axis=(0, 1)),
        )

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift_grid_constant1(self, shift, order):
        # For integer shifts, 'constant' and 'grid-constant' should be equal
        x = numpy.arange(20).reshape((5, 4))
        assert_array_almost_equal(
            ndimage.shift(x, shift, mode='grid-constant', order=order),
            ndimage.shift(x, shift, mode='constant', order=order),
        )

    def test_shift_grid_constant_order1(self):
        x = numpy.array([[1, 2, 3],
                         [4, 5, 6]], dtype=float)
        expected_result = numpy.array([[0.25, 0.75, 1.25],
                                       [1.25, 3.00, 4.00]])
        assert_array_almost_equal(
            ndimage.shift(x, (0.5, 0.5), mode='grid-constant', order=1),
            expected_result,
        )

    @pytest.mark.parametrize('order', range(0, 6))
    def test_shift_reflect(self, order):
        # shift by x.shape results in reflection
        x = numpy.array([[0, 1, 2],
                         [3, 4, 5]])
        assert_array_almost_equal(
            ndimage.shift(x, x.shape, mode='reflect', order=order),
            x[::-1, ::-1],
        )

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('prefilter', [False, True])
    def test_shift_nearest_boundary(self, order, prefilter):
        # verify that shifting at least order // 2 beyond the end of the array
        # gives a value equal to the edge value.
        x = numpy.arange(16)
        kwargs = dict(mode='nearest', order=order, prefilter=prefilter)
        assert_array_almost_equal(
            ndimage.shift(x, order // 2 + 1, **kwargs)[0], x[0],
        )
        assert_array_almost_equal(
            ndimage.shift(x, -order // 2 - 1, **kwargs)[-1], x[-1],
        )

    @pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest',
                                      'mirror', 'reflect'])
    @pytest.mark.parametrize('order', range(6))
    def test_shift_vs_padded(self, order, mode):
        x = numpy.arange(144, dtype=float).reshape(12, 12)
        shift = (0.4, -2.3)

        # manually pad and then extract center to get expected result
        npad = 32
        pad_mode = ndimage_to_numpy_mode.get(mode)
        xp = numpy.pad(x, npad, mode=pad_mode)
        center_slice = tuple([slice(npad, -npad)] * x.ndim)
        expected_result = ndimage.shift(
            xp, shift, mode=mode, order=order)[center_slice]

        assert_allclose(
            ndimage.shift(x, shift, mode=mode, order=order),
            expected_result,
            rtol=1e-7,
        )

    @pytest.mark.parametrize('order', range(0, 6))
    def test_zoom1(self, order):
        for z in [2, [2, 2]]:
            arr = numpy.array(list(range(25))).reshape((5, 5)).astype(float)
            arr = ndimage.zoom(arr, z, order=order)
            assert_equal(arr.shape, (10, 10))
            assert_(numpy.all(arr[-1, :] != 0))
            assert_(numpy.all(arr[-1, :] >= (20 - eps)))
            assert_(numpy.all(arr[0, :] <= (5 + eps)))
            assert_(numpy.all(arr >= (0 - eps)))
            assert_(numpy.all(arr <= (24 + eps)))

    def test_zoom2(self):
        arr = numpy.arange(12).reshape((3, 4))
        out = ndimage.zoom(ndimage.zoom(arr, 2), 0.5)
        assert_array_equal(out, arr)

    def test_zoom3(self):
        arr = numpy.array([[1, 2]])
        out1 = ndimage.zoom(arr, (2, 1))
        out2 = ndimage.zoom(arr, (1, 2))

        assert_array_almost_equal(out1, numpy.array([[1, 2], [1, 2]]))
        assert_array_almost_equal(out2, numpy.array([[1, 1, 2, 2]]))

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_zoom_affine01(self, order, dtype):
        data = numpy.asarray([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=dtype)
        if data.dtype.kind == 'c':
            data -= 1j * data
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed')
            out = ndimage.affine_transform(data, [0.5, 0.5], 0,
                                           (6, 8), order=order)
        assert_array_almost_equal(out[::2, ::2], data)

    def test_zoom_infinity(self):
        # Ticket #1419 regression test
        dim = 8
        ndimage.zoom(numpy.zeros((dim, dim)), 1. / dim, mode='nearest')

    def test_zoom_zoomfactor_one(self):
        # Ticket #1122 regression test
        arr = numpy.zeros((1, 5, 5))
        zoom = (1.0, 2.0, 2.0)

        out = ndimage.zoom(arr, zoom, cval=7)
        ref = numpy.zeros((1, 10, 10))
        assert_array_almost_equal(out, ref)

    def test_zoom_output_shape_roundoff(self):
        arr = numpy.zeros((3, 11, 25))
        zoom = (4.0 / 3, 15.0 / 11, 29.0 / 25)
        out = ndimage.zoom(arr, zoom)
        assert_array_equal(out.shape, (4, 15, 29))

    @pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
    @pytest.mark.parametrize('mode', ['nearest', 'constant', 'wrap', 'reflect',
                                      'mirror', 'grid-wrap', 'grid-mirror',
                                      'grid-constant'])
    def test_zoom_by_int_order0(self, zoom, mode):
        # order 0 zoom should be the same as replication via numpy.kron
        # Note: This is not True for general x shapes when grid_mode is False,
        #       but works here for all modes because the size ratio happens to
        #       always be an integer when x.shape = (2, 2).
        x = numpy.array([[0, 1],
                         [2, 3]], dtype=float)
        # x = numpy.arange(16, dtype=float).reshape(4, 4)
        assert_array_almost_equal(
            ndimage.zoom(x, zoom, order=0, mode=mode),
            numpy.kron(x, numpy.ones(zoom))
        )

    @pytest.mark.parametrize('shape', [(2, 3), (4, 4)])
    @pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
    @pytest.mark.parametrize('mode', ['nearest', 'reflect', 'mirror',
                                      'grid-wrap', 'grid-constant'])
    def test_zoom_grid_by_int_order0(self, shape, zoom, mode):
        # When grid_mode is True,  order 0 zoom should be the same as
        # replication via numpy.kron. The only exceptions to this are the
        # non-grid modes 'constant' and 'wrap'.
        x = numpy.arange(numpy.prod(shape), dtype=float).reshape(shape)
        assert_array_almost_equal(
            ndimage.zoom(x, zoom, order=0, mode=mode, grid_mode=True),
            numpy.kron(x, numpy.ones(zoom))
        )

    @pytest.mark.parametrize('mode', ['constant', 'wrap'])
    def test_zoom_grid_mode_warnings(self, mode):
        # Warn on use of non-grid modes when grid_mode is True
        x = numpy.arange(9, dtype=float).reshape((3, 3))
        with pytest.warns(UserWarning,
                          match="It is recommended to use mode"):
            ndimage.zoom(x, 2, mode=mode, grid_mode=True),

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate01(self, order):
        data = numpy.array([[0, 0, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 0, 0]], dtype=numpy.float64)
        out = ndimage.rotate(data, 0, order=order)
        assert_array_almost_equal(out, data)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate02(self, order):
        data = numpy.array([[0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]], dtype=numpy.float64)
        expected = numpy.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]], dtype=numpy.float64)
        out = ndimage.rotate(data, 90, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    @pytest.mark.parametrize('dtype', [numpy.float64, numpy.complex128])
    def test_rotate03(self, order, dtype):
        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=dtype)
        expected = numpy.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 0]], dtype=dtype)
        if data.dtype.kind == 'c':
            data -= 1j * data
            expected -= 1j * expected
        out = ndimage.rotate(data, 90, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate04(self, order):
        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]], dtype=numpy.float64)
        expected = numpy.array([[0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0]], dtype=numpy.float64)
        out = ndimage.rotate(data, 90, reshape=False, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate05(self, order):
        data = numpy.empty((4, 3, 3))
        for i in range(3):
            data[:, :, i] = numpy.array([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]], dtype=numpy.float64)
        expected = numpy.array([[0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0]], dtype=numpy.float64)
        out = ndimage.rotate(data, 90, order=order)
        for i in range(3):
            assert_array_almost_equal(out[:, :, i], expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate06(self, order):
        data = numpy.empty((3, 4, 3))
        for i in range(3):
            data[:, :, i] = numpy.array([[0, 0, 0, 0],
                                         [0, 1, 1, 0],
                                         [0, 0, 0, 0]], dtype=numpy.float64)
        expected = numpy.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=numpy.float64)
        out = ndimage.rotate(data, 90, order=order)
        for i in range(3):
            assert_array_almost_equal(out[:, :, i], expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate07(self, order):
        data = numpy.array([[[0, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0]]] * 2, dtype=numpy.float64)
        data = data.transpose()
        expected = numpy.array([[[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]]] * 2, dtype=numpy.float64)
        expected = expected.transpose([2, 1, 0])
        out = ndimage.rotate(data, 90, axes=(0, 1), order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', range(0, 6))
    def test_rotate08(self, order):
        data = numpy.array([[[0, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0]]] * 2, dtype=numpy.float64)
        data = data.transpose()
        expected = numpy.array([[[0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0]]] * 2, dtype=numpy.float64)
        expected = expected.transpose()
        out = ndimage.rotate(data, 90, axes=(0, 1), reshape=False, order=order)
        assert_array_almost_equal(out, expected)

    def test_rotate09(self):
        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]] * 2, dtype=numpy.float64)
        with assert_raises(ValueError):
            ndimage.rotate(data, 90, axes=(0, data.ndim))

    def test_rotate10(self):
        data = numpy.arange(45, dtype=numpy.float64).reshape((3, 5, 3))

        # The output of ndimage.rotate before refactoring
        expected = numpy.array([[[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [6.54914793, 7.54914793, 8.54914793],
                                 [10.84520162, 11.84520162, 12.84520162],
                                 [0.0, 0.0, 0.0]],
                                [[6.19286575, 7.19286575, 8.19286575],
                                 [13.4730712, 14.4730712, 15.4730712],
                                 [21.0, 22.0, 23.0],
                                 [28.5269288, 29.5269288, 30.5269288],
                                 [35.80713425, 36.80713425, 37.80713425]],
                                [[0.0, 0.0, 0.0],
                                 [31.15479838, 32.15479838, 33.15479838],
                                 [35.45085207, 36.45085207, 37.45085207],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]]])

        out = ndimage.rotate(data, angle=12, reshape=False)
        assert_array_almost_equal(out, expected)

    def test_rotate_exact_180(self):
        a = numpy.tile(numpy.arange(5), (5, 1))
        b = ndimage.rotate(ndimage.rotate(a, 180), -180)
        assert_equal(a, b)


def test_zoom_output_shape():
    """Ticket #643"""
    x = numpy.arange(12).reshape((3, 4))
    ndimage.zoom(x, 2, output=numpy.zeros((6, 8)))
