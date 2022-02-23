# -*- coding: utf-8 -*-
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import warnings

import pytest
from pytest import raises as assert_raises
from numpy.testing import (
    assert_equal,
    assert_almost_equal, assert_array_equal, assert_array_almost_equal,
    assert_allclose, assert_, assert_warns, assert_array_less,
    suppress_warnings)
from numpy import array, arange
import numpy as np

from scipy.fft import fft
from scipy.ndimage.filters import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
    correlate, correlation_lags, convolve, convolve2d,
    fftconvolve, oaconvolve, choose_conv_method,
    hilbert, hilbert2, lfilter, lfilter_zi, filtfilt, butter, zpk2tf, zpk2sos,
    invres, invresz, vectorstrength, lfiltic, tf2sos, sosfilt, sosfiltfilt,
    sosfilt_zi, tf2zpk, BadCoefficients, detrend, unique_roots, residue,
    residuez)
from scipy.signal.windows import hann
from scipy.signal.signaltools import (_filtfilt_gust, _compute_factors,
                                      _group_poles)
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils


class _TestConvolve:

    def test_basic(self):
        a = [3, 4, 5, 6, 5, 4]
        b = [1, 2, 3]
        c = convolve(a, b)
        assert_array_equal(c, array([3, 10, 22, 28, 32, 32, 23, 12]))

    def test_same(self):
        a = [3, 4, 5]
        b = [1, 2, 3, 4]
        c = convolve(a, b, mode="same")
        assert_array_equal(c, array([10, 22, 34]))

    def test_same_eq(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        c = convolve(a, b, mode="same")
        assert_array_equal(c, array([10, 22, 22]))

    def test_complex(self):
        x = array([1 + 1j, 2 + 1j, 3 + 1j])
        y = array([1 + 1j, 2 + 1j])
        z = convolve(x, y)
        assert_array_equal(z, array([2j, 2 + 6j, 5 + 8j, 5 + 5j]))

    def test_zero_rank(self):
        a = 1289
        b = 4567
        c = convolve(a, b)
        assert_equal(c, a * b)

    def test_broadcastable(self):
        a = np.arange(27).reshape(3, 3, 3)
        b = np.arange(3)
        for i in range(3):
            b_shape = [1]*3
            b_shape[i] = 3
            x = convolve(a, b.reshape(b_shape), method='direct')
            y = convolve(a, b.reshape(b_shape), method='fft')
            assert_allclose(x, y)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        c = convolve(a, b)
        assert_equal(c, a * b)

    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve(a, b)
        d = array([[2, 7, 16, 17, 12],
                   [10, 30, 62, 58, 38],
                   [12, 31, 58, 49, 30]])
        assert_array_equal(c, d)

    def test_input_swapping(self):
        small = arange(8).reshape(2, 2, 2)
        big = 1j * arange(27).reshape(3, 3, 3)
        big += arange(27)[::-1].reshape(3, 3, 3)

        out_array = array(
            [[[0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j],
              [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j],
              [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j],
              [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j]],

             [[104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j],
              [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j],
              [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j],
              [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j]],

             [[68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j],
              [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j],
              [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j],
              [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j]],

             [[32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j],
              [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j],
              [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j],
              [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j]]])

        assert_array_equal(convolve(small, big, 'full'), out_array)
        assert_array_equal(convolve(big, small, 'full'), out_array)
        assert_array_equal(convolve(small, big, 'same'),
                           out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'same'),
                           out_array[0:3, 0:3, 0:3])
        assert_array_equal(convolve(small, big, 'valid'),
                           out_array[1:3, 1:3, 1:3])
        assert_array_equal(convolve(big, small, 'valid'),
                           out_array[1:3, 1:3, 1:3])

    def test_invalid_params(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        assert_raises(ValueError, convolve, a, b, mode='spam')
        assert_raises(ValueError, convolve, a, b, mode='eggs', method='fft')
        assert_raises(ValueError, convolve, a, b, mode='ham', method='direct')
        assert_raises(ValueError, convolve, a, b, mode='full', method='bacon')
        assert_raises(ValueError, convolve, a, b, mode='same', method='bacon')


class TestConvolve(_TestConvolve):

    def test_valid_mode2(self):
        # See gh-5897
        a = [1, 2, 3, 6, 5, 3]
        b = [2, 3, 4, 5, 3, 4, 2, 2, 1]
        expected = [70, 78, 73, 65]

        out = convolve(a, b, 'valid')
        assert_array_equal(out, expected)

        out = convolve(b, a, 'valid')
        assert_array_equal(out, expected)

        a = [1 + 5j, 2 - 1j, 3 + 0j]
        b = [2 - 3j, 1 + 0j]
        expected = [2 - 3j, 8 - 10j]

        out = convolve(a, b, 'valid')
        assert_array_equal(out, expected)

        out = convolve(b, a, 'valid')
        assert_array_equal(out, expected)

    def test_same_mode(self):
        a = [1, 2, 3, 3, 1, 2]
        b = [1, 4, 3, 4, 5, 6, 7, 4, 3, 2, 1, 1, 3]
        c = convolve(a, b, 'same')
        d = array([57, 61, 63, 57, 45, 36])
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, convolve, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, convolve, *(b, a), **{'mode': 'valid'})

    def test_convolve_method(self, n=100):
        types = sum([t for _, t in np.sctypes.items()], [])
        types = {np.dtype(t).name for t in types}

        # These types include 'bool' and all precisions (int8, float32, etc)
        # The removed types throw errors in correlate or fftconvolve
        for dtype in ['complex256', 'complex192', 'float128', 'float96',
                      'str', 'void', 'bytes', 'object', 'unicode', 'string']:
            if dtype in types:
                types.remove(dtype)

        args = [(t1, t2, mode) for t1 in types for t2 in types
                               for mode in ['valid', 'full', 'same']]

        # These are random arrays, which means test is much stronger than
        # convolving testing by convolving two np.ones arrays
        np.random.seed(42)
        array_types = {'i': np.random.choice([0, 1], size=n),
                       'f': np.random.randn(n)}
        array_types['b'] = array_types['u'] = array_types['i']
        array_types['c'] = array_types['f'] + 0.5j*array_types['f']

        for t1, t2, mode in args:
            x1 = array_types[np.dtype(t1).kind].astype(t1)
            x2 = array_types[np.dtype(t2).kind].astype(t2)

            results = {key: convolve(x1, x2, method=key, mode=mode)
                       for key in ['fft', 'direct']}

            assert_equal(results['fft'].dtype, results['direct'].dtype)

            if 'bool' in t1 and 'bool' in t2:
                assert_equal(choose_conv_method(x1, x2), 'direct')
                continue

            # Found by experiment. Found approx smallest value for (rtol, atol)
            # threshold to have tests pass.
            if any([t in {'complex64', 'float32'} for t in [t1, t2]]):
                kwargs = {'rtol': 1.0e-4, 'atol': 1e-6}
            elif 'float16' in [t1, t2]:
                # atol is default for np.allclose
                kwargs = {'rtol': 1e-3, 'atol': 1e-3}
            else:
                # defaults for np.allclose (different from assert_allclose)
                kwargs = {'rtol': 1e-5, 'atol': 1e-8}

            assert_allclose(results['fft'], results['direct'], **kwargs)

    def test_convolve_method_large_input(self):
        # This is really a test that convolving two large integers goes to the
        # direct method even if they're in the fft method.
        for n in [10, 20, 50, 51, 52, 53, 54, 60, 62]:
            z = np.array([2**n], dtype=np.int64)
            fft = convolve(z, z, method='fft')
            direct = convolve(z, z, method='direct')

            # this is the case when integer precision gets to us
            # issue #6076 has more detail, hopefully more tests after resolved
            if n < 50:
                assert_equal(fft, direct)
                assert_equal(fft, 2**(2*n))
                assert_equal(direct, 2**(2*n))

    def test_mismatched_dims(self):
        # Input arrays should have the same number of dimensions
        assert_raises(ValueError, convolve, [1], 2, method='direct')
        assert_raises(ValueError, convolve, 1, [2], method='direct')
        assert_raises(ValueError, convolve, [1], 2, method='fft')
        assert_raises(ValueError, convolve, 1, [2], method='fft')
        assert_raises(ValueError, convolve, [1], [[2]])
        assert_raises(ValueError, convolve, [3], 2)


class _TestConvolve2d:

    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        d = array([[2, 7, 16, 17, 12],
                   [10, 30, 62, 58, 38],
                   [12, 31, 58, 49, 30]])
        e = convolve2d(a, b)
        assert_array_equal(e, d)

    def test_valid_mode(self):
        e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        f = [[1, 2, 3], [3, 4, 5]]
        h = array([[62, 80, 98, 116, 134]])

        g = convolve2d(e, f, 'valid')
        assert_array_equal(g, h)

        # See gh-5897
        g = convolve2d(f, e, 'valid')
        assert_array_equal(g, h)

    def test_valid_mode_complx(self):
        e = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        f = np.array([[1, 2, 3], [3, 4, 5]], dtype=complex) + 1j
        h = array([[62.+24.j, 80.+30.j, 98.+36.j, 116.+42.j, 134.+48.j]])

        g = convolve2d(e, f, 'valid')
        assert_array_almost_equal(g, h)

        # See gh-5897
        g = convolve2d(f, e, 'valid')
        assert_array_equal(g, h)

    def test_fillvalue(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        fillval = 1
        c = convolve2d(a, b, 'full', 'fill', fillval)
        d = array([[24, 26, 31, 34, 32],
                   [28, 40, 62, 64, 52],
                   [32, 46, 67, 62, 48]])
        assert_array_equal(c, d)

    def test_fillvalue_deprecations(self):
        # Deprecated 2017-07, scipy version 1.0.0
        with suppress_warnings() as sup:
            sup.filter(np.ComplexWarning, "Casting complex values to real")
            r = sup.record(DeprecationWarning, "could not cast `fillvalue`")
            convolve2d([[1]], [[1, 2]], fillvalue=1j)
            assert_(len(r) == 1)
            warnings.filterwarnings(
                "error", message="could not cast `fillvalue`",
                category=DeprecationWarning)
            assert_raises(DeprecationWarning, convolve2d, [[1]], [[1, 2]],
                          fillvalue=1j)

        with suppress_warnings():
            warnings.filterwarnings(
                "always", message="`fillvalue` must be scalar or an array ",
                category=DeprecationWarning)
            assert_warns(DeprecationWarning, convolve2d, [[1]], [[1, 2]],
                         fillvalue=[1, 2])
            warnings.filterwarnings(
                "error", message="`fillvalue` must be scalar or an array ",
                category=DeprecationWarning)
            assert_raises(DeprecationWarning, convolve2d, [[1]], [[1, 2]],
                          fillvalue=[1, 2])

    def test_fillvalue_empty(self):
        # Check that fillvalue being empty raises an error:
        assert_raises(ValueError, convolve2d, [[1]], [[1, 2]],
                      fillvalue=[])

    def test_wrap_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve2d(a, b, 'full', 'wrap')
        d = array([[80, 80, 74, 80, 80],
                   [68, 68, 62, 68, 68],
                   [80, 80, 74, 80, 80]])
        assert_array_equal(c, d)

    def test_sym_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve2d(a, b, 'full', 'symm')
        d = array([[34, 30, 44, 62, 66],
                   [52, 48, 62, 80, 84],
                   [82, 78, 92, 110, 114]])
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, convolve2d, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, convolve2d, *(b, a), **{'mode': 'valid'})


class TestConvolve2d(_TestConvolve2d):

    def test_same_mode(self):
        e = [[1, 2, 3], [3, 4, 5]]
        f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        g = convolve2d(e, f, 'same')
        h = array([[22, 28, 34],
                   [80, 98, 116]])
        assert_array_equal(g, h)

    def test_valid_mode2(self):
        # See gh-5897
        e = [[1, 2, 3], [3, 4, 5]]
        f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
        expected = [[62, 80, 98, 116, 134]]

        out = convolve2d(e, f, 'valid')
        assert_array_equal(out, expected)

        out = convolve2d(f, e, 'valid')
        assert_array_equal(out, expected)

        e = [[1 + 1j, 2 - 3j], [3 + 1j, 4 + 0j]]
        f = [[2 - 1j, 3 + 2j, 4 + 0j], [4 - 0j, 5 + 1j, 6 - 3j]]
        expected = [[27 - 1j, 46. + 2j]]

        out = convolve2d(e, f, 'valid')
        assert_array_equal(out, expected)

        # See gh-5897
        out = convolve2d(f, e, 'valid')
        assert_array_equal(out, expected)

    def test_consistency_convolve_funcs(self):
        # Compare np.convolve, signal.convolve, signal.convolve2d
        a = np.arange(5)
        b = np.array([3.2, 1.4, 3])
        for mode in ['full', 'valid', 'same']:
            assert_almost_equal(np.convolve(a, b, mode=mode),
                                signal.convolve(a, b, mode=mode))
            assert_almost_equal(np.squeeze(
                signal.convolve2d([a], [b], mode=mode)),
                signal.convolve(a, b, mode=mode))

    def test_invalid_dims(self):
        assert_raises(ValueError, convolve2d, 3, 4)
        assert_raises(ValueError, convolve2d, [3], [4])
        assert_raises(ValueError, convolve2d, [[[3]]], [[[4]]])

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit("Can't create large array for test")
    def test_large_array(self):
        # Test indexing doesn't overflow an int (gh-10761)
        n = 2**31 // (1000 * np.int64().itemsize)
        _testutils.check_free_memory(2 * n * 1001 * np.int64().itemsize / 1e6)

        # Create a chequered pattern of 1s and 0s
        a = np.zeros(1001 * n, dtype=np.int64)
        a[::2] = 1
        a = np.lib.stride_tricks.as_strided(a, shape=(n, 1000), strides=(8008, 8))

        count = signal.convolve2d(a, [[1, 1]])
        fails = np.where(count > 1)
        assert fails[0].size == 0


class TestFFTConvolve:

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_real(self, axes):
        a = array([1, 2, 3])
        expected = array([1, 4, 10, 12, 9.])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_real_axes(self, axes):
        a = array([1, 2, 3])
        expected = array([1, 4, 10, 12, 9.])

        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_complex(self, axes):
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_complex_axes(self, axes):
        a = array([1 + 1j, 2 + 2j, 3 + 3j])
        expected = array([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['',
                                      None,
                                      [0, 1],
                                      [1, 0],
                                      [0, -1],
                                      [-1, 0],
                                      [-2, 1],
                                      [1, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_real_same(self, axes):
        a = array([[1, 2, 3],
                   [4, 5, 6]])
        expected = array([[1, 4, 10, 12, 9],
                          [8, 26, 56, 54, 36],
                          [16, 40, 73, 60, 36]])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[1, 2],
                                      [2, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-2, 2],
                                      [2, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_real_same_axes(self, axes):
        a = array([[1, 2, 3],
                   [4, 5, 6]])
        expected = array([[1, 4, 10, 12, 9],
                          [8, 26, 56, 54, 36],
                          [16, 40, 73, 60, 36]])

        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['',
                                      None,
                                      [0, 1],
                                      [1, 0],
                                      [0, -1],
                                      [-1, 0],
                                      [-2, 1],
                                      [1, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_complex_same(self, axes):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j],
                   [2 + 1j, 4 + 3j, 6 + 5j]])
        expected = array([
            [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
            [10j, 44j, 118j, 156j, 122j],
            [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]
            ])

        if axes == '':
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[1, 2],
                                      [2, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-2, 2],
                                      [2, -2],
                                      [-2, -1],
                                      [-1, -2]])
    def test_2d_complex_same_axes(self, axes):
        a = array([[1 + 2j, 3 + 4j, 5 + 6j],
                   [2 + 1j, 4 + 3j, 6 + 5j]])
        expected = array([
            [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
            [10j, 44j, 118j, 156j, 122j],
            [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]
            ])

        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_real_same_mode(self, axes):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = array([35., 41., 47.])
        expected_2 = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])

        if axes == '':
            out = fftconvolve(a, b, 'same')
        else:
            out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)

        if axes == '':
            out = fftconvolve(b, a, 'same')
        else:
            out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', [1, -1, [1], [-1]])
    def test_real_same_mode_axes(self, axes):
        a = array([1, 2, 3])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = array([35., 41., 47.])
        expected_2 = array([9., 20., 25., 35., 41., 47., 39., 28., 2.])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected_1 = np.tile(expected_1, [2, 1])
        expected_2 = np.tile(expected_2, [2, 1])

        out = fftconvolve(a, b, 'same', axes=axes)
        assert_array_almost_equal(out, expected_1)

        out = fftconvolve(b, a, 'same', axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_real(self, axes):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1]])
    def test_valid_mode_real_axes(self, axes):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_valid_mode_complex(self, axes):
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        if axes == '':
            out = fftconvolve(a, b, 'valid')
        else:
            out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == '':
            out = fftconvolve(b, a, 'valid')
        else:
            out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_valid_mode_complex_axes(self, axes):
        a = array([3 - 1j, 2 + 7j, 1 + 0j])
        b = array([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = array([45. + 12.j, 30. + 23.j, 48 + 32.j])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

        out = fftconvolve(b, a, 'valid', axes=axes)
        assert_array_almost_equal(out, expected)

    def test_valid_mode_ignore_nonaxes(self):
        # See gh-5897
        a = array([3, 2, 1])
        b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = array([24., 31., 41., 43., 49., 25., 12.])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [1, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'valid', axes=1)
        assert_array_almost_equal(out, expected)

    def test_empty(self):
        # Regression test for #1745: crashes with 0-length input.
        assert_(fftconvolve([], []).size == 0)
        assert_(fftconvolve([5, 6], []).size == 0)
        assert_(fftconvolve([], [7]).size == 0)

    def test_zero_rank(self):
        a = array(4967)
        b = array(3920)
        out = fftconvolve(a, b)
        assert_equal(out, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        out = fftconvolve(a, b)
        assert_equal(out, a * b)

    @pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
    def test_random_data(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, 'full')

        if axes == '':
            out = fftconvolve(a, b, 'full')
        else:
            out = fftconvolve(a, b, 'full', axes=axes)
        assert_(np.allclose(out, expected, rtol=1e-10))

    @pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
    def test_random_data_axes(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, 'full')

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, 'full', axes=axes)
        assert_(np.allclose(out, expected, rtol=1e-10))

    @pytest.mark.parametrize('axes', [[1, 4],
                                      [4, 1],
                                      [1, -1],
                                      [-1, 1],
                                      [-4, 4],
                                      [4, -4],
                                      [-4, -1],
                                      [-1, -4]])
    def test_random_data_multidim_axes(self, axes):
        a_shape, b_shape = (123, 22), (132, 11)
        np.random.seed(1234)
        a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
        b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
        expected = convolve2d(a, b, 'full')

        a = a[:, :, None, None, None]
        b = b[:, :, None, None, None]
        expected = expected[:, :, None, None, None]

        a = np.rollaxis(a.swapaxes(0, 2), 1, 5)
        b = np.rollaxis(b.swapaxes(0, 2), 1, 5)
        expected = np.rollaxis(expected.swapaxes(0, 2), 1, 5)

        # use 1 for dimension 2 in a and 3 in b to test broadcasting
        a = np.tile(a, [2, 1, 3, 1, 1])
        b = np.tile(b, [2, 1, 1, 4, 1])
        expected = np.tile(expected, [2, 1, 3, 4, 1])

        out = fftconvolve(a, b, 'full', axes=axes)
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'n',
        list(range(1, 100)) +
        list(range(1000, 1500)) +
        np.random.RandomState(1234).randint(1001, 10000, 5).tolist())
    def test_many_sizes(self, n):
        a = np.random.rand(n) + 1j * np.random.rand(n)
        b = np.random.rand(n) + 1j * np.random.rand(n)
        expected = np.convolve(a, b, 'full')

        out = fftconvolve(a, b, 'full')
        assert_allclose(out, expected, atol=1e-10)

        out = fftconvolve(a, b, 'full', axes=[0])
        assert_allclose(out, expected, atol=1e-10)


def fftconvolve_err(*args, **kwargs):
    raise RuntimeError('Fell back to fftconvolve')


def gen_oa_shapes(sizes):
    return [(a, b) for a, b in product(sizes, repeat=2)
            if abs(a - b) > 3]


def gen_oa_shapes_2d(sizes):
    shapes0 = gen_oa_shapes(sizes)
    shapes1 = gen_oa_shapes(sizes)
    shapes = [ishapes0+ishapes1 for ishapes0, ishapes1 in
              zip(shapes0, shapes1)]

    modes = ['full', 'valid', 'same']
    return [ishapes+(imode,) for ishapes, imode in product(shapes, modes)
            if imode != 'valid' or
            (ishapes[0] > ishapes[1] and ishapes[2] > ishapes[3]) or
            (ishapes[0] < ishapes[1] and ishapes[2] < ishapes[3])]


def gen_oa_shapes_eq(sizes):
    return [(a, b) for a, b in product(sizes, repeat=2)
            if a >= b]


class TestOAConvolve:
    @pytest.mark.slow()
    @pytest.mark.parametrize('shape_a_0, shape_b_0',
                             gen_oa_shapes_eq(list(range(100)) +
                                              list(range(100, 1000, 23)))
                             )
    def test_real_manylens(self, shape_a_0, shape_b_0):
        a = np.random.rand(shape_a_0)
        b = np.random.rand(shape_b_0)

        expected = fftconvolve(a, b)
        out = oaconvolve(a, b)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('shape_a_0, shape_b_0',
                             gen_oa_shapes([50, 47, 6, 4, 1]))
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
    def test_1d_noaxes(self, shape_a_0, shape_b_0,
                       is_complex, mode, monkeypatch):
        a = np.random.rand(shape_a_0)
        b = np.random.rand(shape_b_0)
        if is_complex:
            a = a + 1j*np.random.rand(shape_a_0)
            b = b + 1j*np.random.rand(shape_b_0)

        expected = fftconvolve(a, b, mode=mode)

        monkeypatch.setattr(signal.signaltools, 'fftconvolve',
                            fftconvolve_err)
        out = oaconvolve(a, b, mode=mode)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [0, 1])
    @pytest.mark.parametrize('shape_a_0, shape_b_0',
                             gen_oa_shapes([50, 47, 6, 4]))
    @pytest.mark.parametrize('shape_a_extra', [1, 3])
    @pytest.mark.parametrize('shape_b_extra', [1, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    @pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
    def test_1d_axes(self, axes, shape_a_0, shape_b_0,
                     shape_a_extra, shape_b_extra,
                     is_complex, mode, monkeypatch):
        ax_a = [shape_a_extra]*2
        ax_b = [shape_b_extra]*2
        ax_a[axes] = shape_a_0
        ax_b[axes] = shape_b_0

        a = np.random.rand(*ax_a)
        b = np.random.rand(*ax_b)
        if is_complex:
            a = a + 1j*np.random.rand(*ax_a)
            b = b + 1j*np.random.rand(*ax_b)

        expected = fftconvolve(a, b, mode=mode, axes=axes)

        monkeypatch.setattr(signal.signaltools, 'fftconvolve',
                            fftconvolve_err)
        out = oaconvolve(a, b, mode=mode, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('shape_a_0, shape_b_0, '
                             'shape_a_1, shape_b_1, mode',
                             gen_oa_shapes_2d([50, 47, 6, 4]))
    @pytest.mark.parametrize('is_complex', [True, False])
    def test_2d_noaxes(self, shape_a_0, shape_b_0,
                       shape_a_1, shape_b_1, mode,
                       is_complex, monkeypatch):
        a = np.random.rand(shape_a_0, shape_a_1)
        b = np.random.rand(shape_b_0, shape_b_1)
        if is_complex:
            a = a + 1j*np.random.rand(shape_a_0, shape_a_1)
            b = b + 1j*np.random.rand(shape_b_0, shape_b_1)

        expected = fftconvolve(a, b, mode=mode)

        monkeypatch.setattr(signal.signaltools, 'fftconvolve',
                            fftconvolve_err)
        out = oaconvolve(a, b, mode=mode)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('axes', [[0, 1], [0, 2], [1, 2]])
    @pytest.mark.parametrize('shape_a_0, shape_b_0, '
                             'shape_a_1, shape_b_1, mode',
                             gen_oa_shapes_2d([50, 47, 6, 4]))
    @pytest.mark.parametrize('shape_a_extra', [1, 3])
    @pytest.mark.parametrize('shape_b_extra', [1, 3])
    @pytest.mark.parametrize('is_complex', [True, False])
    def test_2d_axes(self, axes, shape_a_0, shape_b_0,
                     shape_a_1, shape_b_1, mode,
                     shape_a_extra, shape_b_extra,
                     is_complex, monkeypatch):
        ax_a = [shape_a_extra]*3
        ax_b = [shape_b_extra]*3
        ax_a[axes[0]] = shape_a_0
        ax_b[axes[0]] = shape_b_0
        ax_a[axes[1]] = shape_a_1
        ax_b[axes[1]] = shape_b_1

        a = np.random.rand(*ax_a)
        b = np.random.rand(*ax_b)
        if is_complex:
            a = a + 1j*np.random.rand(*ax_a)
            b = b + 1j*np.random.rand(*ax_b)

        expected = fftconvolve(a, b, mode=mode, axes=axes)

        monkeypatch.setattr(signal.signaltools, 'fftconvolve',
                            fftconvolve_err)
        out = oaconvolve(a, b, mode=mode, axes=axes)

        assert_array_almost_equal(out, expected)

    def test_empty(self):
        # Regression test for #1745: crashes with 0-length input.
        assert_(oaconvolve([], []).size == 0)
        assert_(oaconvolve([5, 6], []).size == 0)
        assert_(oaconvolve([], [7]).size == 0)

    def test_zero_rank(self):
        a = array(4967)
        b = array(3920)
        out = oaconvolve(a, b)
        assert_equal(out, a * b)

    def test_single_element(self):
        a = array([4967])
        b = array([3920])
        out = oaconvolve(a, b)
        assert_equal(out, a * b)


class TestAllFreqConvolves:

    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    def test_invalid_shapes(self, convapproach):
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))
        with assert_raises(ValueError,
                           match="For 'valid' mode, one must be at least "
                           "as large as the other in every dimension"):
            convapproach(a, b, mode='valid')

    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    def test_invalid_shapes_axes(self, convapproach):
        a = np.zeros([5, 6, 2, 1])
        b = np.zeros([5, 6, 3, 1])
        with assert_raises(ValueError,
                           match=r"incompatible shapes for in1 and in2:"
                           r" \(5L?, 6L?, 2L?, 1L?\) and"
                           r" \(5L?, 6L?, 3L?, 1L?\)"):
            convapproach(a, b, axes=[0, 1])

    @pytest.mark.parametrize('a,b',
                             [([1], 2),
                              (1, [2]),
                              ([3], [[2]])])
    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    def test_mismatched_dims(self, a, b, convapproach):
        with assert_raises(ValueError,
                           match="in1 and in2 should have the same"
                           " dimensionality"):
            convapproach(a, b)

    @pytest.mark.parametrize('convapproach',
                             [fftconvolve, oaconvolve])
    def test_invalid_flags(self, convapproach):
        with assert_raises(ValueError,
                           match="acceptable mode flags are 'valid',"
                           " 'same', or 'full'"):
            convapproach([1], [2], mode='chips')

        with assert_raises(ValueError,
                           match="when provided, axes cannot be empty"):
            convapproach([1], [2], axes=[])

        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            convapproach([1], [2], axes=[[1, 2], [3, 4]])

        with assert_raises(ValueError, match="axes must be a scalar or "
                           "iterable of integers"):
            convapproach([1], [2], axes=[1., 2., 3., 4.])

        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            convapproach([1], [2], axes=[1])

        with assert_raises(ValueError,
                           match="axes exceeds dimensionality of input"):
            convapproach([1], [2], axes=[-2])

        with assert_raises(ValueError,
                           match="all axes must be unique"):
            convapproach([1], [2], axes=[0, 0])

    @pytest.mark.parametrize('dtype', [np.longfloat, np.longcomplex])
    def test_longdtype_input(self, dtype):
        x = np.random.random((27, 27)).astype(dtype)
        y = np.random.random((4, 4)).astype(dtype)
        if np.iscomplexobj(dtype()):
            x += .1j
            y -= .1j

        res = fftconvolve(x, y)
        assert_allclose(res, convolve(x, y, method='direct'))
        assert res.dtype == dtype


class TestMedFilt:

    IN = [[50, 50, 50, 50, 50, 92, 18, 27, 65, 46],
          [50, 50, 50, 50, 50, 0, 72, 77, 68, 66],
          [50, 50, 50, 50, 50, 46, 47, 19, 64, 77],
          [50, 50, 50, 50, 50, 42, 15, 29, 95, 35],
          [50, 50, 50, 50, 50, 46, 34, 9, 21, 66],
          [70, 97, 28, 68, 78, 77, 61, 58, 71, 42],
          [64, 53, 44, 29, 68, 32, 19, 68, 24, 84],
          [3, 33, 53, 67, 1, 78, 74, 55, 12, 83],
          [7, 11, 46, 70, 60, 47, 24, 43, 61, 26],
          [32, 61, 88, 7, 39, 4, 92, 64, 45, 61]]

    OUT = [[0, 50, 50, 50, 42, 15, 15, 18, 27, 0],
           [0, 50, 50, 50, 50, 42, 19, 21, 29, 0],
           [50, 50, 50, 50, 50, 47, 34, 34, 46, 35],
           [50, 50, 50, 50, 50, 50, 42, 47, 64, 42],
           [50, 50, 50, 50, 50, 50, 46, 55, 64, 35],
           [33, 50, 50, 50, 50, 47, 46, 43, 55, 26],
           [32, 50, 50, 50, 50, 47, 46, 45, 55, 26],
           [7, 46, 50, 50, 47, 46, 46, 43, 45, 21],
           [0, 32, 33, 39, 32, 32, 43, 43, 43, 0],
           [0, 7, 11, 7, 4, 4, 19, 19, 24, 0]]

    KERNEL_SIZE = [7,3]

    def test_basic(self):
        d = signal.medfilt(self.IN, self.KERNEL_SIZE)
        e = signal.medfilt2d(np.array(self.IN, float), self.KERNEL_SIZE)
        assert_array_equal(d, self.OUT)
        assert_array_equal(d, e)

    @pytest.mark.parametrize('dtype', [np.ubyte, np.byte, np.ushort, np.short,
                                       np.uint, int, np.ulonglong, np.ulonglong,
                                       np.float32, np.float64, np.longdouble])
    def test_types(self, dtype):
        # volume input and output types match
        in_typed = np.array(self.IN, dtype=dtype)
        assert_equal(signal.medfilt(in_typed).dtype, dtype)
        assert_equal(signal.medfilt2d(in_typed).dtype, dtype)

    @pytest.mark.parametrize('dtype', [np.bool_, np.cfloat, np.cdouble,
                                       np.clongdouble, np.float16,])
    def test_invalid_dtypes(self, dtype):
        in_typed = np.array(self.IN, dtype=dtype)
        with pytest.raises(ValueError, match="order_filterND"):
            signal.medfilt(in_typed)

        with pytest.raises(ValueError, match="order_filterND"):
            signal.medfilt2d(in_typed)


    def test_none(self):
        # gh-1651, trac #1124. Ensure this does not segfault.
        with pytest.warns(UserWarning):
            assert_raises(TypeError, signal.medfilt, None)
        # Expand on this test to avoid a regression with possible contiguous
        # numpy arrays that have odd strides. The stride value below gets
        # us into wrong memory if used (but it does not need to be used)
        dummy = np.arange(10, dtype=np.float64)
        a = dummy[5:6]
        a.strides = 16
        assert_(signal.medfilt(a, 1) == 5.)

    def test_refcounting(self):
        # Check a refcounting-related crash
        a = Decimal(123)
        x = np.array([a, a], dtype=object)
        if hasattr(sys, 'getrefcount'):
            n = 2 * sys.getrefcount(a)
        else:
            n = 10
        # Shouldn't segfault:
        with pytest.warns(UserWarning):
            for j in range(n):
                signal.medfilt(x)
        if hasattr(sys, 'getrefcount'):
            assert_(sys.getrefcount(a) < n)
        assert_equal(x, [a, a])

    def test_object(self,):
        in_object = np.array(self.IN, dtype=object)
        out_object = np.array(self.OUT, dtype=object)
        assert_array_equal(signal.medfilt(in_object, self.KERNEL_SIZE),
                           out_object)

    @pytest.mark.parametrize("dtype", [np.ubyte, np.float32, np.float64])
    def test_medfilt2d_parallel(self, dtype):
        in_typed = np.array(self.IN, dtype=dtype)
        expected = np.array(self.OUT, dtype=dtype)

        # This is used to simplify the indexing calculations.
        assert in_typed.shape == expected.shape

        # We'll do the calculation in four chunks. M1 and N1 are the dimensions
        # of the first output chunk. We have to extend the input by half the
        # kernel size to be able to calculate the full output chunk.
        M1 = expected.shape[0] // 2
        N1 = expected.shape[1] // 2
        offM = self.KERNEL_SIZE[0] // 2 + 1
        offN = self.KERNEL_SIZE[1] // 2 + 1

        def apply(chunk):
            # in = slice of in_typed to use.
            # sel = slice of output to crop it to the correct region.
            # out = slice of output array to store in.
            M, N = chunk
            if M == 0:
                Min = slice(0, M1 + offM)
                Msel = slice(0, -offM)
                Mout = slice(0, M1)
            else:
                Min = slice(M1 - offM, None)
                Msel = slice(offM, None)
                Mout = slice(M1, None)
            if N == 0:
                Nin = slice(0, N1 + offN)
                Nsel = slice(0, -offN)
                Nout = slice(0, N1)
            else:
                Nin = slice(N1 - offN, None)
                Nsel = slice(offN, None)
                Nout = slice(N1, None)

            # Do the calculation, but do not write to the output in the threads.
            chunk_data = in_typed[Min, Nin]
            med = signal.medfilt2d(chunk_data, self.KERNEL_SIZE)
            return med[Msel, Nsel], Mout, Nout

        # Give each chunk to a different thread.
        output = np.zeros_like(expected)
        with ThreadPoolExecutor(max_workers=4) as pool:
            chunks = {(0, 0), (0, 1), (1, 0), (1, 1)}
            futures = {pool.submit(apply, chunk) for chunk in chunks}

            # Store each result in the output as it arrives.
            for future in as_completed(futures):
                data, Mslice, Nslice = future.result()
                output[Mslice, Nslice] = data

        assert_array_equal(output, expected)


class TestWiener:

    def test_basic(self):
        g = array([[5, 6, 4, 3],
                   [3, 5, 6, 2],
                   [2, 3, 5, 6],
                   [1, 6, 9, 7]], 'd')
        h = array([[2.16374269, 3.2222222222, 2.8888888889, 1.6666666667],
                   [2.666666667, 4.33333333333, 4.44444444444, 2.8888888888],
                   [2.222222222, 4.4444444444, 5.4444444444, 4.801066874837],
                   [1.33333333333, 3.92735042735, 6.0712560386, 5.0404040404]])
        assert_array_almost_equal(signal.wiener(g), h, decimal=6)
        assert_array_almost_equal(signal.wiener(g, mysize=3), h, decimal=6)


padtype_options = ["mean", "median", "minimum", "maximum", "line"]
padtype_options += _upfirdn_modes


class TestResample:
    def test_basic(self):
        # Some basic tests

        # Regression test for issue #3603.
        # window.shape must equal to sig.shape[0]
        sig = np.arange(128)
        num = 256
        win = signal.get_window(('kaiser', 8.0), 160)
        assert_raises(ValueError, signal.resample, sig, num, window=win)

        # Other degenerate conditions
        assert_raises(ValueError, signal.resample_poly, sig, 'yo', 1)
        assert_raises(ValueError, signal.resample_poly, sig, 1, 0)
        assert_raises(ValueError, signal.resample_poly, sig, 2, 1, padtype='')
        assert_raises(ValueError, signal.resample_poly, sig, 2, 1,
                      padtype='mean', cval=10)

        # test for issue #6505 - should not modify window.shape when axis ≠ 0
        sig2 = np.tile(np.arange(160), (2, 1))
        signal.resample(sig2, num, axis=-1, window=win)
        assert_(win.shape == (160,))

    @pytest.mark.parametrize('window', (None, 'hamming'))
    @pytest.mark.parametrize('N', (20, 19))
    @pytest.mark.parametrize('num', (100, 101, 10, 11))
    def test_rfft(self, N, num, window):
        # Make sure the speed up using rfft gives the same result as the normal
        # way using fft
        x = np.linspace(0, 10, N, endpoint=False)
        y = np.cos(-x**2/6.0)
        assert_allclose(signal.resample(y, num, window=window),
                        signal.resample(y + 0j, num, window=window).real)

        y = np.array([np.cos(-x**2/6.0), np.sin(-x**2/6.0)])
        y_complex = y + 0j
        assert_allclose(
            signal.resample(y, num, axis=1, window=window),
            signal.resample(y_complex, num, axis=1, window=window).real,
            atol=1e-9)

    def test_input_domain(self):
        # Test if both input domain modes produce the same results.
        tsig = np.arange(256) + 0j
        fsig = fft(tsig)
        num = 256
        assert_allclose(
            signal.resample(fsig, num, domain='freq'),
            signal.resample(tsig, num, domain='time'),
            atol=1e-9)

    @pytest.mark.parametrize('nx', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('ny', (1, 2, 3, 5, 8))
    @pytest.mark.parametrize('dtype', ('float', 'complex'))
    def test_dc(self, nx, ny, dtype):
        x = np.array([1] * nx, dtype)
        y = signal.resample(x, ny)
        assert_allclose(y, [1] * ny)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_mutable_window(self, padtype):
        # Test that a mutable window is not modified
        impulse = np.zeros(3)
        window = np.random.RandomState(0).randn(2)
        window_orig = window.copy()
        signal.resample_poly(impulse, 5, 1, window=window, padtype=padtype)
        assert_array_equal(window, window_orig)

    @pytest.mark.parametrize('padtype', padtype_options)
    def test_output_float32(self, padtype):
        # Test that float32 inputs yield a float32 output
        x = np.arange(10, dtype=np.float32)
        h = np.array([1, 1, 1], dtype=np.float32)
        y = signal.resample_poly(x, 1, 2, window=h, padtype=padtype)
        assert(y.dtype == np.float32)

    @pytest.mark.parametrize(
        "method, ext, padtype",
        [("fft", False, None)]
        + list(
            product(
                ["polyphase"], [False, True], padtype_options,
            )
        ),
    )
    def test_resample_methods(self, method, ext, padtype):
        # Test resampling of sinusoids and random noise (1-sec)
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]

        # Sinusoids, windowed to avoid edge artifacts
        t = np.arange(rate) / float(rate)
        freqs = np.array((1., 10., 40.))[:, np.newaxis]
        x = np.sin(2 * np.pi * freqs * t) * hann(rate)

        for rate_to in rates_to:
            t_to = np.arange(rate_to) / float(rate_to)
            y_tos = np.sin(2 * np.pi * freqs * t_to) * hann(rate_to)
            if method == 'fft':
                y_resamps = signal.resample(x, rate_to, axis=-1)
            else:
                if ext and rate_to != rate:
                    # Match default window design
                    g = gcd(rate_to, rate)
                    up = rate_to // g
                    down = rate // g
                    max_rate = max(up, down)
                    f_c = 1. / max_rate
                    half_len = 10 * max_rate
                    window = signal.firwin(2 * half_len + 1, f_c,
                                           window=('kaiser', 5.0))
                    polyargs = {'window': window, 'padtype': padtype}
                else:
                    polyargs = {'padtype': padtype}

                y_resamps = signal.resample_poly(x, rate_to, rate, axis=-1,
                                                 **polyargs)

            for y_to, y_resamp, freq in zip(y_tos, y_resamps, freqs):
                if freq >= 0.5 * rate_to:
                    y_to.fill(0.)  # mostly low-passed away
                    if padtype in ['minimum', 'maximum']:
                        assert_allclose(y_resamp, y_to, atol=3e-1)
                    else:
                        assert_allclose(y_resamp, y_to, atol=1e-3)
                else:
                    assert_array_equal(y_to.shape, y_resamp.shape)
                    corr = np.corrcoef(y_to, y_resamp)[0, 1]
                    assert_(corr > 0.99, msg=(corr, rate, rate_to))

        # Random data
        rng = np.random.RandomState(0)
        x = hann(rate) * np.cumsum(rng.randn(rate))  # low-pass, wind
        for rate_to in rates_to:
            # random data
            t_to = np.arange(rate_to) / float(rate_to)
            y_to = np.interp(t_to, t, x)
            if method == 'fft':
                y_resamp = signal.resample(x, rate_to)
            else:
                y_resamp = signal.resample_poly(x, rate_to, rate,
                                                padtype=padtype)
            assert_array_equal(y_to.shape, y_resamp.shape)
            corr = np.corrcoef(y_to, y_resamp)[0, 1]
            assert_(corr > 0.99, msg=corr)

        # More tests of fft method (Master 0.18.1 fails these)
        if method == 'fft':
            x1 = np.array([1.+0.j, 0.+0.j])
            y1_test = signal.resample(x1, 4)
            # upsampling a complex array
            y1_true = np.array([1.+0.j, 0.5+0.j, 0.+0.j, 0.5+0.j])
            assert_allclose(y1_test, y1_true, atol=1e-12)
            x2 = np.array([1., 0.5, 0., 0.5])
            y2_test = signal.resample(x2, 2)  # downsampling a real array
            y2_true = np.array([1., 0.])
            assert_allclose(y2_test, y2_true, atol=1e-12)

    def test_poly_vs_filtfilt(self):
        # Check that up=1.0 gives same answer as filtfilt + slicing
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)

            # resample_poly assumes zeros outside of signl, whereas filtfilt
            # can only constant-pad. Make them equivalent:
            x[0] = 0
            x[-1] = 0

            for down in down_factors:
                h = signal.firwin(31, 1. / down, window='hamming')
                yf = filtfilt(h, 1.0, x, padtype='constant')[::down]

                # Need to pass convolved version of filter to resample_poly,
                # since filtfilt does forward and backward, but resample_poly
                # only goes forward
                hc = convolve(h, h[::-1])
                y = signal.resample_poly(x, 1, down, window=hc)
                assert_allclose(yf, y, atol=1e-7, rtol=1e-7)

    def test_correlate1d(self):
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = np.random.random((nx,))
                    weights = np.random.random((nweights,))
                    y_g = correlate1d(x, weights[::-1], mode='constant')
                    y_s = signal.resample_poly(
                        x, up=1, down=down, window=weights)
                    assert_allclose(y_g[::down], y_s)


class TestCSpline1DEval:

    def test_basic(self):
        y = array([1, 2, 3, 4, 3, 2, 1, 2, 3.0])
        x = arange(len(y))
        dx = x[1] - x[0]
        cj = signal.cspline1d(y)

        x2 = arange(len(y) * 10.0) / 10.0
        y2 = signal.cspline1d_eval(cj, x2, dx=dx, x0=x[0])

        # make sure interpolated values are on knot points
        assert_array_almost_equal(y2[::10], y, decimal=5)

    def test_complex(self):
        #  create some smoothly varying complex signal to interpolate
        x = np.arange(2)
        y = np.zeros(x.shape, dtype=np.complex64)
        T = 10.0
        f = 1.0 / T
        y = np.exp(2.0J * np.pi * f * x)

        # get the cspline transform
        cy = signal.cspline1d(y)

        # determine new test x value and interpolate
        xnew = np.array([0.5])
        ynew = signal.cspline1d_eval(cy, xnew)

        assert_equal(ynew.dtype, y.dtype)

class TestOrderFilt:

    def test_basic(self):
        assert_array_equal(signal.order_filter([1, 2, 3], [1, 0, 1], 1),
                           [2, 3, 2])


class _TestLinearFilter:

    def generate(self, shape):
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        return self.convert_dtype(x)

    def convert_dtype(self, arr):
        if self.dtype == np.dtype('O'):
            arr = np.asarray(arr)
            out = np.empty(arr.shape, self.dtype)
            iter = np.nditer([arr, out], ['refs_ok','zerosize_ok'],
                        [['readonly'],['writeonly']])
            for x, y in iter:
                y[...] = self.type(x[()])
            return out
        else:
            return np.array(arr, self.dtype, copy=False)

    def test_rank_1_IIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, -0.5])
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_FIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 3, 5, 7, 9.])
        assert_array_almost_equal(lfilter(b, a, x), y_r)

    def test_rank_1_IIR_init_cond(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([0.5, -0.5])
        zi = self.convert_dtype([1, 2])
        y_r = self.convert_dtype([1, 5, 9, 13, 17, 21])
        zf_r = self.convert_dtype([13, -10])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_1_FIR_init_cond(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 1, 1])
        a = self.convert_dtype([1])
        zi = self.convert_dtype([1, 1])
        y_r = self.convert_dtype([1, 2, 3, 6, 9, 12.])
        zf_r = self.convert_dtype([9, 5])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, y_r)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_2_IIR_axis_0(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        y_r2_a0 = self.convert_dtype([[0, 2, 4], [6, 4, 2], [0, 2, 4],
                                      [6, 4, 2]])
        y = lfilter(b, a, x, axis=0)
        assert_array_almost_equal(y_r2_a0, y)

    def test_rank_2_IIR_axis_1(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        y_r2_a1 = self.convert_dtype([[0, 2, 0], [6, -4, 6], [12, -10, 12],
                            [18, -16, 18]])
        y = lfilter(b, a, x, axis=1)
        assert_array_almost_equal(y_r2_a1, y)

    def test_rank_2_IIR_axis_0_init_cond(self):
        x = self.generate((4, 3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        zi = self.convert_dtype(np.ones((4,1)))

        y_r2_a0_1 = self.convert_dtype([[1, 1, 1], [7, -5, 7], [13, -11, 13],
                              [19, -17, 19]])
        zf_r = self.convert_dtype([-5, -17, -29, -41])[:, np.newaxis]
        y, zf = lfilter(b, a, x, axis=1, zi=zi)
        assert_array_almost_equal(y_r2_a0_1, y)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_2_IIR_axis_1_init_cond(self):
        x = self.generate((4,3))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])
        zi = self.convert_dtype(np.ones((1,3)))

        y_r2_a0_0 = self.convert_dtype([[1, 3, 5], [5, 3, 1],
                                        [1, 3, 5], [5, 3, 1]])
        zf_r = self.convert_dtype([[-23, -23, -23]])
        y, zf = lfilter(b, a, x, axis=0, zi=zi)
        assert_array_almost_equal(y_r2_a0_0, y)
        assert_array_almost_equal(zf, zf_r)

    def test_rank_3_IIR(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])

        for axis in range(x.ndim):
            y = lfilter(b, a, x, axis)
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            assert_array_almost_equal(y, y_r)

    def test_rank_3_IIR_init_cond(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, -1])
        a = self.convert_dtype([0.5, 0.5])

        for axis in range(x.ndim):
            zi_shape = list(x.shape)
            zi_shape[axis] = 1
            zi = self.convert_dtype(np.ones(zi_shape))
            zi1 = self.convert_dtype([1])
            y, zf = lfilter(b, a, x, axis, zi)
            lf0 = lambda w: lfilter(b, a, w, zi=zi1)[0]
            lf1 = lambda w: lfilter(b, a, w, zi=zi1)[1]
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            assert_array_almost_equal(y, y_r)
            assert_array_almost_equal(zf, zf_r)

    def test_rank_3_FIR(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])

        for axis in range(x.ndim):
            y = lfilter(b, a, x, axis)
            y_r = np.apply_along_axis(lambda w: lfilter(b, a, w), axis, x)
            assert_array_almost_equal(y, y_r)

    def test_rank_3_FIR_init_cond(self):
        x = self.generate((4, 3, 2))
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])

        for axis in range(x.ndim):
            zi_shape = list(x.shape)
            zi_shape[axis] = 2
            zi = self.convert_dtype(np.ones(zi_shape))
            zi1 = self.convert_dtype([1, 1])
            y, zf = lfilter(b, a, x, axis, zi)
            lf0 = lambda w: lfilter(b, a, w, zi=zi1)[0]
            lf1 = lambda w: lfilter(b, a, w, zi=zi1)[1]
            y_r = np.apply_along_axis(lf0, axis, x)
            zf_r = np.apply_along_axis(lf1, axis, x)
            assert_array_almost_equal(y, y_r)
            assert_array_almost_equal(zf, zf_r)

    def test_zi_pseudobroadcast(self):
        x = self.generate((4, 5, 20))
        b,a = signal.butter(8, 0.2, output='ba')
        b = self.convert_dtype(b)
        a = self.convert_dtype(a)
        zi_size = b.shape[0] - 1

        # lfilter requires x.ndim == zi.ndim exactly.  However, zi can have
        # length 1 dimensions.
        zi_full = self.convert_dtype(np.ones((4, 5, zi_size)))
        zi_sing = self.convert_dtype(np.ones((1, 1, zi_size)))

        y_full, zf_full = lfilter(b, a, x, zi=zi_full)
        y_sing, zf_sing = lfilter(b, a, x, zi=zi_sing)

        assert_array_almost_equal(y_sing, y_full)
        assert_array_almost_equal(zf_full, zf_sing)

        # lfilter does not prepend ones
        assert_raises(ValueError, lfilter, b, a, x, -1, np.ones(zi_size))

    def test_scalar_a(self):
        # a can be a scalar.
        x = self.generate(6)
        b = self.convert_dtype([1, 0, -1])
        a = self.convert_dtype([1])
        y_r = self.convert_dtype([0, 1, 2, 2, 2, 2])

        y = lfilter(b, a[0], x)
        assert_array_almost_equal(y, y_r)

    def test_zi_some_singleton_dims(self):
        # lfilter doesn't really broadcast (no prepending of 1's).  But does
        # do singleton expansion if x and zi have the same ndim.  This was
        # broken only if a subset of the axes were singletons (gh-4681).
        x = self.convert_dtype(np.zeros((3,2,5), 'l'))
        b = self.convert_dtype(np.ones(5, 'l'))
        a = self.convert_dtype(np.array([1,0,0]))
        zi = np.ones((3,1,4), 'l')
        zi[1,:,:] *= 2
        zi[2,:,:] *= 3
        zi = self.convert_dtype(zi)

        zf_expected = self.convert_dtype(np.zeros((3,2,4), 'l'))
        y_expected = np.zeros((3,2,5), 'l')
        y_expected[:,:,:4] = [[[1]], [[2]], [[3]]]
        y_expected = self.convert_dtype(y_expected)

        # IIR
        y_iir, zf_iir = lfilter(b, a, x, -1, zi)
        assert_array_almost_equal(y_iir, y_expected)
        assert_array_almost_equal(zf_iir, zf_expected)

        # FIR
        y_fir, zf_fir = lfilter(b, a[0], x, -1, zi)
        assert_array_almost_equal(y_fir, y_expected)
        assert_array_almost_equal(zf_fir, zf_expected)

    def base_bad_size_zi(self, b, a, x, axis, zi):
        b = self.convert_dtype(b)
        a = self.convert_dtype(a)
        x = self.convert_dtype(x)
        zi = self.convert_dtype(zi)
        assert_raises(ValueError, lfilter, b, a, x, axis, zi)

    def test_bad_size_zi(self):
        # rank 1
        x1 = np.arange(6)
        self.base_bad_size_zi([1], [1], x1, -1, [1])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [[0]])
        self.base_bad_size_zi([1, 1], [1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [[0]])
        self.base_bad_size_zi([1, 1, 1], [1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [[0]])
        self.base_bad_size_zi([1], [1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [[0], [1]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x1, -1, [0, 1, 2, 3])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [[0], [1]])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2])
        self.base_bad_size_zi([1, 1], [1, 1, 1], x1, -1, [0, 1, 2, 3])

        # rank 2
        x2 = np.arange(12).reshape((4,3))
        # for axis=0 zi.shape should == (max(len(a),len(b))-1, 3)
        self.base_bad_size_zi([1], [1], x2, 0, [0])

        # for each of these there are 5 cases tested (in this order):
        # 1. not deep enough, right # elements
        # 2. too deep, right # elements
        # 3. right depth, right # elements, transposed
        # 4. right depth, too few elements
        # 5. right depth, too many elements

        self.base_bad_size_zi([1, 1], [1], x2, 0, [0,1,2])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[[0,1,2]]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0], [1], [2]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0,1]])
        self.base_bad_size_zi([1, 1], [1], x2, 0, [[0,1,2,3]])

        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [0,1,2,3,4,5])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[[0,1,2],[3,4,5]]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0,1],[2,3]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 0, [[0,1,2,3],[4,5,6,7]])

        self.base_bad_size_zi([1], [1, 1], x2, 0, [0,1,2])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[[0,1,2]]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0], [1], [2]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0,1]])
        self.base_bad_size_zi([1], [1, 1], x2, 0, [[0,1,2,3]])

        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [0,1,2,3,4,5])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[[0,1,2],[3,4,5]]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0,1],[2,3]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 0, [[0,1,2,3],[4,5,6,7]])

        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [0,1,2,3,4,5])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[[0,1,2],[3,4,5]]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0,1],[2,3]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 0, [[0,1,2,3],[4,5,6,7]])

        # for axis=1 zi.shape should == (4, max(len(a),len(b))-1)
        self.base_bad_size_zi([1], [1], x2, 1, [0])

        self.base_bad_size_zi([1, 1], [1], x2, 1, [0,1,2,3])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[[0],[1],[2],[3]]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0],[1],[2]])
        self.base_bad_size_zi([1, 1], [1], x2, 1, [[0],[1],[2],[3],[4]])

        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [0,1,2,3,4,5,6,7])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[[0,1],[2,3],[4,5],[6,7]]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0,1,2,3],[4,5,6,7]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1], x2, 1, [[0,1],[2,3],[4,5],[6,7],[8,9]])

        self.base_bad_size_zi([1], [1, 1], x2, 1, [0,1,2,3])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[[0],[1],[2],[3]]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0, 1, 2, 3]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0],[1],[2]])
        self.base_bad_size_zi([1], [1, 1], x2, 1, [[0],[1],[2],[3],[4]])

        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [0,1,2,3,4,5,6,7])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[[0,1],[2,3],[4,5],[6,7]]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0,1,2,3],[4,5,6,7]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1], [1, 1, 1], x2, 1, [[0,1],[2,3],[4,5],[6,7],[8,9]])

        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [0,1,2,3,4,5,6,7])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[[0,1],[2,3],[4,5],[6,7]]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0,1,2,3],[4,5,6,7]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0,1],[2,3],[4,5]])
        self.base_bad_size_zi([1, 1, 1], [1, 1], x2, 1, [[0,1],[2,3],[4,5],[6,7],[8,9]])

    def test_empty_zi(self):
        # Regression test for #880: empty array for zi crashes.
        x = self.generate((5,))
        a = self.convert_dtype([1])
        b = self.convert_dtype([1])
        zi = self.convert_dtype([])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, x)
        assert_equal(zf.dtype, self.dtype)
        assert_equal(zf.size, 0)

    def test_lfiltic_bad_zi(self):
        # Regression test for #3699: bad initial conditions
        a = self.convert_dtype([1])
        b = self.convert_dtype([1])
        # "y" sets the datatype of zi, so it truncates if int
        zi = lfiltic(b, a, [1., 0])
        zi_1 = lfiltic(b, a, [1, 0])
        zi_2 = lfiltic(b, a, [True, False])
        assert_array_equal(zi, zi_1)
        assert_array_equal(zi, zi_2)

    def test_short_x_FIR(self):
        # regression test for #5116
        # x shorter than b, with non None zi fails
        a = self.convert_dtype([1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([7, -72])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, ye)
        assert_array_almost_equal(zf, zfe)

    def test_short_x_IIR(self):
        # regression test for #5116
        # x shorter than b, with non None zi fails
        a = self.convert_dtype([1, 1])
        b = self.convert_dtype([1, 0, -1])
        zi = self.convert_dtype([2, 7])
        x = self.convert_dtype([72])
        ye = self.convert_dtype([74])
        zfe = self.convert_dtype([-67, -72])
        y, zf = lfilter(b, a, x, zi=zi)
        assert_array_almost_equal(y, ye)
        assert_array_almost_equal(zf, zfe)

    def test_do_not_modify_a_b_IIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, -1])
        b0 = b.copy()
        a = self.convert_dtype([0.5, -0.5])
        a0 = a.copy()
        y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.])
        y_f = lfilter(b, a, x)
        assert_array_almost_equal(y_f, y_r)
        assert_equal(b, b0)
        assert_equal(a, a0)

    def test_do_not_modify_a_b_FIR(self):
        x = self.generate((6,))
        b = self.convert_dtype([1, 0, 1])
        b0 = b.copy()
        a = self.convert_dtype([2])
        a0 = a.copy()
        y_r = self.convert_dtype([0, 0.5, 1, 2, 3, 4.])
        y_f = lfilter(b, a, x)
        assert_array_almost_equal(y_f, y_r)
        assert_equal(b, b0)
        assert_equal(a, a0)


class TestLinearFilterFloat32(_TestLinearFilter):
    dtype = np.dtype('f')


class TestLinearFilterFloat64(_TestLinearFilter):
    dtype = np.dtype('d')


class TestLinearFilterFloatExtended(_TestLinearFilter):
    dtype = np.dtype('g')


class TestLinearFilterComplex64(_TestLinearFilter):
    dtype = np.dtype('F')


class TestLinearFilterComplex128(_TestLinearFilter):
    dtype = np.dtype('D')


class TestLinearFilterComplexExtended(_TestLinearFilter):
    dtype = np.dtype('G')

class TestLinearFilterDecimal(_TestLinearFilter):
    dtype = np.dtype('O')

    def type(self, x):
        return Decimal(str(x))


class TestLinearFilterObject(_TestLinearFilter):
    dtype = np.dtype('O')
    type = float


def test_lfilter_bad_object():
    # lfilter: object arrays with non-numeric objects raise TypeError.
    # Regression test for ticket #1452.
    assert_raises(TypeError, lfilter, [1.0], [1.0], [1.0, None, 2.0])
    assert_raises(TypeError, lfilter, [1.0], [None], [1.0, 2.0, 3.0])
    assert_raises(TypeError, lfilter, [None], [1.0], [1.0, 2.0, 3.0])


def test_lfilter_notimplemented_input():
    # Should not crash, gh-7991
    assert_raises(NotImplementedError, lfilter, [2,3], [4,5], [1,2,3,4,5])


@pytest.mark.parametrize('dt', [np.ubyte, np.byte, np.ushort, np.short,
                                np.uint, int, np.ulonglong, np.ulonglong,
                                np.float32, np.float64, np.longdouble,
                                Decimal])
class TestCorrelateReal:
    def _setup_rank1(self, dt):
        a = np.linspace(0, 3, 4).astype(dt)
        b = np.linspace(1, 2, 2).astype(dt)

        y_r = np.array([0, 2, 5, 8, 3]).astype(dt)
        return a, b, y_r

    def equal_tolerance(self, res_dt):
        # default value of keyword
        decimal = 6
        try:
            dt_info = np.finfo(res_dt)
            if hasattr(dt_info, 'resolution'):
                decimal = int(-0.5*np.log10(dt_info.resolution))
        except Exception:
            pass
        return decimal

    def equal_tolerance_fft(self, res_dt):
        # FFT implementations convert longdouble arguments down to
        # double so don't expect better precision, see gh-9520
        if res_dt == np.longdouble:
            return self.equal_tolerance(np.double)
        else:
            return self.equal_tolerance(res_dt)

    def test_method(self, dt):
        if dt == Decimal:
            method = choose_conv_method([Decimal(4)], [Decimal(3)])
            assert_equal(method, 'direct')
        else:
            a, b, y_r = self._setup_rank3(dt)
            y_fft = correlate(a, b, method='fft')
            y_direct = correlate(a, b, method='direct')

            assert_array_almost_equal(y_r, y_fft, decimal=self.equal_tolerance_fft(y_fft.dtype))
            assert_array_almost_equal(y_r, y_direct, decimal=self.equal_tolerance(y_direct.dtype))
            assert_equal(y_fft.dtype, dt)
            assert_equal(y_direct.dtype, dt)

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, 'valid')
        assert_array_almost_equal(y, y_r[1:4])
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, 'valid')
        assert_array_almost_equal(y, y_r[1:4][::-1])
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, 'same')
        assert_array_almost_equal(y, y_r[:-1])
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r)
        assert_equal(y.dtype, dt)

    def _setup_rank3(self, dt):
        a = np.linspace(0, 39, 40).reshape((2, 4, 5), order='F').astype(
            dt)
        b = np.linspace(0, 23, 24).reshape((2, 3, 4), order='F').astype(
            dt)

        y_r = array([[[0., 184., 504., 912., 1360., 888., 472., 160.],
                      [46., 432., 1062., 1840., 2672., 1698., 864., 266.],
                      [134., 736., 1662., 2768., 3920., 2418., 1168., 314.],
                      [260., 952., 1932., 3056., 4208., 2580., 1240., 332.],
                      [202., 664., 1290., 1984., 2688., 1590., 712., 150.],
                      [114., 344., 642., 960., 1280., 726., 296., 38.]],

                     [[23., 400., 1035., 1832., 2696., 1737., 904., 293.],
                      [134., 920., 2166., 3680., 5280., 3306., 1640., 474.],
                      [325., 1544., 3369., 5512., 7720., 4683., 2192., 535.],
                      [571., 1964., 3891., 6064., 8272., 4989., 2324., 565.],
                      [434., 1360., 2586., 3920., 5264., 3054., 1312., 230.],
                      [241., 700., 1281., 1888., 2496., 1383., 532., 39.]],

                     [[22., 214., 528., 916., 1332., 846., 430., 132.],
                      [86., 484., 1098., 1832., 2600., 1602., 772., 206.],
                      [188., 802., 1698., 2732., 3788., 2256., 1018., 218.],
                      [308., 1006., 1950., 2996., 4052., 2400., 1078., 230.],
                      [230., 692., 1290., 1928., 2568., 1458., 596., 78.],
                      [126., 354., 636., 924., 1212., 654., 234., 0.]]],
                    dtype=dt)

        return a, b, y_r

    def test_rank3_valid(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b, "valid")
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5])
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, "valid")
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5][::-1, ::-1, ::-1])
        assert_equal(y.dtype, dt)

    def test_rank3_same(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b, "same")
        assert_array_almost_equal(y, y_r[0:-1, 1:-1, 1:-2])
        assert_equal(y.dtype, dt)

    def test_rank3_all(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b)
        assert_array_almost_equal(y, y_r)
        assert_equal(y.dtype, dt)


class TestCorrelate:
    # Tests that don't depend on dtype

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, correlate, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, correlate, *(b, a), **{'mode': 'valid'})

    def test_invalid_params(self):
        a = [3, 4, 5]
        b = [1, 2, 3]
        assert_raises(ValueError, correlate, a, b, mode='spam')
        assert_raises(ValueError, correlate, a, b, mode='eggs', method='fft')
        assert_raises(ValueError, correlate, a, b, mode='ham', method='direct')
        assert_raises(ValueError, correlate, a, b, mode='full', method='bacon')
        assert_raises(ValueError, correlate, a, b, mode='same', method='bacon')

    def test_mismatched_dims(self):
        # Input arrays should have the same number of dimensions
        assert_raises(ValueError, correlate, [1], 2, method='direct')
        assert_raises(ValueError, correlate, 1, [2], method='direct')
        assert_raises(ValueError, correlate, [1], 2, method='fft')
        assert_raises(ValueError, correlate, 1, [2], method='fft')
        assert_raises(ValueError, correlate, [1], [[2]])
        assert_raises(ValueError, correlate, [3], 2)

    def test_numpy_fastpath(self):
        a = [1, 2, 3]
        b = [4, 5]
        assert_allclose(correlate(a, b, mode='same'), [5, 14, 23])

        a = [1, 2, 3]
        b = [4, 5, 6]
        assert_allclose(correlate(a, b, mode='same'), [17, 32, 23])
        assert_allclose(correlate(a, b, mode='full'), [6, 17, 32, 23, 12])
        assert_allclose(correlate(a, b, mode='valid'), [32])


@pytest.mark.parametrize("mode", ["valid", "same", "full"])
@pytest.mark.parametrize("behind", [True, False])
@pytest.mark.parametrize("input_size", [100, 101, 1000, 1001, 10000, 10001])
def test_correlation_lags(mode, behind, input_size):
    # generate random data
    rng = np.random.RandomState(0)
    in1 = rng.standard_normal(input_size)
    offset = int(input_size/10)
    # generate offset version of array to correlate with
    if behind:
        # y is behind x
        in2 = np.concatenate([rng.standard_normal(offset), in1])
        expected = -offset
    else:
        # y is ahead of x
        in2 = in1[offset:]
        expected = offset
    # cross correlate, returning lag information
    correlation = correlate(in1, in2, mode=mode)
    lags = correlation_lags(in1.size, in2.size, mode=mode)
    # identify the peak
    lag_index = np.argmax(correlation)
    # Check as expected
    assert_equal(lags[lag_index], expected)
    # Correlation and lags shape should match
    assert_equal(lags.shape, correlation.shape)


@pytest.mark.parametrize('dt', [np.csingle, np.cdouble, np.clongdouble])
class TestCorrelateComplex:
    # The decimal precision to be used for comparing results.
    # This value will be passed as the 'decimal' keyword argument of
    # assert_array_almost_equal().
    # Since correlate may chose to use FFT method which converts
    # longdoubles to doubles internally don't expect better precision
    # for longdouble than for double (see gh-9520).

    def decimal(self, dt):
        if dt == np.clongdouble:
            dt = np.cdouble
        return int(2 * np.finfo(dt).precision / 3)

    def _setup_rank1(self, dt, mode):
        np.random.seed(9)
        a = np.random.randn(10).astype(dt)
        a += 1j * np.random.randn(10).astype(dt)
        b = np.random.randn(8).astype(dt)
        b += 1j * np.random.randn(8).astype(dt)

        y_r = (correlate(a.real, b.real, mode=mode) +
               correlate(a.imag, b.imag, mode=mode)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag, mode=mode) +
                     correlate(a.imag, b.real, mode=mode))
        return a, b, y_r

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'valid')
        y = correlate(a, b, 'valid')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, 'valid')
        assert_array_almost_equal(y, y_r[::-1].conj(), decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'same')
        y = correlate(a, b, 'same')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt, 'full')
        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_swap_full(self, dt):
        d = np.array([0.+0.j, 1.+1.j, 2.+2.j], dtype=dt)
        k = np.array([1.+3.j, 2.+4.j, 3.+5.j, 4.+6.j], dtype=dt)
        y = correlate(d, k)
        assert_equal(y, [0.+0.j, 10.-2.j, 28.-6.j, 22.-6.j, 16.-6.j, 8.-4.j])

    def test_swap_same(self, dt):
        d = [0.+0.j, 1.+1.j, 2.+2.j]
        k = [1.+3.j, 2.+4.j, 3.+5.j, 4.+6.j]
        y = correlate(d, k, mode="same")
        assert_equal(y, [10.-2.j, 28.-6.j, 22.-6.j])

    def test_rank3(self, dt):
        a = np.random.randn(10, 8, 6).astype(dt)
        a += 1j * np.random.randn(10, 8, 6).astype(dt)
        b = np.random.randn(8, 6, 4).astype(dt)
        b += 1j * np.random.randn(8, 6, 4).astype(dt)

        y_r = (correlate(a.real, b.real)
               + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))

        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)

    def test_rank0(self, dt):
        a = np.array(np.random.randn()).astype(dt)
        a += 1j * np.array(np.random.randn()).astype(dt)
        b = np.array(np.random.randn()).astype(dt)
        b += 1j * np.array(np.random.randn()).astype(dt)

        y_r = (correlate(a.real, b.real)
               + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))

        y = correlate(a, b, 'full')
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)

        assert_equal(correlate([1], [2j]), correlate(1, 2j))
        assert_equal(correlate([2j], [3j]), correlate(2j, 3j))
        assert_equal(correlate([3j], [4]), correlate(3j, 4))


class TestCorrelate2d:

    def test_consistency_correlate_funcs(self):
        # Compare np.correlate, signal.correlate, signal.correlate2d
        a = np.arange(5)
        b = np.array([3.2, 1.4, 3])
        for mode in ['full', 'valid', 'same']:
            assert_almost_equal(np.correlate(a, b, mode=mode),
                                signal.correlate(a, b, mode=mode))
            assert_almost_equal(np.squeeze(signal.correlate2d([a], [b],
                                                              mode=mode)),
                                signal.correlate(a, b, mode=mode))

            # See gh-5897
            if mode == 'valid':
                assert_almost_equal(np.correlate(b, a, mode=mode),
                                    signal.correlate(b, a, mode=mode))
                assert_almost_equal(np.squeeze(signal.correlate2d([b], [a],
                                                                  mode=mode)),
                                    signal.correlate(b, a, mode=mode))

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = np.arange(1, 7).reshape((2, 3))
        b = np.arange(-6, 0).reshape((3, 2))

        assert_raises(ValueError, signal.correlate2d, *(a, b), **{'mode': 'valid'})
        assert_raises(ValueError, signal.correlate2d, *(b, a), **{'mode': 'valid'})

    def test_complex_input(self):
        assert_equal(signal.correlate2d([[1]], [[2j]]), -2j)
        assert_equal(signal.correlate2d([[2j]], [[3j]]), 6)
        assert_equal(signal.correlate2d([[3j]], [[4]]), 12j)


class TestLFilterZI:

    def test_basic(self):
        a = np.array([1.0, -1.0, 0.5])
        b = np.array([1.0, 0.0, 2.0])
        zi_expected = np.array([5.0, -1.0])
        zi = lfilter_zi(b, a)
        assert_array_almost_equal(zi, zi_expected)

    def test_scale_invariance(self):
        # Regression test.  There was a bug in which b was not correctly
        # rescaled when a[0] was nonzero.
        b = np.array([2, 8, 5])
        a = np.array([1, 1, 8])
        zi1 = lfilter_zi(b, a)
        zi2 = lfilter_zi(2*b, 2*a)
        assert_allclose(zi2, zi1, rtol=1e-12)


class TestFiltFilt:
    filtfilt_kind = 'tf'

    def filtfilt(self, zpk, x, axis=-1, padtype='odd', padlen=None,
                 method='pad', irlen=None):
        if self.filtfilt_kind == 'tf':
            b, a = zpk2tf(*zpk)
            return filtfilt(b, a, x, axis, padtype, padlen, method, irlen)
        elif self.filtfilt_kind == 'sos':
            sos = zpk2sos(*zpk)
            return sosfiltfilt(sos, x, axis, padtype, padlen)

    def test_basic(self):
        zpk = tf2zpk([1, 2, 3], [1, 2, 3])
        out = self.filtfilt(zpk, np.arange(12))
        assert_allclose(out, arange(12), atol=5.28e-11)

    def test_sine(self):
        rate = 2000
        t = np.linspace(0, 1.0, rate + 1)
        # A signal with low frequency and a high frequency.
        xlow = np.sin(5 * 2 * np.pi * t)
        xhigh = np.sin(250 * 2 * np.pi * t)
        x = xlow + xhigh

        zpk = butter(8, 0.125, output='zpk')
        # r is the magnitude of the largest pole.
        r = np.abs(zpk[1]).max()
        eps = 1e-5
        # n estimates the number of steps for the
        # transient to decay by a factor of eps.
        n = int(np.ceil(np.log(eps) / np.log(r)))

        # High order lowpass filter...
        y = self.filtfilt(zpk, x, padlen=n)
        # Result should be just xlow.
        err = np.abs(y - xlow).max()
        assert_(err < 1e-4)

        # A 2D case.
        x2d = np.vstack([xlow, xlow + xhigh])
        y2d = self.filtfilt(zpk, x2d, padlen=n, axis=1)
        assert_equal(y2d.shape, x2d.shape)
        err = np.abs(y2d - xlow).max()
        assert_(err < 1e-4)

        # Use the previous result to check the use of the axis keyword.
        # (Regression test for ticket #1620)
        y2dt = self.filtfilt(zpk, x2d.T, padlen=n, axis=0)
        assert_equal(y2d, y2dt.T)

    def test_axis(self):
        # Test the 'axis' keyword on a 3D array.
        x = np.arange(10.0 * 11.0 * 12.0).reshape(10, 11, 12)
        zpk = butter(3, 0.125, output='zpk')
        y0 = self.filtfilt(zpk, x, padlen=0, axis=0)
        y1 = self.filtfilt(zpk, np.swapaxes(x, 0, 1), padlen=0, axis=1)
        assert_array_equal(y0, np.swapaxes(y1, 0, 1))
        y2 = self.filtfilt(zpk, np.swapaxes(x, 0, 2), padlen=0, axis=2)
        assert_array_equal(y0, np.swapaxes(y2, 0, 2))

    def test_acoeff(self):
        if self.filtfilt_kind != 'tf':
            return  # only necessary for TF
        # test for 'a' coefficient as single number
        out = signal.filtfilt([.5, .5], 1, np.arange(10))
        assert_allclose(out, np.arange(10), rtol=1e-14, atol=1e-14)

    def test_gust_simple(self):
        if self.filtfilt_kind != 'tf':
            pytest.skip('gust only implemented for TF systems')
        # The input array has length 2.  The exact solution for this case
        # was computed "by hand".
        x = np.array([1.0, 2.0])
        b = np.array([0.5])
        a = np.array([1.0, -0.5])
        y, z1, z2 = _filtfilt_gust(b, a, x)
        assert_allclose([z1[0], z2[0]],
                        [0.3*x[0] + 0.2*x[1], 0.2*x[0] + 0.3*x[1]])
        assert_allclose(y, [z1[0] + 0.25*z2[0] + 0.25*x[0] + 0.125*x[1],
                            0.25*z1[0] + z2[0] + 0.125*x[0] + 0.25*x[1]])

    def test_gust_scalars(self):
        if self.filtfilt_kind != 'tf':
            pytest.skip('gust only implemented for TF systems')
        # The filter coefficients are both scalars, so the filter simply
        # multiplies its input by b/a.  When it is used in filtfilt, the
        # factor is (b/a)**2.
        x = np.arange(12)
        b = 3.0
        a = 2.0
        y = filtfilt(b, a, x, method="gust")
        expected = (b/a)**2 * x
        assert_allclose(y, expected)


class TestSOSFiltFilt(TestFiltFilt):
    filtfilt_kind = 'sos'

    def test_equivalence(self):
        """Test equivalence between sosfiltfilt and filtfilt"""
        x = np.random.RandomState(0).randn(1000)
        for order in range(1, 6):
            zpk = signal.butter(order, 0.35, output='zpk')
            b, a = zpk2tf(*zpk)
            sos = zpk2sos(*zpk)
            y = filtfilt(b, a, x)
            y_sos = sosfiltfilt(sos, x)
            assert_allclose(y, y_sos, atol=1e-12, err_msg='order=%s' % order)


def filtfilt_gust_opt(b, a, x):
    """
    An alternative implementation of filtfilt with Gustafsson edges.

    This function computes the same result as
    `scipy.signal.signaltools._filtfilt_gust`, but only 1-d arrays
    are accepted.  The problem is solved using `fmin` from `scipy.optimize`.
    `_filtfilt_gust` is significanly faster than this implementation.
    """
    def filtfilt_gust_opt_func(ics, b, a, x):
        """Objective function used in filtfilt_gust_opt."""
        m = max(len(a), len(b)) - 1
        z0f = ics[:m]
        z0b = ics[m:]
        y_f = lfilter(b, a, x, zi=z0f)[0]
        y_fb = lfilter(b, a, y_f[::-1], zi=z0b)[0][::-1]

        y_b = lfilter(b, a, x[::-1], zi=z0b)[0][::-1]
        y_bf = lfilter(b, a, y_b, zi=z0f)[0]
        value = np.sum((y_fb - y_bf)**2)
        return value

    m = max(len(a), len(b)) - 1
    zi = lfilter_zi(b, a)
    ics = np.concatenate((x[:m].mean()*zi, x[-m:].mean()*zi))
    result = fmin(filtfilt_gust_opt_func, ics, args=(b, a, x),
                  xtol=1e-10, ftol=1e-12,
                  maxfun=10000, maxiter=10000,
                  full_output=True, disp=False)
    opt, fopt, niter, funcalls, warnflag = result
    if warnflag > 0:
        raise RuntimeError("minimization failed in filtfilt_gust_opt: "
                           "warnflag=%d" % warnflag)
    z0f = opt[:m]
    z0b = opt[m:]

    # Apply the forward-backward filter using the computed initial
    # conditions.
    y_b = lfilter(b, a, x[::-1], zi=z0b)[0][::-1]
    y = lfilter(b, a, y_b, zi=z0f)[0]

    return y, z0f, z0b


def check_filtfilt_gust(b, a, shape, axis, irlen=None):
    # Generate x, the data to be filtered.
    np.random.seed(123)
    x = np.random.randn(*shape)

    # Apply filtfilt to x. This is the main calculation to be checked.
    y = filtfilt(b, a, x, axis=axis, method="gust", irlen=irlen)

    # Also call the private function so we can test the ICs.
    yg, zg1, zg2 = _filtfilt_gust(b, a, x, axis=axis, irlen=irlen)

    # filtfilt_gust_opt is an independent implementation that gives the
    # expected result, but it only handles 1-D arrays, so use some looping
    # and reshaping shenanigans to create the expected output arrays.
    xx = np.swapaxes(x, axis, -1)
    out_shape = xx.shape[:-1]
    yo = np.empty_like(xx)
    m = max(len(a), len(b)) - 1
    zo1 = np.empty(out_shape + (m,))
    zo2 = np.empty(out_shape + (m,))
    for indx in product(*[range(d) for d in out_shape]):
        yo[indx], zo1[indx], zo2[indx] = filtfilt_gust_opt(b, a, xx[indx])
    yo = np.swapaxes(yo, -1, axis)
    zo1 = np.swapaxes(zo1, -1, axis)
    zo2 = np.swapaxes(zo2, -1, axis)

    assert_allclose(y, yo, rtol=1e-8, atol=1e-9)
    assert_allclose(yg, yo, rtol=1e-8, atol=1e-9)
    assert_allclose(zg1, zo1, rtol=1e-8, atol=1e-9)
    assert_allclose(zg2, zo2, rtol=1e-8, atol=1e-9)


def test_choose_conv_method():
    for mode in ['valid', 'same', 'full']:
        for ndim in [1, 2]:
            n, k, true_method = 8, 6, 'direct'
            x = np.random.randn(*((n,) * ndim))
            h = np.random.randn(*((k,) * ndim))

            method = choose_conv_method(x, h, mode=mode)
            assert_equal(method, true_method)

            method_try, times = choose_conv_method(x, h, mode=mode, measure=True)
            assert_(method_try in {'fft', 'direct'})
            assert_(type(times) is dict)
            assert_('fft' in times.keys() and 'direct' in times.keys())

        n = 10
        for not_fft_conv_supp in ["complex256", "complex192"]:
            if hasattr(np, not_fft_conv_supp):
                x = np.ones(n, dtype=not_fft_conv_supp)
                h = x.copy()
                assert_equal(choose_conv_method(x, h, mode=mode), 'direct')

        x = np.array([2**51], dtype=np.int64)
        h = x.copy()
        assert_equal(choose_conv_method(x, h, mode=mode), 'direct')

        x = [Decimal(3), Decimal(2)]
        h = [Decimal(1), Decimal(4)]
        assert_equal(choose_conv_method(x, h, mode=mode), 'direct')


def test_filtfilt_gust():
    # Design a filter.
    z, p, k = signal.ellip(3, 0.01, 120, 0.0875, output='zpk')

    # Find the approximate impulse response length of the filter.
    eps = 1e-10
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))

    np.random.seed(123)

    b, a = zpk2tf(z, p, k)
    for irlen in [None, approx_impulse_len]:
        signal_len = 5 * approx_impulse_len

        # 1-d test case
        check_filtfilt_gust(b, a, (signal_len,), 0, irlen)

        # 3-d test case; test each axis.
        for axis in range(3):
            shape = [2, 2, 2]
            shape[axis] = signal_len
            check_filtfilt_gust(b, a, shape, axis, irlen)

    # Test case with length less than 2*approx_impulse_len.
    # In this case, `filtfilt_gust` should behave the same as if
    # `irlen=None` was given.
    length = 2*approx_impulse_len - 50
    check_filtfilt_gust(b, a, (length,), 0, approx_impulse_len)


class TestDecimate:
    def test_bad_args(self):
        x = np.arange(12)
        assert_raises(TypeError, signal.decimate, x, q=0.5, n=1)
        assert_raises(TypeError, signal.decimate, x, q=2, n=0.5)

    def test_basic_IIR(self):
        x = np.arange(12)
        y = signal.decimate(x, 2, n=1, ftype='iir', zero_phase=False).round()
        assert_array_equal(y, x[::2])

    def test_basic_FIR(self):
        x = np.arange(12)
        y = signal.decimate(x, 2, n=1, ftype='fir', zero_phase=False).round()
        assert_array_equal(y, x[::2])

    def test_shape(self):
        # Regression test for ticket #1480.
        z = np.zeros((30, 30))
        d0 = signal.decimate(z, 2, axis=0, zero_phase=False)
        assert_equal(d0.shape, (15, 30))
        d1 = signal.decimate(z, 2, axis=1, zero_phase=False)
        assert_equal(d1.shape, (30, 15))

    def test_phaseshift_FIR(self):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients, "Badly conditioned filter")
            self._test_phaseshift(method='fir', zero_phase=False)

    def test_zero_phase_FIR(self):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients, "Badly conditioned filter")
            self._test_phaseshift(method='fir', zero_phase=True)

    def test_phaseshift_IIR(self):
        self._test_phaseshift(method='iir', zero_phase=False)

    def test_zero_phase_IIR(self):
        self._test_phaseshift(method='iir', zero_phase=True)

    def _test_phaseshift(self, method, zero_phase):
        rate = 120
        rates_to = [15, 20, 30, 40]  # q = 8, 6, 4, 3

        t_tot = int(100)  # Need to let antialiasing filters settle
        t = np.arange(rate*t_tot+1) / float(rate)

        # Sinusoids at 0.8*nyquist, windowed to avoid edge artifacts
        freqs = np.array(rates_to) * 0.8 / 2
        d = (np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t)
             * signal.windows.tukey(t.size, 0.1))

        for rate_to in rates_to:
            q = rate // rate_to
            t_to = np.arange(rate_to*t_tot+1) / float(rate_to)
            d_tos = (np.exp(1j * 2 * np.pi * freqs[:, np.newaxis] * t_to)
                     * signal.windows.tukey(t_to.size, 0.1))

            # Set up downsampling filters, match v0.17 defaults
            if method == 'fir':
                n = 30
                system = signal.dlti(signal.firwin(n + 1, 1. / q,
                                                   window='hamming'), 1.)
            elif method == 'iir':
                n = 8
                wc = 0.8*np.pi/q
                system = signal.dlti(*signal.cheby1(n, 0.05, wc/np.pi))

            # Calculate expected phase response, as unit complex vector
            if zero_phase is False:
                _, h_resps = signal.freqz(system.num, system.den,
                                          freqs/rate*2*np.pi)
                h_resps /= np.abs(h_resps)
            else:
                h_resps = np.ones_like(freqs)

            y_resamps = signal.decimate(d.real, q, n, ftype=system,
                                        zero_phase=zero_phase)

            # Get phase from complex inner product, like CSD
            h_resamps = np.sum(d_tos.conj() * y_resamps, axis=-1)
            h_resamps /= np.abs(h_resamps)
            subnyq = freqs < 0.5*rate_to

            # Complex vectors should be aligned, only compare below nyquist
            assert_allclose(np.angle(h_resps.conj()*h_resamps)[subnyq], 0,
                            atol=1e-3, rtol=1e-3)

    def test_auto_n(self):
        # Test that our value of n is a reasonable choice (depends on
        # the downsampling factor)
        sfreq = 100.
        n = 1000
        t = np.arange(n) / sfreq
        # will alias for decimations (>= 15)
        x = np.sqrt(2. / n) * np.sin(2 * np.pi * (sfreq / 30.) * t)
        assert_allclose(np.linalg.norm(x), 1., rtol=1e-3)
        x_out = signal.decimate(x, 30, ftype='fir')
        assert_array_less(np.linalg.norm(x_out), 0.01)


class TestHilbert:

    def test_bad_args(self):
        x = np.array([1.0 + 0.0j])
        assert_raises(ValueError, hilbert, x)
        x = np.arange(8.0)
        assert_raises(ValueError, hilbert, x, N=0)

    def test_hilbert_theoretical(self):
        # test cases by Ariel Rokem
        decimal = 14

        pi = np.pi
        t = np.arange(0, 2 * pi, pi / 256)
        a0 = np.sin(t)
        a1 = np.cos(t)
        a2 = np.sin(2 * t)
        a3 = np.cos(2 * t)
        a = np.vstack([a0, a1, a2, a3])

        h = hilbert(a)
        h_abs = np.abs(h)
        h_angle = np.angle(h)
        h_real = np.real(h)

        # The real part should be equal to the original signals:
        assert_almost_equal(h_real, a, decimal)
        # The absolute value should be one everywhere, for this input:
        assert_almost_equal(h_abs, np.ones(a.shape), decimal)
        # For the 'slow' sine - the phase should go from -pi/2 to pi/2 in
        # the first 256 bins:
        assert_almost_equal(h_angle[0, :256],
                            np.arange(-pi / 2, pi / 2, pi / 256),
                            decimal)
        # For the 'slow' cosine - the phase should go from 0 to pi in the
        # same interval:
        assert_almost_equal(
            h_angle[1, :256], np.arange(0, pi, pi / 256), decimal)
        # The 'fast' sine should make this phase transition in half the time:
        assert_almost_equal(h_angle[2, :128],
                            np.arange(-pi / 2, pi / 2, pi / 128),
                            decimal)
        # Ditto for the 'fast' cosine:
        assert_almost_equal(
            h_angle[3, :128], np.arange(0, pi, pi / 128), decimal)

        # The imaginary part of hilbert(cos(t)) = sin(t) Wikipedia
        assert_almost_equal(h[1].imag, a0, decimal)

    def test_hilbert_axisN(self):
        # tests for axis and N arguments
        a = np.arange(18).reshape(3, 6)
        # test axis
        aa = hilbert(a, axis=-1)
        assert_equal(hilbert(a.T, axis=0), aa.T)
        # test 1d
        assert_almost_equal(hilbert(a[0]), aa[0], 14)

        # test N
        aan = hilbert(a, N=20, axis=-1)
        assert_equal(aan.shape, [3, 20])
        assert_equal(hilbert(a.T, N=20, axis=0).shape, [20, 3])
        # the next test is just a regression test,
        # no idea whether numbers make sense
        a0hilb = np.array([0.000000000000000e+00 - 1.72015830311905j,
                           1.000000000000000e+00 - 2.047794505137069j,
                           1.999999999999999e+00 - 2.244055555687583j,
                           3.000000000000000e+00 - 1.262750302935009j,
                           4.000000000000000e+00 - 1.066489252384493j,
                           5.000000000000000e+00 + 2.918022706971047j,
                           8.881784197001253e-17 + 3.845658908989067j,
                          -9.444121133484362e-17 + 0.985044202202061j,
                          -1.776356839400251e-16 + 1.332257797702019j,
                          -3.996802888650564e-16 + 0.501905089898885j,
                           1.332267629550188e-16 + 0.668696078880782j,
                          -1.192678053963799e-16 + 0.235487067862679j,
                          -1.776356839400251e-16 + 0.286439612812121j,
                           3.108624468950438e-16 + 0.031676888064907j,
                           1.332267629550188e-16 - 0.019275656884536j,
                          -2.360035624836702e-16 - 0.1652588660287j,
                           0.000000000000000e+00 - 0.332049855010597j,
                           3.552713678800501e-16 - 0.403810179797771j,
                           8.881784197001253e-17 - 0.751023775297729j,
                           9.444121133484362e-17 - 0.79252210110103j])
        assert_almost_equal(aan[0], a0hilb, 14, 'N regression')


class TestHilbert2:

    def test_bad_args(self):
        # x must be real.
        x = np.array([[1.0 + 0.0j]])
        assert_raises(ValueError, hilbert2, x)

        # x must be rank 2.
        x = np.arange(24).reshape(2, 3, 4)
        assert_raises(ValueError, hilbert2, x)

        # Bad value for N.
        x = np.arange(16).reshape(4, 4)
        assert_raises(ValueError, hilbert2, x, N=0)
        assert_raises(ValueError, hilbert2, x, N=(2, 0))
        assert_raises(ValueError, hilbert2, x, N=(2,))


class TestPartialFractionExpansion:
    @staticmethod
    def assert_rp_almost_equal(r, p, r_true, p_true, decimal=7):
        r_true = np.asarray(r_true)
        p_true = np.asarray(p_true)

        distance = np.hypot(abs(p[:, None] - p_true),
                            abs(r[:, None] - r_true))

        rows, cols = linear_sum_assignment(distance)
        assert_almost_equal(p[rows], p_true[cols], decimal=decimal)
        assert_almost_equal(r[rows], r_true[cols], decimal=decimal)

    def test_compute_factors(self):
        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1])
        assert_equal(len(factors), 3)
        assert_almost_equal(factors[0], np.poly([2, 2, 3]))
        assert_almost_equal(factors[1], np.poly([1, 1, 1, 3]))
        assert_almost_equal(factors[2], np.poly([1, 1, 1, 2, 2]))
        assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))

        factors, poly = _compute_factors([1, 2, 3], [3, 2, 1],
                                         include_powers=True)
        assert_equal(len(factors), 6)
        assert_almost_equal(factors[0], np.poly([1, 1, 2, 2, 3]))
        assert_almost_equal(factors[1], np.poly([1, 2, 2, 3]))
        assert_almost_equal(factors[2], np.poly([2, 2, 3]))
        assert_almost_equal(factors[3], np.poly([1, 1, 1, 2, 3]))
        assert_almost_equal(factors[4], np.poly([1, 1, 1, 3]))
        assert_almost_equal(factors[5], np.poly([1, 1, 1, 2, 2]))
        assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))

    def test_group_poles(self):
        unique, multiplicity = _group_poles(
            [1.0, 1.001, 1.003, 2.0, 2.003, 3.0], 0.1, 'min')
        assert_equal(unique, [1.0, 2.0, 3.0])
        assert_equal(multiplicity, [3, 2, 1])

    def test_residue_general(self):
        # Test are taken from issue #4464, note that poles in scipy are
        # in increasing by absolute value order, opposite to MATLAB.
        r, p, k = residue([5, 3, -2, 7], [-4, 0, 8, 3])
        assert_almost_equal(r, [1.3320, -0.6653, -1.4167], decimal=4)
        assert_almost_equal(p, [-0.4093, -1.1644, 1.5737], decimal=4)
        assert_almost_equal(k, [-1.2500], decimal=4)

        r, p, k = residue([-4, 8], [1, 6, 8])
        assert_almost_equal(r, [8, -12])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)

        r, p, k = residue([4, 1], [1, -1, -2])
        assert_almost_equal(r, [1, 3])
        assert_almost_equal(p, [-1, 2])
        assert_equal(k.size, 0)

        r, p, k = residue([4, 3], [2, -3.4, 1.98, -0.406])
        self.assert_rp_almost_equal(
            r, p, [-18.125 - 13.125j, -18.125 + 13.125j, 36.25],
            [0.5 - 0.2j, 0.5 + 0.2j, 0.7])
        assert_equal(k.size, 0)

        r, p, k = residue([2, 1], [1, 5, 8, 4])
        self.assert_rp_almost_equal(r, p, [-1, 1, 3], [-1, -2, -2])
        assert_equal(k.size, 0)

        r, p, k = residue([3, -1.1, 0.88, -2.396, 1.348],
                          [1, -0.7, -0.14, 0.048])
        assert_almost_equal(r, [-3, 4, 1])
        assert_almost_equal(p, [0.2, -0.3, 0.8])
        assert_almost_equal(k, [3, 1])

        r, p, k = residue([1], [1, 2, -3])
        assert_almost_equal(r, [0.25, -0.25])
        assert_almost_equal(p, [1, -3])
        assert_equal(k.size, 0)

        r, p, k = residue([1, 0, -5], [1, 0, 0, 0, -1])
        self.assert_rp_almost_equal(r, p,
                                    [1, 1.5j, -1.5j, -1], [-1, -1j, 1j, 1])
        assert_equal(k.size, 0)

        r, p, k = residue([3, 8, 6], [1, 3, 3, 1])
        self.assert_rp_almost_equal(r, p, [1, 2, 3], [-1, -1, -1])
        assert_equal(k.size, 0)

        r, p, k = residue([3, -1], [1, -3, 2])
        assert_almost_equal(r, [-2, 5])
        assert_almost_equal(p, [1, 2])
        assert_equal(k.size, 0)

        r, p, k = residue([2, 3, -1], [1, -3, 2])
        assert_almost_equal(r, [-4, 13])
        assert_almost_equal(p, [1, 2])
        assert_almost_equal(k, [2])

        r, p, k = residue([7, 2, 3, -1], [1, -3, 2])
        assert_almost_equal(r, [-11, 69])
        assert_almost_equal(p, [1, 2])
        assert_almost_equal(k, [7, 23])

        r, p, k = residue([2, 3, -1], [1, -3, 4, -2])
        self.assert_rp_almost_equal(r, p, [4, -1 + 3.5j, -1 - 3.5j],
                                    [1, 1 - 1j, 1 + 1j])
        assert_almost_equal(k.size, 0)

    def test_residue_leading_zeros(self):
        # Leading zeros in numerator or denominator must not affect the answer.
        r0, p0, k0 = residue([5, 3, -2, 7], [-4, 0, 8, 3])
        r1, p1, k1 = residue([0, 5, 3, -2, 7], [-4, 0, 8, 3])
        r2, p2, k2 = residue([5, 3, -2, 7], [0, -4, 0, 8, 3])
        r3, p3, k3 = residue([0, 0, 5, 3, -2, 7], [0, 0, 0, -4, 0, 8, 3])
        assert_almost_equal(r0, r1)
        assert_almost_equal(r0, r2)
        assert_almost_equal(r0, r3)
        assert_almost_equal(p0, p1)
        assert_almost_equal(p0, p2)
        assert_almost_equal(p0, p3)
        assert_almost_equal(k0, k1)
        assert_almost_equal(k0, k2)
        assert_almost_equal(k0, k3)

    def test_resiude_degenerate(self):
        # Several tests for zero numerator and denominator.
        r, p, k = residue([0, 0], [1, 6, 8])
        assert_almost_equal(r, [0, 0])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)

        r, p, k = residue(0, 1)
        assert_equal(r.size, 0)
        assert_equal(p.size, 0)
        assert_equal(k.size, 0)

        with pytest.raises(ValueError, match="Denominator `a` is zero."):
            residue(1, 0)

    def test_residuez_general(self):
        r, p, k = residuez([1, 6, 6, 2], [1, -(2 + 1j), (1 + 2j), -1j])
        self.assert_rp_almost_equal(r, p, [-2+2.5j, 7.5+7.5j, -4.5-12j],
                                    [1j, 1, 1])
        assert_almost_equal(k, [2j])

        r, p, k = residuez([1, 2, 1], [1, -1, 0.3561])
        self.assert_rp_almost_equal(r, p,
                                    [-0.9041 - 5.9928j, -0.9041 + 5.9928j],
                                    [0.5 + 0.3257j, 0.5 - 0.3257j],
                                    decimal=4)
        assert_almost_equal(k, [2.8082], decimal=4)

        r, p, k = residuez([1, -1], [1, -5, 6])
        assert_almost_equal(r, [-1, 2])
        assert_almost_equal(p, [2, 3])
        assert_equal(k.size, 0)

        r, p, k = residuez([2, 3, 4], [1, 3, 3, 1])
        self.assert_rp_almost_equal(r, p, [4, -5, 3], [-1, -1, -1])
        assert_equal(k.size, 0)

        r, p, k = residuez([1, -10, -4, 4], [2, -2, -4])
        assert_almost_equal(r, [0.5, -1.5])
        assert_almost_equal(p, [-1, 2])
        assert_almost_equal(k, [1.5, -1])

        r, p, k = residuez([18], [18, 3, -4, -1])
        self.assert_rp_almost_equal(r, p,
                                    [0.36, 0.24, 0.4], [0.5, -1/3, -1/3])
        assert_equal(k.size, 0)

        r, p, k = residuez([2, 3], np.polymul([1, -1/2], [1, 1/4]))
        assert_almost_equal(r, [-10/3, 16/3])
        assert_almost_equal(p, [-0.25, 0.5])
        assert_equal(k.size, 0)

        r, p, k = residuez([1, -2, 1], [1, -1])
        assert_almost_equal(r, [0])
        assert_almost_equal(p, [1])
        assert_almost_equal(k, [1, -1])

        r, p, k = residuez(1, [1, -1j])
        assert_almost_equal(r, [1])
        assert_almost_equal(p, [1j])
        assert_equal(k.size, 0)

        r, p, k = residuez(1, [1, -1, 0.25])
        assert_almost_equal(r, [0, 1])
        assert_almost_equal(p, [0.5, 0.5])
        assert_equal(k.size, 0)

        r, p, k = residuez(1, [1, -0.75, .125])
        assert_almost_equal(r, [-1, 2])
        assert_almost_equal(p, [0.25, 0.5])
        assert_equal(k.size, 0)

        r, p, k = residuez([1, 6, 2], [1, -2, 1])
        assert_almost_equal(r, [-10, 9])
        assert_almost_equal(p, [1, 1])
        assert_almost_equal(k, [2])

        r, p, k = residuez([6, 2], [1, -2, 1])
        assert_almost_equal(r, [-2, 8])
        assert_almost_equal(p, [1, 1])
        assert_equal(k.size, 0)

        r, p, k = residuez([1, 6, 6, 2], [1, -2, 1])
        assert_almost_equal(r, [-24, 15])
        assert_almost_equal(p, [1, 1])
        assert_almost_equal(k, [10, 2])

        r, p, k = residuez([1, 0, 1], [1, 0, 0, 0, 0, -1])
        self.assert_rp_almost_equal(r, p,
                                    [0.2618 + 0.1902j, 0.2618 - 0.1902j,
                                     0.4, 0.0382 - 0.1176j, 0.0382 + 0.1176j],
                                    [-0.8090 + 0.5878j, -0.8090 - 0.5878j,
                                     1.0, 0.3090 + 0.9511j, 0.3090 - 0.9511j],
                                    decimal=4)
        assert_equal(k.size, 0)

    def test_residuez_trailing_zeros(self):
        # Trailing zeros in numerator or denominator must not affect the
        # answer.
        r0, p0, k0 = residuez([5, 3, -2, 7], [-4, 0, 8, 3])
        r1, p1, k1 = residuez([5, 3, -2, 7, 0], [-4, 0, 8, 3])
        r2, p2, k2 = residuez([5, 3, -2, 7], [-4, 0, 8, 3, 0])
        r3, p3, k3 = residuez([5, 3, -2, 7, 0, 0], [-4, 0, 8, 3, 0, 0, 0])
        assert_almost_equal(r0, r1)
        assert_almost_equal(r0, r2)
        assert_almost_equal(r0, r3)
        assert_almost_equal(p0, p1)
        assert_almost_equal(p0, p2)
        assert_almost_equal(p0, p3)
        assert_almost_equal(k0, k1)
        assert_almost_equal(k0, k2)
        assert_almost_equal(k0, k3)

    def test_residuez_degenerate(self):
        r, p, k = residuez([0, 0], [1, 6, 8])
        assert_almost_equal(r, [0, 0])
        assert_almost_equal(p, [-2, -4])
        assert_equal(k.size, 0)

        r, p, k = residuez(0, 1)
        assert_equal(r.size, 0)
        assert_equal(p.size, 0)
        assert_equal(k.size, 0)

        with pytest.raises(ValueError, match="Denominator `a` is zero."):
            residuez(1, 0)

        with pytest.raises(ValueError,
                           match="First coefficient of determinant `a` must "
                                 "be non-zero."):
            residuez(1, [0, 1, 2, 3])

    def test_inverse_unique_roots_different_rtypes(self):
        # This test was inspired by github issue 2496.
        r = [3 / 10, -1 / 6, -2 / 15]
        p = [0, -2, -5]
        k = []
        b_expected = [0, 1, 3]
        a_expected = [1, 7, 10, 0]

        # With the default tolerance, the rtype does not matter
        # for this example.
        for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
            b, a = invres(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected)
            assert_allclose(a, a_expected)

            b, a = invresz(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected)
            assert_allclose(a, a_expected)

    def test_inverse_repeated_roots_different_rtypes(self):
        r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]
        p = [0, -2, -2, -5]
        k = []
        b_expected = [0, 0, 1, 3]
        b_expected_z = [-1/6, -2/3, 11/6, 3]
        a_expected = [1, 9, 24, 20, 0]

        for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
            b, a = invres(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected, atol=1e-14)
            assert_allclose(a, a_expected)

            b, a = invresz(r, p, k, rtype=rtype)
            assert_allclose(b, b_expected_z, atol=1e-14)
            assert_allclose(a, a_expected)

    def test_inverse_bad_rtype(self):
        r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]
        p = [0, -2, -2, -5]
        k = []
        with pytest.raises(ValueError, match="`rtype` must be one of"):
            invres(r, p, k, rtype='median')
        with pytest.raises(ValueError, match="`rtype` must be one of"):
            invresz(r, p, k, rtype='median')

    def test_invresz_one_coefficient_bug(self):
        # Regression test for issue in gh-4646.
        r = [1]
        p = [2]
        k = [0]
        b, a = invresz(r, p, k)
        assert_allclose(b, [1.0])
        assert_allclose(a, [1.0, -2.0])

    def test_invres(self):
        b, a = invres([1], [1], [])
        assert_almost_equal(b, [1])
        assert_almost_equal(a, [1, -1])

        b, a = invres([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])
        assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])
        assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])

        b, a = invres([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])
        assert_almost_equal(b, [1, -1 - 1j, 1 - 2j, 0.5 - 3j, 10])
        assert_almost_equal(a, [1, -3 - 1j, 4])

        b, a = invres([-1, 2, 1j, 3 - 1j, 4, -2],
                      [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])
        assert_almost_equal(b, [4 - 1j, -28 + 16j, 40 - 62j, 100 + 24j,
                                -292 + 219j, 192 - 268j])
        assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j,
                                108 - 54j, -81 + 108j])

        b, a = invres([-1, 1j], [1, 1], [1, 2])
        assert_almost_equal(b, [1, 0, -4, 3 + 1j])
        assert_almost_equal(a, [1, -2, 1])

    def test_invresz(self):
        b, a = invresz([1], [1], [])
        assert_almost_equal(b, [1])
        assert_almost_equal(a, [1, -1])

        b, a = invresz([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])
        assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])
        assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])

        b, a = invresz([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])
        assert_almost_equal(b, [2.5, -3 - 1j, 1 - 2j, -1 - 3j, 12])
        assert_almost_equal(a, [1, -3 - 1j, 4])

        b, a = invresz([-1, 2, 1j, 3 - 1j, 4, -2],
                       [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])
        assert_almost_equal(b, [6, -50 + 11j, 100 - 72j, 80 + 58j,
                                -354 + 228j, 234 - 297j])
        assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j,
                                108 - 54j, -81 + 108j])

        b, a = invresz([-1, 1j], [1, 1], [1, 2])
        assert_almost_equal(b, [1j, 1, -3, 2])
        assert_almost_equal(a, [1, -2, 1])

    def test_inverse_scalar_arguments(self):
        b, a = invres(1, 1, 1)
        assert_almost_equal(b, [1, 0])
        assert_almost_equal(a, [1, -1])

        b, a = invresz(1, 1, 1)
        assert_almost_equal(b, [2, -1])
        assert_almost_equal(a, [1, -1])


class TestVectorstrength:

    def test_single_1dperiod(self):
        events = np.array([.5])
        period = 5.
        targ_strength = 1.
        targ_phase = .1

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 0)
        assert_equal(phase.ndim, 0)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_single_2dperiod(self):
        events = np.array([.5])
        period = [1, 2, 5.]
        targ_strength = [1.] * 3
        targ_phase = np.array([.5, .25, .1])

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 1)
        assert_equal(phase.ndim, 1)
        assert_array_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_equal_1dperiod(self):
        events = np.array([.25, .25, .25, .25, .25, .25])
        period = 2
        targ_strength = 1.
        targ_phase = .125

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 0)
        assert_equal(phase.ndim, 0)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_equal_2dperiod(self):
        events = np.array([.25, .25, .25, .25, .25, .25])
        period = [1, 2, ]
        targ_strength = [1.] * 2
        targ_phase = np.array([.25, .125])

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 1)
        assert_equal(phase.ndim, 1)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_spaced_1dperiod(self):
        events = np.array([.1, 1.1, 2.1, 4.1, 10.1])
        period = 1
        targ_strength = 1.
        targ_phase = .1

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 0)
        assert_equal(phase.ndim, 0)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_spaced_2dperiod(self):
        events = np.array([.1, 1.1, 2.1, 4.1, 10.1])
        period = [1, .5]
        targ_strength = [1.] * 2
        targ_phase = np.array([.1, .2])

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 1)
        assert_equal(phase.ndim, 1)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_partial_1dperiod(self):
        events = np.array([.25, .5, .75])
        period = 1
        targ_strength = 1. / 3.
        targ_phase = .5

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 0)
        assert_equal(phase.ndim, 0)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_partial_2dperiod(self):
        events = np.array([.25, .5, .75])
        period = [1., 1., 1., 1.]
        targ_strength = [1. / 3.] * 4
        targ_phase = np.array([.5, .5, .5, .5])

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 1)
        assert_equal(phase.ndim, 1)
        assert_almost_equal(strength, targ_strength)
        assert_almost_equal(phase, 2 * np.pi * targ_phase)

    def test_opposite_1dperiod(self):
        events = np.array([0, .25, .5, .75])
        period = 1.
        targ_strength = 0

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 0)
        assert_equal(phase.ndim, 0)
        assert_almost_equal(strength, targ_strength)

    def test_opposite_2dperiod(self):
        events = np.array([0, .25, .5, .75])
        period = [1.] * 10
        targ_strength = [0.] * 10

        strength, phase = vectorstrength(events, period)

        assert_equal(strength.ndim, 1)
        assert_equal(phase.ndim, 1)
        assert_almost_equal(strength, targ_strength)

    def test_2d_events_ValueError(self):
        events = np.array([[1, 2]])
        period = 1.
        assert_raises(ValueError, vectorstrength, events, period)

    def test_2d_period_ValueError(self):
        events = 1.
        period = np.array([[1]])
        assert_raises(ValueError, vectorstrength, events, period)

    def test_zero_period_ValueError(self):
        events = 1.
        period = 0
        assert_raises(ValueError, vectorstrength, events, period)

    def test_negative_period_ValueError(self):
        events = 1.
        period = -1
        assert_raises(ValueError, vectorstrength, events, period)


def cast_tf2sos(b, a):
    """Convert TF2SOS, casting to complex128 and back to the original dtype."""
    # tf2sos does not support all of the dtypes that we want to check, e.g.:
    #
    #     TypeError: array type complex256 is unsupported in linalg
    #
    # so let's cast, convert, and cast back -- should be fine for the
    # systems and precisions we are testing.
    dtype = np.asarray(b).dtype
    b = np.array(b, np.complex128)
    a = np.array(a, np.complex128)
    return tf2sos(b, a).astype(dtype)


def assert_allclose_cast(actual, desired, rtol=1e-7, atol=0):
    """Wrap assert_allclose while casting object arrays."""
    if actual.dtype.kind == 'O':
        dtype = np.array(actual.flat[0]).dtype
        actual, desired = actual.astype(dtype), desired.astype(dtype)
    assert_allclose(actual, desired, rtol, atol)


@pytest.mark.parametrize('func', (sosfilt, lfilter))
def test_nonnumeric_dtypes(func):
    x = [Decimal(1), Decimal(2), Decimal(3)]
    b = [Decimal(1), Decimal(2), Decimal(3)]
    a = [Decimal(1), Decimal(2), Decimal(3)]
    x = np.array(x)
    assert x.dtype.kind == 'O'
    desired = lfilter(np.array(b, float), np.array(a, float), x.astype(float))
    if func is sosfilt:
        actual = sosfilt([b + a], x)
    else:
        actual = lfilter(b, a, x)
    assert all(isinstance(x, Decimal) for x in actual)
    assert_allclose(actual.astype(float), desired.astype(float))
    # Degenerate cases
    if func is lfilter:
        args = [1., 1.]
    else:
        args = [tf2sos(1., 1.)]

    with pytest.raises(ValueError, match='must be at least 1-D'):
        func(*args, x=1.)


@pytest.mark.parametrize('dt', 'fdgFDGO')
class TestSOSFilt:

    # The test_rank* tests are pulled from _TestLinearFilter
    def test_rank1(self, dt):
        x = np.linspace(0, 5, 6).astype(dt)
        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, -0.5]).astype(dt)

        # Test simple IIR
        y_r = np.array([0, 2, 4, 6, 8, 10.]).astype(dt)
        sos = cast_tf2sos(b, a)
        assert_array_almost_equal(sosfilt(cast_tf2sos(b, a), x), y_r)

        # Test simple FIR
        b = np.array([1, 1]).astype(dt)
        # NOTE: This was changed (rel. to TestLinear...) to add a pole @zero:
        a = np.array([1, 0]).astype(dt)
        y_r = np.array([0, 1, 3, 5, 7, 9.]).astype(dt)
        assert_array_almost_equal(sosfilt(cast_tf2sos(b, a), x), y_r)

        b = [1, 1, 0]
        a = [1, 0, 0]
        x = np.ones(8)
        sos = np.concatenate((b, a))
        sos.shape = (1, 6)
        y = sosfilt(sos, x)
        assert_allclose(y, [1, 2, 2, 2, 2, 2, 2, 2])

    def test_rank2(self, dt):
        shape = (4, 3)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
        x = x.astype(dt)

        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, 0.5]).astype(dt)

        y_r2_a0 = np.array([[0, 2, 4], [6, 4, 2], [0, 2, 4], [6, 4, 2]],
                           dtype=dt)

        y_r2_a1 = np.array([[0, 2, 0], [6, -4, 6], [12, -10, 12],
                            [18, -16, 18]], dtype=dt)

        y = sosfilt(cast_tf2sos(b, a), x, axis=0)
        assert_array_almost_equal(y_r2_a0, y)

        y = sosfilt(cast_tf2sos(b, a), x, axis=1)
        assert_array_almost_equal(y_r2_a1, y)

    def test_rank3(self, dt):
        shape = (4, 3, 2)
        x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)

        b = np.array([1, -1]).astype(dt)
        a = np.array([0.5, 0.5]).astype(dt)

        # Test last axis
        y = sosfilt(cast_tf2sos(b, a), x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                assert_array_almost_equal(y[i, j], lfilter(b, a, x[i, j]))

    def test_initial_conditions(self, dt):
        b1, a1 = signal.butter(2, 0.25, 'low')
        b2, a2 = signal.butter(2, 0.75, 'low')
        b3, a3 = signal.butter(2, 0.75, 'low')
        b = np.convolve(np.convolve(b1, b2), b3)
        a = np.convolve(np.convolve(a1, a2), a3)
        sos = np.array((np.r_[b1, a1], np.r_[b2, a2], np.r_[b3, a3]))

        x = np.random.rand(50).astype(dt)

        # Stopping filtering and continuing
        y_true, zi = lfilter(b, a, x[:20], zi=np.zeros(6))
        y_true = np.r_[y_true, lfilter(b, a, x[20:], zi=zi)[0]]
        assert_allclose_cast(y_true, lfilter(b, a, x))

        y_sos, zi = sosfilt(sos, x[:20], zi=np.zeros((3, 2)))
        y_sos = np.r_[y_sos, sosfilt(sos, x[20:], zi=zi)[0]]
        assert_allclose_cast(y_true, y_sos)

        # Use a step function
        zi = sosfilt_zi(sos)
        x = np.ones(8, dt)
        y, zf = sosfilt(sos, x, zi=zi)

        assert_allclose_cast(y, np.ones(8))
        assert_allclose_cast(zf, zi)

        # Initial condition shape matching
        x.shape = (1, 1) + x.shape  # 3D
        assert_raises(ValueError, sosfilt, sos, x, zi=zi)
        zi_nd = zi.copy()
        zi_nd.shape = (zi.shape[0], 1, 1, zi.shape[-1])
        assert_raises(ValueError, sosfilt, sos, x,
                      zi=zi_nd[:, :, :, [0, 1, 1]])
        y, zf = sosfilt(sos, x, zi=zi_nd)
        assert_allclose_cast(y[0, 0], np.ones(8))
        assert_allclose_cast(zf[:, 0, 0, :], zi)

    def test_initial_conditions_3d_axis1(self, dt):
        # Test the use of zi when sosfilt is applied to axis 1 of a 3-d input.

        # Input array is x.
        x = np.random.RandomState(159).randint(0, 5, size=(2, 15, 3))
        x = x.astype(dt)

        # Design a filter in ZPK format and convert to SOS
        zpk = signal.butter(6, 0.35, output='zpk')
        sos = zpk2sos(*zpk)
        nsections = sos.shape[0]

        # Filter along this axis.
        axis = 1

        # Initial conditions, all zeros.
        shp = list(x.shape)
        shp[axis] = 2
        shp = [nsections] + shp
        z0 = np.zeros(shp)

        # Apply the filter to x.
        yf, zf = sosfilt(sos, x, axis=axis, zi=z0)

        # Apply the filter to x in two stages.
        y1, z1 = sosfilt(sos, x[:, :5, :], axis=axis, zi=z0)
        y2, z2 = sosfilt(sos, x[:, 5:, :], axis=axis, zi=z1)

        # y should equal yf, and z2 should equal zf.
        y = np.concatenate((y1, y2), axis=axis)
        assert_allclose_cast(y, yf, rtol=1e-10, atol=1e-13)
        assert_allclose_cast(z2, zf, rtol=1e-10, atol=1e-13)

        # let's try the "step" initial condition
        zi = sosfilt_zi(sos)
        zi.shape = [nsections, 1, 2, 1]
        zi = zi * x[:, 0:1, :]
        y = sosfilt(sos, x, axis=axis, zi=zi)[0]
        # check it against the TF form
        b, a = zpk2tf(*zpk)
        zi = lfilter_zi(b, a)
        zi.shape = [1, zi.size, 1]
        zi = zi * x[:, 0:1, :]
        y_tf = lfilter(b, a, x, axis=axis, zi=zi)[0]
        assert_allclose_cast(y, y_tf, rtol=1e-10, atol=1e-13)

    def test_bad_zi_shape(self, dt):
        # The shape of zi is checked before using any values in the
        # arguments, so np.empty is fine for creating the arguments.
        x = np.empty((3, 15, 3), dt)
        sos = np.zeros((4, 6))
        zi = np.empty((4, 3, 3, 2))  # Correct shape is (4, 3, 2, 3)
        with pytest.raises(ValueError, match='should be all ones'):
            sosfilt(sos, x, zi=zi, axis=1)
        sos[:, 3] = 1.
        with pytest.raises(ValueError, match='Invalid zi shape'):
            sosfilt(sos, x, zi=zi, axis=1)

    def test_sosfilt_zi(self, dt):
        sos = signal.butter(6, 0.2, output='sos')
        zi = sosfilt_zi(sos)

        y, zf = sosfilt(sos, np.ones(40, dt), zi=zi)
        assert_allclose_cast(zf, zi, rtol=1e-13)

        # Expected steady state value of the step response of this filter:
        ss = np.prod(sos[:, :3].sum(axis=-1) / sos[:, 3:].sum(axis=-1))
        assert_allclose_cast(y, ss, rtol=1e-13)

        # zi as array-like
        _, zf = sosfilt(sos, np.ones(40, dt), zi=zi.tolist())
        assert_allclose_cast(zf, zi, rtol=1e-13)


class TestDeconvolve:

    def test_basic(self):
        # From docstring example
        original = [0, 1, 0, 0, 1, 1, 0, 0]
        impulse_response = [2, 1]
        recorded = [0, 2, 1, 0, 2, 3, 1, 0, 0]
        recovered, remainder = signal.deconvolve(recorded, impulse_response)
        assert_allclose(recovered, original)


class TestDetrend:

    def test_basic(self):
        detrended = detrend(array([1, 2, 3]))
        detrended_exact = array([0, 0, 0])
        assert_array_almost_equal(detrended, detrended_exact)

    def test_copy(self):
        x = array([1, 1.2, 1.5, 1.6, 2.4])
        copy_array = detrend(x, overwrite_data=False)
        inplace = detrend(x, overwrite_data=True)
        assert_array_almost_equal(copy_array, inplace)


class TestUniqueRoots:
    def test_real_no_repeat(self):
        p = [-1.0, -0.5, 0.3, 1.2, 10.0]
        unique, multiplicity = unique_roots(p)
        assert_almost_equal(unique, p, decimal=15)
        assert_equal(multiplicity, np.ones(len(p)))

    def test_real_repeat(self):
        p = [-1.0, -0.95, -0.89, -0.8, 0.5, 1.0, 1.05]

        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='min')
        assert_almost_equal(unique, [-1.0, -0.89, 0.5, 1.0], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='max')
        assert_almost_equal(unique, [-0.95, -0.8, 0.5, 1.05], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='avg')
        assert_almost_equal(unique, [-0.975, -0.845, 0.5, 1.025], decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

    def test_complex_no_repeat(self):
        p = [-1.0, 1.0j, 0.5 + 0.5j, -1.0 - 1.0j, 3.0 + 2.0j]
        unique, multiplicity = unique_roots(p)
        assert_almost_equal(unique, p, decimal=15)
        assert_equal(multiplicity, np.ones(len(p)))

    def test_complex_repeat(self):
        p = [-1.0, -1.0 + 0.05j, -0.95 + 0.15j, -0.90 + 0.15j, 0.0,
             0.5 + 0.5j, 0.45 + 0.55j]

        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='min')
        assert_almost_equal(unique, [-1.0, -0.95 + 0.15j, 0.0, 0.45 + 0.55j],
                            decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='max')
        assert_almost_equal(unique,
                            [-1.0 + 0.05j, -0.90 + 0.15j, 0.0, 0.5 + 0.5j],
                            decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

        unique, multiplicity = unique_roots(p, tol=1e-1, rtype='avg')
        assert_almost_equal(
            unique, [-1.0 + 0.025j, -0.925 + 0.15j, 0.0, 0.475 + 0.525j],
            decimal=15)
        assert_equal(multiplicity, [2, 2, 1, 2])

    def test_gh_4915(self):
        p = np.roots(np.convolve(np.ones(5), np.ones(5)))
        true_roots = [-(-1)**(1/5), (-1)**(4/5), -(-1)**(3/5), (-1)**(2/5)]

        unique, multiplicity = unique_roots(p)
        unique = np.sort(unique)

        assert_almost_equal(np.sort(unique), true_roots, decimal=7)
        assert_equal(multiplicity, [2, 2, 2, 2])

    def test_complex_roots_extra(self):
        unique, multiplicity = unique_roots([1.0, 1.0j, 1.0])
        assert_almost_equal(unique, [1.0, 1.0j], decimal=15)
        assert_equal(multiplicity, [2, 1])

        unique, multiplicity = unique_roots([1, 1 + 2e-9, 1e-9 + 1j], tol=0.1)
        assert_almost_equal(unique, [1.0, 1e-9 + 1.0j], decimal=15)
        assert_equal(multiplicity, [2, 1])

    def test_single_unique_root(self):
        p = np.random.rand(100) + 1j * np.random.rand(100)
        unique, multiplicity = unique_roots(p, 2)
        assert_almost_equal(unique, [np.min(p)], decimal=15)
        assert_equal(multiplicity, [100])
