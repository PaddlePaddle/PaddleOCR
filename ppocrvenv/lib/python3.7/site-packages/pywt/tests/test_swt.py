#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import warnings
from copy import deepcopy
from itertools import combinations, permutations
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
                           assert_raises, assert_array_equal, assert_warns)

import pywt
from pywt._extensions._swt import swt_axis

# Check that float32 and complex64 are preserved.  Other real types get
# converted to float64.
dtypes_in = [np.int8, np.float16, np.float32, np.float64, np.complex64,
             np.complex128]
dtypes_out = [np.float64, np.float32, np.float32, np.float64, np.complex64,
              np.complex128]

# tolerances used in accuracy comparisons
tol_single = 1e-6
tol_double = 1e-13

####
# 1d multilevel swt tests
####


def test_swt_decomposition():
    x = [3, 7, 1, 3, -2, 6, 4, 6]
    db1 = pywt.Wavelet('db1')
    atol = tol_double
    (cA3, cD3), (cA2, cD2), (cA1, cD1) = pywt.swt(x, db1, level=3)
    expected_cA1 = [7.07106781, 5.65685425, 2.82842712, 0.70710678,
                    2.82842712, 7.07106781, 7.07106781, 6.36396103]
    assert_allclose(cA1, expected_cA1, rtol=1e-8, atol=atol)
    expected_cD1 = [-2.82842712, 4.24264069, -1.41421356, 3.53553391,
                    -5.65685425, 1.41421356, -1.41421356, 2.12132034]
    assert_allclose(cD1, expected_cD1, rtol=1e-8, atol=atol)
    expected_cA2 = [7, 4.5, 4, 5.5, 7, 9.5, 10, 8.5]
    assert_allclose(cA2, expected_cA2, rtol=tol_double, atol=atol)
    expected_cD2 = [3, 3.5, 0, -4.5, -3, 0.5, 0, 0.5]
    assert_allclose(cD2, expected_cD2, rtol=tol_double, atol=atol)
    expected_cA3 = [9.89949494, ] * 8
    assert_allclose(cA3, expected_cA3, rtol=1e-8, atol=atol)
    expected_cD3 = [0.00000000, -3.53553391, -4.24264069, -2.12132034,
                    0.00000000, 3.53553391, 4.24264069, 2.12132034]
    assert_allclose(cD3, expected_cD3, rtol=1e-8, atol=atol)

    # level=1, start_level=1 decomposition should match level=2
    res = pywt.swt(cA1, db1, level=1, start_level=1)
    cA2, cD2 = res[0]
    assert_allclose(cA2, expected_cA2, rtol=tol_double, atol=atol)
    assert_allclose(cD2, expected_cD2, rtol=tol_double, atol=atol)

    coeffs = pywt.swt(x, db1)
    assert_(len(coeffs) == 3)
    assert_(pywt.swt_max_level(len(x)), 3)


def test_swt_max_level():
    # odd sized signal will warn about no levels of decomposition possible
    assert_warns(UserWarning, pywt.swt_max_level, 11)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        assert_equal(pywt.swt_max_level(11), 0)

    # no warnings when >= 1 level of decomposition possible
    assert_equal(pywt.swt_max_level(2), 1)     # divisible by 2**1
    assert_equal(pywt.swt_max_level(4*3), 2)    # divisible by 2**2
    assert_equal(pywt.swt_max_level(16), 4)    # divisible by 2**4
    assert_equal(pywt.swt_max_level(16*3), 4)  # divisible by 2**4


def test_swt_axis():
    x = [3, 7, 1, 3, -2, 6, 4, 6]

    db1 = pywt.Wavelet('db1')
    (cA2, cD2), (cA1, cD1) = pywt.swt(x, db1, level=2)

    # test cases use 2D arrays based on tiling x along an axis and then
    # calling swt along the other axis.
    for order in ['C', 'F']:
        # test SWT of 2D data along default axis (-1)
        x_2d = np.asarray(x).reshape((1, -1))
        x_2d = np.concatenate((x_2d, )*5, axis=0)
        if order == 'C':
            x_2d = np.ascontiguousarray(x_2d)
        elif order == 'F':
            x_2d = np.asfortranarray(x_2d)
        (cA2_2d, cD2_2d), (cA1_2d, cD1_2d) = pywt.swt(x_2d, db1, level=2)

        for c in [cA2_2d, cD2_2d, cA1_2d, cD1_2d]:
            assert_(c.shape == x_2d.shape)
        # each row should match the 1D result
        for row in cA1_2d:
            assert_array_equal(row, cA1)
        for row in cA2_2d:
            assert_array_equal(row, cA2)
        for row in cD1_2d:
            assert_array_equal(row, cD1)
        for row in cD2_2d:
            assert_array_equal(row, cD2)

        # test SWT of 2D data along other axis (0)
        x_2d = np.asarray(x).reshape((-1, 1))
        x_2d = np.concatenate((x_2d, )*5, axis=1)
        if order == 'C':
            x_2d = np.ascontiguousarray(x_2d)
        elif order == 'F':
            x_2d = np.asfortranarray(x_2d)
        (cA2_2d, cD2_2d), (cA1_2d, cD1_2d) = pywt.swt(x_2d, db1, level=2,
                                                      axis=0)

        for c in [cA2_2d, cD2_2d, cA1_2d, cD1_2d]:
            assert_(c.shape == x_2d.shape)
        # each column should match the 1D result
        for row in cA1_2d.transpose((1, 0)):
            assert_array_equal(row, cA1)
        for row in cA2_2d.transpose((1, 0)):
            assert_array_equal(row, cA2)
        for row in cD1_2d.transpose((1, 0)):
            assert_array_equal(row, cD1)
        for row in cD2_2d.transpose((1, 0)):
            assert_array_equal(row, cD2)

    # axis too large
    assert_raises(ValueError, pywt.swt, x, db1, level=2, axis=5)


def test_swt_iswt_integration():
    # This function performs a round-trip swt/iswt transform test on
    # all available types of wavelets in PyWavelets - except the
    # 'dmey' wavelet. The latter has been excluded because it does not
    # produce very precise results. This is likely due to the fact
    # that the 'dmey' wavelet is a discrete approximation of a
    # continuous wavelet. All wavelets are tested up to 3 levels. The
    # test validates neither swt or iswt as such, but it does ensure
    # that they are each other's inverse.

    max_level = 3
    wavelets = pywt.wavelist(kind='discrete')
    if 'dmey' in wavelets:
        # The 'dmey' wavelet seems to be a bit special - disregard it for now
        wavelets.remove('dmey')
    for current_wavelet_str in wavelets:
        current_wavelet = pywt.Wavelet(current_wavelet_str)
        input_length_power = int(np.ceil(np.log2(max(
            current_wavelet.dec_len,
            current_wavelet.rec_len))))
        input_length = 2**(input_length_power + max_level - 1)
        X = np.arange(input_length)
        for norm in [True, False]:
            if norm and not current_wavelet.orthogonal:
                # non-orthogonal wavelets to avoid warnings when norm=True
                continue
            for trim_approx in [True, False]:
                coeffs = pywt.swt(X, current_wavelet, max_level,
                                  trim_approx=trim_approx, norm=norm)
                Y = pywt.iswt(coeffs, current_wavelet, norm=norm)
                assert_allclose(Y, X, rtol=1e-5, atol=1e-7)


def test_swt_dtypes():
    wavelet = pywt.Wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        errmsg = "wrong dtype returned for {0} input".format(dt_in)

        # swt
        x = np.ones(8, dtype=dt_in)
        (cA2, cD2), (cA1, cD1) = pywt.swt(x, wavelet, level=2)
        assert_(cA2.dtype == cD2.dtype == cA1.dtype == cD1.dtype == dt_out,
                "swt: " + errmsg)

        # swt2
        x = np.ones((8, 8), dtype=dt_in)
        cA, (cH, cV, cD) = pywt.swt2(x, wavelet, level=1)[0]
        assert_(cA.dtype == cH.dtype == cV.dtype == cD.dtype == dt_out,
                "swt2: " + errmsg)


def test_swt_roundtrip_dtypes():
    # verify perfect reconstruction for all dtypes
    rstate = np.random.RandomState(5)
    wavelet = pywt.Wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        # swt, iswt
        x = rstate.standard_normal((8, )).astype(dt_in)
        c = pywt.swt(x, wavelet, level=2)
        xr = pywt.iswt(c, wavelet)
        assert_allclose(x, xr, rtol=1e-6, atol=1e-7)

        # swt2, iswt2
        x = rstate.standard_normal((8, 8)).astype(dt_in)
        c = pywt.swt2(x, wavelet, level=2)
        xr = pywt.iswt2(c, wavelet)
        assert_allclose(x, xr, rtol=1e-6, atol=1e-7)


def test_swt_default_level_by_axis():
    # make sure default number of levels matches the max level along the axis
    wav = 'db2'
    x = np.ones((2**3, 2**4, 2**5))
    for axis in (0, 1, 2):
        sdec = pywt.swt(x, wav, level=None, start_level=0, axis=axis)
        assert_equal(len(sdec), pywt.swt_max_level(x.shape[axis]))


def test_swt2_ndim_error():
    x = np.ones(8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        assert_raises(ValueError, pywt.swt2, x, 'haar', level=1)


@pytest.mark.slow
def test_swt2_iswt2_integration(wavelets=None):
    # This function performs a round-trip swt2/iswt2 transform test on
    # all available types of wavelets in PyWavelets - except the
    # 'dmey' wavelet. The latter has been excluded because it does not
    # produce very precise results. This is likely due to the fact
    # that the 'dmey' wavelet is a discrete approximation of a
    # continuous wavelet. All wavelets are tested up to 3 levels. The
    # test validates neither swt2 or iswt2 as such, but it does ensure
    # that they are each other's inverse.

    max_level = 3
    if wavelets is None:
        wavelets = pywt.wavelist(kind='discrete')
        if 'dmey' in wavelets:
            # The 'dmey' wavelet is a special case - disregard it for now
            wavelets.remove('dmey')
    for current_wavelet_str in wavelets:
        current_wavelet = pywt.Wavelet(current_wavelet_str)
        input_length_power = int(np.ceil(np.log2(max(
            current_wavelet.dec_len,
            current_wavelet.rec_len))))
        input_length = 2**(input_length_power + max_level - 1)
        X = np.arange(input_length**2).reshape(input_length, input_length)

        for norm in [True, False]:
            if norm and not current_wavelet.orthogonal:
                # non-orthogonal wavelets to avoid warnings when norm=True
                continue
            for trim_approx in [True, False]:
                coeffs = pywt.swt2(X, current_wavelet, max_level,
                                   trim_approx=trim_approx, norm=norm)
                Y = pywt.iswt2(coeffs, current_wavelet, norm=norm)
                assert_allclose(Y, X, rtol=1e-5, atol=1e-5)


def test_swt2_iswt2_quick():
    test_swt2_iswt2_integration(wavelets=['db1', ])


def test_swt2_iswt2_non_square(wavelets=None):
    for nrows in [8, 16, 48]:
        X = np.arange(nrows*32).reshape(nrows, 32)
        current_wavelet = 'db1'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            coeffs = pywt.swt2(X, current_wavelet, level=2)
            Y = pywt.iswt2(coeffs, current_wavelet)
        assert_allclose(Y, X, rtol=tol_single, atol=tol_single)


def test_swt2_axes():
    atol = 1e-14
    current_wavelet = pywt.Wavelet('db2')
    input_length_power = int(np.ceil(np.log2(max(
        current_wavelet.dec_len,
        current_wavelet.rec_len))))
    input_length = 2**(input_length_power)
    X = np.arange(input_length**2).reshape(input_length, input_length)

    (cA1, (cH1, cV1, cD1)) = pywt.swt2(X, current_wavelet, level=1)[0]
    # opposite order
    (cA2, (cH2, cV2, cD2)) = pywt.swt2(X, current_wavelet, level=1,
                                       axes=(1, 0))[0]
    assert_allclose(cA1, cA2, atol=atol)
    assert_allclose(cH1, cV2, atol=atol)
    assert_allclose(cV1, cH2, atol=atol)
    assert_allclose(cD1, cD2, atol=atol)

    # reverify iswt2 restores the orginal data
    r1 = pywt.iswt2([cA1, (cH1, cV1, cD1)], current_wavelet)
    assert_allclose(X, r1, atol=atol)
    r2 = pywt.iswt2([cA2, (cH2, cV2, cD2)], current_wavelet, axes=(1, 0))
    assert_allclose(X, r2, atol=atol)

    # duplicate axes not allowed
    assert_raises(ValueError, pywt.swt2, X, current_wavelet, 1,
                  axes=(0, 0))
    # too few axes
    assert_raises(ValueError, pywt.swt2, X, current_wavelet, 1, axes=(0, ))


def test_swtn_axes():
    atol = 1e-14
    current_wavelet = pywt.Wavelet('db2')
    input_length_power = int(np.ceil(np.log2(max(
        current_wavelet.dec_len,
        current_wavelet.rec_len))))
    input_length = 2**(input_length_power)
    X = np.arange(input_length**2).reshape(input_length, input_length)
    coeffs = pywt.swtn(X, current_wavelet, level=1, axes=None)[0]
    # opposite order
    coeffs2 = pywt.swtn(X, current_wavelet, level=1, axes=(1, 0))[0]
    assert_allclose(coeffs['aa'], coeffs2['aa'], atol=atol)
    assert_allclose(coeffs['ad'], coeffs2['da'], atol=atol)
    assert_allclose(coeffs['da'], coeffs2['ad'], atol=atol)
    assert_allclose(coeffs['dd'], coeffs2['dd'], atol=atol)

    # 0-level transform
    empty = pywt.swtn(X, current_wavelet, level=0)
    assert_equal(empty, [])

    # duplicate axes not allowed
    assert_raises(ValueError, pywt.swtn, X, current_wavelet, 1, axes=(0, 0))

    # data.ndim = 0
    assert_raises(ValueError, pywt.swtn, np.asarray([]), current_wavelet, 1)

    # start_level too large
    assert_raises(ValueError, pywt.swtn, X, current_wavelet,
                  level=1, start_level=2)

    # level < 1 in swt_axis call
    assert_raises(ValueError, swt_axis, X, current_wavelet, level=0,
                  start_level=0)
    # odd-sized data not allowed
    assert_raises(ValueError, swt_axis, X[:-1, :], current_wavelet, level=0,
                  start_level=0, axis=0)


@pytest.mark.slow
def test_swtn_iswtn_integration(wavelets=None):
    # This function performs a round-trip swtn/iswtn transform for various
    # possible combinations of:
    #   1.) 1 out of 2 axes of a 2D array
    #   2.) 2 out of 3 axes of a 3D array
    #
    # To keep test time down, only wavelets of length <= 8 are run.
    #
    # This test does not validate swtn or iswtn individually, but only
    # confirms that iswtn yields an (almost) perfect reconstruction of swtn.
    max_level = 3
    if wavelets is None:
        wavelets = pywt.wavelist(kind='discrete')
        if 'dmey' in wavelets:
            # The 'dmey' wavelet is a special case - disregard it for now
            wavelets.remove('dmey')
    for ndim_transform in range(1, 3):
        ndim = ndim_transform + 1
        for axes in combinations(range(ndim), ndim_transform):
            for current_wavelet_str in wavelets:
                wav = pywt.Wavelet(current_wavelet_str)
                if wav.dec_len > 8:
                    continue  # avoid excessive test duration
                input_length_power = int(np.ceil(np.log2(max(
                    wav.dec_len,
                    wav.rec_len))))
                N = 2**(input_length_power + max_level - 1)
                X = np.arange(N**ndim).reshape((N, )*ndim)

                for norm in [True, False]:
                    if norm and not wav.orthogonal:
                        # non-orthogonal wavelets to avoid warnings
                        continue
                    for trim_approx in [True, False]:
                        coeffs = pywt.swtn(X, wav, max_level, axes=axes,
                                           trim_approx=trim_approx, norm=norm)
                        coeffs_copy = deepcopy(coeffs)
                        Y = pywt.iswtn(coeffs, wav, axes=axes, norm=norm)
                        assert_allclose(Y, X, rtol=1e-5, atol=1e-5)

                # verify the inverse transform didn't modify any coeffs
                for c, c2 in zip(coeffs, coeffs_copy):
                    for k, v in c.items():
                        assert_array_equal(c2[k], v)


def test_swtn_iswtn_quick():
    test_swtn_iswtn_integration(wavelets=['db1', ])


def test_iswtn_errors():
    x = np.arange(8**3).reshape(8, 8, 8)
    max_level = 2
    axes = (0, 1)
    w = pywt.Wavelet('db1')
    coeffs = pywt.swtn(x, w, max_level, axes=axes)

    # more axes than dimensions transformed
    assert_raises(ValueError, pywt.iswtn, coeffs, w, axes=(0, 1, 2))
    # duplicate axes not allowed
    assert_raises(ValueError, pywt.iswtn, coeffs, w, axes=(0, 0))
    # mismatched coefficient size
    coeffs[0]['da'] = coeffs[0]['da'][:-1, :]
    assert_raises(RuntimeError, pywt.iswtn, coeffs, w, axes=axes)


def test_swtn_iswtn_unique_shape_per_axis():
    # test case for gh-460
    _shape = (1, 48, 32)  # unique shape per axis
    wav = 'sym2'
    max_level = 3
    rstate = np.random.RandomState(0)
    for shape in permutations(_shape):
        # transform only along the non-singleton axes
        axes = [ax for ax, s in enumerate(shape) if s != 1]
        x = rstate.standard_normal(shape)
        c = pywt.swtn(x, wav, max_level, axes=axes)
        r = pywt.iswtn(c, wav, axes=axes)
        assert_allclose(x, r, rtol=1e-10, atol=1e-10)


def test_per_axis_wavelets():
    # tests seperate wavelet for each axis.
    rstate = np.random.RandomState(1234)
    data = rstate.randn(16, 16, 16)
    level = 3

    # wavelet can be a string or wavelet object
    wavelets = (pywt.Wavelet('haar'), 'sym2', 'db4')

    coefs = pywt.swtn(data, wavelets, level=level)
    assert_allclose(pywt.iswtn(coefs, wavelets), data, atol=1e-14)

    # 1-tuple also okay
    coefs = pywt.swtn(data, wavelets[:1], level=level)
    assert_allclose(pywt.iswtn(coefs, wavelets[:1]), data, atol=1e-14)

    # length of wavelets doesn't match the length of axes
    assert_raises(ValueError, pywt.swtn, data, wavelets[:2], level)
    assert_raises(ValueError, pywt.iswtn, coefs, wavelets[:2])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        # swt2/iswt2 also support per-axis wavelets/modes
        data2 = data[..., 0]
        coefs2 = pywt.swt2(data2, wavelets[:2], level)
        assert_allclose(pywt.iswt2(coefs2, wavelets[:2]), data2, atol=1e-14)


def test_error_on_continuous_wavelet():
    # A ValueError is raised if a Continuous wavelet is selected
    data = np.ones((16, 16))
    for dec_func, rec_func in zip([pywt.swt, pywt.swt2, pywt.swtn],
                                  [pywt.iswt, pywt.iswt2, pywt.iswtn]):
        for cwave in ['morl', pywt.DiscreteContinuousWavelet('morl')]:
            assert_raises(ValueError, dec_func, data, wavelet=cwave,
                          level=3)

            c = dec_func(data, 'db1', level=3)
            assert_raises(ValueError, rec_func, c, wavelet=cwave)


def test_iswt_mixed_dtypes():
    # Mixed precision inputs give double precision output
    x_real = np.arange(16).astype(np.float64)
    x_complex = x_real + 1j*x_real
    wav = 'sym2'
    for dtype1, dtype2 in [(np.float64, np.float32),
                           (np.float32, np.float64),
                           (np.float16, np.float64),
                           (np.complex128, np.complex64),
                           (np.complex64, np.complex128)]:

        if dtype1 in [np.complex64, np.complex128]:
            x = x_complex
            output_dtype = np.complex128
        else:
            x = x_real
            output_dtype = np.float64

        coeffs = pywt.swt(x, wav, 2)
        # different precision for the approximation coefficients
        coeffs[0] = [coeffs[0][0].astype(dtype1),
                     coeffs[0][1].astype(dtype2)]
        y = pywt.iswt(coeffs, wav)
        assert_equal(output_dtype, y.dtype)
        assert_allclose(y, x, rtol=1e-3, atol=1e-3)


def test_iswt2_mixed_dtypes():
    # Mixed precision inputs give double precision output
    rstate = np.random.RandomState(0)
    x_real = rstate.randn(8, 8)
    x_complex = x_real + 1j*x_real
    wav = 'sym2'
    for dtype1, dtype2 in [(np.float64, np.float32),
                           (np.float32, np.float64),
                           (np.float16, np.float64),
                           (np.complex128, np.complex64),
                           (np.complex64, np.complex128)]:

        if dtype1 in [np.complex64, np.complex128]:
            x = x_complex
            output_dtype = np.complex128
        else:
            x = x_real
            output_dtype = np.float64

        coeffs = pywt.swt2(x, wav, 2)
        # different precision for the approximation coefficients
        coeffs[0] = [coeffs[0][0].astype(dtype1),
                     tuple([c.astype(dtype2) for c in coeffs[0][1]])]
        y = pywt.iswt2(coeffs, wav)
        assert_equal(output_dtype, y.dtype)
        assert_allclose(y, x, rtol=1e-3, atol=1e-3)


def test_iswtn_mixed_dtypes():
    # Mixed precision inputs give double precision output
    rstate = np.random.RandomState(0)
    x_real = rstate.randn(8, 8, 8)
    x_complex = x_real + 1j*x_real
    wav = 'sym2'
    for dtype1, dtype2 in [(np.float64, np.float32),
                           (np.float32, np.float64),
                           (np.float16, np.float64),
                           (np.complex128, np.complex64),
                           (np.complex64, np.complex128)]:

        if dtype1 in [np.complex64, np.complex128]:
            x = x_complex
            output_dtype = np.complex128
        else:
            x = x_real
            output_dtype = np.float64

        coeffs = pywt.swtn(x, wav, 2)
        # different precision for the approximation coefficients
        a = coeffs[0].pop('a' * x.ndim)
        a = a.astype(dtype1)
        coeffs[0] = {k: c.astype(dtype2) for k, c in coeffs[0].items()}
        coeffs[0]['a' * x.ndim] = a
        y = pywt.iswtn(coeffs, wav)
        assert_equal(output_dtype, y.dtype)
        assert_allclose(y, x, rtol=1e-3, atol=1e-3)


def test_swt_zero_size_axes():
    # raise on empty input array
    assert_raises(ValueError, pywt.swt, [], 'db2')

    # >1D case uses a different code path so check there as well
    x = np.ones((1, 4))[0:0, :]  # 2D with a size zero axis
    assert_raises(ValueError, pywt.swtn, x, 'db2', level=1, axes=(0,))


def test_swt_variance_and_energy_preservation():
    """Verify that the 1D SWT partitions variance among the coefficients."""
    # When norm is True and the wavelet is orthogonal, the sum of the
    # variances of the coefficients should equal the variance of the signal.
    wav = 'db2'
    rstate = np.random.RandomState(5)
    x = rstate.randn(256)
    coeffs = pywt.swt(x, wav, trim_approx=True, norm=True)
    variances = [np.var(c) for c in coeffs]
    assert_allclose(np.sum(variances), np.var(x))

    # also verify L2-norm energy preservation property
    assert_allclose(np.linalg.norm(x),
                    np.linalg.norm(np.concatenate(coeffs)))

    # non-orthogonal wavelet with norm=True raises a warning
    assert_warns(UserWarning, pywt.swt, x, 'bior2.2', norm=True)


def test_swt2_variance_and_energy_preservation():
    """Verify that the 2D SWT partitions variance among the coefficients."""
    # When norm is True and the wavelet is orthogonal, the sum of the
    # variances of the coefficients should equal the variance of the signal.
    wav = 'db2'
    rstate = np.random.RandomState(5)
    x = rstate.randn(64, 64)
    coeffs = pywt.swt2(x, wav, level=4, trim_approx=True, norm=True)
    coeff_list = [coeffs[0].ravel()]
    for d in coeffs[1:]:
        for v in d:
            coeff_list.append(v.ravel())
    variances = [np.var(v) for v in coeff_list]
    assert_allclose(np.sum(variances), np.var(x))

    # also verify L2-norm energy preservation property
    assert_allclose(np.linalg.norm(x),
                    np.linalg.norm(np.concatenate(coeff_list)))

    # non-orthogonal wavelet with norm=True raises a warning
    assert_warns(UserWarning, pywt.swt2, x, 'bior2.2', level=4, norm=True)


def test_swtn_variance_and_energy_preservation():
    """Verify that the nD SWT partitions variance among the coefficients."""
    # When norm is True and the wavelet is orthogonal, the sum of the
    # variances of the coefficients should equal the variance of the signal.
    wav = 'db2'
    rstate = np.random.RandomState(5)
    x = rstate.randn(64, 64)
    coeffs = pywt.swtn(x, wav, level=4, trim_approx=True, norm=True)
    coeff_list = [coeffs[0].ravel()]
    for d in coeffs[1:]:
        for k, v in d.items():
            coeff_list.append(v.ravel())
    variances = [np.var(v) for v in coeff_list]
    assert_allclose(np.sum(variances), np.var(x))

    # also verify L2-norm energy preservation property
    assert_allclose(np.linalg.norm(x),
                    np.linalg.norm(np.concatenate(coeff_list)))

    # non-orthogonal wavelet with norm=True raises a warning
    assert_warns(UserWarning, pywt.swtn, x, 'bior2.2', level=4, norm=True)


def test_swt_ravel_and_unravel():
    # When trim_approx=True, all swt functions can user pywt.ravel_coeffs
    for ndim, _swt, _iswt, ravel_type in [
            (1, pywt.swt, pywt.iswt, 'swt'),
            (2, pywt.swt2, pywt.iswt2, 'swt2'),
            (3, pywt.swtn, pywt.iswtn, 'swtn')]:
        x = np.ones((16, ) * ndim)
        c = _swt(x, 'sym2', level=3, trim_approx=True)
        arr, slices, shapes = pywt.ravel_coeffs(c)
        c = pywt.unravel_coeffs(arr, slices, shapes, output_format=ravel_type)
        r = _iswt(c, 'sym2')
        assert_allclose(x, r)
