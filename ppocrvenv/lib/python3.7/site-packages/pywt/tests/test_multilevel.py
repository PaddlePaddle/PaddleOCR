#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import warnings
from itertools import combinations
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_allclose, assert_,
                           assert_equal, assert_raises, assert_raises_regex,
                           assert_array_equal, assert_warns)
import pywt
# Check that float32, float64, complex64, complex128 are preserved.
# Other real types get converted to float64.
# complex256 gets converted to complex128
dtypes_in = [np.int8, np.float16, np.float32, np.float64, np.complex64,
             np.complex128]
dtypes_out = [np.float64, np.float32, np.float32, np.float64, np.complex64,
              np.complex128]

# tolerances used in accuracy comparisons
tol_single = 1e-6
tol_double = 1e-13
dtypes_and_tolerances = [(np.float16, tol_single), (np.float32, tol_single),
                         (np.float64, tol_double), (np.int8, tol_double),
                         (np.complex64, tol_single),
                         (np.complex128, tol_double)]

# test complex256 as well if it is available
try:
    dtypes_in += [np.complex256, ]
    dtypes_out += [np.complex128, ]
    dtypes_and_tolerances += [(np.complex256, tol_double), ]
except AttributeError:
    pass


# determine which wavelets to test
wavelist = pywt.wavelist()
if 'dmey' in wavelist:
    # accuracy is very low for dmey, so omit it
    wavelist.remove('dmey')

# removing wavelets with dwt_possible == False
del_list = []
for wavelet in wavelist:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        if not isinstance(pywt.DiscreteContinuousWavelet(wavelet),
                          pywt.Wavelet):
            del_list.append(wavelet)
for del_ind in del_list:
    wavelist.remove(del_ind)


####
# 1d multilevel dwt tests
####

def test_wavedec():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    db1 = pywt.Wavelet('db1')
    cA3, cD3, cD2, cD1 = pywt.wavedec(x, db1)
    assert_almost_equal(cA3, [8.83883476])
    assert_almost_equal(cD3, [-0.35355339])
    assert_allclose(cD2, [4., -3.5])
    assert_allclose(cD1, [-2.82842712, 0, -4.94974747, -1.41421356])
    assert_(pywt.dwt_max_level(len(x), db1) == 3)


def test_waverec_invalid_inputs():
    # input must be list or tuple
    assert_raises(ValueError, pywt.waverec, np.ones(8), 'haar')

    # input list cannot be empty
    assert_raises(ValueError, pywt.waverec, [], 'haar')

    # 'array_to_coeffs must specify 'output_format' to perform waverec
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    coeffs = pywt.wavedec(x, 'db1')
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    coeffs_from_arr = pywt.array_to_coeffs(arr, coeff_slices)
    message = "Unexpected detail coefficient type"
    assert_raises_regex(ValueError, message, pywt.waverec, coeffs_from_arr,
                        'haar')


def test_waverec_accuracies():
    rstate = np.random.RandomState(1234)
    x0 = rstate.randn(8)
    for dt, tol in dtypes_and_tolerances:
        x = x0.astype(dt)
        if np.iscomplexobj(x):
            x += 1j*rstate.randn(8).astype(x.real.dtype)
        coeffs = pywt.wavedec(x, 'db1')
        assert_allclose(pywt.waverec(coeffs, 'db1'), x, atol=tol, rtol=tol)


def test_waverec_none():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    coeffs = pywt.wavedec(x, 'db1')

    # set some coefficients to None
    coeffs[2] = None
    coeffs[0] = None
    assert_(pywt.waverec(coeffs, 'db1').size, len(x))


def test_waverec_odd_length():
    x = [3, 7, 1, 1, -2, 5]
    coeffs = pywt.wavedec(x, 'db1')
    assert_allclose(pywt.waverec(coeffs, 'db1'), x, rtol=1e-12)


def test_waverec_complex():
    x = np.array([3, 7, 1, 1, -2, 5, 4, 6])
    x = x + 1j
    coeffs = pywt.wavedec(x, 'db1')
    assert_allclose(pywt.waverec(coeffs, 'db1'), x, rtol=1e-12)


def test_multilevel_dtypes_1d():
    # only checks that the result is of the expected type
    wavelet = pywt.Wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        # wavedec, waverec
        x = np.ones(8, dtype=dt_in)
        errmsg = "wrong dtype returned for {0} input".format(dt_in)

        coeffs = pywt.wavedec(x, wavelet, level=2)
        for c in coeffs:
            assert_(c.dtype == dt_out, "wavedec: " + errmsg)
        x_roundtrip = pywt.waverec(coeffs, wavelet)
        assert_(x_roundtrip.dtype == dt_out, "waverec: " + errmsg)


def test_waverec_all_wavelets_modes():
    # test 2D case using all wavelets and modes
    rstate = np.random.RandomState(1234)
    r = rstate.randn(80)
    for wavelet in wavelist:
        for mode in pywt.Modes.modes:
            coeffs = pywt.wavedec(r, wavelet, mode=mode)
            assert_allclose(pywt.waverec(coeffs, wavelet, mode=mode),
                            r, rtol=tol_single, atol=tol_single)

####
# 2d multilevel dwt function tests
####


def test_waverec2_accuracies():
    rstate = np.random.RandomState(1234)
    x0 = rstate.randn(4, 4)
    for dt, tol in dtypes_and_tolerances:
        x = x0.astype(dt)
        if np.iscomplexobj(x):
            x += 1j*rstate.randn(4, 4).astype(x.real.dtype)
        coeffs = pywt.wavedec2(x, 'db1')
        assert_(len(coeffs) == 3)
        assert_allclose(pywt.waverec2(coeffs, 'db1'), x, atol=tol, rtol=tol)


def test_multilevel_dtypes_2d():
    wavelet = pywt.Wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        # wavedec2, waverec2
        x = np.ones((8, 8), dtype=dt_in)
        errmsg = "wrong dtype returned for {0} input".format(dt_in)
        cA, coeffsD2, coeffsD1 = pywt.wavedec2(x, wavelet, level=2)
        assert_(cA.dtype == dt_out, "wavedec2: " + errmsg)
        for c in coeffsD1:
            assert_(c.dtype == dt_out, "wavedec2: " + errmsg)
        for c in coeffsD2:
            assert_(c.dtype == dt_out, "wavedec2: " + errmsg)
        x_roundtrip = pywt.waverec2([cA, coeffsD2, coeffsD1], wavelet)
        assert_(x_roundtrip.dtype == dt_out, "waverec2: " + errmsg)


@pytest.mark.slow
def test_waverec2_all_wavelets_modes():
    # test 2D case using all wavelets and modes
    rstate = np.random.RandomState(1234)
    r = rstate.randn(80, 96)
    for wavelet in wavelist:
        for mode in pywt.Modes.modes:
            coeffs = pywt.wavedec2(r, wavelet, mode=mode)
            assert_allclose(pywt.waverec2(coeffs, wavelet, mode=mode),
                            r, rtol=tol_single, atol=tol_single)


def test_wavedec2_complex():
    data = np.ones((4, 4)) + 1j
    coeffs = pywt.wavedec2(data, 'db1')
    assert_(len(coeffs) == 3)
    assert_allclose(pywt.waverec2(coeffs, 'db1'), data, rtol=1e-12)


def test_wavedec2_invalid_inputs():
    # input array has too few dimensions
    data = np.ones(4)
    assert_raises(ValueError, pywt.wavedec2, data, 'haar')


def test_waverec2_invalid_inputs():
    # input must be list or tuple
    assert_raises(ValueError, pywt.waverec2, np.ones((8, 8)), 'haar')

    # input list cannot be empty
    assert_raises(ValueError, pywt.waverec2, [], 'haar')

    # coefficients from a difference decomposition used as input
    for dec_func in [pywt.wavedec, pywt.wavedecn]:
        coeffs = dec_func(np.ones((8, 8)), 'haar')
        message = "Unexpected detail coefficient type"
        assert_raises_regex(ValueError, message, pywt.waverec2, coeffs,
                            'haar')


def test_waverec2_coeff_shape_mismatch():
    x = np.ones((8, 8))
    coeffs = pywt.wavedec2(x, 'db1')

    # introduce a shape mismatch in the coefficients
    coeffs = list(coeffs)
    coeffs[1] = list(coeffs[1])
    coeffs[1][1] = np.zeros((16, 1))
    assert_raises(ValueError, pywt.waverec2, coeffs, 'db1')


def test_waverec2_odd_length():
    x = np.ones((10, 6))
    coeffs = pywt.wavedec2(x, 'db1')
    assert_allclose(pywt.waverec2(coeffs, 'db1'), x, rtol=1e-12)


def test_waverec2_none_coeffs():
    x = np.arange(24).reshape(6, 4)
    coeffs = pywt.wavedec2(x, 'db1')
    coeffs[1] = (None, None, None)
    assert_(x.shape == pywt.waverec2(coeffs, 'db1').shape)

####
# nd multilevel dwt function tests
####


def test_waverecn():
    rstate = np.random.RandomState(1234)
    # test 1D through 4D cases
    for nd in range(1, 5):
        x = rstate.randn(*(4, )*nd)
        coeffs = pywt.wavedecn(x, 'db1')
        assert_(len(coeffs) == 3)
        assert_allclose(pywt.waverecn(coeffs, 'db1'), x, rtol=tol_double)


def test_waverecn_empty_coeff():
    coeffs = [np.ones((2, 2, 2)), {}, {}]
    assert_equal(pywt.waverecn(coeffs, 'db1').shape, (8, 8, 8))

    assert_equal(pywt.waverecn(coeffs, 'db1').shape, (8, 8, 8))
    coeffs = [np.ones((2, 2, 2)), {}, {'daa': np.ones((4, 4, 4))}]

    coeffs = [np.ones((2, 2, 2)), {}, {}, {'daa': np.ones((8, 8, 8))}]
    assert_equal(pywt.waverecn(coeffs, 'db1').shape, (16, 16, 16))


def test_waverecn_invalid_coeffs():
    # approximation coeffs as None and no valid detail oeffs
    coeffs = [None, {}]
    assert_raises(ValueError, pywt.waverecn, coeffs, 'db1')

    # use of None for a coefficient value
    coeffs = [np.ones((2, 2, 2)), {}, {'daa': None}, ]
    assert_raises(ValueError, pywt.waverecn, coeffs, 'db1')

    # invalid key names in coefficient list
    coeffs = [np.ones((4, 4, 4)), {'daa': np.ones((4, 4, 4)),
                                   'foo': np.ones((4, 4, 4))}]
    assert_raises(ValueError, pywt.waverecn, coeffs, 'db1')

    # mismatched key name lengths
    coeffs = [np.ones((4, 4, 4)), {'daa': np.ones((4, 4, 4)),
                                   'da': np.ones((4, 4, 4))}]
    assert_raises(ValueError, pywt.waverecn, coeffs, 'db1')

    # key name lengths don't match the array dimensions
    coeffs = [[[[1.0]]], {'ad': [[[0.0]]], 'da': [[[0.0]]], 'dd': [[[0.0]]]}]
    assert_raises(ValueError, pywt.waverecn, coeffs, 'db1')

    # input list cannot be empty
    assert_raises(ValueError, pywt.waverecn, [], 'haar')


def test_waverecn_invalid_inputs():

    # coefficients from a difference decomposition used as input
    for dec_func in [pywt.wavedec, pywt.wavedec2]:
        coeffs = dec_func(np.ones((8, 8)), 'haar')
        message = "Unexpected detail coefficient type"
        assert_raises_regex(ValueError, message, pywt.waverecn, coeffs,
                            'haar')


def test_waverecn_lists():
    # support coefficient arrays specified as lists instead of arrays
    coeffs = [[[1.0]], {'ad': [[0.0]], 'da': [[0.0]], 'dd': [[0.0]]}]
    assert_equal(pywt.waverecn(coeffs, 'db1').shape, (2, 2))


def test_waverecn_invalid_coeffs2():
    # shape mismatch should raise an error
    coeffs = [np.ones((4, 4, 4)), {'ada': np.ones((4, 4))}]
    assert_raises(ValueError, pywt.waverecn, coeffs, 'db1')


def test_wavedecn_invalid_inputs():
    # input array has too few dimensions
    data = np.array(0)
    assert_raises(ValueError, pywt.wavedecn, data, 'haar')

    # invalid number of levels
    data = np.ones(16)
    assert_raises(ValueError, pywt.wavedecn, data, 'haar', level=-1)


def test_wavedecn_many_levels():
    # perfect reconstruction even when level > pywt.dwt_max_level
    data = np.arange(64).reshape(8, 8)
    tol = 1e-12
    dec_funcs = [pywt.wavedec, pywt.wavedec2, pywt.wavedecn]
    rec_funcs = [pywt.waverec, pywt.waverec2, pywt.waverecn]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for dec_func, rec_func in zip(dec_funcs, rec_funcs):
            for mode in ['periodization', 'symmetric']:
                    coeffs = dec_func(data, 'haar', mode=mode, level=20)
                    r = rec_func(coeffs, 'haar', mode=mode)
                    assert_allclose(data, r, atol=tol, rtol=tol)


def test_waverecn_accuracies():
    # testing 3D only here
    rstate = np.random.RandomState(1234)
    x0 = rstate.randn(4, 4, 4)
    for dt, tol in dtypes_and_tolerances:
        x = x0.astype(dt)
        if np.iscomplexobj(x):
            x += 1j*rstate.randn(4, 4, 4).astype(x.real.dtype)
        coeffs = pywt.wavedecn(x.astype(dt), 'db1')
        assert_allclose(pywt.waverecn(coeffs, 'db1'), x, atol=tol, rtol=tol)


def test_multilevel_dtypes_nd():
    wavelet = pywt.Wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        # wavedecn, waverecn
        x = np.ones((8, 8), dtype=dt_in)
        errmsg = "wrong dtype returned for {0} input".format(dt_in)
        cA, coeffsD2, coeffsD1 = pywt.wavedecn(x, wavelet, level=2)
        assert_(cA.dtype == dt_out, "wavedecn: " + errmsg)
        for key, c in coeffsD1.items():
            assert_(c.dtype == dt_out, "wavedecn: " + errmsg)
        for key, c in coeffsD2.items():
            assert_(c.dtype == dt_out, "wavedecn: " + errmsg)
        x_roundtrip = pywt.waverecn([cA, coeffsD2, coeffsD1], wavelet)
        assert_(x_roundtrip.dtype == dt_out, "waverecn: " + errmsg)


def test_wavedecn_complex():
    data = np.ones((4, 4, 4)) + 1j
    coeffs = pywt.wavedecn(data, 'db1')
    assert_allclose(pywt.waverecn(coeffs, 'db1'), data, rtol=1e-12)


def test_waverecn_dtypes():
    x = np.ones((4, 4, 4))
    for dt, tol in dtypes_and_tolerances:
        coeffs = pywt.wavedecn(x.astype(dt), 'db1')
        assert_allclose(pywt.waverecn(coeffs, 'db1'), x, atol=tol, rtol=tol)


@pytest.mark.slow
def test_waverecn_all_wavelets_modes():
    # test 2D case using all wavelets and modes
    rstate = np.random.RandomState(1234)
    r = rstate.randn(80, 96)
    for wavelet in wavelist:
        for mode in pywt.Modes.modes:
            coeffs = pywt.wavedecn(r, wavelet, mode=mode)
            assert_allclose(pywt.waverecn(coeffs, wavelet, mode=mode),
                            r, rtol=tol_single, atol=tol_single)


def test_coeffs_to_array():
    # single element list returns the first element
    a_coeffs = [np.arange(8).reshape(2, 4), ]
    arr, arr_slices = pywt.coeffs_to_array(a_coeffs)
    assert_allclose(arr, a_coeffs[0])
    assert_allclose(arr, arr[arr_slices[0]])

    assert_raises(ValueError, pywt.coeffs_to_array, [])
    # invalid second element:  array as in wavedec, but not 1D
    assert_raises(ValueError, pywt.coeffs_to_array, [a_coeffs[0], ] * 2)
    # invalid second element:  tuple as in wavedec2, but not a 3-tuple
    assert_raises(ValueError, pywt.coeffs_to_array, [a_coeffs[0],
                                                     (a_coeffs[0], )])
    # coefficients as None is not supported
    assert_raises(ValueError, pywt.coeffs_to_array, [None, ])
    assert_raises(ValueError, pywt.coeffs_to_array, [a_coeffs,
                                                     (None, None, None)])

    # invalid type for second coefficient list element
    assert_raises(ValueError, pywt.coeffs_to_array, [a_coeffs, None])

    # use an invalid key name in the coef dictionary
    coeffs = [np.array([0]), dict(d=np.array([0]), c=np.array([0]))]
    assert_raises(ValueError, pywt.coeffs_to_array, coeffs)


def test_wavedecn_coeff_reshape_even():
    # verify round trip is correct:
    #   wavedecn - >coeffs_to_array-> array_to_coeffs -> waverecn
    # This is done for wavedec{1, 2, n}
    rng = np.random.RandomState(1234)
    params = {'wavedec': {'d': 1, 'dec': pywt.wavedec, 'rec': pywt.waverec},
              'wavedec2': {'d': 2, 'dec': pywt.wavedec2, 'rec': pywt.waverec2},
              'wavedecn': {'d': 3, 'dec': pywt.wavedecn, 'rec': pywt.waverecn}}
    N = 28
    for f in params:
        x1 = rng.randn(*([N] * params[f]['d']))
        for mode in pywt.Modes.modes:
            for wave in wavelist:
                w = pywt.Wavelet(wave)
                maxlevel = pywt.dwt_max_level(np.min(x1.shape), w.dec_len)
                if maxlevel == 0:
                    continue

                coeffs = params[f]['dec'](x1, w, mode=mode)
                coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
                coeffs2 = pywt.array_to_coeffs(coeff_arr, coeff_slices,
                                               output_format=f)
                x1r = params[f]['rec'](coeffs2, w, mode=mode)

                assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)


def test_wavedecn_coeff_reshape_axes_subset():
    # verify round trip is correct when only a subset of axes are transformed:
    #   wavedecn - >coeffs_to_array-> array_to_coeffs -> waverecn
    # This is done for wavedec{1, 2, n}
    rng = np.random.RandomState(1234)
    mode = 'symmetric'
    w = pywt.Wavelet('db2')
    N = 16
    ndim = 3
    for axes in [(-1, ), (0, ), (1, ), (0, 1), (1, 2), (0, 2), None]:
        x1 = rng.randn(*([N] * ndim))
        coeffs = pywt.wavedecn(x1, w, mode=mode, axes=axes)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, axes=axes)
        if axes is not None:
            # if axes is not None, it must be provided to coeffs_to_array
            assert_raises(ValueError, pywt.coeffs_to_array, coeffs)

        # mismatched axes size
        assert_raises(ValueError, pywt.coeffs_to_array, coeffs,
                      axes=(0, 1, 2, 3))
        assert_raises(ValueError, pywt.coeffs_to_array, coeffs,
                      axes=())

        coeffs2 = pywt.array_to_coeffs(coeff_arr, coeff_slices)
        x1r = pywt.waverecn(coeffs2, w, mode=mode, axes=axes)

        assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)


def test_coeffs_to_array_padding():
    rng = np.random.RandomState(1234)
    x1 = rng.randn(32, 32)
    mode = 'symmetric'
    coeffs = pywt.wavedecn(x1, 'db2', mode=mode)

    # padding=None raises a ValueError when tight packing is not possible
    assert_raises(ValueError, pywt.coeffs_to_array, coeffs, padding=None)

    # set padded values to nan
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, padding=np.nan)
    npad = np.sum(np.isnan(coeff_arr))
    assert_(npad > 0)

    # pad with zeros
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, padding=0)
    assert_(np.sum(np.isnan(coeff_arr)) == 0)
    assert_(np.sum(coeff_arr == 0) == npad)

    # Haar case with N as a power of 2 can be tightly packed
    coeffs_haar = pywt.wavedecn(x1, 'haar', mode=mode)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs_haar, padding=None)
    # shape of coeff_arr will match in this case, but not in general
    assert_equal(coeff_arr.shape, x1.shape)


def test_waverecn_coeff_reshape_odd():
    # verify round trip is correct:
    #   wavedecn - >coeffs_to_array-> array_to_coeffs -> waverecn
    rng = np.random.RandomState(1234)
    x1 = rng.randn(35, 33)
    for mode in pywt.Modes.modes:
        for wave in ['haar', ]:
            w = pywt.Wavelet(wave)
            maxlevel = pywt.dwt_max_level(np.min(x1.shape), w.dec_len)
            if maxlevel == 0:
                continue
            coeffs = pywt.wavedecn(x1, w, mode=mode)
            coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            coeffs2 = pywt.array_to_coeffs(coeff_arr, coeff_slices)
            x1r = pywt.waverecn(coeffs2, w, mode=mode)
            # truncate reconstructed values to original shape
            x1r = x1r[tuple([slice(s) for s in x1.shape])]
            assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)


def test_array_to_coeffs_invalid_inputs():
    coeffs = pywt.wavedecn(np.ones(2), 'haar')
    arr, arr_slices = pywt.coeffs_to_array(coeffs)

    # empty list of array slices
    assert_raises(ValueError, pywt.array_to_coeffs, arr, [])

    # invalid format name
    assert_raises(ValueError, pywt.array_to_coeffs, arr, arr_slices, 'foo')


def test_wavedecn_coeff_ravel():
    # verify round trip is correct:
    #   wavedecn - >ravel_coeffs-> unravel_coeffs -> waverecn
    # This is done for wavedec{1, 2, n}
    rng = np.random.RandomState(1234)
    params = {'wavedec': {'d': 1, 'dec': pywt.wavedec, 'rec': pywt.waverec},
              'wavedec2': {'d': 2, 'dec': pywt.wavedec2, 'rec': pywt.waverec2},
              'wavedecn': {'d': 3, 'dec': pywt.wavedecn, 'rec': pywt.waverecn}}
    N = 12
    for f in params:
        x1 = rng.randn(*([N] * params[f]['d']))
        for mode in pywt.Modes.modes:
            for wave in wavelist:
                w = pywt.Wavelet(wave)
                maxlevel = pywt.dwt_max_level(np.min(x1.shape), w.dec_len)
                if maxlevel == 0:
                    continue

                coeffs = params[f]['dec'](x1, w, mode=mode)
                coeff_arr, slices, shapes = pywt.ravel_coeffs(coeffs)
                coeffs2 = pywt.unravel_coeffs(coeff_arr, slices, shapes,
                                              output_format=f)
                x1r = params[f]['rec'](coeffs2, w, mode=mode)

                assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)


def test_wavedecn_coeff_ravel_zero_level():
    # verify round trip is correct:
    #   wavedecn - >ravel_coeffs-> unravel_coeffs -> waverecn
    # This is done for wavedec{1, 2, n}
    rng = np.random.RandomState(1234)
    params = {'wavedec': {'d': 1, 'dec': pywt.wavedec, 'rec': pywt.waverec},
              'wavedec2': {'d': 2, 'dec': pywt.wavedec2, 'rec': pywt.waverec2},
              'wavedecn': {'d': 3, 'dec': pywt.wavedecn, 'rec': pywt.waverecn}}
    N = 16
    for f in params:
        x1 = rng.randn(*([N] * params[f]['d']))
        for mode in pywt.Modes.modes:
            w = pywt.Wavelet('db2')

            coeffs = params[f]['dec'](x1, w, mode=mode, level=0)
            coeff_arr, slices, shapes = pywt.ravel_coeffs(coeffs)
            coeffs2 = pywt.unravel_coeffs(coeff_arr, slices, shapes,
                                          output_format=f)
            x1r = params[f]['rec'](coeffs2, w, mode=mode)

            assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)


def test_waverecn_coeff_ravel_odd():
    # verify round trip is correct:
    #   wavedecn - >ravel_coeffs-> unravel_coeffs -> waverecn
    rng = np.random.RandomState(1234)
    x1 = rng.randn(35, 33)
    for mode in pywt.Modes.modes:
        for wave in ['haar', ]:
            w = pywt.Wavelet(wave)
            maxlevel = pywt.dwt_max_level(np.min(x1.shape), w.dec_len)
            if maxlevel == 0:
                continue
            coeffs = pywt.wavedecn(x1, w, mode=mode)
            coeff_arr, slices, shapes = pywt.ravel_coeffs(coeffs)
            coeffs2 = pywt.unravel_coeffs(coeff_arr, slices, shapes)
            x1r = pywt.waverecn(coeffs2, w, mode=mode)
            # truncate reconstructed values to original shape
            x1r = x1r[tuple([slice(s) for s in x1.shape])]
            assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)


def test_ravel_wavedec2_with_lists():
    x1 = np.ones((8, 8))
    wav = pywt.Wavelet('haar')
    coeffs = pywt.wavedec2(x1, wav)

    # list [cHn, cVn, cDn] instead of tuple is okay
    coeffs[1:] = [list(c) for c in coeffs[1:]]
    coeff_arr, slices, shapes = pywt.ravel_coeffs(coeffs)
    coeffs2 = pywt.unravel_coeffs(coeff_arr, slices, shapes,
                                  output_format='wavedec2')
    x1r = pywt.waverec2(coeffs2, wav)
    assert_allclose(x1, x1r, rtol=1e-4, atol=1e-4)

    # wrong length list will cause a ValueError
    coeffs[1:] = [list(c[:-1]) for c in coeffs[1:]]  # truncate diag coeffs
    assert_raises(ValueError, pywt.ravel_coeffs, coeffs)


def test_ravel_invalid_input():
    # wavedec ravel does not support any coefficient arrays being set to None
    coeffs = pywt.wavedec(np.ones(8), 'haar')
    coeffs[1] = None
    assert_raises(ValueError, pywt.ravel_coeffs, coeffs)

    # wavedec2 ravel cannot have None or a tuple/list of None
    coeffs = pywt.wavedec2(np.ones((8, 8)), 'haar')
    coeffs[1] = (None, None, None)
    assert_raises(ValueError, pywt.ravel_coeffs, coeffs)
    coeffs[1] = [None, None, None]
    assert_raises(ValueError, pywt.ravel_coeffs, coeffs)
    coeffs[1] = None
    assert_raises(ValueError, pywt.ravel_coeffs, coeffs)

    # wavedecn ravel cannot have any dictionary elements as None
    coeffs = pywt.wavedecn(np.ones((8, 8, 8)), 'haar')
    coeffs[1]['ddd'] = None
    assert_raises(ValueError, pywt.ravel_coeffs, coeffs)


def test_unravel_invalid_inputs():
    coeffs = pywt.wavedecn(np.ones(2), 'haar')
    arr, slices, shapes = pywt.ravel_coeffs(coeffs)

    # empty list for slices or shapes
    assert_raises(ValueError, pywt.unravel_coeffs, arr, slices, [])
    assert_raises(ValueError, pywt.unravel_coeffs, arr, [], shapes)

    # unequal length for slices/shapes
    assert_raises(ValueError, pywt.unravel_coeffs, arr, slices[:-1], shapes)

    # invalid format name
    assert_raises(ValueError, pywt.unravel_coeffs, arr, slices, shapes, 'foo')


def test_wavedecn_shapes_and_size():
    wav = pywt.Wavelet('db2')
    for data_shape in [(33, ), (64, 32), (1, 15, 30)]:
        for axes in [None, 0, -1]:
            for mode in pywt.Modes.modes:
                coeffs = pywt.wavedecn(np.ones(data_shape), wav,
                                       mode=mode, axes=axes)

                # verify that the shapes match the coefficient shapes
                shapes = pywt.wavedecn_shapes(data_shape, wav,
                                              mode=mode, axes=axes)

                assert_equal(coeffs[0].shape, shapes[0])
                expected_size = coeffs[0].size
                for level in range(1, len(coeffs)):
                    for k, v in coeffs[level].items():
                        expected_size += v.size
                        assert_equal(shapes[level][k], v.shape)

                # size can be determined from either the shapes or coeffs
                size = pywt.wavedecn_size(shapes)
                assert_equal(size, expected_size)

                size = pywt.wavedecn_size(coeffs)
                assert_equal(size, expected_size)


def test_dwtn_max_level():
    # predicted and empirical dwtn_max_level match
    for wav in [pywt.Wavelet('db2'), 'sym8']:
        for data_shape in [(33, ), (64, 32), (1, 15, 30)]:
            for axes in [None, 0, -1]:
                for mode in pywt.Modes.modes:
                    coeffs = pywt.wavedecn(np.ones(data_shape), wav,
                                           mode=mode, axes=axes)
                    max_lev = pywt.dwtn_max_level(data_shape, wav, axes)
                    assert_equal(len(coeffs[1:]), max_lev)


def test_waverec_axes_subsets():
    rstate = np.random.RandomState(0)
    data = rstate.standard_normal((8, 8, 8))
    # test all combinations of 1 out of 3 axes transformed
    for axis in [0, 1, 2]:
        coefs = pywt.wavedec(data, 'haar', axis=axis)
        rec = pywt.waverec(coefs, 'haar', axis=axis)
        assert_allclose(rec, data, atol=1e-14)


def test_waverec_axis_db2():
    # test for fix to issue gh-293
    rstate = np.random.RandomState(0)
    data = rstate.standard_normal((16, 16))
    for axis in [0, 1]:
        coefs = pywt.wavedec(data, 'db2', axis=axis)
        rec = pywt.waverec(coefs, 'db2', axis=axis)
        assert_allclose(rec, data, atol=1e-14)


def test_waverec2_axes_subsets():
    rstate = np.random.RandomState(0)
    data = rstate.standard_normal((8, 8, 8))
    # test all combinations of 2 out of 3 axes transformed
    for axes in combinations((0, 1, 2), 2):
        coefs = pywt.wavedec2(data, 'haar', axes=axes)
        rec = pywt.waverec2(coefs, 'haar', axes=axes)
        assert_allclose(rec, data, atol=1e-14)


def test_waverecn_axes_subsets():
    rstate = np.random.RandomState(0)
    data = rstate.standard_normal((8, 8, 8, 8))
    # test all combinations of 3 out of 4 axes transformed
    for axes in combinations((0, 1, 2, 3), 3):
        coefs = pywt.wavedecn(data, 'haar', axes=axes)
        rec = pywt.waverecn(coefs, 'haar', axes=axes)
        assert_allclose(rec, data, atol=1e-14)


def test_waverecn_int_axis():
    # waverecn should also work for axes as an integer
    rstate = np.random.RandomState(0)
    data = rstate.standard_normal((8, 8))
    for axis in [0, 1]:
        coefs = pywt.wavedecn(data, 'haar', axes=axis)
        rec = pywt.waverecn(coefs, 'haar', axes=axis)
        assert_allclose(rec, data, atol=1e-14)


def test_wavedec_axis_error():
    data = np.ones(4)
    # out of range axis not allowed
    assert_raises(ValueError, pywt.wavedec, data, 'haar', axis=1)


def test_waverec_axis_error():
    c = pywt.wavedec(np.ones(4), 'haar')
    # out of range axis not allowed
    assert_raises(ValueError, pywt.waverec, c, 'haar', axis=1)


def test_waverec_shape_mismatch_error():
    c = pywt.wavedec(np.ones(16), 'haar')
    # truncate a detail coefficient to an incorrect shape
    c[3] = c[3][:-1]
    assert_raises(ValueError, pywt.waverec, c, 'haar', axis=1)


def test_wavedec2_axes_errors():
    data = np.ones((4, 4))
    # integer axes not allowed
    assert_raises(TypeError, pywt.wavedec2, data, 'haar', axes=1)
    # non-unique axes not allowed
    assert_raises(ValueError, pywt.wavedec2, data, 'haar', axes=(0, 0))
    # out of range axis not allowed
    assert_raises(ValueError, pywt.wavedec2, data, 'haar', axes=(0, 2))


def test_waverec2_axes_errors():
    data = np.ones((4, 4))
    c = pywt.wavedec2(data, 'haar')
    # integer axes not allowed
    assert_raises(TypeError, pywt.waverec2, c, 'haar', axes=1)
    # non-unique axes not allowed
    assert_raises(ValueError, pywt.waverec2, c, 'haar', axes=(0, 0))
    # out of range axis not allowed
    assert_raises(ValueError, pywt.waverec2, c, 'haar', axes=(0, 2))


def test_wavedecn_axes_errors():
    data = np.ones((8, 8, 8))
    # repeated axes not allowed
    assert_raises(ValueError, pywt.wavedecn, data, 'haar', axes=(1, 1))
    # out of range axis not allowed
    assert_raises(ValueError, pywt.wavedecn, data, 'haar', axes=(0, 1, 3))


def test_waverecn_axes_errors():
    data = np.ones((8, 8, 8))
    c = pywt.wavedecn(data, 'haar')
    # repeated axes not allowed
    assert_raises(ValueError, pywt.waverecn, c, 'haar', axes=(1, 1))
    # out of range axis not allowed
    assert_raises(ValueError, pywt.waverecn, c, 'haar', axes=(0, 1, 3))


def test_per_axis_wavelets_and_modes():
    # tests seperate wavelet and edge mode for each axis.
    rstate = np.random.RandomState(1234)
    data = rstate.randn(24, 24, 16)

    # wavelet can be a string or wavelet object
    wavelets = (pywt.Wavelet('haar'), 'sym2', 'db2')

    # The default number of levels should be the minimum over this list
    max_levels = [pywt._dwt.dwt_max_level(nd, nf) for nd, nf in
                  zip(data.shape, wavelets)]

    # mode can be a string or a Modes enum
    modes = ('symmetric', 'periodization',
             pywt._extensions._pywt.Modes.reflect)

    coefs = pywt.wavedecn(data, wavelets, modes)
    assert_allclose(pywt.waverecn(coefs, wavelets, modes), data, atol=1e-14)
    assert_equal(min(max_levels), len(coefs[1:]))

    coefs = pywt.wavedecn(data, wavelets[:1], modes)
    assert_allclose(pywt.waverecn(coefs, wavelets[:1], modes), data,
                    atol=1e-14)

    coefs = pywt.wavedecn(data, wavelets, modes[:1])
    assert_allclose(pywt.waverecn(coefs, wavelets, modes[:1]), data,
                    atol=1e-14)

    # length of wavelets or modes doesn't match the length of axes
    assert_raises(ValueError, pywt.wavedecn, data, wavelets[:2])
    assert_raises(ValueError, pywt.wavedecn, data, wavelets, mode=modes[:2])
    assert_raises(ValueError, pywt.waverecn, coefs, wavelets[:2])
    assert_raises(ValueError, pywt.waverecn, coefs, wavelets, mode=modes[:2])

    # dwt2/idwt2 also support per-axis wavelets/modes
    data2 = data[..., 0]
    coefs2 = pywt.wavedec2(data2, wavelets[:2], modes[:2])
    assert_allclose(pywt.waverec2(coefs2, wavelets[:2], modes[:2]), data2,
                    atol=1e-14)
    assert_equal(min(max_levels[:2]), len(coefs2[1:]))

# Tests for fully separable multi-level transforms


def test_fswavedecn_fswaverecn_roundtrip():
    # verify proper round trip result for 1D through 4D data
    # same DWT as wavedecn/waverecn so don't need to test all modes/wavelets
    rstate = np.random.RandomState(0)
    for ndim in range(1, 5):
        for dt_in, dt_out in zip(dtypes_in, dtypes_out):
            for levels in (1, None):
                data = rstate.standard_normal((8, )*ndim)
                data = data.astype(dt_in)
                T = pywt.fswavedecn(data, 'haar', levels=levels)
                rec = pywt.fswaverecn(T)
                if data.real.dtype in [np.float32, np.float16]:
                    assert_allclose(rec, data, rtol=1e-6, atol=1e-6)
                else:
                    assert_allclose(rec, data, rtol=1e-14, atol=1e-14)
                assert_(T.coeffs.dtype == dt_out)
                assert_(rec.dtype == dt_out)


def test_fswavedecn_fswaverecn_zero_levels():
    # zero level transform gives coefs matching the original data
    rstate = np.random.RandomState(0)
    ndim = 2
    data = rstate.standard_normal((8, )*ndim)
    T = pywt.fswavedecn(data, 'haar', levels=0)
    assert_array_equal(T.coeffs, data)
    rec = pywt.fswaverecn(T)
    assert_array_equal(T.coeffs, rec)


def test_fswavedecn_fswaverecn_variable_levels():
    # test with differing number of transform levels per axis
    rstate = np.random.RandomState(0)
    ndim = 3
    data = rstate.standard_normal((16, )*ndim)
    T = pywt.fswavedecn(data, 'haar', levels=(1, 2, 3))
    rec = pywt.fswaverecn(T)
    assert_allclose(rec, data, atol=1e-14)

    # levels doesn't match number of axes
    assert_raises(ValueError, pywt.fswavedecn, data, 'haar', levels=(1, 1))
    assert_raises(ValueError, pywt.fswavedecn, data, 'haar', levels=(1, 1, 1, 1))

    # levels too large for array size
    assert_warns(UserWarning, pywt.fswavedecn, data, 'haar',
                 levels=int(np.log2(np.min(data.shape)))+1)


def test_fswavedecn_fswaverecn_variable_wavelets_and_modes():
    # test with differing number of transform levels per axis
    rstate = np.random.RandomState(0)
    ndim = 3
    data = rstate.standard_normal((16, )*ndim)
    wavelets = ('haar', 'db2', 'sym3')
    modes = ('periodic', 'symmetric', 'periodization')
    T = pywt.fswavedecn(data, wavelet=wavelets, mode=modes)
    for ax in range(ndim):
        # expect approx + dwt_max_level detail coeffs along each axis
        assert_equal(len(T.coeff_slices[ax]),
                     pywt.dwt_max_level(data.shape[ax], wavelets[ax])+1)

    rec = pywt.fswaverecn(T)
    assert_allclose(rec, data, atol=1e-14)

    # number of wavelets doesn't match number of axes
    assert_raises(ValueError, pywt.fswavedecn, data, wavelets[:2])

    # number of modes doesn't match number of axes
    assert_raises(ValueError, pywt.fswavedecn, data, wavelets[0], mode=modes[:2])


def test_fswavedecn_fswaverecn_axes_subsets():
    """Fully separable DWT over only a subset of axes"""
    rstate = np.random.RandomState(0)
    # use anisotropic data to result in unique number of levels per axis
    data = rstate.standard_normal((4, 8, 16, 32))
    # test all combinations of 3 out of 4 axes transformed
    for axes in combinations((0, 1, 2, 3), 3):
        T = pywt.fswavedecn(data, 'haar', axes=axes)
        rec = pywt.fswaverecn(T)
        assert_allclose(rec, data, atol=1e-14)

    # some axes exceed data dimensions
    assert_raises(ValueError, pywt.fswavedecn, data, 'haar', axes=(1, 5))


def test_fswavedecnresult():
    data = np.ones((32, 32))
    levels = (1, 2)
    result = pywt.fswavedecn(data, 'sym2', levels=levels)

    # can access the lowpass band via .approx or via __getitem__
    approx_key = (0, ) * data.ndim
    assert_array_equal(result[approx_key], result.approx)

    dkeys = result.detail_keys()
    # the approximation key shouldn't be present in the detail_keys
    assert_(approx_key not in dkeys)

    # can access all detail coefficients and they have matching ndim
    for k in dkeys:
        d = result[k]
        assert_equal(d.ndim, data.ndim)

    # can assign modified coefficients
    result[k] = np.zeros_like(d)

    # assigning a differently sized array raises a ValueError
    assert_raises(ValueError, result.__setitem__,
                  k, np.zeros(tuple([s + 1 for s in d.shape])))

    # warns on assigning with a non-matching dtype
    assert_warns(UserWarning, result.__setitem__,
                 k, np.zeros_like(d).astype(np.float32))

    # all coefficients are stacked into result.coeffs (same ndim)
    assert_equal(result.coeffs.ndim, data.ndim)


def test_error_on_continuous_wavelet():
    # A ValueError is raised if a Continuous wavelet is selected
    data = np.ones((16, 16))
    for dec_fun, rec_fun in zip([pywt.wavedec, pywt.wavedec2, pywt.wavedecn],
                                [pywt.waverec, pywt.waverec2, pywt.waverecn]):
        for cwave in ['morl', pywt.DiscreteContinuousWavelet('morl')]:
            assert_raises(ValueError, dec_fun, data, wavelet=cwave)

            c = dec_fun(data, 'db1')
            assert_raises(ValueError, rec_fun, c, wavelet=cwave)


def test_default_level():
    # default level is the maximum permissible for the transformed axes
    data = np.ones((128, 32, 4))
    wavelet = ('db8', 'db1')
    for dec_func in [pywt.wavedec2, pywt.wavedecn]:
        for axes in [(0, 1), (2, 1), (0, 2)]:
            c = dec_func(data, wavelet, axes=axes)
            max_lev = np.min([pywt.dwt_max_level(data.shape[ax], wav)
                              for ax, wav in zip(axes, wavelet)])
            assert_equal(len(c[1:]), max_lev)

    for ax in [0, 1]:
        c = pywt.wavedecn(data, wavelet[ax], axes=(ax, ))
        assert_equal(len(c[1:]),
                     pywt.dwt_max_level(data.shape[ax], wavelet[ax]))


def test_waverec_mixed_precision():
    rstate = np.random.RandomState(0)
    for func, ifunc, shape in [(pywt.wavedec, pywt.waverec, (8, )),
                               (pywt.wavedec2, pywt.waverec2, (8, 8)),
                               (pywt.wavedecn, pywt.waverecn, (8, 8, 8))]:
        x = rstate.randn(*shape)
        coeffs_real = func(x, 'db1')

        # real: single precision approx, double precision details
        coeffs_real[0] = coeffs_real[0].astype(np.float32)
        r = ifunc(coeffs_real, 'db1')
        assert_allclose(r, x, rtol=1e-7, atol=1e-7)
        assert_equal(r.dtype, np.float64)

        x = x + 1j*x
        coeffs = func(x, 'db1')

        # complex: single precision approx, double precision details
        coeffs[0] = coeffs[0].astype(np.complex64)
        r = ifunc(coeffs, 'db1')
        assert_allclose(r, x, rtol=1e-7, atol=1e-7)
        assert_equal(r.dtype, np.complex128)

        # complex: double precision approx, single precision details
        if x.ndim == 1:
            coeffs[0] = coeffs[0].astype(np.complex128)
            coeffs[1] = coeffs[1].astype(np.complex64)
        if x.ndim == 2:
            coeffs[0] = coeffs[0].astype(np.complex128)
            coeffs[1] = tuple([v.astype(np.complex64) for v in coeffs[1]])
        if x.ndim == 3:
            coeffs[0] = coeffs[0].astype(np.complex128)
            coeffs[1] = {k: v.astype(np.complex64)
                         for k, v in coeffs[1].items()}
        r = ifunc(coeffs, 'db1')
        assert_allclose(r, x, rtol=1e-7, atol=1e-7)
        assert_equal(r.dtype, np.complex128)
