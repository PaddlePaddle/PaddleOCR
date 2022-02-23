#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_allclose, assert_, assert_raises,
                           assert_array_equal)
import pywt

# Check that float32, float64, complex64, complex128 are preserved.
# Other real types get converted to float64.
# complex256 gets converted to complex128
dtypes_in = [np.int8, np.float16, np.float32, np.float64, np.complex64,
             np.complex128]
dtypes_out = [np.float64, np.float32, np.float32, np.float64, np.complex64,
              np.complex128]

# test complex256 as well if it is available
try:
    dtypes_in += [np.complex256, ]
    dtypes_out += [np.complex128, ]
except AttributeError:
    pass


def test_dwt_idwt_basic():
    x = [3, 7, 1, 1, -2, 5, 4, 6]
    cA, cD = pywt.dwt(x, 'db2')
    cA_expect = [5.65685425, 7.39923721, 0.22414387, 3.33677403, 7.77817459]
    cD_expect = [-2.44948974, -1.60368225, -4.44140056, -0.41361256,
                 1.22474487]
    assert_allclose(cA, cA_expect)
    assert_allclose(cD, cD_expect)

    x_roundtrip = pywt.idwt(cA, cD, 'db2')
    assert_allclose(x_roundtrip, x, rtol=1e-10)

    # mismatched dtypes OK
    x_roundtrip2 = pywt.idwt(cA.astype(np.float64), cD.astype(np.float32),
                             'db2')
    assert_allclose(x_roundtrip2, x, rtol=1e-7, atol=1e-7)
    assert_(x_roundtrip2.dtype == np.float64)


def test_idwt_mixed_complex_dtype():
    x = np.arange(8).astype(float)
    x = x + 1j*x[::-1]
    cA, cD = pywt.dwt(x, 'db2')

    x_roundtrip = pywt.idwt(cA, cD, 'db2')
    assert_allclose(x_roundtrip, x, rtol=1e-10)

    # mismatched dtypes OK
    x_roundtrip2 = pywt.idwt(cA.astype(np.complex128), cD.astype(np.complex64),
                             'db2')
    assert_allclose(x_roundtrip2, x, rtol=1e-7, atol=1e-7)
    assert_(x_roundtrip2.dtype == np.complex128)


def test_dwt_idwt_dtypes():
    wavelet = pywt.Wavelet('haar')
    for dt_in, dt_out in zip(dtypes_in, dtypes_out):
        x = np.ones(4, dtype=dt_in)
        errmsg = "wrong dtype returned for {0} input".format(dt_in)

        cA, cD = pywt.dwt(x, wavelet)
        assert_(cA.dtype == cD.dtype == dt_out, "dwt: " + errmsg)

        x_roundtrip = pywt.idwt(cA, cD, wavelet)
        assert_(x_roundtrip.dtype == dt_out, "idwt: " + errmsg)


def test_dwt_idwt_basic_complex():
    x = np.asarray([3, 7, 1, 1, -2, 5, 4, 6])
    x = x + 0.5j*x
    cA, cD = pywt.dwt(x, 'db2')
    cA_expect = np.asarray([5.65685425, 7.39923721, 0.22414387, 3.33677403,
                            7.77817459])
    cA_expect = cA_expect + 0.5j*cA_expect
    cD_expect = np.asarray([-2.44948974, -1.60368225, -4.44140056, -0.41361256,
                            1.22474487])
    cD_expect = cD_expect + 0.5j*cD_expect
    assert_allclose(cA, cA_expect)
    assert_allclose(cD, cD_expect)

    x_roundtrip = pywt.idwt(cA, cD, 'db2')
    assert_allclose(x_roundtrip, x, rtol=1e-10)


def test_dwt_idwt_partial_complex():
    x = np.asarray([3, 7, 1, 1, -2, 5, 4, 6])
    x = x + 0.5j*x

    cA, cD = pywt.dwt(x, 'haar')
    cA_rec_expect = np.array([5.0+2.5j, 5.0+2.5j, 1.0+0.5j, 1.0+0.5j,
                              1.5+0.75j, 1.5+0.75j, 5.0+2.5j, 5.0+2.5j])
    cA_rec = pywt.idwt(cA, None, 'haar')
    assert_allclose(cA_rec, cA_rec_expect)

    cD_rec_expect = np.array([-2.0-1.0j, 2.0+1.0j, 0.0+0.0j, 0.0+0.0j,
                              -3.5-1.75j, 3.5+1.75j, -1.0-0.5j, 1.0+0.5j])
    cD_rec = pywt.idwt(None, cD, 'haar')
    assert_allclose(cD_rec, cD_rec_expect)

    assert_allclose(cA_rec + cD_rec, x)


def test_dwt_wavelet_kwd():
    x = np.array([3, 7, 1, 1, -2, 5, 4, 6])
    w = pywt.Wavelet('sym3')
    cA, cD = pywt.dwt(x, wavelet=w, mode='constant')
    cA_expect = [4.38354585, 3.80302657, 7.31813271, -0.58565539, 4.09727044,
                 7.81994027]
    cD_expect = [-1.33068221, -2.78795192, -3.16825651, -0.67715519,
                 -0.09722957, -0.07045258]
    assert_allclose(cA, cA_expect)
    assert_allclose(cD, cD_expect)


def test_dwt_coeff_len():
    x = np.array([3, 7, 1, 1, -2, 5, 4, 6])
    w = pywt.Wavelet('sym3')
    ln_modes = [pywt.dwt_coeff_len(len(x), w.dec_len, mode) for mode in
                pywt.Modes.modes]

    expected_result = [6, ] * len(pywt.Modes.modes)
    expected_result[pywt.Modes.modes.index('periodization')] = 4

    assert_allclose(ln_modes, expected_result)
    ln_modes = [pywt.dwt_coeff_len(len(x), w, mode) for mode in
                pywt.Modes.modes]
    assert_allclose(ln_modes, expected_result)


def test_idwt_none_input():
    # None input equals arrays of zeros of the right length
    res1 = pywt.idwt([1, 2, 0, 1], None, 'db2', 'symmetric')
    res2 = pywt.idwt([1, 2, 0, 1], [0, 0, 0, 0], 'db2', 'symmetric')
    assert_allclose(res1, res2, rtol=1e-15, atol=1e-15)

    res1 = pywt.idwt(None, [1, 2, 0, 1], 'db2', 'symmetric')
    res2 = pywt.idwt([0, 0, 0, 0], [1, 2, 0, 1], 'db2', 'symmetric')
    assert_allclose(res1, res2, rtol=1e-15, atol=1e-15)

    # Only one argument at a time can be None
    assert_raises(ValueError, pywt.idwt, None, None, 'db2', 'symmetric')


def test_idwt_invalid_input():
    # Too short, min length is 4 for 'db4':
    assert_raises(ValueError, pywt.idwt, [1, 2, 4], [4, 1, 3], 'db4', 'symmetric')


def test_dwt_single_axis():
    x = [[3, 7, 1, 1],
         [-2, 5, 4, 6]]

    cA, cD = pywt.dwt(x, 'db2', axis=-1)

    cA0, cD0 = pywt.dwt(x[0], 'db2')
    cA1, cD1 = pywt.dwt(x[1], 'db2')

    assert_allclose(cA[0], cA0)
    assert_allclose(cA[1], cA1)

    assert_allclose(cD[0], cD0)
    assert_allclose(cD[1], cD1)


def test_idwt_single_axis():
    x = [[3, 7, 1, 1],
         [-2, 5, 4, 6]]

    x = np.asarray(x)
    x = x + 1j*x   # test with complex data
    cA, cD = pywt.dwt(x, 'db2', axis=-1)

    x0 = pywt.idwt(cA[0], cD[0], 'db2', axis=-1)
    x1 = pywt.idwt(cA[1], cD[1], 'db2', axis=-1)

    assert_allclose(x[0], x0)
    assert_allclose(x[1], x1)

def test_dwt_invalid_input():
    x = np.arange(1)
    assert_raises(ValueError, pywt.dwt, x, 'db2', 'reflect')
    assert_raises(ValueError, pywt.dwt, x, 'haar', 'antireflect')


def test_dwt_axis_arg():
    x = [[3, 7, 1, 1],
         [-2, 5, 4, 6]]

    cA_, cD_ = pywt.dwt(x, 'db2', axis=-1)
    cA, cD = pywt.dwt(x, 'db2', axis=1)

    assert_allclose(cA_, cA)
    assert_allclose(cD_, cD)

def test_dwt_axis_invalid_input():  
    x = np.ones((3,1))
    assert_raises(ValueError, pywt.dwt, x, 'db2', 'reflect')

def test_idwt_axis_arg():
    x = [[3, 7, 1, 1],
         [-2, 5, 4, 6]]

    cA, cD = pywt.dwt(x, 'db2', axis=1)

    x_ = pywt.idwt(cA, cD, 'db2', axis=-1)
    x = pywt.idwt(cA, cD, 'db2', axis=1)

    assert_allclose(x_, x)


def test_dwt_idwt_axis_excess():
    x = [[3, 7, 1, 1],
         [-2, 5, 4, 6]]
    # can't transform over axes that aren't there
    assert_raises(ValueError,
                  pywt.dwt, x, 'db2', 'symmetric', axis=2)

    assert_raises(ValueError,
                  pywt.idwt, [1, 2, 4], [4, 1, 3], 'db2', 'symmetric', axis=1)


def test_error_on_continuous_wavelet():
    # A ValueError is raised if a Continuous wavelet is selected
    data = np.ones((32, ))
    for cwave in ['morl', pywt.DiscreteContinuousWavelet('morl')]:
        assert_raises(ValueError, pywt.dwt, data, cwave)

        cA, cD = pywt.dwt(data, 'db1')
        assert_raises(ValueError, pywt.idwt, cA, cD, cwave)


def test_dwt_zero_size_axes():
    # raise on empty input array
    assert_raises(ValueError, pywt.dwt, [], 'db2')

    # >1D case uses a different code path so check there as well
    x = np.ones((1, 4))[0:0, :]  # 2D with a size zero axis
    assert_raises(ValueError, pywt.dwt, x, 'db2', axis=0)


def test_pad_1d():
    x = [1, 2, 3]
    assert_array_equal(pywt.pad(x, (4, 6), 'periodization'),
                       [1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2])
    assert_array_equal(pywt.pad(x, (4, 6), 'periodic'),
                       [3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert_array_equal(pywt.pad(x, (4, 6), 'constant'),
                       [1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3])
    assert_array_equal(pywt.pad(x, (4, 6), 'zero'),
                       [0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0])
    assert_array_equal(pywt.pad(x, (4, 6), 'smooth'),
                       [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert_array_equal(pywt.pad(x, (4, 6), 'symmetric'),
                       [3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3])
    assert_array_equal(pywt.pad(x, (4, 6), 'antisymmetric'),
                       [3, -3, -2, -1, 1, 2, 3, -3, -2, -1, 1, 2, 3])
    assert_array_equal(pywt.pad(x, (4, 6), 'reflect'),
                       [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1])
    assert_array_equal(pywt.pad(x, (4, 6), 'antireflect'),
                       [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # equivalence of various pad_width formats
    assert_array_equal(pywt.pad(x, 4, 'periodic'),
                       pywt.pad(x, (4, 4), 'periodic'))

    assert_array_equal(pywt.pad(x, (4, ), 'periodic'),
                       pywt.pad(x, (4, 4), 'periodic'))

    assert_array_equal(pywt.pad(x, [(4, 4)], 'periodic'),
                       pywt.pad(x, (4, 4), 'periodic'))


def test_pad_errors():
    # negative pad width
    x = [1, 2, 3]
    assert_raises(ValueError, pywt.pad, x, -2, 'periodic')

    # wrong length pad width
    assert_raises(ValueError, pywt.pad, x, (1, 1, 1), 'periodic')

    # invalid mode name
    assert_raises(ValueError, pywt.pad, x, 2, 'bad_mode')


def test_pad_nd():
    for ndim in [2, 3]:
        x = np.arange(4**ndim).reshape((4, ) * ndim)
        if ndim == 2:
            pad_widths = [(2, 1), (2, 3)]
        else:
            pad_widths = [(2, 1), ] * ndim
        for mode in pywt.Modes.modes:
            xp = pywt.pad(x, pad_widths, mode)

            # expected result is the same as applying along axes separably
            xp_expected = x.copy()
            for ax in range(ndim):
                xp_expected = np.apply_along_axis(pywt.pad,
                                                  ax,
                                                  xp_expected,
                                                  pad_widths=[pad_widths[ax]],
                                                  mode=mode)
            assert_array_equal(xp, xp_expected)
