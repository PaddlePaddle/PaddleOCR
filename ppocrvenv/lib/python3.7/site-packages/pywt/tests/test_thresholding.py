from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_, assert_equal

import pywt


float_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
real_dtypes = [np.float32, np.float64]


def _sign(x):
    # Matlab-like sign function (numpy uses a different convention).
    return x / np.abs(x)


def _soft(x, thresh):
    """soft thresholding supporting complex values.

    Notes
    -----
    This version is not robust to zeros in x.
    """
    return _sign(x) * np.maximum(np.abs(x) - thresh, 0)


def test_threshold():
    data = np.linspace(1, 4, 7)

    # soft
    soft_result = [0., 0., 0., 0.5, 1., 1.5, 2.]
    assert_allclose(pywt.threshold(data, 2, 'soft'),
                    np.array(soft_result), rtol=1e-12)
    assert_allclose(pywt.threshold(-data, 2, 'soft'),
                    -np.array(soft_result), rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 1, 'soft'),
                    [[0, 1]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 2, 'soft'),
                    [[0, 0]] * 2, rtol=1e-12)

    # soft thresholding complex values
    assert_allclose(pywt.threshold([[1j, 2j]] * 2, 1, 'soft'),
                    [[0j, 1j]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1+1j, 2+2j]] * 2, 6, 'soft'),
                    [[0, 0]] * 2, rtol=1e-12)
    complex_data = [[1+2j, 2+2j]]*2
    for thresh in [1, 2]:
        assert_allclose(pywt.threshold(complex_data, thresh, 'soft'),
                        _soft(complex_data, thresh), rtol=1e-12)

    # test soft thresholding with non-default substitute argument
    s = 5
    assert_allclose(pywt.threshold([[1j, 2]] * 2, 1.5, 'soft', substitute=s),
                    [[s, 0.5]] * 2, rtol=1e-12)

    # soft: no divide by zero warnings when input contains zeros
    assert_allclose(pywt.threshold(np.zeros(16), 2, 'soft'),
                    np.zeros(16), rtol=1e-12)

    # hard
    hard_result = [0., 0., 2., 2.5, 3., 3.5, 4.]
    assert_allclose(pywt.threshold(data, 2, 'hard'),
                    np.array(hard_result), rtol=1e-12)
    assert_allclose(pywt.threshold(-data, 2, 'hard'),
                    -np.array(hard_result), rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 1, 'hard'),
                    [[1, 2]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 2, 'hard'),
                    [[0, 2]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 2, 'hard', substitute=s),
                    [[s, 2]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1+1j, 2+2j]] * 2, 2, 'hard'),
                    [[0, 2+2j]] * 2, rtol=1e-12)

    # greater
    greater_result = [0., 0., 2., 2.5, 3., 3.5, 4.]
    assert_allclose(pywt.threshold(data, 2, 'greater'),
                    np.array(greater_result), rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 1, 'greater'),
                    [[1, 2]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 2, 'greater'),
                    [[0, 2]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 2, 'greater', substitute=s),
                    [[s, 2]] * 2, rtol=1e-12)
    # greater doesn't allow complex-valued inputs
    assert_raises(ValueError, pywt.threshold, [1j, 2j], 2, 'greater')

    # less
    assert_allclose(pywt.threshold(data, 2, 'less'),
                    np.array([1., 1.5, 2., 0., 0., 0., 0.]), rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 1, 'less'),
                    [[1, 0]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 1, 'less', substitute=s),
                    [[1, s]] * 2, rtol=1e-12)
    assert_allclose(pywt.threshold([[1, 2]] * 2, 2, 'less'),
                    [[1, 2]] * 2, rtol=1e-12)

    # less doesn't allow complex-valued inputs
    assert_raises(ValueError, pywt.threshold, [1j, 2j], 2, 'less')

    # invalid
    assert_raises(ValueError, pywt.threshold, data, 2, 'foo')


def test_nonnegative_garotte():
    thresh = 0.3
    data_real = np.linspace(-1, 1, 100)
    for dtype in float_dtypes:
        if dtype in real_dtypes:
            data = np.asarray(data_real, dtype=dtype)
        else:
            data = np.asarray(data_real + 0.1j, dtype=dtype)
        d_hard = pywt.threshold(data, thresh, 'hard')
        d_soft = pywt.threshold(data, thresh, 'soft')
        d_garotte = pywt.threshold(data, thresh, 'garotte')

        # check dtypes
        assert_equal(d_hard.dtype, data.dtype)
        assert_equal(d_soft.dtype, data.dtype)
        assert_equal(d_garotte.dtype, data.dtype)

        # values < threshold are zero
        lt = np.where(np.abs(data) < thresh)
        assert_(np.all(d_garotte[lt] == 0))

        # values > than the threshold are intermediate between soft and hard
        gt = np.where(np.abs(data) > thresh)
        gt_abs_garotte = np.abs(d_garotte[gt])
        assert_(np.all(gt_abs_garotte < np.abs(d_hard[gt])))
        assert_(np.all(gt_abs_garotte > np.abs(d_soft[gt])))


def test_threshold_firm():
    thresh = 0.2
    thresh2 = 3 * thresh
    data_real = np.linspace(-1, 1, 100)
    for dtype in float_dtypes:
        if dtype in real_dtypes:
            data = np.asarray(data_real, dtype=dtype)
        else:
            data = np.asarray(data_real + 0.1j, dtype=dtype)
        if data.real.dtype == np.float32:
            rtol = atol = 1e-6
        else:
            rtol = atol = 1e-14
        d_hard = pywt.threshold(data, thresh, 'hard')
        d_soft = pywt.threshold(data, thresh, 'soft')
        d_firm = pywt.threshold_firm(data, thresh, thresh2)

        # check dtypes
        assert_equal(d_hard.dtype, data.dtype)
        assert_equal(d_soft.dtype, data.dtype)
        assert_equal(d_firm.dtype, data.dtype)

        # values < threshold are zero
        lt = np.where(np.abs(data) < thresh)
        assert_(np.all(d_firm[lt] == 0))

        # values > than the threshold are equal to hard-thresholding
        gt = np.where(np.abs(data) >= thresh2)
        assert_allclose(np.abs(d_hard[gt]), np.abs(d_firm[gt]),
                        rtol=rtol, atol=atol)

        # other values are intermediate between soft and hard thresholding
        mt = np.where(np.logical_and(np.abs(data) > thresh,
                                     np.abs(data) < thresh2))
        mt_abs_firm = np.abs(d_firm[mt])
        assert_(np.all(mt_abs_firm < np.abs(d_hard[mt])))
        assert_(np.all(mt_abs_firm > np.abs(d_soft[mt])))
