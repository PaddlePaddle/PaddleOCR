"""Test for correctness of color distance functions"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (deltaE_cie76, deltaE_ciede94,
                                   deltaE_ciede2000, deltaE_cmc)


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ciede2000_dE(dtype, channel_axis):
    data = load_ciede2000_data()
    N = len(data)
    lab1 = np.zeros((N, 3), dtype=dtype)
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']

    lab2 = np.zeros((N, 3), dtype=dtype)
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']

    lab1 = np.moveaxis(lab1, source=-1, destination=channel_axis)
    lab2 = np.moveaxis(lab2, source=-1, destination=channel_axis)
    dE2 = deltaE_ciede2000(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)

    rtol = 1e-2 if dtype == np.float32 else 1e-4
    assert_allclose(dE2, data['dE'], rtol=rtol)


def load_ciede2000_data():
    dtype = [('pair', int),
             ('1', int),
             ('L1', float),
             ('a1', float),
             ('b1', float),
             ('a1_prime', float),
             ('C1_prime', float),
             ('h1_prime', float),
             ('hbar_prime', float),
             ('G', float),
             ('T', float),
             ('SL', float),
             ('SC', float),
             ('SH', float),
             ('RT', float),
             ('dE', float),
             ('2', int),
             ('L2', float),
             ('a2', float),
             ('b2', float),
             ('a2_prime', float),
             ('C2_prime', float),
             ('h2_prime', float),
             ]

    # note: ciede_test_data.txt contains several intermediate quantities
    path = fetch('color/tests/ciede2000_test_data.txt')
    return np.loadtxt(path, dtype=dtype)


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cie76(dtype, channel_axis):
    data = load_ciede2000_data()
    N = len(data)
    lab1 = np.zeros((N, 3), dtype=dtype)
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']

    lab2 = np.zeros((N, 3), dtype=dtype)
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']

    lab1 = np.moveaxis(lab1, source=-1, destination=channel_axis)
    lab2 = np.moveaxis(lab2, source=-1, destination=channel_axis)
    dE2 = deltaE_cie76(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    oracle = np.array([
        4.00106328, 6.31415011, 9.1776999, 2.06270077, 2.36957073,
        2.91529271, 2.23606798, 2.23606798, 4.98000036, 4.9800004,
        4.98000044, 4.98000049, 4.98000036, 4.9800004, 4.98000044,
        3.53553391, 36.86800781, 31.91002977, 30.25309901, 27.40894015,
        0.89242934, 0.7972, 0.8583065, 0.82982507, 3.1819238,
        2.21334297, 1.53890382, 4.60630929, 6.58467989, 3.88641412,
        1.50514845, 2.3237848, 0.94413208, 1.31910843
    ])
    rtol = 1e-5 if dtype == np.float32 else 1e-8
    assert_allclose(dE2, oracle, rtol=rtol)


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ciede94(dtype, channel_axis):
    data = load_ciede2000_data()
    N = len(data)
    lab1 = np.zeros((N, 3), dtype=dtype)
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']

    lab2 = np.zeros((N, 3), dtype=dtype)
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']

    lab1 = np.moveaxis(lab1, source=-1, destination=channel_axis)
    lab2 = np.moveaxis(lab2, source=-1, destination=channel_axis)
    dE2 = deltaE_ciede94(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    oracle = np.array([
        1.39503887, 1.93410055, 2.45433566, 0.68449187, 0.6695627,
        0.69194527, 2.23606798, 2.03163832, 4.80069441, 4.80069445,
        4.80069449, 4.80069453, 4.80069441, 4.80069445, 4.80069449,
        3.40774352, 34.6891632, 29.44137328, 27.91408781, 24.93766082,
        0.82213163, 0.71658427, 0.8048753, 0.75284394, 1.39099471,
        1.24808929, 1.29795787, 1.82045088, 2.55613309, 1.42491303,
        1.41945261, 2.3225685, 0.93853308, 1.30654464
    ])
    rtol = 1e-5 if dtype == np.float32 else 1e-8
    assert_allclose(dE2, oracle, rtol=rtol)


@pytest.mark.parametrize("channel_axis", [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cmc(dtype, channel_axis):
    data = load_ciede2000_data()
    N = len(data)
    lab1 = np.zeros((N, 3), dtype=dtype)
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']

    lab2 = np.zeros((N, 3), dtype=dtype)
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']

    lab1 = np.moveaxis(lab1, source=-1, destination=channel_axis)
    lab2 = np.moveaxis(lab2, source=-1, destination=channel_axis)
    dE2 = deltaE_cmc(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    oracle = np.array([
        1.73873611, 2.49660844, 3.30494501, 0.85735576, 0.88332927,
        0.97822692, 3.50480874, 2.87930032, 6.5783807, 6.57838075,
        6.5783808, 6.57838086, 6.67492321, 6.67492326, 6.67492331,
        4.66852997, 42.10875485, 39.45889064, 38.36005919, 33.93663807,
        1.14400168, 1.00600419, 1.11302547, 1.05335328, 1.42822951,
        1.2548143, 1.76838061, 2.02583367, 3.08695508, 1.74893533,
        1.90095165, 1.70258148, 1.80317207, 2.44934417
    ])
    rtol = 1e-5 if dtype == np.float32 else 1e-8
    assert_allclose(dE2, oracle, rtol=rtol)

    # Equal or close colors make `delta_e.get_dH2` function to return
    # negative values resulting in NaNs when passed to sqrt (see #1908
    # issue on Github):
    lab1 = lab2
    expected = np.zeros_like(oracle)
    assert_almost_equal(
        deltaE_cmc(lab1, lab2, channel_axis=channel_axis), expected, decimal=6
    )

    lab2[0, 0] += np.finfo(float).eps
    assert_almost_equal(
        deltaE_cmc(lab1, lab2, channel_axis=channel_axis), expected, decimal=6
    )


def test_cmc_single_item():
    # Single item case:
    lab1 = lab2 = np.array([0., 1.59607713, 0.87755709])
    assert_equal(deltaE_cmc(lab1, lab2), 0)

    lab2[0] += np.finfo(float).eps
    assert_equal(deltaE_cmc(lab1, lab2), 0)


def test_single_color_cie76():
    lab1 = (0.5, 0.5, 0.5)
    lab2 = (0.4, 0.4, 0.4)
    deltaE_cie76(lab1, lab2)


def test_single_color_ciede94():
    lab1 = (0.5, 0.5, 0.5)
    lab2 = (0.4, 0.4, 0.4)
    deltaE_ciede94(lab1, lab2)


def test_single_color_ciede2000():
    lab1 = (0.5, 0.5, 0.5)
    lab2 = (0.4, 0.4, 0.4)
    deltaE_ciede2000(lab1, lab2)


def test_single_color_cmc():
    lab1 = (0.5, 0.5, 0.5)
    lab2 = (0.4, 0.4, 0.4)
    deltaE_cmc(lab1, lab2)
