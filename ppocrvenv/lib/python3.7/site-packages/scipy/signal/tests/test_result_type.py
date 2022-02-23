# Regressions tests on result types of some signal functions

import numpy as np
from numpy.testing import assert_

import pytest

from scipy.signal import (decimate,
                          lfilter_zi,
                          lfiltic,
                          sos2tf,
                          sosfilt_zi)


def test_decimate():
    ones_f32 = np.ones(32, dtype=np.float32)
    assert_(decimate(ones_f32, 2).dtype == np.float32)

    ones_i64 = np.ones(32, dtype=np.int64)
    assert_(decimate(ones_i64, 2).dtype == np.float64)
    

def test_lfilter_zi():
    b_f32 = np.array([1, 2, 3], dtype=np.float32)
    a_f32 = np.array([4, 5, 6], dtype=np.float32)
    assert_(lfilter_zi(b_f32, a_f32).dtype == np.float32)


def test_lfiltic():
    # this would return f32 when given a mix of f32 / f64 args
    b_f32 = np.array([1, 2, 3], dtype=np.float32)
    a_f32 = np.array([4, 5, 6], dtype=np.float32)
    x_f32 = np.ones(32, dtype=np.float32)
    
    b_f64 = b_f32.astype(np.float64)
    a_f64 = a_f32.astype(np.float64)
    x_f64 = x_f32.astype(np.float64)

    assert_(lfiltic(b_f64, a_f32, x_f32).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f64, x_f32).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f32, x_f64).dtype == np.float64)
    assert_(lfiltic(b_f32, a_f32, x_f32, x_f64).dtype == np.float64)


def test_sos2tf():
    sos_f32 = np.array([[4, 5, 6, 1, 2, 3]], dtype=np.float32)
    b, a = sos2tf(sos_f32)
    assert_(b.dtype == np.float32)
    assert_(a.dtype == np.float32)


def test_sosfilt_zi():
    sos_f32 = np.array([[4, 5, 6, 1, 2, 3]], dtype=np.float32)
    assert_(sosfilt_zi(sos_f32).dtype == np.float32)
