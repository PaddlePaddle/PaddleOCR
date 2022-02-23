""" Testing

"""

import numpy as np

from numpy.testing import assert_array_equal, assert_

from scipy.io.matlab.mio_utils import squeeze_element, chars_to_strings


def test_squeeze_element():
    a = np.zeros((1,3))
    assert_array_equal(np.squeeze(a), squeeze_element(a))
    # 0-D output from squeeze gives scalar
    sq_int = squeeze_element(np.zeros((1,1), dtype=float))
    assert_(isinstance(sq_int, float))
    # Unless it's a structured array
    sq_sa = squeeze_element(np.zeros((1,1),dtype=[('f1', 'f')]))
    assert_(isinstance(sq_sa, np.ndarray))
    # Squeezing empty arrays maintain their dtypes.
    sq_empty = squeeze_element(np.empty(0, np.uint8))
    assert sq_empty.dtype == np.uint8


def test_chars_strings():
    # chars as strings
    strings = ['learn ', 'python', 'fast  ', 'here  ']
    str_arr = np.array(strings, dtype='U6')  # shape (4,)
    chars = [list(s) for s in strings]
    char_arr = np.array(chars, dtype='U1')  # shape (4,6)
    assert_array_equal(chars_to_strings(char_arr), str_arr)
    ca2d = char_arr.reshape((2,2,6))
    sa2d = str_arr.reshape((2,2))
    assert_array_equal(chars_to_strings(ca2d), sa2d)
    ca3d = char_arr.reshape((1,2,2,6))
    sa3d = str_arr.reshape((1,2,2))
    assert_array_equal(chars_to_strings(ca3d), sa3d)
    # Fortran ordered arrays
    char_arrf = np.array(chars, dtype='U1', order='F')  # shape (4,6)
    assert_array_equal(chars_to_strings(char_arrf), str_arr)
    # empty array
    arr = np.array([['']], dtype='U1')
    out_arr = np.array([''], dtype='U1')
    assert_array_equal(chars_to_strings(arr), out_arr)
