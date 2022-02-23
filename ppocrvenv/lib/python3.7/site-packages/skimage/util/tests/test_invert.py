import numpy as np
from skimage import dtype_limits
from skimage.util.dtype import dtype_range
from skimage.util import invert

from skimage._shared.testing import assert_array_equal


def test_invert_bool():
    dtype = 'bool'
    image = np.zeros((3, 3), dtype=dtype)
    upper_dtype_limit = dtype_limits(image, clip_negative=False)[1]
    image[1, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype) + upper_dtype_limit
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_uint8():
    dtype = 'uint8'
    image = np.zeros((3, 3), dtype=dtype)
    upper_dtype_limit = dtype_limits(image, clip_negative=False)[1]
    image[1, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype) + upper_dtype_limit
    expected[1, :] = 0
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_int8():
    dtype = 'int8'
    image = np.zeros((3, 3), dtype=dtype)
    lower_dtype_limit, upper_dtype_limit = \
        dtype_limits(image, clip_negative=False)
    image[1, :] = lower_dtype_limit
    image[2, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype)
    expected[2, :] = lower_dtype_limit
    expected[1, :] = upper_dtype_limit
    expected[0, :] = -1
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_float64_signed():
    dtype = 'float64'
    image = np.zeros((3, 3), dtype=dtype)
    lower_dtype_limit, upper_dtype_limit = \
        dtype_limits(image, clip_negative=False)
    image[1, :] = lower_dtype_limit
    image[2, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype)
    expected[2, :] = lower_dtype_limit
    expected[1, :] = upper_dtype_limit
    result = invert(image, signed_float=True)
    assert_array_equal(expected, result)


def test_invert_float64_unsigned():
    dtype = 'float64'
    image = np.zeros((3, 3), dtype=dtype)
    lower_dtype_limit, upper_dtype_limit = \
        dtype_limits(image, clip_negative=True)
    image[2, :] = upper_dtype_limit
    expected = np.zeros((3, 3), dtype=dtype)
    expected[0, :] = upper_dtype_limit
    expected[1, :] = upper_dtype_limit
    result = invert(image)
    assert_array_equal(expected, result)


def test_invert_roundtrip():
    for t, limits in dtype_range.items():
        image = np.array(limits, dtype=t)
        expected = invert(invert(image))
        assert_array_equal(image, expected)
