import sys
import warnings

import numpy as np
import pytest

from skimage._shared import testing
from skimage._shared.utils import (check_nD, deprecate_kwarg,
                                   _validate_interpolation_order,
                                   change_default_value, remove_arg,
                                   _supported_float_type,
                                   channel_as_last_axis)

complex_dtypes = [np.complex64, np.complex128]
if hasattr(np, 'complex256'):
    complex_dtypes += [np.complex256]

have_numpydoc = False
try:
    import numpydoc
    have_numpydoc = True
except ImportError:
    pass


def test_remove_argument():

    @remove_arg('arg1', changed_version='0.12')
    def foo(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return arg0, arg1, arg2

    @remove_arg('arg1', changed_version='0.12',
                help_msg="Some indication on future behavior")
    def bar(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return arg0, arg1, arg2

    # Assert warning messages
    expected_msg = ("arg1 argument is deprecated and will be removed "
                    "in version 0.12. To avoid this warning, "
                    "please do not use the arg1 argument. Please see "
                    "foo documentation for more details.")

    with pytest.warns(FutureWarning) as record:
        assert foo(0, 1) == (0, 1, 1)

    assert str(record[0].message) == expected_msg

    with pytest.warns(FutureWarning) as record:
        assert foo(0, arg1=1) == (0, 1, 1)

    assert str(record[0].message) == expected_msg

    expected_msg = ("arg1 argument is deprecated and will be removed "
                    "in version 0.12. To avoid this warning, "
                    "please do not use the arg1 argument. Please see "
                    "bar documentation for more details."
                    " Some indication on future behavior")

    with pytest.warns(FutureWarning) as record:
        assert bar(0, 1) == (0, 1, 1)

    assert str(record[0].message) == expected_msg

    with pytest.warns(FutureWarning) as record:
        assert bar(0, arg1=1) == (0, 1, 1)

    assert str(record[0].message) == expected_msg
    with warnings.catch_warnings(record=True) as recorded:
        # No kwargs
        assert foo(0) == (0, 0, 1)
        assert foo(0, arg2=0) == (0, 0, 0)

        # Function name and doc is preserved
        assert foo.__name__ == 'foo'
        if sys.flags.optimize < 2:
            # if PYTHONOPTIMIZE is set to 2, docstrings are stripped
            assert foo.__doc__ == 'Expected docstring'
    # Assert no warnings were raised
    assert len(recorded) == 0


def test_change_default_value():

    @change_default_value('arg1', new_value=-1, changed_version='0.12')
    def foo(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return arg0, arg1, arg2

    @change_default_value('arg1', new_value=-1, changed_version='0.12',
                          warning_msg="Custom warning message")
    def bar(arg0, arg1=0, arg2=1):
        """Expected docstring"""
        return arg0, arg1, arg2

    # Assert warning messages
    with pytest.warns(FutureWarning) as record:
        assert foo(0) == (0, 0, 1)
        assert bar(0) == (0, 0, 1)

    expected_msg = ("The new recommended value for arg1 is -1. Until "
                    "version 0.12, the default arg1 value is 0. From "
                    "version 0.12, the arg1 default value will be -1. "
                    "To avoid this warning, please explicitly set arg1 value.")

    assert str(record[0].message) == expected_msg
    assert str(record[1].message) == "Custom warning message"

    # Assert that nothing happens if arg1 is set
    with warnings.catch_warnings(record=True) as recorded:
        # No kwargs
        assert foo(0, 2) == (0, 2, 1)
        assert foo(0, arg1=0) == (0, 0, 1)

        # Function name and doc is preserved
        assert foo.__name__ == 'foo'
        if sys.flags.optimize < 2:
            # if PYTHONOPTIMIZE is set to 2, docstrings are stripped
            assert foo.__doc__ == 'Expected docstring'
    # Assert no warnings were raised
    assert len(recorded) == 0


def test_deprecate_kwarg():

    @deprecate_kwarg({'old_arg1': 'new_arg1'}, '0.19')
    def foo(arg0, new_arg1=1, arg2=None):
        """Expected docstring"""
        return arg0, new_arg1, arg2

    @deprecate_kwarg({'old_arg1': 'new_arg1'},
                     deprecated_version='0.19',
                     warning_msg="Custom warning message")
    def bar(arg0, new_arg1=1, arg2=None):
        """Expected docstring"""
        return arg0, new_arg1, arg2

    # Assert that the DeprecationWarning is raised when the deprecated
    # argument name is used and that the reasult is valid
    with pytest.warns(FutureWarning) as record:
        assert foo(0, old_arg1=1) == (0, 1, None)
        assert bar(0, old_arg1=1) == (0, 1, None)

    msg = ("`old_arg1` is a deprecated argument name "
           "for `foo`. Please use `new_arg1` instead.")
    assert str(record[0].message) == msg
    assert str(record[1].message) == "Custom warning message"

    # Assert that nothing happens when the function is called with the
    # new API
    with warnings.catch_warnings(record=True) as recorded:
        # No kwargs
        assert foo(0) == (0, 1, None)
        assert foo(0, 2) == (0, 2, None)
        assert foo(0, 1, 2) == (0, 1, 2)
        # Kwargs without deprecated argument
        assert foo(0, new_arg1=1, arg2=2) == (0, 1, 2)
        assert foo(0, new_arg1=2) == (0, 2, None)
        assert foo(0, arg2=2) == (0, 1, 2)
        assert foo(0, 1, arg2=2) == (0, 1, 2)
        # Function name and doc is preserved
        assert foo.__name__ == 'foo'
        if sys.flags.optimize < 2:
            # if PYTHONOPTIMIZE is set to 2, docstrings are stripped
            if not have_numpydoc:
                assert foo.__doc__ == """Expected docstring"""
            else:
                assert foo.__doc__ == """Expected docstring


    Other Parameters
    ----------------
    old_arg1 : DEPRECATED
        Deprecated in favor of `new_arg1`.

        .. deprecated:: 0.19
"""

    assert len(recorded) == 0


def test_check_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        check_nD(x, 2)


@pytest.mark.parametrize('dtype', [bool, int, np.uint8, np.uint16,
                                   float, np.float32, np.float64])
@pytest.mark.parametrize('order', [None, -1, 0, 1, 2, 3, 4, 5, 6])
def test_validate_interpolation_order(dtype, order):
    if order is None:
        # Default order
        assert (_validate_interpolation_order(dtype, None) == 0
                if dtype == bool else 1)
    elif order < 0 or order > 5:
        # Order not in valid range
        with testing.raises(ValueError):
            _validate_interpolation_order(dtype, order)
    elif dtype == bool and order != 0:
        # Deprecated order for bool array
        with pytest.raises(ValueError):
            _validate_interpolation_order(bool, order)
    else:
        # Valid use case
        assert _validate_interpolation_order(dtype, order) == order


@pytest.mark.parametrize(
    'dtype',
    [bool, np.float16, np.float32, np.float64, np.uint8, np.uint16, np.uint32,
     np.uint64, np.int8, np.int16, np.int32, np.int64]
)
def test_supported_float_dtype_real(dtype):
    float_dtype = _supported_float_type(dtype)
    if dtype in [np.float16, np.float32]:
        assert float_dtype == np.float32
    else:
        assert float_dtype == np.float64


@pytest.mark.parametrize('dtype', complex_dtypes)
@pytest.mark.parametrize('allow_complex', [False, True])
def test_supported_float_dtype_complex(dtype, allow_complex):
    if allow_complex:
        float_dtype = _supported_float_type(dtype, allow_complex=allow_complex)
        if dtype == np.complex64:
            assert float_dtype == np.complex64
        else:
            assert float_dtype == np.complex128
    else:
        with testing.raises(ValueError):
            _supported_float_type(dtype, allow_complex=allow_complex)


@pytest.mark.parametrize(
    'dtype', ['f', 'float32', np.float32, np.dtype(np.float32)]
)
def test_supported_float_dtype_input_kinds(dtype):
    assert _supported_float_type(dtype) == np.float32


@pytest.mark.parametrize(
    'dtypes, expected',
    [
        ((np.float16, np.float64), np.float64),
        ([np.float32, np.uint16, np.int8], np.float64),
        ({np.float32, np.float16}, np.float32),
    ]
)
def test_supported_float_dtype_sequence(dtypes, expected):
    float_dtype = _supported_float_type(dtypes)
    assert float_dtype == expected


@channel_as_last_axis(multichannel_output=False)
def _decorated_channel_axis_size(x, *, channel_axis=None):
    if channel_axis is None:
        return None
    assert channel_axis == -1
    return x.shape[-1]


@testing.parametrize('channel_axis', [None, 0, 1, 2, -1, -2, -3])
def test_decorated_channel_axis_shape(channel_axis):
    # Verify that channel_as_last_axis modifies the channel_axis as expected

    # need unique size per axis here
    x = np.zeros((2, 3, 4))

    size = _decorated_channel_axis_size(x, channel_axis=channel_axis)
    if channel_axis is None:
        assert size is None
    else:
        assert size == x.shape[channel_axis]
