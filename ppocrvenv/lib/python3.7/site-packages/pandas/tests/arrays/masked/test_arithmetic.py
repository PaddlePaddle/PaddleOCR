from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray

# integer dtypes
arrays = [pd.array([1, 2, 3, None], dtype=dtype) for dtype in tm.ALL_EA_INT_DTYPES]
scalars: list[Any] = [2] * len(arrays)
# floating dtypes
arrays += [pd.array([0.1, 0.2, 0.3, None], dtype=dtype) for dtype in tm.FLOAT_EA_DTYPES]
scalars += [0.2, 0.2]
# boolean
arrays += [pd.array([True, False, True, None], dtype="boolean")]
scalars += [False]


@pytest.fixture(params=zip(arrays, scalars), ids=[a.dtype.name for a in arrays])
def data(request):
    return request.param


def check_skip(data, op_name):
    if isinstance(data.dtype, pd.BooleanDtype) and "sub" in op_name:
        pytest.skip("subtract not implemented for boolean")


# Test equivalence of scalars, numpy arrays with array ops
# -----------------------------------------------------------------------------


def test_array_scalar_like_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    scalar_array = pd.array([scalar] * len(data), dtype=data.dtype)

    # TODO also add len-1 array (np.array([scalar], dtype=data.dtype.numpy_dtype))
    for scalar in [scalar, data.dtype.type(scalar)]:
        result = op(data, scalar)
        expected = op(data, scalar_array)
        tm.assert_extension_array_equal(result, expected)


def test_array_NA(data, all_arithmetic_operators):
    if "truediv" in all_arithmetic_operators:
        pytest.skip("division with pd.NA raises")
    data, _ = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    scalar = pd.NA
    scalar_array = pd.array([pd.NA] * len(data), dtype=data.dtype)

    result = op(data, scalar)
    expected = op(data, scalar_array)
    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_equivalence(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    numpy_array = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
    pd_array = pd.array(numpy_array, dtype=data.dtype)

    result = op(data, numpy_array)
    expected = op(data, pd_array)
    if isinstance(expected, ExtensionArray):
        tm.assert_extension_array_equal(result, expected)
    else:
        # TODO div still gives float ndarray -> remove this once we have Float EA
        tm.assert_numpy_array_equal(result, expected)


# Test equivalence with Series and DataFrame ops
# -----------------------------------------------------------------------------


def test_frame(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    # DataFrame with scalar
    df = pd.DataFrame({"A": data})

    result = op(df, scalar)
    expected = pd.DataFrame({"A": op(data, scalar)})
    tm.assert_frame_equal(result, expected)


def test_series(data, all_arithmetic_operators):
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)
    check_skip(data, all_arithmetic_operators)

    s = pd.Series(data)

    # Series with scalar
    result = op(s, scalar)
    expected = pd.Series(op(data, scalar))
    tm.assert_series_equal(result, expected)

    # Series with np.ndarray
    other = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
    result = op(s, other)
    expected = pd.Series(op(data, other))
    tm.assert_series_equal(result, expected)

    # Series with pd.array
    other = pd.array([scalar] * len(data), dtype=data.dtype)
    result = op(s, other)
    expected = pd.Series(op(data, other))
    tm.assert_series_equal(result, expected)

    # Series with Series
    other = pd.Series([scalar] * len(data), dtype=data.dtype)
    result = op(s, other)
    expected = pd.Series(op(data, other.array))
    tm.assert_series_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_object(data, all_arithmetic_operators):
    data, _ = data

    op = all_arithmetic_operators
    opa = getattr(data, op)

    # 2d -> return NotImplemented
    result = opa(pd.DataFrame({"A": data}))
    assert result is NotImplemented

    msg = r"can only perform ops with 1-d structures"
    with pytest.raises(NotImplementedError, match=msg):
        opa(np.arange(len(data)).reshape(-1, len(data)))


def test_error_len_mismatch(data, all_arithmetic_operators):
    # operating with a list-like with non-matching length raises
    data, scalar = data
    op = tm.get_op_from_name(all_arithmetic_operators)

    other = [scalar] * (len(data) - 1)

    for other in [other, np.array(other)]:
        with pytest.raises(ValueError, match="Lengths must match"):
            op(data, other)

        s = pd.Series(data)
        with pytest.raises(ValueError, match="Lengths must match"):
            op(s, other)


@pytest.mark.parametrize("op", ["__neg__", "__abs__", "__invert__"])
def test_unary_op_does_not_propagate_mask(data, op, request):
    # https://github.com/pandas-dev/pandas/issues/39943
    data, _ = data
    if data.dtype in ["Float32", "Float64"] and op == "__invert__":
        request.node.add_marker(
            pytest.mark.xfail(
                raises=TypeError, reason="invert is not implemented for float ea dtypes"
            )
        )
    s = pd.Series(data)
    result = getattr(s, op)()
    expected = result.copy(deep=True)
    s[0] = None
    tm.assert_series_equal(result, expected)
