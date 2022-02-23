import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import FloatingArray


@pytest.fixture
def data():
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


@pytest.fixture
def left_array():
    return pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")


@pytest.fixture
def right_array():
    return pd.array([True, False, None] * 3, dtype="boolean")


# Basic test for the arithmetic array ops
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opname, exp",
    [
        ("add", [True, True, None, True, False, None, None, None, None]),
        ("mul", [True, False, None, False, False, None, None, None, None]),
    ],
    ids=["add", "mul"],
)
def test_add_mul(left_array, right_array, opname, exp):
    op = getattr(operator, opname)
    result = op(left_array, right_array)
    expected = pd.array(exp, dtype="boolean")
    tm.assert_extension_array_equal(result, expected)


def test_sub(left_array, right_array):
    msg = (
        r"numpy boolean subtract, the `-` operator, is (?:deprecated|not supported), "
        r"use the bitwise_xor, the `\^` operator, or the logical_xor function instead\."
    )
    with pytest.raises(TypeError, match=msg):
        left_array - right_array


def test_div(left_array, right_array):
    result = left_array / right_array
    expected = FloatingArray(
        np.array(
            [1.0, np.inf, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            dtype="float64",
        ),
        np.array([False, False, True, False, False, True, True, True, True]),
    )
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "opname",
    [
        "floordiv",
        "mod",
        pytest.param(
            "pow", marks=pytest.mark.xfail(reason="TODO follow int8 behaviour? GH34686")
        ),
    ],
)
def test_op_int8(left_array, right_array, opname):
    op = getattr(operator, opname)
    result = op(left_array, right_array)
    expected = op(left_array.astype("Int8"), right_array.astype("Int8"))
    tm.assert_extension_array_equal(result, expected)


# Test generic characteristics / errors
# -----------------------------------------------------------------------------


def test_error_invalid_values(data, all_arithmetic_operators):
    # invalid ops

    op = all_arithmetic_operators
    s = pd.Series(data)
    ops = getattr(s, op)

    # invalid scalars
    msg = (
        "did not contain a loop with signature matching types|"
        "BooleanArray cannot perform the operation|"
        "not supported for the input types, and the inputs could not be safely coerced "
        "to any supported types according to the casting rule ''safe''"
    )
    with pytest.raises(TypeError, match=msg):
        ops("foo")
    msg = (
        r"unsupported operand type\(s\) for|"
        "Concatenation operation is not implemented for NumPy arrays"
    )
    with pytest.raises(TypeError, match=msg):
        ops(pd.Timestamp("20180101"))

    # invalid array-likes
    if op not in ("__mul__", "__rmul__"):
        # TODO(extension) numpy's mul with object array sees booleans as numbers
        msg = (
            r"unsupported operand type\(s\) for|can only concatenate str|"
            "not all arguments converted during string formatting"
        )
        with pytest.raises(TypeError, match=msg):
            ops(pd.Series("foo", index=s.index))
