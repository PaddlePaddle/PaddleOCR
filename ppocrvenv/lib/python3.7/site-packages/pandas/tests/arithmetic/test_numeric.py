# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for numeric dtypes
from __future__ import annotations

from collections import abc
from decimal import Decimal
from itertools import combinations
import operator
from typing import Any

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Float64Index,
    Index,
    Int64Index,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    UInt64Index,
    array,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr


@pytest.fixture(params=[Index, Series, tm.to_array])
def box_pandas_1d_array(request):
    """
    Fixture to test behavior for Index, Series and tm.to_array classes
    """
    return request.param


def adjust_negative_zero(zero, expected):
    """
    Helper to adjust the expected result if we are dividing by -0.0
    as opposed to 0.0
    """
    if np.signbit(np.array(zero)).any():
        # All entries in the `zero` fixture should be either
        #  all-negative or no-negative.
        assert np.signbit(np.array(zero)).all()

        expected *= -1

    return expected


# TODO: remove this kludge once mypy stops giving false positives here
# List comprehension has incompatible type List[PandasObject]; expected List[RangeIndex]
#  See GH#29725
ser_or_index: list[Any] = [Series, Index]
lefts: list[Any] = [RangeIndex(10, 40, 10)]
lefts.extend(
    [
        cls([10, 20, 30], dtype=dtype)
        for dtype in ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"]
        for cls in ser_or_index
    ]
)

# ------------------------------------------------------------------
# Comparisons


class TestNumericComparisons:
    def test_operator_series_comparison_zerorank(self):
        # GH#13006
        result = np.float64(0) > Series([1, 2, 3])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)
        result = Series([1, 2, 3]) < np.float64(0)
        expected = Series([1, 2, 3]) < 0.0
        tm.assert_series_equal(result, expected)
        result = np.array([0, 1, 2])[0] > Series([0, 1, 2])
        expected = 0.0 > Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_df_numeric_cmp_dt64_raises(self):
        # GH#8932, GH#22163
        ts = pd.Timestamp.now()
        df = pd.DataFrame({"x": range(5)})

        msg = (
            "'[<>]' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
        )
        with pytest.raises(TypeError, match=msg):
            df > ts
        with pytest.raises(TypeError, match=msg):
            df < ts
        with pytest.raises(TypeError, match=msg):
            ts < df
        with pytest.raises(TypeError, match=msg):
            ts > df

        assert not (df == ts).any().any()
        assert (df != ts).all().all()

    def test_compare_invalid(self):
        # GH#8058
        # ops testing
        a = Series(np.random.randn(5), name=0)
        b = Series(np.random.randn(5))
        b.name = pd.Timestamp("2000-01-01")
        tm.assert_series_equal(a / b, 1 / (b / a))

    def test_numeric_cmp_string_numexpr_path(self, box_with_array):
        # GH#36377, GH#35700
        box = box_with_array
        xbox = box if box is not Index else np.ndarray

        obj = Series(np.random.randn(10 ** 5))
        obj = tm.box_expected(obj, box, transpose=False)

        result = obj == "a"

        expected = Series(np.zeros(10 ** 5, dtype=bool))
        expected = tm.box_expected(expected, xbox, transpose=False)
        tm.assert_equal(result, expected)

        result = obj != "a"
        tm.assert_equal(result, ~expected)

        msg = "Invalid comparison between dtype=float64 and str"
        with pytest.raises(TypeError, match=msg):
            obj < "a"


# ------------------------------------------------------------------
# Numeric dtypes Arithmetic with Datetime/Timedelta Scalar


class TestNumericArraylikeArithmeticWithDatetimeLike:

    # TODO: also check name retentention
    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    def test_mul_td64arr(self, left, box_cls):
        # GH#22390
        right = np.array([1, 2, 3], dtype="m8[s]")
        right = box_cls(right)

        expected = TimedeltaIndex(["10s", "40s", "90s"])
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)

        result = left * right
        tm.assert_equal(result, expected)

        result = right * left
        tm.assert_equal(result, expected)

    # TODO: also check name retentention
    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    def test_div_td64arr(self, left, box_cls):
        # GH#22390
        right = np.array([10, 40, 90], dtype="m8[s]")
        right = box_cls(right)

        expected = TimedeltaIndex(["1s", "2s", "3s"])
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)

        result = right / left
        tm.assert_equal(result, expected)

        result = right // left
        tm.assert_equal(result, expected)

        msg = "Cannot divide"
        with pytest.raises(TypeError, match=msg):
            left / right

        with pytest.raises(TypeError, match=msg):
            left // right

    # TODO: also test Tick objects;
    #  see test_numeric_arr_rdiv_tdscalar for note on these failing
    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[s]"),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[ms]"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_numeric_arr_mul_tdscalar(self, scalar_td, numeric_idx, box_with_array):
        # GH#19333
        box = box_with_array
        index = numeric_idx
        expected = TimedeltaIndex([Timedelta(days=n) for n in range(len(index))])

        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)

        result = index * scalar_td
        tm.assert_equal(result, expected)

        commute = scalar_td * index
        tm.assert_equal(commute, expected)

    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_numeric_arr_mul_tdscalar_numexpr_path(self, scalar_td, box_with_array):
        box = box_with_array

        arr = np.arange(2 * 10 ** 4).astype(np.int64)
        obj = tm.box_expected(arr, box, transpose=False)

        expected = arr.view("timedelta64[D]").astype("timedelta64[ns]")
        expected = tm.box_expected(expected, box, transpose=False)

        result = obj * scalar_td
        tm.assert_equal(result, expected)

        result = scalar_td * obj
        tm.assert_equal(result, expected)

    def test_numeric_arr_rdiv_tdscalar(self, three_days, numeric_idx, box_with_array):
        box = box_with_array

        index = numeric_idx[1:3]

        expected = TimedeltaIndex(["3 Days", "36 Hours"])

        index = tm.box_expected(index, box)
        expected = tm.box_expected(expected, box)

        result = three_days / index
        tm.assert_equal(result, expected)

        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            index / three_days

    @pytest.mark.parametrize(
        "other",
        [
            Timedelta(hours=31),
            Timedelta(hours=31).to_pytimedelta(),
            Timedelta(hours=31).to_timedelta64(),
            Timedelta(hours=31).to_timedelta64().astype("m8[h]"),
            np.timedelta64("NaT"),
            np.timedelta64("NaT", "D"),
            pd.offsets.Minute(3),
            pd.offsets.Second(0),
        ],
    )
    def test_add_sub_timedeltalike_invalid(self, numeric_idx, other, box_with_array):
        box = box_with_array

        left = tm.box_expected(numeric_idx, box)
        msg = (
            "unsupported operand type|"
            "Addition/subtraction of integers and integer-arrays|"
            "Instead of adding/subtracting|"
            "cannot use operands with types dtype|"
            "Concatenation operation is not implemented for NumPy arrays"
        )
        with pytest.raises(TypeError, match=msg):
            left + other
        with pytest.raises(TypeError, match=msg):
            other + left
        with pytest.raises(TypeError, match=msg):
            left - other
        with pytest.raises(TypeError, match=msg):
            other - left

    @pytest.mark.parametrize(
        "other",
        [
            pd.Timestamp.now().to_pydatetime(),
            pd.Timestamp.now(tz="UTC").to_pydatetime(),
            pd.Timestamp.now().to_datetime64(),
            pd.NaT,
        ],
    )
    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    def test_add_sub_datetimelike_invalid(self, numeric_idx, other, box_with_array):
        # GH#28080 numeric+datetime64 should raise; Timestamp raises
        #  NullFrequencyError instead of TypeError so is excluded.
        box = box_with_array
        left = tm.box_expected(numeric_idx, box)

        msg = "|".join(
            [
                "unsupported operand type",
                "Cannot (add|subtract) NaT (to|from) ndarray",
                "Addition/subtraction of integers and integer-arrays",
                "Concatenation operation is not implemented for NumPy arrays",
                # pd.array vs np.datetime64 case
                r"operand type\(s\) all returned NotImplemented from __array_ufunc__",
                "can only perform ops with numeric values",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            left + other
        with pytest.raises(TypeError, match=msg):
            other + left
        with pytest.raises(TypeError, match=msg):
            left - other
        with pytest.raises(TypeError, match=msg):
            other - left


# ------------------------------------------------------------------
# Arithmetic


class TestDivisionByZero:
    def test_div_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # We only adjust for Index, because Series does not yet apply
        #  the adjustment correctly.
        expected2 = adjust_negative_zero(zero, expected)

        result = idx / zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype("i8") / np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_floordiv_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # We only adjust for Index, because Series does not yet apply
        #  the adjustment correctly.
        expected2 = adjust_negative_zero(zero, expected)

        result = idx // zero
        tm.assert_index_equal(result, expected2)
        ser_compat = Series(idx).astype("i8") // np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_mod_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        result = idx % zero
        tm.assert_index_equal(result, expected)
        ser_compat = Series(idx).astype("i8") % np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(result))

    def test_divmod_zero(self, zero, numeric_idx):
        idx = numeric_idx

        exleft = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        exright = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        exleft = adjust_negative_zero(zero, exleft)

        result = divmod(idx, zero)
        tm.assert_index_equal(result[0], exleft)
        tm.assert_index_equal(result[1], exright)

    @pytest.mark.parametrize("op", [operator.truediv, operator.floordiv])
    def test_div_negative_zero(self, zero, numeric_idx, op):
        # Check that -1 / -0.0 returns np.inf, not -np.inf
        if isinstance(numeric_idx, UInt64Index):
            return
        idx = numeric_idx - 3

        expected = Index([-np.inf, -np.inf, -np.inf, np.nan, np.inf], dtype=np.float64)
        expected = adjust_negative_zero(zero, expected)

        result = op(idx, zero)
        tm.assert_index_equal(result, expected)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_div_ser(self, switch_numexpr_min_elements, dtype1, any_real_dtype):
        # no longer do integer div for any ops, but deal with the 0's
        dtype2 = any_real_dtype

        first = Series([3, 4, 5, 8], name="first").astype(dtype1)
        second = Series([0, 0, 0, 3], name="second").astype(dtype2)

        with np.errstate(all="ignore"):
            expected = Series(
                first.values.astype(np.float64) / second.values,
                dtype="float64",
                name=None,
            )
        expected.iloc[0:3] = np.inf
        if first.dtype == "int64" and second.dtype == "float32":
            # when using numexpr, the casting rules are slightly different
            # and int64/float32 combo results in float32 instead of float64
            if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                expected = expected.astype("float32")

        result = first / second
        tm.assert_series_equal(result, expected)
        assert not result.equals(second / first)

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_divmod_zero(self, dtype1, any_real_dtype):
        # GH#26987
        dtype2 = any_real_dtype
        left = Series([1, 1]).astype(dtype1)
        right = Series([0, 2]).astype(dtype2)

        # GH#27321 pandas convention is to set 1 // 0 to np.inf, as opposed
        #  to numpy which sets to np.nan; patch `expected[0]` below
        expected = left // right, left % right
        expected = list(expected)
        expected[0] = expected[0].astype(np.float64)
        expected[0][0] = np.inf
        result = divmod(left, right)

        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod case
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_ser_divmod_inf(self):
        left = Series([np.inf, 1.0])
        right = Series([np.inf, 2.0])

        expected = left // right, left % right
        result = divmod(left, right)

        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod case
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_rdiv_zero_compat(self):
        # GH#8674
        zero_array = np.array([0] * 5)
        data = np.random.randn(5)
        expected = Series([0.0] * 5)

        result = zero_array / Series(data)
        tm.assert_series_equal(result, expected)

        result = Series(zero_array) / data
        tm.assert_series_equal(result, expected)

        result = Series(zero_array) / Series(data)
        tm.assert_series_equal(result, expected)

    def test_div_zero_inf_signs(self):
        # GH#9144, inf signing
        ser = Series([-1, 0, 1], name="first")
        expected = Series([-np.inf, np.nan, np.inf], name="first")

        result = ser / 0
        tm.assert_series_equal(result, expected)

    def test_rdiv_zero(self):
        # GH#9144
        ser = Series([-1, 0, 1], name="first")
        expected = Series([0.0, np.nan, 0.0], name="first")

        result = 0 / ser
        tm.assert_series_equal(result, expected)

    def test_floordiv_div(self):
        # GH#9144
        ser = Series([-1, 0, 1], name="first")

        result = ser // 0
        expected = Series([-np.inf, np.nan, np.inf], name="first")
        tm.assert_series_equal(result, expected)

    def test_df_div_zero_df(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        result = df / df

        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({"first": first, "second": second})
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_array(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        first = Series([1.0, 1.0, 1.0, 1.0])
        second = Series([np.nan, np.nan, np.nan, 1])
        expected = pd.DataFrame({"first": first, "second": second})

        with np.errstate(all="ignore"):
            arr = df.values.astype("float") / df.values
        result = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_int(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        result = df / 0
        expected = pd.DataFrame(np.inf, index=df.index, columns=df.columns)
        expected.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result, expected)

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") / 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_div_zero_series_does_not_commute(self):
        # integer div, but deal with the 0's (GH#9144)
        df = pd.DataFrame(np.random.randn(10, 5))
        ser = df[0]
        res = ser / df
        res2 = df / ser
        assert not res.fillna(0).equals(res2.fillna(0))

    # ------------------------------------------------------------------
    # Mod By Zero

    def test_df_mod_zero_df(self, using_array_manager):
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        # this is technically wrong, as the integer portion is coerced to float
        first = Series([0, 0, 0, 0])
        if not using_array_manager:
            # INFO(ArrayManager) BlockManager doesn't preserve dtype per column
            # while ArrayManager performs op column-wisedoes and thus preserves
            # dtype if possible
            first = first.astype("float64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})
        result = df % df
        tm.assert_frame_equal(result, expected)

        # GH#38939 If we dont pass copy=False, df is consolidated and
        #  result["first"] is float64 instead of int64
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}, copy=False)
        first = Series([0, 0, 0, 0], dtype="int64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})
        result = df % df
        tm.assert_frame_equal(result, expected)

    def test_df_mod_zero_array(self):
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        # this is technically wrong, as the integer portion is coerced to float
        # ###
        first = Series([0, 0, 0, 0], dtype="float64")
        second = Series([np.nan, np.nan, np.nan, 0])
        expected = pd.DataFrame({"first": first, "second": second})

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values % df.values
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns, dtype="float64")
        result2.iloc[0:3, 1] = np.nan
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_int(self):
        # GH#3590, modulo as ints
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        result = df % 0
        expected = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

        # numpy has a slightly different (wrong) treatment
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") % 0
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_series_does_not_commute(self):
        # GH#3590, modulo as ints
        # not commutative with series
        df = pd.DataFrame(np.random.randn(10, 5))
        ser = df[0]
        res = ser % df
        res2 = df % ser
        assert not res.fillna(0).equals(res2.fillna(0))


class TestMultiplicationDivision:
    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__
    # for non-timestamp/timedelta/period dtypes

    def test_divide_decimal(self, box_with_array):
        # resolves issue GH#9787
        box = box_with_array
        ser = Series([Decimal(10)])
        expected = Series([Decimal(5)])

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = ser / Decimal(2)

        tm.assert_equal(result, expected)

        result = ser // Decimal(2)
        tm.assert_equal(result, expected)

    def test_div_equiv_binop(self):
        # Test Series.div as well as Series.__div__
        # float/integer issue
        # GH#7785
        first = Series([1, 0], name="first")
        second = Series([-0.01, -0.02], name="second")
        expected = Series([-0.01, -np.inf])

        result = second.div(first)
        tm.assert_series_equal(result, expected, check_names=False)

        result = second / first
        tm.assert_series_equal(result, expected)

    def test_div_int(self, numeric_idx):
        idx = numeric_idx
        result = idx / 1
        expected = idx.astype("float64")
        tm.assert_index_equal(result, expected)

        result = idx / 2
        expected = Index(idx.values / 2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul, operator.floordiv])
    def test_mul_int_identity(self, op, numeric_idx, box_with_array):
        idx = numeric_idx
        idx = tm.box_expected(idx, box_with_array)

        result = op(idx, 1)
        tm.assert_equal(result, idx)

    def test_mul_int_array(self, numeric_idx):
        idx = numeric_idx
        didx = idx * idx

        result = idx * np.array(5, dtype="int64")
        tm.assert_index_equal(result, idx * 5)

        arr_dtype = "uint64" if isinstance(idx, UInt64Index) else "int64"
        result = idx * np.arange(5, dtype=arr_dtype)
        tm.assert_index_equal(result, didx)

    def test_mul_int_series(self, numeric_idx):
        idx = numeric_idx
        didx = idx * idx

        arr_dtype = "uint64" if isinstance(idx, UInt64Index) else "int64"
        result = idx * Series(np.arange(5, dtype=arr_dtype))
        tm.assert_series_equal(result, Series(didx))

    def test_mul_float_series(self, numeric_idx):
        idx = numeric_idx
        rng5 = np.arange(5, dtype="float64")

        result = idx * Series(rng5 + 0.1)
        expected = Series(rng5 * (rng5 + 0.1))
        tm.assert_series_equal(result, expected)

    def test_mul_index(self, numeric_idx):
        # in general not true for RangeIndex
        idx = numeric_idx
        if not isinstance(idx, RangeIndex):
            result = idx * idx
            tm.assert_index_equal(result, idx ** 2)

    def test_mul_datelike_raises(self, numeric_idx):
        idx = numeric_idx
        msg = "cannot perform __rmul__ with this index type"
        with pytest.raises(TypeError, match=msg):
            idx * pd.date_range("20130101", periods=5)

    def test_mul_size_mismatch_raises(self, numeric_idx):
        idx = numeric_idx
        msg = "operands could not be broadcast together"
        with pytest.raises(ValueError, match=msg):
            idx * idx[0:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    @pytest.mark.parametrize("op", [operator.pow, ops.rpow])
    def test_pow_float(self, op, numeric_idx, box_with_array):
        # test power calculations both ways, GH#14973
        box = box_with_array
        idx = numeric_idx
        expected = Float64Index(op(idx.values, 2.0))

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        result = op(idx, 2.0)
        tm.assert_equal(result, expected)

    def test_modulo(self, numeric_idx, box_with_array):
        # GH#9244
        box = box_with_array
        idx = numeric_idx
        expected = Index(idx.values % 2)

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        result = idx % 2
        tm.assert_equal(result, expected)

    def test_divmod_scalar(self, numeric_idx):
        idx = numeric_idx

        result = divmod(idx, 2)
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, 2)

        expected = Index(div), Index(mod)
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    def test_divmod_ndarray(self, numeric_idx):
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        result = divmod(idx, other)
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        expected = Index(div), Index(mod)
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    def test_divmod_series(self, numeric_idx):
        idx = numeric_idx
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        result = divmod(idx, Series(other))
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        expected = Series(div), Series(mod)
        for r, e in zip(result, expected):
            tm.assert_series_equal(r, e)

    @pytest.mark.parametrize("other", [np.nan, 7, -23, 2.718, -3.14, np.inf])
    def test_ops_np_scalar(self, other):
        vals = np.random.randn(5, 3)
        f = lambda x: pd.DataFrame(
            x, index=list("ABCDE"), columns=["jim", "joe", "jolie"]
        )

        df = f(vals)

        tm.assert_frame_equal(df / np.array(other), f(vals / other))
        tm.assert_frame_equal(np.array(other) * df, f(vals * other))
        tm.assert_frame_equal(df + np.array(other), f(vals + other))
        tm.assert_frame_equal(np.array(other) - df, f(other - vals))

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_operators_frame(self):
        # rpow does not work with DataFrame
        ts = tm.makeTimeSeries()
        ts.name = "ts"

        df = pd.DataFrame({"A": ts})

        tm.assert_series_equal(ts + ts, ts + df["A"], check_names=False)
        tm.assert_series_equal(ts ** ts, ts ** df["A"], check_names=False)
        tm.assert_series_equal(ts < ts, ts < df["A"], check_names=False)
        tm.assert_series_equal(ts / ts, ts / df["A"], check_names=False)

    # TODO: this came from tests.series.test_analytics, needs cleanup and
    #  de-duplication with test_modulo above
    def test_modulo2(self):
        with np.errstate(all="ignore"):

            # GH#3590, modulo as ints
            p = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values, dtype="float64")
            expected.iloc[0:3] = np.nan
            tm.assert_series_equal(result, expected)

            result = p["first"] % 0
            expected = Series(np.nan, index=p.index, name="first")
            tm.assert_series_equal(result, expected)

            p = p.astype("float64")
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values)
            tm.assert_series_equal(result, expected)

            p = p.astype("float64")
            result = p["first"] % p["second"]
            result2 = p["second"] % p["first"]
            assert not result.equals(result2)

    def test_modulo_zero_int(self):
        # GH#9144
        with np.errstate(all="ignore"):
            s = Series([0, 1])

            result = s % 0
            expected = Series([np.nan, np.nan])
            tm.assert_series_equal(result, expected)

            result = 0 % s
            expected = Series([np.nan, 0.0])
            tm.assert_series_equal(result, expected)


class TestAdditionSubtraction:
    # __add__, __sub__, __radd__, __rsub__, __iadd__, __isub__
    # for non-timestamp/timedelta/period dtypes

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2], index=list("ABD"), name="x"),
                Series([3.0, 4.0, np.nan, np.nan], index=list("ABCD"), name="x"),
            ),
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),
                Series([2, 2, 2, 2], index=list("ABCD"), name="x"),
                Series([3, 4, 5, np.nan], index=list("ABCD"), name="x"),
            ),
        ],
    )
    def test_add_series(self, first, second, expected):
        # GH#1134
        tm.assert_series_equal(first + second, expected)
        tm.assert_series_equal(second + first, expected)

    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),
                pd.DataFrame({"x": [2, 2, 2]}, index=list("ABD")),
                pd.DataFrame({"x": [3.0, 4.0, np.nan, np.nan]}, index=list("ABCD")),
            ),
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),
                pd.DataFrame({"x": [2, 2, 2, 2]}, index=list("ABCD")),
                pd.DataFrame({"x": [3, 4, 5, np.nan]}, index=list("ABCD")),
            ),
        ],
    )
    def test_add_frames(self, first, second, expected):
        # GH#1134
        tm.assert_frame_equal(first + second, expected)
        tm.assert_frame_equal(second + first, expected)

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_series_frame_radd_bug(self):
        # GH#353
        vals = Series(tm.rands_array(5, 10))
        result = "foo_" + vals
        expected = vals.map(lambda x: "foo_" + x)
        tm.assert_series_equal(result, expected)

        frame = pd.DataFrame({"vals": vals})
        result = "foo_" + frame
        expected = pd.DataFrame({"vals": vals.map(lambda x: "foo_" + x)})
        tm.assert_frame_equal(result, expected)

        ts = tm.makeTimeSeries()
        ts.name = "ts"

        # really raise this time
        now = pd.Timestamp.now().to_pydatetime()
        msg = "|".join(
            [
                "unsupported operand type",
                # wrong error message, see https://github.com/numpy/numpy/issues/18832
                "Concatenation operation",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            now + ts

        with pytest.raises(TypeError, match=msg):
            ts + now

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_datetime64_with_index(self):
        # arithmetic integer ops with an index
        ser = Series(np.random.randn(5))
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)

        # GH#4629
        # arithmetic datetime64 ops with an index
        ser = Series(
            pd.date_range("20130101", periods=5),
            index=pd.date_range("20130101", periods=5),
        )
        expected = ser - ser.index.to_series()
        result = ser - ser.index
        tm.assert_series_equal(result, expected)

        msg = "cannot subtract period"
        with pytest.raises(TypeError, match=msg):
            # GH#18850
            result = ser - ser.index.to_period()

        df = pd.DataFrame(
            np.random.randn(5, 2), index=pd.date_range("20130101", periods=5)
        )
        df["date"] = pd.Timestamp("20130102")
        df["expected"] = df["date"] - df.index.to_series()
        df["result"] = df["date"] - df.index
        tm.assert_series_equal(df["result"], df["expected"], check_names=False)

    # TODO: taken from tests.frame.test_operators, needs cleanup
    def test_frame_operators(self, float_frame):
        frame = float_frame
        frame2 = pd.DataFrame(float_frame, columns=["D", "C", "B", "A"])

        garbage = np.random.random(4)
        colSeries = Series(garbage, index=np.array(frame.columns))

        idSum = frame + frame
        seriesSum = frame + colSeries

        for col, series in idSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] * 2
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

        for col, series in seriesSum.items():
            for idx, val in series.items():
                origVal = frame[col][idx] + colSeries[col]
                if not np.isnan(val):
                    assert val == origVal
                else:
                    assert np.isnan(origVal)

        added = frame2 + frame2
        expected = frame2 * 2
        tm.assert_frame_equal(added, expected)

        df = pd.DataFrame({"a": ["a", None, "b"]})
        tm.assert_frame_equal(df + df, pd.DataFrame({"a": ["aa", np.nan, "bb"]}))

        # Test for issue #10181
        for dtype in ("float", "int64"):
            frames = [
                pd.DataFrame(dtype=dtype),
                pd.DataFrame(columns=["A"], dtype=dtype),
                pd.DataFrame(index=[0], dtype=dtype),
            ]
            for df in frames:
                assert (df + df).equals(df)
                tm.assert_frame_equal(df + df, df)

    # TODO: taken from tests.series.test_operators; needs cleanup
    def test_series_operators(self):
        def _check_op(series, other, op, pos_only=False):
            left = np.abs(series) if pos_only else series
            right = np.abs(other) if pos_only else other

            cython_or_numpy = op(left, right)
            python = left.combine(right, op)
            if isinstance(other, Series) and not other.index.equals(series.index):
                python.index = python.index._with_freq(None)
            tm.assert_series_equal(cython_or_numpy, python)

        def check(series, other):
            simple_ops = ["add", "sub", "mul", "truediv", "floordiv", "mod"]

            for opname in simple_ops:
                _check_op(series, other, getattr(operator, opname))

            _check_op(series, other, operator.pow, pos_only=True)

            _check_op(series, other, ops.radd)
            _check_op(series, other, ops.rsub)
            _check_op(series, other, ops.rtruediv)
            _check_op(series, other, ops.rfloordiv)
            _check_op(series, other, ops.rmul)
            _check_op(series, other, ops.rpow, pos_only=True)
            _check_op(series, other, ops.rmod)

        tser = tm.makeTimeSeries().rename("ts")
        check(tser, tser * 2)
        check(tser, tser[::2])
        check(tser, 5)

        def check_comparators(series, other):
            _check_op(series, other, operator.gt)
            _check_op(series, other, operator.ge)
            _check_op(series, other, operator.eq)
            _check_op(series, other, operator.lt)
            _check_op(series, other, operator.le)

        check_comparators(tser, 5)
        check_comparators(tser, tser + 1)

    # TODO: taken from tests.series.test_operators; needs cleanup
    def test_divmod(self):
        def check(series, other):
            results = divmod(series, other)
            if isinstance(other, abc.Iterable) and len(series) != len(other):
                # if the lengths don't match, this is the test where we use
                # `tser[::2]`. Pad every other value in `other_np` with nan.
                other_np = []
                for n in other:
                    other_np.append(n)
                    other_np.append(np.nan)
            else:
                other_np = other
            other_np = np.asarray(other_np)
            with np.errstate(all="ignore"):
                expecteds = divmod(series.values, np.asarray(other_np))

            for result, expected in zip(results, expecteds):
                # check the values, name, and index separately
                tm.assert_almost_equal(np.asarray(result), expected)

                assert result.name == series.name
                tm.assert_index_equal(result.index, series.index._with_freq(None))

        tser = tm.makeTimeSeries().rename("ts")
        check(tser, tser * 2)
        check(tser, tser[::2])
        check(tser, 5)

    def test_series_divmod_zero(self):
        # Check that divmod uses pandas convention for division by zero,
        #  which does not match numpy.
        # pandas convention has
        #  1/0 == np.inf
        #  -1/0 == -np.inf
        #  1/-0.0 == -np.inf
        #  -1/-0.0 == np.inf
        tser = tm.makeTimeSeries().rename("ts")
        other = tser * 0

        result = divmod(tser, other)
        exp1 = Series([np.inf] * len(tser), index=tser.index, name="ts")
        exp2 = Series([np.nan] * len(tser), index=tser.index, name="ts")
        tm.assert_series_equal(result[0], exp1)
        tm.assert_series_equal(result[1], exp2)


class TestUFuncCompat:
    @pytest.mark.parametrize(
        "holder",
        [Int64Index, UInt64Index, Float64Index, RangeIndex, Series],
    )
    def test_ufunc_compat(self, holder):
        box = Series if holder is Series else Index

        if holder is RangeIndex:
            idx = RangeIndex(0, 5)
        else:
            idx = holder(np.arange(5, dtype="int64"))
        result = np.sin(idx)
        expected = box(np.sin(np.arange(5, dtype="int64")))
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("holder", [Int64Index, UInt64Index, Float64Index, Series])
    def test_ufunc_coercions(self, holder):
        idx = holder([1, 2, 3, 4, 5], name="x")
        box = Series if holder is Series else Index

        result = np.sqrt(idx)
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Float64Index(np.sqrt(np.array([1, 2, 3, 4, 5])), name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = np.divide(idx, 2.0)
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Float64Index([0.5, 1.0, 1.5, 2.0, 2.5], name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        # _evaluate_numeric_binop
        result = idx + 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Float64Index([3.0, 4.0, 5.0, 6.0, 7.0], name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx - 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Float64Index([-1.0, 0.0, 1.0, 2.0, 3.0], name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx * 1.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Float64Index([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

        result = idx / 2.0
        assert result.dtype == "f8" and isinstance(result, box)
        exp = Float64Index([0.5, 1.0, 1.5, 2.0, 2.5], name="x")
        exp = tm.box_expected(exp, box)
        tm.assert_equal(result, exp)

    @pytest.mark.parametrize("holder", [Int64Index, UInt64Index, Float64Index, Series])
    def test_ufunc_multiple_return_values(self, holder):
        obj = holder([1, 2, 3], name="x")
        box = Series if holder is Series else Index

        result = np.modf(obj)
        assert isinstance(result, tuple)
        exp1 = Float64Index([0.0, 0.0, 0.0], name="x")
        exp2 = Float64Index([1.0, 2.0, 3.0], name="x")
        tm.assert_equal(result[0], tm.box_expected(exp1, box))
        tm.assert_equal(result[1], tm.box_expected(exp2, box))

    def test_ufunc_at(self):
        s = Series([0, 1, 2], index=[1, 2, 3], name="x")
        np.add.at(s, [0, 2], 10)
        expected = Series([10, 1, 12], index=[1, 2, 3], name="x")
        tm.assert_series_equal(s, expected)


class TestObjectDtypeEquivalence:
    # Tests that arithmetic operations match operations executed elementwise

    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_nan(self, dtype, box_with_array):
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([np.nan, np.nan, np.nan], dtype=dtype)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = np.nan + ser
        tm.assert_equal(result, expected)

        result = ser + np.nan
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_int(self, dtype, box_with_array):
        box = box_with_array
        ser = Series([1, 2, 3], dtype=dtype)
        expected = Series([2, 3, 4], dtype=dtype)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = 1 + ser
        tm.assert_equal(result, expected)

        result = ser + 1
        tm.assert_equal(result, expected)

    # TODO: moved from tests.series.test_operators; needs cleanup
    @pytest.mark.parametrize(
        "op",
        [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
    )
    def test_operators_reverse_object(self, op):
        # GH#56
        arr = Series(np.random.randn(10), index=np.arange(10), dtype=object)

        result = op(1.0, arr)
        expected = op(1.0, arr.astype(float))
        tm.assert_series_equal(result.astype(float), expected)


class TestNumericArithmeticUnsorted:
    # Tests in this class have been moved from type-specific test modules
    #  but not yet sorted, parametrized, and de-duplicated

    def check_binop(self, ops, scalars, idxs):
        for op in ops:
            for a, b in combinations(idxs, 2):
                result = op(a, b)
                expected = op(Int64Index(a), Int64Index(b))
                tm.assert_index_equal(result, expected)
            for idx in idxs:
                for scalar in scalars:
                    result = op(idx, scalar)
                    expected = op(Int64Index(idx), scalar)
                    tm.assert_index_equal(result, expected)

    def test_binops(self):
        ops = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ]
        scalars = [-1, 1, 2]
        idxs = [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ]
        self.check_binop(ops, scalars, idxs)

    def test_binops_pow(self):
        # numpy does not allow powers of negative integers so test separately
        # https://github.com/numpy/numpy/pull/8127
        ops = [pow]
        scalars = [1, 2]
        idxs = [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)]
        self.check_binop(ops, scalars, idxs)

    # TODO: mod, divmod?
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
            operator.pow,
        ],
    )
    def test_arithmetic_with_frame_or_series(self, op):
        # check that we return NotImplemented when operating with Series
        # or DataFrame
        index = RangeIndex(5)
        other = Series(np.random.randn(5))

        expected = op(Series(index), other)
        result = op(index, other)
        tm.assert_series_equal(result, expected)

        other = pd.DataFrame(np.random.randn(2, 5))
        expected = op(pd.DataFrame([index, index]), other)
        result = op(index, other)
        tm.assert_frame_equal(result, expected)

    def test_numeric_compat2(self):
        # validate that we are handling the RangeIndex overrides to numeric ops
        # and returning RangeIndex where possible

        idx = RangeIndex(0, 10, 2)

        result = idx * 2
        expected = RangeIndex(0, 20, 4)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx + 2
        expected = RangeIndex(2, 12, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx - 2
        expected = RangeIndex(-2, 8, 2)
        tm.assert_index_equal(result, expected, exact=True)

        result = idx / 2
        expected = RangeIndex(0, 5, 1).astype("float64")
        tm.assert_index_equal(result, expected, exact=True)

        result = idx / 4
        expected = RangeIndex(0, 10, 2) / 4
        tm.assert_index_equal(result, expected, exact=True)

        result = idx // 1
        expected = idx
        tm.assert_index_equal(result, expected, exact=True)

        # __mul__
        result = idx * idx
        expected = Index(idx.values * idx.values)
        tm.assert_index_equal(result, expected, exact=True)

        # __pow__
        idx = RangeIndex(0, 1000, 2)
        result = idx ** 2
        expected = idx._int64index ** 2
        tm.assert_index_equal(Index(result.values), expected, exact=True)

        # __floordiv__
        cases_exact = [
            (RangeIndex(0, 1000, 2), 2, RangeIndex(0, 500, 1)),
            (RangeIndex(-99, -201, -3), -3, RangeIndex(33, 67, 1)),
            (RangeIndex(0, 1000, 1), 2, RangeIndex(0, 1000, 1)._int64index // 2),
            (
                RangeIndex(0, 100, 1),
                2.0,
                RangeIndex(0, 100, 1)._int64index // 2.0,
            ),
            (RangeIndex(0), 50, RangeIndex(0)),
            (RangeIndex(2, 4, 2), 3, RangeIndex(0, 1, 1)),
            (RangeIndex(-5, -10, -6), 4, RangeIndex(-2, -1, 1)),
            (RangeIndex(-100, -200, 3), 2, RangeIndex(0)),
        ]
        for idx, div, expected in cases_exact:
            tm.assert_index_equal(idx // div, expected, exact=True)

    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("delta", [1, 0, -1])
    def test_addsub_arithmetic(self, dtype, delta):
        # GH#8142
        delta = dtype(delta)
        index = Index([10, 11, 12], dtype=dtype)
        result = index + delta
        expected = Index(index.values + delta, dtype=dtype)
        tm.assert_index_equal(result, expected)

        # this subtraction used to fail
        result = index - delta
        expected = Index(index.values - delta, dtype=dtype)
        tm.assert_index_equal(result, expected)

        tm.assert_index_equal(index + index, 2 * index)
        tm.assert_index_equal(index - index, 0 * index)
        assert not (index - index).empty


def test_fill_value_inf_masking():
    # GH #27464 make sure we mask 0/1 with Inf and not NaN
    df = pd.DataFrame({"A": [0, 1, 2], "B": [1.1, None, 1.1]})

    other = pd.DataFrame({"A": [1.1, 1.2, 1.3]}, index=[0, 2, 3])

    result = df.rfloordiv(other, fill_value=1)

    expected = pd.DataFrame(
        {"A": [np.inf, 1.0, 0.0, 1.0], "B": [0.0, np.nan, 0.0, np.nan]}
    )
    tm.assert_frame_equal(result, expected)


def test_dataframe_div_silenced():
    # GH#26793
    pdf1 = pd.DataFrame(
        {
            "A": np.arange(10),
            "B": [np.nan, 1, 2, 3, 4] * 2,
            "C": [np.nan] * 10,
            "D": np.arange(10),
        },
        index=list("abcdefghij"),
        columns=list("ABCD"),
    )
    pdf2 = pd.DataFrame(
        np.random.randn(10, 4), index=list("abcdefghjk"), columns=list("ABCX")
    )
    with tm.assert_produces_warning(None):
        pdf1.div(pdf2, fill_value=0)


@pytest.mark.parametrize(
    "data, expected_data",
    [([0, 1, 2], [0, 2, 4])],
)
def test_integer_array_add_list_like(
    box_pandas_1d_array, box_1d_array, data, expected_data
):
    # GH22606 Verify operators with IntegerArray and list-likes
    arr = array(data, dtype="Int64")
    container = box_pandas_1d_array(arr)
    left = container + box_1d_array(data)
    right = box_1d_array(data) + container

    if Series == box_pandas_1d_array:
        assert_function = tm.assert_series_equal
        expected = Series(expected_data, dtype="Int64")
    elif Series == box_1d_array:
        assert_function = tm.assert_series_equal
        expected = Series(expected_data, dtype="object")
    elif Index in (box_pandas_1d_array, box_1d_array):
        assert_function = tm.assert_index_equal
        expected = Int64Index(expected_data)
    else:
        assert_function = tm.assert_numpy_array_equal
        expected = np.array(expected_data, dtype="object")

    assert_function(left, expected)
    assert_function(right, expected)


def test_sub_multiindex_swapped_levels():
    # GH 9952
    df = pd.DataFrame(
        {"a": np.random.randn(6)},
        index=pd.MultiIndex.from_product(
            [["a", "b"], [0, 1, 2]], names=["levA", "levB"]
        ),
    )
    df2 = df.copy()
    df2.index = df2.index.swaplevel(0, 1)
    result = df - df2
    expected = pd.DataFrame([0.0] * 6, columns=["a"], index=df.index)
    tm.assert_frame_equal(result, expected)
