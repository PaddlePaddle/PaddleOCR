from datetime import datetime
import operator

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    Series,
    bdate_range,
)
import pandas._testing as tm
from pandas.core import ops


class TestSeriesLogicalOps:
    @pytest.mark.parametrize("bool_op", [operator.and_, operator.or_, operator.xor])
    def test_bool_operators_with_nas(self, bool_op):
        # boolean &, |, ^ should work with object arrays and propagate NAs
        ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
        ser[::2] = np.nan

        mask = ser.isna()
        filled = ser.fillna(ser[0])

        result = bool_op(ser < ser[9], ser > ser[3])

        expected = bool_op(filled < filled[9], filled > filled[3])
        expected[mask] = False
        tm.assert_series_equal(result, expected)

    def test_logical_operators_bool_dtype_with_empty(self):
        # GH#9016: support bitwise op for integer types
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)
        s_empty = Series([], dtype=object)

        res = s_tft & s_empty
        expected = s_fff
        tm.assert_series_equal(res, expected)

        res = s_tft | s_empty
        expected = s_tft
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_int_dtype(self):
        # GH#9016: support bitwise op for integer types

        # TODO: unused
        # s_0101 = Series([0, 1, 0, 1])

        s_0123 = Series(range(4), dtype="int64")
        s_3333 = Series([3] * 4)
        s_4444 = Series([4] * 4)

        res = s_0123 & s_3333
        expected = Series(range(4), dtype="int64")
        tm.assert_series_equal(res, expected)

        res = s_0123 | s_4444
        expected = Series(range(4, 8), dtype="int64")
        tm.assert_series_equal(res, expected)

        s_1111 = Series([1] * 4, dtype="int8")
        res = s_0123 & s_1111
        expected = Series([0, 1, 0, 1], dtype="int64")
        tm.assert_series_equal(res, expected)

        res = s_0123.astype(np.int16) | s_1111.astype(np.int32)
        expected = Series([1, 1, 3, 3], dtype="int32")
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_int_scalar(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        res = s_0123 & 0
        expected = Series([0] * 4)
        tm.assert_series_equal(res, expected)

        res = s_0123 & 1
        expected = Series([0, 1, 0, 1])
        tm.assert_series_equal(res, expected)

    def test_logical_operators_int_dtype_with_float(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.NaN
        with pytest.raises(TypeError, match=msg):
            s_0123 & 3.14
        msg = "unsupported operand type.+for &:"
        with pytest.raises(TypeError, match=msg):
            s_0123 & [0.1, 4, 3.14, 2]
        with pytest.raises(TypeError, match=msg):
            s_0123 & np.array([0.1, 4, 3.14, 2])
        with pytest.raises(TypeError, match=msg):
            s_0123 & Series([0.1, 4, -3.14, 2])

    def test_logical_operators_int_dtype_with_str(self):
        s_1111 = Series([1] * 4, dtype="int8")
        msg = "Cannot perform 'and_' with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s_1111 & "a"
        with pytest.raises(TypeError, match="unsupported operand.+for &"):
            s_1111 & ["a", "b", "c", "d"]

    def test_logical_operators_int_dtype_with_bool(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        expected = Series([False] * 4)

        result = s_0123 & False
        tm.assert_series_equal(result, expected)

        result = s_0123 & [False]
        tm.assert_series_equal(result, expected)

        result = s_0123 & (False,)
        tm.assert_series_equal(result, expected)

        result = s_0123 ^ False
        expected = Series([False, True, True, True])
        tm.assert_series_equal(result, expected)

    def test_logical_operators_int_dtype_with_object(self):
        # GH#9016: support bitwise op for integer types
        s_0123 = Series(range(4), dtype="int64")

        result = s_0123 & Series([False, np.NaN, False, False])
        expected = Series([False] * 4)
        tm.assert_series_equal(result, expected)

        s_abNd = Series(["a", "b", np.NaN, "d"])
        with pytest.raises(TypeError, match="unsupported.* 'int' and 'str'"):
            s_0123 & s_abNd

    def test_logical_operators_bool_dtype_with_int(self):
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_fff = Series([False, False, False], index=index)

        res = s_tft & 0
        expected = s_fff
        tm.assert_series_equal(res, expected)

        res = s_tft & 1
        expected = s_tft
        tm.assert_series_equal(res, expected)

    def test_logical_ops_bool_dtype_with_ndarray(self):
        # make sure we operate on ndarray the same as Series
        left = Series([True, True, True, False, True])
        right = [True, False, None, True, np.nan]

        expected = Series([True, False, False, False, False])
        result = left & right
        tm.assert_series_equal(result, expected)
        result = left & np.array(right)
        tm.assert_series_equal(result, expected)
        result = left & Index(right)
        tm.assert_series_equal(result, expected)
        result = left & Series(right)
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, True, True, True])
        result = left | right
        tm.assert_series_equal(result, expected)
        result = left | np.array(right)
        tm.assert_series_equal(result, expected)
        result = left | Index(right)
        tm.assert_series_equal(result, expected)
        result = left | Series(right)
        tm.assert_series_equal(result, expected)

        expected = Series([False, True, True, True, True])
        result = left ^ right
        tm.assert_series_equal(result, expected)
        result = left ^ np.array(right)
        tm.assert_series_equal(result, expected)
        result = left ^ Index(right)
        tm.assert_series_equal(result, expected)
        result = left ^ Series(right)
        tm.assert_series_equal(result, expected)

    def test_logical_operators_int_dtype_with_bool_dtype_and_reindex(self):
        # GH#9016: support bitwise op for integer types

        # with non-matching indexes, logical operators will cast to object
        #  before operating
        index = list("bca")

        s_tft = Series([True, False, True], index=index)
        s_tft = Series([True, False, True], index=index)
        s_tff = Series([True, False, False], index=index)

        s_0123 = Series(range(4), dtype="int64")

        # s_0123 will be all false now because of reindexing like s_tft
        expected = Series([False] * 7, index=[0, 1, 2, 3, "a", "b", "c"])
        result = s_tft & s_0123
        tm.assert_series_equal(result, expected)

        expected = Series([False] * 7, index=[0, 1, 2, 3, "a", "b", "c"])
        result = s_0123 & s_tft
        tm.assert_series_equal(result, expected)

        s_a0b1c0 = Series([1], list("b"))

        res = s_tft & s_a0b1c0
        expected = s_tff.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

        res = s_tft | s_a0b1c0
        expected = s_tft.reindex(list("abc"))
        tm.assert_series_equal(res, expected)

    def test_scalar_na_logical_ops_corners(self):
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, 10])

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        with pytest.raises(TypeError, match=msg):
            s & datetime(2005, 1, 1)

        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan

        expected = Series(True, index=s.index)
        expected[::2] = False
        result = s & list(s)
        tm.assert_series_equal(result, expected)

    def test_scalar_na_logical_ops_corners_aligns(self):
        s = Series([2, 3, 4, 5, 6, 7, 8, 9, datetime(2005, 1, 1)])
        s[::2] = np.nan
        d = DataFrame({"A": s})

        expected = DataFrame(False, index=range(9), columns=["A"] + list(range(9)))

        result = s & d
        tm.assert_frame_equal(result, expected)

        result = d & s
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
    def test_logical_ops_with_index(self, op):
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

        result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)

        result = op(ser, idx2)
        tm.assert_series_equal(result, expected)

    def test_reversed_xor_with_index_returns_index(self):
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        msg = "operating as a set operation"

        expected = Index.symmetric_difference(idx1, ser)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = idx1 ^ ser
        tm.assert_index_equal(result, expected)

        expected = Index.symmetric_difference(idx2, ser)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = idx2 ^ ser
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "op",
        [
            pytest.param(
                ops.rand_,
                marks=pytest.mark.xfail(
                    reason="GH#22092 Index __and__ returns Index intersection",
                    raises=AssertionError,
                ),
            ),
            pytest.param(
                ops.ror_,
                marks=pytest.mark.xfail(
                    reason="GH#22092 Index __or__ returns Index union",
                    raises=AssertionError,
                ),
            ),
        ],
    )
    def test_reversed_logical_op_with_index_returns_series(self, op):
        # GH#22092, GH#19792
        ser = Series([True, True, False, False])
        idx1 = Index([True, False, True, False])
        idx2 = Index([1, 0, 1, 0])

        msg = "operating as a set operation"

        expected = Series(op(idx1.values, ser.values))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = op(ser, idx1)
        tm.assert_series_equal(result, expected)

        expected = Series(op(idx2.values, ser.values))
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = op(ser, idx2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (ops.rand_, Index([False, True])),
            (ops.ror_, Index([False, True])),
            (ops.rxor, Index([])),
        ],
    )
    def test_reverse_ops_with_index(self, op, expected):
        # https://github.com/pandas-dev/pandas/pull/23628
        # multi-set Index ops are buggy, so let's avoid duplicates...
        ser = Series([True, False])
        idx = Index([False, True])

        msg = "operating as a set operation"
        with tm.assert_produces_warning(
            FutureWarning, match=msg, check_stacklevel=False
        ):
            # behaving as set ops is deprecated, will become logical ops
            result = op(ser, idx)
        tm.assert_index_equal(result, expected)

    def test_logical_ops_label_based(self):
        # GH#4947
        # logical ops should be label based

        a = Series([True, False, True], list("bca"))
        b = Series([False, True, False], list("abc"))

        expected = Series([False, True, False], list("abc"))
        result = a & b
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, False], list("abc"))
        result = a | b
        tm.assert_series_equal(result, expected)

        expected = Series([True, False, False], list("abc"))
        result = a ^ b
        tm.assert_series_equal(result, expected)

        # rhs is bigger
        a = Series([True, False, True], list("bca"))
        b = Series([False, True, False, True], list("abcd"))

        expected = Series([False, True, False, False], list("abcd"))
        result = a & b
        tm.assert_series_equal(result, expected)

        expected = Series([True, True, False, False], list("abcd"))
        result = a | b
        tm.assert_series_equal(result, expected)

        # filling

        # vs empty
        empty = Series([], dtype=object)

        result = a & empty.copy()
        expected = Series([False, False, False], list("bca"))
        tm.assert_series_equal(result, expected)

        result = a | empty.copy()
        expected = Series([True, False, True], list("bca"))
        tm.assert_series_equal(result, expected)

        # vs non-matching
        result = a & Series([1], ["z"])
        expected = Series([False, False, False, False], list("abcz"))
        tm.assert_series_equal(result, expected)

        result = a | Series([1], ["z"])
        expected = Series([True, True, False, False], list("abcz"))
        tm.assert_series_equal(result, expected)

        # identity
        # we would like s[s|e] == s to hold for any e, whether empty or not
        for e in [
            empty.copy(),
            Series([1], ["z"]),
            Series(np.nan, b.index),
            Series(np.nan, a.index),
        ]:
            result = a[a | e]
            tm.assert_series_equal(result, a[a])

        for e in [Series(["z"])]:
            result = a[a | e]
            tm.assert_series_equal(result, a[a])

        # vs scalars
        index = list("bca")
        t = Series([True, False, True])

        for v in [True, 1, 2]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, True, True], index=index)
            tm.assert_series_equal(result, expected)

        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        for v in [np.nan, "foo"]:
            with pytest.raises(TypeError, match=msg):
                t | v

        for v in [False, 0]:
            result = Series([True, False, True], index=index) | v
            expected = Series([True, False, True], index=index)
            tm.assert_series_equal(result, expected)

        for v in [True, 1]:
            result = Series([True, False, True], index=index) & v
            expected = Series([True, False, True], index=index)
            tm.assert_series_equal(result, expected)

        for v in [False, 0]:
            result = Series([True, False, True], index=index) & v
            expected = Series([False, False, False], index=index)
            tm.assert_series_equal(result, expected)
        msg = "Cannot perform.+with a dtyped.+array and scalar of type"
        for v in [np.nan]:
            with pytest.raises(TypeError, match=msg):
                t & v

    def test_logical_ops_df_compat(self):
        # GH#1134
        s1 = Series([True, False, True], index=list("ABC"), name="x")
        s2 = Series([True, True, False], index=list("ABD"), name="x")

        exp = Series([True, False, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 & s2, exp)
        tm.assert_series_equal(s2 & s1, exp)

        # True | np.nan => True
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s1 | s2, exp_or1)
        # np.nan | True => np.nan, filled with False
        exp_or = Series([True, True, False, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s2 | s1, exp_or)

        # DataFrame doesn't fill nan with False
        tm.assert_frame_equal(s1.to_frame() & s2.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s2.to_frame() & s1.to_frame(), exp.to_frame())

        exp = DataFrame({"x": [True, True, np.nan, np.nan]}, index=list("ABCD"))
        tm.assert_frame_equal(s1.to_frame() | s2.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s2.to_frame() | s1.to_frame(), exp_or.to_frame())

        # different length
        s3 = Series([True, False, True], index=list("ABC"), name="x")
        s4 = Series([True, True, True, True], index=list("ABCD"), name="x")

        exp = Series([True, False, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 & s4, exp)
        tm.assert_series_equal(s4 & s3, exp)

        # np.nan | True => np.nan, filled with False
        exp_or1 = Series([True, True, True, False], index=list("ABCD"), name="x")
        tm.assert_series_equal(s3 | s4, exp_or1)
        # True | np.nan => True
        exp_or = Series([True, True, True, True], index=list("ABCD"), name="x")
        tm.assert_series_equal(s4 | s3, exp_or)

        tm.assert_frame_equal(s3.to_frame() & s4.to_frame(), exp.to_frame())
        tm.assert_frame_equal(s4.to_frame() & s3.to_frame(), exp.to_frame())

        tm.assert_frame_equal(s3.to_frame() | s4.to_frame(), exp_or1.to_frame())
        tm.assert_frame_equal(s4.to_frame() | s3.to_frame(), exp_or.to_frame())
