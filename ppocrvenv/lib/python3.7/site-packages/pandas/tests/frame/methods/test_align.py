import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameAlign:
    def test_frame_align_aware(self):
        idx1 = date_range("2001", periods=5, freq="H", tz="US/Eastern")
        idx2 = date_range("2001", periods=5, freq="2H", tz="US/Eastern")
        df1 = DataFrame(np.random.randn(len(idx1), 3), idx1)
        df2 = DataFrame(np.random.randn(len(idx2), 3), idx2)
        new1, new2 = df1.align(df2)
        assert df1.index.tz == new1.index.tz
        assert df2.index.tz == new2.index.tz

        # different timezones convert to UTC

        # frame with frame
        df1_central = df1.tz_convert("US/Central")
        new1, new2 = df1.align(df1_central)
        assert new1.index.tz == pytz.UTC
        assert new2.index.tz == pytz.UTC

        # frame with Series
        new1, new2 = df1.align(df1_central[0], axis=0)
        assert new1.index.tz == pytz.UTC
        assert new2.index.tz == pytz.UTC

        df1[0].align(df1_central, axis=0)
        assert new1.index.tz == pytz.UTC
        assert new2.index.tz == pytz.UTC

    def test_align_float(self, float_frame):
        af, bf = float_frame.align(float_frame)
        assert af._mgr is not float_frame._mgr

        af, bf = float_frame.align(float_frame, copy=False)
        assert af._mgr is float_frame._mgr

        # axis = 0
        other = float_frame.iloc[:-5, :3]
        af, bf = float_frame.align(other, axis=0, fill_value=-1)

        tm.assert_index_equal(bf.columns, other.columns)

        # test fill value
        join_idx = float_frame.index.join(other.index)
        diff_a = float_frame.index.difference(join_idx)
        diff_b = other.index.difference(join_idx)
        diff_a_vals = af.reindex(diff_a).values
        diff_b_vals = bf.reindex(diff_b).values
        assert (diff_a_vals == -1).all()

        af, bf = float_frame.align(other, join="right", axis=0)
        tm.assert_index_equal(bf.columns, other.columns)
        tm.assert_index_equal(bf.index, other.index)
        tm.assert_index_equal(af.index, other.index)

        # axis = 1
        other = float_frame.iloc[:-5, :3].copy()
        af, bf = float_frame.align(other, axis=1)
        tm.assert_index_equal(bf.columns, float_frame.columns)
        tm.assert_index_equal(bf.index, other.index)

        # test fill value
        join_idx = float_frame.index.join(other.index)
        diff_a = float_frame.index.difference(join_idx)
        diff_b = other.index.difference(join_idx)
        diff_a_vals = af.reindex(diff_a).values

        # TODO(wesm): unused?
        diff_b_vals = bf.reindex(diff_b).values  # noqa

        assert (diff_a_vals == -1).all()

        af, bf = float_frame.align(other, join="inner", axis=1)
        tm.assert_index_equal(bf.columns, other.columns)

        af, bf = float_frame.align(other, join="inner", axis=1, method="pad")
        tm.assert_index_equal(bf.columns, other.columns)

        af, bf = float_frame.align(
            other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=None
        )
        tm.assert_index_equal(bf.index, Index([]))

        af, bf = float_frame.align(
            other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=0
        )
        tm.assert_index_equal(bf.index, Index([]))

        # Try to align DataFrame to Series along bad axis
        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            float_frame.align(af.iloc[0, :3], join="inner", axis=2)

        # align dataframe to series with broadcast or not
        idx = float_frame.index
        s = Series(range(len(idx)), index=idx)

        left, right = float_frame.align(s, axis=0)
        tm.assert_index_equal(left.index, float_frame.index)
        tm.assert_index_equal(right.index, float_frame.index)
        assert isinstance(right, Series)

        left, right = float_frame.align(s, broadcast_axis=1)
        tm.assert_index_equal(left.index, float_frame.index)
        expected = {c: s for c in float_frame.columns}
        expected = DataFrame(
            expected, index=float_frame.index, columns=float_frame.columns
        )
        tm.assert_frame_equal(right, expected)

        # see gh-9558
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = df[df["a"] == 2]
        expected = DataFrame([[2, 5]], index=[1], columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

        result = df.where(df["a"] == 2, 0)
        expected = DataFrame({"a": [0, 2, 0], "b": [0, 5, 0]})
        tm.assert_frame_equal(result, expected)

    def test_align_int(self, int_frame):
        # test other non-float types
        other = DataFrame(index=range(5), columns=["A", "B", "C"])

        af, bf = int_frame.align(other, join="inner", axis=1, method="pad")
        tm.assert_index_equal(bf.columns, other.columns)

    def test_align_mixed_type(self, float_string_frame):

        af, bf = float_string_frame.align(
            float_string_frame, join="inner", axis=1, method="pad"
        )
        tm.assert_index_equal(bf.columns, float_string_frame.columns)

    def test_align_mixed_float(self, mixed_float_frame):
        # mixed floats/ints
        other = DataFrame(index=range(5), columns=["A", "B", "C"])

        af, bf = mixed_float_frame.align(
            other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=0
        )
        tm.assert_index_equal(bf.index, Index([]))

    def test_align_mixed_int(self, mixed_int_frame):
        other = DataFrame(index=range(5), columns=["A", "B", "C"])

        af, bf = mixed_int_frame.align(
            other.iloc[:, 0], join="inner", axis=1, method=None, fill_value=0
        )
        tm.assert_index_equal(bf.index, Index([]))

    @pytest.mark.parametrize(
        "l_ordered,r_ordered,expected",
        [
            [True, True, pd.CategoricalIndex],
            [True, False, Index],
            [False, True, Index],
            [False, False, pd.CategoricalIndex],
        ],
    )
    def test_align_categorical(self, l_ordered, r_ordered, expected):
        # GH-28397
        df_1 = DataFrame(
            {
                "A": np.arange(6, dtype="int64"),
                "B": Series(list("aabbca")).astype(
                    pd.CategoricalDtype(list("cab"), ordered=l_ordered)
                ),
            }
        ).set_index("B")
        df_2 = DataFrame(
            {
                "A": np.arange(5, dtype="int64"),
                "B": Series(list("babca")).astype(
                    pd.CategoricalDtype(list("cab"), ordered=r_ordered)
                ),
            }
        ).set_index("B")

        aligned_1, aligned_2 = df_1.align(df_2)
        assert isinstance(aligned_1.index, expected)
        assert isinstance(aligned_2.index, expected)
        tm.assert_index_equal(aligned_1.index, aligned_2.index)

    def test_align_multiindex(self):
        # GH#10665
        # same test cases as test_align_multiindex in test_series.py

        midx = pd.MultiIndex.from_product(
            [range(2), range(3), range(2)], names=("a", "b", "c")
        )
        idx = Index(range(2), name="b")
        df1 = DataFrame(np.arange(12, dtype="int64"), index=midx)
        df2 = DataFrame(np.arange(2, dtype="int64"), index=idx)

        # these must be the same results (but flipped)
        res1l, res1r = df1.align(df2, join="left")
        res2l, res2r = df2.align(df1, join="right")

        expl = df1
        tm.assert_frame_equal(expl, res1l)
        tm.assert_frame_equal(expl, res2r)
        expr = DataFrame([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
        tm.assert_frame_equal(expr, res1r)
        tm.assert_frame_equal(expr, res2l)

        res1l, res1r = df1.align(df2, join="right")
        res2l, res2r = df2.align(df1, join="left")

        exp_idx = pd.MultiIndex.from_product(
            [range(2), range(2), range(2)], names=("a", "b", "c")
        )
        expl = DataFrame([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
        tm.assert_frame_equal(expl, res1l)
        tm.assert_frame_equal(expl, res2r)
        expr = DataFrame([0, 0, 1, 1] * 2, index=exp_idx)
        tm.assert_frame_equal(expr, res1r)
        tm.assert_frame_equal(expr, res2l)

    def test_align_series_combinations(self):
        df = DataFrame({"a": [1, 3, 5], "b": [1, 3, 5]}, index=list("ACE"))
        s = Series([1, 2, 4], index=list("ABD"), name="x")

        # frame + series
        res1, res2 = df.align(s, axis=0)
        exp1 = DataFrame(
            {"a": [1, np.nan, 3, np.nan, 5], "b": [1, np.nan, 3, np.nan, 5]},
            index=list("ABCDE"),
        )
        exp2 = Series([1, 2, np.nan, 4, np.nan], index=list("ABCDE"), name="x")

        tm.assert_frame_equal(res1, exp1)
        tm.assert_series_equal(res2, exp2)

        # series + frame
        res1, res2 = s.align(df)
        tm.assert_series_equal(res1, exp2)
        tm.assert_frame_equal(res2, exp1)

    def _check_align(self, a, b, axis, fill_axis, how, method, limit=None):
        aa, ab = a.align(
            b, axis=axis, join=how, method=method, limit=limit, fill_axis=fill_axis
        )

        join_index, join_columns = None, None

        ea, eb = a, b
        if axis is None or axis == 0:
            join_index = a.index.join(b.index, how=how)
            ea = ea.reindex(index=join_index)
            eb = eb.reindex(index=join_index)

        if axis is None or axis == 1:
            join_columns = a.columns.join(b.columns, how=how)
            ea = ea.reindex(columns=join_columns)
            eb = eb.reindex(columns=join_columns)

        ea = ea.fillna(axis=fill_axis, method=method, limit=limit)
        eb = eb.fillna(axis=fill_axis, method=method, limit=limit)

        tm.assert_frame_equal(aa, ea)
        tm.assert_frame_equal(ab, eb)

    @pytest.mark.parametrize("meth", ["pad", "bfill"])
    @pytest.mark.parametrize("ax", [0, 1, None])
    @pytest.mark.parametrize("fax", [0, 1])
    @pytest.mark.parametrize("how", ["inner", "outer", "left", "right"])
    def test_align_fill_method(self, how, meth, ax, fax, float_frame):
        df = float_frame
        self._check_align_fill(df, how, meth, ax, fax)

    def _check_align_fill(self, frame, kind, meth, ax, fax):
        left = frame.iloc[0:4, :10]
        right = frame.iloc[2:, 6:]
        empty = frame.iloc[:0, :0]

        self._check_align(left, right, axis=ax, fill_axis=fax, how=kind, method=meth)
        self._check_align(
            left, right, axis=ax, fill_axis=fax, how=kind, method=meth, limit=1
        )

        # empty left
        self._check_align(empty, right, axis=ax, fill_axis=fax, how=kind, method=meth)
        self._check_align(
            empty, right, axis=ax, fill_axis=fax, how=kind, method=meth, limit=1
        )

        # empty right
        self._check_align(left, empty, axis=ax, fill_axis=fax, how=kind, method=meth)
        self._check_align(
            left, empty, axis=ax, fill_axis=fax, how=kind, method=meth, limit=1
        )

        # both empty
        self._check_align(empty, empty, axis=ax, fill_axis=fax, how=kind, method=meth)
        self._check_align(
            empty, empty, axis=ax, fill_axis=fax, how=kind, method=meth, limit=1
        )
