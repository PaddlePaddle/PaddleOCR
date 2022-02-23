import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm

AGG_FUNCTIONS = [
    "sum",
    "prod",
    "min",
    "max",
    "median",
    "mean",
    "skew",
    "mad",
    "std",
    "var",
    "sem",
]


class TestMultiLevel:
    def test_reindex_level(self, multiindex_year_month_day_dataframe_random_data):
        # axis=0
        ymd = multiindex_year_month_day_dataframe_random_data

        with tm.assert_produces_warning(FutureWarning):
            month_sums = ymd.sum(level="month")
        result = month_sums.reindex(ymd.index, level=1)
        expected = ymd.groupby(level="month").transform(np.sum)

        tm.assert_frame_equal(result, expected)

        # Series
        result = month_sums["A"].reindex(ymd.index, level=1)
        expected = ymd["A"].groupby(level="month").transform(np.sum)
        tm.assert_series_equal(result, expected, check_names=False)

        # axis=1
        with tm.assert_produces_warning(FutureWarning):
            month_sums = ymd.T.sum(axis=1, level="month")
        result = month_sums.reindex(columns=ymd.index, level=1)
        expected = ymd.groupby(level="month").transform(np.sum).T
        tm.assert_frame_equal(result, expected)

    def test_binops_level(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data

        def _check_op(opname):
            op = getattr(DataFrame, opname)
            with tm.assert_produces_warning(FutureWarning):
                month_sums = ymd.sum(level="month")
            result = op(ymd, month_sums, level="month")

            broadcasted = ymd.groupby(level="month").transform(np.sum)
            expected = op(ymd, broadcasted)
            tm.assert_frame_equal(result, expected)

            # Series
            op = getattr(Series, opname)
            result = op(ymd["A"], month_sums["A"], level="month")
            broadcasted = ymd["A"].groupby(level="month").transform(np.sum)
            expected = op(ymd["A"], broadcasted)
            expected.name = "A"
            tm.assert_series_equal(result, expected)

        _check_op("sub")
        _check_op("add")
        _check_op("mul")
        _check_op("div")

    def test_reindex(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        expected = frame.iloc[[0, 3]]
        reindexed = frame.loc[[("foo", "one"), ("bar", "one")]]
        tm.assert_frame_equal(reindexed, expected)

    def test_reindex_preserve_levels(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ymd = multiindex_year_month_day_dataframe_random_data

        new_index = ymd.index[::10]
        chunk = ymd.reindex(new_index)
        assert chunk.index is new_index

        chunk = ymd.loc[new_index]
        assert chunk.index is new_index

        ymdT = ymd.T
        chunk = ymdT.reindex(columns=new_index)
        assert chunk.columns is new_index

        chunk = ymdT.loc[:, new_index]
        assert chunk.columns is new_index

    def test_groupby_transform(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        s = frame["A"]
        grouper = s.index.get_level_values(0)

        grouped = s.groupby(grouper)

        applied = grouped.apply(lambda x: x * 2)
        expected = grouped.transform(lambda x: x * 2)
        result = applied.reindex(expected.index)
        tm.assert_series_equal(result, expected, check_names=False)

    def test_groupby_corner(self):
        midx = MultiIndex(
            levels=[["foo"], ["bar"], ["baz"]],
            codes=[[0], [0], [0]],
            names=["one", "two", "three"],
        )
        df = DataFrame([np.random.rand(4)], columns=["a", "b", "c", "d"], index=midx)
        # should work
        df.groupby(level="three")

    def test_groupby_level_no_obs(self):
        # #1697
        midx = MultiIndex.from_tuples(
            [
                ("f1", "s1"),
                ("f1", "s2"),
                ("f2", "s1"),
                ("f2", "s2"),
                ("f3", "s1"),
                ("f3", "s2"),
            ]
        )
        df = DataFrame([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], columns=midx)
        df1 = df.loc(axis=1)[df.columns.map(lambda u: u[0] in ["f2", "f3"])]

        grouped = df1.groupby(axis=1, level=0)
        result = grouped.sum()
        assert (result.columns == ["f2", "f3"]).all()

    def test_setitem_with_expansion_multiindex_columns(
        self, multiindex_year_month_day_dataframe_random_data
    ):
        ymd = multiindex_year_month_day_dataframe_random_data

        df = ymd[:5].T
        df[2000, 1, 10] = df[2000, 1, 7]
        assert isinstance(df.columns, MultiIndex)
        assert (df[2000, 1, 10] == df[2000, 1, 7]).all()

    def test_alignment(self):
        x = Series(
            data=[1, 2, 3], index=MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3)])
        )

        y = Series(
            data=[4, 5, 6], index=MultiIndex.from_tuples([("Z", 1), ("Z", 2), ("B", 3)])
        )

        res = x - y
        exp_index = x.index.union(y.index)
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        tm.assert_series_equal(res, exp)

        # hit non-monotonic code path
        res = x[::-1] - y[::-1]
        exp_index = x.index.union(y.index)
        exp = x.reindex(exp_index) - y.reindex(exp_index)
        tm.assert_series_equal(res, exp)

    @pytest.mark.parametrize("op", AGG_FUNCTIONS)
    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("sort", [True, False])
    def test_series_group_min_max(
        self, op, level, skipna, sort, series_with_multilevel_index
    ):
        # GH 17537
        ser = series_with_multilevel_index

        grouped = ser.groupby(level=level, sort=sort)
        # skipna=True
        leftside = grouped.agg(lambda x: getattr(x, op)(skipna=skipna))
        with tm.assert_produces_warning(FutureWarning):
            rightside = getattr(ser, op)(level=level, skipna=skipna)
        if sort:
            rightside = rightside.sort_index(level=level)
        tm.assert_series_equal(leftside, rightside)

    @pytest.mark.parametrize("op", AGG_FUNCTIONS)
    @pytest.mark.parametrize("level", [0, 1])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize("sort", [True, False])
    def test_frame_group_ops(
        self, op, level, axis, skipna, sort, multiindex_dataframe_random_data
    ):
        # GH 17537
        frame = multiindex_dataframe_random_data

        frame.iloc[1, [1, 2]] = np.nan
        frame.iloc[7, [0, 1]] = np.nan

        level_name = frame.index.names[level]

        if axis == 0:
            frame = frame
        else:
            frame = frame.T

        grouped = frame.groupby(level=level, axis=axis, sort=sort)

        pieces = []

        def aggf(x):
            pieces.append(x)
            return getattr(x, op)(skipna=skipna, axis=axis)

        leftside = grouped.agg(aggf)
        with tm.assert_produces_warning(FutureWarning):
            rightside = getattr(frame, op)(level=level, axis=axis, skipna=skipna)
        if sort:
            rightside = rightside.sort_index(level=level, axis=axis)
            frame = frame.sort_index(level=level, axis=axis)

        # for good measure, groupby detail
        level_index = frame._get_axis(axis).levels[level].rename(level_name)

        tm.assert_index_equal(leftside._get_axis(axis), level_index)
        tm.assert_index_equal(rightside._get_axis(axis), level_index)

        tm.assert_frame_equal(leftside, rightside)

    def test_std_var_pass_ddof(self):
        index = MultiIndex.from_arrays(
            [np.arange(5).repeat(10), np.tile(np.arange(10), 5)]
        )
        df = DataFrame(np.random.randn(len(index), 5), index=index)

        for meth in ["var", "std"]:
            ddof = 4
            alt = lambda x: getattr(x, meth)(ddof=ddof)

            with tm.assert_produces_warning(FutureWarning):
                result = getattr(df[0], meth)(level=0, ddof=ddof)
            expected = df[0].groupby(level=0).agg(alt)
            tm.assert_series_equal(result, expected)

            with tm.assert_produces_warning(FutureWarning):
                result = getattr(df, meth)(level=0, ddof=ddof)
            expected = df.groupby(level=0).agg(alt)
            tm.assert_frame_equal(result, expected)

    def test_agg_multiple_levels(
        self, multiindex_year_month_day_dataframe_random_data, frame_or_series
    ):
        ymd = multiindex_year_month_day_dataframe_random_data
        if frame_or_series is Series:
            ymd = ymd["A"]

        with tm.assert_produces_warning(FutureWarning):
            result = ymd.sum(level=["year", "month"])
        expected = ymd.groupby(level=["year", "month"]).sum()
        tm.assert_equal(result, expected)

    def test_groupby_multilevel(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.groupby(level=[0, 1]).mean()

        k1 = ymd.index.get_level_values(0)
        k2 = ymd.index.get_level_values(1)

        expected = ymd.groupby([k1, k2]).mean()

        # TODO groupby with level_values drops names
        tm.assert_frame_equal(result, expected, check_names=False)
        assert result.index.names == ymd.index.names[:2]

        result2 = ymd.groupby(level=ymd.index.names[:2]).mean()
        tm.assert_frame_equal(result, result2)

    def test_groupby_multilevel_with_transform(self):
        pass

    def test_multilevel_consolidate(self):
        index = MultiIndex.from_tuples(
            [("foo", "one"), ("foo", "two"), ("bar", "one"), ("bar", "two")]
        )
        df = DataFrame(np.random.randn(4, 4), index=index, columns=index)
        df["Totals", ""] = df.sum(1)
        df = df._consolidate()

    def test_level_with_tuples(self):
        index = MultiIndex(
            levels=[[("foo", "bar", 0), ("foo", "baz", 0), ("foo", "qux", 0)], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        series = Series(np.random.randn(6), index=index)
        frame = DataFrame(np.random.randn(6, 4), index=index)

        result = series[("foo", "bar", 0)]
        result2 = series.loc[("foo", "bar", 0)]
        expected = series[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        with pytest.raises(KeyError, match=r"^\(\('foo', 'bar', 0\), 2\)$"):
            series[("foo", "bar", 0), 2]

        result = frame.loc[("foo", "bar", 0)]
        result2 = frame.xs(("foo", "bar", 0))
        expected = frame[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        index = MultiIndex(
            levels=[[("foo", "bar"), ("foo", "baz"), ("foo", "qux")], [0, 1]],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]],
        )

        series = Series(np.random.randn(6), index=index)
        frame = DataFrame(np.random.randn(6, 4), index=index)

        result = series[("foo", "bar")]
        result2 = series.loc[("foo", "bar")]
        expected = series[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        result = frame.loc[("foo", "bar")]
        result2 = frame.xs(("foo", "bar"))
        expected = frame[:2]
        expected.index = expected.index.droplevel(0)
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_reindex_level_partial_selection(self, multiindex_dataframe_random_data):
        frame = multiindex_dataframe_random_data

        result = frame.reindex(["foo", "qux"], level=0)
        expected = frame.iloc[[0, 1, 2, 7, 8, 9]]
        tm.assert_frame_equal(result, expected)

        result = frame.T.reindex(["foo", "qux"], axis=1, level=0)
        tm.assert_frame_equal(result, expected.T)

        result = frame.loc[["foo", "qux"]]
        tm.assert_frame_equal(result, expected)

        result = frame["A"].loc[["foo", "qux"]]
        tm.assert_series_equal(result, expected["A"])

        result = frame.T.loc[:, ["foo", "qux"]]
        tm.assert_frame_equal(result, expected.T)

    @pytest.mark.parametrize("d", [4, "d"])
    def test_empty_frame_groupby_dtypes_consistency(self, d):
        # GH 20888
        group_keys = ["a", "b", "c"]
        df = DataFrame({"a": [1], "b": [2], "c": [3], "d": [d]})

        g = df[df.a == 2].groupby(group_keys)
        result = g.first().index
        expected = MultiIndex(
            levels=[[1], [2], [3]], codes=[[], [], []], names=["a", "b", "c"]
        )

        tm.assert_index_equal(result, expected)

    def test_duplicate_groupby_issues(self):
        idx_tp = [
            ("600809", "20061231"),
            ("600809", "20070331"),
            ("600809", "20070630"),
            ("600809", "20070331"),
        ]
        dt = ["demo", "demo", "demo", "demo"]

        idx = MultiIndex.from_tuples(idx_tp, names=["STK_ID", "RPT_Date"])
        s = Series(dt, index=idx)

        result = s.groupby(s.index).first()
        assert len(result) == 3

    def test_subsets_multiindex_dtype(self):
        # GH 20757
        data = [["x", 1]]
        columns = [("a", "b", np.nan), ("a", "c", 0.0)]
        df = DataFrame(data, columns=MultiIndex.from_tuples(columns))
        expected = df.dtypes.a.b
        result = df.a.b.dtypes
        tm.assert_series_equal(result, expected)


class TestSorted:
    """everything you wanted to test about sorting"""

    def test_sort_non_lexsorted(self):
        # degenerate case where we sort but don't
        # have a satisfying result :<
        # GH 15797
        idx = MultiIndex(
            [["A", "B", "C"], ["c", "b", "a"]], [[0, 1, 2, 0, 1, 2], [0, 2, 1, 1, 0, 2]]
        )

        df = DataFrame({"col": range(len(idx))}, index=idx, dtype="int64")
        assert df.index.is_monotonic is False

        sorted = df.sort_index()
        assert sorted.index.is_monotonic is True

        expected = DataFrame(
            {"col": [1, 4, 5, 2]},
            index=MultiIndex.from_tuples(
                [("B", "a"), ("B", "c"), ("C", "a"), ("C", "b")]
            ),
            dtype="int64",
        )
        result = sorted.loc[pd.IndexSlice["B":"C", "a":"c"], :]
        tm.assert_frame_equal(result, expected)
