import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby


class TestRolling:
    def setup_method(self):
        self.frame = DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})

    def test_mutated(self):

        msg = r"groupby\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            self.frame.groupby("A", foo=1)

        g = self.frame.groupby("A")
        assert not g.mutated
        g = get_groupby(self.frame, by="A", mutated=True)
        assert g.mutated

    def test_getitem(self):
        g = self.frame.groupby("A")
        g_mutated = get_groupby(self.frame, by="A", mutated=True)

        expected = g_mutated.B.apply(lambda x: x.rolling(2).mean())

        result = g.rolling(2).mean().B
        tm.assert_series_equal(result, expected)

        result = g.rolling(2).B.mean()
        tm.assert_series_equal(result, expected)

        result = g.B.rolling(2).mean()
        tm.assert_series_equal(result, expected)

        result = self.frame.B.groupby(self.frame.A).rolling(2).mean()
        tm.assert_series_equal(result, expected)

    def test_getitem_multiple(self):

        # GH 13174
        g = self.frame.groupby("A")
        r = g.rolling(2, min_periods=0)
        g_mutated = get_groupby(self.frame, by="A", mutated=True)
        expected = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())

        result = r.B.count()
        tm.assert_series_equal(result, expected)

        result = r.B.count()
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            "min",
            "max",
            pytest.param(
                "count",
                marks=pytest.mark.filterwarnings("ignore:min_periods:FutureWarning"),
            ),
            "kurt",
            "skew",
        ],
    )
    def test_rolling(self, f):
        g = self.frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.rolling(4), f)())
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_rolling_ddof(self, f):
        g = self.frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r, f)(ddof=1)
        expected = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_rolling_quantile(self, interpolation):
        g = self.frame.groupby("A")
        r = g.rolling(window=4)

        result = r.quantile(0.4, interpolation=interpolation)
        expected = g.apply(
            lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation)
        )
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f, expected_val", [["corr", 1], ["cov", 0.5]])
    def test_rolling_corr_cov_other_same_size_as_groups(self, f, expected_val):
        # GH 42915
        df = DataFrame(
            {"value": range(10), "idx1": [1] * 5 + [2] * 5, "idx2": [1, 2, 3, 4, 5] * 2}
        ).set_index(["idx1", "idx2"])
        other = DataFrame({"value": range(5), "idx2": [1, 2, 3, 4, 5]}).set_index(
            "idx2"
        )
        result = getattr(df.groupby(level=0).rolling(2), f)(other)
        expected_data = ([np.nan] + [expected_val] * 4) * 2
        expected = DataFrame(
            expected_data,
            columns=["value"],
            index=MultiIndex.from_arrays(
                [
                    [1] * 5 + [2] * 5,
                    [1] * 5 + [2] * 5,
                    list(range(1, 6)) * 2,
                ],
                names=["idx1", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f):
        g = self.frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r, f)(self.frame)

        def func(x):
            return getattr(x.rolling(4), f)(self.frame)

        expected = g.apply(func)
        # GH 39591: The grouped column should be all np.nan
        # (groupby.apply inserts 0s for cov)
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_pairwise(self, f):
        g = self.frame.groupby("A")
        r = g.rolling(window=4)

        result = getattr(r.B, f)(pairwise=True)

        def func(x):
            return getattr(x.B.rolling(4), f)(pairwise=True)

        expected = g.apply(func)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, expected_values",
        [("cov", [[1.0, 1.0], [1.0, 4.0]]), ("corr", [[1.0, 0.5], [0.5, 1.0]])],
    )
    def test_rolling_corr_cov_unordered(self, func, expected_values):
        # GH 43386
        df = DataFrame(
            {
                "a": ["g1", "g2", "g1", "g1"],
                "b": [0, 0, 1, 2],
                "c": [2, 0, 6, 4],
            }
        )
        rol = df.groupby("a").rolling(3)
        result = getattr(rol, func)()
        expected = DataFrame(
            {
                "b": 4 * [np.nan] + expected_values[0] + 2 * [np.nan],
                "c": 4 * [np.nan] + expected_values[1] + 2 * [np.nan],
            },
            index=MultiIndex.from_tuples(
                [
                    ("g1", 0, "b"),
                    ("g1", 0, "c"),
                    ("g1", 2, "b"),
                    ("g1", 2, "c"),
                    ("g1", 3, "b"),
                    ("g1", 3, "c"),
                    ("g2", 1, "b"),
                    ("g2", 1, "c"),
                ],
                names=["a", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply(self, raw):
        g = self.frame.groupby("A")
        r = g.rolling(window=4)

        # reduction
        result = r.apply(lambda x: x.sum(), raw=raw)
        expected = g.apply(lambda x: x.rolling(4).apply(lambda y: y.sum(), raw=raw))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    def test_rolling_apply_mutability(self):
        # GH 14013
        df = DataFrame({"A": ["foo"] * 3 + ["bar"] * 3, "B": [1] * 6})
        g = df.groupby("A")

        mi = MultiIndex.from_tuples(
            [("bar", 3), ("bar", 4), ("bar", 5), ("foo", 0), ("foo", 1), ("foo", 2)]
        )

        mi.names = ["A", None]
        # Grouped column should not be a part of the output
        expected = DataFrame([np.nan, 2.0, 2.0] * 2, columns=["B"], index=mi)

        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

        # Call an arbitrary function on the groupby
        g.sum()

        # Make sure nothing has been mutated
        result = g.rolling(window=2).sum()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("expected_value,raw_value", [[1.0, True], [0.0, False]])
    def test_groupby_rolling(self, expected_value, raw_value):
        # GH 31754

        def foo(x):
            return int(isinstance(x, np.ndarray))

        df = DataFrame({"id": [1, 1, 1], "value": [1, 2, 3]})
        result = df.groupby("id").value.rolling(1).apply(foo, raw=raw_value)
        expected = Series(
            [expected_value] * 3,
            index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=["id", None]),
            name="value",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self):
        # GH 35552
        series = Series(range(1, 6))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 5,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))),
        )
        tm.assert_series_equal(result, expected)

        series = Series(range(1, 5))
        result = series.groupby(series).rolling(center=True, window=3).mean()
        expected = Series(
            [np.nan] * 4,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))),
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 6, "b": range(11)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                    ("b", 10),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": ["a"] * 5 + ["b"] * 5, "b": range(10)})
        result = df.groupby("a").rolling(center=True, window=3).mean()
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_center_on(self):
        # GH 37141
        df = DataFrame(
            data={
                "Date": date_range("2020-01-01", "2020-01-10"),
                "gb": ["group_1"] * 6 + ["group_2"] * 4,
                "value": range(10),
            }
        )
        result = (
            df.groupby("gb")
            .rolling(6, on="Date", center=True, min_periods=1)
            .value.mean()
        )
        expected = Series(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5],
            name="value",
            index=MultiIndex.from_tuples(
                (
                    ("group_1", Timestamp("2020-01-01")),
                    ("group_1", Timestamp("2020-01-02")),
                    ("group_1", Timestamp("2020-01-03")),
                    ("group_1", Timestamp("2020-01-04")),
                    ("group_1", Timestamp("2020-01-05")),
                    ("group_1", Timestamp("2020-01-06")),
                    ("group_2", Timestamp("2020-01-07")),
                    ("group_2", Timestamp("2020-01-08")),
                    ("group_2", Timestamp("2020-01-09")),
                    ("group_2", Timestamp("2020-01-10")),
                ),
                names=["gb", "Date"],
            ),
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("min_periods", [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods):
        # GH 36040
        df = DataFrame({"group": ["A"] * 10 + ["B"] * 10, "data": range(20)})

        window_size = 5
        result = (
            df.groupby("group")
            .rolling(window_size, center=True, min_periods=min_periods)
            .mean()
        )
        result = result.reset_index()[["group", "data"]]

        grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean = [x + 10.0 for x in grp_A_mean]

        num_nans = max(0, min_periods - 3)  # For window_size of 5
        nans = [np.nan] * num_nans
        grp_A_expected = nans + grp_A_mean[num_nans : 10 - num_nans] + nans
        grp_B_expected = nans + grp_B_mean[num_nans : 10 - num_nans] + nans

        expected = DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10, "data": grp_A_expected + grp_B_expected}
        )

        tm.assert_frame_equal(result, expected)

    def test_groupby_subselect_rolling(self):
        # GH 35486
        df = DataFrame(
            {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0], "c": [10, 20, 30, 20]}
        )
        result = df.groupby("a")[["b"]].rolling(2).max()
        expected = DataFrame(
            [np.nan, np.nan, 2.0, np.nan],
            columns=["b"],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
        )
        tm.assert_frame_equal(result, expected)

        result = df.groupby("a")["b"].rolling(2).max()
        expected = Series(
            [np.nan, np.nan, 2.0, np.nan],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
            name="b",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_custom_indexer(self):
        # GH 35557
        class SimpleIndexer(BaseIndexer):
            def get_window_bounds(
                self, num_values=0, min_periods=None, center=None, closed=None
            ):
                min_periods = self.window_size if min_periods is None else 0
                end = np.arange(num_values, dtype=np.int64) + 1
                start = end.copy() - self.window_size
                start[start < 0] = min_periods
                return start, end

        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5
        )
        result = (
            df.groupby(df.index)
            .rolling(SimpleIndexer(window_size=3), min_periods=1)
            .sum()
        )
        expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_subset_with_closed(self):
        # GH 35549
        df = DataFrame(
            {
                "column1": range(6),
                "column2": range(6),
                "group": 3 * ["A", "B"],
                "date": [Timestamp("2019-01-01")] * 6,
            }
        )
        result = (
            df.groupby("group").rolling("1D", on="date", closed="left")["column1"].sum()
        )
        expected = Series(
            [np.nan, 0.0, 2.0, np.nan, 1.0, 4.0],
            index=MultiIndex.from_tuples(
                [("A", Timestamp("2019-01-01"))] * 3
                + [("B", Timestamp("2019-01-01"))] * 3,
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_subset_rolling_subset_with_closed(self):
        # GH 35549
        df = DataFrame(
            {
                "column1": range(6),
                "column2": range(6),
                "group": 3 * ["A", "B"],
                "date": [Timestamp("2019-01-01")] * 6,
            }
        )

        result = (
            df.groupby("group")[["column1", "date"]]
            .rolling("1D", on="date", closed="left")["column1"]
            .sum()
        )
        expected = Series(
            [np.nan, 0.0, 2.0, np.nan, 1.0, 4.0],
            index=MultiIndex.from_tuples(
                [("A", Timestamp("2019-01-01"))] * 3
                + [("B", Timestamp("2019-01-01"))] * 3,
                names=["group", "date"],
            ),
            name="column1",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_groupby_rolling_index_changed(self, func):
        # GH: #36018 nlevels of MultiIndex changed
        ds = Series(
            [1, 2, 2],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("c", "z")], names=["1", "2"]
            ),
            name="a",
        )

        result = getattr(ds.groupby(ds).rolling(2), func)()
        expected = Series(
            [np.nan, np.nan, 2.0],
            index=MultiIndex.from_tuples(
                [(1, "a", "x"), (2, "a", "y"), (2, "c", "z")], names=["a", "1", "2"]
            ),
            name="a",
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_empty_frame(self):
        # GH 36197
        expected = DataFrame({"s1": []})
        result = expected.groupby("s1").rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns="s1")
        # GH-38057 from_tuples gives empty object dtype, we now get float/int levels
        # expected.index = MultiIndex.from_tuples([], names=["s1", None])
        expected.index = MultiIndex.from_product(
            [Index([], dtype="float64"), Index([], dtype="int64")], names=["s1", None]
        )
        tm.assert_frame_equal(result, expected)

        expected = DataFrame({"s1": [], "s2": []})
        result = expected.groupby(["s1", "s2"]).rolling(window=1).sum()
        # GH 32262
        expected = expected.drop(columns=["s1", "s2"])
        expected.index = MultiIndex.from_product(
            [
                Index([], dtype="float64"),
                Index([], dtype="float64"),
                Index([], dtype="int64"),
            ],
            names=["s1", "s2", None],
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_string_index(self):
        # GH: 36727
        df = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9)],
                ["B", "group_1", Timestamp(2019, 1, 2, 9)],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9)],
                ["H", "group_1", Timestamp(2019, 1, 6, 9)],
                ["E", "group_2", Timestamp(2019, 1, 20, 9)],
            ],
            columns=["index", "group", "eventTime"],
        ).set_index("index")

        groups = df.groupby("group")
        df["count_to_date"] = groups.cumcount()
        rolling_groups = groups.rolling("10d", on="eventTime")
        result = rolling_groups.apply(lambda df: df.shape[0])
        expected = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9), 1.0],
                ["B", "group_1", Timestamp(2019, 1, 2, 9), 2.0],
                ["H", "group_1", Timestamp(2019, 1, 6, 9), 3.0],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9), 1.0],
                ["E", "group_2", Timestamp(2019, 1, 20, 9), 1.0],
            ],
            columns=["index", "group", "eventTime", "count_to_date"],
        ).set_index(["group", "index"])
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_no_sort(self):
        # GH 36889
        result = (
            DataFrame({"foo": [2, 1], "bar": [2, 1]})
            .groupby("foo", sort=False)
            .rolling(1)
            .min()
        )
        expected = DataFrame(
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            columns=["foo", "bar"],
            index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=["foo", None]),
        )
        # GH 32262
        expected = expected.drop(columns="foo")
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_count_closed_on(self):
        # GH 35869
        df = DataFrame(
            {
                "column1": range(6),
                "column2": range(6),
                "group": 3 * ["A", "B"],
                "date": date_range(end="20190101", periods=6),
            }
        )
        result = (
            df.groupby("group")
            .rolling("3d", on="date", closed="left")["column1"]
            .count()
        )
        expected = Series(
            [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0],
            name="column1",
            index=MultiIndex.from_tuples(
                [
                    ("A", Timestamp("2018-12-27")),
                    ("A", Timestamp("2018-12-29")),
                    ("A", Timestamp("2018-12-31")),
                    ("B", Timestamp("2018-12-28")),
                    ("B", Timestamp("2018-12-30")),
                    ("B", Timestamp("2019-01-01")),
                ],
                names=["group", "date"],
            ),
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        ("func", "kwargs"),
        [("rolling", {"window": 2, "min_periods": 1}), ("expanding", {})],
    )
    def test_groupby_rolling_sem(self, func, kwargs):
        # GH: 26476
        df = DataFrame(
            [["a", 1], ["a", 2], ["b", 1], ["b", 2], ["b", 3]], columns=["a", "b"]
        )
        result = getattr(df.groupby("a"), func)(**kwargs).sem()
        expected = DataFrame(
            {"a": [np.nan] * 5, "b": [np.nan, 0.70711, np.nan, 0.70711, 0.70711]},
            index=MultiIndex.from_tuples(
                [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("b", 4)], names=["a", None]
            ),
        )
        # GH 32262
        expected = expected.drop(columns="a")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("rollings", "key"), [({"on": "a"}, "a"), ({"on": None}, "index")]
    )
    def test_groupby_rolling_nans_in_index(self, rollings, key):
        # GH: 34617
        df = DataFrame(
            {
                "a": to_datetime(["2020-06-01 12:00", "2020-06-01 14:00", np.nan]),
                "b": [1, 2, 3],
                "c": [1, 1, 1],
            }
        )
        if key == "index":
            df = df.set_index("a")
        with pytest.raises(ValueError, match=f"{key} must be monotonic"):
            df.groupby("c").rolling("60min", **rollings)

    @pytest.mark.parametrize("group_keys", [True, False])
    def test_groupby_rolling_group_keys(self, group_keys):
        # GH 37641
        # GH 38523: GH 37641 actually was not a bug.
        # group_keys only applies to groupby.apply directly
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        s = Series([1, 2, 3], index=index)
        result = s.groupby(["idx1", "idx2"], group_keys=group_keys).rolling(1).mean()
        expected = Series(
            [1.0, 2.0, 3.0],
            index=MultiIndex.from_tuples(
                [
                    ("val1", "val1", "val1", "val1"),
                    ("val1", "val1", "val1", "val1"),
                    ("val2", "val2", "val2", "val2"),
                ],
                names=["idx1", "idx2", "idx1", "idx2"],
            ),
        )
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_index_level_and_column_label(self):
        # The groupby keys should not appear as a resulting column
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        df = DataFrame({"A": [1, 1, 2], "B": range(3)}, index=index)
        result = df.groupby(["idx1", "A"]).rolling(1).mean()
        expected = DataFrame(
            {"B": [0.0, 1.0, 2.0]},
            index=MultiIndex.from_tuples(
                [
                    ("val1", 1, "val1", "val1"),
                    ("val1", 1, "val1", "val1"),
                    ("val2", 2, "val2", "val2"),
                ],
                names=["idx1", "A", "idx1", "idx2"],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_resulting_multiindex(self):
        # a few different cases checking the created MultiIndex of the result
        # https://github.com/pandas-dev/pandas/pull/38057

        # grouping by 1 columns -> 2-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4})
        result = df.groupby("b").rolling(3).mean()
        expected_index = MultiIndex.from_tuples(
            [(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)],
            names=["b", None],
        )
        tm.assert_index_equal(result.index, expected_index)

        # grouping by 2 columns -> 3-level MI as result
        df = DataFrame({"a": np.arange(12.0), "b": [1, 2] * 6, "c": [1, 2, 3, 4] * 3})
        result = df.groupby(["b", "c"]).rolling(2).sum()
        expected_index = MultiIndex.from_tuples(
            [
                (1, 1, 0),
                (1, 1, 4),
                (1, 1, 8),
                (1, 3, 2),
                (1, 3, 6),
                (1, 3, 10),
                (2, 2, 1),
                (2, 2, 5),
                (2, 2, 9),
                (2, 4, 3),
                (2, 4, 7),
                (2, 4, 11),
            ],
            names=["b", "c", None],
        )
        tm.assert_index_equal(result.index, expected_index)

        # grouping with 1 level on dataframe with 2-level MI -> 3-level MI as result
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4, "c": [1, 2, 3, 4] * 2})
        df = df.set_index("c", append=True)
        result = df.groupby("b").rolling(3).mean()
        expected_index = MultiIndex.from_tuples(
            [
                (1, 0, 1),
                (1, 2, 3),
                (1, 4, 1),
                (1, 6, 3),
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 2),
                (2, 7, 4),
            ],
            names=["b", None, "c"],
        )
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_object_doesnt_affect_groupby_apply(self):
        # GH 39732
        g = self.frame.groupby("A")
        expected = g.apply(lambda x: x.rolling(4).sum()).index
        _ = g.rolling(window=4)
        result = g.apply(lambda x: x.rolling(4).sum()).index
        tm.assert_index_equal(result, expected)
        assert not g.mutated
        assert not g.grouper.mutated

    @pytest.mark.parametrize(
        "columns", [MultiIndex.from_tuples([("A", ""), ("B", "C")]), ["A", "B"]]
    )
    def test_by_column_not_in_values(self, columns):
        # GH 32262
        df = DataFrame([[1, 0]] * 20 + [[2, 0]] * 12 + [[3, 0]] * 8, columns=columns)
        g = df.groupby("A")
        original_obj = g.obj.copy(deep=True)
        r = g.rolling(4)
        result = r.sum()
        assert "A" not in result.columns
        tm.assert_frame_equal(g.obj, original_obj)

    def test_groupby_level(self):
        # GH 38523, 38787
        arrays = [
            ["Falcon", "Falcon", "Parrot", "Parrot"],
            ["Captive", "Wild", "Captive", "Wild"],
        ]
        index = MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
        df = DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
        result = df.groupby(level=0)["Max Speed"].rolling(2).sum()
        expected = Series(
            [np.nan, 740.0, np.nan, 50.0],
            index=MultiIndex.from_tuples(
                [
                    ("Falcon", "Falcon", "Captive"),
                    ("Falcon", "Falcon", "Wild"),
                    ("Parrot", "Parrot", "Captive"),
                    ("Parrot", "Parrot", "Wild"),
                ],
                names=["Animal", "Animal", "Type"],
            ),
            name="Max Speed",
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "by, expected_data",
        [
            [["id"], {"num": [100.0, 150.0, 150.0, 200.0]}],
            [
                ["id", "index"],
                {
                    "date": [
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                    ],
                    "num": [100.0, 200.0, 150.0, 250.0],
                },
            ],
        ],
    )
    def test_as_index_false(self, by, expected_data):
        # GH 39433
        data = [
            ["A", "2018-01-01", 100.0],
            ["A", "2018-01-02", 200.0],
            ["B", "2018-01-01", 150.0],
            ["B", "2018-01-02", 250.0],
        ]
        df = DataFrame(data, columns=["id", "date", "num"])
        df["date"] = to_datetime(df["date"])
        df = df.set_index(["date"])

        gp_by = [getattr(df, attr) for attr in by]
        result = (
            df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
        )

        expected = {"id": ["A", "A", "B", "B"]}
        expected.update(expected_data)
        expected = DataFrame(
            expected,
            index=df.index,
        )
        tm.assert_frame_equal(result, expected)


class TestExpanding:
    def setup_method(self):
        self.frame = DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})

    @pytest.mark.parametrize(
        "f", ["sum", "mean", "min", "max", "count", "kurt", "skew"]
    )
    def test_expanding(self, f):
        g = self.frame.groupby("A")
        r = g.expanding()

        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.expanding(), f)())
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["std", "var"])
    def test_expanding_ddof(self, f):
        g = self.frame.groupby("A")
        r = g.expanding()

        result = getattr(r, f)(ddof=0)
        expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    def test_expanding_quantile(self, interpolation):
        g = self.frame.groupby("A")
        r = g.expanding()

        result = r.quantile(0.4, interpolation=interpolation)
        expected = g.apply(
            lambda x: x.expanding().quantile(0.4, interpolation=interpolation)
        )
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_expanding_corr_cov(self, f):
        g = self.frame.groupby("A")
        r = g.expanding()

        result = getattr(r, f)(self.frame)

        def func(x):
            return getattr(x.expanding(), f)(self.frame)

        expected = g.apply(func)
        # GH 39591: groupby.apply returns 1 instead of nan for windows
        # with all nan values
        null_idx = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan
        # GH 39591: The grouped column should be all np.nan
        # (groupby.apply inserts 0s for cov)
        expected["A"] = np.nan
        tm.assert_frame_equal(result, expected)

        result = getattr(r.B, f)(pairwise=True)

        def func(x):
            return getattr(x.B.expanding(), f)(pairwise=True)

        expected = g.apply(func)
        tm.assert_series_equal(result, expected)

    def test_expanding_apply(self, raw):
        g = self.frame.groupby("A")
        r = g.expanding()

        # reduction
        result = r.apply(lambda x: x.sum(), raw=raw)
        expected = g.apply(lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw))
        # groupby.apply doesn't drop the grouped-by column
        expected = expected.drop("A", axis=1)
        # GH 39732
        expected_index = MultiIndex.from_arrays([self.frame["A"], range(40)])
        expected.index = expected_index
        tm.assert_frame_equal(result, expected)


class TestEWM:
    @pytest.mark.parametrize(
        "method, expected_data",
        [
            ["mean", [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666]],
            ["std", [np.nan, 0.707107, 0.963624, 1.177164]],
            ["var", [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]],
        ],
    )
    def test_methods(self, method, expected_data):
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

        expected = df.groupby("A").apply(lambda x: getattr(x.ewm(com=1.0), method)())
        # There may be a bug in the above statement; not returning the correct index
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    @pytest.mark.parametrize(
        "method, expected_data",
        [["corr", [np.nan, 1.0, 1.0, 1]], ["cov", [np.nan, 0.5, 0.928571, 1.385714]]],
    )
    def test_pairwise_methods(self, method, expected_data):
        # GH 16037
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0, "B"),
                    ("a", 1, "B"),
                    ("a", 2, "B"),
                    ("a", 3, "B"),
                ],
                names=["A", None, None],
            ),
        )
        tm.assert_frame_equal(result, expected)

        expected = df.groupby("A").apply(lambda x: getattr(x.ewm(com=1.0), method)())
        tm.assert_frame_equal(result, expected)

    def test_times(self, times_frame):
        # GH 40951
        halflife = "23 days"
        result = times_frame.groupby("A").ewm(halflife=halflife, times="C").mean()
        expected = DataFrame(
            {
                "B": [
                    0.0,
                    0.507534,
                    1.020088,
                    1.537661,
                    0.0,
                    0.567395,
                    1.221209,
                    0.0,
                    0.653141,
                    1.195003,
                ]
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 3),
                    ("a", 6),
                    ("a", 9),
                    ("b", 1),
                    ("b", 4),
                    ("b", 7),
                    ("c", 2),
                    ("c", 5),
                    ("c", 8),
                ],
                names=["A", None],
            ),
        )
        tm.assert_frame_equal(result, expected)

    def test_times_vs_apply(self, times_frame):
        # GH 40951
        halflife = "23 days"
        result = times_frame.groupby("A").ewm(halflife=halflife, times="C").mean()
        expected = (
            times_frame.groupby("A")
            .apply(lambda x: x.ewm(halflife=halflife, times="C").mean())
            .iloc[[0, 3, 6, 9, 1, 4, 7, 2, 5, 8]]
            .reset_index(drop=True)
        )
        tm.assert_frame_equal(result.reset_index(drop=True), expected)

    def test_times_array(self, times_frame):
        # GH 40951
        halflife = "23 days"
        result = times_frame.groupby("A").ewm(halflife=halflife, times="C").mean()
        expected = (
            times_frame.groupby("A")
            .ewm(halflife=halflife, times=times_frame["C"].values)
            .mean()
        )
        tm.assert_frame_equal(result, expected)
