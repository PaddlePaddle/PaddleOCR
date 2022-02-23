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

import pandas.tseries.offsets as offsets


class TestRollingTS:

    # rolling time-series friendly
    # xref GH13327

    def setup_method(self, method):

        self.regular = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        ).set_index("A")

        self.ragged = DataFrame({"B": range(5)})
        self.ragged.index = [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130101 09:00:02"),
            Timestamp("20130101 09:00:03"),
            Timestamp("20130101 09:00:05"),
            Timestamp("20130101 09:00:06"),
        ]

    def test_doc_string(self):

        df = DataFrame(
            {"B": [0, 1, 2, np.nan, 4]},
            index=[
                Timestamp("20130101 09:00:00"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:05"),
                Timestamp("20130101 09:00:06"),
            ],
        )
        df
        df.rolling("2s").sum()

    def test_invalid_window_non_int(self):

        # not a valid freq
        msg = "passed window foobar is not compatible with a datetimelike index"
        with pytest.raises(ValueError, match=msg):
            self.regular.rolling(window="foobar")
        # not a datetimelike index
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            self.regular.reset_index().rolling(window="foobar")

    @pytest.mark.parametrize("freq", ["2MS", offsets.MonthBegin(2)])
    def test_invalid_window_nonfixed(self, freq):

        # non-fixed freqs
        msg = "\\<2 \\* MonthBegins\\> is a non-fixed frequency"
        with pytest.raises(ValueError, match=msg):
            self.regular.rolling(window=freq)

    @pytest.mark.parametrize("freq", ["1D", offsets.Day(2), "2ms"])
    def test_valid_window(self, freq):
        self.regular.rolling(window=freq)

    @pytest.mark.parametrize("minp", [1.0, "foo", np.array([1, 2, 3])])
    def test_invalid_minp(self, minp):
        # non-integer min_periods
        msg = (
            r"local variable 'minp' referenced before assignment|"
            "min_periods must be an integer"
        )
        with pytest.raises(ValueError, match=msg):
            self.regular.rolling(window="1D", min_periods=minp)

    def test_on(self):

        df = self.regular

        # not a valid column
        msg = (
            r"invalid on specified as foobar, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2s", on="foobar")

        # column is valid
        df = df.copy()
        df["C"] = date_range("20130101", periods=len(df))
        df.rolling(window="2d", on="C").sum()

        # invalid columns
        msg = "window must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.rolling(window="2d", on="B")

        # ok even though on non-selected
        df.rolling(window="2d", on="C").B.sum()

    def test_monotonic_on(self):

        # on/index must be monotonic
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )

        assert df.A.is_monotonic
        df.rolling("2s", on="A").sum()

        df = df.set_index("A")
        assert df.index.is_monotonic
        df.rolling("2s").sum()

    def test_non_monotonic_on(self):
        # GH 19248
        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        )
        df = df.set_index("A")
        non_monotonic_index = df.index.to_list()
        non_monotonic_index[0] = non_monotonic_index[3]
        df.index = non_monotonic_index

        assert not df.index.is_monotonic

        msg = "index must be monotonic"
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s").sum()

        df = df.reset_index()

        msg = (
            r"invalid on specified as A, must be a column "
            "\\(of DataFrame\\), an Index or None"
        )
        with pytest.raises(ValueError, match=msg):
            df.rolling("2s", on="A").sum()

    def test_frame_on(self):

        df = DataFrame(
            {"B": range(5), "C": date_range("20130101 09:00:00", periods=5, freq="3s")}
        )

        df["A"] = [
            Timestamp("20130101 09:00:00"),
            Timestamp("20130101 09:00:02"),
            Timestamp("20130101 09:00:03"),
            Timestamp("20130101 09:00:05"),
            Timestamp("20130101 09:00:06"),
        ]

        # we are doing simulating using 'on'
        expected = df.set_index("A").rolling("2s").B.sum().reset_index(drop=True)

        result = df.rolling("2s", on="A").B.sum()
        tm.assert_series_equal(result, expected)

        # test as a frame
        # we should be ignoring the 'on' as an aggregation column
        # note that the expected is setting, computing, and resetting
        # so the columns need to be switched compared
        # to the actual result where they are ordered as in the
        # original
        expected = (
            df.set_index("A").rolling("2s")[["B"]].sum().reset_index()[["B", "A"]]
        )

        result = df.rolling("2s", on="A")[["B"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_frame_on2(self):

        # using multiple aggregation columns
        df = DataFrame(
            {
                "A": [0, 1, 2, 3, 4],
                "B": [0, 1, 2, np.nan, 4],
                "C": Index(
                    [
                        Timestamp("20130101 09:00:00"),
                        Timestamp("20130101 09:00:02"),
                        Timestamp("20130101 09:00:03"),
                        Timestamp("20130101 09:00:05"),
                        Timestamp("20130101 09:00:06"),
                    ]
                ),
            },
            columns=["A", "C", "B"],
        )

        expected1 = DataFrame(
            {"A": [0.0, 1, 3, 3, 7], "B": [0, 1, 3, np.nan, 4], "C": df["C"]},
            columns=["A", "C", "B"],
        )

        result = df.rolling("2s", on="C").sum()
        expected = expected1
        tm.assert_frame_equal(result, expected)

        expected = Series([0, 1, 3, np.nan, 4], name="B")
        result = df.rolling("2s", on="C").B.sum()
        tm.assert_series_equal(result, expected)

        expected = expected1[["A", "B", "C"]]
        result = df.rolling("2s", on="C")[["A", "B", "C"]].sum()
        tm.assert_frame_equal(result, expected)

    def test_basic_regular(self):

        df = self.regular.copy()

        df.index = date_range("20130101", periods=5, freq="D")
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="1D").sum()
        tm.assert_frame_equal(result, expected)

        df.index = date_range("20130101", periods=5, freq="2D")
        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(window=1, min_periods=1).sum()
        result = df.rolling(window="2D", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(window=1).sum()
        result = df.rolling(window="2D").sum()
        tm.assert_frame_equal(result, expected)

    def test_min_periods(self):

        # compare for min_periods
        df = self.regular

        # these slightly different
        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.rolling(2, min_periods=1).sum()
        result = df.rolling("2s", min_periods=1).sum()
        tm.assert_frame_equal(result, expected)

    def test_closed(self):

        # xref GH13965

        df = DataFrame(
            {"A": [1] * 5},
            index=[
                Timestamp("20130101 09:00:01"),
                Timestamp("20130101 09:00:02"),
                Timestamp("20130101 09:00:03"),
                Timestamp("20130101 09:00:04"),
                Timestamp("20130101 09:00:06"),
            ],
        )

        # closed must be 'right', 'left', 'both', 'neither'
        msg = "closed must be 'right', 'left', 'both' or 'neither'"
        with pytest.raises(ValueError, match=msg):
            self.regular.rolling(window="2s", closed="blabla")

        expected = df.copy()
        expected["A"] = [1.0, 2, 2, 2, 1]
        result = df.rolling("2s", closed="right").sum()
        tm.assert_frame_equal(result, expected)

        # default should be 'right'
        result = df.rolling("2s").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [1.0, 2, 3, 3, 2]
        result = df.rolling("2s", closed="both").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 2, 2, 1]
        result = df.rolling("2s", closed="left").sum()
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        expected["A"] = [np.nan, 1.0, 1, 1, np.nan]
        result = df.rolling("2s", closed="neither").sum()
        tm.assert_frame_equal(result, expected)

    def test_ragged_sum(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 3, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=2).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, np.nan, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s").sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 5, 7]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="4s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="4s", min_periods=3).sum()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 3, 6, 9]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).sum()
        expected = df.copy()
        expected["B"] = [0.0, 1, 3, 6, 10]
        tm.assert_frame_equal(result, expected)

    def test_ragged_mean(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).mean()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_median(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).median()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_quantile(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).quantile(0.5)
        expected = df.copy()
        expected["B"] = [0.0, 1, 1.5, 3.0, 3.5]
        tm.assert_frame_equal(result, expected)

    def test_ragged_std(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).std(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="1s", min_periods=1).std(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).std(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] + [0.5] * 4
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).std(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan, 0.707107, 1.0, 1.0, 1.290994]
        tm.assert_frame_equal(result, expected)

    def test_ragged_var(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).var(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="1s", min_periods=1).var(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="3s", min_periods=1).var(ddof=0)
        expected = df.copy()
        expected["B"] = [0.0] + [0.25] * 4
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).var(ddof=1)
        expected = df.copy()
        expected["B"] = [np.nan, 0.5, 1.0, 1.0, 1 + 2 / 3.0]
        tm.assert_frame_equal(result, expected)

    def test_ragged_skew(self):

        df = self.ragged
        result = df.rolling(window="3s", min_periods=1).skew()
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).skew()
        expected = df.copy()
        expected["B"] = [np.nan] * 2 + [0.0, 0.0, 0.0]
        tm.assert_frame_equal(result, expected)

    def test_ragged_kurt(self):

        df = self.ragged
        result = df.rolling(window="3s", min_periods=1).kurt()
        expected = df.copy()
        expected["B"] = [np.nan] * 5
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).kurt()
        expected = df.copy()
        expected["B"] = [np.nan] * 4 + [-1.2]
        tm.assert_frame_equal(result, expected)

    def test_ragged_count(self):

        df = self.ragged
        result = df.rolling(window="1s", min_periods=1).count()
        expected = df.copy()
        expected["B"] = [1.0, 1, 1, 1, 1]
        tm.assert_frame_equal(result, expected)

        df = self.ragged
        result = df.rolling(window="1s").count()
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).count()
        expected = df.copy()
        expected["B"] = [1.0, 1, 2, 1, 2]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=2).count()
        expected = df.copy()
        expected["B"] = [np.nan, np.nan, 2, np.nan, 2]
        tm.assert_frame_equal(result, expected)

    def test_regular_min(self):

        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [0.0, 1, 2, 3, 4]}
        ).set_index("A")
        result = df.rolling("1s").min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": [5, 4, 3, 4, 5]}
        ).set_index("A")

        tm.assert_frame_equal(result, expected)
        result = df.rolling("2s").min()
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling("5s").min()
        expected = df.copy()
        expected["B"] = [5.0, 4, 3, 3, 3]
        tm.assert_frame_equal(result, expected)

    def test_ragged_min(self):

        df = self.ragged

        result = df.rolling(window="1s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 1, 1, 3, 3]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).min()
        expected = df.copy()
        expected["B"] = [0.0, 0, 0, 1, 1]
        tm.assert_frame_equal(result, expected)

    def test_perf_min(self):

        N = 10000

        dfp = DataFrame(
            {"B": np.random.randn(N)}, index=date_range("20130101", periods=N, freq="s")
        )
        expected = dfp.rolling(2, min_periods=1).min()
        result = dfp.rolling("2s").min()
        assert ((result - expected) < 0.01).all().bool()

        expected = dfp.rolling(200, min_periods=1).min()
        result = dfp.rolling("200s").min()
        assert ((result - expected) < 0.01).all().bool()

    def test_ragged_max(self):

        df = self.ragged

        result = df.rolling(window="1s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="2s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

        result = df.rolling(window="5s", min_periods=1).max()
        expected = df.copy()
        expected["B"] = [0.0, 1, 2, 3, 4]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "freq, op, result_data",
        [
            ("ms", "min", [0.0] * 10),
            ("ms", "mean", [0.0] * 9 + [2.0 / 9]),
            ("ms", "max", [0.0] * 9 + [2.0]),
            ("s", "min", [0.0] * 10),
            ("s", "mean", [0.0] * 9 + [2.0 / 9]),
            ("s", "max", [0.0] * 9 + [2.0]),
            ("min", "min", [0.0] * 10),
            ("min", "mean", [0.0] * 9 + [2.0 / 9]),
            ("min", "max", [0.0] * 9 + [2.0]),
            ("h", "min", [0.0] * 10),
            ("h", "mean", [0.0] * 9 + [2.0 / 9]),
            ("h", "max", [0.0] * 9 + [2.0]),
            ("D", "min", [0.0] * 10),
            ("D", "mean", [0.0] * 9 + [2.0 / 9]),
            ("D", "max", [0.0] * 9 + [2.0]),
        ],
    )
    def test_freqs_ops(self, freq, op, result_data):
        # GH 21096
        index = date_range(start="2018-1-1 01:00:00", freq=f"1{freq}", periods=10)
        s = Series(data=0, index=index)
        s.iloc[1] = np.nan
        s.iloc[-1] = 2
        result = getattr(s.rolling(window=f"10{freq}"), op)()
        expected = Series(data=result_data, index=index)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "f",
        [
            "sum",
            "mean",
            pytest.param(
                "count",
                marks=pytest.mark.filterwarnings("ignore:min_periods:FutureWarning"),
            ),
            "median",
            "std",
            "var",
            "kurt",
            "skew",
            "min",
            "max",
        ],
    )
    def test_all(self, f):

        # simple comparison of integer vs time-based windowing
        df = self.regular * 2
        er = df.rolling(window=1)
        r = df.rolling(window="1s")

        result = getattr(r, f)()
        expected = getattr(er, f)()
        tm.assert_frame_equal(result, expected)

        result = r.quantile(0.5)
        expected = er.quantile(0.5)
        tm.assert_frame_equal(result, expected)

    def test_all2(self, arithmetic_win_operators):
        f = arithmetic_win_operators
        # more sophisticated comparison of integer vs.
        # time-based windowing
        df = DataFrame(
            {"B": np.arange(50)}, index=date_range("20130101", periods=50, freq="H")
        )
        # in-range data
        dft = df.between_time("09:00", "16:00")

        r = dft.rolling(window="5H")

        result = getattr(r, f)()

        # we need to roll the days separately
        # to compare with a time-based roll
        # finally groupby-apply will return a multi-index
        # so we need to drop the day
        def agg_by_day(x):
            x = x.between_time("09:00", "16:00")
            return getattr(x.rolling(5, min_periods=1), f)()

        expected = (
            df.groupby(df.index.day).apply(agg_by_day).reset_index(level=0, drop=True)
        )

        tm.assert_frame_equal(result, expected)

    def test_groupby_monotonic(self):

        # GH 15130
        # we don't need to validate monotonicity when grouping

        data = [
            ["David", "1/1/2015", 100],
            ["David", "1/5/2015", 500],
            ["David", "5/30/2015", 50],
            ["David", "7/25/2015", 50],
            ["Ryan", "1/4/2014", 100],
            ["Ryan", "1/19/2015", 500],
            ["Ryan", "3/31/2016", 50],
            ["Joe", "7/1/2015", 100],
            ["Joe", "9/9/2015", 500],
            ["Joe", "10/15/2015", 50],
        ]

        df = DataFrame(data=data, columns=["name", "date", "amount"])
        df["date"] = to_datetime(df["date"])

        expected = (
            df.set_index("date")
            .groupby("name")
            .apply(lambda x: x.rolling("180D")["amount"].sum())
        )
        result = df.groupby("name").rolling("180D", on="date")["amount"].sum()
        tm.assert_series_equal(result, expected)

    def test_non_monotonic(self):
        # GH 13966 (similar to #15130, closed by #15175)

        dates = date_range(start="2016-01-01 09:30:00", periods=20, freq="s")
        df = DataFrame(
            {
                "A": [1] * 20 + [2] * 12 + [3] * 8,
                "B": np.concatenate((dates, dates)),
                "C": np.arange(40),
            }
        )

        result = df.groupby("A").rolling("4s", on="B").C.mean()
        expected = (
            df.set_index("B").groupby("A").apply(lambda x: x.rolling("4s")["C"].mean())
        )
        tm.assert_series_equal(result, expected)

        df2 = df.sort_values("B")
        result = df2.groupby("A").rolling("4s", on="B").C.mean()
        tm.assert_series_equal(result, expected)

    def test_rolling_cov_offset(self):
        # GH16058

        idx = date_range("2017-01-01", periods=24, freq="1h")
        ss = Series(np.arange(len(idx)), index=idx)

        result = ss.rolling("2h").cov()
        expected = Series([np.nan] + [0.5] * (len(idx) - 1), index=idx)
        tm.assert_series_equal(result, expected)

        expected2 = ss.rolling(2, min_periods=1).cov()
        tm.assert_series_equal(result, expected2)

        result = ss.rolling("3h").cov()
        expected = Series([np.nan, 0.5] + [1.0] * (len(idx) - 2), index=idx)
        tm.assert_series_equal(result, expected)

        expected2 = ss.rolling(3, min_periods=1).cov()
        tm.assert_series_equal(result, expected2)

    def test_rolling_on_decreasing_index(self):
        # GH-19248, GH-32385
        index = [
            Timestamp("20190101 09:00:30"),
            Timestamp("20190101 09:00:27"),
            Timestamp("20190101 09:00:20"),
            Timestamp("20190101 09:00:18"),
            Timestamp("20190101 09:00:10"),
        ]

        df = DataFrame({"column": [3, 4, 4, 5, 6]}, index=index)
        result = df.rolling("5s").min()
        expected = DataFrame({"column": [3.0, 3.0, 4.0, 4.0, 6.0]}, index=index)
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_empty(self):
        # GH-32385
        df = DataFrame({"column": []}, index=[])
        result = df.rolling("5s").min()
        expected = DataFrame({"column": []}, index=[])
        tm.assert_frame_equal(result, expected)

    def test_rolling_on_multi_index_level(self):
        # GH-15584
        df = DataFrame(
            {"column": range(6)},
            index=MultiIndex.from_product(
                [date_range("20190101", periods=3), range(2)], names=["date", "seq"]
            ),
        )
        result = df.rolling("10d", on=df.index.get_level_values("date")).sum()
        expected = DataFrame(
            {"column": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]}, index=df.index
        )
        tm.assert_frame_equal(result, expected)
