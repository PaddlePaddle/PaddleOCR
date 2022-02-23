import datetime as dt
from datetime import datetime

import dateutil
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    to_timedelta,
)
import pandas._testing as tm


class TestDatetimeConcat:
    def test_concat_datetime64_block(self):
        from pandas.core.indexes.datetimes import date_range

        rng = date_range("1/1/2000", periods=10)

        df = DataFrame({"time": rng})

        result = concat([df, df])
        assert (result.iloc[:10]["time"] == rng).all()
        assert (result.iloc[10:]["time"] == rng).all()

    def test_concat_datetime_datetime64_frame(self):
        # GH#2624
        rows = []
        rows.append([datetime(2010, 1, 1), 1])
        rows.append([datetime(2010, 1, 2), "hi"])

        df2_obj = DataFrame.from_records(rows, columns=["date", "test"])

        ind = date_range(start="2000/1/1", freq="D", periods=10)
        df1 = DataFrame({"date": ind, "test": range(10)})

        # it works!
        concat([df1, df2_obj])

    def test_concat_datetime_timezone(self):
        # GH 18523
        idx1 = date_range("2011-01-01", periods=3, freq="H", tz="Europe/Paris")
        idx2 = date_range(start=idx1[0], end=idx1[-1], freq="H")
        df1 = DataFrame({"a": [1, 2, 3]}, index=idx1)
        df2 = DataFrame({"b": [1, 2, 3]}, index=idx2)
        result = concat([df1, df2], axis=1)

        exp_idx = (
            DatetimeIndex(
                [
                    "2011-01-01 00:00:00+01:00",
                    "2011-01-01 01:00:00+01:00",
                    "2011-01-01 02:00:00+01:00",
                ],
                freq="H",
            )
            .tz_convert("UTC")
            .tz_convert("Europe/Paris")
        )

        expected = DataFrame(
            [[1, 1], [2, 2], [3, 3]], index=exp_idx, columns=["a", "b"]
        )

        tm.assert_frame_equal(result, expected)

        idx3 = date_range("2011-01-01", periods=3, freq="H", tz="Asia/Tokyo")
        df3 = DataFrame({"b": [1, 2, 3]}, index=idx3)
        result = concat([df1, df3], axis=1)

        exp_idx = DatetimeIndex(
            [
                "2010-12-31 15:00:00+00:00",
                "2010-12-31 16:00:00+00:00",
                "2010-12-31 17:00:00+00:00",
                "2010-12-31 23:00:00+00:00",
                "2011-01-01 00:00:00+00:00",
                "2011-01-01 01:00:00+00:00",
            ]
        )

        expected = DataFrame(
            [
                [np.nan, 1],
                [np.nan, 2],
                [np.nan, 3],
                [1, np.nan],
                [2, np.nan],
                [3, np.nan],
            ],
            index=exp_idx,
            columns=["a", "b"],
        )

        tm.assert_frame_equal(result, expected)

        # GH 13783: Concat after resample
        result = concat([df1.resample("H").mean(), df2.resample("H").mean()], sort=True)
        expected = DataFrame(
            {"a": [1, 2, 3] + [np.nan] * 3, "b": [np.nan] * 3 + [1, 2, 3]},
            index=idx1.append(idx1),
        )
        tm.assert_frame_equal(result, expected)

    def test_concat_datetimeindex_freq(self):
        # GH 3232
        # Monotonic index result
        dr = date_range("01-Jan-2013", periods=100, freq="50L", tz="UTC")
        data = list(range(100))
        expected = DataFrame(data, index=dr)
        result = concat([expected[:50], expected[50:]])
        tm.assert_frame_equal(result, expected)

        # Non-monotonic index result
        result = concat([expected[50:], expected[:50]])
        expected = DataFrame(data[50:] + data[:50], index=dr[50:].append(dr[:50]))
        expected.index._data.freq = None
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_datetime_object_index(self):
        # https://github.com/pandas-dev/pandas/issues/11058
        idx = Index(
            [dt.date(2013, 1, 1), dt.date(2014, 1, 1), dt.date(2015, 1, 1)],
            dtype="object",
        )

        s = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [
                    [1, 2],
                    idx[:-1],
                ],
                names=["first", "second"],
            ),
        )
        s2 = Series(
            ["a", "b"],
            index=MultiIndex.from_arrays(
                [[1, 2], idx[::2]],
                names=["first", "second"],
            ),
        )
        mi = MultiIndex.from_arrays(
            [[1, 2, 2], idx],
            names=["first", "second"],
        )
        assert mi.levels[1].dtype == object

        expected = DataFrame(
            [["a", "a"], ["b", np.nan], [np.nan, "b"]],
            index=mi,
        )
        result = concat([s, s2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_NaT_series(self):
        # GH 11693
        # test for merging NaT series with datetime series.
        x = Series(
            date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="US/Eastern")
        )
        y = Series(pd.NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])

        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # all NaT with tz
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns, US/Eastern]")
        result = concat([y, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # without tz
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h"))
        y = Series(date_range("20151124 10:00", "20151124 11:00", freq="1h"))
        y[:] = pd.NaT
        expected = Series([x[0], x[1], pd.NaT, pd.NaT])
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # all NaT without tz
        x[:] = pd.NaT
        expected = Series(pd.NaT, index=range(4), dtype="datetime64[ns]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_concat_NaT_dataframes(self, tz):
        # GH 12396

        first = DataFrame([[pd.NaT], [pd.NaT]])
        first = first.apply(lambda x: x.dt.tz_localize(tz))
        second = DataFrame(
            [[Timestamp("2015/01/01", tz=tz)], [Timestamp("2016/01/01", tz=tz)]],
            index=[2, 3],
        )
        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz),
                Timestamp("2016/01/01", tz=tz),
            ]
        )

        result = concat([first, second], axis=0)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    @pytest.mark.parametrize("s", [pd.NaT, Timestamp("20150101")])
    def test_concat_NaT_dataframes_all_NaT_axis_0(self, tz1, tz2, s):
        # GH 12396

        # tz-naive
        first = DataFrame([[pd.NaT], [pd.NaT]]).apply(lambda x: x.dt.tz_localize(tz1))
        second = DataFrame([s]).apply(lambda x: x.dt.tz_localize(tz2))

        result = concat([first, second], axis=0)
        expected = DataFrame(Series([pd.NaT, pd.NaT, s], index=[0, 1, 0]))
        expected = expected.apply(lambda x: x.dt.tz_localize(tz2))
        if tz1 != tz2:
            expected = expected.astype(object)

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_dataframes_all_NaT_axis_1(self, tz1, tz2):
        # GH 12396

        first = DataFrame(Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1))
        second = DataFrame(Series([pd.NaT]).dt.tz_localize(tz2), columns=[1])
        expected = DataFrame(
            {
                0: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1),
                1: Series([pd.NaT, pd.NaT]).dt.tz_localize(tz2),
            }
        )
        result = concat([first, second], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("tz1", [None, "UTC"])
    @pytest.mark.parametrize("tz2", [None, "UTC"])
    def test_concat_NaT_series_dataframe_all_NaT(self, tz1, tz2):
        # GH 12396

        # tz-naive
        first = Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1)
        second = DataFrame(
            [
                [Timestamp("2015/01/01", tz=tz2)],
                [Timestamp("2016/01/01", tz=tz2)],
            ],
            index=[2, 3],
        )

        expected = DataFrame(
            [
                pd.NaT,
                pd.NaT,
                Timestamp("2015/01/01", tz=tz2),
                Timestamp("2016/01/01", tz=tz2),
            ]
        )
        if tz1 != tz2:
            expected = expected.astype(object)

        result = concat([first, second])
        tm.assert_frame_equal(result, expected)


class TestTimezoneConcat:
    def test_concat_tz_series(self):
        # gh-11755: tz and no tz
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        y = Series(date_range("2012-01-01", "2012-01-02"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # gh-11887: concat tz and object
        x = Series(date_range("20151124 08:00", "20151124 09:00", freq="1h", tz="UTC"))
        y = Series(["a", "b"])
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

        # see gh-12217 and gh-12306
        # Concatenating two UTC times
        first = DataFrame([[datetime(2016, 1, 1)]])
        first[0] = first[0].dt.tz_localize("UTC")

        second = DataFrame([[datetime(2016, 1, 2)]])
        second[0] = second[0].dt.tz_localize("UTC")

        result = concat([first, second])
        assert result[0].dtype == "datetime64[ns, UTC]"

        # Concatenating two London times
        first = DataFrame([[datetime(2016, 1, 1)]])
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 2)]])
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        assert result[0].dtype == "datetime64[ns, Europe/London]"

        # Concatenating 2+1 London times
        first = DataFrame([[datetime(2016, 1, 1)], [datetime(2016, 1, 2)]])
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 3)]])
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        assert result[0].dtype == "datetime64[ns, Europe/London]"

        # Concat'ing 1+2 London times
        first = DataFrame([[datetime(2016, 1, 1)]])
        first[0] = first[0].dt.tz_localize("Europe/London")

        second = DataFrame([[datetime(2016, 1, 2)], [datetime(2016, 1, 3)]])
        second[0] = second[0].dt.tz_localize("Europe/London")

        result = concat([first, second])
        assert result[0].dtype == "datetime64[ns, Europe/London]"

    def test_concat_tz_series_tzlocal(self):
        # see gh-13583
        x = [
            Timestamp("2011-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2011-02-01", tz=dateutil.tz.tzlocal()),
        ]
        y = [
            Timestamp("2012-01-01", tz=dateutil.tz.tzlocal()),
            Timestamp("2012-02-01", tz=dateutil.tz.tzlocal()),
        ]

        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y))
        assert result.dtype == "datetime64[ns, tzlocal()]"

    def test_concat_tz_series_with_datetimelike(self):
        # see gh-12620: tz and timedelta
        x = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-02-01", tz="US/Eastern"),
        ]
        y = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

        # tz and period
        y = [pd.Period("2011-03", freq="M"), pd.Period("2011-04", freq="M")]
        result = concat([Series(x), Series(y)], ignore_index=True)
        tm.assert_series_equal(result, Series(x + y, dtype="object"))

    def test_concat_tz_frame(self):
        df2 = DataFrame(
            {
                "A": Timestamp("20130102", tz="US/Eastern"),
                "B": Timestamp("20130603", tz="CET"),
            },
            index=range(5),
        )

        # concat
        df3 = concat([df2.A.to_frame(), df2.B.to_frame()], axis=1)
        tm.assert_frame_equal(df2, df3)

    def test_concat_multiple_tzs(self):
        # GH#12467
        # combining datetime tz-aware and naive DataFrames
        ts1 = Timestamp("2015-01-01", tz=None)
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="EST")

        df1 = DataFrame({"time": [ts1]})
        df2 = DataFrame({"time": [ts2]})
        df3 = DataFrame({"time": [ts3]})

        results = concat([df1, df2]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts2]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df1, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts1, ts3]}, dtype=object)
        tm.assert_frame_equal(results, expected)

        results = concat([df2, df3]).reset_index(drop=True)
        expected = DataFrame({"time": [ts2, ts3]})
        tm.assert_frame_equal(results, expected)

    @pytest.mark.filterwarnings("ignore:Timestamp.freq is deprecated:FutureWarning")
    def test_concat_multiindex_with_tz(self):
        # GH 6606
        df = DataFrame(
            {
                "dt": [
                    datetime(2014, 1, 1),
                    datetime(2014, 1, 2),
                    datetime(2014, 1, 3),
                ],
                "b": ["A", "B", "C"],
                "c": [1, 2, 3],
                "d": [4, 5, 6],
            }
        )
        df["dt"] = df["dt"].apply(lambda d: Timestamp(d, tz="US/Pacific"))
        df = df.set_index(["dt", "b"])

        exp_idx1 = DatetimeIndex(
            ["2014-01-01", "2014-01-02", "2014-01-03"] * 2, tz="US/Pacific", name="dt"
        )
        exp_idx2 = Index(["A", "B", "C"] * 2, name="b")
        exp_idx = MultiIndex.from_arrays([exp_idx1, exp_idx2])
        expected = DataFrame(
            {"c": [1, 2, 3] * 2, "d": [4, 5, 6] * 2}, index=exp_idx, columns=["c", "d"]
        )

        result = concat([df, df])
        tm.assert_frame_equal(result, expected)

    def test_concat_tz_not_aligned(self):
        # GH#22796
        ts = pd.to_datetime([1, 2]).tz_localize("UTC")
        a = DataFrame({"A": ts})
        b = DataFrame({"A": ts, "B": ts})
        result = concat([a, b], sort=True, ignore_index=True)
        expected = DataFrame(
            {"A": list(ts) + list(ts), "B": [pd.NaT, pd.NaT] + list(ts)}
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "t1",
        [
            "2015-01-01",
            pytest.param(
                pd.NaT,
                marks=pytest.mark.xfail(
                    reason="GH23037 incorrect dtype when concatenating"
                ),
            ),
        ],
    )
    def test_concat_tz_NaT(self, t1):
        # GH#22796
        # Concatenating tz-aware multicolumn DataFrames
        ts1 = Timestamp(t1, tz="UTC")
        ts2 = Timestamp("2015-01-01", tz="UTC")
        ts3 = Timestamp("2015-01-01", tz="UTC")

        df1 = DataFrame([[ts1, ts2]])
        df2 = DataFrame([[ts3]])

        result = concat([df1, df2])
        expected = DataFrame([[ts1, ts2], [ts3, pd.NaT]], index=[0, 0])

        tm.assert_frame_equal(result, expected)

    def test_concat_tz_with_empty(self):
        # GH 9188
        result = concat(
            [DataFrame(date_range("2000", periods=1, tz="UTC")), DataFrame()]
        )
        expected = DataFrame(date_range("2000", periods=1, tz="UTC"))
        tm.assert_frame_equal(result, expected)


class TestPeriodConcat:
    def test_concat_period_series(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="D"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="Period[D]")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)

    def test_concat_period_multiple_freq_series(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-10-01", "2016-01-01"], freq="M"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

    def test_concat_period_other_series(self):
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="M"))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

        # non-period
        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(DatetimeIndex(["2015-11-01", "2015-12-01"]))
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"

        x = Series(pd.PeriodIndex(["2015-11-01", "2015-12-01"], freq="D"))
        y = Series(["A", "B"])
        expected = Series([x[0], x[1], y[0], y[1]], dtype="object")
        result = concat([x, y], ignore_index=True)
        tm.assert_series_equal(result, expected)
        assert result.dtype == "object"


def test_concat_timedelta64_block():
    rng = to_timedelta(np.arange(10), unit="s")

    df = DataFrame({"time": rng})

    result = concat([df, df])
    tm.assert_frame_equal(result.iloc[:10], df)
    tm.assert_frame_equal(result.iloc[10:], df)
