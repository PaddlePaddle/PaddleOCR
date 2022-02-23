from textwrap import dedent

import numpy as np
import pytest

import pandas.util._test_decorators as td
from pandas.util._test_decorators import async_mark

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    TimedeltaIndex,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range

test_frame = DataFrame(
    {"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)},
    index=date_range("1/1/2000", freq="s", periods=40),
)


@async_mark()
@td.check_file_leaks
async def test_tab_complete_ipython6_warning(ip):
    from IPython.core.completer import provisionalcompleter

    code = dedent(
        """\
    import pandas._testing as tm
    s = tm.makeTimeSeries()
    rs = s.resample("D")
    """
    )
    await ip.run_code(code)

    # GH 31324 newer jedi version raises Deprecation warning;
    #  appears resolved 2021-02-02
    with tm.assert_produces_warning(None):
        with provisionalcompleter("ignore"):
            list(ip.Completer.completions("rs.", 1))


def test_deferred_with_groupby():

    # GH 12486
    # support deferred resample ops with groupby
    data = [
        ["2010-01-01", "A", 2],
        ["2010-01-02", "A", 3],
        ["2010-01-05", "A", 8],
        ["2010-01-10", "A", 7],
        ["2010-01-13", "A", 3],
        ["2010-01-01", "B", 5],
        ["2010-01-03", "B", 2],
        ["2010-01-04", "B", 1],
        ["2010-01-11", "B", 7],
        ["2010-01-14", "B", 3],
    ]

    df = DataFrame(data, columns=["date", "id", "score"])
    df.date = pd.to_datetime(df.date)

    def f(x):
        return x.set_index("date").resample("D").asfreq()

    expected = df.groupby("id").apply(f)
    result = df.set_index("date").groupby("id").resample("D").asfreq()
    tm.assert_frame_equal(result, expected)

    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W"),
            "group": [1, 1, 2, 2],
            "val": [5, 6, 7, 8],
        }
    ).set_index("date")

    def f(x):
        return x.resample("1D").ffill()

    expected = df.groupby("group").apply(f)
    result = df.groupby("group").resample("1D").ffill()
    tm.assert_frame_equal(result, expected)


def test_getitem():
    g = test_frame.groupby("A")

    expected = g.B.apply(lambda x: x.resample("2s").mean())

    result = g.resample("2s").B.mean()
    tm.assert_series_equal(result, expected)

    result = g.B.resample("2s").mean()
    tm.assert_series_equal(result, expected)

    result = g.resample("2s").mean().B
    tm.assert_series_equal(result, expected)


def test_getitem_multiple():

    # GH 13174
    # multiple calls after selection causing an issue with aliasing
    data = [{"id": 1, "buyer": "A"}, {"id": 2, "buyer": "B"}]
    df = DataFrame(data, index=date_range("2016-01-01", periods=2))
    r = df.groupby("id").resample("1D")
    result = r["buyer"].count()
    expected = Series(
        [1, 1],
        index=pd.MultiIndex.from_tuples(
            [(1, Timestamp("2016-01-01")), (2, Timestamp("2016-01-02"))],
            names=["id", None],
        ),
        name="buyer",
    )
    tm.assert_series_equal(result, expected)

    result = r["buyer"].count()
    tm.assert_series_equal(result, expected)


def test_groupby_resample_on_api_with_getitem():
    # GH 17813
    df = DataFrame(
        {"id": list("aabbb"), "date": date_range("1-1-2016", periods=5), "data": 1}
    )
    exp = df.set_index("date").groupby("id").resample("2D")["data"].sum()
    result = df.groupby("id").resample("2D", on="date")["data"].sum()
    tm.assert_series_equal(result, exp)


def test_groupby_with_origin():
    # GH 31809

    freq = "1399min"  # prime number that is smaller than 24h
    start, end = "1/1/2000 00:00:00", "1/31/2000 00:00"
    middle = "1/15/2000 00:00:00"

    rng = date_range(start, end, freq="1231min")  # prime number
    ts = Series(np.random.randn(len(rng)), index=rng)
    ts2 = ts[middle:end]

    # proves that grouper without a fixed origin does not work
    # when dealing with unusual frequencies
    simple_grouper = pd.Grouper(freq=freq)
    count_ts = ts.groupby(simple_grouper).agg("count")
    count_ts = count_ts[middle:end]
    count_ts2 = ts2.groupby(simple_grouper).agg("count")
    with pytest.raises(AssertionError, match="Index are different"):
        tm.assert_index_equal(count_ts.index, count_ts2.index)

    # test origin on 1970-01-01 00:00:00
    origin = Timestamp(0)
    adjusted_grouper = pd.Grouper(freq=freq, origin=origin)
    adjusted_count_ts = ts.groupby(adjusted_grouper).agg("count")
    adjusted_count_ts = adjusted_count_ts[middle:end]
    adjusted_count_ts2 = ts2.groupby(adjusted_grouper).agg("count")
    tm.assert_series_equal(adjusted_count_ts, adjusted_count_ts2)

    # test origin on 2049-10-18 20:00:00
    origin_future = Timestamp(0) + pd.Timedelta("1399min") * 30_000
    adjusted_grouper2 = pd.Grouper(freq=freq, origin=origin_future)
    adjusted2_count_ts = ts.groupby(adjusted_grouper2).agg("count")
    adjusted2_count_ts = adjusted2_count_ts[middle:end]
    adjusted2_count_ts2 = ts2.groupby(adjusted_grouper2).agg("count")
    tm.assert_series_equal(adjusted2_count_ts, adjusted2_count_ts2)

    # both grouper use an adjusted timestamp that is a multiple of 1399 min
    # they should be equals even if the adjusted_timestamp is in the future
    tm.assert_series_equal(adjusted_count_ts, adjusted2_count_ts2)


def test_nearest():

    # GH 17496
    # Resample nearest
    index = date_range("1/1/2000", periods=3, freq="T")
    result = Series(range(3), index=index).resample("20s").nearest()

    expected = Series(
        [0, 0, 1, 1, 1, 2, 2],
        index=pd.DatetimeIndex(
            [
                "2000-01-01 00:00:00",
                "2000-01-01 00:00:20",
                "2000-01-01 00:00:40",
                "2000-01-01 00:01:00",
                "2000-01-01 00:01:20",
                "2000-01-01 00:01:40",
                "2000-01-01 00:02:00",
            ],
            dtype="datetime64[ns]",
            freq="20S",
        ),
    )
    tm.assert_series_equal(result, expected)


def test_methods():
    g = test_frame.groupby("A")
    r = g.resample("2s")

    for f in ["first", "last", "median", "sem", "sum", "mean", "min", "max"]:
        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)())
        tm.assert_frame_equal(result, expected)

    for f in ["size"]:
        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)())
        tm.assert_series_equal(result, expected)

    for f in ["count"]:
        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)())
        tm.assert_frame_equal(result, expected)

    # series only
    for f in ["nunique"]:
        result = getattr(r.B, f)()
        expected = g.B.apply(lambda x: getattr(x.resample("2s"), f)())
        tm.assert_series_equal(result, expected)

    for f in ["nearest", "backfill", "ffill", "asfreq"]:
        result = getattr(r, f)()
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)())
        tm.assert_frame_equal(result, expected)

    result = r.ohlc()
    expected = g.apply(lambda x: x.resample("2s").ohlc())
    tm.assert_frame_equal(result, expected)

    for f in ["std", "var"]:
        result = getattr(r, f)(ddof=1)
        expected = g.apply(lambda x: getattr(x.resample("2s"), f)(ddof=1))
        tm.assert_frame_equal(result, expected)


def test_apply():

    g = test_frame.groupby("A")
    r = g.resample("2s")

    # reduction
    expected = g.resample("2s").sum()

    def f(x):
        return x.resample("2s").sum()

    result = r.apply(f)
    tm.assert_frame_equal(result, expected)

    def f(x):
        return x.resample("2s").apply(lambda y: y.sum())

    result = g.apply(f)
    # y.sum() results in int64 instead of int32 on 32-bit architectures
    expected = expected.astype("int64")
    tm.assert_frame_equal(result, expected)


def test_apply_with_mutated_index():
    # GH 15169
    index = date_range("1-1-2015", "12-31-15", freq="D")
    df = DataFrame(data={"col1": np.random.rand(len(index))}, index=index)

    def f(x):
        s = Series([1, 2], index=["a", "b"])
        return s

    expected = df.groupby(pd.Grouper(freq="M")).apply(f)

    result = df.resample("M").apply(f)
    tm.assert_frame_equal(result, expected)

    # A case for series
    expected = df["col1"].groupby(pd.Grouper(freq="M")).apply(f)
    result = df["col1"].resample("M").apply(f)
    tm.assert_series_equal(result, expected)


def test_apply_columns_multilevel():
    # GH 16231
    cols = pd.MultiIndex.from_tuples([("A", "a", "", "one"), ("B", "b", "i", "two")])
    ind = date_range(start="2017-01-01", freq="15Min", periods=8)
    df = DataFrame(np.array([0] * 16).reshape(8, 2), index=ind, columns=cols)
    agg_dict = {col: (np.sum if col[3] == "one" else np.mean) for col in df.columns}
    result = df.resample("H").apply(lambda x: agg_dict[x.name](x))
    expected = DataFrame(
        2 * [[0, 0.0]],
        index=date_range(start="2017-01-01", freq="1H", periods=2),
        columns=pd.MultiIndex.from_tuples(
            [("A", "a", "", "one"), ("B", "b", "i", "two")]
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_groupby_with_label():
    # GH 13235
    index = date_range("2000-01-01", freq="2D", periods=5)
    df = DataFrame(index=index, data={"col0": [0, 0, 1, 1, 2], "col1": [1, 1, 1, 1, 1]})
    result = df.groupby("col0").resample("1W", label="left").sum()

    mi = [
        np.array([0, 0, 1, 2]),
        pd.to_datetime(
            np.array(["1999-12-26", "2000-01-02", "2000-01-02", "2000-01-02"])
        ),
    ]
    mindex = pd.MultiIndex.from_arrays(mi, names=["col0", None])
    expected = DataFrame(
        data={"col0": [0, 0, 2, 2], "col1": [1, 1, 2, 1]}, index=mindex
    )

    tm.assert_frame_equal(result, expected)


def test_consistency_with_window():

    # consistent return values with window
    df = test_frame
    expected = pd.Int64Index([1, 2, 3], name="A")
    result = df.groupby("A").resample("2s").mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)

    result = df.groupby("A").rolling(20).mean()
    assert result.index.nlevels == 2
    tm.assert_index_equal(result.index.levels[0], expected)


def test_median_duplicate_columns():
    # GH 14233

    df = DataFrame(
        np.random.randn(20, 3),
        columns=list("aaa"),
        index=date_range("2012-01-01", periods=20, freq="s"),
    )
    df2 = df.copy()
    df2.columns = ["a", "b", "c"]
    expected = df2.resample("5s").median()
    result = df.resample("5s").median()
    expected.columns = result.columns
    tm.assert_frame_equal(result, expected)


def test_apply_to_one_column_of_df():
    # GH: 36951
    df = DataFrame(
        {"col": range(10), "col1": range(10, 20)},
        index=date_range("2012-01-01", periods=10, freq="20min"),
    )

    # access "col" via getattr -> make sure we handle AttributeError
    result = df.resample("H").apply(lambda group: group.col.sum())
    expected = Series(
        [3, 12, 21, 9], index=date_range("2012-01-01", periods=4, freq="H")
    )
    tm.assert_series_equal(result, expected)

    # access "col" via _getitem__ -> make sure we handle KeyErrpr
    result = df.resample("H").apply(lambda group: group["col"].sum())
    tm.assert_series_equal(result, expected)


def test_resample_groupby_agg():
    # GH: 33548
    df = DataFrame(
        {
            "cat": [
                "cat_1",
                "cat_1",
                "cat_2",
                "cat_1",
                "cat_2",
                "cat_1",
                "cat_2",
                "cat_1",
            ],
            "num": [5, 20, 22, 3, 4, 30, 10, 50],
            "date": [
                "2019-2-1",
                "2018-02-03",
                "2020-3-11",
                "2019-2-2",
                "2019-2-2",
                "2018-12-4",
                "2020-3-11",
                "2020-12-12",
            ],
        }
    )
    df["date"] = pd.to_datetime(df["date"])

    resampled = df.groupby("cat").resample("Y", on="date")
    expected = resampled.sum()
    result = resampled.agg({"num": "sum"})

    tm.assert_frame_equal(result, expected)


def test_resample_groupby_agg_listlike():
    # GH 42905
    ts = Timestamp("2021-02-28 00:00:00")
    df = DataFrame({"class": ["beta"], "value": [69]}, index=Index([ts], name="date"))
    resampled = df.groupby("class").resample("M")["value"]
    result = resampled.agg(["sum", "size"])
    expected = DataFrame(
        [[69, 1]],
        index=pd.MultiIndex.from_tuples([("beta", ts)], names=["class", "date"]),
        columns=["sum", "size"],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_empty(keys):
    # GH 26411
    df = DataFrame([], columns=["a", "b"], index=TimedeltaIndex([]))
    result = df.groupby(keys).resample(rule=pd.to_timedelta("00:00:01")).mean()
    expected = DataFrame(columns=["a", "b"]).set_index(keys, drop=False)
    if len(keys) == 1:
        expected.index.name = keys[0]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("consolidate", [True, False])
def test_resample_groupby_agg_object_dtype_all_nan(consolidate):
    # https://github.com/pandas-dev/pandas/issues/39329

    dates = date_range("2020-01-01", periods=15, freq="D")
    df1 = DataFrame({"key": "A", "date": dates, "col1": range(15), "col_object": "val"})
    df2 = DataFrame({"key": "B", "date": dates, "col1": range(15)})
    df = pd.concat([df1, df2], ignore_index=True)
    if consolidate:
        df = df._consolidate()

    result = df.groupby(["key"]).resample("W", on="date").min()
    idx = pd.MultiIndex.from_arrays(
        [
            ["A"] * 3 + ["B"] * 3,
            pd.to_datetime(["2020-01-05", "2020-01-12", "2020-01-19"] * 2),
        ],
        names=["key", "date"],
    )
    expected = DataFrame(
        {
            "key": ["A"] * 3 + ["B"] * 3,
            "date": pd.to_datetime(["2020-01-01", "2020-01-06", "2020-01-13"] * 2),
            "col1": [0, 5, 12] * 2,
            "col_object": ["val"] * 3 + [np.nan] * 3,
        },
        index=idx,
    )
    tm.assert_frame_equal(result, expected)
