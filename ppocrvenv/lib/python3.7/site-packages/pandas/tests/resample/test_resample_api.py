from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range

dti = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="Min")

test_series = Series(np.random.rand(len(dti)), dti)
_test_frame = DataFrame({"A": test_series, "B": test_series, "C": np.arange(len(dti))})


@pytest.fixture
def test_frame():
    return _test_frame.copy()


def test_str():

    r = test_series.resample("H")
    assert (
        "DatetimeIndexResampler [freq=<Hour>, axis=0, closed=left, "
        "label=left, convention=start, origin=start_day]" in str(r)
    )

    r = test_series.resample("H", origin="2000-01-01")
    assert (
        "DatetimeIndexResampler [freq=<Hour>, axis=0, closed=left, "
        "label=left, convention=start, origin=2000-01-01 00:00:00]" in str(r)
    )


def test_api():

    r = test_series.resample("H")
    result = r.mean()
    assert isinstance(result, Series)
    assert len(result) == 217

    r = test_series.to_frame().resample("H")
    result = r.mean()
    assert isinstance(result, DataFrame)
    assert len(result) == 217


def test_groupby_resample_api():

    # GH 12448
    # .groupby(...).resample(...) hitting warnings
    # when appropriate
    df = DataFrame(
        {
            "date": date_range(start="2016-01-01", periods=4, freq="W"),
            "group": [1, 1, 2, 2],
            "val": [5, 6, 7, 8],
        }
    ).set_index("date")

    # replication step
    i = (
        date_range("2016-01-03", periods=8).tolist()
        + date_range("2016-01-17", periods=8).tolist()
    )
    index = pd.MultiIndex.from_arrays([[1] * 8 + [2] * 8, i], names=["group", "date"])
    expected = DataFrame({"val": [5] * 7 + [6] + [7] * 7 + [8]}, index=index)
    result = df.groupby("group").apply(lambda x: x.resample("1D").ffill())[["val"]]
    tm.assert_frame_equal(result, expected)


def test_groupby_resample_on_api():

    # GH 15021
    # .groupby(...).resample(on=...) results in an unexpected
    # keyword warning.
    df = DataFrame(
        {
            "key": ["A", "B"] * 5,
            "dates": date_range("2016-01-01", periods=10),
            "values": np.random.randn(10),
        }
    )

    expected = df.set_index("dates").groupby("key").resample("D").mean()

    result = df.groupby("key").resample("D", on="dates").mean()
    tm.assert_frame_equal(result, expected)


def test_pipe(test_frame):
    # GH17905

    # series
    r = test_series.resample("H")
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_series_equal(result, expected)

    # dataframe
    r = test_frame.resample("H")
    expected = r.max() - r.mean()
    result = r.pipe(lambda x: x.max() - x.mean())
    tm.assert_frame_equal(result, expected)


def test_getitem(test_frame):

    r = test_frame.resample("H")
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns)

    r = test_frame.resample("H")["B"]
    assert r._selected_obj.name == test_frame.columns[1]

    # technically this is allowed
    r = test_frame.resample("H")["A", "B"]
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])

    r = test_frame.resample("H")["A", "B"]
    tm.assert_index_equal(r._selected_obj.columns, test_frame.columns[[0, 1]])


@pytest.mark.parametrize("key", [["D"], ["A", "D"]])
def test_select_bad_cols(key, test_frame):
    g = test_frame.resample("H")
    # 'A' should not be referenced as a bad column...
    # will have to rethink regex if you change message!
    msg = r"^\"Columns not found: 'D'\"$"
    with pytest.raises(KeyError, match=msg):
        g[key]


def test_attribute_access(test_frame):

    r = test_frame.resample("H")
    tm.assert_series_equal(r.A.sum(), r["A"].sum())


def test_api_compat_before_use():

    # make sure that we are setting the binner
    # on these attributes
    for attr in ["groups", "ngroups", "indices"]:
        rng = date_range("1/1/2012", periods=100, freq="S")
        ts = Series(np.arange(len(rng)), index=rng)
        rs = ts.resample("30s")

        # before use
        getattr(rs, attr)

        # after grouper is initialized is ok
        rs.mean()
        getattr(rs, attr)


def tests_skip_nuisance(test_frame):

    df = test_frame
    df["D"] = "foo"
    r = df.resample("H")
    result = r[["A", "B"]].sum()
    expected = pd.concat([r.A.sum(), r.B.sum()], axis=1)
    tm.assert_frame_equal(result, expected)

    expected = r[["A", "B", "C"]].sum()
    result = r.sum()
    tm.assert_frame_equal(result, expected)


def test_downsample_but_actually_upsampling():

    # this is reindex / asfreq
    rng = date_range("1/1/2012", periods=100, freq="S")
    ts = Series(np.arange(len(rng), dtype="int64"), index=rng)
    result = ts.resample("20s").asfreq()
    expected = Series(
        [0, 20, 40, 60, 80],
        index=date_range("2012-01-01 00:00:00", freq="20s", periods=5),
    )
    tm.assert_series_equal(result, expected)


def test_combined_up_downsampling_of_irregular():

    # since we are really doing an operation like this
    # ts2.resample('2s').mean().ffill()
    # preserve these semantics

    rng = date_range("1/1/2012", periods=100, freq="S")
    ts = Series(np.arange(len(rng)), index=rng)
    ts2 = ts.iloc[[0, 1, 2, 3, 5, 7, 11, 15, 16, 25, 30]]

    result = ts2.resample("2s").mean().ffill()
    expected = Series(
        [
            0.5,
            2.5,
            5.0,
            7.0,
            7.0,
            11.0,
            11.0,
            15.0,
            16.0,
            16.0,
            16.0,
            16.0,
            25.0,
            25.0,
            25.0,
            30.0,
        ],
        index=pd.DatetimeIndex(
            [
                "2012-01-01 00:00:00",
                "2012-01-01 00:00:02",
                "2012-01-01 00:00:04",
                "2012-01-01 00:00:06",
                "2012-01-01 00:00:08",
                "2012-01-01 00:00:10",
                "2012-01-01 00:00:12",
                "2012-01-01 00:00:14",
                "2012-01-01 00:00:16",
                "2012-01-01 00:00:18",
                "2012-01-01 00:00:20",
                "2012-01-01 00:00:22",
                "2012-01-01 00:00:24",
                "2012-01-01 00:00:26",
                "2012-01-01 00:00:28",
                "2012-01-01 00:00:30",
            ],
            dtype="datetime64[ns]",
            freq="2S",
        ),
    )
    tm.assert_series_equal(result, expected)


def test_transform():

    r = test_series.resample("20min")
    expected = test_series.groupby(pd.Grouper(freq="20min")).transform("mean")
    result = r.transform("mean")
    tm.assert_series_equal(result, expected)


def test_fillna():

    # need to upsample here
    rng = date_range("1/1/2012", periods=10, freq="2S")
    ts = Series(np.arange(len(rng), dtype="int64"), index=rng)
    r = ts.resample("s")

    expected = r.ffill()
    result = r.fillna(method="ffill")
    tm.assert_series_equal(result, expected)

    expected = r.bfill()
    result = r.fillna(method="bfill")
    tm.assert_series_equal(result, expected)

    msg = (
        r"Invalid fill method\. Expecting pad \(ffill\), backfill "
        r"\(bfill\) or nearest\. Got 0"
    )
    with pytest.raises(ValueError, match=msg):
        r.fillna(0)


def test_apply_without_aggregation():

    # both resample and groupby should work w/o aggregation
    r = test_series.resample("20min")
    g = test_series.groupby(pd.Grouper(freq="20min"))

    for t in [g, r]:
        result = t.apply(lambda x: x)
        tm.assert_series_equal(result, test_series)


def test_agg_consistency():

    # make sure that we are consistent across
    # similar aggregations with and w/o selection list
    df = DataFrame(
        np.random.randn(1000, 3),
        index=date_range("1/1/2012", freq="S", periods=1000),
        columns=["A", "B", "C"],
    )

    r = df.resample("3T")

    msg = r"Column\(s\) \['r1', 'r2'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({"r1": "mean", "r2": "sum"})


def test_agg_consistency_int_str_column_mix():
    # GH#39025
    df = DataFrame(
        np.random.randn(1000, 2),
        index=date_range("1/1/2012", freq="S", periods=1000),
        columns=[1, "a"],
    )

    r = df.resample("3T")

    msg = r"Column\(s\) \[2, 'b'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        r.agg({2: "mean", "b": "sum"})


# TODO: once GH 14008 is fixed, move these tests into
# `Base` test class


def test_agg():
    # test with all three Resampler apis and TimeGrouper

    np.random.seed(1234)
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    df = DataFrame(np.random.rand(10, 2), columns=list("AB"), index=index)
    df_col = df.reset_index()
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), df.index], names=["index", "date"]
    )
    r = df.resample("2D")
    cases = [
        r,
        df_col.resample("2D", on="date"),
        df_mult.resample("2D", level="date"),
        df.groupby(pd.Grouper(freq="2D")),
    ]

    a_mean = r["A"].mean()
    a_std = r["A"].std()
    a_sum = r["A"].sum()
    b_mean = r["B"].mean()
    b_std = r["B"].std()
    b_sum = r["B"].sum()

    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_product([["A", "B"], ["mean", "std"]])
    for t in cases:
        result = t.aggregate([np.mean, np.std])
        tm.assert_frame_equal(result, expected)

    expected = pd.concat([a_mean, b_std], axis=1)
    for t in cases:
        result = t.aggregate({"A": np.mean, "B": np.std})
        tm.assert_frame_equal(result, expected, check_like=True)

    expected = pd.concat([a_mean, a_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples([("A", "mean"), ("A", "std")])
    for t in cases:
        result = t.aggregate({"A": ["mean", "std"]})
        tm.assert_frame_equal(result, expected)

    expected = pd.concat([a_mean, a_sum], axis=1)
    expected.columns = ["mean", "sum"]
    for t in cases:
        result = t["A"].aggregate(["mean", "sum"])
    tm.assert_frame_equal(result, expected)

    msg = "nested renamer is not supported"
    for t in cases:
        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t.aggregate({"A": {"mean": "mean", "sum": "sum"}})

    expected = pd.concat([a_mean, a_sum, b_mean, b_sum], axis=1)
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "mean"), ("A", "sum"), ("B", "mean2"), ("B", "sum2")]
    )
    for t in cases:
        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t.aggregate(
                {
                    "A": {"mean": "mean", "sum": "sum"},
                    "B": {"mean2": "mean", "sum2": "sum"},
                }
            )

    expected = pd.concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    )
    for t in cases:
        result = t.aggregate({"A": ["mean", "std"], "B": ["mean", "std"]})
        tm.assert_frame_equal(result, expected, check_like=True)

    expected = pd.concat([a_mean, a_sum, b_mean, b_sum], axis=1)
    expected.columns = pd.MultiIndex.from_tuples(
        [
            ("r1", "A", "mean"),
            ("r1", "A", "sum"),
            ("r2", "B", "mean"),
            ("r2", "B", "sum"),
        ]
    )


def test_agg_misc():
    # test with all three Resampler apis and TimeGrouper

    np.random.seed(1234)
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    df = DataFrame(np.random.rand(10, 2), columns=list("AB"), index=index)
    df_col = df.reset_index()
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), df.index], names=["index", "date"]
    )

    r = df.resample("2D")
    cases = [
        r,
        df_col.resample("2D", on="date"),
        df_mult.resample("2D", level="date"),
        df.groupby(pd.Grouper(freq="2D")),
    ]

    # passed lambda
    for t in cases:
        result = t.agg({"A": np.sum, "B": lambda x: np.std(x, ddof=1)})
        rcustom = t["B"].apply(lambda x: np.std(x, ddof=1))
        expected = pd.concat([r["A"].sum(), rcustom], axis=1)
        tm.assert_frame_equal(result, expected, check_like=True)

    # agg with renamers
    expected = pd.concat(
        [t["A"].sum(), t["B"].sum(), t["A"].mean(), t["B"].mean()], axis=1
    )
    expected.columns = pd.MultiIndex.from_tuples(
        [("result1", "A"), ("result1", "B"), ("result2", "A"), ("result2", "B")]
    )

    msg = r"Column\(s\) \['result1', 'result2'\] do not exist"
    for t in cases:
        with pytest.raises(KeyError, match=msg):
            t[["A", "B"]].agg({"result1": np.sum, "result2": np.mean})

    # agg with different hows
    expected = pd.concat(
        [t["A"].sum(), t["A"].std(), t["B"].mean(), t["B"].std()], axis=1
    )
    expected.columns = pd.MultiIndex.from_tuples(
        [("A", "sum"), ("A", "std"), ("B", "mean"), ("B", "std")]
    )
    for t in cases:
        result = t.agg({"A": ["sum", "std"], "B": ["mean", "std"]})
        tm.assert_frame_equal(result, expected, check_like=True)

    # equivalent of using a selection list / or not
    for t in cases:
        result = t[["A", "B"]].agg({"A": ["sum", "std"], "B": ["mean", "std"]})
        tm.assert_frame_equal(result, expected, check_like=True)

    msg = "nested renamer is not supported"

    # series like aggs
    for t in cases:
        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t["A"].agg({"A": ["sum", "std"]})

        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t["A"].agg({"A": ["sum", "std"], "B": ["mean", "std"]})

    # errors
    # invalid names in the agg specification
    msg = r"Column\(s\) \['B'\] do not exist"
    for t in cases:
        with pytest.raises(KeyError, match=msg):
            t[["A"]].agg({"A": ["sum", "std"], "B": ["mean", "std"]})


def test_agg_nested_dicts():

    np.random.seed(1234)
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")
    index.name = "date"
    df = DataFrame(np.random.rand(10, 2), columns=list("AB"), index=index)
    df_col = df.reset_index()
    df_mult = df_col.copy()
    df_mult.index = pd.MultiIndex.from_arrays(
        [range(10), df.index], names=["index", "date"]
    )
    r = df.resample("2D")
    cases = [
        r,
        df_col.resample("2D", on="date"),
        df_mult.resample("2D", level="date"),
        df.groupby(pd.Grouper(freq="2D")),
    ]

    msg = "nested renamer is not supported"
    for t in cases:
        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t.aggregate({"r1": {"A": ["mean", "sum"]}, "r2": {"B": ["mean", "sum"]}})

    for t in cases:

        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t[["A", "B"]].agg(
                {"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}}
            )

        with pytest.raises(pd.core.base.SpecificationError, match=msg):
            t.agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})


def test_try_aggregate_non_existing_column():
    # GH 16766
    data = [
        {"dt": datetime(2017, 6, 1, 0), "x": 1.0, "y": 2.0},
        {"dt": datetime(2017, 6, 1, 1), "x": 2.0, "y": 2.0},
        {"dt": datetime(2017, 6, 1, 2), "x": 3.0, "y": 1.5},
    ]
    df = DataFrame(data).set_index("dt")

    # Error as we don't have 'z' column
    msg = r"Column\(s\) \['z'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.resample("30T").agg({"x": ["mean"], "y": ["median"], "z": ["sum"]})


def test_selection_api_validation():
    # GH 13500
    index = date_range(datetime(2005, 1, 1), datetime(2005, 1, 10), freq="D")

    rng = np.arange(len(index), dtype=np.int64)
    df = DataFrame(
        {"date": index, "a": rng},
        index=pd.MultiIndex.from_arrays([rng, index], names=["v", "d"]),
    )
    df_exp = DataFrame({"a": rng}, index=index)

    # non DatetimeIndex
    msg = (
        "Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'Int64Index'"
    )
    with pytest.raises(TypeError, match=msg):
        df.resample("2D", level="v")

    msg = "The Grouper cannot specify both a key and a level!"
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", on="date", level="d")

    msg = "unhashable type: 'list'"
    with pytest.raises(TypeError, match=msg):
        df.resample("2D", on=["a", "date"])

    msg = r"\"Level \['a', 'date'\] not found\""
    with pytest.raises(KeyError, match=msg):
        df.resample("2D", level=["a", "date"])

    # upsampling not allowed
    msg = (
        "Upsampling from level= or on= selection is not supported, use "
        r"\.set_index\(\.\.\.\) to explicitly set index to datetime-like"
    )
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", level="d").asfreq()
    with pytest.raises(ValueError, match=msg):
        df.resample("2D", on="date").asfreq()

    exp = df_exp.resample("2D").sum()
    exp.index.name = "date"
    tm.assert_frame_equal(exp, df.resample("2D", on="date").sum())

    exp.index.name = "d"
    tm.assert_frame_equal(exp, df.resample("2D", level="d").sum())


@pytest.mark.parametrize(
    "col_name", ["t2", "t2x", "t2q", "T_2M", "t2p", "t2m", "t2m1", "T2M"]
)
def test_agg_with_datetime_index_list_agg_func(col_name):
    # GH 22660
    # The parametrized column names would get converted to dates by our
    # date parser. Some would result in OutOfBoundsError (ValueError) while
    # others would result in OverflowError when passed into Timestamp.
    # We catch these errors and move on to the correct branch.
    df = DataFrame(
        list(range(200)),
        index=date_range(
            start="2017-01-01", freq="15min", periods=200, tz="Europe/Berlin"
        ),
        columns=[col_name],
    )
    result = df.resample("1d").aggregate(["mean"])
    expected = DataFrame(
        [47.5, 143.5, 195.5],
        index=date_range(start="2017-01-01", freq="D", periods=3, tz="Europe/Berlin"),
        columns=pd.MultiIndex(levels=[[col_name], ["mean"]], codes=[[0], [0]]),
    )
    tm.assert_frame_equal(result, expected)


def test_resample_agg_readonly():
    # GH#31710 cython needs to allow readonly data
    index = date_range("2020-01-01", "2020-01-02", freq="1h")
    arr = np.zeros_like(index)
    arr.setflags(write=False)

    ser = Series(arr, index=index)
    rs = ser.resample("1D")

    expected = Series([pd.Timestamp(0), pd.Timestamp(0)], index=index[::24])

    result = rs.agg("last")
    tm.assert_series_equal(result, expected)

    result = rs.agg("first")
    tm.assert_series_equal(result, expected)

    result = rs.agg("max")
    tm.assert_series_equal(result, expected)

    result = rs.agg("min")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods",
    [
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end",
            None,
            [0, 18, 27, 63],
            "20001002 00:26:00",
            4,
        ),
        (
            "20200101 8:26:35",
            "20200101 9:31:58",
            "77s",
            [1] * 51,
            "7min",
            "end",
            "right",
            [1, 6, 5, 6, 5, 6, 5, 6, 5, 6],
            "2020-01-01 09:30:45",
            10,
        ),
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end",
            "left",
            [0, 18, 27, 39, 24],
            "20001002 00:43:00",
            5,
        ),
        (
            "2000-10-01 23:30:00",
            "2000-10-02 00:26:00",
            "7min",
            [0, 3, 6, 9, 12, 15, 18, 21, 24],
            "17min",
            "end_day",
            None,
            [3, 15, 45, 45],
            "2000-10-02 00:29:00",
            4,
        ),
    ],
)
def test_end_and_end_day_origin(
    start,
    end,
    freq,
    data,
    resample_freq,
    origin,
    closed,
    exp_data,
    exp_end,
    exp_periods,
):
    rng = date_range(start, end, freq=freq)
    ts = Series(data, index=rng)

    res = ts.resample(resample_freq, origin=origin, closed=closed).sum()
    expected = Series(
        exp_data,
        index=date_range(end=exp_end, freq=resample_freq, periods=exp_periods),
    )

    tm.assert_series_equal(res, expected)
