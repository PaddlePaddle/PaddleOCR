from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import timedelta_range

from pandas.tseries.offsets import (
    BDay,
    Minute,
)

DATE_RANGE = (date_range, "dti", datetime(2005, 1, 1), datetime(2005, 1, 10))
PERIOD_RANGE = (period_range, "pi", datetime(2005, 1, 1), datetime(2005, 1, 10))
TIMEDELTA_RANGE = (timedelta_range, "tdi", "1 day", "10 day")

all_ts = pytest.mark.parametrize(
    "_index_factory,_series_name,_index_start,_index_end",
    [DATE_RANGE, PERIOD_RANGE, TIMEDELTA_RANGE],
)


@pytest.fixture()
def _index_factory():
    return period_range


@pytest.fixture
def create_index(_index_factory):
    def _create_index(*args, **kwargs):
        """return the _index_factory created using the args, kwargs"""
        return _index_factory(*args, **kwargs)

    return _create_index


# new test to check that all FutureWarning are triggered
def test_deprecating_on_loffset_and_base():
    # GH 31809

    idx = date_range("2001-01-01", periods=4, freq="T")
    df = DataFrame(data=4 * [range(2)], index=idx, columns=["a", "b"])

    with tm.assert_produces_warning(FutureWarning):
        pd.Grouper(freq="10s", base=0)
    with tm.assert_produces_warning(FutureWarning):
        pd.Grouper(freq="10s", loffset="0s")

    # not checking the stacklevel for .groupby().resample() because it's complicated to
    # reconcile it with the stacklevel for Series.resample() and DataFrame.resample();
    # see GH #37603
    with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
        df.groupby("a").resample("3T", base=0).sum()
    with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
        df.groupby("a").resample("3T", loffset="0s").sum()
    msg = "'offset' and 'base' cannot be present at the same time"
    with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
        with pytest.raises(ValueError, match=msg):
            df.groupby("a").resample("3T", base=0, offset=0).sum()

    with tm.assert_produces_warning(FutureWarning):
        df.resample("3T", base=0).sum()
    with tm.assert_produces_warning(FutureWarning):
        df.resample("3T", loffset="0s").sum()


@all_ts
@pytest.mark.parametrize("arg", ["mean", {"value": "mean"}, ["mean"]])
def test_resample_loffset_arg_type(frame, create_index, arg):
    # GH 13218, 15002
    df = frame
    expected_means = [df.values[i : i + 2].mean() for i in range(0, len(df.values), 2)]
    expected_index = create_index(df.index[0], periods=len(df.index) / 2, freq="2D")

    # loffset coerces PeriodIndex to DateTimeIndex
    if isinstance(expected_index, PeriodIndex):
        expected_index = expected_index.to_timestamp()

    expected_index += timedelta(hours=2)
    expected = DataFrame({"value": expected_means}, index=expected_index)

    with tm.assert_produces_warning(FutureWarning):
        result_agg = df.resample("2D", loffset="2H").agg(arg)

    if isinstance(arg, list):
        expected.columns = pd.MultiIndex.from_tuples([("value", "mean")])

    tm.assert_frame_equal(result_agg, expected)


@pytest.mark.parametrize(
    "loffset", [timedelta(minutes=1), "1min", Minute(1), np.timedelta64(1, "m")]
)
def test_resample_loffset(loffset):
    # GH 7687
    rng = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="min")
    s = Series(np.random.randn(14), index=rng)

    with tm.assert_produces_warning(FutureWarning):
        result = s.resample(
            "5min", closed="right", label="right", loffset=loffset
        ).mean()
    idx = date_range("1/1/2000", periods=4, freq="5min")
    expected = Series(
        [s[0], s[1:6].mean(), s[6:11].mean(), s[11:].mean()],
        index=idx + timedelta(minutes=1),
    )
    tm.assert_series_equal(result, expected)
    assert result.index.freq == Minute(5)

    # from daily
    dti = date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq="D")
    ser = Series(np.random.rand(len(dti)), dti)

    # to weekly
    result = ser.resample("w-sun").last()
    business_day_offset = BDay()
    with tm.assert_produces_warning(FutureWarning):
        expected = ser.resample("w-sun", loffset=-business_day_offset).last()
    assert result.index[0] - business_day_offset == expected.index[0]


def test_resample_loffset_upsample():
    # GH 20744
    rng = date_range("1/1/2000 00:00:00", "1/1/2000 00:13:00", freq="min")
    s = Series(np.random.randn(14), index=rng)

    with tm.assert_produces_warning(FutureWarning):
        result = s.resample(
            "5min", closed="right", label="right", loffset=timedelta(minutes=1)
        ).ffill()
    idx = date_range("1/1/2000", periods=4, freq="5min")
    expected = Series([s[0], s[5], s[10], s[-1]], index=idx + timedelta(minutes=1))

    tm.assert_series_equal(result, expected)


def test_resample_loffset_count():
    # GH 12725
    start_time = "1/1/2000 00:00:00"
    rng = date_range(start_time, periods=100, freq="S")
    ts = Series(np.random.randn(len(rng)), index=rng)

    with tm.assert_produces_warning(FutureWarning):
        result = ts.resample("10S", loffset="1s").count()

    expected_index = date_range(start_time, periods=10, freq="10S") + timedelta(
        seconds=1
    )
    expected = Series(10, index=expected_index)

    tm.assert_series_equal(result, expected)

    # Same issue should apply to .size() since it goes through
    #   same code path
    with tm.assert_produces_warning(FutureWarning):
        result = ts.resample("10S", loffset="1s").size()

    tm.assert_series_equal(result, expected)


def test_resample_base():
    rng = date_range("1/1/2000 00:00:00", "1/1/2000 02:00", freq="s")
    ts = Series(np.random.randn(len(rng)), index=rng)

    with tm.assert_produces_warning(FutureWarning):
        resampled = ts.resample("5min", base=2).mean()
    exp_rng = date_range("12/31/1999 23:57:00", "1/1/2000 01:57", freq="5min")
    tm.assert_index_equal(resampled.index, exp_rng)


def test_resample_float_base():
    # GH25161
    dt = pd.to_datetime(
        ["2018-11-26 16:17:43.51", "2018-11-26 16:17:44.51", "2018-11-26 16:17:45.51"]
    )
    s = Series(np.arange(3), index=dt)

    base = 17 + 43.51 / 60
    with tm.assert_produces_warning(FutureWarning):
        result = s.resample("3min", base=base).size()
    expected = Series(
        3, index=pd.DatetimeIndex(["2018-11-26 16:17:43.51"], freq="3min")
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kind", ["period", None, "timestamp"])
@pytest.mark.parametrize("agg_arg", ["mean", {"value": "mean"}, ["mean"]])
def test_loffset_returns_datetimeindex(frame, kind, agg_arg):
    # make sure passing loffset returns DatetimeIndex in all cases
    # basic method taken from Base.test_resample_loffset_arg_type()
    df = frame
    expected_means = [df.values[i : i + 2].mean() for i in range(0, len(df.values), 2)]
    expected_index = period_range(df.index[0], periods=len(df.index) / 2, freq="2D")

    # loffset coerces PeriodIndex to DateTimeIndex
    expected_index = expected_index.to_timestamp()
    expected_index += timedelta(hours=2)
    expected = DataFrame({"value": expected_means}, index=expected_index)

    with tm.assert_produces_warning(FutureWarning):
        result_agg = df.resample("2D", loffset="2H", kind=kind).agg(agg_arg)
    if isinstance(agg_arg, list):
        expected.columns = pd.MultiIndex.from_tuples([("value", "mean")])
    tm.assert_frame_equal(result_agg, expected)


@pytest.mark.parametrize(
    "start,end,start_freq,end_freq,base,offset",
    [
        ("19910905", "19910909 03:00", "H", "24H", 10, "10H"),
        ("19910905", "19910909 12:00", "H", "24H", 10, "10H"),
        ("19910905", "19910909 23:00", "H", "24H", 10, "10H"),
        ("19910905 10:00", "19910909", "H", "24H", 10, "10H"),
        ("19910905 10:00", "19910909 10:00", "H", "24H", 10, "10H"),
        ("19910905", "19910909 10:00", "H", "24H", 10, "10H"),
        ("19910905 12:00", "19910909", "H", "24H", 10, "10H"),
        ("19910905 12:00", "19910909 03:00", "H", "24H", 10, "10H"),
        ("19910905 12:00", "19910909 12:00", "H", "24H", 10, "10H"),
        ("19910905 12:00", "19910909 12:00", "H", "24H", 34, "34H"),
        ("19910905 12:00", "19910909 12:00", "H", "17H", 10, "10H"),
        ("19910905 12:00", "19910909 12:00", "H", "17H", 3, "3H"),
        ("19910905 12:00", "19910909 1:00", "H", "M", 3, "3H"),
        ("19910905", "19910913 06:00", "2H", "24H", 10, "10H"),
        ("19910905", "19910905 01:39", "Min", "5Min", 3, "3Min"),
        ("19910905", "19910905 03:18", "2Min", "5Min", 3, "3Min"),
    ],
)
def test_resample_with_non_zero_base(start, end, start_freq, end_freq, base, offset):
    # GH 23882
    s = Series(0, index=period_range(start, end, freq=start_freq))
    s = s + np.arange(len(s))
    with tm.assert_produces_warning(FutureWarning):
        result = s.resample(end_freq, base=base).mean()
    result = result.to_timestamp(end_freq)

    # test that the replacement argument 'offset' works
    result_offset = s.resample(end_freq, offset=offset).mean()
    result_offset = result_offset.to_timestamp(end_freq)
    tm.assert_series_equal(result, result_offset)

    # to_timestamp casts 24H -> D
    result = result.asfreq(end_freq) if end_freq == "24H" else result
    with tm.assert_produces_warning(FutureWarning):
        expected = s.to_timestamp().resample(end_freq, base=base).mean()
    if end_freq == "M":
        # TODO: is non-tick the relevant characteristic? (GH 33815)
        expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)


def test_resample_base_with_timedeltaindex():
    # GH 10530
    rng = timedelta_range(start="0s", periods=25, freq="s")
    ts = Series(np.random.randn(len(rng)), index=rng)

    with tm.assert_produces_warning(FutureWarning):
        with_base = ts.resample("2s", base=5).mean()
    without_base = ts.resample("2s").mean()

    exp_without_base = timedelta_range(start="0s", end="25s", freq="2s")
    exp_with_base = timedelta_range(start="5s", end="29s", freq="2s")

    tm.assert_index_equal(without_base.index, exp_without_base)
    tm.assert_index_equal(with_base.index, exp_with_base)


def test_interpolate_posargs_deprecation():
    # GH 41485
    idx = pd.to_datetime(["1992-08-27 07:46:48", "1992-08-27 07:46:59"])
    s = Series([1, 4], index=idx)

    msg = (
        r"In a future version of pandas all arguments of Resampler\.interpolate "
        r"except for the argument 'method' will be keyword-only"
    )

    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.resample("3s").interpolate("linear", 0)

    idx = pd.to_datetime(
        [
            "1992-08-27 07:46:48",
            "1992-08-27 07:46:51",
            "1992-08-27 07:46:54",
            "1992-08-27 07:46:57",
        ]
    )
    expected = Series([1.0, 1.0, 1.0, 1.0], index=idx)

    expected.index._data.freq = "3s"
    tm.assert_series_equal(result, expected)
