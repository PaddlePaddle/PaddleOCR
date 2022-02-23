import warnings

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    concat,
    isna,
    notna,
)
import pandas._testing as tm

import pandas.tseries.offsets as offsets


def f(x):
    # suppress warnings about empty slices, as we are deliberately testing
    # with a 0-length Series
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*(empty slice|0 for slice).*",
            category=RuntimeWarning,
        )
        return x[np.isfinite(x)].mean()


def test_series(raw, series):
    result = series.rolling(50).apply(f, raw=raw)
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.iloc[-1], np.mean(series[-50:]))


def test_frame(raw, frame):
    result = frame.rolling(50).apply(f, raw=raw)
    assert isinstance(result, DataFrame)
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(np.mean, axis=0, raw=raw),
        check_names=False,
    )


def test_time_rule_series(raw, series):
    win = 25
    minp = 10
    ser = series[::2].resample("B").mean()
    series_result = ser.rolling(window=win, min_periods=minp).apply(f, raw=raw)
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result[-1], np.mean(trunc_series))


def test_time_rule_frame(raw, frame):
    win = 25
    minp = 10
    frm = frame[::2].resample("B").mean()
    frame_result = frm.rolling(window=win, min_periods=minp).apply(f, raw=raw)
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(np.mean, raw=raw),
        check_names=False,
    )


def test_nans(raw):
    obj = Series(np.random.randn(50))
    obj[:10] = np.NaN
    obj[-10:] = np.NaN

    result = obj.rolling(50, min_periods=30).apply(f, raw=raw)
    tm.assert_almost_equal(result.iloc[-1], np.mean(obj[10:-10]))

    # min_periods is working correctly
    result = obj.rolling(20, min_periods=15).apply(f, raw=raw)
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.randn(20))
    result = obj2.rolling(10, min_periods=5).apply(f, raw=raw)
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    result0 = obj.rolling(20, min_periods=0).apply(f, raw=raw)
    result1 = obj.rolling(20, min_periods=1).apply(f, raw=raw)
    tm.assert_almost_equal(result0, result1)


@pytest.mark.parametrize("minp", [0, 99, 100])
def test_min_periods(raw, series, minp):
    result = series.rolling(len(series) + 1, min_periods=minp).apply(f, raw=raw)
    expected = series.rolling(len(series), min_periods=minp).apply(f, raw=raw)
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


def test_center(raw):
    obj = Series(np.random.randn(50))
    obj[:10] = np.NaN
    obj[-10:] = np.NaN

    result = obj.rolling(20, min_periods=15, center=True).apply(f, raw=raw)
    expected = (
        concat([obj, Series([np.NaN] * 9)])
        .rolling(20, min_periods=15)
        .apply(f, raw=raw)[9:]
        .reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


def test_center_reindex_series(raw, series):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]
    minp = 10

    series_xp = (
        series.reindex(list(series.index) + s)
        .rolling(window=25, min_periods=minp)
        .apply(f, raw=raw)
        .shift(-12)
        .reindex(series.index)
    )
    series_rs = series.rolling(window=25, min_periods=minp, center=True).apply(
        f, raw=raw
    )
    tm.assert_series_equal(series_xp, series_rs)


def test_center_reindex_frame(raw, frame):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]
    minp = 10

    frame_xp = (
        frame.reindex(list(frame.index) + s)
        .rolling(window=25, min_periods=minp)
        .apply(f, raw=raw)
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = frame.rolling(window=25, min_periods=minp, center=True).apply(f, raw=raw)
    tm.assert_frame_equal(frame_xp, frame_rs)
