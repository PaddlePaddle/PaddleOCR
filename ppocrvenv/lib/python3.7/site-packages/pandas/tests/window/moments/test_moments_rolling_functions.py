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


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],
        [np.nansum, "sum", {}],
        pytest.param(
            lambda x: np.isfinite(x).astype(float).sum(),
            "count",
            {},
            marks=pytest.mark.filterwarnings("ignore:min_periods:FutureWarning"),
        ),
        [np.median, "median", {}],
        [np.min, "min", {}],
        [np.max, "max", {}],
        [lambda x: np.std(x, ddof=1), "std", {}],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],
        [lambda x: np.var(x, ddof=1), "var", {}],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],
    ],
)
def test_series(series, compare_func, roll_func, kwargs):
    result = getattr(series.rolling(50), roll_func)(**kwargs)
    assert isinstance(result, Series)
    tm.assert_almost_equal(result.iloc[-1], compare_func(series[-50:]))


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],
        [np.nansum, "sum", {}],
        pytest.param(
            lambda x: np.isfinite(x).astype(float).sum(),
            "count",
            {},
            marks=pytest.mark.filterwarnings("ignore:min_periods:FutureWarning"),
        ),
        [np.median, "median", {}],
        [np.min, "min", {}],
        [np.max, "max", {}],
        [lambda x: np.std(x, ddof=1), "std", {}],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],
        [lambda x: np.var(x, ddof=1), "var", {}],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],
    ],
)
def test_frame(raw, frame, compare_func, roll_func, kwargs):
    result = getattr(frame.rolling(50), roll_func)(**kwargs)
    assert isinstance(result, DataFrame)
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(compare_func, axis=0, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs, minp",
    [
        [np.mean, "mean", {}, 10],
        [np.nansum, "sum", {}, 10],
        [lambda x: np.isfinite(x).astype(float).sum(), "count", {}, 0],
        [np.median, "median", {}, 10],
        [np.min, "min", {}, 10],
        [np.max, "max", {}, 10],
        [lambda x: np.std(x, ddof=1), "std", {}, 10],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}, 10],
        [lambda x: np.var(x, ddof=1), "var", {}, 10],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}, 10],
    ],
)
def test_time_rule_series(series, compare_func, roll_func, kwargs, minp):
    win = 25
    ser = series[::2].resample("B").mean()
    series_result = getattr(ser.rolling(window=win, min_periods=minp), roll_func)(
        **kwargs
    )
    last_date = series_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_series = series[::2].truncate(prev_date, last_date)
    tm.assert_almost_equal(series_result[-1], compare_func(trunc_series))


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs, minp",
    [
        [np.mean, "mean", {}, 10],
        [np.nansum, "sum", {}, 10],
        [lambda x: np.isfinite(x).astype(float).sum(), "count", {}, 0],
        [np.median, "median", {}, 10],
        [np.min, "min", {}, 10],
        [np.max, "max", {}, 10],
        [lambda x: np.std(x, ddof=1), "std", {}, 10],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}, 10],
        [lambda x: np.var(x, ddof=1), "var", {}, 10],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}, 10],
    ],
)
def test_time_rule_frame(raw, frame, compare_func, roll_func, kwargs, minp):
    win = 25
    frm = frame[::2].resample("B").mean()
    frame_result = getattr(frm.rolling(window=win, min_periods=minp), roll_func)(
        **kwargs
    )
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()

    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(compare_func, raw=raw),
        check_names=False,
    )


@pytest.mark.parametrize(
    "compare_func, roll_func, kwargs",
    [
        [np.mean, "mean", {}],
        [np.nansum, "sum", {}],
        [np.median, "median", {}],
        [np.min, "min", {}],
        [np.max, "max", {}],
        [lambda x: np.std(x, ddof=1), "std", {}],
        [lambda x: np.std(x, ddof=0), "std", {"ddof": 0}],
        [lambda x: np.var(x, ddof=1), "var", {}],
        [lambda x: np.var(x, ddof=0), "var", {"ddof": 0}],
    ],
)
def test_nans(compare_func, roll_func, kwargs):
    obj = Series(np.random.randn(50))
    obj[:10] = np.NaN
    obj[-10:] = np.NaN

    result = getattr(obj.rolling(50, min_periods=30), roll_func)(**kwargs)
    tm.assert_almost_equal(result.iloc[-1], compare_func(obj[10:-10]))

    # min_periods is working correctly
    result = getattr(obj.rolling(20, min_periods=15), roll_func)(**kwargs)
    assert isna(result.iloc[23])
    assert not isna(result.iloc[24])

    assert not isna(result.iloc[-6])
    assert isna(result.iloc[-5])

    obj2 = Series(np.random.randn(20))
    result = getattr(obj2.rolling(10, min_periods=5), roll_func)(**kwargs)
    assert isna(result.iloc[3])
    assert notna(result.iloc[4])

    if roll_func != "sum":
        result0 = getattr(obj.rolling(20, min_periods=0), roll_func)(**kwargs)
        result1 = getattr(obj.rolling(20, min_periods=1), roll_func)(**kwargs)
        tm.assert_almost_equal(result0, result1)


def test_nans_count():
    obj = Series(np.random.randn(50))
    obj[:10] = np.NaN
    obj[-10:] = np.NaN
    result = obj.rolling(50, min_periods=30).count()
    tm.assert_almost_equal(
        result.iloc[-1], np.isfinite(obj[10:-10]).astype(float).sum()
    )


@pytest.mark.parametrize(
    "roll_func, kwargs",
    [
        ["mean", {}],
        ["sum", {}],
        ["median", {}],
        ["min", {}],
        ["max", {}],
        ["std", {}],
        ["std", {"ddof": 0}],
        ["var", {}],
        ["var", {"ddof": 0}],
    ],
)
@pytest.mark.parametrize("minp", [0, 99, 100])
def test_min_periods(series, minp, roll_func, kwargs):
    result = getattr(series.rolling(len(series) + 1, min_periods=minp), roll_func)(
        **kwargs
    )
    expected = getattr(series.rolling(len(series), min_periods=minp), roll_func)(
        **kwargs
    )
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


def test_min_periods_count(series):
    result = series.rolling(len(series) + 1, min_periods=0).count()
    expected = series.rolling(len(series), min_periods=0).count()
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))

    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


@pytest.mark.parametrize(
    "roll_func, kwargs, minp",
    [
        ["mean", {}, 15],
        ["sum", {}, 15],
        ["count", {}, 0],
        ["median", {}, 15],
        ["min", {}, 15],
        ["max", {}, 15],
        ["std", {}, 15],
        ["std", {"ddof": 0}, 15],
        ["var", {}, 15],
        ["var", {"ddof": 0}, 15],
    ],
)
def test_center(roll_func, kwargs, minp):
    obj = Series(np.random.randn(50))
    obj[:10] = np.NaN
    obj[-10:] = np.NaN

    result = getattr(obj.rolling(20, min_periods=minp, center=True), roll_func)(
        **kwargs
    )
    expected = getattr(
        concat([obj, Series([np.NaN] * 9)]).rolling(20, min_periods=minp), roll_func
    )(**kwargs)[9:].reset_index(drop=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "roll_func, kwargs, minp, fill_value",
    [
        ["mean", {}, 10, None],
        ["sum", {}, 10, None],
        ["count", {}, 0, 0],
        ["median", {}, 10, None],
        ["min", {}, 10, None],
        ["max", {}, 10, None],
        ["std", {}, 10, None],
        ["std", {"ddof": 0}, 10, None],
        ["var", {}, 10, None],
        ["var", {"ddof": 0}, 10, None],
    ],
)
def test_center_reindex_series(series, roll_func, kwargs, minp, fill_value):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    series_xp = (
        getattr(
            series.reindex(list(series.index) + s).rolling(window=25, min_periods=minp),
            roll_func,
        )(**kwargs)
        .shift(-12)
        .reindex(series.index)
    )
    series_rs = getattr(
        series.rolling(window=25, min_periods=minp, center=True), roll_func
    )(**kwargs)
    if fill_value is not None:
        series_xp = series_xp.fillna(fill_value)
    tm.assert_series_equal(series_xp, series_rs)


@pytest.mark.parametrize(
    "roll_func, kwargs, minp, fill_value",
    [
        ["mean", {}, 10, None],
        ["sum", {}, 10, None],
        ["count", {}, 0, 0],
        ["median", {}, 10, None],
        ["min", {}, 10, None],
        ["max", {}, 10, None],
        ["std", {}, 10, None],
        ["std", {"ddof": 0}, 10, None],
        ["var", {}, 10, None],
        ["var", {"ddof": 0}, 10, None],
    ],
)
def test_center_reindex_frame(frame, roll_func, kwargs, minp, fill_value):
    # shifter index
    s = [f"x{x:d}" for x in range(12)]

    frame_xp = (
        getattr(
            frame.reindex(list(frame.index) + s).rolling(window=25, min_periods=minp),
            roll_func,
        )(**kwargs)
        .shift(-12)
        .reindex(frame.index)
    )
    frame_rs = getattr(
        frame.rolling(window=25, min_periods=minp, center=True), roll_func
    )(**kwargs)
    if fill_value is not None:
        frame_xp = frame_xp.fillna(fill_value)
    tm.assert_frame_equal(frame_xp, frame_rs)
