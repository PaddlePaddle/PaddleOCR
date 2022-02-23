from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.compat import (
    is_platform_arm,
    is_platform_mac,
)
from pandas.errors import UnsupportedFunctionCall

from pandas import (
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.window import Rolling


def test_doc_string():

    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df
    df.rolling(2).sum()
    df.rolling(2, min_periods=1).sum()


def test_constructor(frame_or_series):
    # GH 12669

    c = frame_or_series(range(5)).rolling

    # valid
    c(0)
    c(window=2)
    c(window=2, min_periods=1)
    c(window=2, min_periods=1, center=True)
    c(window=2, min_periods=1, center=False)

    # GH 13383

    msg = "window must be an integer 0 or greater"

    with pytest.raises(ValueError, match=msg):
        c(-1)


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_invalid_constructor(frame_or_series, w):
    # not valid

    c = frame_or_series(range(5)).rolling

    msg = (
        "window must be an integer|"
        "passed window foo is not compatible with a datetimelike index"
    )
    with pytest.raises(ValueError, match=msg):
        c(window=w)

    msg = "min_periods must be an integer"
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=w)

    msg = "center must be a boolean"
    with pytest.raises(ValueError, match=msg):
        c(window=2, min_periods=1, center=w)


@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window):
    # GH 15440
    n = 10
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    expected_data = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))

    result = df.rolling(window=window).sum()
    expected = DataFrame(
        {"value": expected_data},
        index=date_range("2015-12-24", periods=n, freq="D"),
    )
    tm.assert_frame_equal(result, expected)
    expected = df.rolling("3D").sum()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("window", [timedelta(days=3), Timedelta(days=3), "3D"])
def test_constructor_timedelta_window_and_minperiods(window, raw):
    # GH 15305
    n = 10
    df = DataFrame(
        {"value": np.arange(n)},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    expected = DataFrame(
        {"value": np.append([np.NaN, 1.0], np.arange(3.0, 27.0, 3))},
        index=date_range("2017-08-08", periods=n, freq="D"),
    )
    result_roll_sum = df.rolling(window=window, min_periods=2).sum()
    result_roll_generic = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)


@pytest.mark.parametrize("method", ["std", "mean", "sum", "max", "min", "var"])
def test_numpy_compat(method):
    # see gh-12811
    r = Rolling(Series([2, 4, 6]), window=2)

    msg = "numpy operations are not valid with window objects"

    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(r, method)(1, 2, 3)
    with pytest.raises(UnsupportedFunctionCall, match=msg):
        getattr(r, method)(dtype=np.float64)


def test_closed_fixed(closed, arithmetic_win_operators):
    # GH 34315
    func_name = arithmetic_win_operators
    df_fixed = DataFrame({"A": [0, 1, 2, 3, 4]})
    df_time = DataFrame({"A": [0, 1, 2, 3, 4]}, index=date_range("2020", periods=5))

    result = getattr(
        df_fixed.rolling(2, closed=closed, min_periods=1),
        func_name,
    )()
    expected = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1),
        func_name,
    )().reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "closed, window_selections",
    [
        (
            "both",
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ],
        ),
        (
            "left",
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
        ),
        (
            "right",
            [
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
                [False, False, False, False, True],
            ],
        ),
        (
            "neither",
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
        ),
    ],
)
def test_datetimelike_centered_selections(
    closed, window_selections, arithmetic_win_operators
):
    # GH 34315
    func_name = arithmetic_win_operators
    df_time = DataFrame(
        {"A": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range("2020", periods=5)
    )

    expected = DataFrame(
        {"A": [getattr(df_time["A"].iloc[s], func_name)() for s in window_selections]},
        index=date_range("2020", periods=5),
    )

    if func_name == "sem":
        kwargs = {"ddof": 0}
    else:
        kwargs = {}

    result = getattr(
        df_time.rolling("2D", closed=closed, min_periods=1, center=True),
        func_name,
    )(**kwargs)

    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "window,closed,expected",
    [
        ("3s", "right", [3.0, 3.0, 3.0]),
        ("3s", "both", [3.0, 3.0, 3.0]),
        ("3s", "left", [3.0, 3.0, 3.0]),
        ("3s", "neither", [3.0, 3.0, 3.0]),
        ("2s", "right", [3.0, 2.0, 2.0]),
        ("2s", "both", [3.0, 3.0, 3.0]),
        ("2s", "left", [1.0, 3.0, 3.0]),
        ("2s", "neither", [1.0, 2.0, 2.0]),
    ],
)
def test_datetimelike_centered_offset_covers_all(
    window, closed, expected, frame_or_series
):
    # GH 42753

    index = [
        Timestamp("20130101 09:00:01"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:02"),
    ]
    df = frame_or_series([1, 1, 1], index=index)

    result = df.rolling(window, closed=closed, center=True).sum()
    expected = frame_or_series(expected, index=index)
    tm.assert_equal(result, expected)


def test_even_number_window_alignment():
    # see discussion in GH 38780
    s = Series(range(3), index=date_range(start="2020-01-01", freq="D", periods=3))

    # behavior of index- and datetime-based windows differs here!
    # s.rolling(window=2, min_periods=1, center=True).mean()

    result = s.rolling(window="2D", min_periods=1, center=True).mean()

    expected = Series([0.5, 1.5, 2], index=s.index)

    tm.assert_series_equal(result, expected)


def test_closed_fixed_binary_col(center):
    # GH 34315
    data = [0, 1, 1, 0, 0, 1, 0, 1]
    df = DataFrame(
        {"binary_col": data},
        index=date_range(start="2020-01-01", freq="min", periods=len(data)),
    )

    if center:
        expected_data = [2 / 3, 0.5, 0.4, 0.5, 0.428571, 0.5, 0.571429, 0.5]
    else:
        expected_data = [np.nan, 0, 0.5, 2 / 3, 0.5, 0.4, 0.5, 0.428571]

    expected = DataFrame(
        expected_data,
        columns=["binary_col"],
        index=date_range(start="2020-01-01", freq="min", periods=len(expected_data)),
    )

    rolling = df.rolling(window=len(df), closed="left", min_periods=1, center=center)
    result = rolling.mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("closed", ["neither", "left"])
def test_closed_empty(closed, arithmetic_win_operators):
    # GH 26005
    func_name = arithmetic_win_operators
    ser = Series(data=np.arange(5), index=date_range("2000", periods=5, freq="2D"))
    roll = ser.rolling("1D", closed=closed)

    result = getattr(roll, func_name)()
    expected = Series([np.nan] * 5, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry(func):
    # GH24718
    ser = Series(data=[2], index=date_range("2000", periods=1))
    result = getattr(ser.rolling("10D", closed="left"), func)()
    tm.assert_series_equal(result, Series([np.nan], index=ser.index))


@pytest.mark.parametrize("func", ["min", "max"])
def test_closed_one_entry_groupby(func):
    # GH24718
    ser = DataFrame(
        data={"A": [1, 1, 2], "B": [3, 2, 1]},
        index=date_range("2000", periods=3),
    )
    result = getattr(
        ser.groupby("A", sort=False)["B"].rolling("10D", closed="left"), func
    )()
    exp_idx = MultiIndex.from_arrays(arrays=[[1, 1, 2], ser.index], names=("A", None))
    expected = Series(data=[np.nan, 3, np.nan], index=exp_idx, name="B")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("input_dtype", ["int", "float"])
@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "both", [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("min", "neither", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("min", "left", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", "right", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "both", [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("max", "neither", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("max", "left", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_closed_min_max_datetime(input_dtype, func, closed, expected):
    # see gh-21704
    ser = Series(
        data=np.arange(10).astype(input_dtype),
        index=date_range("2000", periods=10),
    )

    result = getattr(ser.rolling("3D", closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)


def test_closed_uneven():
    # see gh-21704
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))

    # uneven
    ser = ser.drop(index=ser.index[[1, 5]])
    result = ser.rolling("3D", closed="left").min()
    expected = Series([np.nan, 0, 0, 2, 3, 4, 6, 6], index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func,closed,expected",
    [
        ("min", "right", [np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "both", [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("min", "neither", [np.nan, np.nan, 0, 1, 2, 3, 4, 5, np.nan, np.nan]),
        ("min", "left", [np.nan, np.nan, 0, 0, 1, 2, 3, 4, 5, np.nan]),
        ("max", "right", [np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan, np.nan]),
        ("max", "both", [np.nan, 1, 2, 3, 4, 5, 6, 6, 6, np.nan]),
        ("max", "neither", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan]),
        ("max", "left", [np.nan, np.nan, 1, 2, 3, 4, 5, 6, 6, np.nan]),
    ],
)
def test_closed_min_max_minp(func, closed, expected):
    # see gh-21704
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))
    ser[ser.index[-3:]] = np.nan
    result = getattr(ser.rolling("3D", min_periods=2, closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "closed,expected",
    [
        ("right", [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]),
        ("both", [0, 0.5, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("neither", [np.nan, 0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
        ("left", [np.nan, 0, 0.5, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_closed_median_quantile(closed, expected):
    # GH 26005
    ser = Series(data=np.arange(10), index=date_range("2000", periods=10))
    roll = ser.rolling("3D", closed=closed)
    expected = Series(expected, index=ser.index)

    result = roll.median()
    tm.assert_series_equal(result, expected)

    result = roll.quantile(0.5)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("roller", ["1s", 1])
def tests_empty_df_rolling(roller):
    # GH 15819 Verifies that datetime and integer rolling windows can be
    # applied to empty DataFrames
    expected = DataFrame()
    result = DataFrame().rolling(roller).sum()
    tm.assert_frame_equal(result, expected)

    # Verifies that datetime and integer rolling windows can be applied to
    # empty DataFrames with datetime index
    expected = DataFrame(index=DatetimeIndex([]))
    result = DataFrame(index=DatetimeIndex([])).rolling(roller).sum()
    tm.assert_frame_equal(result, expected)


def test_empty_window_median_quantile():
    # GH 26005
    expected = Series([np.nan, np.nan, np.nan])
    roll = Series(np.arange(3)).rolling(0)

    result = roll.median()
    tm.assert_series_equal(result, expected)

    result = roll.quantile(0.1)
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero():
    # https://github.com/pandas-dev/pandas/pull/18921
    # minp=0
    x = Series([np.nan])
    result = x.rolling(1, min_periods=0).sum()
    expected = Series([0.0])
    tm.assert_series_equal(result, expected)

    # minp=1
    result = x.rolling(1, min_periods=1).sum()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_missing_minp_zero_variable():
    # https://github.com/pandas-dev/pandas/pull/18921
    x = Series(
        [np.nan] * 4,
        index=DatetimeIndex(["2017-01-01", "2017-01-04", "2017-01-06", "2017-01-07"]),
    )
    result = x.rolling(Timedelta("2d"), min_periods=0).sum()
    expected = Series(0.0, index=x.index)
    tm.assert_series_equal(result, expected)


def test_multi_index_names():

    # GH 16789, 16825
    cols = MultiIndex.from_product([["A", "B"], ["C", "D", "E"]], names=["1", "2"])
    df = DataFrame(np.ones((10, 6)), columns=cols)
    result = df.rolling(3).cov()

    tm.assert_index_equal(result.columns, df.columns)
    assert result.index.names == [None, "1", "2"]


def test_rolling_axis_sum(axis_frame):
    # see gh-23372.
    df = DataFrame(np.ones((10, 20)))
    axis = df._get_axis_number(axis_frame)

    if axis == 0:
        expected = DataFrame({i: [np.nan] * 2 + [3.0] * 8 for i in range(20)})
    else:
        # axis == 1
        expected = DataFrame([[np.nan] * 2 + [3.0] * 18] * 10)

    result = df.rolling(3, axis=axis_frame).sum()
    tm.assert_frame_equal(result, expected)


def test_rolling_axis_count(axis_frame):
    # see gh-26055
    df = DataFrame({"x": range(3), "y": range(3)})

    axis = df._get_axis_number(axis_frame)

    if axis in [0, "index"]:
        expected = DataFrame({"x": [1.0, 2.0, 2.0], "y": [1.0, 2.0, 2.0]})
    else:
        expected = DataFrame({"x": [1.0, 1.0, 1.0], "y": [2.0, 2.0, 2.0]})

    result = df.rolling(2, axis=axis_frame, min_periods=0).count()
    tm.assert_frame_equal(result, expected)


def test_readonly_array():
    # GH-27766
    arr = np.array([1, 3, np.nan, 3, 5])
    arr.setflags(write=False)
    result = Series(arr).rolling(2).mean()
    expected = Series([np.nan, 2, np.nan, np.nan, 4])
    tm.assert_series_equal(result, expected)


def test_rolling_datetime(axis_frame, tz_naive_fixture):
    # GH-28192
    tz = tz_naive_fixture
    df = DataFrame(
        {i: [1] * 2 for i in date_range("2019-8-01", "2019-08-03", freq="D", tz=tz)}
    )
    if axis_frame in [0, "index"]:
        result = df.T.rolling("2D", axis=axis_frame).sum().T
    else:
        result = df.rolling("2D", axis=axis_frame).sum()
    expected = DataFrame(
        {
            **{
                i: [1.0] * 2
                for i in date_range("2019-8-01", periods=1, freq="D", tz=tz)
            },
            **{
                i: [2.0] * 2
                for i in date_range("2019-8-02", "2019-8-03", freq="D", tz=tz)
            },
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "center, expected_data",
    [
        (
            True,
            (
                [88.0] * 7
                + [97.0] * 9
                + [98.0]
                + [99.0] * 21
                + [95.0] * 16
                + [93.0] * 5
                + [89.0] * 5
                + [96.0] * 21
                + [94.0] * 14
                + [90.0] * 13
                + [88.0] * 2
                + [90.0] * 9
                + [96.0] * 21
                + [95.0] * 6
                + [91.0]
                + [87.0] * 6
                + [92.0] * 21
                + [83.0] * 2
                + [86.0] * 10
                + [87.0] * 5
                + [98.0] * 21
                + [97.0] * 14
                + [93.0] * 7
                + [87.0] * 4
                + [86.0] * 4
                + [95.0] * 21
                + [85.0] * 14
                + [83.0] * 2
                + [76.0] * 5
                + [81.0] * 2
                + [98.0] * 21
                + [95.0] * 14
                + [91.0] * 7
                + [86.0]
                + [93.0] * 3
                + [95.0] * 29
                + [77.0] * 2
            ),
        ),
        (
            False,
            (
                [np.nan] * 2
                + [88.0] * 16
                + [97.0] * 9
                + [98.0]
                + [99.0] * 21
                + [95.0] * 16
                + [93.0] * 5
                + [89.0] * 5
                + [96.0] * 21
                + [94.0] * 14
                + [90.0] * 13
                + [88.0] * 2
                + [90.0] * 9
                + [96.0] * 21
                + [95.0] * 6
                + [91.0]
                + [87.0] * 6
                + [92.0] * 21
                + [83.0] * 2
                + [86.0] * 10
                + [87.0] * 5
                + [98.0] * 21
                + [97.0] * 14
                + [93.0] * 7
                + [87.0] * 4
                + [86.0] * 4
                + [95.0] * 21
                + [85.0] * 14
                + [83.0] * 2
                + [76.0] * 5
                + [81.0] * 2
                + [98.0] * 21
                + [95.0] * 14
                + [91.0] * 7
                + [86.0]
                + [93.0] * 3
                + [95.0] * 20
            ),
        ),
    ],
)
def test_rolling_window_as_string(center, expected_data):
    # see gh-22590
    date_today = datetime.now()
    days = date_range(date_today, date_today + timedelta(365), freq="D")

    npr = np.random.RandomState(seed=421)

    data = npr.randint(1, high=100, size=len(days))
    df = DataFrame({"DateCol": days, "metric": data})

    df.set_index("DateCol", inplace=True)
    result = df.rolling(window="21D", min_periods=2, closed="left", center=center)[
        "metric"
    ].agg("max")

    index = days.rename("DateCol")
    index = index._with_freq(None)
    expected = Series(expected_data, index=index, name="metric")
    tm.assert_series_equal(result, expected)


def test_min_periods1():
    # GH#6795
    df = DataFrame([0, 1, 2, 1, 0], columns=["a"])
    result = df["a"].rolling(3, center=True, min_periods=1).max()
    expected = Series([1.0, 2.0, 2.0, 2.0, 1.0], name="a")
    tm.assert_series_equal(result, expected)


def test_rolling_count_with_min_periods(frame_or_series):
    # GH 26996
    result = frame_or_series(range(5)).rolling(3, min_periods=3).count()
    expected = frame_or_series([np.nan, np.nan, 3.0, 3.0, 3.0])
    tm.assert_equal(result, expected)


def test_rolling_count_default_min_periods_with_null_values(frame_or_series):
    # GH 26996
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 3.0]

    # GH 31302
    with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
        result = frame_or_series(values).rolling(3).count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "df,expected,window,min_periods",
    [
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            3,
            None,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            2,
            1,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            2,
            2,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            1,
            1,
        ),
        (
            DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            1,
            0,
        ),
        (DataFrame({"A": [1], "B": [4]}), [], 2, None),
        (DataFrame({"A": [1], "B": [4]}), [], 2, 1),
        (DataFrame(), [({}, [])], 2, None),
        (
            DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}),
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            3,
            2,
        ),
    ],
)
def test_iter_rolling_dataframe(df, expected, window, min_periods):
    # GH 11704
    expected = [DataFrame(values, index=index) for (values, index) in expected]

    for (expected, actual) in zip(
        expected, df.rolling(window, min_periods=min_periods)
    ):
        tm.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "expected,window",
    [
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [2, 3], "B": [5, 6]}, [1, 2]),
            ],
            "2D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            "3D",
        ),
        (
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [2], "B": [5]}, [1]),
                ({"A": [3], "B": [6]}, [2]),
            ],
            "1D",
        ),
    ],
)
def test_iter_rolling_on_dataframe(expected, window):
    # GH 11704, 40373
    df = DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [4, 5, 6, 7, 8],
            "C": date_range(start="2016-01-01", periods=5, freq="D"),
        }
    )

    expected = [
        DataFrame(values, index=df.loc[index, "C"]) for (values, index) in expected
    ]
    for (expected, actual) in zip(expected, df.rolling(window, on="C")):
        tm.assert_frame_equal(actual, expected)


def test_iter_rolling_on_dataframe_unordered():
    # GH 43386
    df = DataFrame({"a": ["x", "y", "x"], "b": [0, 1, 2]})
    results = list(df.groupby("a").rolling(2))
    expecteds = [df.iloc[idx, [1]] for idx in [[0], [0, 2], [1]]]
    for result, expected in zip(results, expecteds):
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ser,expected,window, min_periods",
    [
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            None,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])],
            3,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            1,
        ),
        (
            Series([1, 2, 3]),
            [([1], [0]), ([1, 2], [0, 1]), ([2, 3], [1, 2])],
            2,
            2,
        ),
        (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 0),
        (Series([1, 2, 3]), [([1], [0]), ([2], [1]), ([3], [2])], 1, 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2, 0),
        (Series([], dtype="int64"), [], 2, 1),
    ],
)
def test_iter_rolling_series(ser, expected, window, min_periods):
    # GH 11704
    expected = [Series(values, index=index) for (values, index) in expected]

    for (expected, actual) in zip(
        expected, ser.rolling(window, min_periods=min_periods)
    ):
        tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "expected,expected_index,window",
    [
        (
            [[0], [1], [2], [3], [4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-02", periods=1, freq="D"),
                date_range("2020-01-03", periods=1, freq="D"),
                date_range("2020-01-04", periods=1, freq="D"),
                date_range("2020-01-05", periods=1, freq="D"),
            ],
            "1D",
        ),
        (
            [[0], [0, 1], [1, 2], [2, 3], [3, 4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-02", periods=2, freq="D"),
                date_range("2020-01-03", periods=2, freq="D"),
                date_range("2020-01-04", periods=2, freq="D"),
            ],
            "2D",
        ),
        (
            [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [
                date_range("2020-01-01", periods=1, freq="D"),
                date_range("2020-01-01", periods=2, freq="D"),
                date_range("2020-01-01", periods=3, freq="D"),
                date_range("2020-01-02", periods=3, freq="D"),
                date_range("2020-01-03", periods=3, freq="D"),
            ],
            "3D",
        ),
    ],
)
def test_iter_rolling_datetime(expected, expected_index, window):
    # GH 11704
    ser = Series(range(5), index=date_range(start="2020-01-01", periods=5, freq="D"))

    expected = [
        Series(values, index=idx) for (values, idx) in zip(expected, expected_index)
    ]

    for (expected, actual) in zip(expected, ser.rolling(window)):
        tm.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "grouping,_index",
    [
        (
            {"level": 0},
            MultiIndex.from_tuples(
                [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)], names=[None, None]
            ),
        ),
        (
            {"by": "X"},
            MultiIndex.from_tuples(
                [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1)], names=["X", None]
            ),
        ),
    ],
)
def test_rolling_positional_argument(grouping, _index, raw):
    # GH 34605

    def scaled_sum(*args):
        if len(args) < 2:
            raise ValueError("The function needs two arguments")
        array, scale = args
        return array.sum() / scale

    df = DataFrame(data={"X": range(5)}, index=[0, 0, 1, 1, 1])

    expected = DataFrame(data={"X": [0.0, 0.5, 1.0, 1.5, 2.0]}, index=_index)
    # GH 40341
    if "by" in grouping:
        expected = expected.drop(columns="X", errors="ignore")
    result = df.groupby(**grouping).rolling(1).apply(scaled_sum, raw=raw, args=(2,))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("add", [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add):
    # GH: 36031 implementing kahan summation
    df = DataFrame(
        {"A": [3002399751580331.0 + add, -0.0, -0.0]},
        index=[
            Timestamp("19700101 09:00:00"),
            Timestamp("19700101 09:00:03"),
            Timestamp("19700101 09:00:06"),
        ],
    )
    result = (
        df.resample("1s").ffill().rolling("3s", closed="left", min_periods=3).mean()
    )
    dates = date_range("19700101 09:00:00", periods=7, freq="S")
    expected = DataFrame(
        {
            "A": [
                np.nan,
                np.nan,
                np.nan,
                3002399751580330.5,
                2001599834386887.25,
                1000799917193443.625,
                0.0,
            ]
        },
        index=dates,
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_numerical_accuracy_kahan_sum():
    # GH: 13254
    df = DataFrame([2.186, -1.647, 0.0, 0.0, 0.0, 0.0], columns=["x"])
    result = df["x"].rolling(3).sum()
    expected = Series([np.nan, np.nan, 0.539, -1.647, 0.0, 0.0], name="x")
    tm.assert_series_equal(result, expected)


def test_rolling_numerical_accuracy_jump():
    # GH: 32761
    index = date_range(start="2020-01-01", end="2020-01-02", freq="60s").append(
        DatetimeIndex(["2020-01-03"])
    )
    data = np.random.rand(len(index))

    df = DataFrame({"data": data}, index=index)
    result = df.rolling("60s").mean()
    tm.assert_frame_equal(result, df[["data"]])


def test_rolling_numerical_accuracy_small_values():
    # GH: 10319
    s = Series(
        data=[0.00012456, 0.0003, -0.0, -0.0],
        index=date_range("1999-02-03", "1999-02-06"),
    )
    result = s.rolling(1).mean()
    tm.assert_series_equal(result, s)


def test_rolling_numerical_too_large_numbers():
    # GH: 11645
    dates = date_range("2015-01-01", periods=10, freq="D")
    ds = Series(data=range(10), index=dates, dtype=np.float64)
    ds[2] = -9e33
    result = ds.rolling(5).mean()
    expected = Series(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.8e33,
            -1.8e33,
            -1.8e33,
            5.0,
            6.0,
            7.0,
        ],
        index=dates,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "value"),
    [("sum", 2.0), ("max", 1.0), ("min", 1.0), ("mean", 1.0), ("median", 1.0)],
)
def test_rolling_mixed_dtypes_axis_1(func, value):
    # GH: 20649
    df = DataFrame(1, index=[1, 2], columns=["a", "b", "c"])
    df["c"] = 1.0
    result = getattr(df.rolling(window=2, min_periods=1, axis=1), func)()
    expected = DataFrame(
        {"a": [1.0, 1.0], "b": [value, value], "c": [value, value]},
        index=[1, 2],
    )
    tm.assert_frame_equal(result, expected)


def test_rolling_axis_one_with_nan():
    # GH: 35596
    df = DataFrame(
        [
            [0, 1, 2, 4, np.nan, np.nan, np.nan],
            [0, 1, 2, np.nan, np.nan, np.nan, np.nan],
            [0, 2, 2, np.nan, 2, np.nan, 1],
        ]
    )
    result = df.rolling(window=7, min_periods=1, axis="columns").sum()
    expected = DataFrame(
        [
            [0.0, 1.0, 3.0, 7.0, 7.0, 7.0, 7.0],
            [0.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [0.0, 2.0, 4.0, 4.0, 6.0, 6.0, 7.0],
        ]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    ["test", to_datetime("2019-12-31"), to_timedelta("1 days 06:05:01.00003")],
)
def test_rolling_axis_1_non_numeric_dtypes(value):
    # GH: 20649
    df = DataFrame({"a": [1, 2]})
    df["b"] = value
    result = df.rolling(window=2, min_periods=1, axis=1).sum()
    expected = DataFrame({"a": [1.0, 2.0]})
    tm.assert_frame_equal(result, expected)


def test_rolling_on_df_transposed():
    # GH: 32724
    df = DataFrame({"A": [1, None], "B": [4, 5], "C": [7, 8]})
    expected = DataFrame({"A": [1.0, np.nan], "B": [5.0, 5.0], "C": [11.0, 13.0]})
    result = df.rolling(min_periods=1, window=2, axis=1).sum()
    tm.assert_frame_equal(result, expected)

    result = df.T.rolling(min_periods=1, window=2).sum().T
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    [
        (
            period_range(start="2020-01-01 08:00", end="2020-01-01 08:08", freq="T"),
            "2T",
        ),
        (
            period_range(start="2020-01-01 08:00", end="2020-01-01 12:00", freq="30T"),
            "1h",
        ),
    ],
)
@pytest.mark.parametrize(
    ("func", "values"),
    [
        ("min", [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]),
        ("max", [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]),
        ("sum", [np.nan, 0, 1, 3, 5, 7, 9, 11, 13]),
    ],
)
def test_rolling_period_index(index, window, func, values):
    # GH: 34225
    ds = Series([0, 1, 2, 3, 4, 5, 6, 7, 8], index=index)
    result = getattr(ds.rolling(window, closed="left"), func)()
    expected = Series(values, index=index)
    tm.assert_series_equal(result, expected)


def test_rolling_sem(frame_or_series):
    # GH: 26476
    obj = frame_or_series([0, 1, 2])
    result = obj.rolling(2, min_periods=1).sem()
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    expected = Series([np.nan] + [0.7071067811865476] * 2)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(is_platform_arm() and not is_platform_mac(), reason="GH 38921")
@pytest.mark.parametrize(
    ("func", "third_value", "values"),
    [
        ("var", 1, [5e33, 0, 0.5, 0.5, 2, 0]),
        ("std", 1, [7.071068e16, 0, 0.7071068, 0.7071068, 1.414214, 0]),
        ("var", 2, [5e33, 0.5, 0, 0.5, 2, 0]),
        ("std", 2, [7.071068e16, 0.7071068, 0, 0.7071068, 1.414214, 0]),
    ],
)
def test_rolling_var_numerical_issues(func, third_value, values):
    # GH: 37051
    ds = Series([99999999999999999, 1, third_value, 2, 3, 1, 1])
    result = getattr(ds.rolling(2), func)()
    expected = Series([np.nan] + values)
    tm.assert_series_equal(result, expected)


def test_timeoffset_as_window_parameter_for_corr():
    # GH: 28266
    exp = DataFrame(
        {
            "B": [
                np.nan,
                np.nan,
                0.9999999999999998,
                -1.0,
                1.0,
                -0.3273268353539892,
                0.9999999999999998,
                1.0,
                0.9999999999999998,
                1.0,
            ],
            "A": [
                np.nan,
                np.nan,
                -1.0,
                1.0000000000000002,
                -0.3273268353539892,
                0.9999999999999966,
                1.0,
                1.0000000000000002,
                1.0,
                1.0000000000000002,
            ],
        },
        index=MultiIndex.from_tuples(
            [
                (Timestamp("20130101 09:00:00"), "B"),
                (Timestamp("20130101 09:00:00"), "A"),
                (Timestamp("20130102 09:00:02"), "B"),
                (Timestamp("20130102 09:00:02"), "A"),
                (Timestamp("20130103 09:00:03"), "B"),
                (Timestamp("20130103 09:00:03"), "A"),
                (Timestamp("20130105 09:00:05"), "B"),
                (Timestamp("20130105 09:00:05"), "A"),
                (Timestamp("20130106 09:00:06"), "B"),
                (Timestamp("20130106 09:00:06"), "A"),
            ]
        ),
    )

    df = DataFrame(
        {"B": [0, 1, 2, 4, 3], "A": [7, 4, 6, 9, 3]},
        index=[
            Timestamp("20130101 09:00:00"),
            Timestamp("20130102 09:00:02"),
            Timestamp("20130103 09:00:03"),
            Timestamp("20130105 09:00:05"),
            Timestamp("20130106 09:00:06"),
        ],
    )

    res = df.rolling(window="3d").corr()

    tm.assert_frame_equal(exp, res)


@pytest.mark.parametrize("method", ["var", "sum", "mean", "skew", "kurt", "min", "max"])
def test_rolling_decreasing_indices(method):
    """
    Make sure that decreasing indices give the same results as increasing indices.

    GH 36933
    """
    df = DataFrame({"values": np.arange(-15, 10) ** 2})
    df_reverse = DataFrame({"values": df["values"][::-1]}, index=df.index[::-1])

    increasing = getattr(df.rolling(window=5), method)()
    decreasing = getattr(df_reverse.rolling(window=5), method)()

    assert np.abs(decreasing.values[::-1][:-4] - increasing.values[4:]).max() < 1e-12


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            "var",
            [
                float("nan"),
                43.0,
                float("nan"),
                136.333333,
                43.5,
                94.966667,
                182.0,
                318.0,
            ],
        ),
        (
            "mean",
            [float("nan"), 7.5, float("nan"), 21.5, 6.0, 9.166667, 13.0, 17.5],
        ),
        (
            "sum",
            [float("nan"), 30.0, float("nan"), 86.0, 30.0, 55.0, 91.0, 140.0],
        ),
        (
            "skew",
            [
                float("nan"),
                0.709296,
                float("nan"),
                0.407073,
                0.984656,
                0.919184,
                0.874674,
                0.842418,
            ],
        ),
        (
            "kurt",
            [
                float("nan"),
                -0.5916711736073559,
                float("nan"),
                -1.0028993131317954,
                -0.06103844629409494,
                -0.254143227116194,
                -0.37362637362637585,
                -0.45439658241367054,
            ],
        ),
    ],
)
def test_rolling_non_monotonic(method, expected):
    """
    Make sure the (rare) branch of non-monotonic indices is covered by a test.

    output from 1.1.3 is assumed to be the expected output. Output of sum/mean has
    manually been verified.

    GH 36933.
    """
    # Based on an example found in computation.rst
    use_expanding = [True, False, True, False, True, True, True, True]
    df = DataFrame({"values": np.arange(len(use_expanding)) ** 2})

    class CustomIndexer(BaseIndexer):
        def get_window_bounds(self, num_values, min_periods, center, closed):
            start = np.empty(num_values, dtype=np.int64)
            end = np.empty(num_values, dtype=np.int64)
            for i in range(num_values):
                if self.use_expanding[i]:
                    start[i] = 0
                    end[i] = i + 1
                else:
                    start[i] = i
                    end[i] = i + self.window_size
            return start, end

    indexer = CustomIndexer(window_size=4, use_expanding=use_expanding)

    result = getattr(df.rolling(indexer), method)()
    expected = DataFrame({"values": expected})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("index", "window"),
    [
        ([0, 1, 2, 3, 4], 2),
        (date_range("2001-01-01", freq="D", periods=5), "2D"),
    ],
)
def test_rolling_corr_timedelta_index(index, window):
    # GH: 31286
    x = Series([1, 2, 3, 4, 5], index=index)
    y = x.copy()
    x[0:2] = 0.0
    result = x.rolling(window).corr(y)
    expected = Series([np.nan, np.nan, 1, 1, 1], index=index)
    tm.assert_almost_equal(result, expected)


def test_groupby_rolling_nan_included():
    # GH 35542
    data = {"group": ["g1", np.nan, "g1", "g2", np.nan], "B": [0, 1, 2, 3, 4]}
    df = DataFrame(data)
    result = df.groupby("group", dropna=False).rolling(1, min_periods=1).mean()
    expected = DataFrame(
        {"B": [0.0, 2.0, 3.0, 1.0, 4.0]},
        # GH-38057 from_tuples puts the NaNs in the codes, result expects them
        # to be in the levels, at the moment
        # index=MultiIndex.from_tuples(
        #     [("g1", 0), ("g1", 2), ("g2", 3), (np.nan, 1), (np.nan, 4)],
        #     names=["group", None],
        # ),
        index=MultiIndex(
            [["g1", "g2", np.nan], [0, 1, 2, 3, 4]],
            [[0, 0, 1, 2, 2], [0, 2, 3, 1, 4]],
            names=["group", None],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_rolling_skew_kurt_numerical_stability(method):
    # GH#6929
    ser = Series(np.random.rand(10))
    ser_copy = ser.copy()
    expected = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(ser, ser_copy)
    ser = ser + 50000
    result = getattr(ser.rolling(3), method)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("method", "values"),
    [
        ("skew", [2.0, 0.854563, 0.0, 1.999984]),
        ("kurt", [4.0, -1.289256, -1.2, 3.999946]),
    ],
)
def test_rolling_skew_kurt_large_value_range(method, values):
    # GH: 37557
    s = Series([3000000, 1, 1, 2, 3, 4, 999])
    result = getattr(s.rolling(4), method)()
    expected = Series([np.nan] * 3 + values)
    tm.assert_series_equal(result, expected)


def test_invalid_method():
    with pytest.raises(ValueError, match="method must be 'table' or 'single"):
        Series(range(1)).rolling(1, method="foo")


@pytest.mark.parametrize("window", [1, "1d"])
def test_rolling_descending_date_order_with_offset(window, frame_or_series):
    # GH#40002
    idx = date_range(start="2020-01-01", end="2020-01-03", freq="1d")
    obj = frame_or_series(range(1, 4), index=idx)
    result = obj.rolling("1d", closed="left").sum()
    expected = frame_or_series([np.nan, 1, 2], index=idx)
    tm.assert_equal(result, expected)

    result = obj.iloc[::-1].rolling("1d", closed="left").sum()
    idx = date_range(start="2020-01-03", end="2020-01-01", freq="-1d")
    expected = frame_or_series([np.nan, 3, 2], index=idx)
    tm.assert_equal(result, expected)


def test_rolling_var_floating_artifact_precision():
    # GH 37051
    s = Series([7, 5, 5, 5])
    result = s.rolling(3).var()
    expected = Series([np.nan, np.nan, 4 / 3, 0])
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)


def test_rolling_std_small_values():
    # GH 37051
    s = Series(
        [
            0.00000054,
            0.00000053,
            0.00000054,
        ]
    )
    result = s.rolling(2).std()
    expected = Series([np.nan, 7.071068e-9, 7.071068e-9])
    tm.assert_series_equal(result, expected, atol=1.0e-15, rtol=1.0e-15)


@pytest.mark.parametrize(
    "start, exp_values",
    [
        (1, [0.03, 0.0155, 0.0155, 0.011, 0.01025]),
        (2, [0.001, 0.001, 0.0015, 0.00366666]),
    ],
)
def test_rolling_mean_all_nan_window_floating_artifacts(start, exp_values):
    # GH#41053
    df = DataFrame(
        [
            0.03,
            0.03,
            0.001,
            np.NaN,
            0.002,
            0.008,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            np.NaN,
            0.005,
            0.2,
        ]
    )

    values = exp_values + [
        0.00366666,
        0.005,
        0.005,
        0.008,
        np.NaN,
        np.NaN,
        0.005,
        0.102500,
    ]
    expected = DataFrame(
        values,
        index=list(range(start, len(values) + start)),
    )
    result = df.iloc[start:].rolling(5, min_periods=0).mean()
    tm.assert_frame_equal(result, expected)


def test_rolling_sum_all_nan_window_floating_artifacts():
    # GH#41053
    df = DataFrame([0.002, 0.008, 0.005, np.NaN, np.NaN, np.NaN])
    result = df.rolling(3, min_periods=0).sum()
    expected = DataFrame([0.002, 0.010, 0.015, 0.013, 0.005, 0.0])
    tm.assert_frame_equal(result, expected)


def test_rolling_zero_window():
    # GH 22719
    s = Series(range(1))
    result = s.rolling(0).min()
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)
