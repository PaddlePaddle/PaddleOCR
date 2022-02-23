from datetime import datetime

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
)
import pandas._testing as tm


def _rolling_consistency_cases():
    for window in [1, 2, 3, 10, 20]:
        for min_periods in {0, 1, 2, 3, 4, window}:
            if min_periods and (min_periods > window):
                continue
            for center in [False, True]:
                yield window, min_periods, center


# binary moments
def test_rolling_cov(series):
    A = series
    B = A + np.random.randn(len(A))

    result = A.rolling(window=50, min_periods=25).cov(B)
    tm.assert_almost_equal(result[-1], np.cov(A[-50:], B[-50:])[0, 1])


def test_rolling_corr(series):
    A = series
    B = A + np.random.randn(len(A))

    result = A.rolling(window=50, min_periods=25).corr(B)
    tm.assert_almost_equal(result[-1], np.corrcoef(A[-50:], B[-50:])[0, 1])

    # test for correct bias correction
    a = tm.makeTimeSeries()
    b = tm.makeTimeSeries()
    a[:5] = np.nan
    b[:10] = np.nan

    result = a.rolling(window=len(a), min_periods=1).corr(b)
    tm.assert_almost_equal(result[-1], a.corr(b))


@pytest.mark.parametrize("func", ["cov", "corr"])
def test_rolling_pairwise_cov_corr(func, frame):
    result = getattr(frame.rolling(window=10, min_periods=5), func)()
    result = result.loc[(slice(None), 1), 5]
    result.index = result.index.droplevel(1)
    expected = getattr(frame[1].rolling(window=10, min_periods=5), func)(frame[5])
    tm.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("method", ["corr", "cov"])
def test_flex_binary_frame(method, frame):
    series = frame[1]

    res = getattr(series.rolling(window=10), method)(frame)
    res2 = getattr(frame.rolling(window=10), method)(series)
    exp = frame.apply(lambda x: getattr(series.rolling(window=10), method)(x))

    tm.assert_frame_equal(res, exp)
    tm.assert_frame_equal(res2, exp)

    frame2 = frame.copy()
    frame2.values[:] = np.random.randn(*frame2.shape)

    res3 = getattr(frame.rolling(window=10), method)(frame2)
    exp = DataFrame(
        {k: getattr(frame[k].rolling(window=10), method)(frame2[k]) for k in frame}
    )
    tm.assert_frame_equal(res3, exp)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum])
def test_rolling_apply_consistency_sum_nans(
    consistency_data, window, min_periods, center, f
):
    x, is_constant, no_nans = consistency_data

    if f is np.nansum and min_periods == 0:
        pass
    else:
        rolling_f_result = x.rolling(
            window=window, min_periods=min_periods, center=center
        ).sum()
        rolling_apply_f_result = x.rolling(
            window=window, min_periods=min_periods, center=center
        ).apply(func=f, raw=True)
        tm.assert_equal(rolling_f_result, rolling_apply_f_result)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_rolling_apply_consistency_sum_no_nans(
    consistency_data, window, min_periods, center, f
):
    x, is_constant, no_nans = consistency_data

    if no_nans:
        if f is np.nansum and min_periods == 0:
            pass
        else:
            rolling_f_result = x.rolling(
                window=window, min_periods=min_periods, center=center
            ).sum()
            rolling_apply_f_result = x.rolling(
                window=window, min_periods=min_periods, center=center
            ).apply(func=f, raw=True)
            tm.assert_equal(rolling_f_result, rolling_apply_f_result)


@pytest.mark.parametrize("window", range(7))
def test_rolling_corr_with_zero_variance(window):
    # GH 18430
    s = Series(np.zeros(20))
    other = Series(np.arange(20))

    assert s.rolling(window=window).corr(other=other).isna().all()


def test_corr_sanity():
    # GH 3155
    df = DataFrame(
        np.array(
            [
                [0.87024726, 0.18505595],
                [0.64355431, 0.3091617],
                [0.92372966, 0.50552513],
                [0.00203756, 0.04520709],
                [0.84780328, 0.33394331],
                [0.78369152, 0.63919667],
            ]
        )
    )

    res = df[0].rolling(5, center=True).corr(df[1])
    assert all(np.abs(np.nan_to_num(x)) <= 1 for x in res)

    df = DataFrame(np.random.rand(30, 2))
    res = df[0].rolling(5, center=True).corr(df[1])
    assert all(np.abs(np.nan_to_num(x)) <= 1 for x in res)


def test_rolling_cov_diff_length():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.rolling(window=3, min_periods=2).cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.rolling(window=3, min_periods=2).cov(s2a)
    tm.assert_series_equal(result, expected)


def test_rolling_corr_diff_length():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.rolling(window=3, min_periods=2).corr(s2)
    expected = Series([None, None, 1.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.rolling(window=3, min_periods=2).corr(s2a)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).max(),
        lambda x: x.rolling(window=10, min_periods=5).min(),
        lambda x: x.rolling(window=10, min_periods=5).sum(),
        lambda x: x.rolling(window=10, min_periods=5).mean(),
        lambda x: x.rolling(window=10, min_periods=5).std(),
        lambda x: x.rolling(window=10, min_periods=5).var(),
        lambda x: x.rolling(window=10, min_periods=5).skew(),
        lambda x: x.rolling(window=10, min_periods=5).kurt(),
        lambda x: x.rolling(window=10, min_periods=5).quantile(quantile=0.5),
        lambda x: x.rolling(window=10, min_periods=5).median(),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True),
        pytest.param(
            lambda x: x.rolling(win_type="boxcar", window=10, min_periods=5).mean(),
            marks=td.skip_if_no_scipy,
        ),
    ],
)
def test_rolling_functions_window_non_shrinkage(f):
    # GH 7764
    s = Series(range(4))
    s_expected = Series(np.nan, index=s.index)
    df = DataFrame([[1, 5], [3, 2], [3, 9], [-1, 0]], columns=["A", "B"])
    df_expected = DataFrame(np.nan, index=df.index, columns=df.columns)

    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df_result = f(df)
    tm.assert_frame_equal(df_result, df_expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.rolling(window=10, min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.rolling(window=10, min_periods=5).corr(x, pairwise=True)),
    ],
)
def test_rolling_functions_window_non_shrinkage_binary(f):

    # corr/cov return a MI DataFrame
    df = DataFrame(
        [[1, 5], [3, 2], [3, 9], [-1, 0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(range(4), name="bar"),
    )
    df_expected = DataFrame(
        columns=Index(["A", "B"], name="foo"),
        index=MultiIndex.from_product([df.index, df.columns], names=["bar", "foo"]),
        dtype="float64",
    )
    df_result = f(df)
    tm.assert_frame_equal(df_result, df_expected)


def test_rolling_skew_edge_cases():

    all_nan = Series([np.NaN] * 5)

    # yields all NaN (0 variance)
    d = Series([1] * 5)
    x = d.rolling(window=5).skew()
    tm.assert_series_equal(all_nan, x)

    # yields all NaN (window too small)
    d = Series(np.random.randn(5))
    x = d.rolling(window=2).skew()
    tm.assert_series_equal(all_nan, x)

    # yields [NaN, NaN, NaN, 0.177994, 1.548824]
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    expected = Series([np.NaN, np.NaN, np.NaN, 0.177994, 1.548824])
    x = d.rolling(window=4).skew()
    tm.assert_series_equal(expected, x)


def test_rolling_kurt_edge_cases():

    all_nan = Series([np.NaN] * 5)

    # yields all NaN (0 variance)
    d = Series([1] * 5)
    x = d.rolling(window=5).kurt()
    tm.assert_series_equal(all_nan, x)

    # yields all NaN (window too small)
    d = Series(np.random.randn(5))
    x = d.rolling(window=3).kurt()
    tm.assert_series_equal(all_nan, x)

    # yields [NaN, NaN, NaN, 1.224307, 2.671499]
    d = Series([-1.50837035, -0.1297039, 0.19501095, 1.73508164, 0.41941401])
    expected = Series([np.NaN, np.NaN, np.NaN, 1.224307, 2.671499])
    x = d.rolling(window=4).kurt()
    tm.assert_series_equal(expected, x)


def test_rolling_skew_eq_value_fperr():
    # #18804 all rolling skew for all equal values should return Nan
    a = Series([1.1] * 15).rolling(window=10).skew()
    assert np.isnan(a).all()


def test_rolling_kurt_eq_value_fperr():
    # #18804 all rolling kurt for all equal values should return Nan
    a = Series([1.1] * 15).rolling(window=10).kurt()
    assert np.isnan(a).all()


def test_rolling_max_gh6297():
    """Replicate result expected in GH #6297"""
    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 2 datapoints on one of the days
    indices.append(datetime(1975, 1, 3, 6, 0))
    series = Series(range(1, 7), index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    expected = Series(
        [1.0, 2.0, 6.0, 4.0, 5.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    x = series.resample("D").max().rolling(window=1).max()
    tm.assert_series_equal(expected, x)


def test_rolling_max_resample():

    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 3 datapoints on last day (4, 10, and 20)
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(0, 5)) + [10, 20], index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    # Default how should be max
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 20.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    x = series.resample("D").max().rolling(window=1).max()
    tm.assert_series_equal(expected, x)

    # Now specify median (10.0)
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 10.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    x = series.resample("D").median().rolling(window=1).max()
    tm.assert_series_equal(expected, x)

    # Now specify mean (4+10+20)/3
    v = (4.0 + 10.0 + 20.0) / 3.0
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, v],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    x = series.resample("D").mean().rolling(window=1).max()
    tm.assert_series_equal(expected, x)


def test_rolling_min_resample():

    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 3 datapoints on last day (4, 10, and 20)
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(0, 5)) + [10, 20], index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    # Default how should be min
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    r = series.resample("D").min().rolling(window=1)
    tm.assert_series_equal(expected, r.min())


def test_rolling_median_resample():

    indices = [datetime(1975, 1, i) for i in range(1, 6)]
    # So that we can have 3 datapoints on last day (4, 10, and 20)
    indices.append(datetime(1975, 1, 5, 1))
    indices.append(datetime(1975, 1, 5, 2))
    series = Series(list(range(0, 5)) + [10, 20], index=indices)
    # Use floats instead of ints as values
    series = series.map(lambda x: float(x))
    # Sort chronologically
    series = series.sort_index()

    # Default how should be median
    expected = Series(
        [0.0, 1.0, 2.0, 3.0, 10],
        index=DatetimeIndex([datetime(1975, 1, i, 0) for i in range(1, 6)], freq="D"),
    )
    x = series.resample("D").median().rolling(window=1).median()
    tm.assert_series_equal(expected, x)


def test_rolling_median_memory_error():
    # GH11722
    n = 20000
    Series(np.random.randn(n)).rolling(window=2, center=False).median()
    Series(np.random.randn(n)).rolling(window=2, center=False).median()


@pytest.mark.parametrize(
    "data_type",
    [np.dtype(f"f{width}") for width in [4, 8]]
    + [np.dtype(f"{sign}{width}") for width in [1, 2, 4, 8] for sign in "ui"],
)
def test_rolling_min_max_numeric_types(data_type):
    # GH12373

    # Just testing that these don't throw exceptions and that
    # the return type is float64. Other tests will cover quantitative
    # correctness
    result = DataFrame(np.arange(20, dtype=data_type)).rolling(window=5).max()
    assert result.dtypes[0] == np.dtype("f8")
    result = DataFrame(np.arange(20, dtype=data_type)).rolling(window=5).min()
    assert result.dtypes[0] == np.dtype("f8")


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.rolling(window=10, min_periods=0).count(),
        lambda x: x.rolling(window=10, min_periods=5).cov(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).corr(x, pairwise=False),
        lambda x: x.rolling(window=10, min_periods=5).max(),
        lambda x: x.rolling(window=10, min_periods=5).min(),
        lambda x: x.rolling(window=10, min_periods=5).sum(),
        lambda x: x.rolling(window=10, min_periods=5).mean(),
        lambda x: x.rolling(window=10, min_periods=5).std(),
        lambda x: x.rolling(window=10, min_periods=5).var(),
        lambda x: x.rolling(window=10, min_periods=5).skew(),
        lambda x: x.rolling(window=10, min_periods=5).kurt(),
        lambda x: x.rolling(window=10, min_periods=5).quantile(0.5),
        lambda x: x.rolling(window=10, min_periods=5).median(),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=False),
        lambda x: x.rolling(window=10, min_periods=5).apply(sum, raw=True),
        pytest.param(
            lambda x: x.rolling(win_type="boxcar", window=10, min_periods=5).mean(),
            marks=td.skip_if_no_scipy,
        ),
    ],
)
def test_moment_functions_zero_length(f):
    # GH 8056
    s = Series(dtype=np.float64)
    s_expected = s
    df1 = DataFrame()
    df1_expected = df1
    df2 = DataFrame(columns=["a"])
    df2["a"] = df2["a"].astype("float64")
    df2_expected = df2

    s_result = f(s)
    tm.assert_series_equal(s_result, s_expected)

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.rolling(window=10, min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.rolling(window=10, min_periods=5).corr(x, pairwise=True)),
    ],
)
def test_moment_functions_zero_length_pairwise(f):

    df1 = DataFrame()
    df2 = DataFrame(columns=Index(["a"], name="foo"), index=Index([], name="bar"))
    df2["a"] = df2["a"].astype("float64")

    df1_expected = DataFrame(
        index=MultiIndex.from_product([df1.index, df1.columns]), columns=Index([])
    )
    df2_expected = DataFrame(
        index=MultiIndex.from_product([df2.index, df2.columns], names=["bar", "foo"]),
        columns=Index(["a"], name="foo"),
        dtype="float64",
    )

    df1_result = f(df1)
    tm.assert_frame_equal(df1_result, df1_expected)

    df2_result = f(df2)
    tm.assert_frame_equal(df2_result, df2_expected)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(consistency_data, window, min_periods, center, ddof):
    x, is_constant, no_nans = consistency_data

    mean_x = x.rolling(window=window, min_periods=min_periods, center=center).mean()
    var_x = x.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    assert not (var_x < 0).any().any()

    if ddof == 0:
        # check that biased var(x) == mean(x^2) - mean(x)^2
        mean_x2 = (
            (x * x)
            .rolling(window=window, min_periods=min_periods, center=center)
            .mean()
        )
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(
    consistency_data, window, min_periods, center, ddof
):
    x, is_constant, no_nans = consistency_data

    if is_constant:
        count_x = x.rolling(
            window=window, min_periods=min_periods, center=center
        ).count()
        var_x = x.rolling(window=window, min_periods=min_periods, center=center).var(
            ddof=ddof
        )

        # check that variance of constant series is identically 0
        assert not (var_x > 0).any().any()
        expected = x * np.nan
        expected[count_x >= max(min_periods, 1)] = 0.0
        if ddof == 1:
            expected[count_x < 2] = np.nan
        tm.assert_equal(var_x, expected)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_std(consistency_data, window, min_periods, center, ddof):
    x, is_constant, no_nans = consistency_data

    var_x = x.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    std_x = x.rolling(window=window, min_periods=min_periods, center=center).std(
        ddof=ddof
    )
    assert not (var_x < 0).any().any()
    assert not (std_x < 0).any().any()

    # check that var(x) == std(x)^2
    tm.assert_equal(var_x, std_x * std_x)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_cov(consistency_data, window, min_periods, center, ddof):
    x, is_constant, no_nans = consistency_data
    var_x = x.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=ddof
    )
    assert not (var_x < 0).any().any()

    cov_x_x = x.rolling(window=window, min_periods=min_periods, center=center).cov(
        x, ddof=ddof
    )
    assert not (cov_x_x < 0).any().any()

    # check that var(x) == cov(x, x)
    tm.assert_equal(var_x, cov_x_x)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
@pytest.mark.parametrize("ddof", [0, 1])
def test_rolling_consistency_series_cov_corr(
    consistency_data, window, min_periods, center, ddof
):
    x, is_constant, no_nans = consistency_data

    if isinstance(x, Series):
        var_x_plus_y = (
            (x + x)
            .rolling(window=window, min_periods=min_periods, center=center)
            .var(ddof=ddof)
        )
        var_x = x.rolling(window=window, min_periods=min_periods, center=center).var(
            ddof=ddof
        )
        var_y = x.rolling(window=window, min_periods=min_periods, center=center).var(
            ddof=ddof
        )
        cov_x_y = x.rolling(window=window, min_periods=min_periods, center=center).cov(
            x, ddof=ddof
        )
        # check that cov(x, y) == (var(x+y) - var(x) -
        # var(y)) / 2
        tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

        # check that corr(x, y) == cov(x, y) / (std(x) *
        # std(y))
        corr_x_y = x.rolling(
            window=window, min_periods=min_periods, center=center
        ).corr(x)
        std_x = x.rolling(window=window, min_periods=min_periods, center=center).std(
            ddof=ddof
        )
        std_y = x.rolling(window=window, min_periods=min_periods, center=center).std(
            ddof=ddof
        )
        tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

        if ddof == 0:
            # check that biased cov(x, y) == mean(x*y) -
            # mean(x)*mean(y)
            mean_x = x.rolling(
                window=window, min_periods=min_periods, center=center
            ).mean()
            mean_y = x.rolling(
                window=window, min_periods=min_periods, center=center
            ).mean()
            mean_x_times_y = (
                (x * x)
                .rolling(window=window, min_periods=min_periods, center=center)
                .mean()
            )
            tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
def test_rolling_consistency_mean(consistency_data, window, min_periods, center):
    x, is_constant, no_nans = consistency_data

    result = x.rolling(window=window, min_periods=min_periods, center=center).mean()
    expected = (
        x.rolling(window=window, min_periods=min_periods, center=center)
        .sum()
        .divide(
            x.rolling(window=window, min_periods=min_periods, center=center).count()
        )
    )
    tm.assert_equal(result, expected.astype("float64"))


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
def test_rolling_consistency_constant(consistency_data, window, min_periods, center):
    x, is_constant, no_nans = consistency_data

    if is_constant:
        count_x = x.rolling(
            window=window, min_periods=min_periods, center=center
        ).count()
        mean_x = x.rolling(window=window, min_periods=min_periods, center=center).mean()
        # check that correlation of a series with itself is either 1 or NaN
        corr_x_x = x.rolling(
            window=window, min_periods=min_periods, center=center
        ).corr(x)

        exp = x.max() if isinstance(x, Series) else x.max().max()

        # check mean of constant series
        expected = x * np.nan
        expected[count_x >= max(min_periods, 1)] = exp
        tm.assert_equal(mean_x, expected)

        # check correlation of constant series with itself is NaN
        expected[:] = np.nan
        tm.assert_equal(corr_x_x, expected)


@pytest.mark.parametrize(
    "window,min_periods,center", list(_rolling_consistency_cases())
)
def test_rolling_consistency_var_debiasing_factors(
    consistency_data, window, min_periods, center
):
    x, is_constant, no_nans = consistency_data

    # check variance debiasing factors
    var_unbiased_x = x.rolling(
        window=window, min_periods=min_periods, center=center
    ).var()
    var_biased_x = x.rolling(window=window, min_periods=min_periods, center=center).var(
        ddof=0
    )
    var_debiasing_factors_x = (
        x.rolling(window=window, min_periods=min_periods, center=center)
        .count()
        .divide(
            (
                x.rolling(window=window, min_periods=min_periods, center=center).count()
                - 1.0
            ).replace(0.0, np.nan)
        )
    )
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)
