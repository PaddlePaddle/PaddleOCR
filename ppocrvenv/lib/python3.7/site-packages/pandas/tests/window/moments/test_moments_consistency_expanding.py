import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
)
import pandas._testing as tm


def test_expanding_corr(series):
    A = series.dropna()
    B = (A + np.random.randn(len(A)))[:-5]

    result = A.expanding().corr(B)

    rolling_result = A.rolling(window=len(A), min_periods=1).corr(B)

    tm.assert_almost_equal(rolling_result, result)


def test_expanding_count(series):
    result = series.expanding(min_periods=0).count()
    tm.assert_almost_equal(
        result, series.rolling(window=len(series), min_periods=0).count()
    )


def test_expanding_quantile(series):
    result = series.expanding().quantile(0.5)

    rolling_result = series.rolling(window=len(series), min_periods=1).quantile(0.5)

    tm.assert_almost_equal(result, rolling_result)


def test_expanding_cov(series):
    A = series
    B = (A + np.random.randn(len(A)))[:-5]

    result = A.expanding().cov(B)

    rolling_result = A.rolling(window=len(A), min_periods=1).cov(B)

    tm.assert_almost_equal(rolling_result, result)


def test_expanding_cov_pairwise(frame):
    result = frame.expanding().cov()

    rolling_result = frame.rolling(window=len(frame), min_periods=1).cov()

    tm.assert_frame_equal(result, rolling_result)


def test_expanding_corr_pairwise(frame):
    result = frame.expanding().corr()

    rolling_result = frame.rolling(window=len(frame), min_periods=1).corr()
    tm.assert_frame_equal(result, rolling_result)


@pytest.mark.parametrize(
    "func,static_comp",
    [("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min)],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_func(func, static_comp, frame_or_series):
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result = getattr(data.expanding(min_periods=1, axis=0), func)()
    assert isinstance(result, frame_or_series)

    if frame_or_series is Series:
        tm.assert_almost_equal(result[10], static_comp(data[:11]))
    else:
        tm.assert_series_equal(
            result.iloc[10], static_comp(data[:11]), check_names=False
        )


@pytest.mark.parametrize(
    "func,static_comp",
    [("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min)],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_min_periods(func, static_comp):
    ser = Series(np.random.randn(50))

    result = getattr(ser.expanding(min_periods=30, axis=0), func)()
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))

    # min_periods is working correctly
    result = getattr(ser.expanding(min_periods=15, axis=0), func)()
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])

    ser2 = Series(np.random.randn(20))
    result = getattr(ser2.expanding(min_periods=5, axis=0), func)()
    assert isna(result[3])
    assert notna(result[4])

    # min_periods=0
    result0 = getattr(ser.expanding(min_periods=0, axis=0), func)()
    result1 = getattr(ser.expanding(min_periods=1, axis=0), func)()
    tm.assert_almost_equal(result0, result1)

    result = getattr(ser.expanding(min_periods=1, axis=0), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))


def test_expanding_apply(engine_and_raw, frame_or_series):
    engine, raw = engine_and_raw
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    result = data.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isinstance(result, frame_or_series)

    if frame_or_series is Series:
        tm.assert_almost_equal(result[9], np.mean(data[:11]))
    else:
        tm.assert_series_equal(result.iloc[9], np.mean(data[:11]), check_names=False)


def test_expanding_min_periods_apply(engine_and_raw):
    engine, raw = engine_and_raw
    ser = Series(np.random.randn(50))

    result = ser.expanding(min_periods=30).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))

    # min_periods is working correctly
    result = ser.expanding(min_periods=15).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])

    ser2 = Series(np.random.randn(20))
    result = ser2.expanding(min_periods=5).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    assert isna(result[3])
    assert notna(result[4])

    # min_periods=0
    result0 = ser.expanding(min_periods=0).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    result1 = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    tm.assert_almost_equal(result0, result1)

    result = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum])
def test_expanding_apply_consistency_sum_nans(consistency_data, min_periods, f):
    x, is_constant, no_nans = consistency_data

    if f is np.nansum and min_periods == 0:
        pass
    else:
        expanding_f_result = x.expanding(min_periods=min_periods).sum()
        expanding_apply_f_result = x.expanding(min_periods=min_periods).apply(
            func=f, raw=True
        )
        tm.assert_equal(expanding_f_result, expanding_apply_f_result)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("f", [lambda v: Series(v).sum(), np.nansum, np.sum])
def test_expanding_apply_consistency_sum_no_nans(consistency_data, min_periods, f):

    x, is_constant, no_nans = consistency_data

    if no_nans:
        if f is np.nansum and min_periods == 0:
            pass
        else:
            expanding_f_result = x.expanding(min_periods=min_periods).sum()
            expanding_apply_f_result = x.expanding(min_periods=min_periods).apply(
                func=f, raw=True
            )
            tm.assert_equal(expanding_f_result, expanding_apply_f_result)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var(consistency_data, min_periods, ddof):
    x, is_constant, no_nans = consistency_data

    mean_x = x.expanding(min_periods=min_periods).mean()
    var_x = x.expanding(min_periods=min_periods).var(ddof=ddof)
    assert not (var_x < 0).any().any()

    if ddof == 0:
        # check that biased var(x) == mean(x^2) - mean(x)^2
        mean_x2 = (x * x).expanding(min_periods=min_periods).mean()
        tm.assert_equal(var_x, mean_x2 - (mean_x * mean_x))


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ddof", [0, 1])
def test_moments_consistency_var_constant(consistency_data, min_periods, ddof):
    x, is_constant, no_nans = consistency_data

    if is_constant:
        count_x = x.expanding(min_periods=min_periods).count()
        var_x = x.expanding(min_periods=min_periods).var(ddof=ddof)

        # check that variance of constant series is identically 0
        assert not (var_x > 0).any().any()
        expected = x * np.nan
        expected[count_x >= max(min_periods, 1)] = 0.0
        if ddof == 1:
            expected[count_x < 2] = np.nan
        tm.assert_equal(var_x, expected)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_std(consistency_data, min_periods, ddof):
    x, is_constant, no_nans = consistency_data

    var_x = x.expanding(min_periods=min_periods).var(ddof=ddof)
    std_x = x.expanding(min_periods=min_periods).std(ddof=ddof)
    assert not (var_x < 0).any().any()
    assert not (std_x < 0).any().any()

    # check that var(x) == std(x)^2
    tm.assert_equal(var_x, std_x * std_x)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_cov(consistency_data, min_periods, ddof):
    x, is_constant, no_nans = consistency_data
    var_x = x.expanding(min_periods=min_periods).var(ddof=ddof)
    assert not (var_x < 0).any().any()

    cov_x_x = x.expanding(min_periods=min_periods).cov(x, ddof=ddof)
    assert not (cov_x_x < 0).any().any()

    # check that var(x) == cov(x, x)
    tm.assert_equal(var_x, cov_x_x)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ddof", [0, 1])
def test_expanding_consistency_series_cov_corr(consistency_data, min_periods, ddof):
    x, is_constant, no_nans = consistency_data

    if isinstance(x, Series):
        var_x_plus_y = (x + x).expanding(min_periods=min_periods).var(ddof=ddof)
        var_x = x.expanding(min_periods=min_periods).var(ddof=ddof)
        var_y = x.expanding(min_periods=min_periods).var(ddof=ddof)
        cov_x_y = x.expanding(min_periods=min_periods).cov(x, ddof=ddof)
        # check that cov(x, y) == (var(x+y) - var(x) -
        # var(y)) / 2
        tm.assert_equal(cov_x_y, 0.5 * (var_x_plus_y - var_x - var_y))

        # check that corr(x, y) == cov(x, y) / (std(x) *
        # std(y))
        corr_x_y = x.expanding(min_periods=min_periods).corr(x)
        std_x = x.expanding(min_periods=min_periods).std(ddof=ddof)
        std_y = x.expanding(min_periods=min_periods).std(ddof=ddof)
        tm.assert_equal(corr_x_y, cov_x_y / (std_x * std_y))

        if ddof == 0:
            # check that biased cov(x, y) == mean(x*y) -
            # mean(x)*mean(y)
            mean_x = x.expanding(min_periods=min_periods).mean()
            mean_y = x.expanding(min_periods=min_periods).mean()
            mean_x_times_y = (x * x).expanding(min_periods=min_periods).mean()
            tm.assert_equal(cov_x_y, mean_x_times_y - (mean_x * mean_y))


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
def test_expanding_consistency_mean(consistency_data, min_periods):
    x, is_constant, no_nans = consistency_data

    result = x.expanding(min_periods=min_periods).mean()
    expected = (
        x.expanding(min_periods=min_periods).sum()
        / x.expanding(min_periods=min_periods).count()
    )
    tm.assert_equal(result, expected.astype("float64"))


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
def test_expanding_consistency_constant(consistency_data, min_periods):
    x, is_constant, no_nans = consistency_data

    if is_constant:
        count_x = x.expanding().count()
        mean_x = x.expanding(min_periods=min_periods).mean()
        # check that correlation of a series with itself is either 1 or NaN
        corr_x_x = x.expanding(min_periods=min_periods).corr(x)

        exp = x.max() if isinstance(x, Series) else x.max().max()

        # check mean of constant series
        expected = x * np.nan
        expected[count_x >= max(min_periods, 1)] = exp
        tm.assert_equal(mean_x, expected)

        # check correlation of constant series with itself is NaN
        expected[:] = np.nan
        tm.assert_equal(corr_x_x, expected)


@pytest.mark.parametrize("min_periods", [0, 1, 2, 3, 4])
def test_expanding_consistency_var_debiasing_factors(consistency_data, min_periods):
    x, is_constant, no_nans = consistency_data

    # check variance debiasing factors
    var_unbiased_x = x.expanding(min_periods=min_periods).var()
    var_biased_x = x.expanding(min_periods=min_periods).var(ddof=0)
    var_debiasing_factors_x = x.expanding().count() / (
        x.expanding().count() - 1.0
    ).replace(0.0, np.nan)
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.expanding(min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.expanding(min_periods=5).corr(x, pairwise=True)),
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
    "f",
    [
        lambda x: x.expanding().count(),
        lambda x: x.expanding(min_periods=5).cov(x, pairwise=False),
        lambda x: x.expanding(min_periods=5).corr(x, pairwise=False),
        lambda x: x.expanding(min_periods=5).max(),
        lambda x: x.expanding(min_periods=5).min(),
        lambda x: x.expanding(min_periods=5).sum(),
        lambda x: x.expanding(min_periods=5).mean(),
        lambda x: x.expanding(min_periods=5).std(),
        lambda x: x.expanding(min_periods=5).var(),
        lambda x: x.expanding(min_periods=5).skew(),
        lambda x: x.expanding(min_periods=5).kurt(),
        lambda x: x.expanding(min_periods=5).quantile(0.5),
        lambda x: x.expanding(min_periods=5).median(),
        lambda x: x.expanding(min_periods=5).apply(sum, raw=False),
        lambda x: x.expanding(min_periods=5).apply(sum, raw=True),
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


def test_expanding_apply_empty_series(engine_and_raw):
    engine, raw = engine_and_raw
    ser = Series([], dtype=np.float64)
    tm.assert_series_equal(
        ser, ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine)
    )


def test_expanding_apply_min_periods_0(engine_and_raw):
    # GH 8080
    engine, raw = engine_and_raw
    s = Series([None, None, None])
    result = s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    expected = Series([1.0, 2.0, 3.0])
    tm.assert_series_equal(result, expected)


def test_expanding_cov_diff_index():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.expanding().cov(s2)
    expected = Series([None, None, 2.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().cov(s2a)
    tm.assert_series_equal(result, expected)

    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().cov(s2)
    expected = Series([None, None, None, 4.5])
    tm.assert_series_equal(result, expected)


def test_expanding_corr_diff_index():
    # GH 7512
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    result = s1.expanding().corr(s2)
    expected = Series([None, None, 1.0])
    tm.assert_series_equal(result, expected)

    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().corr(s2a)
    tm.assert_series_equal(result, expected)

    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().corr(s2)
    expected = Series([None, None, None, 1.0])
    tm.assert_series_equal(result, expected)


def test_expanding_cov_pairwise_diff_length():
    # GH 7512
    df1 = DataFrame([[1, 5], [3, 2], [3, 9]], columns=Index(["A", "B"], name="foo"))
    df1a = DataFrame(
        [[1, 5], [3, 9]], index=[0, 2], columns=Index(["A", "B"], name="foo")
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]], columns=Index(["X", "Y"], name="foo")
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=[0, 2], columns=Index(["X", "Y"], name="foo")
    )
    # TODO: xref gh-15826
    # .loc is not preserving the names
    result1 = df1.expanding().cov(df2, pairwise=True).loc[2]
    result2 = df1.expanding().cov(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().cov(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().cov(df2a, pairwise=True).loc[2]
    expected = DataFrame(
        [[-3.0, -6.0], [-5.0, -10.0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(["X", "Y"], name="foo"),
    )
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_corr_pairwise_diff_length():
    # GH 7512
    df1 = DataFrame(
        [[1, 2], [3, 2], [3, 4]], columns=["A", "B"], index=Index(range(3), name="bar")
    )
    df1a = DataFrame(
        [[1, 2], [3, 4]], index=Index([0, 2], name="bar"), columns=["A", "B"]
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]],
        columns=["X", "Y"],
        index=Index(range(3), name="bar"),
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=Index([0, 2], name="bar"), columns=["X", "Y"]
    )
    result1 = df1.expanding().corr(df2, pairwise=True).loc[2]
    result2 = df1.expanding().corr(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().corr(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().corr(df2a, pairwise=True).loc[2]
    expected = DataFrame(
        [[-1.0, -1.0], [-1.0, -1.0]], columns=["A", "B"], index=Index(["X", "Y"])
    )
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_apply_args_kwargs(engine_and_raw):
    def mean_w_arg(x, const):
        return np.mean(x) + const

    engine, raw = engine_and_raw

    df = DataFrame(np.random.rand(20, 3))

    expected = df.expanding().apply(np.mean, engine=engine, raw=raw) + 20.0

    result = df.expanding().apply(mean_w_arg, engine=engine, raw=raw, args=(20,))
    tm.assert_frame_equal(result, expected)

    result = df.expanding().apply(mean_w_arg, raw=raw, kwargs={"const": 20})
    tm.assert_frame_equal(result, expected)
