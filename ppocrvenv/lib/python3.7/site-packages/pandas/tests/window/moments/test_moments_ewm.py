import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


@pytest.mark.parametrize("name", ["var", "std", "mean"])
def test_ewma_series(series, name):
    series_result = getattr(series.ewm(com=10), name)()
    assert isinstance(series_result, Series)


@pytest.mark.parametrize("name", ["var", "std", "mean"])
def test_ewma_frame(frame, name):
    frame_result = getattr(frame.ewm(com=10), name)()
    assert isinstance(frame_result, DataFrame)


def test_ewma_adjust():
    vals = Series(np.zeros(1000))
    vals[5] = 1
    result = vals.ewm(span=100, adjust=False).mean().sum()
    assert np.abs(result - 1) < 1e-2


@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("ignore_na", [True, False])
def test_ewma_cases(adjust, ignore_na):
    # try adjust/ignore_na args matrix

    s = Series([1.0, 2.0, 4.0, 8.0])

    if adjust:
        expected = Series([1.0, 1.6, 2.736842, 4.923077])
    else:
        expected = Series([1.0, 1.333333, 2.222222, 4.148148])

    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()
    tm.assert_series_equal(result, expected)


def test_ewma_nan_handling():
    s = Series([1.0] + [np.nan] * 5 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([1.0] * len(s)))

    s = Series([np.nan] * 2 + [1.0] + [np.nan] * 2 + [1.0])
    result = s.ewm(com=5).mean()
    tm.assert_series_equal(result, Series([np.nan] * 2 + [1.0] * 4))


@pytest.mark.parametrize(
    "s, adjust, ignore_na, w",
    [
        (
            Series([np.nan, 1.0, 101.0]),
            True,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), 1.0],
        ),
        (
            Series([np.nan, 1.0, 101.0]),
            True,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), 1.0],
        ),
        (
            Series([np.nan, 1.0, 101.0]),
            False,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([np.nan, 1.0, 101.0]),
            False,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            True,
            False,
            [(1.0 - (1.0 / (1.0 + 2.0))) ** 2, np.nan, 1.0],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            True,
            True,
            [(1.0 - (1.0 / (1.0 + 2.0))), np.nan, 1.0],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            False,
            False,
            [(1.0 - (1.0 / (1.0 + 2.0))) ** 2, np.nan, (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([1.0, np.nan, 101.0]),
            False,
            True,
            [(1.0 - (1.0 / (1.0 + 2.0))), np.nan, (1.0 / (1.0 + 2.0))],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            True,
            False,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))) ** 3, np.nan, np.nan, 1.0, np.nan],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            True,
            True,
            [np.nan, (1.0 - (1.0 / (1.0 + 2.0))), np.nan, np.nan, 1.0, np.nan],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            False,
            False,
            [
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                np.nan,
                (1.0 / (1.0 + 2.0)),
                np.nan,
            ],
        ),
        (
            Series([np.nan, 1.0, np.nan, np.nan, 101.0, np.nan]),
            False,
            True,
            [
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                np.nan,
                np.nan,
                (1.0 / (1.0 + 2.0)),
                np.nan,
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            True,
            False,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                1.0,
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            True,
            True,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 2,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))),
                1.0,
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            False,
            False,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 3,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) * (1.0 / (1.0 + 2.0)),
                (1.0 / (1.0 + 2.0))
                * ((1.0 - (1.0 / (1.0 + 2.0))) ** 2 + (1.0 / (1.0 + 2.0))),
            ],
        ),
        (
            Series([1.0, np.nan, 101.0, 50.0]),
            False,
            True,
            [
                (1.0 - (1.0 / (1.0 + 2.0))) ** 2,
                np.nan,
                (1.0 - (1.0 / (1.0 + 2.0))) * (1.0 / (1.0 + 2.0)),
                (1.0 / (1.0 + 2.0)),
            ],
        ),
    ],
)
def test_ewma_nan_handling_cases(s, adjust, ignore_na, w):
    # GH 7603
    expected = (s.multiply(w).cumsum() / Series(w).cumsum()).fillna(method="ffill")
    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()

    tm.assert_series_equal(result, expected)
    if ignore_na is False:
        # check that ignore_na defaults to False
        result = s.ewm(com=2.0, adjust=adjust).mean()
        tm.assert_series_equal(result, expected)


def test_ewma_span_com_args(series):
    A = series.ewm(com=9.5).mean()
    B = series.ewm(span=20).mean()
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20)

    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        series.ewm().mean()


def test_ewma_halflife_arg(series):
    A = series.ewm(com=13.932726172912965).mean()
    B = series.ewm(halflife=10.0).mean()
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        series.ewm(span=20, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, halflife=50)
    with pytest.raises(ValueError, match=msg):
        series.ewm(com=9.5, span=20, halflife=50)
    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        series.ewm()


def test_ewm_alpha():
    # GH 10789
    arr = np.random.randn(100)
    locs = np.arange(20, 40)
    arr[locs] = np.NaN

    s = Series(arr)
    a = s.ewm(alpha=0.61722699889169674).mean()
    b = s.ewm(com=0.62014947789973052).mean()
    c = s.ewm(span=2.240298955799461).mean()
    d = s.ewm(halflife=0.721792864318).mean()
    tm.assert_series_equal(a, b)
    tm.assert_series_equal(a, c)
    tm.assert_series_equal(a, d)


def test_ewm_alpha_arg(series):
    # GH 10789
    s = series
    msg = "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        s.ewm()

    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=10.0, alpha=0.5)


def test_ewm_domain_checks():
    # GH 12492
    arr = np.random.randn(100)
    locs = np.arange(20, 40)
    arr[locs] = np.NaN

    s = Series(arr)
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=-0.1)
    s.ewm(com=0.0)
    s.ewm(com=0.1)

    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.9)
    s.ewm(span=1.0)
    s.ewm(span=1.1)

    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=0.0)
    s.ewm(halflife=0.1)

    msg = "alpha must satisfy: 0 < alpha <= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=0.0)
    s.ewm(alpha=0.1)
    s.ewm(alpha=1.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=1.1)


@pytest.mark.parametrize("method", ["mean", "std", "var"])
def test_ew_empty_series(method):
    vals = Series([], dtype=np.float64)

    ewm = vals.ewm(3)
    result = getattr(ewm, method)()
    tm.assert_almost_equal(result, vals)


@pytest.mark.parametrize("min_periods", [0, 1])
@pytest.mark.parametrize("name", ["mean", "var", "std"])
def test_ew_min_periods(min_periods, name):
    # excluding NaNs correctly
    arr = np.random.randn(50)
    arr[:10] = np.NaN
    arr[-10:] = np.NaN
    s = Series(arr)

    # check min_periods
    # GH 7898
    result = getattr(s.ewm(com=50, min_periods=2), name)()
    assert result[:11].isna().all()
    assert not result[11:].isna().any()

    result = getattr(s.ewm(com=50, min_periods=min_periods), name)()
    if name == "mean":
        assert result[:10].isna().all()
        assert not result[10:].isna().any()
    else:
        # ewm.std, ewm.var (with bias=False) require at least
        # two values
        assert result[:11].isna().all()
        assert not result[11:].isna().any()

    # check series of length 0
    result = getattr(Series(dtype=object).ewm(com=50, min_periods=min_periods), name)()
    tm.assert_series_equal(result, Series(dtype="float64"))

    # check series of length 1
    result = getattr(Series([1.0]).ewm(50, min_periods=min_periods), name)()
    if name == "mean":
        tm.assert_series_equal(result, Series([1.0]))
    else:
        # ewm.std, ewm.var with bias=False require at least
        # two values
        tm.assert_series_equal(result, Series([np.NaN]))

    # pass in ints
    result2 = getattr(Series(np.arange(50)).ewm(span=10), name)()
    assert result2.dtype == np.float_
