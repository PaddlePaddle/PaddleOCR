import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


def test_centered_axis_validation():

    # ok
    Series(np.ones(10)).rolling(window=3, center=True, axis=0).mean()

    # bad axis
    msg = "No axis named 1 for object type Series"
    with pytest.raises(ValueError, match=msg):
        Series(np.ones(10)).rolling(window=3, center=True, axis=1).mean()

    # ok ok
    DataFrame(np.ones((10, 10))).rolling(window=3, center=True, axis=0).mean()
    DataFrame(np.ones((10, 10))).rolling(window=3, center=True, axis=1).mean()

    # bad axis
    msg = "No axis named 2 for object type DataFrame"
    with pytest.raises(ValueError, match=msg):
        (DataFrame(np.ones((10, 10))).rolling(window=3, center=True, axis=2).mean())


@td.skip_if_no_scipy
def test_cmov_mean():
    # GH 8238
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    result = Series(vals).rolling(5, center=True).mean()
    expected_values = [
        np.nan,
        np.nan,
        9.962,
        11.27,
        11.564,
        12.516,
        12.818,
        12.952,
        np.nan,
        np.nan,
    ]
    expected = Series(expected_values)
    tm.assert_series_equal(expected, result)


@td.skip_if_no_scipy
def test_cmov_window():
    # GH 8238
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    result = Series(vals).rolling(5, win_type="boxcar", center=True).mean()
    expected_values = [
        np.nan,
        np.nan,
        9.962,
        11.27,
        11.564,
        12.516,
        12.818,
        12.952,
        np.nan,
        np.nan,
    ]
    expected = Series(expected_values)
    tm.assert_series_equal(expected, result)


@td.skip_if_no_scipy
def test_cmov_window_corner():
    # GH 8238
    # all nan
    vals = Series([np.nan] * 10)
    result = vals.rolling(5, center=True, win_type="boxcar").mean()
    assert np.isnan(result).all()

    # empty
    vals = Series([], dtype=object)
    result = vals.rolling(5, center=True, win_type="boxcar").mean()
    assert len(result) == 0

    # shorter than window
    vals = Series(np.random.randn(5))
    result = vals.rolling(10, win_type="boxcar").mean()
    assert np.isnan(result).all()
    assert len(result) == 5


@td.skip_if_no_scipy
@pytest.mark.parametrize(
    "f,xp",
    [
        (
            "mean",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [9.252, 9.392],
                [8.644, 9.906],
                [8.87, 10.208],
                [6.81, 8.588],
                [7.792, 8.644],
                [9.05, 7.824],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
        (
            "std",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [3.789706, 4.068313],
                [3.429232, 3.237411],
                [3.589269, 3.220810],
                [3.405195, 2.380655],
                [3.281839, 2.369869],
                [3.676846, 1.801799],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
        (
            "var",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [14.36187, 16.55117],
                [11.75963, 10.48083],
                [12.88285, 10.37362],
                [11.59535, 5.66752],
                [10.77047, 5.61628],
                [13.51920, 3.24648],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
        (
            "sum",
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [46.26, 46.96],
                [43.22, 49.53],
                [44.35, 51.04],
                [34.05, 42.94],
                [38.96, 43.22],
                [45.25, 39.12],
                [np.nan, np.nan],
                [np.nan, np.nan],
            ],
        ),
    ],
)
def test_cmov_window_frame(f, xp):
    # Gh 8238
    df = DataFrame(
        np.array(
            [
                [12.18, 3.64],
                [10.18, 9.16],
                [13.24, 14.61],
                [4.51, 8.11],
                [6.15, 11.44],
                [9.14, 6.21],
                [11.31, 10.67],
                [2.94, 6.51],
                [9.42, 8.39],
                [12.44, 7.34],
            ]
        )
    )
    xp = DataFrame(np.array(xp))

    roll = df.rolling(5, win_type="boxcar", center=True)
    rs = getattr(roll, f)()

    tm.assert_frame_equal(xp, rs)


@td.skip_if_no_scipy
def test_cmov_window_na_min_periods():
    # min_periods
    vals = Series(np.random.randn(10))
    vals[4] = np.nan
    vals[8] = np.nan

    xp = vals.rolling(5, min_periods=4, center=True).mean()
    rs = vals.rolling(5, win_type="boxcar", min_periods=4, center=True).mean()
    tm.assert_series_equal(xp, rs)


@td.skip_if_no_scipy
def test_cmov_window_regular(win_types):
    # GH 8238
    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])
    xps = {
        "hamming": [
            np.nan,
            np.nan,
            8.71384,
            9.56348,
            12.38009,
            14.03687,
            13.8567,
            11.81473,
            np.nan,
            np.nan,
        ],
        "triang": [
            np.nan,
            np.nan,
            9.28667,
            10.34667,
            12.00556,
            13.33889,
            13.38,
            12.33667,
            np.nan,
            np.nan,
        ],
        "barthann": [
            np.nan,
            np.nan,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            14.0825,
            11.5675,
            np.nan,
            np.nan,
        ],
        "bohman": [
            np.nan,
            np.nan,
            7.61599,
            9.1764,
            12.83559,
            14.17267,
            14.65923,
            11.10401,
            np.nan,
            np.nan,
        ],
        "blackmanharris": [
            np.nan,
            np.nan,
            6.97691,
            9.16438,
            13.05052,
            14.02156,
            15.10512,
            10.74574,
            np.nan,
            np.nan,
        ],
        "nuttall": [
            np.nan,
            np.nan,
            7.04618,
            9.16786,
            13.02671,
            14.03559,
            15.05657,
            10.78514,
            np.nan,
            np.nan,
        ],
        "blackman": [
            np.nan,
            np.nan,
            7.73345,
            9.17869,
            12.79607,
            14.20036,
            14.57726,
            11.16988,
            np.nan,
            np.nan,
        ],
        "bartlett": [
            np.nan,
            np.nan,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            14.0825,
            11.5675,
            np.nan,
            np.nan,
        ],
    }

    xp = Series(xps[win_types])
    rs = Series(vals).rolling(5, win_type=win_types, center=True).mean()
    tm.assert_series_equal(xp, rs)


@td.skip_if_no_scipy
def test_cmov_window_regular_linear_range(win_types):
    # GH 8238
    vals = np.array(range(10), dtype=float)
    xp = vals.copy()
    xp[:2] = np.nan
    xp[-2:] = np.nan
    xp = Series(xp)

    rs = Series(vals).rolling(5, win_type=win_types, center=True).mean()
    tm.assert_series_equal(xp, rs)


@td.skip_if_no_scipy
def test_cmov_window_regular_missing_data(win_types):
    # GH 8238
    vals = np.array(
        [6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, np.nan, 10.63, 14.48]
    )
    xps = {
        "bartlett": [
            np.nan,
            np.nan,
            9.70333,
            10.5225,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            15.61667,
            13.655,
        ],
        "blackman": [
            np.nan,
            np.nan,
            9.04582,
            11.41536,
            7.73345,
            9.17869,
            12.79607,
            14.20036,
            15.8706,
            13.655,
        ],
        "barthann": [
            np.nan,
            np.nan,
            9.70333,
            10.5225,
            8.4425,
            9.1925,
            12.5575,
            14.3675,
            15.61667,
            13.655,
        ],
        "bohman": [
            np.nan,
            np.nan,
            8.9444,
            11.56327,
            7.61599,
            9.1764,
            12.83559,
            14.17267,
            15.90976,
            13.655,
        ],
        "hamming": [
            np.nan,
            np.nan,
            9.59321,
            10.29694,
            8.71384,
            9.56348,
            12.38009,
            14.20565,
            15.24694,
            13.69758,
        ],
        "nuttall": [
            np.nan,
            np.nan,
            8.47693,
            12.2821,
            7.04618,
            9.16786,
            13.02671,
            14.03673,
            16.08759,
            13.65553,
        ],
        "triang": [
            np.nan,
            np.nan,
            9.33167,
            9.76125,
            9.28667,
            10.34667,
            12.00556,
            13.82125,
            14.49429,
            13.765,
        ],
        "blackmanharris": [
            np.nan,
            np.nan,
            8.42526,
            12.36824,
            6.97691,
            9.16438,
            13.05052,
            14.02175,
            16.1098,
            13.65509,
        ],
    }

    xp = Series(xps[win_types])
    rs = Series(vals).rolling(5, win_type=win_types, min_periods=3).mean()
    tm.assert_series_equal(xp, rs)


@td.skip_if_no_scipy
def test_cmov_window_special(win_types_special):
    # GH 8238
    kwds = {
        "kaiser": {"beta": 1.0},
        "gaussian": {"std": 1.0},
        "general_gaussian": {"p": 2.0, "sig": 2.0},
        "exponential": {"tau": 10},
    }

    vals = np.array([6.95, 15.21, 4.72, 9.12, 13.81, 13.49, 16.68, 9.48, 10.63, 14.48])

    xps = {
        "gaussian": [
            np.nan,
            np.nan,
            8.97297,
            9.76077,
            12.24763,
            13.89053,
            13.65671,
            12.01002,
            np.nan,
            np.nan,
        ],
        "general_gaussian": [
            np.nan,
            np.nan,
            9.85011,
            10.71589,
            11.73161,
            13.08516,
            12.95111,
            12.74577,
            np.nan,
            np.nan,
        ],
        "kaiser": [
            np.nan,
            np.nan,
            9.86851,
            11.02969,
            11.65161,
            12.75129,
            12.90702,
            12.83757,
            np.nan,
            np.nan,
        ],
        "exponential": [
            np.nan,
            np.nan,
            9.83364,
            11.10472,
            11.64551,
            12.66138,
            12.92379,
            12.83770,
            np.nan,
            np.nan,
        ],
    }

    xp = Series(xps[win_types_special])
    rs = (
        Series(vals)
        .rolling(5, win_type=win_types_special, center=True)
        .mean(**kwds[win_types_special])
    )
    tm.assert_series_equal(xp, rs)


@td.skip_if_no_scipy
def test_cmov_window_special_linear_range(win_types_special):
    # GH 8238
    kwds = {
        "kaiser": {"beta": 1.0},
        "gaussian": {"std": 1.0},
        "general_gaussian": {"p": 2.0, "sig": 2.0},
        "slepian": {"width": 0.5},
        "exponential": {"tau": 10},
    }

    vals = np.array(range(10), dtype=float)
    xp = vals.copy()
    xp[:2] = np.nan
    xp[-2:] = np.nan
    xp = Series(xp)

    rs = (
        Series(vals)
        .rolling(5, win_type=win_types_special, center=True)
        .mean(**kwds[win_types_special])
    )
    tm.assert_series_equal(xp, rs)


def test_rolling_min_min_periods():
    a = Series([1, 2, 3, 4, 5])
    result = a.rolling(window=100, min_periods=1).min()
    expected = Series(np.ones(len(a)))
    tm.assert_series_equal(result, expected)
    msg = "min_periods 5 must be <= window 3"
    with pytest.raises(ValueError, match=msg):
        Series([1, 2, 3]).rolling(window=3, min_periods=5).min()


def test_rolling_max_min_periods():
    a = Series([1, 2, 3, 4, 5], dtype=np.float64)
    b = a.rolling(window=100, min_periods=1).max()
    tm.assert_almost_equal(a, b)
    msg = "min_periods 5 must be <= window 3"
    with pytest.raises(ValueError, match=msg):
        Series([1, 2, 3]).rolling(window=3, min_periods=5).max()


def test_rolling_quantile_np_percentile():
    # #9413: Tests that rolling window's quantile default behavior
    # is analogous to Numpy's percentile
    row = 10
    col = 5
    idx = date_range("20100101", periods=row, freq="B")
    df = DataFrame(np.random.rand(row * col).reshape((row, -1)), index=idx)

    df_quantile = df.quantile([0.25, 0.5, 0.75], axis=0)
    np_percentile = np.percentile(df, [25, 50, 75], axis=0)

    tm.assert_almost_equal(df_quantile.values, np.array(np_percentile))


@pytest.mark.parametrize("quantile", [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0],
        [0.0, np.nan, 0.2, np.nan, 0.4],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5],
        [0.5],
        [np.nan, 0.7, 0.6],
    ],
)
def test_rolling_quantile_interpolation_options(quantile, interpolation, data):
    # Tests that rolling window's quantile behavior is analogous to
    # Series' quantile for each interpolation option
    s = Series(data)

    q1 = s.quantile(quantile, interpolation)
    q2 = s.expanding(min_periods=1).quantile(quantile, interpolation).iloc[-1]

    if np.isnan(q1):
        assert np.isnan(q2)
    else:
        assert q1 == q2


def test_invalid_quantile_value():
    data = np.arange(5)
    s = Series(data)

    msg = "Interpolation 'invalid' is not supported"
    with pytest.raises(ValueError, match=msg):
        s.rolling(len(data), min_periods=1).quantile(0.5, interpolation="invalid")


def test_rolling_quantile_param():
    ser = Series([0.0, 0.1, 0.5, 0.9, 1.0])
    msg = "quantile value -0.1 not in \\[0, 1\\]"
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(-0.1)

    msg = "quantile value 10.0 not in \\[0, 1\\]"
    with pytest.raises(ValueError, match=msg):
        ser.rolling(3).quantile(10.0)

    msg = "must be real number, not str"
    with pytest.raises(TypeError, match=msg):
        ser.rolling(3).quantile("foo")


def test_rolling_std_1obs():
    vals = Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = vals.rolling(1, min_periods=1).std()
    expected = Series([np.nan] * 5)
    tm.assert_series_equal(result, expected)

    result = vals.rolling(1, min_periods=1).std(ddof=0)
    expected = Series([0.0] * 5)
    tm.assert_series_equal(result, expected)

    result = Series([np.nan, np.nan, 3, 4, 5]).rolling(3, min_periods=2).std()
    assert np.isnan(result[2])


def test_rolling_std_neg_sqrt():
    # unit test from Bottleneck

    # Test move_nanstd for neg sqrt.

    a = Series(
        [
            0.0011448196318903589,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
            0.00028718669878572767,
        ]
    )
    b = a.rolling(window=3).std()
    assert np.isfinite(b[2:]).all()

    b = a.ewm(span=3).std()
    assert np.isfinite(b[2:]).all()
