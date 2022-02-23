import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


@pytest.mark.parametrize("bad_raw", [None, 1, 0])
def test_rolling_apply_invalid_raw(bad_raw):
    with pytest.raises(ValueError, match="raw parameter must be `True` or `False`"):
        Series(range(3)).rolling(1).apply(len, raw=bad_raw)


def test_rolling_apply_out_of_bounds(engine_and_raw):
    # gh-1850
    engine, raw = engine_and_raw

    vals = Series([1, 2, 3, 4])

    result = vals.rolling(10).apply(np.sum, engine=engine, raw=raw)
    assert result.isna().all()

    result = vals.rolling(10, min_periods=1).apply(np.sum, engine=engine, raw=raw)
    expected = Series([1, 3, 6, 10], dtype=float)
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize("window", [2, "2s"])
def test_rolling_apply_with_pandas_objects(window):
    # 5071
    df = DataFrame(
        {"A": np.random.randn(5), "B": np.random.randint(0, 10, size=5)},
        index=date_range("20130101", periods=5, freq="s"),
    )

    # we have an equal spaced timeseries index
    # so simulate removing the first period
    def f(x):
        if x.index[0] == df.index[0]:
            return np.nan
        return x.iloc[-1]

    result = df.rolling(window).apply(f, raw=False)
    expected = df.iloc[2:].reindex_like(df)
    tm.assert_frame_equal(result, expected)

    with tm.external_error_raised(AttributeError):
        df.rolling(window).apply(f, raw=True)


def test_rolling_apply(engine_and_raw):
    engine, raw = engine_and_raw

    expected = Series([], dtype="float64")
    result = expected.rolling(10).apply(lambda x: x.mean(), engine=engine, raw=raw)
    tm.assert_series_equal(result, expected)

    # gh-8080
    s = Series([None, None, None])
    result = s.rolling(2, min_periods=0).apply(lambda x: len(x), engine=engine, raw=raw)
    expected = Series([1.0, 2.0, 2.0])
    tm.assert_series_equal(result, expected)

    result = s.rolling(2, min_periods=0).apply(len, engine=engine, raw=raw)
    tm.assert_series_equal(result, expected)


def test_all_apply(engine_and_raw):
    engine, raw = engine_and_raw

    df = (
        DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        ).set_index("A")
        * 2
    )
    er = df.rolling(window=1)
    r = df.rolling(window="1s")

    result = r.apply(lambda x: 1, engine=engine, raw=raw)
    expected = er.apply(lambda x: 1, engine=engine, raw=raw)
    tm.assert_frame_equal(result, expected)


def test_ragged_apply(engine_and_raw):
    engine, raw = engine_and_raw

    df = DataFrame({"B": range(5)})
    df.index = [
        Timestamp("20130101 09:00:00"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:03"),
        Timestamp("20130101 09:00:05"),
        Timestamp("20130101 09:00:06"),
    ]

    f = lambda x: 1
    result = df.rolling(window="1s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)

    result = df.rolling(window="2s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)

    result = df.rolling(window="5s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)


def test_invalid_engine():
    with pytest.raises(ValueError, match="engine must be either 'numba' or 'cython'"):
        Series(range(1)).rolling(1).apply(lambda x: x, engine="foo")


def test_invalid_engine_kwargs_cython():
    with pytest.raises(ValueError, match="cython engine does not accept engine_kwargs"):
        Series(range(1)).rolling(1).apply(
            lambda x: x, engine="cython", engine_kwargs={"nopython": False}
        )


def test_invalid_raw_numba():
    with pytest.raises(
        ValueError, match="raw must be `True` when using the numba engine"
    ):
        Series(range(1)).rolling(1).apply(lambda x: x, raw=False, engine="numba")


@pytest.mark.parametrize("args_kwargs", [[None, {"par": 10}], [(10,), None]])
def test_rolling_apply_args_kwargs(args_kwargs):
    # GH 33433
    def foo(x, par):
        return np.sum(x + par)

    df = DataFrame({"gr": [1, 1], "a": [1, 2]})

    idx = Index(["gr", "a"])
    expected = DataFrame([[11.0, 11.0], [11.0, 12.0]], columns=idx)

    result = df.rolling(1).apply(foo, args=args_kwargs[0], kwargs=args_kwargs[1])
    tm.assert_frame_equal(result, expected)

    midx = MultiIndex.from_tuples([(1, 0), (1, 1)], names=["gr", None])
    expected = Series([11.0, 12.0], index=midx, name="a")

    gb_rolling = df.groupby("gr")["a"].rolling(1)

    result = gb_rolling.apply(foo, args=args_kwargs[0], kwargs=args_kwargs[1])
    tm.assert_series_equal(result, expected)
