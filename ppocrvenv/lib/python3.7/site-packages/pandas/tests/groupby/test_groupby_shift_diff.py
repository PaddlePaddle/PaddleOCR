import numpy as np
import pytest

from pandas import (
    DataFrame,
    NaT,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


def test_group_shift_with_null_key():
    # This test is designed to replicate the segfault in issue #13813.
    n_rows = 1200

    # Generate a moderately large dataframe with occasional missing
    # values in column `B`, and then group by [`A`, `B`]. This should
    # force `-1` in `labels` array of `g.grouper.group_info` exactly
    # at those places, where the group-by key is partially missing.
    df = DataFrame(
        [(i % 12, i % 3 if i % 3 else np.nan, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i % 3 and i < n_rows - 12 else np.nan) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1)

    tm.assert_frame_equal(result, expected)


def test_group_shift_with_fill_value():
    # GH #24128
    n_rows = 24
    df = DataFrame(
        [(i % 12, i % 3, i) for i in range(n_rows)],
        dtype=float,
        columns=["A", "B", "Z"],
        index=None,
    )
    g = df.groupby(["A", "B"])

    expected = DataFrame(
        [(i + 12 if i < n_rows - 12 else 0) for i in range(n_rows)],
        dtype=float,
        columns=["Z"],
        index=None,
    )
    result = g.shift(-1, fill_value=0)[["Z"]]

    tm.assert_frame_equal(result, expected)


def test_group_shift_lose_timezone():
    # GH 30134
    now_dt = Timestamp.utcnow()
    df = DataFrame({"a": [1, 1], "date": now_dt})
    result = df.groupby("a").shift(0).iloc[0]
    expected = Series({"date": now_dt}, name=result.name)
    tm.assert_series_equal(result, expected)


def test_group_diff_real(any_real_dtype):
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [1, 2, 3, 4, 5]}, dtype=any_real_dtype)
    result = df.groupby("a")["b"].diff()
    exp_dtype = "float"
    if any_real_dtype in ["int8", "int16", "float32"]:
        exp_dtype = "float32"
    expected = Series([np.nan, np.nan, np.nan, 1.0, 3.0], dtype=exp_dtype, name="b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-02"),
            Timestamp("2013-01-03"),
        ],
        [Timedelta("5 days"), Timedelta("6 days"), Timedelta("7 days")],
    ],
)
def test_group_diff_datetimelike(data):
    df = DataFrame({"a": [1, 2, 2], "b": data})
    result = df.groupby("a")["b"].diff()
    expected = Series([NaT, NaT, Timedelta("1 days")], name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_bool():
    df = DataFrame({"a": [1, 2, 3, 3, 2], "b": [True, True, False, False, True]})
    result = df.groupby("a")["b"].diff()
    expected = Series([np.nan, np.nan, np.nan, False, False], name="b")
    tm.assert_series_equal(result, expected)


def test_group_diff_object_raises(object_dtype):
    df = DataFrame(
        {"a": ["foo", "bar", "bar"], "b": ["baz", "foo", "foo"]}, dtype=object_dtype
    )
    with pytest.raises(TypeError, match=r"unsupported operand type\(s\) for -"):
        df.groupby("a")["b"].diff()
