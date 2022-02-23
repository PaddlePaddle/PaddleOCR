import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(
    "a_vals,b_vals",
    [
        # Ints
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([1, 2, 3, 4, 5], [4, 3, 2, 1]),
        # Floats
        ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]),
        # Missing data
        ([1.0, np.nan, 3.0, np.nan, 5.0], [5.0, np.nan, 3.0, np.nan, 1.0]),
        ([np.nan, 4.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 2.0, np.nan]),
        # Timestamps
        (
            list(pd.date_range("1/1/18", freq="D", periods=5)),
            list(pd.date_range("1/1/18", freq="D", periods=5))[::-1],
        ),
        # All NA
        ([np.nan] * 5, [np.nan] * 5),
    ],
)
@pytest.mark.parametrize("q", [0, 0.25, 0.5, 0.75, 1])
def test_quantile(interpolation, a_vals, b_vals, q):
    if interpolation == "nearest" and q == 0.5 and b_vals == [4, 3, 2, 1]:
        pytest.skip(
            "Unclear numpy expectation for nearest result with equidistant data"
        )

    a_expected = pd.Series(a_vals).quantile(q, interpolation=interpolation)
    b_expected = pd.Series(b_vals).quantile(q, interpolation=interpolation)

    df = DataFrame(
        {"key": ["a"] * len(a_vals) + ["b"] * len(b_vals), "val": a_vals + b_vals}
    )

    expected = DataFrame(
        [a_expected, b_expected], columns=["val"], index=Index(["a", "b"], name="key")
    )
    result = df.groupby("key").quantile(q, interpolation=interpolation)

    tm.assert_frame_equal(result, expected)


def test_quantile_array():
    # https://github.com/pandas-dev/pandas/issues/27526
    df = DataFrame({"A": [0, 1, 2, 3, 4]})
    result = df.groupby([0, 0, 1, 1, 1]).quantile([0.25])

    index = pd.MultiIndex.from_product([[0, 1], [0.25]])
    expected = DataFrame({"A": [0.25, 2.50]}, index=index)
    tm.assert_frame_equal(result, expected)

    df = DataFrame({"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]})
    index = pd.MultiIndex.from_product([[0, 1], [0.25, 0.75]])

    result = df.groupby([0, 0, 1, 1]).quantile([0.25, 0.75])
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.25, 2.75], "B": [4.25, 4.75, 6.25, 6.75]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array2():
    # https://github.com/pandas-dev/pandas/pull/28085#issuecomment-524066959
    df = DataFrame(
        np.random.RandomState(0).randint(0, 5, size=(10, 3)), columns=list("ABC")
    )
    result = df.groupby("A").quantile([0.3, 0.7])
    expected = DataFrame(
        {
            "B": [0.9, 2.1, 2.2, 3.4, 1.6, 2.4, 2.3, 2.7, 0.0, 0.0],
            "C": [1.2, 2.8, 1.8, 3.0, 0.0, 0.0, 1.9, 3.1, 3.0, 3.0],
        },
        index=pd.MultiIndex.from_product(
            [[0, 1, 2, 3, 4], [0.3, 0.7]], names=["A", None]
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array_no_sort():
    df = DataFrame({"A": [0, 1, 2], "B": [3, 4, 5]})
    result = df.groupby([1, 0, 1], sort=False).quantile([0.25, 0.5, 0.75])
    expected = DataFrame(
        {"A": [0.5, 1.0, 1.5, 1.0, 1.0, 1.0], "B": [3.5, 4.0, 4.5, 4.0, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.25, 0.5, 0.75]]),
    )
    tm.assert_frame_equal(result, expected)

    result = df.groupby([1, 0, 1], sort=False).quantile([0.75, 0.25])
    expected = DataFrame(
        {"A": [1.5, 0.5, 1.0, 1.0], "B": [4.5, 3.5, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.75, 0.25]]),
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array_multiple_levels():
    df = DataFrame(
        {"A": [0, 1, 2], "B": [3, 4, 5], "c": ["a", "a", "a"], "d": ["a", "a", "b"]}
    )
    result = df.groupby(["c", "d"]).quantile([0.25, 0.75])
    index = pd.MultiIndex.from_tuples(
        [("a", "a", 0.25), ("a", "a", 0.75), ("a", "b", 0.25), ("a", "b", 0.75)],
        names=["c", "d", None],
    )
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.0, 2.0], "B": [3.25, 3.75, 5.0, 5.0]}, index=index
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("frame_size", [(2, 3), (100, 10)])
@pytest.mark.parametrize("groupby", [[0], [0, 1]])
@pytest.mark.parametrize("q", [[0.5, 0.6]])
def test_groupby_quantile_with_arraylike_q_and_int_columns(frame_size, groupby, q):
    # GH30289
    nrow, ncol = frame_size
    df = DataFrame(np.array([ncol * [_ % 4] for _ in range(nrow)]), columns=range(ncol))

    idx_levels = [list(range(min(nrow, 4)))] * len(groupby) + [q]
    idx_codes = [[x for x in range(min(nrow, 4)) for _ in q]] * len(groupby) + [
        list(range(len(q))) * min(nrow, 4)
    ]
    expected_index = pd.MultiIndex(
        levels=idx_levels, codes=idx_codes, names=groupby + [None]
    )
    expected_values = [
        [float(x)] * (ncol - len(groupby)) for x in range(min(nrow, 4)) for _ in q
    ]
    expected_columns = [x for x in range(ncol) if x not in groupby]
    expected = DataFrame(
        expected_values, index=expected_index, columns=expected_columns
    )
    result = df.groupby(groupby).quantile(q)

    tm.assert_frame_equal(result, expected)


def test_quantile_raises():
    df = DataFrame([["foo", "a"], ["foo", "b"], ["foo", "c"]], columns=["key", "val"])

    with pytest.raises(TypeError, match="cannot be performed against 'object' dtypes"):
        with tm.assert_produces_warning(
            FutureWarning, match="Dropping invalid columns"
        ):
            df.groupby("key").quantile()


def test_quantile_out_of_bounds_q_raises():
    # https://github.com/pandas-dev/pandas/issues/27470
    df = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    g = df.groupby([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Got '50.0' instead"):
        g.quantile(50)

    with pytest.raises(ValueError, match="Got '-1.0' instead"):
        g.quantile(-1)


def test_quantile_missing_group_values_no_segfaults():
    # GH 28662
    data = np.array([1.0, np.nan, 1.0])
    df = DataFrame({"key": data, "val": range(3)})

    # Random segfaults; would have been guaranteed in loop
    grp = df.groupby("key")
    for _ in range(100):
        grp.quantile()


@pytest.mark.parametrize(
    "key, val, expected_key, expected_val",
    [
        ([1.0, np.nan, 3.0, np.nan], range(4), [1.0, 3.0], [0.0, 2.0]),
        ([1.0, np.nan, 2.0, 2.0], range(4), [1.0, 2.0], [0.0, 2.5]),
        (["a", "b", "b", np.nan], range(4), ["a", "b"], [0, 1.5]),
        ([0], [42], [0], [42.0]),
        ([], [], np.array([], dtype="float64"), np.array([], dtype="float64")),
    ],
)
def test_quantile_missing_group_values_correct_results(
    key, val, expected_key, expected_val
):
    # GH 28662, GH 33200, GH 33569
    df = DataFrame({"key": key, "val": val})

    expected = DataFrame(
        expected_val, index=Index(expected_key, name="key"), columns=["val"]
    )

    grp = df.groupby("key")

    result = grp.quantile(0.5)
    tm.assert_frame_equal(result, expected)

    result = grp.quantile()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.array([1, 0, None] * 2, dtype="Int64"),
        pd.array([True, False, None] * 2, dtype="boolean"),
    ],
)
@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
def test_groupby_quantile_nullable_array(values, q):
    # https://github.com/pandas-dev/pandas/issues/33136
    df = DataFrame({"a": ["x"] * 3 + ["y"] * 3, "b": values})
    result = df.groupby("a")["b"].quantile(q)

    if isinstance(q, list):
        idx = pd.MultiIndex.from_product((["x", "y"], q), names=["a", None])
        true_quantiles = [0.0, 0.5, 1.0]
    else:
        idx = Index(["x", "y"], name="a")
        true_quantiles = [0.5]

    expected = pd.Series(true_quantiles * 2, index=idx, name="b")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
def test_groupby_quantile_skips_invalid_dtype(q):
    df = DataFrame({"a": [1], "b": [2.0], "c": ["x"]})

    warn = None if isinstance(q, list) else FutureWarning
    with tm.assert_produces_warning(warn, match="Dropping invalid columns"):
        result = df.groupby("a").quantile(q)

    expected = df.groupby("a")[["b"]].quantile(q)
    tm.assert_frame_equal(result, expected)


def test_groupby_quantile_NA_float(any_float_allowed_nullable_dtype):
    # GH#42849
    df = DataFrame(
        {"x": [1, 1], "y": [0.2, np.nan]}, dtype=any_float_allowed_nullable_dtype
    )
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series([0.2], dtype=float, index=Index(df["x"][:1]), name="y")
    tm.assert_series_equal(expected, result)

    result = df.groupby("x")["y"].quantile([0.5, 0.75])
    expected = pd.Series(
        [0.2] * 2,
        index=pd.MultiIndex.from_arrays(
            [Index(df["x"]), [0.5, 0.75]], names=["x", None]
        ),
        name="y",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_quantile_NA_int(any_nullable_int_dtype):
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [2, 5]}, dtype=any_nullable_int_dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series([3.5], dtype=float, index=Index(df["x"][:1]), name="y")
    tm.assert_series_equal(expected, result)

    result = df.groupby("x").quantile(0.5)
    expected = DataFrame({"y": 3.5}, index=Index(df["x"][:1]))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Float64", "Float32"])
def test_groupby_quantile_allNA_column(dtype):
    # GH#42849
    df = DataFrame({"x": [1, 1], "y": [pd.NA] * 2}, dtype=dtype)
    result = df.groupby("x")["y"].quantile(0.5)
    expected = pd.Series([np.nan], dtype=float, index=Index(df["x"][:1]), name="y")
    tm.assert_series_equal(expected, result)


def test_groupby_timedelta_quantile():
    # GH: 29485
    df = DataFrame(
        {"value": pd.to_timedelta(np.arange(4), unit="s"), "group": [1, 1, 2, 2]}
    )
    result = df.groupby("group").quantile(0.99)
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),
                pd.Timedelta("0 days 00:00:02.990000"),
            ]
        },
        index=Index([1, 2], name="group"),
    )
    tm.assert_frame_equal(result, expected)


def test_columns_groupby_quantile():
    # GH 33795
    df = DataFrame(
        np.arange(12).reshape(3, -1),
        index=list("XYZ"),
        columns=pd.Series(list("ABAB"), name="col"),
    )
    result = df.groupby("col", axis=1).quantile(q=[0.8, 0.2])
    expected = DataFrame(
        [
            [1.6, 0.4, 2.6, 1.4],
            [5.6, 4.4, 6.6, 5.4],
            [9.6, 8.4, 10.6, 9.4],
        ],
        index=list("XYZ"),
        columns=pd.MultiIndex.from_tuples(
            [("A", 0.8), ("A", 0.2), ("B", 0.8), ("B", 0.2)], names=["col", None]
        ),
    )

    tm.assert_frame_equal(result, expected)
