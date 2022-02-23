""" test with the .transform """
from io import StringIO

import numpy as np
import pytest

from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_timedelta64_dtype,
)

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.core.groupby.generic import (
    DataFrameGroupBy,
    SeriesGroupBy,
)


def assert_fp_equal(a, b):
    assert (np.abs(a - b) < 1e-12).all()


def test_transform():
    data = Series(np.arange(9) // 3, index=np.arange(9))

    index = np.arange(9)
    np.random.shuffle(index)
    data = data.reindex(index)

    grouped = data.groupby(lambda x: x // 3)

    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12

    # GH 8046
    # make sure that we preserve the input order

    df = DataFrame(
        np.arange(6, dtype="int64").reshape(3, 2), columns=["a", "b"], index=[0, 2, 1]
    )
    key = [0, 0, 1]
    expected = (
        df.sort_index()
        .groupby(key)
        .transform(lambda x: x - x.mean())
        .groupby(key)
        .mean()
    )
    result = df.groupby(key).transform(lambda x: x - x.mean()).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    def demean(arr):
        return arr - arr.mean()

    people = DataFrame(
        np.random.randn(5, 5),
        columns=["a", "b", "c", "d", "e"],
        index=["Joe", "Steve", "Wes", "Jim", "Travis"],
    )
    key = ["one", "two", "one", "two", "one"]
    result = people.groupby(key).transform(demean).groupby(key).mean()
    expected = people.groupby(key).apply(demean).groupby(key).mean()
    tm.assert_frame_equal(result, expected)

    # GH 8430
    df = tm.makeTimeDataFrame()
    g = df.groupby(pd.Grouper(freq="M"))
    g.transform(lambda x: x - 1)

    # GH 9700
    df = DataFrame({"a": range(5, 10), "b": range(5)})
    result = df.groupby("a").transform(max)
    expected = DataFrame({"b": range(5)})
    tm.assert_frame_equal(result, expected)


def test_transform_fast():

    df = DataFrame({"id": np.arange(100000) / 3, "val": np.random.randn(100000)})

    grp = df.groupby("id")["val"]

    values = np.repeat(grp.mean().values, ensure_platform_int(grp.count().values))
    expected = Series(values, index=df.index, name="val")

    result = grp.transform(np.mean)
    tm.assert_series_equal(result, expected)

    result = grp.transform("mean")
    tm.assert_series_equal(result, expected)

    # GH 12737
    df = DataFrame(
        {
            "grouping": [0, 1, 1, 3],
            "f": [1.1, 2.1, 3.1, 4.5],
            "d": date_range("2014-1-1", "2014-1-4"),
            "i": [1, 2, 3, 4],
        },
        columns=["grouping", "f", "i", "d"],
    )
    result = df.groupby("grouping").transform("first")

    dates = [
        Timestamp("2014-1-1"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-4"),
    ]
    expected = DataFrame(
        {"f": [1.1, 2.1, 2.1, 4.5], "d": dates, "i": [1, 2, 2, 4]},
        columns=["f", "i", "d"],
    )
    tm.assert_frame_equal(result, expected)

    # selection
    result = df.groupby("grouping")[["f", "i"]].transform("first")
    expected = expected[["f", "i"]]
    tm.assert_frame_equal(result, expected)

    # dup columns
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["g", "a", "a"])
    result = df.groupby("g").transform("first")
    expected = df.drop("g", axis=1)
    tm.assert_frame_equal(result, expected)


def test_transform_broadcast(tsframe, ts):
    grouped = ts.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)

    tm.assert_index_equal(result.index, ts.index)
    for _, gp in grouped:
        assert_fp_equal(result.reindex(gp.index), gp.mean())

    grouped = tsframe.groupby(lambda x: x.month)
    result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    for _, gp in grouped:
        agged = gp.mean()
        res = result.reindex(gp.index)
        for col in tsframe:
            assert_fp_equal(res[col], agged[col])

    # group columns
    grouped = tsframe.groupby({"A": 0, "B": 0, "C": 1, "D": 1}, axis=1)
    result = grouped.transform(np.mean)
    tm.assert_index_equal(result.index, tsframe.index)
    tm.assert_index_equal(result.columns, tsframe.columns)
    for _, gp in grouped:
        agged = gp.mean(1)
        res = result.reindex(columns=gp.columns)
        for idx in gp.index:
            assert_fp_equal(res.xs(idx), agged[idx])


def test_transform_axis_1(request, transformation_func, using_array_manager):
    # GH 36308
    if using_array_manager and transformation_func == "pct_change":
        # TODO(ArrayManager) column-wise shift
        request.node.add_marker(
            pytest.mark.xfail(reason="ArrayManager: shift axis=1 not yet implemented")
        )
    warn = None
    if transformation_func == "tshift":
        warn = FutureWarning

        request.node.add_marker(pytest.mark.xfail(reason="tshift is deprecated"))
    args = ("ffill",) if transformation_func == "fillna" else ()

    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"])
    with tm.assert_produces_warning(warn):
        result = df.groupby([0, 0, 1], axis=1).transform(transformation_func, *args)
        expected = df.T.groupby([0, 0, 1]).transform(transformation_func, *args).T

    if transformation_func == "diff":
        # Result contains nans, so transpose coerces to float
        expected["b"] = expected["b"].astype("int64")

    # cumcount returns Series; the rest are DataFrame
    tm.assert_equal(result, expected)


def test_transform_axis_ts(tsframe):

    # make sure that we are setting the axes
    # correctly when on axis=0 or 1
    # in the presence of a non-monotonic indexer
    # GH12713

    base = tsframe.iloc[0:5]
    r = len(base.index)
    c = len(base.columns)
    tso = DataFrame(
        np.random.randn(r, c), index=base.index, columns=base.columns, dtype="float64"
    )
    # monotonic
    ts = tso
    grouped = ts.groupby(lambda x: x.weekday())
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean())
    tm.assert_frame_equal(result, expected)

    ts = ts.T
    grouped = ts.groupby(lambda x: x.weekday(), axis=1)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: (x.T - x.mean(1)).T)
    tm.assert_frame_equal(result, expected)

    # non-monotonic
    ts = tso.iloc[[1, 0] + list(range(2, len(base)))]
    grouped = ts.groupby(lambda x: x.weekday())
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: x - x.mean())
    tm.assert_frame_equal(result, expected)

    ts = ts.T
    grouped = ts.groupby(lambda x: x.weekday(), axis=1)
    result = ts - grouped.transform("mean")
    expected = grouped.apply(lambda x: (x.T - x.mean(1)).T)
    tm.assert_frame_equal(result, expected)


def test_transform_dtype():
    # GH 9807
    # Check transform dtype output is preserved
    df = DataFrame([[1, 3], [2, 3]])
    result = df.groupby(1).transform("mean")
    expected = DataFrame([[1.5], [1.5]])
    tm.assert_frame_equal(result, expected)


def test_transform_bug():
    # GH 5712
    # transforming on a datetime column
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    result = df.groupby("A")["B"].transform(lambda x: x.rank(ascending=False))
    expected = Series(np.arange(5, 0, step=-1), name="B", dtype="float64")
    tm.assert_series_equal(result, expected)


def test_transform_numeric_to_boolean():
    # GH 16875
    # inconsistency in transforming boolean values
    expected = Series([True, True], name="A")

    df = DataFrame({"A": [1.1, 2.2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)

    df = DataFrame({"A": [1, 2], "B": [1, 2]})
    result = df.groupby("B").A.transform(lambda x: True)
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_timedelta():
    # GH 15429
    # transforming a datetime to timedelta
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    expected = Series([Timestamp("20130101") - Timestamp("20130101")] * 5, name="A")

    # this does date math without changing result type in transform
    base_time = df["A"][0]
    result = (
        df.groupby("A")["A"].transform(lambda x: x.max() - x.min() + base_time)
        - base_time
    )
    tm.assert_series_equal(result, expected)

    # this does date math and causes the transform to return timedelta
    result = df.groupby("A")["A"].transform(lambda x: x.max() - x.min())
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_numeric():
    # GH 10972
    # convert dt to float
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.mean()
    )

    expected = Series([-0.5, 0.5], name="b")
    tm.assert_series_equal(result, expected)

    # convert dt to int
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.min()
    )

    expected = Series([0, 1], name="b")
    tm.assert_series_equal(result, expected)


def test_transform_casting():
    # 13046
    data = """
    idx     A         ID3              DATETIME
    0   B-028  b76cd912ff "2014-10-08 13:43:27"
    1   B-054  4a57ed0b02 "2014-10-08 14:26:19"
    2   B-076  1a682034f8 "2014-10-08 14:29:01"
    3   B-023  b76cd912ff "2014-10-08 18:39:34"
    4   B-023  f88g8d7sds "2014-10-08 18:40:18"
    5   B-033  b76cd912ff "2014-10-08 18:44:30"
    6   B-032  b76cd912ff "2014-10-08 18:46:00"
    7   B-037  b76cd912ff "2014-10-08 18:52:15"
    8   B-046  db959faf02 "2014-10-08 18:59:59"
    9   B-053  b76cd912ff "2014-10-08 19:17:48"
    10  B-065  b76cd912ff "2014-10-08 19:21:38"
    """
    df = pd.read_csv(
        StringIO(data), sep=r"\s+", index_col=[0], parse_dates=["DATETIME"]
    )

    result = df.groupby("ID3")["DATETIME"].transform(lambda x: x.diff())
    assert is_timedelta64_dtype(result.dtype)

    result = df[["ID3", "DATETIME"]].groupby("ID3").transform(lambda x: x.diff())
    assert is_timedelta64_dtype(result.DATETIME.dtype)


def test_transform_multiple(ts):
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])

    grouped.transform(lambda x: x * 2)
    grouped.transform(np.mean)


def test_dispatch_transform(tsframe):
    df = tsframe[::5].reindex(tsframe.index)

    grouped = df.groupby(lambda x: x.month)

    filled = grouped.fillna(method="pad")
    fillit = lambda x: x.fillna(method="pad")
    expected = df.groupby(lambda x: x.month).transform(fillit)
    tm.assert_frame_equal(filled, expected)


def test_transform_transformation_func(request, transformation_func):
    # GH 30918
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "bar", "bar", "baz"],
            "B": [1, 2, np.nan, 3, 3, np.nan, 4],
        },
        index=date_range("2020-01-01", "2020-01-07"),
    )

    if transformation_func == "cumcount":
        test_op = lambda x: x.transform("cumcount")
        mock_op = lambda x: Series(range(len(x)), x.index)
    elif transformation_func == "fillna":
        test_op = lambda x: x.transform("fillna", value=0)
        mock_op = lambda x: x.fillna(value=0)
    elif transformation_func == "tshift":
        msg = (
            "Current behavior of groupby.tshift is inconsistent with other "
            "transformations. See GH34452 for more details"
        )
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    else:
        test_op = lambda x: x.transform(transformation_func)
        mock_op = lambda x: getattr(x, transformation_func)()

    result = test_op(df.groupby("A"))
    groups = [df[["B"]].iloc[:4], df[["B"]].iloc[4:6], df[["B"]].iloc[6:]]
    expected = concat([mock_op(g) for g in groups])

    if transformation_func == "cumcount":
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_frame_equal(result, expected)


def test_transform_select_columns(df):
    f = lambda x: x.mean()
    result = df.groupby("A")[["C", "D"]].transform(f)

    selection = df[["C", "D"]]
    expected = selection.groupby(df["A"]).transform(f)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("duplicates", [True, False])
def test_transform_exclude_nuisance(df, duplicates):
    # case that goes through _transform_item_by_item

    if duplicates:
        # make sure we work with duplicate columns GH#41427
        df.columns = ["A", "C", "C", "D"]

    # this also tests orderings in transform between
    # series/frame to make sure it's consistent
    expected = {}
    grouped = df.groupby("A")

    gbc = grouped["C"]
    warn = FutureWarning if duplicates else None
    with tm.assert_produces_warning(warn, match="Dropping invalid columns"):
        expected["C"] = gbc.transform(np.mean)
    if duplicates:
        # squeeze 1-column DataFrame down to Series
        expected["C"] = expected["C"]["C"]

        assert isinstance(gbc.obj, DataFrame)
        assert isinstance(gbc, DataFrameGroupBy)
    else:
        assert isinstance(gbc, SeriesGroupBy)
        assert isinstance(gbc.obj, Series)

    expected["D"] = grouped["D"].transform(np.mean)
    expected = DataFrame(expected)
    with tm.assert_produces_warning(FutureWarning, match="Dropping invalid columns"):
        result = df.groupby("A").transform(np.mean)

    tm.assert_frame_equal(result, expected)


def test_transform_function_aliases(df):
    with tm.assert_produces_warning(FutureWarning, match="Dropping invalid columns"):
        result = df.groupby("A").transform("mean")
        expected = df.groupby("A").transform(np.mean)
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A")["C"].transform("mean")
    expected = df.groupby("A")["C"].transform(np.mean)
    tm.assert_series_equal(result, expected)


def test_series_fast_transform_date():
    # GH 13191
    df = DataFrame(
        {"grouping": [np.nan, 1, 1, 3], "d": date_range("2014-1-1", "2014-1-4")}
    )
    result = df.groupby("grouping")["d"].transform("first")
    dates = [
        pd.NaT,
        Timestamp("2014-1-2"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-4"),
    ]
    expected = Series(dates, name="d")
    tm.assert_series_equal(result, expected)


def test_transform_length():
    # GH 9697
    df = DataFrame({"col1": [1, 1, 2, 2], "col2": [1, 2, 3, np.nan]})
    expected = Series([3.0] * 4)

    def nsum(x):
        return np.nansum(x)

    results = [
        df.groupby("col1").transform(sum)["col2"],
        df.groupby("col1")["col2"].transform(sum),
        df.groupby("col1").transform(nsum)["col2"],
        df.groupby("col1")["col2"].transform(nsum),
    ]
    for result in results:
        tm.assert_series_equal(result, expected, check_names=False)


def test_transform_coercion():

    # 14457
    # when we are transforming be sure to not coerce
    # via assignment
    df = DataFrame({"A": ["a", "a"], "B": [0, 1]})
    g = df.groupby("A")

    expected = g.transform(np.mean)
    result = g.transform(lambda x: np.mean(x))
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_int():

    # GH 3740, make sure that we might upcast on item-by-item transform

    # floats
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": Series(1, dtype="float64"),
            "C": Series([1, 2, 3, 1, 2, 3], dtype="float64"),
            "D": "foo",
        }
    )
    with np.errstate(all="ignore"):
        with tm.assert_produces_warning(
            FutureWarning, match="Dropping invalid columns"
        ):
            result = df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
    expected = DataFrame(
        {"B": np.nan, "C": Series([-1, 0, 1, -1, 0, 1], dtype="float64")}
    )
    tm.assert_frame_equal(result, expected)

    # int case
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": 1,
            "C": [1, 2, 3, 1, 2, 3],
            "D": "foo",
        }
    )
    with np.errstate(all="ignore"):
        with tm.assert_produces_warning(
            FutureWarning, match="Dropping invalid columns"
        ):
            result = df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
    expected = DataFrame({"B": np.nan, "C": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    tm.assert_frame_equal(result, expected)

    # int that needs float conversion
    s = Series([2, 3, 4, 10, 5, -1])
    df = DataFrame({"A": [1, 1, 1, 2, 2, 2], "B": 1, "C": s, "D": "foo"})
    with np.errstate(all="ignore"):
        with tm.assert_produces_warning(
            FutureWarning, match="Dropping invalid columns"
        ):
            result = df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())

    s1 = s.iloc[0:3]
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = s.iloc[3:6]
    s2 = (s2 - s2.mean()) / s2.std()
    expected = DataFrame({"B": np.nan, "C": concat([s1, s2])})
    tm.assert_frame_equal(result, expected)

    # int doesn't get downcasted
    with tm.assert_produces_warning(FutureWarning, match="Dropping invalid columns"):
        result = df.groupby("A").transform(lambda x: x * 2 / 2)
    expected = DataFrame({"B": 1.0, "C": [2.0, 3.0, 4.0, 10.0, 5.0, -1.0]})
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_nan_group():
    # GH 9941
    df = DataFrame({"a": range(10), "b": [1, 1, 2, 3, np.nan, 4, 4, 5, 5, 5]})
    result = df.groupby(df.b)["a"].transform(max)
    expected = Series([1.0, 1.0, 2.0, 3.0, np.nan, 6.0, 6.0, 9.0, 9.0, 9.0], name="a")
    tm.assert_series_equal(result, expected)


def test_transform_mixed_type():
    index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "c": np.tile(["a", "b", "c"], 2),
            "v": np.arange(1.0, 7.0),
        },
        index=index,
    )

    def f(group):
        group["g"] = group["d"] * 2
        return group[:1]

    grouped = df.groupby("c")
    result = grouped.apply(f)

    assert result["d"].dtype == np.float64

    # this is by definition a mutating operation!
    with pd.option_context("mode.chained_assignment", None):
        for key, group in grouped:
            res = f(group)
            tm.assert_frame_equal(res, result.loc[key])


@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),
        ("cumsum", (), lambda x: x.cumsum()),
        ("shift", (-1,), lambda x: x.shift(-1)),
        ("shift", (1,), lambda x: x.shift()),
    ],
)
def test_cython_transform_series(op, args, targop):
    # GH 4095
    s = Series(np.random.randn(1000))
    s_missing = s.copy()
    s_missing.iloc[2:10] = np.nan
    labels = np.random.randint(0, 50, size=1000).astype(float)

    # series
    for data in [s, s_missing]:
        # print(data.head())
        expected = data.groupby(labels).transform(targop)

        tm.assert_series_equal(expected, data.groupby(labels).transform(op, *args))
        tm.assert_series_equal(expected, getattr(data.groupby(labels), op)(*args))


@pytest.mark.parametrize("op", ["cumprod", "cumsum"])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize(
    "input, exp",
    [
        # When everything is NaN
        ({"key": ["b"] * 10, "value": np.nan}, Series([np.nan] * 10, name="value")),
        # When there is a single NaN
        (
            {"key": ["b"] * 10 + ["a"] * 2, "value": [3] * 3 + [np.nan] + [3] * 8},
            {
                ("cumprod", False): [3.0, 9.0, 27.0] + [np.nan] * 7 + [3.0, 9.0],
                ("cumprod", True): [
                    3.0,
                    9.0,
                    27.0,
                    np.nan,
                    81.0,
                    243.0,
                    729.0,
                    2187.0,
                    6561.0,
                    19683.0,
                    3.0,
                    9.0,
                ],
                ("cumsum", False): [3.0, 6.0, 9.0] + [np.nan] * 7 + [3.0, 6.0],
                ("cumsum", True): [
                    3.0,
                    6.0,
                    9.0,
                    np.nan,
                    12.0,
                    15.0,
                    18.0,
                    21.0,
                    24.0,
                    27.0,
                    3.0,
                    6.0,
                ],
            },
        ),
    ],
)
def test_groupby_cum_skipna(op, skipna, input, exp):
    df = DataFrame(input)
    result = df.groupby("key")["value"].transform(op, skipna=skipna)
    if isinstance(exp, dict):
        expected = exp[(op, skipna)]
    else:
        expected = exp
    expected = Series(expected, name="value")
    tm.assert_series_equal(expected, result)


@pytest.mark.slow
@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),
        ("cumsum", (), lambda x: x.cumsum()),
        ("shift", (-1,), lambda x: x.shift(-1)),
        ("shift", (1,), lambda x: x.shift()),
    ],
)
def test_cython_transform_frame(op, args, targop):
    s = Series(np.random.randn(1000))
    s_missing = s.copy()
    s_missing.iloc[2:10] = np.nan
    labels = np.random.randint(0, 50, size=1000).astype(float)
    strings = list("qwertyuiopasdfghjklz")
    strings_missing = strings[:]
    strings_missing[5] = np.nan
    df = DataFrame(
        {
            "float": s,
            "float_missing": s_missing,
            "int": [1, 1, 1, 1, 2] * 200,
            "datetime": date_range("1990-1-1", periods=1000),
            "timedelta": pd.timedelta_range(1, freq="s", periods=1000),
            "string": strings * 50,
            "string_missing": strings_missing * 50,
        },
        columns=[
            "float",
            "float_missing",
            "int",
            "datetime",
            "timedelta",
            "string",
            "string_missing",
        ],
    )
    df["cat"] = df["string"].astype("category")

    df2 = df.copy()
    df2.index = MultiIndex.from_product([range(100), range(10)])

    # DataFrame - Single and MultiIndex,
    # group by values, index level, columns
    for df in [df, df2]:
        for gb_target in [
            {"by": labels},
            {"level": 0},
            {"by": "string"},
        ]:  # {"by": 'string_missing'}]:
            # {"by": ['int','string']}]:

            gb = df.groupby(**gb_target)
            # allowlisted methods set the selection before applying
            # bit a of hack to make sure the cythonized shift
            # is equivalent to pre 0.17.1 behavior
            if op == "shift":
                gb._set_group_selection()

            if op != "shift" and "int" not in gb_target:
                # numeric apply fastpath promotes dtype so have
                # to apply separately and concat
                i = gb[["int"]].apply(targop)
                f = gb[["float", "float_missing"]].apply(targop)
                expected = concat([f, i], axis=1)
            else:
                expected = gb.apply(targop)

            expected = expected.sort_index(axis=1)
            tm.assert_frame_equal(expected, gb.transform(op, *args).sort_index(axis=1))
            tm.assert_frame_equal(expected, getattr(gb, op)(*args).sort_index(axis=1))
            # individual columns
            for c in df:
                if (
                    c not in ["float", "int", "float_missing"]
                    and op != "shift"
                    and not (c == "timedelta" and op == "cumsum")
                ):
                    msg = "|".join(
                        [
                            "does not support .* operations",
                            ".* is not supported for object dtype",
                            "is not implemented for this dtype",
                        ]
                    )
                    with pytest.raises(TypeError, match=msg):
                        gb[c].transform(op)
                    with pytest.raises(TypeError, match=msg):
                        getattr(gb[c], op)()
                else:
                    expected = gb[c].apply(targop)
                    expected.name = c
                    tm.assert_series_equal(expected, gb[c].transform(op, *args))
                    tm.assert_series_equal(expected, getattr(gb[c], op)(*args))


def test_transform_with_non_scalar_group():
    # GH 10165
    cols = MultiIndex.from_tuples(
        [
            ("syn", "A"),
            ("mis", "A"),
            ("non", "A"),
            ("syn", "C"),
            ("mis", "C"),
            ("non", "C"),
            ("syn", "T"),
            ("mis", "T"),
            ("non", "T"),
            ("syn", "G"),
            ("mis", "G"),
            ("non", "G"),
        ]
    )
    df = DataFrame(
        np.random.randint(1, 10, (4, 12)), columns=cols, index=["A", "C", "G", "T"]
    )

    msg = "transform must return a scalar value for each group.*"
    with pytest.raises(ValueError, match=msg):
        df.groupby(axis=1, level=1).transform(lambda z: z.div(z.sum(axis=1), axis=0))


@pytest.mark.parametrize(
    "cols,exp,comp_func",
    [
        ("a", Series([1, 1, 1], name="a"), tm.assert_series_equal),
        (
            ["a", "c"],
            DataFrame({"a": [1, 1, 1], "c": [1, 1, 1]}),
            tm.assert_frame_equal,
        ),
    ],
)
@pytest.mark.parametrize("agg_func", ["count", "rank", "size"])
def test_transform_numeric_ret(cols, exp, comp_func, agg_func, request):
    if agg_func == "size" and isinstance(cols, list):
        # https://github.com/pytest-dev/pytest/issues/6300
        # workaround to xfail fixture/param permutations
        reason = "'size' transformation not supported with NDFrameGroupy"
        request.node.add_marker(pytest.mark.xfail(reason=reason))

    # GH 19200
    df = DataFrame(
        {"a": date_range("2018-01-01", periods=3), "b": range(3), "c": range(7, 10)}
    )

    warn = FutureWarning
    if isinstance(exp, Series) or agg_func != "size":
        warn = None
    with tm.assert_produces_warning(warn, match="Dropping invalid columns"):
        result = df.groupby("b")[cols].transform(agg_func)

    if agg_func == "rank":
        exp = exp.astype("float")

    comp_func(result, exp)


def test_transform_ffill():
    # GH 24211
    data = [["a", 0.0], ["a", float("nan")], ["b", 1.0], ["b", float("nan")]]
    df = DataFrame(data, columns=["key", "values"])
    result = df.groupby("key").transform("ffill")
    expected = DataFrame({"values": [0.0, 0.0, 1.0, 1.0]})
    tm.assert_frame_equal(result, expected)
    result = df.groupby("key")["values"].transform("ffill")
    expected = Series([0.0, 0.0, 1.0, 1.0], name="values")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("mix_groupings", [True, False])
@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize("val1,val2", [("foo", "bar"), (1, 2), (1.0, 2.0)])
@pytest.mark.parametrize(
    "fill_method,limit,exp_vals",
    [
        (
            "ffill",
            None,
            [np.nan, np.nan, "val1", "val1", "val1", "val2", "val2", "val2"],
        ),
        ("ffill", 1, [np.nan, np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan]),
        (
            "bfill",
            None,
            ["val1", "val1", "val1", "val2", "val2", "val2", np.nan, np.nan],
        ),
        ("bfill", 1, [np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan, np.nan]),
    ],
)
def test_group_fill_methods(
    mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals
):
    vals = [np.nan, np.nan, val1, np.nan, np.nan, val2, np.nan, np.nan]
    _exp_vals = list(exp_vals)
    # Overwrite placeholder values
    for index, exp_val in enumerate(_exp_vals):
        if exp_val == "val1":
            _exp_vals[index] = val1
        elif exp_val == "val2":
            _exp_vals[index] = val2

    # Need to modify values and expectations depending on the
    # Series / DataFrame that we ultimately want to generate
    if mix_groupings:  # ['a', 'b', 'a, 'b', ...]
        keys = ["a", "b"] * len(vals)

        def interweave(list_obj):
            temp = []
            for x in list_obj:
                temp.extend([x, x])

            return temp

        _exp_vals = interweave(_exp_vals)
        vals = interweave(vals)
    else:  # ['a', 'a', 'a', ... 'b', 'b', 'b']
        keys = ["a"] * len(vals) + ["b"] * len(vals)
        _exp_vals = _exp_vals * 2
        vals = vals * 2

    df = DataFrame({"key": keys, "val": vals})
    if as_series:
        result = getattr(df.groupby("key")["val"], fill_method)(limit=limit)
        exp = Series(_exp_vals, name="val")
        tm.assert_series_equal(result, exp)
    else:
        result = getattr(df.groupby("key"), fill_method)(limit=limit)
        exp = DataFrame({"val": _exp_vals})
        tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize("fill_method", ["ffill", "bfill"])
def test_pad_stable_sorting(fill_method):
    # GH 21207
    x = [0] * 20
    y = [np.nan] * 10 + [1] * 10

    if fill_method == "bfill":
        y = y[::-1]

    df = DataFrame({"x": x, "y": y})
    expected = df.drop("x", axis=1)

    result = getattr(df.groupby("x"), fill_method)()

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize(
    "freq",
    [
        None,
        pytest.param(
            "D",
            marks=pytest.mark.xfail(
                reason="GH#23918 before method uses freq in vectorized approach"
            ),
        ),
    ],
)
@pytest.mark.parametrize("periods", [1, -1])
@pytest.mark.parametrize("fill_method", ["ffill", "bfill", None])
@pytest.mark.parametrize("limit", [None, 1])
def test_pct_change(test_series, freq, periods, fill_method, limit):
    # GH  21200, 21621, 30463
    vals = [3, np.nan, np.nan, np.nan, 1, 2, 4, 10, np.nan, 4]
    keys = ["a", "b"]
    key_v = np.repeat(keys, len(vals))
    df = DataFrame({"key": key_v, "vals": vals * 2})

    df_g = df
    if fill_method is not None:
        df_g = getattr(df.groupby("key"), fill_method)(limit=limit)
    grp = df_g.groupby(df.key)

    expected = grp["vals"].obj / grp["vals"].shift(periods) - 1

    if test_series:
        result = df.groupby("key")["vals"].pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq
        )
        tm.assert_series_equal(result, expected)
    else:
        result = df.groupby("key").pct_change(
            periods=periods, fill_method=fill_method, limit=limit, freq=freq
        )
        tm.assert_frame_equal(result, expected.to_frame("vals"))


@pytest.mark.parametrize(
    "func, expected_status",
    [
        ("ffill", ["shrt", "shrt", "lng", np.nan, "shrt", "ntrl", "ntrl"]),
        ("bfill", ["shrt", "lng", "lng", "shrt", "shrt", "ntrl", np.nan]),
    ],
)
def test_ffill_bfill_non_unique_multilevel(func, expected_status):
    # GH 19437
    date = pd.to_datetime(
        [
            "2018-01-01",
            "2018-01-01",
            "2018-01-01",
            "2018-01-01",
            "2018-01-02",
            "2018-01-01",
            "2018-01-02",
        ]
    )
    symbol = ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL", "TSLA", "TSLA"]
    status = ["shrt", np.nan, "lng", np.nan, "shrt", "ntrl", np.nan]

    df = DataFrame({"date": date, "symbol": symbol, "status": status})
    df = df.set_index(["date", "symbol"])
    result = getattr(df.groupby("symbol")["status"], func)()

    index = MultiIndex.from_tuples(
        tuples=list(zip(*[date, symbol])), names=["date", "symbol"]
    )
    expected = Series(expected_status, index=index, name="status")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [np.any, np.all])
def test_any_all_np_func(func):
    # GH 20653
    df = DataFrame(
        [["foo", True], [np.nan, True], ["foo", True]], columns=["key", "val"]
    )

    exp = Series([True, np.nan, True], name="val")

    res = df.groupby("key")["val"].transform(func)
    tm.assert_series_equal(res, exp)


def test_groupby_transform_rename():
    # https://github.com/pandas-dev/pandas/issues/23461
    def demean_rename(x):
        result = x - x.mean()

        if isinstance(x, Series):
            return result

        result = result.rename(columns={c: "{c}_demeaned" for c in result.columns})

        return result

    df = DataFrame({"group": list("ababa"), "value": [1, 1, 1, 2, 2]})
    expected = DataFrame({"value": [-1.0 / 3, -0.5, -1.0 / 3, 0.5, 2.0 / 3]})

    result = df.groupby("group").transform(demean_rename)
    tm.assert_frame_equal(result, expected)
    result_single = df.groupby("group").value.transform(demean_rename)
    tm.assert_series_equal(result_single, expected["value"])


@pytest.mark.parametrize("func", [min, max, np.min, np.max, "first", "last"])
def test_groupby_transform_timezone_column(func):
    # GH 24198
    ts = pd.to_datetime("now", utc=True).tz_convert("Asia/Singapore")
    result = DataFrame({"end_time": [ts], "id": [1]})
    result["max_end_time"] = result.groupby("id").end_time.transform(func)
    expected = DataFrame([[ts, 1, ts]], columns=["end_time", "id", "max_end_time"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"]),
        ("idxmax", ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"]),
    ],
)
def test_groupby_transform_with_datetimes(func, values):
    # GH 15306
    dates = date_range("1/1/2011", periods=10, freq="D")

    stocks = DataFrame({"price": np.arange(10.0)}, index=dates)
    stocks["week_id"] = dates.isocalendar().week

    result = stocks.groupby(stocks["week_id"])["price"].transform(func)

    expected = Series(data=pd.to_datetime(values), index=dates, name="price")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["cumsum", "cumprod", "cummin", "cummax"])
def test_transform_absent_categories(func):
    # GH 16771
    # cython transforms with more groups than rows
    x_vals = [1]
    x_cats = range(2)
    y = [1]
    df = DataFrame({"x": Categorical(x_vals, x_cats), "y": y})
    result = getattr(df.y.groupby(df.x), func)()
    expected = df.y
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["ffill", "bfill", "shift"])
@pytest.mark.parametrize("key, val", [("level", 0), ("by", Series([0]))])
def test_ffill_not_in_axis(func, key, val):
    # GH 21521
    df = DataFrame([[np.nan]])
    result = getattr(df.groupby(**{key: val}), func)()
    expected = df

    tm.assert_frame_equal(result, expected)


def test_transform_invalid_name_raises():
    # GH#27486
    df = DataFrame({"a": [0, 1, 1, 2]})
    g = df.groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("some_arbitrary_name")

    # method exists on the object, but is not a valid transformation/agg
    assert hasattr(g, "aggregate")  # make sure the method exists
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("aggregate")

    # Test SeriesGroupBy
    g = df["a"].groupby(["a", "b", "b", "c"])
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("some_arbitrary_name")


@pytest.mark.parametrize(
    "obj",
    [
        DataFrame(
            {"a": [0, 0, 0, 1, 1, 1], "b": range(6)},
            index=["A", "B", "C", "D", "E", "F"],
        ),
        Series([0, 0, 0, 1, 1, 1], index=["A", "B", "C", "D", "E", "F"]),
    ],
)
def test_transform_agg_by_name(request, reduction_func, obj):
    func = reduction_func
    g = obj.groupby(np.repeat([0, 1], 3))

    if func == "ngroup":  # GH#27468
        request.node.add_marker(
            pytest.mark.xfail(reason="TODO: g.transform('ngroup') doesn't work")
        )
    if func == "size" and obj.ndim == 2:  # GH#27469
        request.node.add_marker(
            pytest.mark.xfail(reason="TODO: g.transform('size') doesn't work")
        )
    if func == "corrwith" and isinstance(obj, Series):  # GH#32293
        request.node.add_marker(
            pytest.mark.xfail(reason="TODO: implement SeriesGroupBy.corrwith")
        )

    args = {"nth": [0], "quantile": [0.5], "corrwith": [obj]}.get(func, [])

    warn = None
    if isinstance(obj, DataFrame) and func == "size":
        warn = FutureWarning

    with tm.assert_produces_warning(warn, match="Dropping invalid columns"):
        result = g.transform(func, *args)

    # this is the *definition* of a transformation
    tm.assert_index_equal(result.index, obj.index)
    if hasattr(obj, "columns"):
        tm.assert_index_equal(result.columns, obj.columns)

    # verify that values were broadcasted across each group
    assert len(set(DataFrame(result).iloc[-3:, -1])) == 1


def test_transform_lambda_with_datetimetz():
    # GH 27496
    df = DataFrame(
        {
            "time": [
                Timestamp("2010-07-15 03:14:45"),
                Timestamp("2010-11-19 18:47:06"),
            ],
            "timezone": ["Etc/GMT+4", "US/Eastern"],
        }
    )
    result = df.groupby(["timezone"])["time"].transform(
        lambda x: x.dt.tz_localize(x.name)
    )
    expected = Series(
        [
            Timestamp("2010-07-15 03:14:45", tz="Etc/GMT+4"),
            Timestamp("2010-11-19 18:47:06", tz="US/Eastern"),
        ],
        name="time",
    )
    tm.assert_series_equal(result, expected)


def test_transform_fastpath_raises():
    # GH#29631 case where fastpath defined in groupby.generic _choose_path
    #  raises, but slow_path does not

    df = DataFrame({"A": [1, 1, 2, 2], "B": [1, -1, 1, 2]})
    gb = df.groupby("A")

    def func(grp):
        # we want a function such that func(frame) fails but func.apply(frame)
        #  works
        if grp.ndim == 2:
            # Ensure that fast_path fails
            raise NotImplementedError("Don't cross the streams")
        return grp * 2

    # Check that the fastpath raises, see _transform_general
    obj = gb._obj_with_exclusions
    gen = gb.grouper.get_iterator(obj, axis=gb.axis)
    fast_path, slow_path = gb._define_paths(func)
    _, group = next(gen)

    with pytest.raises(NotImplementedError, match="Don't cross the streams"):
        fast_path(group)

    result = gb.transform(func)

    expected = DataFrame([2, -2, 2, 4], columns=["B"])
    tm.assert_frame_equal(result, expected)


def test_transform_lambda_indexing():
    # GH 7883
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "flux", "foo", "flux"],
            "B": ["one", "one", "two", "three", "two", "six", "five", "three"],
            "C": range(8),
            "D": range(8),
            "E": range(8),
        }
    )
    df = df.set_index(["A", "B"])
    df = df.sort_index()
    result = df.groupby(level="A").transform(lambda x: x.iloc[-1])
    expected = DataFrame(
        {
            "C": [3, 3, 7, 7, 4, 4, 4, 4],
            "D": [3, 3, 7, 7, 4, 4, 4, 4],
            "E": [3, 3, 7, 7, 4, 4, 4, 4],
        },
        index=MultiIndex.from_tuples(
            [
                ("bar", "one"),
                ("bar", "three"),
                ("flux", "six"),
                ("flux", "three"),
                ("foo", "five"),
                ("foo", "one"),
                ("foo", "two"),
                ("foo", "two"),
            ],
            names=["A", "B"],
        ),
    )
    tm.assert_frame_equal(result, expected)


def test_categorical_and_not_categorical_key(observed):
    # Checks that groupby-transform, when grouping by both a categorical
    # and a non-categorical key, doesn't try to expand the output to include
    # non-observed categories but instead matches the input shape.
    # GH 32494
    df_with_categorical = DataFrame(
        {
            "A": Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "B": [1, 2, 3],
            "C": ["a", "b", "a"],
        }
    )
    df_without_categorical = DataFrame(
        {"A": ["a", "b", "a"], "B": [1, 2, 3], "C": ["a", "b", "a"]}
    )

    # DataFrame case
    result = df_with_categorical.groupby(["A", "C"], observed=observed).transform("sum")
    expected = df_without_categorical.groupby(["A", "C"]).transform("sum")
    tm.assert_frame_equal(result, expected)
    expected_explicit = DataFrame({"B": [4, 2, 4]})
    tm.assert_frame_equal(result, expected_explicit)

    # Series case
    result = df_with_categorical.groupby(["A", "C"], observed=observed)["B"].transform(
        "sum"
    )
    expected = df_without_categorical.groupby(["A", "C"])["B"].transform("sum")
    tm.assert_series_equal(result, expected)
    expected_explicit = Series([4, 2, 4], name="B")
    tm.assert_series_equal(result, expected_explicit)


def test_string_rank_grouping():
    # GH 19354
    df = DataFrame({"A": [1, 1, 2], "B": [1, 2, 3]})
    result = df.groupby("A").transform("rank")
    expected = DataFrame({"B": [1.0, 2.0, 1.0]})
    tm.assert_frame_equal(result, expected)
