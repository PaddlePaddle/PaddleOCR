import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm


def test_first_last_nth(df):
    # tests for first / last / nth
    grouped = df.groupby("A")
    first = grouped.first()
    expected = df.loc[[1, 0], ["B", "C", "D"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)

    nth = grouped.nth(0)
    tm.assert_frame_equal(nth, expected)

    last = grouped.last()
    expected = df.loc[[5, 7], ["B", "C", "D"]]
    expected.index = Index(["bar", "foo"], name="A")
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(-1)
    tm.assert_frame_equal(nth, expected)

    nth = grouped.nth(1)
    expected = df.loc[[2, 3], ["B", "C", "D"]].copy()
    expected.index = Index(["foo", "bar"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(nth, expected)

    # it works!
    grouped["B"].first()
    grouped["B"].last()
    grouped["B"].nth(0)

    df.loc[df["A"] == "foo", "B"] = np.nan
    assert isna(grouped["B"].first()["foo"])
    assert isna(grouped["B"].last()["foo"])
    assert isna(grouped["B"].nth(0)["foo"])

    # v0.14.0 whatsnew
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")
    result = g.first()
    expected = df.iloc[[1, 2]].set_index("A")
    tm.assert_frame_equal(result, expected)

    expected = df.iloc[[1, 2]].set_index("A")
    result = g.nth(0, dropna="any")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_na_object(method, nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32123
    groups = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]}).groupby("a")
    result = getattr(groups, method)()

    if method == "first":
        values = [1, 3]
    else:
        values = [2, 3]

    values = np.array(values, dtype=result["b"].dtype)
    idx = Index([1, 2], name="a")
    expected = DataFrame({"b": values}, index=idx)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("index", [0, -1])
def test_nth_with_na_object(index, nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32123
    groups = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]}).groupby("a")
    result = groups.nth(index)

    if index == 0:
        values = [1, 3]
    else:
        values = [2, nulls_fixture]

    values = np.array(values, dtype=result["b"].dtype)
    idx = Index([1, 2], name="a")
    expected = DataFrame({"b": values}, index=idx)

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_None(method):
    # https://github.com/pandas-dev/pandas/issues/32800
    # None should be preserved as object dtype
    df = DataFrame.from_dict({"id": ["a"], "value": [None]})
    groups = df.groupby("id", as_index=False)
    result = getattr(groups, method)()

    tm.assert_frame_equal(result, df)


@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize(
    "df, expected",
    [
        (
            DataFrame({"id": "a", "value": [None, "foo", np.nan]}),
            DataFrame({"value": ["foo"]}, index=Index(["a"], name="id")),
        ),
        (
            DataFrame({"id": "a", "value": [np.nan]}, dtype=object),
            DataFrame({"value": [None]}, index=Index(["a"], name="id")),
        ),
    ],
)
def test_first_last_with_None_expanded(method, df, expected):
    # GH 32800, 38286
    result = getattr(df.groupby("id"), method)()
    tm.assert_frame_equal(result, expected)


def test_first_last_nth_dtypes(df_mixed_floats):

    df = df_mixed_floats.copy()
    df["E"] = True
    df["F"] = 1

    # tests for first / last / nth
    grouped = df.groupby("A")
    first = grouped.first()
    expected = df.loc[[1, 0], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(first, expected)

    last = grouped.last()
    expected = df.loc[[5, 7], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(1)
    expected = df.loc[[3, 2], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    tm.assert_frame_equal(nth, expected)

    # GH 2763, first/last shifting dtypes
    idx = list(range(10))
    idx.append(9)
    s = Series(data=range(11), index=idx, name="IntCol")
    assert s.dtype == "int64"
    f = s.groupby(level=0).first()
    assert f.dtype == "int64"


def test_first_last_nth_nan_dtype():
    # GH 33591
    df = DataFrame({"data": ["A"], "nans": Series([np.nan], dtype=object)})

    grouped = df.groupby("data")
    expected = df.set_index("data").nans
    tm.assert_series_equal(grouped.nans.first(), expected)
    tm.assert_series_equal(grouped.nans.last(), expected)
    tm.assert_series_equal(grouped.nans.nth(-1), expected)
    tm.assert_series_equal(grouped.nans.nth(0), expected)


def test_first_strings_timestamps():
    # GH 11244
    test = DataFrame(
        {
            Timestamp("2012-01-01 00:00:00"): ["a", "b"],
            Timestamp("2012-01-02 00:00:00"): ["c", "d"],
            "name": ["e", "e"],
            "aaaa": ["f", "g"],
        }
    )
    result = test.groupby("name").first()
    expected = DataFrame(
        [["a", "c", "f"]],
        columns=Index([Timestamp("2012-01-01"), Timestamp("2012-01-02"), "aaaa"]),
        index=Index(["e"], name="name"),
    )
    tm.assert_frame_equal(result, expected)


def test_nth():
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")

    tm.assert_frame_equal(g.nth(0), df.iloc[[0, 2]].set_index("A"))
    tm.assert_frame_equal(g.nth(1), df.iloc[[1]].set_index("A"))
    tm.assert_frame_equal(g.nth(2), df.loc[[]].set_index("A"))
    tm.assert_frame_equal(g.nth(-1), df.iloc[[1, 2]].set_index("A"))
    tm.assert_frame_equal(g.nth(-2), df.iloc[[0]].set_index("A"))
    tm.assert_frame_equal(g.nth(-3), df.loc[[]].set_index("A"))
    tm.assert_series_equal(g.B.nth(0), df.set_index("A").B.iloc[[0, 2]])
    tm.assert_series_equal(g.B.nth(1), df.set_index("A").B.iloc[[1]])
    tm.assert_frame_equal(g[["B"]].nth(0), df.loc[[0, 2], ["A", "B"]].set_index("A"))

    exp = df.set_index("A")
    tm.assert_frame_equal(g.nth(0, dropna="any"), exp.iloc[[1, 2]])
    tm.assert_frame_equal(g.nth(-1, dropna="any"), exp.iloc[[1, 2]])

    exp["B"] = np.nan
    tm.assert_frame_equal(g.nth(7, dropna="any"), exp.iloc[[1, 2]])
    tm.assert_frame_equal(g.nth(2, dropna="any"), exp.iloc[[1, 2]])

    # out of bounds, regression from 0.13.1
    # GH 6621
    df = DataFrame(
        {
            "color": {0: "green", 1: "green", 2: "red", 3: "red", 4: "red"},
            "food": {0: "ham", 1: "eggs", 2: "eggs", 3: "ham", 4: "pork"},
            "two": {
                0: 1.5456590000000001,
                1: -0.070345000000000005,
                2: -2.4004539999999999,
                3: 0.46206000000000003,
                4: 0.52350799999999997,
            },
            "one": {
                0: 0.56573799999999996,
                1: -0.9742360000000001,
                2: 1.033801,
                3: -0.78543499999999999,
                4: 0.70422799999999997,
            },
        }
    ).set_index(["color", "food"])

    result = df.groupby(level=0, as_index=False).nth(2)
    expected = df.iloc[[-1]]
    tm.assert_frame_equal(result, expected)

    result = df.groupby(level=0, as_index=False).nth(3)
    expected = df.loc[[]]
    tm.assert_frame_equal(result, expected)

    # GH 7559
    # from the vbench
    df = DataFrame(np.random.randint(1, 10, (100, 2)), dtype="int64")
    s = df[1]
    g = df[0]
    expected = s.groupby(g).first()
    expected2 = s.groupby(g).apply(lambda x: x.iloc[0])
    tm.assert_series_equal(expected2, expected, check_names=False)
    assert expected.name == 1
    assert expected2.name == 1

    # validate first
    v = s[g == 1].iloc[0]
    assert expected.iloc[0] == v
    assert expected2.iloc[0] == v

    # this is NOT the same as .first (as sorted is default!)
    # as it keeps the order in the series (and not the group order)
    # related GH 7287
    expected = s.groupby(g, sort=False).first()
    result = s.groupby(g, sort=False).nth(0, dropna="all")
    tm.assert_series_equal(result, expected)

    with pytest.raises(ValueError, match="For a DataFrame groupby"):
        s.groupby(g, sort=False).nth(0, dropna=True)

    # doc example
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A")
    result = g.B.nth(0, dropna="all")
    expected = g.B.first()
    tm.assert_series_equal(result, expected)

    # test multiple nth values
    df = DataFrame([[1, np.nan], [1, 3], [1, 4], [5, 6], [5, 7]], columns=["A", "B"])
    g = df.groupby("A")

    tm.assert_frame_equal(g.nth(0), df.iloc[[0, 3]].set_index("A"))
    tm.assert_frame_equal(g.nth([0]), df.iloc[[0, 3]].set_index("A"))
    tm.assert_frame_equal(g.nth([0, 1]), df.iloc[[0, 1, 3, 4]].set_index("A"))
    tm.assert_frame_equal(g.nth([0, -1]), df.iloc[[0, 2, 3, 4]].set_index("A"))
    tm.assert_frame_equal(g.nth([0, 1, 2]), df.iloc[[0, 1, 2, 3, 4]].set_index("A"))
    tm.assert_frame_equal(g.nth([0, 1, -1]), df.iloc[[0, 1, 2, 3, 4]].set_index("A"))
    tm.assert_frame_equal(g.nth([2]), df.iloc[[2]].set_index("A"))
    tm.assert_frame_equal(g.nth([3, 4]), df.loc[[]].set_index("A"))

    business_dates = pd.date_range(start="4/1/2014", end="6/30/2014", freq="B")
    df = DataFrame(1, index=business_dates, columns=["a", "b"])
    # get the first, fourth and last two business days for each month
    key = [df.index.year, df.index.month]
    result = df.groupby(key, as_index=False).nth([0, 3, -2, -1])
    expected_dates = pd.to_datetime(
        [
            "2014/4/1",
            "2014/4/4",
            "2014/4/29",
            "2014/4/30",
            "2014/5/1",
            "2014/5/6",
            "2014/5/29",
            "2014/5/30",
            "2014/6/2",
            "2014/6/5",
            "2014/6/27",
            "2014/6/30",
        ]
    )
    expected = DataFrame(1, columns=["a", "b"], index=expected_dates)
    tm.assert_frame_equal(result, expected)


def test_nth_multi_index(three_group):
    # PR 9090, related to issue 8979
    # test nth on MultiIndex, should match .first()
    grouped = three_group.groupby(["A", "B"])
    result = grouped.nth(0)
    expected = grouped.first()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_first, expected_last",
    [
        (
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
        ),
        (
            {
                "id": ["A", "B", "A"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                ],
                "foo": [1, 2, 3],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [1, 2],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [3, 2],
            },
        ),
    ],
)
def test_first_last_tz(data, expected_first, expected_last):
    # GH15884
    # Test that the timezone is retained when calling first
    # or last on groupby with as_index=False

    df = DataFrame(data)

    result = df.groupby("id", as_index=False).first()
    expected = DataFrame(expected_first)
    cols = ["id", "time", "foo"]
    tm.assert_frame_equal(result[cols], expected[cols])

    result = df.groupby("id", as_index=False)["time"].first()
    tm.assert_frame_equal(result, expected[["id", "time"]])

    result = df.groupby("id", as_index=False).last()
    expected = DataFrame(expected_last)
    cols = ["id", "time", "foo"]
    tm.assert_frame_equal(result[cols], expected[cols])

    result = df.groupby("id", as_index=False)["time"].last()
    tm.assert_frame_equal(result, expected[["id", "time"]])


@pytest.mark.parametrize(
    "method, ts, alpha",
    [
        ["first", Timestamp("2013-01-01", tz="US/Eastern"), "a"],
        ["last", Timestamp("2013-01-02", tz="US/Eastern"), "b"],
    ],
)
def test_first_last_tz_multi_column(method, ts, alpha):
    # GH 21603
    category_string = Series(list("abc")).astype("category")
    df = DataFrame(
        {
            "group": [1, 1, 2],
            "category_string": category_string,
            "datetimetz": pd.date_range("20130101", periods=3, tz="US/Eastern"),
        }
    )
    result = getattr(df.groupby("group"), method)()
    expected = DataFrame(
        {
            "category_string": pd.Categorical(
                [alpha, "c"], dtype=category_string.dtype
            ),
            "datetimetz": [ts, Timestamp("2013-01-03", tz="US/Eastern")],
        },
        index=Index([1, 2], name="group"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.array([True, False], dtype="boolean"),
        pd.array([1, 2], dtype="Int64"),
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        pd.to_timedelta([1, 2], unit="D"),
    ],
)
@pytest.mark.parametrize("function", ["first", "last", "min", "max"])
def test_first_last_extension_array_keeps_dtype(values, function):
    # https://github.com/pandas-dev/pandas/issues/33071
    # https://github.com/pandas-dev/pandas/issues/32194
    df = DataFrame({"a": [1, 2], "b": values})
    grouped = df.groupby("a")
    idx = Index([1, 2], name="a")
    expected_series = Series(values, name="b", index=idx)
    expected_frame = DataFrame({"b": values}, index=idx)

    result_series = getattr(grouped["b"], function)()
    tm.assert_series_equal(result_series, expected_series)

    result_frame = grouped.agg({"b": function})
    tm.assert_frame_equal(result_frame, expected_frame)


def test_nth_multi_index_as_expected():
    # PR 9090, related to issue 8979
    # test nth on MultiIndex
    three_group = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
        }
    )
    grouped = three_group.groupby(["A", "B"])
    result = grouped.nth(0)
    expected = DataFrame(
        {"C": ["dull", "dull", "dull", "dull"]},
        index=MultiIndex.from_arrays(
            [["bar", "bar", "foo", "foo"], ["one", "two", "one", "two"]],
            names=["A", "B"],
        ),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, n, expected_rows",
    [
        ("head", -1, []),
        ("head", 0, []),
        ("head", 1, [0, 2]),
        ("head", 7, [0, 1, 2]),
        ("tail", -1, []),
        ("tail", 0, []),
        ("tail", 1, [1, 2]),
        ("tail", 7, [0, 1, 2]),
    ],
)
@pytest.mark.parametrize("columns", [None, [], ["A"], ["B"], ["A", "B"]])
@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_head_tail(op, n, expected_rows, columns, as_index):
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    g = df.groupby("A", as_index=as_index)
    expected = df.iloc[expected_rows]
    if columns is not None:
        g = g[columns]
        expected = expected[columns]
    result = getattr(g, op)(n)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, n, expected_cols",
    [
        ("head", -1, []),
        ("head", 0, []),
        ("head", 1, [0, 2]),
        ("head", 7, [0, 1, 2]),
        ("tail", -1, []),
        ("tail", 0, []),
        ("tail", 1, [1, 2]),
        ("tail", 7, [0, 1, 2]),
    ],
)
def test_groupby_head_tail_axis_1(op, n, expected_cols):
    # GH 9772
    df = DataFrame(
        [[1, 2, 3], [1, 4, 5], [2, 6, 7], [3, 8, 9]], columns=["A", "B", "C"]
    )
    g = df.groupby([0, 0, 1], axis=1)
    expected = df.iloc[:, expected_cols]
    result = getattr(g, op)(n)
    tm.assert_frame_equal(result, expected)


def test_group_selection_cache():
    # GH 12839 nth, head, and tail should return same result consistently
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    expected = df.iloc[[0, 2]].set_index("A")

    g = df.groupby("A")
    result1 = g.head(n=2)
    result2 = g.nth(0)
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, expected)

    g = df.groupby("A")
    result1 = g.tail(n=2)
    result2 = g.nth(0)
    tm.assert_frame_equal(result1, df)
    tm.assert_frame_equal(result2, expected)

    g = df.groupby("A")
    result1 = g.nth(0)
    result2 = g.head(n=2)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, df)

    g = df.groupby("A")
    result1 = g.nth(0)
    result2 = g.tail(n=2)
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, df)


def test_nth_empty():
    # GH 16064
    df = DataFrame(index=[0], columns=["a", "b", "c"])
    result = df.groupby("a").nth(10)
    expected = DataFrame(index=Index([], name="a"), columns=["b", "c"])
    tm.assert_frame_equal(result, expected)

    result = df.groupby(["a", "b"]).nth(10)
    expected = DataFrame(
        index=MultiIndex([[], []], [[], []], names=["a", "b"]), columns=["c"]
    )
    tm.assert_frame_equal(result, expected)


def test_nth_column_order():
    # GH 20760
    # Check that nth preserves column order
    df = DataFrame(
        [[1, "b", 100], [1, "a", 50], [1, "a", np.nan], [2, "c", 200], [2, "d", 150]],
        columns=["A", "C", "B"],
    )
    result = df.groupby("A").nth(0)
    expected = DataFrame(
        [["b", 100.0], ["c", 200.0]], columns=["C", "B"], index=Index([1, 2], name="A")
    )
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A").nth(-1, dropna="any")
    expected = DataFrame(
        [["a", 50.0], ["d", 150.0]], columns=["C", "B"], index=Index([1, 2], name="A")
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dropna", [None, "any", "all"])
def test_nth_nan_in_grouper(dropna):
    # GH 26011
    df = DataFrame(
        [[np.nan, 0, 1], ["abc", 2, 3], [np.nan, 4, 5], ["def", 6, 7], [np.nan, 8, 9]],
        columns=list("abc"),
    )
    result = df.groupby("a").nth(0, dropna=dropna)
    expected = DataFrame(
        [[2, 3], [6, 7]], columns=list("bc"), index=Index(["abc", "def"], name="a")
    )

    tm.assert_frame_equal(result, expected)


def test_first_categorical_and_datetime_data_nat():
    # GH 20520
    df = DataFrame(
        {
            "group": ["first", "first", "second", "third", "third"],
            "time": 5 * [np.datetime64("NaT")],
            "categories": Series(["a", "b", "c", "a", "b"], dtype="category"),
        }
    )
    result = df.groupby("group").first()
    expected = DataFrame(
        {
            "time": 3 * [np.datetime64("NaT")],
            "categories": Series(["a", "c", "a"]).astype(
                pd.CategoricalDtype(["a", "b", "c"])
            ),
        }
    )
    expected.index = Index(["first", "second", "third"], name="group")
    tm.assert_frame_equal(result, expected)


def test_first_multi_key_groupbby_categorical():
    # GH 22512
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2],
            "B": [100, 100, 200, 100, 100],
            "C": ["apple", "orange", "mango", "mango", "orange"],
            "D": ["jupiter", "mercury", "mars", "venus", "venus"],
        }
    )
    df = df.astype({"D": "category"})
    result = df.groupby(by=["A", "B"]).first()
    expected = DataFrame(
        {
            "C": ["apple", "mango", "mango"],
            "D": Series(["jupiter", "mars", "venus"]).astype(
                pd.CategoricalDtype(["jupiter", "mars", "mercury", "venus"])
            ),
        }
    )
    expected.index = MultiIndex.from_tuples(
        [(1, 100), (1, 200), (2, 100)], names=["A", "B"]
    )
    tm.assert_frame_equal(result, expected)
