import datetime as dt
from string import ascii_lowercase

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    NaT,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


@pytest.mark.slow
@pytest.mark.parametrize("n", 10 ** np.arange(2, 6))
@pytest.mark.parametrize("m", [10, 100, 1000])
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("dropna", [False, True])
def test_series_groupby_nunique(n, m, sort, dropna):
    def check_nunique(df, keys, as_index=True):
        original_df = df.copy()
        gr = df.groupby(keys, as_index=as_index, sort=sort)
        left = gr["julie"].nunique(dropna=dropna)

        gr = df.groupby(keys, as_index=as_index, sort=sort)
        right = gr["julie"].apply(Series.nunique, dropna=dropna)
        if not as_index:
            right = right.reset_index(drop=True)

        if as_index:
            tm.assert_series_equal(left, right, check_names=False)
        else:
            tm.assert_frame_equal(left, right, check_names=False)
        tm.assert_frame_equal(df, original_df)

    days = date_range("2015-08-23", periods=10)

    frame = DataFrame(
        {
            "jim": np.random.choice(list(ascii_lowercase), n),
            "joe": np.random.choice(days, n),
            "julie": np.random.randint(0, m, n),
        }
    )

    check_nunique(frame, ["jim"])
    check_nunique(frame, ["jim", "joe"])

    frame.loc[1::17, "jim"] = None
    frame.loc[3::37, "joe"] = None
    frame.loc[7::19, "julie"] = None
    frame.loc[8::19, "julie"] = None
    frame.loc[9::19, "julie"] = None

    check_nunique(frame, ["jim"])
    check_nunique(frame, ["jim", "joe"])
    check_nunique(frame, ["jim"], as_index=False)
    check_nunique(frame, ["jim", "joe"], as_index=False)


def test_nunique():
    df = DataFrame({"A": list("abbacc"), "B": list("abxacc"), "C": list("abbacx")})

    expected = DataFrame({"A": list("abc"), "B": [1, 2, 1], "C": [1, 1, 2]})
    result = df.groupby("A", as_index=False).nunique()
    tm.assert_frame_equal(result, expected)

    # as_index
    expected.index = list("abc")
    expected.index.name = "A"
    expected = expected.drop(columns="A")
    result = df.groupby("A").nunique()
    tm.assert_frame_equal(result, expected)

    # with na
    result = df.replace({"x": None}).groupby("A").nunique(dropna=False)
    tm.assert_frame_equal(result, expected)

    # dropna
    expected = DataFrame({"B": [1] * 3, "C": [1] * 3}, index=list("abc"))
    expected.index.name = "A"
    result = df.replace({"x": None}).groupby("A").nunique()
    tm.assert_frame_equal(result, expected)


def test_nunique_with_object():
    # GH 11077
    data = DataFrame(
        [
            [100, 1, "Alice"],
            [200, 2, "Bob"],
            [300, 3, "Charlie"],
            [-400, 4, "Dan"],
            [500, 5, "Edith"],
        ],
        columns=["amount", "id", "name"],
    )

    result = data.groupby(["id", "amount"])["name"].nunique()
    index = MultiIndex.from_arrays([data.id, data.amount])
    expected = Series([1] * 5, name="name", index=index)
    tm.assert_series_equal(result, expected)


def test_nunique_with_empty_series():
    # GH 12553
    data = Series(name="name", dtype=object)
    result = data.groupby(level=0).nunique()
    expected = Series(name="name", dtype="int64")
    tm.assert_series_equal(result, expected)


def test_nunique_with_timegrouper():
    # GH 13453
    test = DataFrame(
        {
            "time": [
                Timestamp("2016-06-28 09:35:35"),
                Timestamp("2016-06-28 16:09:30"),
                Timestamp("2016-06-28 16:46:28"),
            ],
            "data": ["1", "2", "3"],
        }
    ).set_index("time")
    result = test.groupby(pd.Grouper(freq="h"))["data"].nunique()
    expected = test.groupby(pd.Grouper(freq="h"))["data"].apply(Series.nunique)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "key, data, dropna, expected",
    [
        (
            ["x", "x", "x"],
            [Timestamp("2019-01-01"), NaT, Timestamp("2019-01-01")],
            True,
            Series([1], index=pd.Index(["x"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x"],
            [dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1)],
            True,
            Series([1], index=pd.Index(["x"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x", "y", "y"],
            [dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1)],
            False,
            Series([2, 2], index=pd.Index(["x", "y"], name="key"), name="data"),
        ),
        (
            ["x", "x", "x", "x", "y"],
            [dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1), NaT, dt.date(2019, 1, 1)],
            False,
            Series([2, 1], index=pd.Index(["x", "y"], name="key"), name="data"),
        ),
    ],
)
def test_nunique_with_NaT(key, data, dropna, expected):
    # GH 27951
    df = DataFrame({"key": key, "data": data})
    result = df.groupby(["key"])["data"].nunique(dropna=dropna)
    tm.assert_series_equal(result, expected)


def test_nunique_preserves_column_level_names():
    # GH 23222
    test = DataFrame([1, 2, 2], columns=pd.Index(["A"], name="level_0"))
    result = test.groupby([0, 0, 0]).nunique()
    expected = DataFrame([2], columns=test.columns)
    tm.assert_frame_equal(result, expected)


def test_nunique_transform_with_datetime():
    # GH 35109 - transform with nunique on datetimes results in integers
    df = DataFrame(date_range("2008-12-31", "2009-01-02"), columns=["date"])
    result = df.groupby([0, 0, 1])["date"].transform("nunique")
    expected = Series([2, 2, 1], name="date")
    tm.assert_series_equal(result, expected)
