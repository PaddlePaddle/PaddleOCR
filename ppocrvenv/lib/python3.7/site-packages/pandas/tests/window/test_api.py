import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timestamp,
    concat,
    date_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.base import SpecificationError


def test_getitem():
    frame = DataFrame(np.random.randn(5, 5))
    r = frame.rolling(window=5)
    tm.assert_index_equal(r._selected_obj.columns, frame.columns)

    r = frame.rolling(window=5)[1]
    assert r._selected_obj.name == frame.columns[1]

    # technically this is allowed
    r = frame.rolling(window=5)[1, 3]
    tm.assert_index_equal(r._selected_obj.columns, frame.columns[[1, 3]])

    r = frame.rolling(window=5)[[1, 3]]
    tm.assert_index_equal(r._selected_obj.columns, frame.columns[[1, 3]])


def test_select_bad_cols():
    df = DataFrame([[1, 2]], columns=["A", "B"])
    g = df.rolling(window=5)
    with pytest.raises(KeyError, match="Columns not found: 'C'"):
        g[["C"]]
    with pytest.raises(KeyError, match="^[^A]+$"):
        # A should not be referenced as a bad column...
        # will have to rethink regex if you change message!
        g[["A", "C"]]


def test_attribute_access():

    df = DataFrame([[1, 2]], columns=["A", "B"])
    r = df.rolling(window=5)
    tm.assert_series_equal(r.A.sum(), r["A"].sum())
    msg = "'Rolling' object has no attribute 'F'"
    with pytest.raises(AttributeError, match=msg):
        r.F


def tests_skip_nuisance():

    df = DataFrame({"A": range(5), "B": range(5, 10), "C": "foo"})
    r = df.rolling(window=3)
    result = r[["A", "B"]].sum()
    expected = DataFrame(
        {"A": [np.nan, np.nan, 3, 6, 9], "B": [np.nan, np.nan, 18, 21, 24]},
        columns=list("AB"),
    )
    tm.assert_frame_equal(result, expected)


def test_skip_sum_object_raises():
    df = DataFrame({"A": range(5), "B": range(5, 10), "C": "foo"})
    r = df.rolling(window=3)
    result = r.sum()
    expected = DataFrame(
        {"A": [np.nan, np.nan, 3, 6, 9], "B": [np.nan, np.nan, 18, 21, 24]},
        columns=list("AB"),
    )
    tm.assert_frame_equal(result, expected)


def test_agg():
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    r = df.rolling(window=3)
    a_mean = r["A"].mean()
    a_std = r["A"].std()
    a_sum = r["A"].sum()
    b_mean = r["B"].mean()
    b_std = r["B"].std()

    result = r.aggregate([np.mean, np.std])
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = MultiIndex.from_product([["A", "B"], ["mean", "std"]])
    tm.assert_frame_equal(result, expected)

    result = r.aggregate({"A": np.mean, "B": np.std})

    expected = concat([a_mean, b_std], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)

    result = r.aggregate({"A": ["mean", "std"]})
    expected = concat([a_mean, a_std], axis=1)
    expected.columns = MultiIndex.from_tuples([("A", "mean"), ("A", "std")])
    tm.assert_frame_equal(result, expected)

    result = r["A"].aggregate(["mean", "sum"])
    expected = concat([a_mean, a_sum], axis=1)
    expected.columns = ["mean", "sum"]
    tm.assert_frame_equal(result, expected)

    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        # using a dict with renaming
        r.aggregate({"A": {"mean": "mean", "sum": "sum"}})

    with pytest.raises(SpecificationError, match=msg):
        r.aggregate(
            {"A": {"mean": "mean", "sum": "sum"}, "B": {"mean2": "mean", "sum2": "sum"}}
        )

    result = r.aggregate({"A": ["mean", "std"], "B": ["mean", "std"]})
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)

    exp_cols = [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    expected.columns = MultiIndex.from_tuples(exp_cols)
    tm.assert_frame_equal(result, expected, check_like=True)


def test_agg_apply(raw):

    # passed lambda
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    r = df.rolling(window=3)
    a_sum = r["A"].sum()

    result = r.agg({"A": np.sum, "B": lambda x: np.std(x, ddof=1)})
    rcustom = r["B"].apply(lambda x: np.std(x, ddof=1), raw=raw)
    expected = concat([a_sum, rcustom], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)


def test_agg_consistency():

    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})
    r = df.rolling(window=3)

    result = r.agg([np.sum, np.mean]).columns
    expected = MultiIndex.from_product([list("AB"), ["sum", "mean"]])
    tm.assert_index_equal(result, expected)

    result = r["A"].agg([np.sum, np.mean]).columns
    expected = Index(["sum", "mean"])
    tm.assert_index_equal(result, expected)

    result = r.agg({"A": [np.sum, np.mean]}).columns
    expected = MultiIndex.from_tuples([("A", "sum"), ("A", "mean")])
    tm.assert_index_equal(result, expected)


def test_agg_nested_dicts():

    # API change for disallowing these types of nested dicts
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})
    r = df.rolling(window=3)

    msg = "nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        r.aggregate({"r1": {"A": ["mean", "sum"]}, "r2": {"B": ["mean", "sum"]}})

    expected = concat(
        [r["A"].mean(), r["A"].std(), r["B"].mean(), r["B"].std()], axis=1
    )
    expected.columns = MultiIndex.from_tuples(
        [("ra", "mean"), ("ra", "std"), ("rb", "mean"), ("rb", "std")]
    )
    with pytest.raises(SpecificationError, match=msg):
        r[["A", "B"]].agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})

    with pytest.raises(SpecificationError, match=msg):
        r.agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})


def test_count_nonnumeric_types():
    # GH12541
    cols = [
        "int",
        "float",
        "string",
        "datetime",
        "timedelta",
        "periods",
        "fl_inf",
        "fl_nan",
        "str_nan",
        "dt_nat",
        "periods_nat",
    ]
    dt_nat_col = [Timestamp("20170101"), Timestamp("20170203"), Timestamp(None)]

    df = DataFrame(
        {
            "int": [1, 2, 3],
            "float": [4.0, 5.0, 6.0],
            "string": list("abc"),
            "datetime": date_range("20170101", periods=3),
            "timedelta": timedelta_range("1 s", periods=3, freq="s"),
            "periods": [
                Period("2012-01"),
                Period("2012-02"),
                Period("2012-03"),
            ],
            "fl_inf": [1.0, 2.0, np.Inf],
            "fl_nan": [1.0, 2.0, np.NaN],
            "str_nan": ["aa", "bb", np.NaN],
            "dt_nat": dt_nat_col,
            "periods_nat": [
                Period("2012-01"),
                Period("2012-02"),
                Period(None),
            ],
        },
        columns=cols,
    )

    expected = DataFrame(
        {
            "int": [1.0, 2.0, 2.0],
            "float": [1.0, 2.0, 2.0],
            "string": [1.0, 2.0, 2.0],
            "datetime": [1.0, 2.0, 2.0],
            "timedelta": [1.0, 2.0, 2.0],
            "periods": [1.0, 2.0, 2.0],
            "fl_inf": [1.0, 2.0, 2.0],
            "fl_nan": [1.0, 2.0, 1.0],
            "str_nan": [1.0, 2.0, 1.0],
            "dt_nat": [1.0, 2.0, 1.0],
            "periods_nat": [1.0, 2.0, 1.0],
        },
        columns=cols,
    )

    result = df.rolling(window=2, min_periods=0).count()
    tm.assert_frame_equal(result, expected)

    result = df.rolling(1, min_periods=0).count()
    expected = df.notna().astype(float)
    tm.assert_frame_equal(result, expected)


def test_preserve_metadata():
    # GH 10565
    s = Series(np.arange(100), name="foo")

    s2 = s.rolling(30).sum()
    s3 = s.rolling(20).sum()
    assert s2.name == "foo"
    assert s3.name == "foo"


@pytest.mark.parametrize(
    "func,window_size,expected_vals",
    [
        (
            "rolling",
            2,
            [
                [np.nan, np.nan, np.nan, np.nan],
                [15.0, 20.0, 25.0, 20.0],
                [25.0, 30.0, 35.0, 30.0],
                [np.nan, np.nan, np.nan, np.nan],
                [20.0, 30.0, 35.0, 30.0],
                [35.0, 40.0, 60.0, 40.0],
                [60.0, 80.0, 85.0, 80],
            ],
        ),
        (
            "expanding",
            None,
            [
                [10.0, 10.0, 20.0, 20.0],
                [15.0, 20.0, 25.0, 20.0],
                [20.0, 30.0, 30.0, 20.0],
                [10.0, 10.0, 30.0, 30.0],
                [20.0, 30.0, 35.0, 30.0],
                [26.666667, 40.0, 50.0, 30.0],
                [40.0, 80.0, 60.0, 30.0],
            ],
        ),
    ],
)
def test_multiple_agg_funcs(func, window_size, expected_vals):
    # GH 15072
    df = DataFrame(
        [
            ["A", 10, 20],
            ["A", 20, 30],
            ["A", 30, 40],
            ["B", 10, 30],
            ["B", 30, 40],
            ["B", 40, 80],
            ["B", 80, 90],
        ],
        columns=["stock", "low", "high"],
    )

    f = getattr(df.groupby("stock"), func)
    if window_size:
        window = f(window_size)
    else:
        window = f()

    index = MultiIndex.from_tuples(
        [("A", 0), ("A", 1), ("A", 2), ("B", 3), ("B", 4), ("B", 5), ("B", 6)],
        names=["stock", None],
    )
    columns = MultiIndex.from_tuples(
        [("low", "mean"), ("low", "max"), ("high", "mean"), ("high", "min")]
    )
    expected = DataFrame(expected_vals, index=index, columns=columns)

    result = window.agg({"low": ["mean", "max"], "high": ["mean", "min"]})

    tm.assert_frame_equal(result, expected)


def test_is_datetimelike_deprecated():
    s = Series(range(1)).rolling(1)
    with tm.assert_produces_warning(FutureWarning):
        assert not s.is_datetimelike


@pytest.mark.filterwarnings("ignore:min_periods:FutureWarning")
def test_dont_modify_attributes_after_methods(
    arithmetic_win_operators, closed, center, min_periods
):
    # GH 39554
    roll_obj = Series(range(1)).rolling(
        1, center=center, closed=closed, min_periods=min_periods
    )
    expected = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    getattr(roll_obj, arithmetic_win_operators)()
    result = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    assert result == expected
