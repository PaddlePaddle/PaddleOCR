from datetime import datetime
from decimal import Decimal
from io import StringIO

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    read_csv,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.base import SpecificationError
import pandas.core.common as com


def test_repr():
    # GH18203
    result = repr(Grouper(key="A", level="B"))
    expected = "Grouper(key='A', level='B', axis=0, sort=False)"
    assert result == expected


@pytest.mark.parametrize("dtype", ["int64", "int32", "float64", "float32"])
def test_basic(dtype):

    data = Series(np.arange(9) // 3, index=np.arange(9), dtype=dtype)

    index = np.arange(9)
    np.random.shuffle(index)
    data = data.reindex(index)

    grouped = data.groupby(lambda x: x // 3)

    for k, v in grouped:
        assert len(v) == 3

    agged = grouped.aggregate(np.mean)
    assert agged[1] == 1

    tm.assert_series_equal(agged, grouped.agg(np.mean))  # shorthand
    tm.assert_series_equal(agged, grouped.mean())
    tm.assert_series_equal(grouped.agg(np.sum), grouped.sum())

    expected = grouped.apply(lambda x: x * x.sum())
    transformed = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12
    tm.assert_series_equal(transformed, expected)

    value_grouped = data.groupby(data)
    tm.assert_series_equal(
        value_grouped.aggregate(np.mean), agged, check_index_type=False
    )

    # complex agg
    agged = grouped.aggregate([np.mean, np.std])

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped.aggregate({"one": np.mean, "two": np.std})

    group_constants = {0: 10, 1: 20, 2: 30}
    agged = grouped.agg(lambda x: group_constants[x.name] + x.mean())
    assert agged[1] == 21

    # corner cases
    msg = "Must produce aggregated value"
    # exception raised is type Exception
    with pytest.raises(Exception, match=msg):
        grouped.aggregate(lambda x: x * 2)


def test_groupby_nonobject_dtype(mframe, df_mixed_floats):
    key = mframe.index.codes[0]
    grouped = mframe.groupby(key)
    result = grouped.sum()

    expected = mframe.groupby(key.astype("O")).sum()
    tm.assert_frame_equal(result, expected)

    # GH 3911, mixed frame non-conversion
    df = df_mixed_floats.copy()
    df["value"] = range(len(df))

    def max_value(group):
        return group.loc[group["value"].idxmax()]

    applied = df.groupby("A").apply(max_value)
    result = applied.dtypes
    expected = df.dtypes
    tm.assert_series_equal(result, expected)


def test_groupby_return_type():

    # GH2893, return a reduced type
    df1 = DataFrame(
        [
            {"val1": 1, "val2": 20},
            {"val1": 1, "val2": 19},
            {"val1": 2, "val2": 27},
            {"val1": 2, "val2": 12},
        ]
    )

    def func(dataf):
        return dataf["val2"] - dataf["val2"].mean()

    with tm.assert_produces_warning(FutureWarning):
        result = df1.groupby("val1", squeeze=True).apply(func)
    assert isinstance(result, Series)

    df2 = DataFrame(
        [
            {"val1": 1, "val2": 20},
            {"val1": 1, "val2": 19},
            {"val1": 1, "val2": 27},
            {"val1": 1, "val2": 12},
        ]
    )

    def func(dataf):
        return dataf["val2"] - dataf["val2"].mean()

    with tm.assert_produces_warning(FutureWarning):
        result = df2.groupby("val1", squeeze=True).apply(func)
    assert isinstance(result, Series)

    # GH3596, return a consistent type (regression in 0.11 from 0.10.1)
    df = DataFrame([[1, 1], [1, 1]], columns=["X", "Y"])
    with tm.assert_produces_warning(FutureWarning):
        result = df.groupby("X", squeeze=False).count()
    assert isinstance(result, DataFrame)


def test_inconsistent_return_type():
    # GH5592
    # inconsistent return type
    df = DataFrame(
        {
            "A": ["Tiger", "Tiger", "Tiger", "Lamb", "Lamb", "Pony", "Pony"],
            "B": Series(np.arange(7), dtype="int64"),
            "C": date_range("20130101", periods=7),
        }
    )

    def f(grp):
        return grp.iloc[0]

    expected = df.groupby("A").first()[["B"]]
    result = df.groupby("A").apply(f)[["B"]]
    tm.assert_frame_equal(result, expected)

    def f(grp):
        if grp.name == "Tiger":
            return None
        return grp.iloc[0]

    result = df.groupby("A").apply(f)[["B"]]
    e = expected.copy()
    e.loc["Tiger"] = np.nan
    tm.assert_frame_equal(result, e)

    def f(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0]

    result = df.groupby("A").apply(f)[["B"]]
    e = expected.copy()
    e.loc["Pony"] = np.nan
    tm.assert_frame_equal(result, e)

    # 5592 revisited, with datetimes
    def f(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0]

    result = df.groupby("A").apply(f)[["C"]]
    e = df.groupby("A").first()[["C"]]
    e.loc["Pony"] = pd.NaT
    tm.assert_frame_equal(result, e)

    # scalar outputs
    def f(grp):
        if grp.name == "Pony":
            return None
        return grp.iloc[0].loc["C"]

    result = df.groupby("A").apply(f)
    e = df.groupby("A").first()["C"].copy()
    e.loc["Pony"] = np.nan
    e.name = None
    tm.assert_series_equal(result, e)


def test_pass_args_kwargs(ts, tsframe):
    def f(x, q=None, axis=0):
        return np.percentile(x, q, axis=axis)

    g = lambda x: np.percentile(x, 80, axis=0)

    # Series
    ts_grouped = ts.groupby(lambda x: x.month)
    agg_result = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result = ts_grouped.transform(np.percentile, 80, axis=0)

    agg_expected = ts_grouped.quantile(0.8)
    trans_expected = ts_grouped.transform(g)

    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    # DataFrame
    for as_index in [True, False]:
        df_grouped = tsframe.groupby(lambda x: x.month, as_index=as_index)
        agg_result = df_grouped.agg(np.percentile, 80, axis=0)
        apply_result = df_grouped.apply(DataFrame.quantile, 0.8)
        expected = df_grouped.quantile(0.8)
        tm.assert_frame_equal(apply_result, expected, check_names=False)
        tm.assert_frame_equal(agg_result, expected)

        agg_result = df_grouped.agg(f, q=80)
        apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
        tm.assert_frame_equal(agg_result, expected)
        tm.assert_frame_equal(apply_result, expected, check_names=False)


@pytest.mark.parametrize("as_index", [True, False])
def test_pass_args_kwargs_duplicate_columns(tsframe, as_index):
    # go through _aggregate_frame with self.axis == 0 and duplicate columns
    tsframe.columns = ["A", "B", "A", "C"]
    gb = tsframe.groupby(lambda x: x.month, as_index=as_index)

    res = gb.agg(np.percentile, 80, axis=0)

    ex_data = {
        1: tsframe[tsframe.index.month == 1].quantile(0.8),
        2: tsframe[tsframe.index.month == 2].quantile(0.8),
    }
    expected = DataFrame(ex_data).T
    if not as_index:
        # TODO: try to get this more consistent?
        expected.index = Index(range(2))

    tm.assert_frame_equal(res, expected)


def test_len():
    df = tm.makeTimeDataFrame()
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    assert len(grouped) == len(df)

    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    expected = len({(x.year, x.month) for x in df.index})
    assert len(grouped) == expected

    # issue 11016
    df = DataFrame({"a": [np.nan] * 3, "b": [1, 2, 3]})
    assert len(df.groupby("a")) == 0
    assert len(df.groupby("b")) == 3
    assert len(df.groupby(["a", "b"])) == 3


def test_basic_regression():
    # regression
    result = Series([1.0 * x for x in list(range(1, 10)) * 10])

    data = np.random.random(1100) * 10.0
    groupings = Series(data)

    grouped = result.groupby(groupings)
    grouped.mean()


@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "int64", "int32", "int16", "int8"]
)
def test_with_na_groups(dtype):
    index = Index(np.arange(10))
    values = Series(np.ones(10), index, dtype=dtype)
    labels = Series(
        [np.nan, "foo", "bar", "bar", np.nan, np.nan, "bar", "bar", np.nan, "foo"],
        index=index,
    )

    # this SHOULD be an int
    grouped = values.groupby(labels)
    agged = grouped.agg(len)
    expected = Series([4, 2], index=["bar", "foo"])

    tm.assert_series_equal(agged, expected, check_dtype=False)

    # assert issubclass(agged.dtype.type, np.integer)

    # explicitly return a float from my function
    def f(x):
        return float(len(x))

    agged = grouped.agg(f)
    expected = Series([4.0, 2.0], index=["bar", "foo"])

    tm.assert_series_equal(agged, expected)


def test_indices_concatenation_order():

    # GH 2808

    def f1(x):
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=["b", "c"])
            res = DataFrame(columns=["a"], index=multiindex)
            return res
        else:
            y = y.set_index(["b", "c"])
            return y

    def f2(x):
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(["b", "c"])
            return y

    def f3(x):
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(
                levels=[[]] * 2, codes=[[]] * 2, names=["foo", "bar"]
            )
            res = DataFrame(columns=["a", "b"], index=multiindex)
            return res
        else:
            return y

    df = DataFrame({"a": [1, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    df2 = DataFrame({"a": [3, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    # correct result
    result1 = df.groupby("a").apply(f1)
    result2 = df2.groupby("a").apply(f1)
    tm.assert_frame_equal(result1, result2)

    # should fail (not the same number of levels)
    msg = "Cannot concat indices that do not have the same number of levels"
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f2)

    # should fail (incorrect shape)
    with pytest.raises(AssertionError, match=msg):
        df.groupby("a").apply(f3)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f3)


def test_attr_wrapper(ts):
    grouped = ts.groupby(lambda x: x.weekday())

    result = grouped.std()
    expected = grouped.agg(lambda x: np.std(x, ddof=1))
    tm.assert_series_equal(result, expected)

    # this is pretty cool
    result = grouped.describe()
    expected = {name: gp.describe() for name, gp in grouped}
    expected = DataFrame(expected).T
    tm.assert_frame_equal(result, expected)

    # get attribute
    result = grouped.dtype
    expected = grouped.agg(lambda x: x.dtype)
    tm.assert_series_equal(result, expected)

    # make sure raises error
    msg = "'SeriesGroupBy' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        getattr(grouped, "foo")


def test_frame_groupby(tsframe):
    grouped = tsframe.groupby(lambda x: x.weekday())

    # aggregate
    aggregated = grouped.aggregate(np.mean)
    assert len(aggregated) == 5
    assert len(aggregated.columns) == 4

    # by string
    tscopy = tsframe.copy()
    tscopy["weekday"] = [x.weekday() for x in tscopy.index]
    stragged = tscopy.groupby("weekday").aggregate(np.mean)
    tm.assert_frame_equal(stragged, aggregated, check_names=False)

    # transform
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed = grouped.transform(lambda x: x - x.mean())
    assert len(transformed) == 30
    assert len(transformed.columns) == 4

    # transform propagate
    transformed = grouped.transform(lambda x: x.mean())
    for name, group in grouped:
        mean = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)

    # iterate
    for weekday, group in grouped:
        assert group.index[0].weekday() == weekday

    # groups / group_indices
    groups = grouped.groups
    indices = grouped.indices

    for k, v in groups.items():
        samething = tsframe.index.take(indices[k])
        assert (samething == v).all()


def test_frame_groupby_columns(tsframe):
    mapping = {"A": 0, "B": 0, "C": 1, "D": 1}
    grouped = tsframe.groupby(mapping, axis=1)

    # aggregate
    aggregated = grouped.aggregate(np.mean)
    assert len(aggregated) == len(tsframe)
    assert len(aggregated.columns) == 2

    # transform
    tf = lambda x: x - x.mean()
    groupedT = tsframe.T.groupby(mapping, axis=0)
    tm.assert_frame_equal(groupedT.transform(tf).T, grouped.transform(tf))

    # iterate
    for k, v in grouped:
        assert len(v.columns) == 2


def test_frame_set_name_single(df):
    grouped = df.groupby("A")

    result = grouped.mean()
    assert result.index.name == "A"

    result = df.groupby("A", as_index=False).mean()
    assert result.index.name != "A"

    result = grouped.agg(np.mean)
    assert result.index.name == "A"

    result = grouped.agg({"C": np.mean, "D": np.std})
    assert result.index.name == "A"

    result = grouped["C"].mean()
    assert result.index.name == "A"
    result = grouped["C"].agg(np.mean)
    assert result.index.name == "A"
    result = grouped["C"].agg([np.mean, np.std])
    assert result.index.name == "A"

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"foo": np.mean, "bar": np.std})


def test_multi_func(df):
    col1 = df["A"]
    col2 = df["B"]

    grouped = df.groupby([col1.get, col2.get])
    agged = grouped.mean()
    expected = df.groupby(["A", "B"]).mean()

    # TODO groupby get drops names
    tm.assert_frame_equal(
        agged.loc[:, ["C", "D"]], expected.loc[:, ["C", "D"]], check_names=False
    )

    # some "groups" with no data
    df = DataFrame(
        {
            "v1": np.random.randn(6),
            "v2": np.random.randn(6),
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
        },
        index=["one", "two", "three", "four", "five", "six"],
    )
    # only verify that it works for now
    grouped = df.groupby(["k1", "k2"])
    grouped.agg(np.sum)


def test_multi_key_multiple_functions(df):
    grouped = df.groupby(["A", "B"])["C"]

    agged = grouped.agg([np.mean, np.std])
    expected = DataFrame({"mean": grouped.agg(np.mean), "std": grouped.agg(np.std)})
    tm.assert_frame_equal(agged, expected)


def test_frame_multi_key_function_list():
    data = DataFrame(
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
            "D": np.random.randn(11),
            "E": np.random.randn(11),
            "F": np.random.randn(11),
        }
    )

    grouped = data.groupby(["A", "B"])
    funcs = [np.mean, np.std]
    agged = grouped.agg(funcs)
    expected = pd.concat(
        [grouped["D"].agg(funcs), grouped["E"].agg(funcs), grouped["F"].agg(funcs)],
        keys=["D", "E", "F"],
        axis=1,
    )
    assert isinstance(agged.index, MultiIndex)
    assert isinstance(expected.index, MultiIndex)
    tm.assert_frame_equal(agged, expected)


@pytest.mark.parametrize("op", [lambda x: x.sum(), lambda x: x.mean()])
def test_groupby_multiple_columns(df, op):
    data = df
    grouped = data.groupby(["A", "B"])

    result1 = op(grouped)

    keys = []
    values = []
    for n1, gp1 in data.groupby("A"):
        for n2, gp2 in gp1.groupby("B"):
            keys.append((n1, n2))
            values.append(op(gp2.loc[:, ["C", "D"]]))

    mi = MultiIndex.from_tuples(keys, names=["A", "B"])
    expected = pd.concat(values, axis=1).T
    expected.index = mi

    # a little bit crude
    for col in ["C", "D"]:
        result_col = op(grouped[col])
        pivoted = result1[col]
        exp = expected[col]
        tm.assert_series_equal(result_col, exp)
        tm.assert_series_equal(pivoted, exp)

    # test single series works the same
    result = data["C"].groupby([data["A"], data["B"]]).mean()
    expected = data.groupby(["A", "B"]).mean()["C"]

    tm.assert_series_equal(result, expected)


def test_as_index_select_column():
    # GH 5764
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    result = df.groupby("A", as_index=False)["B"].get_group(1)
    expected = Series([2, 4], name="B")
    tm.assert_series_equal(result, expected)

    result = df.groupby("A", as_index=False)["B"].apply(lambda x: x.cumsum())
    expected = Series(
        [2, 6, 6], name="B", index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 2)])
    )
    tm.assert_series_equal(result, expected)


def test_groupby_as_index_select_column_sum_empty_df():
    # GH 35246
    df = DataFrame(columns=Index(["A", "B", "C"], name="alpha"))
    left = df.groupby(by="A", as_index=False)["B"].sum(numeric_only=False)

    expected = DataFrame(columns=df.columns[:2], index=range(0))
    tm.assert_frame_equal(left, expected)


def test_groupby_as_index_agg(df):
    grouped = df.groupby("A", as_index=False)

    # single-key

    result = grouped.agg(np.mean)
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)

    result2 = grouped.agg({"C": np.mean, "D": np.sum})
    expected2 = grouped.mean()
    expected2["D"] = grouped.sum()["D"]
    tm.assert_frame_equal(result2, expected2)

    grouped = df.groupby("A", as_index=True)

    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"Q": np.sum})

    # multi-key

    grouped = df.groupby(["A", "B"], as_index=False)

    result = grouped.agg(np.mean)
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)

    result2 = grouped.agg({"C": np.mean, "D": np.sum})
    expected2 = grouped.mean()
    expected2["D"] = grouped.sum()["D"]
    tm.assert_frame_equal(result2, expected2)

    expected3 = grouped["C"].sum()
    expected3 = DataFrame(expected3).rename(columns={"C": "Q"})
    result3 = grouped["C"].agg({"Q": np.sum})
    tm.assert_frame_equal(result3, expected3)

    # GH7115 & GH8112 & GH8582
    df = DataFrame(np.random.randint(0, 100, (50, 3)), columns=["jim", "joe", "jolie"])
    ts = Series(np.random.randint(5, 10, 50), name="jim")

    gr = df.groupby(ts)
    gr.nth(0)  # invokes set_selection_from_grouper internally
    tm.assert_frame_equal(gr.apply(sum), df.groupby(ts).apply(sum))

    for attr in ["mean", "max", "count", "idxmax", "cumsum", "all"]:
        gr = df.groupby(ts, as_index=False)
        left = getattr(gr, attr)()

        gr = df.groupby(ts.values, as_index=True)
        right = getattr(gr, attr)().reset_index(drop=True)

        tm.assert_frame_equal(left, right)


def test_ops_not_as_index(reduction_func):
    # GH 10355, 21090
    # Using as_index=False should not modify grouped column

    if reduction_func in ("corrwith",):
        pytest.skip("Test not applicable")

    if reduction_func in ("nth", "ngroup"):
        pytest.skip("Skip until behavior is determined (GH #5755)")

    df = DataFrame(np.random.randint(0, 5, size=(100, 2)), columns=["a", "b"])
    expected = getattr(df.groupby("a"), reduction_func)()
    if reduction_func == "size":
        expected = expected.rename("size")
    expected = expected.reset_index()

    g = df.groupby("a", as_index=False)

    result = getattr(g, reduction_func)()
    tm.assert_frame_equal(result, expected)

    result = g.agg(reduction_func)
    tm.assert_frame_equal(result, expected)

    result = getattr(g["b"], reduction_func)()
    tm.assert_frame_equal(result, expected)

    result = g["b"].agg(reduction_func)
    tm.assert_frame_equal(result, expected)


def test_as_index_series_return_frame(df):
    grouped = df.groupby("A", as_index=False)
    grouped2 = df.groupby(["A", "B"], as_index=False)

    result = grouped["C"].agg(np.sum)
    expected = grouped.agg(np.sum).loc[:, ["A", "C"]]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)

    result2 = grouped2["C"].agg(np.sum)
    expected2 = grouped2.agg(np.sum).loc[:, ["A", "B", "C"]]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)

    result = grouped["C"].sum()
    expected = grouped.sum().loc[:, ["A", "C"]]
    assert isinstance(result, DataFrame)
    tm.assert_frame_equal(result, expected)

    result2 = grouped2["C"].sum()
    expected2 = grouped2.sum().loc[:, ["A", "B", "C"]]
    assert isinstance(result2, DataFrame)
    tm.assert_frame_equal(result2, expected2)


def test_as_index_series_column_slice_raises(df):
    # GH15072
    grouped = df.groupby("A", as_index=False)
    msg = r"Column\(s\) C already selected"

    with pytest.raises(IndexError, match=msg):
        grouped["C"].__getitem__("D")


def test_groupby_as_index_cython(df):
    data = df

    # single-key
    grouped = data.groupby("A", as_index=False)
    result = grouped.mean()
    expected = data.groupby(["A"]).mean()
    expected.insert(0, "A", expected.index)
    expected.index = np.arange(len(expected))
    tm.assert_frame_equal(result, expected)

    # multi-key
    grouped = data.groupby(["A", "B"], as_index=False)
    result = grouped.mean()
    expected = data.groupby(["A", "B"]).mean()

    arrays = list(zip(*expected.index.values))
    expected.insert(0, "A", arrays[0])
    expected.insert(1, "B", arrays[1])
    expected.index = np.arange(len(expected))
    tm.assert_frame_equal(result, expected)


def test_groupby_as_index_series_scalar(df):
    grouped = df.groupby(["A", "B"], as_index=False)

    # GH #421

    result = grouped["C"].agg(len)
    expected = grouped.agg(len).loc[:, ["A", "B", "C"]]
    tm.assert_frame_equal(result, expected)


def test_groupby_as_index_corner(df, ts):
    msg = "as_index=False only valid with DataFrame"
    with pytest.raises(TypeError, match=msg):
        ts.groupby(lambda x: x.weekday(), as_index=False)

    msg = "as_index=False only valid for axis=0"
    with pytest.raises(ValueError, match=msg):
        df.groupby(lambda x: x.lower(), as_index=False, axis=1)


def test_groupby_multiple_key(df):
    df = tm.makeTimeDataFrame()
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    agged = grouped.sum()
    tm.assert_almost_equal(df.values, agged.values)

    grouped = df.T.groupby(
        [lambda x: x.year, lambda x: x.month, lambda x: x.day], axis=1
    )

    agged = grouped.agg(lambda x: x.sum())
    tm.assert_index_equal(agged.index, df.columns)
    tm.assert_almost_equal(df.T.values, agged.values)

    agged = grouped.agg(lambda x: x.sum())
    tm.assert_almost_equal(df.T.values, agged.values)


def test_groupby_multi_corner(df):
    # test that having an all-NA column doesn't mess you up
    df = df.copy()
    df["bad"] = np.nan
    agged = df.groupby(["A", "B"]).mean()

    expected = df.groupby(["A", "B"]).mean()
    expected["bad"] = np.nan

    tm.assert_frame_equal(agged, expected)


def test_omit_nuisance(df):
    grouped = df.groupby("A")

    result = grouped.mean()
    expected = df.loc[:, ["A", "C", "D"]].groupby("A").mean()
    tm.assert_frame_equal(result, expected)

    agged = grouped.agg(np.mean)
    exp = grouped.mean()
    tm.assert_frame_equal(agged, exp)

    df = df.loc[:, ["A", "C", "D"]]
    df["E"] = datetime.now()
    grouped = df.groupby("A")
    result = grouped.agg(np.sum)
    expected = grouped.sum()
    tm.assert_frame_equal(result, expected)

    # won't work with axis = 1
    grouped = df.groupby({"A": 0, "C": 0, "D": 1, "E": 1}, axis=1)
    msg = "'DatetimeArray' does not implement reduction 'sum'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(lambda x: x.sum(0, numeric_only=False))


def test_omit_nuisance_sem(df):
    # GH 38774 - sem should work with nuisance columns
    grouped = df.groupby("A")
    result = grouped.sem()
    expected = df.loc[:, ["A", "C", "D"]].groupby("A").sem()
    tm.assert_frame_equal(result, expected)


def test_omit_nuisance_python_multiple(three_group):
    grouped = three_group.groupby(["A", "B"])

    agged = grouped.agg(np.mean)
    exp = grouped.mean()
    tm.assert_frame_equal(agged, exp)


def test_empty_groups_corner(mframe):
    # handle empty groups
    df = DataFrame(
        {
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
            "k3": ["foo", "bar"] * 3,
            "v1": np.random.randn(6),
            "v2": np.random.randn(6),
        }
    )

    grouped = df.groupby(["k1", "k2"])
    result = grouped.agg(np.mean)
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)

    grouped = mframe[3:5].groupby(level=0)
    agged = grouped.apply(lambda x: x.mean())
    agged_A = grouped["A"].apply(np.mean)
    tm.assert_series_equal(agged["A"], agged_A)
    assert agged.index.name == "first"


def test_nonsense_func():
    df = DataFrame([0])
    msg = r"unsupported operand type\(s\) for \+: 'int' and 'str'"
    with pytest.raises(TypeError, match=msg):
        df.groupby(lambda x: x + "foo")


def test_wrap_aggregated_output_multindex(mframe):
    df = mframe.T
    df["baz", "two"] = "peekaboo"

    keys = [np.array([0, 0, 1]), np.array([0, 0, 1])]
    agged = df.groupby(keys).agg(np.mean)
    assert isinstance(agged.columns, MultiIndex)

    def aggfun(ser):
        if ser.name == ("foo", "one"):
            raise TypeError
        else:
            return ser.sum()

    with tm.assert_produces_warning(FutureWarning, match="Dropping invalid columns"):
        agged2 = df.groupby(keys).aggregate(aggfun)
    assert len(agged2.columns) + 1 == len(df.columns)


def test_groupby_level_apply(mframe):

    result = mframe.groupby(level=0).count()
    assert result.index.name == "first"
    result = mframe.groupby(level=1).count()
    assert result.index.name == "second"

    result = mframe["A"].groupby(level=0).count()
    assert result.index.name == "first"


def test_groupby_level_mapper(mframe):
    deleveled = mframe.reset_index()

    mapper0 = {"foo": 0, "bar": 0, "baz": 1, "qux": 1}
    mapper1 = {"one": 0, "two": 0, "three": 1}

    result0 = mframe.groupby(mapper0, level=0).sum()
    result1 = mframe.groupby(mapper1, level=1).sum()

    mapped_level0 = np.array([mapper0.get(x) for x in deleveled["first"]])
    mapped_level1 = np.array([mapper1.get(x) for x in deleveled["second"]])
    expected0 = mframe.groupby(mapped_level0).sum()
    expected1 = mframe.groupby(mapped_level1).sum()
    expected0.index.name, expected1.index.name = "first", "second"

    tm.assert_frame_equal(result0, expected0)
    tm.assert_frame_equal(result1, expected1)


def test_groupby_level_nonmulti():
    # GH 1313, GH 13901
    s = Series([1, 2, 3, 10, 4, 5, 20, 6], Index([1, 2, 3, 1, 4, 5, 2, 6], name="foo"))
    expected = Series([11, 22, 3, 4, 5, 6], Index(range(1, 7), name="foo"))

    result = s.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[0]).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=-1).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[-1]).sum()
    tm.assert_series_equal(result, expected)

    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=1)
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=-2)
    msg = "No group keys passed!"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[])
    msg = "multiple levels only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 0])
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 1])
    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[1])


def test_groupby_complex():
    # GH 12902
    a = Series(data=np.arange(4) * (1 + 2j), index=[0, 0, 1, 1])
    expected = Series((1 + 2j, 5 + 10j))

    result = a.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning):
        result = a.sum(level=0)
    tm.assert_series_equal(result, expected)


def test_groupby_series_indexed_differently():
    s1 = Series(
        [5.0, -9.0, 4.0, 100.0, -5.0, 55.0, 6.7],
        index=Index(["a", "b", "c", "d", "e", "f", "g"]),
    )
    s2 = Series(
        [1.0, 1.0, 4.0, 5.0, 5.0, 7.0], index=Index(["a", "b", "d", "f", "g", "h"])
    )

    grouped = s1.groupby(s2)
    agged = grouped.mean()
    exp = s1.groupby(s2.reindex(s1.index).get).mean()
    tm.assert_series_equal(agged, exp)


def test_groupby_with_hier_columns():
    tuples = list(
        zip(
            *[
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ]
        )
    )
    index = MultiIndex.from_tuples(tuples)
    columns = MultiIndex.from_tuples(
        [("A", "cat"), ("B", "dog"), ("B", "cat"), ("A", "dog")]
    )
    df = DataFrame(np.random.randn(8, 4), index=index, columns=columns)

    result = df.groupby(level=0).mean()
    tm.assert_index_equal(result.columns, columns)

    result = df.groupby(level=0, axis=1).mean()
    tm.assert_index_equal(result.index, df.index)

    result = df.groupby(level=0).agg(np.mean)
    tm.assert_index_equal(result.columns, columns)

    result = df.groupby(level=0).apply(lambda x: x.mean())
    tm.assert_index_equal(result.columns, columns)

    result = df.groupby(level=0, axis=1).agg(lambda x: x.mean(1))
    tm.assert_index_equal(result.columns, Index(["A", "B"]))
    tm.assert_index_equal(result.index, df.index)

    # add a nuisance column
    sorted_columns, _ = columns.sortlevel(0)
    df["A", "foo"] = "bar"
    result = df.groupby(level=0).mean()
    tm.assert_index_equal(result.columns, df.columns[:-1])


def test_grouping_ndarray(df):
    grouped = df.groupby(df["A"].values)

    result = grouped.sum()
    expected = df.groupby("A").sum()
    tm.assert_frame_equal(
        result, expected, check_names=False
    )  # Note: no names when grouping by value


def test_groupby_wrong_multi_labels():
    data = """index,foo,bar,baz,spam,data
0,foo1,bar1,baz1,spam2,20
1,foo1,bar2,baz1,spam3,30
2,foo2,bar2,baz1,spam2,40
3,foo1,bar1,baz2,spam1,50
4,foo3,bar1,baz2,spam1,60"""

    data = read_csv(StringIO(data), index_col=0)

    grouped = data.groupby(["foo", "bar", "baz", "spam"])

    result = grouped.agg(np.mean)
    expected = grouped.mean()
    tm.assert_frame_equal(result, expected)


def test_groupby_series_with_name(df):
    result = df.groupby(df["A"]).mean()
    result2 = df.groupby(df["A"], as_index=False).mean()
    assert result.index.name == "A"
    assert "A" in result2

    result = df.groupby([df["A"], df["B"]]).mean()
    result2 = df.groupby([df["A"], df["B"]], as_index=False).mean()
    assert result.index.names == ("A", "B")
    assert "A" in result2
    assert "B" in result2


def test_seriesgroupby_name_attr(df):
    # GH 6265
    result = df.groupby("A")["C"]
    assert result.count().name == "C"
    assert result.mean().name == "C"

    testFunc = lambda x: np.sum(x) * 2
    assert result.agg(testFunc).name == "C"


def test_consistency_name():
    # GH 12363

    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.randn(8) + 1.0,
            "D": np.arange(8),
        }
    )

    expected = df.groupby(["A"]).B.count()
    result = df.B.groupby(df.A).count()
    tm.assert_series_equal(result, expected)


def test_groupby_name_propagation(df):
    # GH 6124
    def summarize(df, name=None):
        return Series({"count": 1, "mean": 2, "omissions": 3}, name=name)

    def summarize_random_name(df):
        # Provide a different name for each Series.  In this case, groupby
        # should not attempt to propagate the Series name since they are
        # inconsistent.
        return Series({"count": 1, "mean": 2, "omissions": 3}, name=df.iloc[0]["A"])

    metrics = df.groupby("A").apply(summarize)
    assert metrics.columns.name is None
    metrics = df.groupby("A").apply(summarize, "metrics")
    assert metrics.columns.name == "metrics"
    metrics = df.groupby("A").apply(summarize_random_name)
    assert metrics.columns.name is None


def test_groupby_nonstring_columns():
    df = DataFrame([np.arange(10) for x in range(10)])
    grouped = df.groupby(0)
    result = grouped.mean()
    expected = df.groupby(df[0]).mean()
    tm.assert_frame_equal(result, expected)


def test_groupby_mixed_type_columns():
    # GH 13432, unorderable types in py3
    df = DataFrame([[0, 1, 2]], columns=["A", "B", 0])
    expected = DataFrame([[1, 2]], columns=["B", 0], index=Index([0], name="A"))

    result = df.groupby("A").first()
    tm.assert_frame_equal(result, expected)

    result = df.groupby("A").sum()
    tm.assert_frame_equal(result, expected)


# TODO: Ensure warning isn't emitted in the first place
@pytest.mark.filterwarnings("ignore:Mean of:RuntimeWarning")
def test_cython_grouper_series_bug_noncontig():
    arr = np.empty((100, 100))
    arr.fill(np.nan)
    obj = Series(arr[:, 0])
    inds = np.tile(range(10), 10)

    result = obj.groupby(inds).agg(Series.median)
    assert result.isna().all()


def test_series_grouper_noncontig_index():
    index = Index(tm.rands_array(10, 100))

    values = Series(np.random.randn(50), index=index[::2])
    labels = np.random.randint(0, 5, 50)

    # it works!
    grouped = values.groupby(labels)

    # accessing the index elements causes segfault
    f = lambda x: len(set(map(id, x.index)))
    grouped.agg(f)


def test_convert_objects_leave_decimal_alone():

    s = Series(range(5))
    labels = np.array(["a", "b", "c", "d", "e"], dtype="O")

    def convert_fast(x):
        return Decimal(str(x.mean()))

    def convert_force_pure(x):
        # base will be length 0
        assert len(x.values.base) > 0
        return Decimal(str(x.mean()))

    grouped = s.groupby(labels)

    result = grouped.agg(convert_fast)
    assert result.dtype == np.object_
    assert isinstance(result[0], Decimal)

    result = grouped.agg(convert_force_pure)
    assert result.dtype == np.object_
    assert isinstance(result[0], Decimal)


def test_groupby_dtype_inference_empty():
    # GH 6733
    df = DataFrame({"x": [], "range": np.arange(0, dtype="int64")})
    assert df["x"].dtype == np.float64

    result = df.groupby("x").first()
    exp_index = Index([], name="x", dtype=np.float64)
    expected = DataFrame({"range": Series([], index=exp_index, dtype="int64")})
    tm.assert_frame_equal(result, expected, by_blocks=True)


def test_groupby_unit64_float_conversion():
    #  GH: 30859 groupby converts unit64 to floats sometimes
    df = DataFrame({"first": [1], "second": [1], "value": [16148277970000000000]})
    result = df.groupby(["first", "second"])["value"].max()
    expected = Series(
        [16148277970000000000],
        MultiIndex.from_product([[1], [1]], names=["first", "second"]),
        name="value",
    )
    tm.assert_series_equal(result, expected)


def test_groupby_list_infer_array_like(df):
    result = df.groupby(list(df["A"])).mean()
    expected = df.groupby(df["A"]).mean()
    tm.assert_frame_equal(result, expected, check_names=False)

    with pytest.raises(KeyError, match=r"^'foo'$"):
        df.groupby(list(df["A"][:-1]))

    # pathological case of ambiguity
    df = DataFrame({"foo": [0, 1], "bar": [3, 4], "val": np.random.randn(2)})

    result = df.groupby(["foo", "bar"]).mean()
    expected = df.groupby([df["foo"], df["bar"]]).mean()[["val"]]


def test_groupby_keys_same_size_as_index():
    # GH 11185
    freq = "s"
    index = date_range(
        start=Timestamp("2015-09-29T11:34:44-0700"), periods=2, freq=freq
    )
    df = DataFrame([["A", 10], ["B", 15]], columns=["metric", "values"], index=index)
    result = df.groupby([Grouper(level=0, freq=freq), "metric"]).mean()
    expected = df.set_index([df.index, "metric"]).astype(float)

    tm.assert_frame_equal(result, expected)


def test_groupby_one_row():
    # GH 11741
    msg = r"^'Z'$"
    df1 = DataFrame(np.random.randn(1, 4), columns=list("ABCD"))
    with pytest.raises(KeyError, match=msg):
        df1.groupby("Z")
    df2 = DataFrame(np.random.randn(2, 4), columns=list("ABCD"))
    with pytest.raises(KeyError, match=msg):
        df2.groupby("Z")


def test_groupby_nat_exclude():
    # GH 6992
    df = DataFrame(
        {
            "values": np.random.randn(8),
            "dt": [
                np.nan,
                Timestamp("2013-01-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-01-01"),
            ],
            "str": [np.nan, "a", np.nan, "a", np.nan, "a", np.nan, "b"],
        }
    )
    grouped = df.groupby("dt")

    expected = [Index([1, 7]), Index([3, 5])]
    keys = sorted(grouped.groups.keys())
    assert len(keys) == 2
    for k, e in zip(keys, expected):
        # grouped.groups keys are np.datetime64 with system tz
        # not to be affected by tz, only compare values
        tm.assert_index_equal(grouped.groups[k], e)

    # confirm obj is not filtered
    tm.assert_frame_equal(grouped.grouper.groupings[0].obj, df)
    assert grouped.ngroups == 2

    expected = {
        Timestamp("2013-01-01 00:00:00"): np.array([1, 7], dtype=np.intp),
        Timestamp("2013-02-01 00:00:00"): np.array([3, 5], dtype=np.intp),
    }

    for k in grouped.indices:
        tm.assert_numpy_array_equal(grouped.indices[k], expected[k])

    tm.assert_frame_equal(grouped.get_group(Timestamp("2013-01-01")), df.iloc[[1, 7]])
    tm.assert_frame_equal(grouped.get_group(Timestamp("2013-02-01")), df.iloc[[3, 5]])

    with pytest.raises(KeyError, match=r"^NaT$"):
        grouped.get_group(pd.NaT)

    nan_df = DataFrame(
        {"nan": [np.nan, np.nan, np.nan], "nat": [pd.NaT, pd.NaT, pd.NaT]}
    )
    assert nan_df["nan"].dtype == "float64"
    assert nan_df["nat"].dtype == "datetime64[ns]"

    for key in ["nan", "nat"]:
        grouped = nan_df.groupby(key)
        assert grouped.groups == {}
        assert grouped.ngroups == 0
        assert grouped.indices == {}
        with pytest.raises(KeyError, match=r"^nan$"):
            grouped.get_group(np.nan)
        with pytest.raises(KeyError, match=r"^NaT$"):
            grouped.get_group(pd.NaT)


def test_groupby_two_group_keys_all_nan():
    # GH #36842: Grouping over two group keys shouldn't raise an error
    df = DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan], "c": [1, 2]})
    result = df.groupby(["a", "b"]).indices
    assert result == {}


def test_groupby_2d_malformed():
    d = DataFrame(index=range(2))
    d["group"] = ["g1", "g2"]
    d["zeros"] = [0, 0]
    d["ones"] = [1, 1]
    d["label"] = ["l1", "l2"]
    tmp = d.groupby(["group"]).mean()
    res_values = np.array([[0.0, 1.0], [0.0, 1.0]])
    tm.assert_index_equal(tmp.columns, Index(["zeros", "ones"]))
    tm.assert_numpy_array_equal(tmp.values, res_values)


def test_int32_overflow():
    B = np.concatenate((np.arange(10000), np.arange(10000), np.arange(5000)))
    A = np.arange(25000)
    df = DataFrame({"A": A, "B": B, "C": A, "D": B, "E": np.random.randn(25000)})

    left = df.groupby(["A", "B", "C", "D"]).sum()
    right = df.groupby(["D", "C", "B", "A"]).sum()
    assert len(left) == len(right)


def test_groupby_sort_multi():
    df = DataFrame(
        {
            "a": ["foo", "bar", "baz"],
            "b": [3, 2, 1],
            "c": [0, 1, 2],
            "d": np.random.randn(3),
        }
    )

    tups = [tuple(row) for row in df[["a", "b", "c"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["a", "b", "c"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[1, 2, 0]])

    tups = [tuple(row) for row in df[["c", "a", "b"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["c", "a", "b"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups)

    tups = [tuple(x) for x in df[["b", "c", "a"]].values]
    tups = com.asarray_tuplesafe(tups)
    result = df.groupby(["b", "c", "a"], sort=True).sum()
    tm.assert_numpy_array_equal(result.index.values, tups[[2, 1, 0]])

    df = DataFrame(
        {"a": [0, 1, 2, 0, 1, 2], "b": [0, 0, 0, 1, 1, 1], "d": np.random.randn(6)}
    )
    grouped = df.groupby(["a", "b"])["d"]
    result = grouped.sum()

    def _check_groupby(df, result, keys, field, f=lambda x: x.sum()):
        tups = [tuple(row) for row in df[keys].values]
        tups = com.asarray_tuplesafe(tups)
        expected = f(df.groupby(tups)[field])
        for k, v in expected.items():
            assert result[k] == v

    _check_groupby(df, result, ["a", "b"], "d")


def test_dont_clobber_name_column():
    df = DataFrame(
        {"key": ["a", "a", "a", "b", "b", "b"], "name": ["foo", "bar", "baz"] * 2}
    )

    result = df.groupby("key").apply(lambda x: x)
    tm.assert_frame_equal(result, df)


def test_skip_group_keys():

    tsf = tm.makeTimeDataFrame()

    grouped = tsf.groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values(by="A")[:3])

    pieces = [group.sort_values(by="A")[:3] for key, group in grouped]

    expected = pd.concat(pieces)
    tm.assert_frame_equal(result, expected)

    grouped = tsf["A"].groupby(lambda x: x.month, group_keys=False)
    result = grouped.apply(lambda x: x.sort_values()[:3])

    pieces = [group.sort_values()[:3] for key, group in grouped]

    expected = pd.concat(pieces)
    tm.assert_series_equal(result, expected)


def test_no_nonsense_name(float_frame):
    # GH #995
    s = float_frame["C"].copy()
    s.name = None

    result = s.groupby(float_frame["A"]).agg(np.sum)
    assert result.name is None


def test_multifunc_sum_bug():
    # GH #1065
    x = DataFrame(np.arange(9).reshape(3, 3))
    x["test"] = 0
    x["fl"] = [1.3, 1.5, 1.6]

    grouped = x.groupby("test")
    result = grouped.agg({"fl": "sum", 2: "size"})
    assert result["fl"].dtype == np.float64


def test_handle_dict_return_value(df):
    def f(group):
        return {"max": group.max(), "min": group.min()}

    def g(group):
        return Series({"max": group.max(), "min": group.min()})

    result = df.groupby("A")["C"].apply(f)
    expected = df.groupby("A")["C"].apply(g)

    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("grouper", ["A", ["A", "B"]])
def test_set_group_name(df, grouper):
    def f(group):
        assert group.name is not None
        return group

    def freduce(group):
        assert group.name is not None
        return group.sum()

    def foo(x):
        return freduce(x)

    grouped = df.groupby(grouper)

    # make sure all these work
    grouped.apply(f)
    grouped.aggregate(freduce)
    grouped.aggregate({"C": freduce, "D": freduce})
    grouped.transform(f)

    grouped["C"].apply(f)
    grouped["C"].aggregate(freduce)
    grouped["C"].aggregate([freduce, foo])
    grouped["C"].transform(f)


def test_group_name_available_in_inference_pass():
    # gh-15062
    df = DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": np.arange(6)})

    names = []

    def f(group):
        names.append(group.name)
        return group.copy()

    df.groupby("a", sort=False, group_keys=False).apply(f)

    expected_names = [0, 1, 2]
    assert names == expected_names


def test_no_dummy_key_names(df):
    # see gh-1291
    result = df.groupby(df["A"].values).sum()
    assert result.index.name is None

    result = df.groupby([df["A"].values, df["B"].values]).sum()
    assert result.index.names == (None, None)


def test_groupby_sort_multiindex_series():
    # series multiindex groupby sort argument was not being passed through
    # _compress_group_index
    # GH 9444
    index = MultiIndex(
        levels=[[1, 2], [1, 2]],
        codes=[[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]],
        names=["a", "b"],
    )
    mseries = Series([0, 1, 2, 3, 4, 5], index=index)
    index = MultiIndex(
        levels=[[1, 2], [1, 2]], codes=[[0, 0, 1], [1, 0, 0]], names=["a", "b"]
    )
    mseries_result = Series([0, 2, 4], index=index)

    result = mseries.groupby(level=["a", "b"], sort=False).first()
    tm.assert_series_equal(result, mseries_result)
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function():

    periods = 1000
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    def agg_before(func, fix=False):
        """
        Run an aggregate func on the subset of data.
        """

        def _func(data):
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            if fix:
                data[data.index[0]]
            if len(d) == 0:
                return None
            return func(d)

        return _func

    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    closure_good = grouped.agg({"high": agg_before(np.max, True)})

    tm.assert_frame_equal(closure_bad, closure_good)


def test_groupby_multiindex_missing_pair():
    # GH9049
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    df = df.set_index(["group1", "group2"])
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)

    res = df_grouped.agg("sum")
    idx = MultiIndex.from_tuples(
        [("a", "c"), ("a", "d"), ("b", "c")], names=["group1", "group2"]
    )
    exp = DataFrame([[2], [1], [5]], index=idx, columns=["value"])

    tm.assert_frame_equal(res, exp)


def test_groupby_multiindex_not_lexsorted():
    # GH 11640

    # define the lexsorted version
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")], names=["b", "c"]
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()

    # define the non-lexsorted version
    not_lexsorted_df = DataFrame(
        columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
    )
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()

    # compare the results
    tm.assert_frame_equal(lexsorted_df, not_lexsorted_df)

    expected = lexsorted_df.groupby("a").mean()
    with tm.assert_produces_warning(PerformanceWarning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)

    # a transforming function should work regardless of sort
    # GH 14776
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    assert not df.index._is_lexsorted()

    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(level=level, sort=sort).apply(DataFrame.drop_duplicates)
            expected = df
            tm.assert_frame_equal(expected, result)

            result = (
                df.sort_index()
                .groupby(level=level, sort=sort)
                .apply(DataFrame.drop_duplicates)
            )
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location():
    # checking we don't have any label/location confusion in the
    # wake of GH5375
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    g = df.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)

    ser = df[0]
    g = ser.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)

    #  and again, with a generic Index of floats
    df.index = df.index.astype(float)
    g = df.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = df.iloc[[1, 3, 4]]
    tm.assert_frame_equal(actual, expected)

    ser = df[0]
    g = ser.groupby(list("ababb"))
    actual = g.filter(lambda x: len(x) > 2)
    expected = ser.take([1, 3, 4])
    tm.assert_series_equal(actual, expected)


def test_transform_doesnt_clobber_ints():
    # GH 7972
    n = 6
    x = np.arange(n)
    df = DataFrame({"a": x // 2, "b": 2.0 * x, "c": 3.0 * x})
    df2 = DataFrame({"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x})

    gb = df.groupby("a")
    result = gb.transform("mean")

    gb2 = df2.groupby("a")
    expected = gb2.transform("mean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column", ["int_groups", "string_groups", ["int_groups", "string_groups"]]
)
def test_groupby_preserves_sort(sort_column, group_column):
    # Test to ensure that groupby always preserves sort order of original
    # object. Issue #8588 and #9651

    df = DataFrame(
        {
            "int_groups": [3, 1, 0, 1, 0, 3, 3, 3],
            "string_groups": ["z", "a", "z", "a", "a", "g", "g", "g"],
            "ints": [8, 7, 4, 5, 2, 9, 1, 1],
            "floats": [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
            "strings": ["z", "d", "a", "e", "word", "word2", "42", "47"],
        }
    )

    # Try sorting on different types and with different group types

    df = df.sort_values(by=sort_column)
    g = df.groupby(group_column)

    def test_sort(x):
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))

    g.apply(test_sort)


def test_pivot_table_values_key_error():
    # This test is designed to replicate the error in issue #14938
    df = DataFrame(
        {
            "eventDate": date_range(datetime.today(), periods=20, freq="M").tolist(),
            "thename": range(0, 20),
        }
    )

    df["year"] = df.set_index("eventDate").index.year
    df["month"] = df.set_index("eventDate").index.month

    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(
            index="year", columns="month", values="badname", aggfunc="count"
        )


@pytest.mark.parametrize("columns", ["C", ["C"]])
@pytest.mark.parametrize("keys", [["A"], ["A", "B"]])
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range(0, 1, 1, tz="US/Eastern"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
)
@pytest.mark.parametrize("method", ["attr", "agg", "apply"])
@pytest.mark.parametrize(
    "op", ["idxmax", "idxmin", "mad", "min", "max", "sum", "prod", "skew"]
)
@pytest.mark.filterwarnings("ignore:Dropping invalid columns:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*Select only valid:FutureWarning")
def test_empty_groupby(columns, keys, values, method, op, request):
    # GH8093 & GH26411
    override_dtype = None

    if (
        isinstance(values, Categorical)
        and not isinstance(columns, list)
        and op in ["sum", "prod"]
        and method != "apply"
    ):
        # handled below GH#41291
        pass
    elif isinstance(values, Categorical) and len(keys) == 1 and method == "apply":
        mark = pytest.mark.xfail(raises=TypeError, match="'str' object is not callable")
        request.node.add_marker(mark)
    elif (
        isinstance(values, Categorical)
        and len(keys) == 1
        and op in ["idxmax", "idxmin"]
    ):
        mark = pytest.mark.xfail(
            raises=ValueError, match="attempt to get arg(min|max) of an empty sequence"
        )
        request.node.add_marker(mark)
    elif (
        isinstance(values, Categorical)
        and len(keys) == 1
        and not isinstance(columns, list)
    ):
        mark = pytest.mark.xfail(
            raises=TypeError, match="'Categorical' does not implement"
        )
        request.node.add_marker(mark)
    elif (
        isinstance(values, Categorical)
        and len(keys) == 1
        and op in ["mad", "min", "max", "sum", "prod", "skew"]
    ):
        mark = pytest.mark.xfail(
            raises=AssertionError, match="(DataFrame|Series) are different"
        )
        request.node.add_marker(mark)
    elif (
        isinstance(values, Categorical)
        and len(keys) == 2
        and op in ["min", "max", "sum"]
        and method != "apply"
    ):
        mark = pytest.mark.xfail(
            raises=AssertionError, match="(DataFrame|Series) are different"
        )
        request.node.add_marker(mark)
    elif (
        isinstance(values, pd.core.arrays.BooleanArray)
        and op in ["sum", "prod"]
        and method != "apply"
    ):
        # We expect to get Int64 back for these
        override_dtype = "Int64"

    if isinstance(values[0], bool) and op in ("prod", "sum") and method != "apply":
        # sum/product of bools is an integer
        override_dtype = "int64"

    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))

    if hasattr(values, "dtype"):
        # check that we did the construction right
        assert (df.dtypes == values.dtype).all()

    df = df.iloc[:0]

    gb = df.groupby(keys)[columns]

    def get_result():
        if method == "attr":
            return getattr(gb, op)()
        else:
            return getattr(gb, method)(op)

    if columns == "C":
        # i.e. SeriesGroupBy
        if op in ["prod", "sum"]:
            # ops that require more than just ordered-ness
            if method != "apply":
                # FIXME: apply goes through different code path
                if df.dtypes[0].kind == "M":
                    # GH#41291
                    # datetime64 -> prod and sum are invalid
                    msg = "datetime64 type does not support"
                    with pytest.raises(TypeError, match=msg):
                        get_result()

                    return
                elif isinstance(values, Categorical):
                    # GH#41291
                    msg = "category type does not support"
                    with pytest.raises(TypeError, match=msg):
                        get_result()

                    return
    else:
        # ie. DataFrameGroupBy
        if op in ["prod", "sum"]:
            # ops that require more than just ordered-ness
            if method != "apply":
                # FIXME: apply goes through different code path
                if df.dtypes[0].kind == "M":
                    # GH#41291
                    # datetime64 -> prod and sum are invalid
                    result = get_result()

                    # with numeric_only=True, these are dropped, and we get
                    # an empty DataFrame back
                    expected = df.set_index(keys)[[]]
                    tm.assert_equal(result, expected)
                    return

                elif isinstance(values, Categorical):
                    # GH#41291
                    # Categorical doesn't implement sum or prod
                    result = get_result()

                    # with numeric_only=True, these are dropped, and we get
                    # an empty DataFrame back
                    expected = df.set_index(keys)[[]]
                    if len(keys) != 1 and op == "prod":
                        # TODO: why just prod and not sum?
                        # Categorical is special without 'observed=True'
                        lev = Categorical([0], dtype=values.dtype)
                        mi = MultiIndex.from_product([lev, lev], names=["A", "B"])
                        expected = DataFrame([], columns=[], index=mi)

                    tm.assert_equal(result, expected)
                    return

                elif df.dtypes[0] == object:
                    # FIXME: the test is actually wrong here, xref #41341
                    result = get_result()
                    # In this case we have list-of-list, will raise TypeError,
                    # and subsequently be dropped as nuisance columns
                    expected = df.set_index(keys)[[]]
                    tm.assert_equal(result, expected)
                    return

    result = get_result()
    expected = df.set_index(keys)[columns]
    if override_dtype is not None:
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        expected.index.name = keys[0]
    tm.assert_equal(result, expected)


def test_tuple_as_grouping():
    # https://github.com/pandas-dev/pandas/issues/18314
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )

    with pytest.raises(KeyError, match=r"('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))

    result = df.groupby(("a", "b"))["c"].sum()
    expected = Series([4], name="c", index=Index([1], name=("a", "b")))
    tm.assert_series_equal(result, expected)


def test_tuple_correct_keyerror():
    # https://github.com/pandas-dev/pandas/issues/18798
    df = DataFrame(1, index=range(3), columns=MultiIndex.from_product([[1, 2], [3, 4]]))
    with pytest.raises(KeyError, match=r"^\(7, 8\)$"):
        df.groupby((7, 8)).mean()


def test_groupby_agg_ohlc_non_first():
    # GH 21716
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    expected = DataFrame(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        columns=MultiIndex.from_tuples(
            (
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ),
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])

    tm.assert_frame_equal(result, expected)


def test_groupby_multiindex_nat():
    # GH 9236
    values = [
        (pd.NaT, "a"),
        (datetime(2012, 1, 2), "a"),
        (datetime(2012, 1, 2), "b"),
        (datetime(2012, 1, 3), "a"),
    ]
    mi = MultiIndex.from_tuples(values, names=["date", None])
    ser = Series([3, 2, 2.5, 4], index=mi)

    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=["a", "b"])
    tm.assert_series_equal(result, expected)


def test_groupby_empty_list_raises():
    # GH 5289
    values = zip(range(10), range(10))
    df = DataFrame(values, columns=["apple", "b"])
    msg = "Grouper and axis must be same length"
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])


def test_groupby_multiindex_series_keys_len_equal_group_axis():
    # GH 25704
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series(data=[1, 2], index=ri)
    result = s.groupby(["first", "third"]).sum()

    index_array = [["x"], ["k"]]
    index_names = ["first", "third"]
    ei = MultiIndex.from_arrays(index_array, names=index_names)
    expected = Series([3], index=ei)

    tm.assert_series_equal(result, expected)


def test_groupby_groups_in_BaseGrouper():
    # GH 26326
    # Test if DataFrame grouped with a pandas.Grouper has correct groups
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    df = DataFrame({"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi)
    result = df.groupby([Grouper(level="alpha"), "beta"])
    expected = df.groupby(["alpha", "beta"])
    assert result.groups == expected.groups

    result = df.groupby(["beta", Grouper(level="alpha")])
    expected = df.groupby(["beta", "alpha"])
    assert result.groups == expected.groups


@pytest.mark.parametrize("group_name", ["x", ["x"]])
def test_groupby_axis_1(group_name):
    # GH 27614
    df = DataFrame(
        np.arange(12).reshape(3, 4), index=[0, 1, 0], columns=[10, 20, 10, 20]
    )
    df.index.name = "y"
    df.columns.name = "x"

    results = df.groupby(group_name, axis=1).sum()
    expected = df.T.groupby(group_name).sum().T
    tm.assert_frame_equal(results, expected)

    # test on MI column
    iterables = [["bar", "baz", "foo"], ["one", "two"]]
    mi = MultiIndex.from_product(iterables=iterables, names=["x", "x1"])
    df = DataFrame(np.arange(18).reshape(3, 6), index=[0, 1, 0], columns=mi)
    results = df.groupby(group_name, axis=1).sum()
    expected = df.T.groupby(group_name).sum().T
    tm.assert_frame_equal(results, expected)


@pytest.mark.parametrize(
    "op, expected",
    [
        (
            "shift",
            {
                "time": [
                    None,
                    None,
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    None,
                    None,
                ]
            },
        ),
        (
            "bfill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
        (
            "ffill",
            {
                "time": [
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 12:00:00"),
                    Timestamp("2019-01-01 12:30:00"),
                    Timestamp("2019-01-01 14:00:00"),
                    Timestamp("2019-01-01 14:30:00"),
                ]
            },
        ),
    ],
)
def test_shift_bfill_ffill_tz(tz_naive_fixture, op, expected):
    # GH19995, GH27992: Check that timezone does not drop in shift, bfill, and ffill
    tz = tz_naive_fixture
    data = {
        "id": ["A", "B", "A", "B", "A", "B"],
        "time": [
            Timestamp("2019-01-01 12:00:00"),
            Timestamp("2019-01-01 12:30:00"),
            None,
            None,
            Timestamp("2019-01-01 14:00:00"),
            Timestamp("2019-01-01 14:30:00"),
        ],
    }
    df = DataFrame(data).assign(time=lambda x: x.time.dt.tz_localize(tz))

    grouped = df.groupby("id")
    result = getattr(grouped, op)()
    expected = DataFrame(expected).assign(time=lambda x: x.time.dt.tz_localize(tz))
    tm.assert_frame_equal(result, expected)


def test_groupby_only_none_group():
    # see GH21624
    # this was crashing with "ValueError: Length of passed values is 1, index implies 0"
    df = DataFrame({"g": [None], "x": 1})
    actual = df.groupby("g")["x"].transform("sum")
    expected = Series([np.nan], name="x")

    tm.assert_series_equal(actual, expected)


def test_groupby_duplicate_index():
    # GH#29189 the groupby call here used to raise
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    gb = ser.groupby(level=0)

    result = gb.mean()
    expected = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "idx",
    [
        Index(["a", "a"], name="foo"),
        MultiIndex.from_tuples((("a", "a"), ("a", "a")), names=["foo", "bar"]),
    ],
)
@pytest.mark.filterwarnings("ignore:tshift is deprecated:FutureWarning")
def test_dup_labels_output_shape(groupby_func, idx):
    if groupby_func in {"size", "ngroup", "cumcount"}:
        pytest.skip("Not applicable")

    df = DataFrame([[1, 1]], columns=idx)
    grp_by = df.groupby([0])

    args = []
    if groupby_func in {"fillna", "nth"}:
        args.append(0)
    elif groupby_func == "corrwith":
        args.append(df)
    elif groupby_func == "tshift":
        df.index = [Timestamp("today")]
        args.extend([1, "D"])

    result = getattr(grp_by, groupby_func)(*args)

    assert result.shape == (1, 2)
    tm.assert_index_equal(result.columns, idx)


def test_groupby_crash_on_nunique(axis):
    # Fix following 30253
    dti = date_range("2016-01-01", periods=2, name="foo")
    df = DataFrame({("A", "B"): [1, 2], ("A", "C"): [1, 3], ("D", "B"): [0, 0]})
    df.columns.names = ("bar", "baz")
    df.index = dti

    axis_number = df._get_axis_number(axis)
    if not axis_number:
        df = df.T

    gb = df.groupby(axis=axis_number, level=0)
    result = gb.nunique()

    expected = DataFrame({"A": [1, 2], "D": [1, 1]}, index=dti)
    expected.columns.name = "bar"
    if not axis_number:
        expected = expected.T

    tm.assert_frame_equal(result, expected)

    if axis_number == 0:
        # same thing, but empty columns
        gb2 = df[[]].groupby(axis=axis_number, level=0)
        exp = expected[[]]
    else:
        # same thing, but empty rows
        gb2 = df.loc[[]].groupby(axis=axis_number, level=0)
        # default for empty when we can't infer a dtype is float64
        exp = expected.loc[[]].astype(np.float64)

    res = gb2.nunique()
    tm.assert_frame_equal(res, exp)


def test_groupby_list_level():
    # GH 9790
    expected = DataFrame(np.arange(0, 9).reshape(3, 3), dtype=float)
    result = expected.groupby(level=[0]).mean()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "max_seq_items, expected",
    [
        (5, "{0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}"),
        (4, "{0: [0], 1: [1], 2: [2], 3: [3], ...}"),
        (1, "{0: [0], ...}"),
    ],
)
def test_groups_repr_truncates(max_seq_items, expected):
    # GH 1135
    df = DataFrame(np.random.randn(5, 1))
    df["a"] = df.index

    with pd.option_context("display.max_seq_items", max_seq_items):
        result = df.groupby("a").groups.__repr__()
        assert result == expected

        result = df.groupby(np.array(df.a)).groups.__repr__()
        assert result == expected


def test_group_on_two_row_multiindex_returns_one_tuple_key():
    # GH 18451
    df = DataFrame([{"a": 1, "b": 2, "c": 99}, {"a": 1, "b": 2, "c": 88}])
    df = df.set_index(["a", "b"])

    grp = df.groupby(["a", "b"])
    result = grp.indices
    expected = {(1, 2): np.array([0, 1], dtype=np.int64)}

    assert len(result) == 1
    key = (1, 2)
    assert (result[key] == expected[key]).all()


@pytest.mark.parametrize(
    "klass, attr, value",
    [
        (DataFrame, "level", "a"),
        (DataFrame, "as_index", False),
        (DataFrame, "sort", False),
        (DataFrame, "group_keys", False),
        (DataFrame, "squeeze", True),
        (DataFrame, "observed", True),
        (DataFrame, "dropna", False),
        pytest.param(
            Series,
            "axis",
            1,
            marks=pytest.mark.xfail(
                reason="GH 35443: Attribute currently not passed on to series"
            ),
        ),
        (Series, "level", "a"),
        (Series, "as_index", False),
        (Series, "sort", False),
        (Series, "group_keys", False),
        (Series, "squeeze", True),
        (Series, "observed", True),
        (Series, "dropna", False),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:The `squeeze` parameter is deprecated:FutureWarning"
)
def test_subsetting_columns_keeps_attrs(klass, attr, value):
    # GH 9959 - When subsetting columns, don't drop attributes
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    if attr != "axis":
        df = df.set_index("a")

    expected = df.groupby("a", **{attr: value})
    result = expected[["b"]] if klass is DataFrame else expected["b"]
    assert getattr(result, attr) == getattr(expected, attr)


def test_subsetting_columns_axis_1():
    # GH 37725
    g = DataFrame({"A": [1], "B": [2], "C": [3]}).groupby([0, 0, 1], axis=1)
    match = "Cannot subset columns when using axis=1"
    with pytest.raises(ValueError, match=match):
        g[["A", "B"]].sum()


@pytest.mark.parametrize("func", ["sum", "any", "shift"])
def test_groupby_column_index_name_lost(func):
    # GH: 29764 groupby loses index sometimes
    expected = Index(["a"], name="idx")
    df = DataFrame([[1]], columns=expected)
    df_grouped = df.groupby([1])
    result = getattr(df_grouped, func)().columns
    tm.assert_index_equal(result, expected)


def test_groupby_duplicate_columns():
    # GH: 31735
    df = DataFrame(
        {"A": ["f", "e", "g", "h"], "B": ["a", "b", "c", "d"], "C": [1, 2, 3, 4]}
    ).astype(object)
    df.columns = ["A", "B", "B"]
    result = df.groupby([0, 0, 0, 0]).min()
    expected = DataFrame([["e", "a", 1]], columns=["A", "B", "B"])
    tm.assert_frame_equal(result, expected)


def test_groupby_series_with_tuple_name():
    # GH 37755
    ser = Series([1, 2, 3, 4], index=[1, 1, 2, 2], name=("a", "a"))
    ser.index.name = ("b", "b")
    result = ser.groupby(level=0).last()
    expected = Series([2, 4], index=[1, 2], name=("a", "a"))
    expected.index.name = ("b", "b")
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(not IS64, reason="GH#38778: fail on 32-bit system")
@pytest.mark.parametrize(
    "func, values", [("sum", [97.0, 98.0]), ("mean", [24.25, 24.5])]
)
def test_groupby_numerical_stability_sum_mean(func, values):
    # GH#38778
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = getattr(df.groupby("group"), func)()
    expected = DataFrame({"a": values, "b": values}, index=Index([1, 2], name="group"))
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(not IS64, reason="GH#38778: fail on 32-bit system")
def test_groupby_numerical_stability_cumsum():
    # GH#38934
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    result = df.groupby("group").cumsum()
    exp_data = (
        [1e16] * 2 + [1e16 + 96, 1e16 + 98] + [5e15 + 97, 5e15 + 98] + [97.0, 98.0]
    )
    expected = DataFrame({"a": exp_data, "b": exp_data})
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_groupby_mean_duplicate_index(rand_series_with_duplicate_datetimeindex):
    dups = rand_series_with_duplicate_datetimeindex
    result = dups.groupby(level=0).mean()
    expected = dups.groupby(dups.index).mean()
    tm.assert_series_equal(result, expected)


def test_groupby_all_nan_groups_drop():
    # GH 15036
    s = Series([1, 2, 3], [np.nan, np.nan, np.nan])
    result = s.groupby(s.index).sum()
    expected = Series([], index=Index([], dtype=np.float64), dtype=np.int64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_empty_multi_column(as_index, numeric_only):
    # GH 15106 & GH 41998
    df = DataFrame(data=[], columns=["A", "B", "C"])
    gb = df.groupby(["A", "B"], as_index=as_index)
    result = gb.sum(numeric_only=numeric_only)
    if as_index:
        index = MultiIndex([[], []], [[], []], names=["A", "B"])
        columns = ["C"] if not numeric_only else []
    else:
        index = RangeIndex(0)
        columns = ["A", "B", "C"] if not numeric_only else ["A", "B"]
    expected = DataFrame([], columns=columns, index=index)
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_non_numeric_dtype():
    # GH #43108
    df = DataFrame(
        [["M", [1]], ["M", [1]], ["W", [10]], ["W", [20]]], columns=["MW", "v"]
    )

    expected = DataFrame(
        {
            "v": [[1, 1], [10, 20]],
        },
        index=Index(["M", "W"], dtype="object", name="MW"),
    )

    gb = df.groupby(by=["MW"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_multi_non_numeric_dtype():
    # GH #42395
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": [Timedelta(i * 10, "days") for i in range(1, 6)],
        }
    )

    expected = DataFrame(
        {
            "y": [Timedelta(i, "days") for i in range(7, 9)],
            "z": [Timedelta(i * 10, "days") for i in range(7, 9)],
        },
        index=Index([0, 1], dtype="int64", name="x"),
    )

    gb = df.groupby(by=["x"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_aggregation_numeric_with_non_numeric_dtype():
    # GH #43108
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": list(range(1, 6)),
        }
    )

    expected = DataFrame(
        {"z": [7, 8]},
        index=Index([0, 1], dtype="int64", name="x"),
    )

    gb = df.groupby(by=["x"])
    result = gb.sum()
    tm.assert_frame_equal(result, expected)


def test_groupby_filtered_df_std():
    # GH 16174
    dicts = [
        {"filter_col": False, "groupby_col": True, "bool_col": True, "float_col": 10.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 20.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 30.5},
    ]
    df = DataFrame(dicts)

    df_filter = df[df["filter_col"] == True]  # noqa:E712
    dfgb = df_filter.groupby("groupby_col")
    result = dfgb.std()
    expected = DataFrame(
        [[0.0, 0.0, 7.071068]],
        columns=["filter_col", "bool_col", "float_col"],
        index=Index([True], name="groupby_col"),
    )
    tm.assert_frame_equal(result, expected)


def test_datetime_categorical_multikey_groupby_indices():
    # GH 26859
    df = DataFrame(
        {
            "a": Series(list("abc")),
            "b": Series(
                to_datetime(["2018-01-01", "2018-02-01", "2018-03-01"]),
                dtype="category",
            ),
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    result = df.groupby(["a", "b"]).indices
    expected = {
        ("a", Timestamp("2018-01-01 00:00:00")): np.array([0]),
        ("b", Timestamp("2018-02-01 00:00:00")): np.array([1]),
        ("c", Timestamp("2018-03-01 00:00:00")): np.array([2]),
    }
    assert result == expected
