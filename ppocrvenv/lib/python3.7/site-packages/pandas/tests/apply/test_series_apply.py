from collections import (
    Counter,
    defaultdict,
)
from itertools import chain

import numpy as np
import pytest

from pandas.core.dtypes.common import is_number

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
    isna,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels


def test_series_map_box_timedelta():
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=5, freq="h"))

    def f(x):
        return x.total_seconds()

    ser.map(f)
    ser.apply(f)
    DataFrame(ser).applymap(f)


def test_apply(datetime_series):
    with np.errstate(all="ignore"):
        tm.assert_series_equal(datetime_series.apply(np.sqrt), np.sqrt(datetime_series))

        # element-wise apply
        import math

        tm.assert_series_equal(datetime_series.apply(math.exp), np.exp(datetime_series))

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    rs = s.apply(lambda x: x)
    tm.assert_series_equal(s, rs)

    # check all metadata (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.apply(lambda x: x)
    tm.assert_series_equal(s, rs)


def test_apply_same_length_inference_bug():
    s = Series([1, 2])

    def f(x):
        return (x, x + 1)

    result = s.apply(f)
    expected = s.map(f)
    tm.assert_series_equal(result, expected)

    s = Series([1, 2, 3])
    result = s.apply(f)
    expected = s.map(f)
    tm.assert_series_equal(result, expected)


def test_apply_dont_convert_dtype():
    s = Series(np.random.randn(10))

    def f(x):
        return x if x > 0 else np.nan

    result = s.apply(f, convert_dtype=False)
    assert result.dtype == object


def test_with_string_args(datetime_series):

    for arg in ["sum", "mean", "min", "max", "std"]:
        result = datetime_series.apply(arg)
        expected = getattr(datetime_series, arg)()
        assert result == expected


def test_apply_args():
    s = Series(["foo,bar"])

    result = s.apply(str.split, args=(",",))
    assert result[0] == ["foo", "bar"]
    assert isinstance(result[0], list)


@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args, kwargs, increment):
    # GH 43357
    def f(x, a=0, b=0, c=0):
        return x + a + 10 * b + 100 * c

    s = Series([1, 2])
    result = s.agg(f, 0, *args, **kwargs)
    expected = s + increment
    tm.assert_series_equal(result, expected)


def test_series_map_box_timestamps():
    # GH#2689, GH#2627
    ser = Series(pd.date_range("1/1/2000", periods=10))

    def func(x):
        return (x.hour, x.day, x.month)

    # it works!
    ser.map(func)
    ser.apply(func)


def test_series_map_stringdtype(any_string_dtype):
    # map test on StringDType, GH#40823
    ser1 = Series(
        data=["cat", "dog", "rabbit"],
        index=["id1", "id2", "id3"],
        dtype=any_string_dtype,
    )
    ser2 = Series(data=["id3", "id2", "id1", "id7000"], dtype=any_string_dtype)
    result = ser2.map(ser1)
    expected = Series(data=["rabbit", "dog", "cat", pd.NA], dtype=any_string_dtype)

    tm.assert_series_equal(result, expected)


def test_apply_box():
    # ufunc will not be boxed. Same test cases as the test_map_box
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    s = Series(vals)
    assert s.dtype == "datetime64[ns]"
    # boxed value must be Timestamp instance
    res = s.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)

    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    s = Series(vals)
    assert s.dtype == "datetime64[ns, US/Eastern]"
    res = s.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)

    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    s = Series(vals)
    assert s.dtype == "timedelta64[ns]"
    res = s.apply(lambda x: f"{type(x).__name__}_{x.days}")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)

    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    s = Series(vals)
    assert s.dtype == "Period[M]"
    res = s.apply(lambda x: f"{type(x).__name__}_{x.freqstr}")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


def test_apply_datetimetz():
    values = pd.date_range("2011-01-01", "2011-01-02", freq="H").tz_localize(
        "Asia/Tokyo"
    )
    s = Series(values, name="XX")

    result = s.apply(lambda x: x + pd.offsets.Day())
    exp_values = pd.date_range("2011-01-02", "2011-01-03", freq="H").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    # change dtype
    # GH 14506 : Returned dtype changed from int32 to int64
    result = s.apply(lambda x: x.hour)
    exp = Series(list(range(24)) + [0], name="XX", dtype=np.int64)
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x):
        if not isinstance(x, pd.Timestamp):
            raise ValueError
        return str(x.tz)

    result = s.map(f)
    exp = Series(["Asia/Tokyo"] * 25, name="XX")
    tm.assert_series_equal(result, exp)


def test_apply_categorical():
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    ser = Series(values, name="XX", index=list("abcdefg"))
    result = ser.apply(lambda x: x.lower())

    # should be categorical dtype when the number of categories are
    # the same
    values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp.values)

    result = ser.apply(lambda x: "A")
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object


@pytest.mark.parametrize("series", [["1-1", "1-1", np.NaN], ["1-1", "1-2", np.NaN]])
def test_apply_categorical_with_nan_values(series):
    # GH 20714 bug fixed in: GH 24275
    s = Series(series, dtype="category")
    result = s.apply(lambda x: x.split("-")[0])
    result = result.astype(object)
    expected = Series(["1", "1", np.NaN], dtype="category")
    expected = expected.astype(object)
    tm.assert_series_equal(result, expected)


def test_apply_empty_integer_series_with_datetime_index():
    # GH 21245
    s = Series([], index=pd.date_range(start="2018-01-01", periods=0), dtype=int)
    result = s.apply(lambda x: x)
    tm.assert_series_equal(result, s)


def test_transform(string_series):
    # transforming functions

    with np.errstate(all="ignore"):

        f_sqrt = np.sqrt(string_series)
        f_abs = np.abs(string_series)

        # ufunc
        result = string_series.apply(np.sqrt)
        expected = f_sqrt.copy()
        tm.assert_series_equal(result, expected)

        # list-like
        result = string_series.apply([np.sqrt])
        expected = f_sqrt.to_frame().copy()
        expected.columns = ["sqrt"]
        tm.assert_frame_equal(result, expected)

        result = string_series.apply(["sqrt"])
        tm.assert_frame_equal(result, expected)

        # multiple items in list
        # these are in the order as if we are applying both functions per
        # series and then concatting
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["sqrt", "absolute"]
        result = string_series.apply([np.sqrt, np.abs])
        tm.assert_frame_equal(result, expected)

        # dict, provide renaming
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["foo", "bar"]
        expected = expected.unstack().rename("series")

        result = string_series.apply({"foo": np.sqrt, "bar": np.abs})
        tm.assert_series_equal(result.reindex_like(expected), expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_partial_failure(op, request):
    # GH 35964
    if op in ("ffill", "bfill", "pad", "backfill", "shift"):
        request.node.add_marker(
            pytest.mark.xfail(
                raises=AssertionError, reason=f"{op} is successful on any dtype"
            )
        )
    if op in ("rank", "fillna"):
        pytest.skip(f"{op} doesn't raise TypeError on object")

    # Using object makes most transform kernels fail
    ser = Series(3 * [object])

    expected = ser.transform(["shift"])
    result = ser.transform([op, "shift"])
    tm.assert_equal(result, expected)

    expected = ser.transform({"B": "shift"})
    result = ser.transform({"A": op, "B": "shift"})
    tm.assert_equal(result, expected)

    expected = ser.transform({"B": ["shift"]})
    result = ser.transform({"A": [op], "B": ["shift"]})
    tm.assert_equal(result, expected)

    expected = ser.transform({"A": ["shift"], "B": [op]})
    result = ser.transform({"A": [op, "shift"], "B": [op]})
    tm.assert_equal(result, expected)


def test_transform_partial_failure_valueerror():
    # GH 40211
    match = ".*did not transform successfully and did not raise a TypeError"

    def noop(x):
        return x

    def raising_op(_):
        raise ValueError

    ser = Series(3 * [object])

    expected = ser.transform([noop])
    with tm.assert_produces_warning(FutureWarning, match=match):
        result = ser.transform([noop, raising_op])
    tm.assert_equal(result, expected)

    expected = ser.transform({"B": noop})
    with tm.assert_produces_warning(FutureWarning, match=match):
        result = ser.transform({"A": raising_op, "B": noop})
    tm.assert_equal(result, expected)

    expected = ser.transform({"B": [noop]})
    with tm.assert_produces_warning(FutureWarning, match=match):
        result = ser.transform({"A": [raising_op], "B": [noop]})
    tm.assert_equal(result, expected)

    expected = ser.transform({"A": [noop], "B": [noop]})
    with tm.assert_produces_warning(FutureWarning, match=match, check_stacklevel=False):
        result = ser.transform({"A": [noop, raising_op], "B": [noop]})
    tm.assert_equal(result, expected)


def test_demo():
    # demonstration tests
    s = Series(range(6), dtype="int64", name="series")

    result = s.agg(["min", "max"])
    expected = Series([0, 5], index=["min", "max"], name="series")
    tm.assert_series_equal(result, expected)

    result = s.agg({"foo": "min"})
    expected = Series([0], index=["foo"], name="series")
    tm.assert_series_equal(result, expected)


def test_agg_apply_evaluate_lambdas_the_same(string_series):
    # test that we are evaluating row-by-row first
    # before vectorized evaluation
    result = string_series.apply(lambda x: str(x))
    expected = string_series.agg(lambda x: str(x))
    tm.assert_series_equal(result, expected)

    result = string_series.apply(str)
    expected = string_series.agg(str)
    tm.assert_series_equal(result, expected)


def test_with_nested_series(datetime_series):
    # GH 2316
    # .agg with a reducer and a transform, what to do
    result = datetime_series.apply(lambda x: Series([x, x ** 2], index=["x", "x^2"]))
    expected = DataFrame({"x": datetime_series, "x^2": datetime_series ** 2})
    tm.assert_frame_equal(result, expected)

    result = datetime_series.agg(lambda x: Series([x, x ** 2], index=["x", "x^2"]))
    tm.assert_frame_equal(result, expected)


def test_replicate_describe(string_series):
    # this also tests a result set that is all scalars
    expected = string_series.describe()
    result = string_series.apply(
        {
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": lambda x: x.quantile(0.25),
            "50%": "median",
            "75%": lambda x: x.quantile(0.75),
            "max": "max",
        }
    )
    tm.assert_series_equal(result, expected)


def test_reduce(string_series):
    # reductions with named functions
    result = string_series.agg(["sum", "mean"])
    expected = Series(
        [string_series.sum(), string_series.mean()],
        ["sum", "mean"],
        name=string_series.name,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_non_callable_aggregates(how):
    # test agg using non-callable series attributes
    # GH 39116 - expand to apply
    s = Series([1, 2, None])

    # Calling agg w/ just a string arg same as calling s.arg
    result = getattr(s, how)("size")
    expected = s.size
    assert result == expected

    # test when mixed w/ callable reducers
    result = getattr(s, how)(["size", "count", "mean"])
    expected = Series({"size": 3.0, "count": 2.0, "mean": 1.5})
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("sum", 0),
                ("max", np.nan),
                ("min", np.nan),
                ("all", True),
                ("any", False),
                ("mean", np.nan),
                ("prod", 1),
                ("std", np.nan),
                ("var", np.nan),
                ("median", np.nan),
            ],
        ),
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("sum", 6),
                ("max", 3),
                ("min", 1),
                ("all", True),
                ("any", True),
                ("mean", 2),
                ("prod", 6),
                ("std", 1),
                ("var", 1),
                ("median", 2),
            ],
        ),
        tm.get_cython_table_params(
            Series("a b c".split()),
            [
                ("sum", "abc"),
                ("max", "c"),
                ("min", "a"),
                ("all", True),
                ("any", True),
            ],
        ),
    ),
)
def test_agg_cython_table(series, func, expected):
    # GH21224
    # test reducing functions in
    # pandas.core.base.SelectionMixin._cython_table
    result = series.agg(func)
    if is_number(expected):
        assert np.isclose(result, expected, equal_nan=True)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "series, func, expected",
    chain(
        tm.get_cython_table_params(
            Series(dtype=np.float64),
            [
                ("cumprod", Series([], Index([]), dtype=np.float64)),
                ("cumsum", Series([], Index([]), dtype=np.float64)),
            ],
        ),
        tm.get_cython_table_params(
            Series([np.nan, 1, 2, 3]),
            [
                ("cumprod", Series([np.nan, 1, 2, 6])),
                ("cumsum", Series([np.nan, 1, 3, 6])),
            ],
        ),
        tm.get_cython_table_params(
            Series("a b c".split()), [("cumsum", Series(["a", "ab", "abc"]))]
        ),
    ),
)
def test_agg_cython_table_transform(series, func, expected):
    # GH21224
    # test transforming functions in
    # pandas.core.base.SelectionMixin._cython_table (cumprod, cumsum)
    result = series.agg(func)
    tm.assert_series_equal(result, expected)


def test_series_apply_no_suffix_index():
    # GH36189
    s = Series([4] * 3)
    result = s.apply(["sum", lambda x: x.sum(), lambda x: x.sum()])
    expected = Series([12, 12, 12], index=["sum", "<lambda>", "<lambda>"])

    tm.assert_series_equal(result, expected)


def test_map(datetime_series):
    index, data = tm.getMixedTypeDict()

    source = Series(data["B"], index=data["C"])
    target = Series(data["C"][:4], index=data["D"][:4])

    merged = target.map(source)

    for k, v in merged.items():
        assert v == source[target[k]]

    # input could be a dict
    merged = target.map(source.to_dict())

    for k, v in merged.items():
        assert v == source[target[k]]

    # function
    result = datetime_series.map(lambda x: x * 2)
    tm.assert_series_equal(result, datetime_series * 2)

    # GH 10324
    a = Series([1, 2, 3, 4])
    b = Series(["even", "odd", "even", "odd"], dtype="category")
    c = Series(["even", "odd", "even", "odd"])

    exp = Series(["odd", "even", "odd", np.nan], dtype="category")
    tm.assert_series_equal(a.map(b), exp)
    exp = Series(["odd", "even", "odd", np.nan])
    tm.assert_series_equal(a.map(c), exp)

    a = Series(["a", "b", "c", "d"])
    b = Series([1, 2, 3, 4], index=pd.CategoricalIndex(["b", "c", "d", "e"]))
    c = Series([1, 2, 3, 4], index=Index(["b", "c", "d", "e"]))

    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, 1, 2, 3])
    tm.assert_series_equal(a.map(c), exp)

    a = Series(["a", "b", "c", "d"])
    b = Series(
        ["B", "C", "D", "E"],
        dtype="category",
        index=pd.CategoricalIndex(["b", "c", "d", "e"]),
    )
    c = Series(["B", "C", "D", "E"], index=Index(["b", "c", "d", "e"]))

    exp = Series(
        pd.Categorical([np.nan, "B", "C", "D"], categories=["B", "C", "D", "E"])
    )
    tm.assert_series_equal(a.map(b), exp)
    exp = Series([np.nan, "B", "C", "D"])
    tm.assert_series_equal(a.map(c), exp)


def test_map_empty(index):
    if isinstance(index, MultiIndex):
        pytest.skip("Initializing a Series from a MultiIndex is not supported")

    s = Series(index)
    result = s.map({})

    expected = Series(np.nan, index=s.index)
    tm.assert_series_equal(result, expected)


def test_map_compat():
    # related GH 8024
    s = Series([True, True, False], index=[1, 2, 3])
    result = s.map({True: "foo", False: "bar"})
    expected = Series(["foo", "foo", "bar"], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_map_int():
    left = Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4})
    right = Series({1: 11, 2: 22, 3: 33})

    assert left.dtype == np.float_
    assert issubclass(right.dtype.type, np.integer)

    merged = left.map(right)
    assert merged.dtype == np.float_
    assert isna(merged["d"])
    assert not isna(merged["c"])


def test_map_type_inference():
    s = Series(range(3))
    s2 = s.map(lambda x: np.where(x == 0, 0, 1))
    assert issubclass(s2.dtype.type, np.integer)


def test_map_decimal(string_series):
    from decimal import Decimal

    result = string_series.map(lambda x: Decimal(str(x)))
    assert result.dtype == np.object_
    assert isinstance(result[0], Decimal)


def test_map_na_exclusion():
    s = Series([1.5, np.nan, 3, np.nan, 5])

    result = s.map(lambda x: x * 2, na_action="ignore")
    exp = s * 2
    tm.assert_series_equal(result, exp)


def test_map_dict_with_tuple_keys():
    """
    Due to new MultiIndex-ing behaviour in v0.14.0,
    dicts with tuple keys passed to map were being
    converted to a multi-index, preventing tuple values
    from being mapped properly.
    """
    # GH 18496
    df = DataFrame({"a": [(1,), (2,), (3, 4), (5, 6)]})
    label_mappings = {(1,): "A", (2,): "B", (3, 4): "A", (5, 6): "B"}

    df["labels"] = df["a"].map(label_mappings)
    df["expected_labels"] = Series(["A", "B", "A", "B"], index=df.index)
    # All labels should be filled now
    tm.assert_series_equal(df["labels"], df["expected_labels"], check_names=False)


def test_map_counter():
    s = Series(["a", "b", "c"], index=[1, 2, 3])
    counter = Counter()
    counter["b"] = 5
    counter["c"] += 1
    result = s.map(counter)
    expected = Series([0, 5, 1], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_map_defaultdict():
    s = Series([1, 2, 3], index=["a", "b", "c"])
    default_dict = defaultdict(lambda: "blank")
    default_dict[1] = "stuff"
    result = s.map(default_dict)
    expected = Series(["stuff", "blank", "blank"], index=["a", "b", "c"])
    tm.assert_series_equal(result, expected)


def test_map_dict_na_key():
    # https://github.com/pandas-dev/pandas/issues/17648
    # Checks that np.nan key is appropriately mapped
    s = Series([1, 2, np.nan])
    expected = Series(["a", "b", "c"])
    result = s.map({1: "a", 2: "b", np.nan: "c"})
    tm.assert_series_equal(result, expected)


def test_map_dict_subclass_with_missing():
    """
    Test Series.map with a dictionary subclass that defines __missing__,
    i.e. sets a default value (GH #15999).
    """

    class DictWithMissing(dict):
        def __missing__(self, key):
            return "missing"

    s = Series([1, 2, 3])
    dictionary = DictWithMissing({3: "three"})
    result = s.map(dictionary)
    expected = Series(["missing", "missing", "three"])
    tm.assert_series_equal(result, expected)


def test_map_dict_subclass_without_missing():
    class DictWithoutMissing(dict):
        pass

    s = Series([1, 2, 3])
    dictionary = DictWithoutMissing({3: "three"})
    result = s.map(dictionary)
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_abc_mapping(non_dict_mapping_subclass):
    # https://github.com/pandas-dev/pandas/issues/29733
    # Check collections.abc.Mapping support as mapper for Series.map
    s = Series([1, 2, 3])
    not_a_dictionary = non_dict_mapping_subclass({3: "three"})
    result = s.map(not_a_dictionary)
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_abc_mapping_with_missing(non_dict_mapping_subclass):
    # https://github.com/pandas-dev/pandas/issues/29733
    # Check collections.abc.Mapping support as mapper for Series.map
    class NonDictMappingWithMissing(non_dict_mapping_subclass):
        def __missing__(key):
            return "missing"

    s = Series([1, 2, 3])
    not_a_dictionary = NonDictMappingWithMissing({3: "three"})
    result = s.map(not_a_dictionary)
    # __missing__ is a dict concept, not a Mapping concept,
    # so it should not change the result!
    expected = Series([np.nan, np.nan, "three"])
    tm.assert_series_equal(result, expected)


def test_map_box():
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    s = Series(vals)
    assert s.dtype == "datetime64[ns]"
    # boxed value must be Timestamp instance
    res = s.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    tm.assert_series_equal(res, exp)

    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    s = Series(vals)
    assert s.dtype == "datetime64[ns, US/Eastern]"
    res = s.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    tm.assert_series_equal(res, exp)

    # timedelta
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    s = Series(vals)
    assert s.dtype == "timedelta64[ns]"
    res = s.apply(lambda x: f"{type(x).__name__}_{x.days}")
    exp = Series(["Timedelta_1", "Timedelta_2"])
    tm.assert_series_equal(res, exp)

    # period
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    s = Series(vals)
    assert s.dtype == "Period[M]"
    res = s.apply(lambda x: f"{type(x).__name__}_{x.freqstr}")
    exp = Series(["Period_M", "Period_M"])
    tm.assert_series_equal(res, exp)


def test_map_categorical():
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    s = Series(values, name="XX", index=list("abcdefg"))

    result = s.map(lambda x: x.lower())
    exp_values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(exp_values, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp_values)

    result = s.map(lambda x: "A")
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    tm.assert_series_equal(result, exp)
    assert result.dtype == object


def test_map_datetimetz():
    values = pd.date_range("2011-01-01", "2011-01-02", freq="H").tz_localize(
        "Asia/Tokyo"
    )
    s = Series(values, name="XX")

    # keep tz
    result = s.map(lambda x: x + pd.offsets.Day())
    exp_values = pd.date_range("2011-01-02", "2011-01-03", freq="H").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    tm.assert_series_equal(result, exp)

    # change dtype
    # GH 14506 : Returned dtype changed from int32 to int64
    result = s.map(lambda x: x.hour)
    exp = Series(list(range(24)) + [0], name="XX", dtype=np.int64)
    tm.assert_series_equal(result, exp)

    # not vectorized
    def f(x):
        if not isinstance(x, pd.Timestamp):
            raise ValueError
        return str(x.tz)

    result = s.map(f)
    exp = Series(["Asia/Tokyo"] * 25, name="XX")
    tm.assert_series_equal(result, exp)


@pytest.mark.parametrize(
    "vals,mapping,exp",
    [
        (list("abc"), {np.nan: "not NaN"}, [np.nan] * 3 + ["not NaN"]),
        (list("abc"), {"a": "a letter"}, ["a letter"] + [np.nan] * 3),
        (list(range(3)), {0: 42}, [42] + [np.nan] * 3),
    ],
)
def test_map_missing_mixed(vals, mapping, exp):
    # GH20495
    s = Series(vals + [np.nan])
    result = s.map(mapping)

    tm.assert_series_equal(result, Series(exp))


@pytest.mark.parametrize(
    "dti,exp",
    [
        (
            Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])),
            DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype="int64"),
        ),
        (
            tm.makeTimeSeries(nper=30),
            DataFrame(np.repeat([[1, 2]], 30, axis=0), dtype="int64"),
        ),
    ],
)
@pytest.mark.parametrize("aware", [True, False])
def test_apply_series_on_date_time_index_aware_series(dti, exp, aware):
    # GH 25959
    # Calling apply on a localized time series should not cause an error
    if aware:
        index = dti.tz_localize("UTC").index
    else:
        index = dti.index
    result = Series(index).apply(lambda x: Series([1, 2]))
    tm.assert_frame_equal(result, exp)


def test_apply_scaler_on_date_time_index_aware_series():
    # GH 25959
    # Calling apply on a localized time series should not cause an error
    series = tm.makeTimeSeries(nper=30).tz_localize("UTC")
    result = Series(series.index).apply(lambda x: 1)
    tm.assert_series_equal(result, Series(np.ones(30), dtype="int64"))


def test_map_float_to_string_precision():
    # GH 13228
    ser = Series(1 / 3)
    result = ser.map(lambda val: str(val)).to_dict()
    expected = {0: "0.3333333333333333"}
    assert result == expected


def test_apply_to_timedelta():
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).apply(pd.to_timedelta)
    # FIXME: dont leave commented-out
    # Can't compare until apply on a Series gives the correct dtype
    # assert_series_equal(a, b)

    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]

    a = pd.to_timedelta(list_of_strings)  # noqa
    with tm.assert_produces_warning(FutureWarning, match="Inferring timedelta64"):
        ser = Series(list_of_strings)
    b = ser.apply(pd.to_timedelta)  # noqa
    # Can't compare until apply on a Series gives the correct dtype
    # assert_series_equal(a, b)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sum], ["sum"]),
        ([np.sum, np.mean], ["sum", "mean"]),
        (np.array([np.sum]), ["sum"]),
        (np.array([np.sum, np.mean]), ["sum", "mean"]),
    ],
)
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_listlike_reducer(string_series, ops, names, how):
    # GH 39140
    expected = Series({name: op(string_series) for name, op in zip(names, ops)})
    expected.name = "series"
    result = getattr(string_series, how)(ops)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sum},
        {"A": np.sum, "B": np.mean},
        Series({"A": np.sum}),
        Series({"A": np.sum, "B": np.mean}),
    ],
)
@pytest.mark.parametrize("how", ["agg", "apply"])
def test_apply_dictlike_reducer(string_series, ops, how):
    # GH 39140
    expected = Series({name: op(string_series) for name, op in ops.items()})
    expected.name = string_series.name
    result = getattr(string_series, how)(ops)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
def test_apply_listlike_transformer(string_series, ops, names):
    # GH 39140
    with np.errstate(all="ignore"):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.apply(ops)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sqrt},
        {"A": np.sqrt, "B": np.exp},
        Series({"A": np.sqrt}),
        Series({"A": np.sqrt, "B": np.exp}),
    ],
)
def test_apply_dictlike_transformer(string_series, ops):
    # GH 39140
    with np.errstate(all="ignore"):
        expected = concat({name: op(string_series) for name, op in ops.items()})
        expected.name = string_series.name
        result = string_series.apply(ops)
        tm.assert_series_equal(result, expected)
