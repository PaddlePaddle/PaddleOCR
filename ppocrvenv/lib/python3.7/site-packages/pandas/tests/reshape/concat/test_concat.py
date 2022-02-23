from collections import (
    abc,
    deque,
)
from decimal import Decimal
from warnings import catch_warnings

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.core.construction import create_series_with_explicit_dtype
from pandas.tests.extension.decimal import to_decimal


class TestConcatenate:
    def test_append_concat(self):
        # GH#1815
        d1 = date_range("12/31/1990", "12/31/1999", freq="A-DEC")
        d2 = date_range("12/31/2000", "12/31/2009", freq="A-DEC")

        s1 = Series(np.random.randn(10), d1)
        s2 = Series(np.random.randn(10), d2)

        s1 = s1.to_period()
        s2 = s2.to_period()

        # drops index
        result = concat([s1, s2])
        assert isinstance(result.index, PeriodIndex)
        assert result.index[0] == s1.index[0]

    # TODO(ArrayManager) using block internals to verify, needs rewrite
    @td.skip_array_manager_invalid_test
    def test_concat_copy(self):
        df = DataFrame(np.random.randn(4, 3))
        df2 = DataFrame(np.random.randint(0, 10, size=4).reshape(4, 1))
        df3 = DataFrame({5: "foo"}, index=range(4))

        # These are actual copies.
        result = concat([df, df2, df3], axis=1, copy=True)

        for b in result._mgr.blocks:
            assert b.values.base is None

        # These are the same.
        result = concat([df, df2, df3], axis=1, copy=False)

        for b in result._mgr.blocks:
            if b.dtype.kind == "f":
                assert b.values.base is df._mgr.blocks[0].values.base
            elif b.dtype.kind in ["i", "u"]:
                assert b.values.base is df2._mgr.blocks[0].values.base
            elif b.is_object:
                assert b.values.base is not None

        # Float block was consolidated.
        df4 = DataFrame(np.random.randn(4, 1))
        result = concat([df, df2, df3, df4], axis=1, copy=False)
        for b in result._mgr.blocks:
            if b.dtype.kind == "f":
                assert b.values.base is None
            elif b.dtype.kind in ["i", "u"]:
                assert b.values.base is df2._mgr.blocks[0].values.base
            elif b.is_object:
                assert b.values.base is not None

    def test_concat_with_group_keys(self):
        df = DataFrame(np.random.randn(4, 3))
        df2 = DataFrame(np.random.randn(4, 4))

        # axis=0
        df = DataFrame(np.random.randn(3, 4))
        df2 = DataFrame(np.random.randn(4, 4))

        result = concat([df, df2], keys=[0, 1])
        exp_index = MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2, 3]]
        )
        expected = DataFrame(np.r_[df.values, df2.values], index=exp_index)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df], keys=[0, 1])
        exp_index2 = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        expected = DataFrame(np.r_[df.values, df.values], index=exp_index2)
        tm.assert_frame_equal(result, expected)

        # axis=1
        df = DataFrame(np.random.randn(4, 3))
        df2 = DataFrame(np.random.randn(4, 4))

        result = concat([df, df2], keys=[0, 1], axis=1)
        expected = DataFrame(np.c_[df.values, df2.values], columns=exp_index)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df], keys=[0, 1], axis=1)
        expected = DataFrame(np.c_[df.values, df.values], columns=exp_index2)
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_specific_levels(self):
        df = DataFrame(np.random.randn(10, 4))
        pieces = [df.iloc[:, [0, 1]], df.iloc[:, [2]], df.iloc[:, [3]]]
        level = ["three", "two", "one", "zero"]
        result = concat(
            pieces,
            axis=1,
            keys=["one", "two", "three"],
            levels=[level],
            names=["group_key"],
        )

        tm.assert_index_equal(result.columns.levels[0], Index(level, name="group_key"))
        tm.assert_index_equal(result.columns.levels[1], Index([0, 1, 2, 3]))

        assert result.columns.names == ["group_key", None]

    @pytest.mark.parametrize("mapping", ["mapping", "dict"])
    def test_concat_mapping(self, mapping, non_dict_mapping_subclass):
        constructor = dict if mapping == "dict" else non_dict_mapping_subclass
        frames = constructor(
            {
                "foo": DataFrame(np.random.randn(4, 3)),
                "bar": DataFrame(np.random.randn(4, 3)),
                "baz": DataFrame(np.random.randn(4, 3)),
                "qux": DataFrame(np.random.randn(4, 3)),
            }
        )

        sorted_keys = list(frames.keys())

        result = concat(frames)
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys)
        tm.assert_frame_equal(result, expected)

        result = concat(frames, axis=1)
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys, axis=1)
        tm.assert_frame_equal(result, expected)

        keys = ["baz", "foo", "bar"]
        result = concat(frames, keys=keys)
        expected = concat([frames[k] for k in keys], keys=keys)
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_and_levels(self):
        df = DataFrame(np.random.randn(1, 3))
        df2 = DataFrame(np.random.randn(1, 4))

        levels = [["foo", "baz"], ["one", "two"]]
        names = ["first", "second"]
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
            names=names,
        )
        expected = concat([df, df2, df, df2])
        exp_index = MultiIndex(
            levels=levels + [[0]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]],
            names=names + [None],
        )
        expected.index = exp_index

        tm.assert_frame_equal(result, expected)

        # no names
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
        )
        assert result.index.names == (None,) * 3

        # no levels
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            names=["first", "second"],
        )
        assert result.index.names == ("first", "second", None)
        tm.assert_index_equal(
            result.index.levels[0], Index(["baz", "foo"], name="first")
        )

    def test_concat_keys_levels_no_overlap(self):
        # GH #1406
        df = DataFrame(np.random.randn(1, 3), index=["a"])
        df2 = DataFrame(np.random.randn(1, 4), index=["b"])

        msg = "Values not found in passed level"
        with pytest.raises(ValueError, match=msg):
            concat([df, df], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

        msg = "Key one not in level"
        with pytest.raises(ValueError, match=msg):
            concat([df, df2], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

    def test_crossed_dtypes_weird_corner(self):
        columns = ["A", "B", "C", "D"]
        df1 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="f8"),
                "B": np.array([1, 2, 3, 4], dtype="i8"),
                "C": np.array([1, 2, 3, 4], dtype="f8"),
                "D": np.array([1, 2, 3, 4], dtype="i8"),
            },
            columns=columns,
        )

        df2 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="i8"),
                "B": np.array([1, 2, 3, 4], dtype="f8"),
                "C": np.array([1, 2, 3, 4], dtype="i8"),
                "D": np.array([1, 2, 3, 4], dtype="f8"),
            },
            columns=columns,
        )

        appended = df1.append(df2, ignore_index=True)
        expected = DataFrame(
            np.concatenate([df1.values, df2.values], axis=0), columns=columns
        )
        tm.assert_frame_equal(appended, expected)

        df = DataFrame(np.random.randn(1, 3), index=["a"])
        df2 = DataFrame(np.random.randn(1, 4), index=["b"])
        result = concat([df, df2], keys=["one", "two"], names=["first", "second"])
        assert result.index.names == ("first", "second")

    def test_with_mixed_tuples(self, sort):
        # 10697
        # columns have mixed tuples, so handle properly
        df1 = DataFrame({"A": "foo", ("B", 1): "bar"}, index=range(2))
        df2 = DataFrame({"B": "foo", ("B", 1): "bar"}, index=range(2))

        # it works
        concat([df1, df2], sort=sort)

    def test_concat_mixed_objs(self):

        # concat mixed series/frames
        # G2385

        # axis 1
        index = date_range("01-Jan-2013", periods=10, freq="H")
        arr = np.arange(10, dtype="int64")
        s1 = Series(arr, index=index)
        s2 = Series(arr, index=index)
        df = DataFrame(arr.reshape(-1, 1), index=index)

        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 0]
        )
        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 1]
        )
        result = concat([s1, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        result = concat([s1, s2, s1], axis=1)
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            np.repeat(arr, 5).reshape(-1, 5), index=index, columns=[0, 0, 1, 2, 3]
        )
        result = concat([s1, df, s2, s2, s1], axis=1)
        tm.assert_frame_equal(result, expected)

        # with names
        s1.name = "foo"
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, 0]
        )
        result = concat([s1, df, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        s2.name = "bar"
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, "bar"]
        )
        result = concat([s1, df, s2], axis=1)
        tm.assert_frame_equal(result, expected)

        # ignore index
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        result = concat([s1, df, s2], axis=1, ignore_index=True)
        tm.assert_frame_equal(result, expected)

        # axis 0
        expected = DataFrame(
            np.tile(arr, 3).reshape(-1, 1), index=index.tolist() * 3, columns=[0]
        )
        result = concat([s1, df, s2])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(np.tile(arr, 3).reshape(-1, 1), columns=[0])
        result = concat([s1, df, s2], ignore_index=True)
        tm.assert_frame_equal(result, expected)

    def test_dtype_coerceion(self):

        # 12411
        df = DataFrame({"date": [pd.Timestamp("20130101").tz_localize("UTC"), pd.NaT]})

        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 12045
        import datetime

        df = DataFrame(
            {"date": [datetime.datetime(2012, 1, 1), datetime.datetime(1012, 1, 2)]}
        )
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 11594
        df = DataFrame({"text": ["some words"] + [None] * 9})
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        tm.assert_series_equal(result.dtypes, df.dtypes)

    def test_concat_single_with_key(self):
        df = DataFrame(np.random.randn(10, 4))

        result = concat([df], keys=["foo"])
        expected = concat([df, df], keys=["foo", "bar"])
        tm.assert_frame_equal(result, expected[:10])

    def test_concat_no_items_raises(self):
        with pytest.raises(ValueError, match="No objects to concatenate"):
            concat([])

    def test_concat_exclude_none(self):
        df = DataFrame(np.random.randn(10, 4))

        pieces = [df[:5], None, None, df[5:]]
        result = concat(pieces)
        tm.assert_frame_equal(result, df)
        with pytest.raises(ValueError, match="All objects passed were None"):
            concat([None, None])

    def test_concat_keys_with_none(self):
        # #1649
        df0 = DataFrame([[10, 20, 30], [10, 20, 30], [10, 20, 30]])

        result = concat({"a": None, "b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        expected = concat({"b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        tm.assert_frame_equal(result, expected)

        result = concat(
            [None, df0, df0[:2], df0[:1], df0], keys=["a", "b", "c", "d", "e"]
        )
        expected = concat([df0, df0[:2], df0[:1], df0], keys=["b", "c", "d", "e"])
        tm.assert_frame_equal(result, expected)

    def test_concat_bug_1719(self):
        ts1 = tm.makeTimeSeries()
        ts2 = tm.makeTimeSeries()[::2]

        # to join with union
        # these two are of different length!
        left = concat([ts1, ts2], join="outer", axis=1)
        right = concat([ts2, ts1], join="outer", axis=1)

        assert len(left) == len(right)

    def test_concat_bug_2972(self):
        ts0 = Series(np.zeros(5))
        ts1 = Series(np.ones(5))
        ts0.name = ts1.name = "same name"
        result = concat([ts0, ts1], axis=1)

        expected = DataFrame({0: ts0, 1: ts1})
        expected.columns = ["same name", "same name"]
        tm.assert_frame_equal(result, expected)

    def test_concat_bug_3602(self):

        # GH 3602, duplicate columns
        df1 = DataFrame(
            {
                "firmNo": [0, 0, 0, 0],
                "prc": [6, 6, 6, 6],
                "stringvar": ["rrr", "rrr", "rrr", "rrr"],
            }
        )
        df2 = DataFrame(
            {"C": [9, 10, 11, 12], "misc": [1, 2, 3, 4], "prc": [6, 6, 6, 6]}
        )
        expected = DataFrame(
            [
                [0, 6, "rrr", 9, 1, 6],
                [0, 6, "rrr", 10, 2, 6],
                [0, 6, "rrr", 11, 3, 6],
                [0, 6, "rrr", 12, 4, 6],
            ]
        )
        expected.columns = ["firmNo", "prc", "stringvar", "C", "misc", "prc"]

        result = concat([df1, df2], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_concat_iterables(self):
        # GH8645 check concat works with tuples, list, generators, and weird
        # stuff like deque and custom iterables
        df1 = DataFrame([1, 2, 3])
        df2 = DataFrame([4, 5, 6])
        expected = DataFrame([1, 2, 3, 4, 5, 6])
        tm.assert_frame_equal(concat((df1, df2), ignore_index=True), expected)
        tm.assert_frame_equal(concat([df1, df2], ignore_index=True), expected)
        tm.assert_frame_equal(
            concat((df for df in (df1, df2)), ignore_index=True), expected
        )
        tm.assert_frame_equal(concat(deque((df1, df2)), ignore_index=True), expected)

        class CustomIterator1:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index):
                try:
                    return {0: df1, 1: df2}[index]
                except KeyError as err:
                    raise IndexError from err

        tm.assert_frame_equal(concat(CustomIterator1(), ignore_index=True), expected)

        class CustomIterator2(abc.Iterable):
            def __iter__(self):
                yield df1
                yield df2

        tm.assert_frame_equal(concat(CustomIterator2(), ignore_index=True), expected)

    def test_concat_order(self):
        # GH 17344
        dfs = [DataFrame(index=range(3), columns=["a", 1, None])]
        dfs += [DataFrame(index=range(3), columns=[None, 1, "a"]) for i in range(100)]

        result = concat(dfs, sort=True).columns
        expected = dfs[0].columns
        tm.assert_index_equal(result, expected)

    def test_concat_different_extension_dtypes_upcasts(self):
        a = Series(pd.array([1, 2], dtype="Int64"))
        b = Series(to_decimal([1, 2]))

        result = concat([a, b], ignore_index=True)
        expected = Series([1, 2, Decimal(1), Decimal(2)], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_concat_ordered_dict(self):
        # GH 21510
        expected = concat(
            [Series(range(3)), Series(range(4))], keys=["First", "Another"]
        )
        result = concat({"First": Series(range(3)), "Another": Series(range(4))})
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pdt", [Series, DataFrame])
@pytest.mark.parametrize("dt", np.sctypes["float"])
def test_concat_no_unnecessary_upcast(dt, pdt):
    # GH 13247
    dims = pdt(dtype=object).ndim

    dfs = [
        pdt(np.array([1], dtype=dt, ndmin=dims)),
        pdt(np.array([np.nan], dtype=dt, ndmin=dims)),
        pdt(np.array([5], dtype=dt, ndmin=dims)),
    ]
    x = concat(dfs)
    assert x.values.dtype == dt


@pytest.mark.parametrize("pdt", [create_series_with_explicit_dtype, DataFrame])
@pytest.mark.parametrize("dt", np.sctypes["int"])
def test_concat_will_upcast(dt, pdt):
    with catch_warnings(record=True):
        dims = pdt().ndim
        dfs = [
            pdt(np.array([1], dtype=dt, ndmin=dims)),
            pdt(np.array([np.nan], ndmin=dims)),
            pdt(np.array([5], dtype=dt, ndmin=dims)),
        ]
        x = concat(dfs)
        assert x.values.dtype == "float64"


def test_concat_empty_and_non_empty_frame_regression():
    # GH 18178 regression test
    df1 = DataFrame({"foo": [1]})
    df2 = DataFrame({"foo": []})
    expected = DataFrame({"foo": [1.0]})
    result = concat([df1, df2])
    tm.assert_frame_equal(result, expected)


def test_concat_sparse():
    # GH 23557
    a = Series(SparseArray([0, 1, 2]))
    expected = DataFrame(data=[[0, 0], [1, 1], [2, 2]]).astype(
        pd.SparseDtype(np.int64, 0)
    )
    result = concat([a, a], axis=1)
    tm.assert_frame_equal(result, expected)


def test_concat_dense_sparse():
    # GH 30668
    a = Series(pd.arrays.SparseArray([1, None]), dtype=float)
    b = Series([1], dtype=float)
    expected = Series(data=[1, None, 1], index=[0, 1, 0]).astype(
        pd.SparseDtype(np.float64, None)
    )
    result = concat([a, b], axis=0)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", [["e", "f", "f"], ["f", "e", "f"]])
def test_duplicate_keys(keys):
    # GH 33654
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s1 = Series([7, 8, 9], name="c")
    s2 = Series([10, 11, 12], name="d")
    result = concat([df, s1, s2], axis=1, keys=keys)
    expected_values = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1], "c"), (keys[2], "d")]
    )
    expected = DataFrame(expected_values, columns=expected_columns)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "obj",
    [
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
)
def test_concat_preserves_subclass(obj):
    # GH28330 -- preserve subclass

    result = concat([obj, obj])
    assert isinstance(result, type(obj))


def test_concat_frame_axis0_extension_dtypes():
    # preserve extension dtype (through common_dtype mechanism)
    df1 = DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})
    df2 = DataFrame({"a": np.array([4, 5, 6])})

    result = concat([df1, df2], ignore_index=True)
    expected = DataFrame({"a": [1, 2, 3, 4, 5, 6]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)

    result = concat([df2, df1], ignore_index=True)
    expected = DataFrame({"a": [4, 5, 6, 1, 2, 3]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


def test_concat_preserves_extension_int64_dtype():
    # GH 24768
    df_a = DataFrame({"a": [-1]}, dtype="Int64")
    df_b = DataFrame({"b": [1]}, dtype="Int64")
    result = concat([df_a, df_b], ignore_index=True)
    expected = DataFrame({"a": [-1, None], "b": [None, 1]}, dtype="Int64")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("keys", "integrity"),
    [
        (["red"] * 3, True),
        (["red"] * 3, False),
        (["red", "blue", "red"], False),
        (["red", "blue", "red"], True),
    ],
)
def test_concat_repeated_keys(keys, integrity):
    # GH: 20816
    series_list = [Series({"a": 1}), Series({"b": 2}), Series({"c": 3})]
    result = concat(series_list, keys=keys, verify_integrity=integrity)
    tuples = list(zip(keys, ["a", "b", "c"]))
    expected = Series([1, 2, 3], index=MultiIndex.from_tuples(tuples))
    tm.assert_series_equal(result, expected)


def test_concat_null_object_with_dti():
    # GH#40841
    dti = pd.DatetimeIndex(
        ["2021-04-08 21:21:14+00:00"], dtype="datetime64[ns, UTC]", name="Time (UTC)"
    )
    right = DataFrame(data={"C": [0.5274]}, index=dti)

    idx = Index([None], dtype="object", name="Maybe Time (UTC)")
    left = DataFrame(data={"A": [None], "B": [np.nan]}, index=idx)

    result = concat([left, right], axis="columns")

    exp_index = Index([None, dti[0]], dtype=object)
    expected = DataFrame(
        {"A": [None, None], "B": [np.nan, np.nan], "C": [np.nan, 0.5274]},
        index=exp_index,
    )
    tm.assert_frame_equal(result, expected)


def test_concat_multiindex_with_empty_rangeindex():
    # GH#41234
    mi = MultiIndex.from_tuples([("B", 1), ("C", 1)])
    df1 = DataFrame([[1, 2]], columns=mi)
    df2 = DataFrame(index=[1], columns=pd.RangeIndex(0))

    result = concat([df1, df2])
    expected = DataFrame([[1, 2], [np.nan, np.nan]], columns=mi)
    tm.assert_frame_equal(result, expected)


def test_concat_posargs_deprecation():
    # https://github.com/pandas-dev/pandas/issues/41485
    df = DataFrame([[1, 2, 3]], index=["a"])
    df2 = DataFrame([[4, 5, 6]], index=["b"])

    msg = (
        "In a future version of pandas all arguments of concat "
        "except for the argument 'objs' will be keyword-only"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = concat([df, df2], 0)
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], index=["a", "b"])
    tm.assert_frame_equal(result, expected)
