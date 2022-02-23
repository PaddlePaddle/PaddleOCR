from collections import OrderedDict
from datetime import (
    datetime,
    timedelta,
)

from dateutil.tz import tzoffset
import numpy as np
import numpy.ma as ma
import pytest

from pandas._libs import (
    iNaT,
    lib,
)
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_categorical_dtype,
    is_datetime64tz_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    MultiIndex,
    NaT,
    Period,
    RangeIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    IntervalArray,
    period_array,
)
from pandas.core.internals.blocks import NumericBlock


class TestSeriesConstructors:
    @pytest.mark.parametrize(
        "constructor,check_index_type",
        [
            # NOTE: some overlap with test_constructor_empty but that test does not
            # test for None or an empty generator.
            # test_constructor_pass_none tests None but only with the index also
            # passed.
            (lambda: Series(), True),
            (lambda: Series(None), True),
            (lambda: Series({}), True),
            (lambda: Series(()), False),  # creates a RangeIndex
            (lambda: Series([]), False),  # creates a RangeIndex
            (lambda: Series(_ for _ in []), False),  # creates a RangeIndex
            (lambda: Series(data=None), True),
            (lambda: Series(data={}), True),
            (lambda: Series(data=()), False),  # creates a RangeIndex
            (lambda: Series(data=[]), False),  # creates a RangeIndex
            (lambda: Series(data=(_ for _ in [])), False),  # creates a RangeIndex
        ],
    )
    def test_empty_constructor(self, constructor, check_index_type):
        # TODO: share with frame test of the same name
        with tm.assert_produces_warning(DeprecationWarning):
            expected = Series()
            result = constructor()

        assert len(result.index) == 0
        tm.assert_series_equal(result, expected, check_index_type=check_index_type)

    def test_invalid_dtype(self):
        # GH15520
        msg = "not understood"
        invalid_list = [Timestamp, "Timestamp", list]
        for dtype in invalid_list:
            with pytest.raises(TypeError, match=msg):
                Series([], name="time", dtype=dtype)

    def test_invalid_compound_dtype(self):
        # GH#13296
        c_dtype = np.dtype([("a", "i8"), ("b", "f4")])
        cdt_arr = np.array([(1, 0.4), (256, -13)], dtype=c_dtype)

        with pytest.raises(ValueError, match="Use DataFrame instead"):
            Series(cdt_arr, index=["A", "B"])

    def test_scalar_conversion(self):

        # Pass in scalar is disabled
        scalar = Series(0.5)
        assert not isinstance(scalar, float)

        # Coercion
        assert float(Series([1.0])) == 1.0
        assert int(Series([1.0])) == 1

    def test_scalar_extension_dtype(self, ea_scalar_and_dtype):
        # GH 28401

        ea_scalar, ea_dtype = ea_scalar_and_dtype

        ser = Series(ea_scalar, index=range(3))
        expected = Series([ea_scalar] * 3, dtype=ea_dtype)

        assert ser.dtype == ea_dtype
        tm.assert_series_equal(ser, expected)

    def test_constructor(self, datetime_series):
        with tm.assert_produces_warning(DeprecationWarning):
            empty_series = Series()
        assert datetime_series.index._is_all_dates

        # Pass in Series
        derived = Series(datetime_series)
        assert derived.index._is_all_dates

        assert tm.equalContents(derived.index, datetime_series.index)
        # Ensure new index is not created
        assert id(datetime_series.index) == id(derived.index)

        # Mixed type Series
        mixed = Series(["hello", np.NaN], index=[0, 1])
        assert mixed.dtype == np.object_
        assert mixed[1] is np.NaN

        assert not empty_series.index._is_all_dates
        with tm.assert_produces_warning(DeprecationWarning):
            assert not Series().index._is_all_dates

        # exception raised is of type ValueError GH35744
        with pytest.raises(ValueError, match="Data must be 1-dimensional"):
            Series(np.random.randn(3, 3), index=np.arange(3))

        mixed.name = "Series"
        rs = Series(mixed).name
        xp = "Series"
        assert rs == xp

        # raise on MultiIndex GH4187
        m = MultiIndex.from_arrays([[1, 2], [3, 4]])
        msg = "initializing a Series from a MultiIndex is not supported"
        with pytest.raises(NotImplementedError, match=msg):
            Series(m)

    @pytest.mark.parametrize("input_class", [list, dict, OrderedDict])
    def test_constructor_empty(self, input_class):
        with tm.assert_produces_warning(DeprecationWarning):
            empty = Series()
            empty2 = Series(input_class())

        # these are Index() and RangeIndex() which don't compare type equal
        # but are just .equals
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # With explicit dtype:
        empty = Series(dtype="float64")
        empty2 = Series(input_class(), dtype="float64")
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # GH 18515 : with dtype=category:
        empty = Series(dtype="category")
        empty2 = Series(input_class(), dtype="category")
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        if input_class is not list:
            # With index:
            with tm.assert_produces_warning(DeprecationWarning):
                empty = Series(index=range(10))
                empty2 = Series(input_class(), index=range(10))
            tm.assert_series_equal(empty, empty2)

            # With index and dtype float64:
            empty = Series(np.nan, index=range(10))
            empty2 = Series(input_class(), index=range(10), dtype="float64")
            tm.assert_series_equal(empty, empty2)

            # GH 19853 : with empty string, index and dtype str
            empty = Series("", dtype=str, index=range(3))
            empty2 = Series("", index=range(3))
            tm.assert_series_equal(empty, empty2)

    @pytest.mark.parametrize("input_arg", [np.nan, float("nan")])
    def test_constructor_nan(self, input_arg):
        empty = Series(dtype="float64", index=range(10))
        empty2 = Series(input_arg, index=range(10))

        tm.assert_series_equal(empty, empty2, check_index_type=False)

    @pytest.mark.parametrize(
        "dtype",
        ["f8", "i8", "M8[ns]", "m8[ns]", "category", "object", "datetime64[ns, UTC]"],
    )
    @pytest.mark.parametrize("index", [None, Index([])])
    def test_constructor_dtype_only(self, dtype, index):
        # GH-20865
        result = Series(dtype=dtype, index=index)
        assert result.dtype == dtype
        assert len(result) == 0

    def test_constructor_no_data_index_order(self):
        with tm.assert_produces_warning(DeprecationWarning):
            result = Series(index=["b", "a", "c"])
        assert result.index.tolist() == ["b", "a", "c"]

    def test_constructor_no_data_string_type(self):
        # GH 22477
        result = Series(index=[1], dtype=str)
        assert np.isnan(result.iloc[0])

    @pytest.mark.parametrize("item", ["entry", "ѐ", 13])
    def test_constructor_string_element_string_type(self, item):
        # GH 22477
        result = Series(item, index=[1], dtype=str)
        assert result.iloc[0] == str(item)

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # https://github.com/pandas-dev/pandas/issues/21083
        ser = Series(["x", None], dtype=string_dtype)
        result = ser.isna()
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)
        assert ser.iloc[1] is None

        ser = Series(["x", np.nan], dtype=string_dtype)
        assert np.isnan(ser.iloc[1])

    def test_constructor_series(self):
        index1 = ["d", "b", "a", "c"]
        index2 = sorted(index1)
        s1 = Series([4, 7, -5, 3], index=index1)
        s2 = Series(s1, index=index2)

        tm.assert_series_equal(s2, s1.sort_index())

    def test_constructor_iterable(self):
        # GH 21987
        class Iter:
            def __iter__(self):
                yield from range(10)

        expected = Series(list(range(10)), dtype="int64")
        result = Series(Iter(), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_sequence(self):
        # GH 21987
        expected = Series(list(range(10)), dtype="int64")
        result = Series(range(10), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_single_str(self):
        # GH 21987
        expected = Series(["abc"])
        result = Series("abc")
        tm.assert_series_equal(result, expected)

    def test_constructor_list_like(self):

        # make sure that we are coercing different
        # list-likes to standard dtypes and not
        # platform specific
        expected = Series([1, 2, 3], dtype="int64")
        for obj in [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3], dtype="int64")]:
            result = Series(obj, index=[0, 1, 2])
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float64"])
    def test_constructor_index_dtype(self, dtype):
        # GH 17088

        s = Series(Index([0, 2, 4]), dtype=dtype)
        assert s.dtype == dtype

    @pytest.mark.parametrize(
        "input_vals",
        [
            ([1, 2]),
            (["1", "2"]),
            (list(date_range("1/1/2011", periods=2, freq="H"))),
            (list(date_range("1/1/2011", periods=2, freq="H", tz="US/Eastern"))),
            ([Interval(left=0, right=5)]),
        ],
    )
    def test_constructor_list_str(self, input_vals, string_dtype):
        # GH 16605
        # Ensure that data elements from a list are converted to strings
        # when dtype is str, 'str', or 'U'
        result = Series(input_vals, dtype=string_dtype)
        expected = Series(input_vals).astype(string_dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype):
        result = Series([1.0, 2.0, np.nan], dtype=string_dtype)
        expected = Series(["1.0", "2.0", np.nan], dtype=object)
        tm.assert_series_equal(result, expected)
        assert np.isnan(result[2])

    def test_constructor_generator(self):
        gen = (i for i in range(10))

        result = Series(gen)
        exp = Series(range(10))
        tm.assert_series_equal(result, exp)

        # same but with non-default index
        gen = (i for i in range(10))
        result = Series(gen, index=range(10, 20))
        exp.index = range(10, 20)
        tm.assert_series_equal(result, exp)

    def test_constructor_map(self):
        # GH8909
        m = map(lambda x: x, range(10))

        result = Series(m)
        exp = Series(range(10))
        tm.assert_series_equal(result, exp)

        # same but with non-default index
        m = map(lambda x: x, range(10))
        result = Series(m, index=range(10, 20))
        exp.index = range(10, 20)
        tm.assert_series_equal(result, exp)

    def test_constructor_categorical(self):
        cat = Categorical([0, 1, 2, 0, 1, 2], ["a", "b", "c"], fastpath=True)
        res = Series(cat)
        tm.assert_categorical_equal(res.values, cat)

        # can cast to a new dtype
        result = Series(Categorical([1, 2, 3]), dtype="int64")
        expected = Series([1, 2, 3], dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_construct_from_categorical_with_dtype(self):
        # GH12574
        cat = Series(Categorical([1, 2, 3]), dtype="category")
        assert is_categorical_dtype(cat)
        assert is_categorical_dtype(cat.dtype)

    def test_construct_intlist_values_category_dtype(self):
        ser = Series([1, 2, 3], dtype="category")
        assert is_categorical_dtype(ser)
        assert is_categorical_dtype(ser.dtype)

    def test_constructor_categorical_with_coercion(self):
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"])
        # test basic creation / coercion of categoricals
        s = Series(factor, name="A")
        assert s.dtype == "category"
        assert len(s) == len(factor)
        str(s.values)
        str(s)

        # in a frame
        df = DataFrame({"A": factor})
        result = df["A"]
        tm.assert_series_equal(result, s)
        result = df.iloc[:, 0]
        tm.assert_series_equal(result, s)
        assert len(df) == len(factor)
        str(df.values)
        str(df)

        df = DataFrame({"A": s})
        result = df["A"]
        tm.assert_series_equal(result, s)
        assert len(df) == len(factor)
        str(df.values)
        str(df)

        # multiples
        df = DataFrame({"A": s, "B": s, "C": 1})
        result1 = df["A"]
        result2 = df["B"]
        tm.assert_series_equal(result1, s)
        tm.assert_series_equal(result2, s, check_names=False)
        assert result2.name == "B"
        assert len(df) == len(factor)
        str(df.values)
        str(df)

    def test_constructor_categorical_with_coercion2(self):
        # GH8623
        x = DataFrame(
            [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
            columns=["person_id", "person_name"],
        )
        x["person_name"] = Categorical(x.person_name)  # doing this breaks transform

        expected = x.iloc[0].person_name
        result = x.person_name.iloc[0]
        assert result == expected

        result = x.person_name[0]
        assert result == expected

        result = x.person_name.loc[0]
        assert result == expected

    def test_constructor_series_to_categorical(self):
        # see GH#16524: test conversion of Series to Categorical
        series = Series(["a", "b", "c"])

        result = Series(series, dtype="category")
        expected = Series(["a", "b", "c"], dtype="category")

        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_dtype(self):
        result = Series(
            ["a", "b"], dtype=CategoricalDtype(["a", "b", "c"], ordered=True)
        )
        assert is_categorical_dtype(result.dtype) is True
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))
        assert result.cat.ordered

        result = Series(["a", "b"], dtype=CategoricalDtype(["b", "a"]))
        assert is_categorical_dtype(result.dtype)
        tm.assert_index_equal(result.cat.categories, Index(["b", "a"]))
        assert result.cat.ordered is False

        # GH 19565 - Check broadcasting of scalar with Categorical dtype
        result = Series(
            "a", index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        expected = Series(
            ["a", "a"], index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_string(self):
        # GH 26336: the string 'category' maintains existing CategoricalDtype
        cdt = CategoricalDtype(categories=list("dabc"), ordered=True)
        expected = Series(list("abcabc"), dtype=cdt)

        # Series(Categorical, dtype='category') keeps existing dtype
        cat = Categorical(list("abcabc"), dtype=cdt)
        result = Series(cat, dtype="category")
        tm.assert_series_equal(result, expected)

        # Series(Series[Categorical], dtype='category') keeps existing dtype
        result = Series(result, dtype="category")
        tm.assert_series_equal(result, expected)

    def test_categorical_sideeffects_free(self):
        # Passing a categorical to a Series and then changing values in either
        # the series or the categorical should not change the values in the
        # other one, IF you specify copy!
        cat = Categorical(["a", "b", "c", "a"])
        s = Series(cat, copy=True)
        assert s.cat is not cat
        s.cat.categories = [1, 2, 3]
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(s.__array__(), exp_s)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # setting
        s[0] = 2
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # however, copy is False by default
        # so this WILL change values
        cat = Categorical(["a", "b", "c", "a"])
        s = Series(cat)
        assert s.values is cat
        s.cat.categories = [1, 2, 3]
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s)
        tm.assert_numpy_array_equal(cat.__array__(), exp_s)

        s[0] = 2
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)
        tm.assert_numpy_array_equal(cat.__array__(), exp_s2)

    def test_unordered_compare_equal(self):
        left = Series(["a", "b", "c"], dtype=CategoricalDtype(["a", "b"]))
        right = Series(Categorical(["a", "b", np.nan], categories=["a", "b"]))
        tm.assert_series_equal(left, right)

    def test_constructor_maskedarray(self):
        data = ma.masked_all((3,), dtype=float)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan])
        tm.assert_series_equal(result, expected)

        data[0] = 0.0
        data[2] = 2.0
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([0.0, np.nan, 2.0], index=index)
        tm.assert_series_equal(result, expected)

        data[1] = 1.0
        result = Series(data, index=index)
        expected = Series([0.0, 1.0, 2.0], index=index)
        tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype=int)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan], dtype=float)
        tm.assert_series_equal(result, expected)

        data[0] = 0
        data[2] = 2
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([0, np.nan, 2], index=index, dtype=float)
        tm.assert_series_equal(result, expected)

        data[1] = 1
        result = Series(data, index=index)
        expected = Series([0, 1, 2], index=index, dtype=int)
        tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype=bool)
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan], dtype=object)
        tm.assert_series_equal(result, expected)

        data[0] = True
        data[2] = False
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series([True, np.nan, False], index=index, dtype=object)
        tm.assert_series_equal(result, expected)

        data[1] = True
        result = Series(data, index=index)
        expected = Series([True, True, False], index=index, dtype=bool)
        tm.assert_series_equal(result, expected)

        data = ma.masked_all((3,), dtype="M8[ns]")
        result = Series(data)
        expected = Series([iNaT, iNaT, iNaT], dtype="M8[ns]")
        tm.assert_series_equal(result, expected)

        data[0] = datetime(2001, 1, 1)
        data[2] = datetime(2001, 1, 3)
        index = ["a", "b", "c"]
        result = Series(data, index=index)
        expected = Series(
            [datetime(2001, 1, 1), iNaT, datetime(2001, 1, 3)],
            index=index,
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

        data[1] = datetime(2001, 1, 2)
        result = Series(data, index=index)
        expected = Series(
            [datetime(2001, 1, 1), datetime(2001, 1, 2), datetime(2001, 1, 3)],
            index=index,
            dtype="M8[ns]",
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_maskedarray_hardened(self):
        # Check numpy masked arrays with hard masks -- from GH24574
        data = ma.masked_all((3,), dtype=float).harden_mask()
        result = Series(data)
        expected = Series([np.nan, np.nan, np.nan])
        tm.assert_series_equal(result, expected)

    def test_series_ctor_plus_datetimeindex(self):
        rng = date_range("20090415", "20090519", freq="B")
        data = {k: 1 for k in rng}

        result = Series(data, index=rng)
        assert result.index is rng

    def test_constructor_default_index(self):
        s = Series([0, 1, 2])
        tm.assert_index_equal(s.index, Index(range(3)), exact=True)

    @pytest.mark.parametrize(
        "input",
        [
            [1, 2, 3],
            (1, 2, 3),
            list(range(3)),
            Categorical(["a", "b", "a"]),
            (i for i in range(3)),
            map(lambda x: x, range(3)),
        ],
    )
    def test_constructor_index_mismatch(self, input):
        # GH 19342
        # test that construction of a Series with an index of different length
        # raises an error
        msg = r"Length of values \(3\) does not match length of index \(4\)"
        with pytest.raises(ValueError, match=msg):
            Series(input, index=np.arange(4))

    def test_constructor_numpy_scalar(self):
        # GH 19342
        # construction with a numpy scalar
        # should not raise
        result = Series(np.array(100), index=np.arange(4), dtype="int64")
        expected = Series(100, index=np.arange(4), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_broadcast_list(self):
        # GH 19342
        # construction with single-element container and index
        # should raise
        msg = r"Length of values \(1\) does not match length of index \(3\)"
        with pytest.raises(ValueError, match=msg):
            Series(["foo"], index=["a", "b", "c"])

    def test_constructor_corner(self):
        df = tm.makeTimeDataFrame()
        objs = [df, df]
        s = Series(objs, index=[0, 1])
        assert isinstance(s, Series)

    def test_constructor_sanitize(self):
        s = Series(np.array([1.0, 1.0, 8.0]), dtype="i8")
        assert s.dtype == np.dtype("i8")

        s = Series(np.array([1.0, 1.0, np.nan]), copy=True, dtype="i8")
        assert s.dtype == np.dtype("f8")

    def test_constructor_copy(self):
        # GH15125
        # test dtype parameter has no side effects on copy=True
        for data in [[1.0], np.array([1.0])]:
            x = Series(data)
            y = Series(x, copy=True, dtype=float)

            # copy=True maintains original data in Series
            tm.assert_series_equal(x, y)

            # changes to origin of copy does not affect the copy
            x[0] = 2.0
            assert not x.equals(y)
            assert x[0] == 2.0
            assert y[0] == 1.0

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite test
    @pytest.mark.parametrize(
        "index",
        [
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170101", periods=3),
            timedelta_range("1 day", periods=3),
            period_range("2012Q1", periods=3, freq="Q"),
            Index(list("abc")),
            pd.Int64Index([1, 2, 3]),
            RangeIndex(0, 3),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_limit_copies(self, index):
        # GH 17449
        # limit copies of input
        s = Series(index)

        # we make 1 copy; this is just a smoke test here
        assert s._mgr.blocks[0].values is not index

    def test_constructor_pass_none(self):
        with tm.assert_produces_warning(DeprecationWarning):
            s = Series(None, index=range(5))
        assert s.dtype == np.float64

        s = Series(None, index=range(5), dtype=object)
        assert s.dtype == np.object_

        # GH 7431
        # inference on the index
        with tm.assert_produces_warning(DeprecationWarning):
            s = Series(index=np.array([None]))
            expected = Series(index=Index([None]))
        tm.assert_series_equal(s, expected)

    def test_constructor_pass_nan_nat(self):
        # GH 13467
        exp = Series([np.nan, np.nan], dtype=np.float64)
        assert exp.dtype == np.float64
        tm.assert_series_equal(Series([np.nan, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, np.nan])), exp)

        exp = Series([NaT, NaT])
        assert exp.dtype == "datetime64[ns]"
        tm.assert_series_equal(Series([NaT, NaT]), exp)
        tm.assert_series_equal(Series(np.array([NaT, NaT])), exp)

        tm.assert_series_equal(Series([NaT, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([NaT, np.nan])), exp)

        tm.assert_series_equal(Series([np.nan, NaT]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, NaT])), exp)

    def test_constructor_cast(self):
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            Series(["a", "b", "c"], dtype=float)

    def test_constructor_signed_int_overflow_deprecation(self):
        # GH#41734 disallow silent overflow
        msg = "Values are too large to be losslessly cast"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([1, 200, 923442], dtype="int8")

        expected = Series([1, -56, 50], dtype="int8")
        tm.assert_series_equal(ser, expected)

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([1, 200, 923442], dtype="uint8")

        expected = Series([1, 200, 50], dtype="uint8")
        tm.assert_series_equal(ser, expected)

    def test_constructor_unsigned_dtype_overflow(self, uint_dtype):
        # see gh-15832
        msg = "Trying to coerce negative values to unsigned integers"
        with pytest.raises(OverflowError, match=msg):
            Series([-1], dtype=uint_dtype)

    def test_constructor_coerce_float_fail(self, any_int_dtype):
        # see gh-15832
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            Series([1, 2, 3.5], dtype=any_int_dtype)

    def test_constructor_coerce_float_valid(self, float_dtype):
        s = Series([1, 2, 3.5], dtype=float_dtype)
        expected = Series([1, 2, 3.5]).astype(float_dtype)
        tm.assert_series_equal(s, expected)

    def test_constructor_invalid_coerce_ints_with_float_nan(self, any_int_dtype):
        # GH 22585

        msg = "cannot convert float NaN to integer"
        with pytest.raises(ValueError, match=msg):
            Series([1, 2, np.nan], dtype=any_int_dtype)

    def test_constructor_dtype_no_cast(self):
        # see gh-1572
        s = Series([1, 2, 3])
        s2 = Series(s, dtype=np.int64)

        s2[1] = 5
        assert s[1] == 5

    def test_constructor_datelike_coercion(self):

        # GH 9477
        # incorrectly inferring on dateimelike looking when object dtype is
        # specified
        s = Series([Timestamp("20130101"), "NOV"], dtype=object)
        assert s.iloc[0] == Timestamp("20130101")
        assert s.iloc[1] == "NOV"
        assert s.dtype == object

    def test_constructor_datelike_coercion2(self):
        # the dtype was being reset on the slicing and re-inferred to datetime
        # even thought the blocks are mixed
        belly = "216 3T19".split()
        wing1 = "2T15 4H19".split()
        wing2 = "416 4T20".split()
        mat = pd.to_datetime("2016-01-22 2019-09-07".split())
        df = DataFrame({"wing1": wing1, "wing2": wing2, "mat": mat}, index=belly)

        result = df.loc["3T19"]
        assert result.dtype == object
        result = df.loc["216"]
        assert result.dtype == object

    def test_constructor_mixed_int_and_timestamp(self, frame_or_series):
        # specifically Timestamp with nanos, not datetimes
        objs = [Timestamp(9), 10, NaT.value]
        result = frame_or_series(objs, dtype="M8[ns]")

        expected = frame_or_series([Timestamp(9), Timestamp(10), NaT])
        tm.assert_equal(result, expected)

    def test_constructor_datetimes_with_nulls(self):
        # gh-15869
        for arr in [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
        ]:
            result = Series(arr)
            assert result.dtype == "M8[ns]"

    def test_constructor_dtype_datetime64(self):

        s = Series(iNaT, dtype="M8[ns]", index=range(5))
        assert isna(s).all()

        # in theory this should be all nulls, but since
        # we are not specifying a dtype is ambiguous
        s = Series(iNaT, index=range(5))
        assert not isna(s).all()

        s = Series(np.nan, dtype="M8[ns]", index=range(5))
        assert isna(s).all()

        s = Series([datetime(2001, 1, 2, 0, 0), iNaT], dtype="M8[ns]")
        assert isna(s[1])
        assert s.dtype == "M8[ns]"

        s = Series([datetime(2001, 1, 2, 0, 0), np.nan], dtype="M8[ns]")
        assert isna(s[1])
        assert s.dtype == "M8[ns]"

    def test_constructor_dtype_datetime64_10(self):
        # GH3416
        dates = [
            np.datetime64(datetime(2013, 1, 1)),
            np.datetime64(datetime(2013, 1, 2)),
            np.datetime64(datetime(2013, 1, 3)),
        ]

        s = Series(dates)
        assert s.dtype == "M8[ns]"

        s.iloc[0] = np.nan
        assert s.dtype == "M8[ns]"

        # GH3414 related
        expected = Series(
            [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)],
            dtype="datetime64[ns]",
        )

        result = Series(Series(dates).view(np.int64) / 1000000, dtype="M8[ms]")
        tm.assert_series_equal(result, expected)

        result = Series(dates, dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

        expected = Series(
            [NaT, datetime(2013, 1, 2), datetime(2013, 1, 3)], dtype="datetime64[ns]"
        )
        result = Series([np.nan] + dates[1:], dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

        dts = Series(dates, dtype="datetime64[ns]")

        # valid astype
        with tm.assert_produces_warning(FutureWarning):
            # astype(np.int64) deprecated
            dts.astype("int64")

        # invalid casting
        msg = r"cannot astype a datetimelike from \[datetime64\[ns\]\] to \[int32\]"
        with pytest.raises(TypeError, match=msg):
            dts.astype("int32")

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        with tm.assert_produces_warning(FutureWarning):
            # astype(np.int64) deprecated
            result = Series(dts, dtype=np.int64)
            expected = Series(dts.astype(np.int64))
        tm.assert_series_equal(result, expected)

    def test_constructor_dtype_datetime64_9(self):
        # invalid dates can be help as object
        result = Series([datetime(2, 1, 1)])
        assert result[0] == datetime(2, 1, 1, 0, 0)

        result = Series([datetime(3000, 1, 1)])
        assert result[0] == datetime(3000, 1, 1, 0, 0)

    def test_constructor_dtype_datetime64_8(self):
        # don't mix types
        result = Series([Timestamp("20130101"), 1], index=["a", "b"])
        assert result["a"] == Timestamp("20130101")
        assert result["b"] == 1

    def test_constructor_dtype_datetime64_7(self):
        # GH6529
        # coerce datetime64 non-ns properly
        dates = date_range("01-Jan-2015", "01-Dec-2015", freq="M")
        values2 = dates.view(np.ndarray).astype("datetime64[ns]")
        expected = Series(values2, index=dates)

        for dtype in ["s", "D", "ms", "us", "ns"]:
            values1 = dates.view(np.ndarray).astype(f"M8[{dtype}]")
            result = Series(values1, dates)
            tm.assert_series_equal(result, expected)

        # GH 13876
        # coerce to non-ns to object properly
        expected = Series(values2, index=dates, dtype=object)
        for dtype in ["s", "D", "ms", "us", "ns"]:
            values1 = dates.view(np.ndarray).astype(f"M8[{dtype}]")
            result = Series(values1, index=dates, dtype=object)
            tm.assert_series_equal(result, expected)

        # leave datetime.date alone
        dates2 = np.array([d.date() for d in dates.to_pydatetime()], dtype=object)
        series1 = Series(dates2, dates)
        tm.assert_numpy_array_equal(series1.values, dates2)
        assert series1.dtype == object

    def test_constructor_dtype_datetime64_6(self):
        # these will correctly infer a datetime
        msg = "containing strings is deprecated"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([None, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == "datetime64[ns]"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([np.nan, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == "datetime64[ns]"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([NaT, None, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == "datetime64[ns]"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([NaT, np.nan, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == "datetime64[ns]"

    def test_constructor_dtype_datetime64_5(self):
        # tz-aware (UTC and other tz's)
        # GH 8411
        dr = date_range("20130101", periods=3)
        assert Series(dr).iloc[0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        assert str(Series(dr).iloc[0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        assert str(Series(dr).iloc[0].tz) == "US/Eastern"

    def test_constructor_dtype_datetime64_4(self):
        # non-convertible
        s = Series([1479596223000, -1479590, NaT])
        assert s.dtype == "object"
        assert s[2] is NaT
        assert "NaT" in str(s)

    def test_constructor_dtype_datetime64_3(self):
        # if we passed a NaT it remains
        s = Series([datetime(2010, 1, 1), datetime(2, 1, 1), NaT])
        assert s.dtype == "object"
        assert s[2] is NaT
        assert "NaT" in str(s)

    def test_constructor_dtype_datetime64_2(self):
        # if we passed a nan it remains
        s = Series([datetime(2010, 1, 1), datetime(2, 1, 1), np.nan])
        assert s.dtype == "object"
        assert s[2] is np.nan
        assert "NaN" in str(s)

    def test_constructor_with_datetime_tz(self):

        # 8260
        # support datetime64 with tz

        dr = date_range("20130101", periods=3, tz="US/Eastern")
        s = Series(dr)
        assert s.dtype.name == "datetime64[ns, US/Eastern]"
        assert s.dtype == "datetime64[ns, US/Eastern]"
        assert is_datetime64tz_dtype(s.dtype)
        assert "datetime64[ns, US/Eastern]" in str(s)

        # export
        result = s.values
        assert isinstance(result, np.ndarray)
        assert result.dtype == "datetime64[ns]"

        exp = DatetimeIndex(result)
        exp = exp.tz_localize("UTC").tz_convert(tz=s.dt.tz)
        tm.assert_index_equal(dr, exp)

        # indexing
        result = s.iloc[0]
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")
        result = s[0]
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")

        result = s[Series([True, True, False], index=s.index)]
        tm.assert_series_equal(result, s[0:2])

        result = s.iloc[0:1]
        tm.assert_series_equal(result, Series(dr[0:1]))

        # concat
        result = pd.concat([s.iloc[0:1], s.iloc[1:]])
        tm.assert_series_equal(result, s)

        # short str
        assert "datetime64[ns, US/Eastern]" in str(s)

        # formatting with NaT
        result = s.shift()
        assert "datetime64[ns, US/Eastern]" in str(result)
        assert "NaT" in str(result)

        # long str
        t = Series(date_range("20130101", periods=1000, tz="US/Eastern"))
        assert "datetime64[ns, US/Eastern]" in str(t)

        result = DatetimeIndex(s, freq="infer")
        tm.assert_index_equal(result, dr)

    def test_constructor_with_datetime_tz4(self):
        # inference
        s = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific"),
            ]
        )
        assert s.dtype == "datetime64[ns, US/Pacific]"
        assert lib.infer_dtype(s, skipna=True) == "datetime64"

    def test_constructor_with_datetime_tz3(self):
        s = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Eastern"),
            ]
        )
        assert s.dtype == "object"
        assert lib.infer_dtype(s, skipna=True) == "datetime"

    def test_constructor_with_datetime_tz2(self):
        # with all NaT
        s = Series(NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        expected = Series(DatetimeIndex(["NaT", "NaT"], tz="US/Eastern"))
        tm.assert_series_equal(s, expected)

    @pytest.mark.parametrize("arr_dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("dtype", ["M8", "m8"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_construction_to_datetimelike_unit(self, arr_dtype, dtype, unit):
        # tests all units
        # gh-19223
        dtype = f"{dtype}[{unit}]"
        arr = np.array([1, 2, 3], dtype=arr_dtype)
        s = Series(arr)
        result = s.astype(dtype)
        expected = Series(arr.astype(dtype))

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("arg", ["2013-01-01 00:00:00", NaT, np.nan, None])
    def test_constructor_with_naive_string_and_datetimetz_dtype(self, arg):
        # GH 17415: With naive string
        result = Series([arg], dtype="datetime64[ns, CET]")
        expected = Series(Timestamp(arg)).dt.tz_localize("CET")
        tm.assert_series_equal(result, expected)

    def test_constructor_datetime64_bigendian(self):
        # GH#30976
        ms = np.datetime64(1, "ms")
        arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")

        result = Series(arr)
        expected = Series([Timestamp(ms)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("interval_constructor", [IntervalIndex, IntervalArray])
    def test_construction_interval(self, interval_constructor):
        # construction from interval & array of intervals
        intervals = interval_constructor.from_breaks(np.arange(3), closed="right")
        result = Series(intervals)
        assert result.dtype == "interval[int64, right]"
        tm.assert_index_equal(Index(result.values), Index(intervals))

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_infer_interval(self, data_constructor):
        # GH 23563: consistent closed results in interval dtype
        data = [Interval(0, 1), Interval(0, 2), None]
        result = Series(data_constructor(data))
        expected = Series(IntervalArray(data))
        assert result.dtype == "interval[float64, right]"
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_interval_mixed_closed(self, data_constructor):
        # GH 23563: mixed closed results in object dtype (not interval dtype)
        data = [Interval(0, 1, closed="both"), Interval(0, 2, closed="neither")]
        result = Series(data_constructor(data))
        assert result.dtype == object
        assert result.tolist() == data

    def test_construction_consistency(self):

        # make sure that we are not re-localizing upon construction
        # GH 14928
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

        result = Series(ser, dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        result = Series(ser.dt.tz_convert("UTC"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        msg = "will interpret the data as wall-times"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # deprecate behavior inconsistent with DatetimeIndex GH#33401
            result = Series(ser.values, dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        with tm.assert_produces_warning(None):
            # one suggested alternative to the deprecated usage
            middle = Series(ser.values).dt.tz_localize("UTC")
            result = middle.dt.tz_convert(ser.dtype.tz)
        tm.assert_series_equal(result, ser)

        with tm.assert_produces_warning(None):
            # the other suggested alternative to the deprecated usage
            result = Series(ser.values.view("int64"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    def test_constructor_infer_period(self, data_constructor):
        data = [Period("2000", "D"), Period("2001", "D"), None]
        result = Series(data_constructor(data))
        expected = Series(period_array(data))
        tm.assert_series_equal(result, expected)
        assert result.dtype == "Period[D]"

    @pytest.mark.xfail(reason="PeriodDtype Series not supported yet")
    def test_construct_from_ints_including_iNaT_scalar_period_dtype(self):
        series = Series([0, 1000, 2000, pd._libs.iNaT], dtype="period[D]")

        val = series[3]
        assert isna(val)

        series[2] = val
        assert isna(series[2])

    def test_constructor_period_incompatible_frequency(self):
        data = [Period("2000", "D"), Period("2001", "A")]
        result = Series(data)
        assert result.dtype == object
        assert result.tolist() == data

    def test_constructor_periodindex(self):
        # GH7932
        # converting a PeriodIndex when put in a Series

        pi = period_range("20130101", periods=5, freq="D")
        s = Series(pi)
        assert s.dtype == "Period[D]"
        expected = Series(pi.astype(object))
        tm.assert_series_equal(s, expected)

    def test_constructor_dict(self):
        d = {"a": 0.0, "b": 1.0, "c": 2.0}

        result = Series(d)
        expected = Series(d, index=sorted(d.keys()))
        tm.assert_series_equal(result, expected)

        result = Series(d, index=["b", "c", "d", "a"])
        expected = Series([1, 2, np.nan, 0], index=["b", "c", "d", "a"])
        tm.assert_series_equal(result, expected)

        pidx = tm.makePeriodIndex(100)
        d = {pidx[0]: 0, pidx[1]: 1}
        result = Series(d, index=pidx)
        expected = Series(np.nan, pidx, dtype=np.float64)
        expected.iloc[0] = 0
        expected.iloc[1] = 1
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_list_value_explicit_dtype(self):
        # GH 18625
        d = {"a": [[2], [3], [4]]}
        result = Series(d, index=["a"], dtype="object")
        expected = Series(d, index=["a"])
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_order(self):
        # GH19018
        # initialization ordering: by insertion order if python>= 3.6, else
        # order by value
        d = {"b": 1, "a": 0, "c": 2}
        result = Series(d)
        expected = Series([1, 0, 2], index=list("bac"))
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_extension(self, ea_scalar_and_dtype):
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        d = {"a": ea_scalar}
        result = Series(d, index=["a"])
        expected = Series(ea_scalar, index=["a"], dtype=ea_dtype)

        assert result.dtype == ea_dtype

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # GH 18480
        d = {1: "a", value: "b", float("nan"): "c", 4: "d"}
        result = Series(d).sort_values()
        expected = Series(["a", "b", "c", "d"], index=[1, value, np.nan, 4])
        tm.assert_series_equal(result, expected)

        # MultiIndex:
        d = {(1, 1): "a", (2, np.nan): "b", (3, value): "c"}
        result = Series(d).sort_values()
        expected = Series(
            ["a", "b", "c"], index=Index([(1, 1), (2, np.nan), (3, value)])
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_datetime64_index(self):
        # GH 9456

        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]
        values = [42544017.198965244, 1234565, 40512335.181958228, -1]

        def create_data(constructor):
            return dict(zip((constructor(x) for x in dates_as_str), values))

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        expected = Series(values, (Timestamp(x) for x in dates_as_str))

        result_datetime64 = Series(data_datetime64)
        result_datetime = Series(data_datetime)
        result_Timestamp = Series(data_Timestamp)

        tm.assert_series_equal(result_datetime64, expected)
        tm.assert_series_equal(result_datetime, expected)
        tm.assert_series_equal(result_Timestamp, expected)

    def test_constructor_dict_tuple_indexer(self):
        # GH 12948
        data = {(1, 1, None): -1.0}
        result = Series(data)
        expected = Series(
            -1.0, index=MultiIndex(levels=[[1], [1], [np.nan]], codes=[[0], [0], [-1]])
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_mapping(self, non_dict_mapping_subclass):
        # GH 29788
        ndm = non_dict_mapping_subclass({3: "three"})
        result = Series(ndm)
        expected = Series(["three"], index=[3])

        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        data = [(1, 1), (2, 2), (2, 3)]
        s = Series(data)
        assert list(s) == data

    def test_constructor_tuple_of_tuples(self):
        data = ((1, 1), (2, 2), (2, 3))
        s = Series(data)
        assert tuple(s) == data

    def test_constructor_dict_of_tuples(self):
        data = {(1, 2): 3, (None, 5): 6}
        result = Series(data).sort_values()
        expected = Series([3, 6], index=MultiIndex.from_tuples([(1, 2), (None, 5)]))
        tm.assert_series_equal(result, expected)

    # https://github.com/pandas-dev/pandas/issues/22698
    @pytest.mark.filterwarnings("ignore:elementwise comparison:FutureWarning")
    def test_fromDict(self):
        data = {"a": 0, "b": 1, "c": 2, "d": 3}

        series = Series(data)
        tm.assert_is_sorted(series.index)

        data = {"a": 0, "b": "1", "c": "2", "d": datetime.now()}
        series = Series(data)
        assert series.dtype == np.object_

        data = {"a": 0, "b": "1", "c": "2", "d": "3"}
        series = Series(data)
        assert series.dtype == np.object_

        data = {"a": "0", "b": "1"}
        series = Series(data, dtype=float)
        assert series.dtype == np.float64

    def test_fromValue(self, datetime_series):

        nans = Series(np.NaN, index=datetime_series.index, dtype=np.float64)
        assert nans.dtype == np.float_
        assert len(nans) == len(datetime_series)

        strings = Series("foo", index=datetime_series.index)
        assert strings.dtype == np.object_
        assert len(strings) == len(datetime_series)

        d = datetime.now()
        dates = Series(d, index=datetime_series.index)
        assert dates.dtype == "M8[ns]"
        assert len(dates) == len(datetime_series)

        # GH12336
        # Test construction of categorical series from value
        categorical = Series(0, index=datetime_series.index, dtype="category")
        expected = Series(0, index=datetime_series.index).astype("category")
        assert categorical.dtype == "category"
        assert len(categorical) == len(datetime_series)
        tm.assert_series_equal(categorical, expected)

    def test_constructor_dtype_timedelta64(self):

        # basic
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1)])
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1), timedelta(days=2), np.timedelta64(1, "s")])

        assert td.dtype == "timedelta64[ns]"

        # mixed with NaT
        td = Series([timedelta(days=1), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        td = Series([timedelta(days=1), np.nan], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        td = Series([np.timedelta64(300000000), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"

        # improved inference
        # GH5689
        td = Series([np.timedelta64(300000000), NaT])
        assert td.dtype == "timedelta64[ns]"

        # because iNaT is int, not coerced to timedelta
        td = Series([np.timedelta64(300000000), iNaT])
        assert td.dtype == "object"

        td = Series([np.timedelta64(300000000), np.nan])
        assert td.dtype == "timedelta64[ns]"

        td = Series([NaT, np.timedelta64(300000000)])
        assert td.dtype == "timedelta64[ns]"

        td = Series([np.timedelta64(1, "s")])
        assert td.dtype == "timedelta64[ns]"

        # FIXME: dont leave commented-out
        # these are frequency conversion astypes
        # for t in ['s', 'D', 'us', 'ms']:
        #    with pytest.raises(TypeError):
        #        td.astype('m8[%s]' % t)

        # valid astype
        with tm.assert_produces_warning(FutureWarning):
            # astype(int64) deprecated
            td.astype("int64")

        # invalid casting
        msg = r"cannot astype a datetimelike from \[timedelta64\[ns\]\] to \[int32\]"
        with pytest.raises(TypeError, match=msg):
            td.astype("int32")

        # this is an invalid casting
        msg = "|".join(
            [
                "Could not convert object to NumPy timedelta",
                "Could not convert 'foo' to NumPy timedelta",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            Series([timedelta(days=1), "foo"], dtype="m8[ns]")

        # leave as object here
        td = Series([timedelta(days=i) for i in range(3)] + ["foo"])
        assert td.dtype == "object"

        # these will correctly infer a timedelta
        msg = "containing strings is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([None, NaT, "1 Day"])
        assert ser.dtype == "timedelta64[ns]"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([np.nan, NaT, "1 Day"])
        assert ser.dtype == "timedelta64[ns]"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([NaT, None, "1 Day"])
        assert ser.dtype == "timedelta64[ns]"

        with tm.assert_produces_warning(FutureWarning, match=msg):
            ser = Series([NaT, np.nan, "1 Day"])
        assert ser.dtype == "timedelta64[ns]"

    # GH 16406
    def test_constructor_mixed_tz(self):
        s = Series([Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")])
        expected = Series(
            [Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(s, expected)

    def test_NaT_scalar(self):
        series = Series([0, 1000, 2000, iNaT], dtype="M8[ns]")

        val = series[3]
        assert isna(val)

        series[2] = val
        assert isna(series[2])

    def test_NaT_cast(self):
        # GH10747
        result = Series([np.nan]).astype("M8[ns]")
        expected = Series([NaT])
        tm.assert_series_equal(result, expected)

    def test_constructor_name_hashable(self):
        for n in [777, 777.0, "name", datetime(2001, 11, 11), (1,), "\u05D0"]:
            for data in [[1, 2, 3], np.ones(3), {"a": 0, "b": 1}]:
                s = Series(data, name=n)
                assert s.name == n

    def test_constructor_name_unhashable(self):
        msg = r"Series\.name must be a hashable type"
        for n in [["name_list"], np.ones(2), {1: 2}]:
            for data in [["name_list"], np.ones(2), {1: 2}]:
                with pytest.raises(TypeError, match=msg):
                    Series(data, name=n)

    def test_auto_conversion(self):
        series = Series(list(date_range("1/1/2000", periods=10)))
        assert series.dtype == "M8[ns]"

    def test_convert_non_ns(self):
        # convert from a numpy array of non-ns timedelta64
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")
        s = Series(arr)
        expected = Series(timedelta_range("00:00:01", periods=3, freq="s"))
        tm.assert_series_equal(s, expected)

        # convert from a numpy array of non-ns datetime64
        # note that creating a numpy datetime64 is in LOCAL time!!!!
        # seems to work for M8[D], but not for M8[s]
        # TODO: is the above comment still accurate/needed?

        arr = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )
        ser = Series(arr)
        expected = Series(date_range("20130101", periods=3, freq="D"))
        tm.assert_series_equal(ser, expected)

        arr = np.array(
            ["2013-01-01 00:00:01", "2013-01-01 00:00:02", "2013-01-01 00:00:03"],
            dtype="datetime64[s]",
        )
        ser = Series(arr)
        expected = Series(date_range("20130101 00:00:01", periods=3, freq="s"))
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),
            timedelta_range("1 day", periods=10),
            period_range("2000-Q1", periods=10, freq="Q"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_cant_cast_datetimelike(self, index):

        # floats are not ok
        # strip Index to convert PeriodIndex -> Period
        # We don't care whether the error message says
        # PeriodIndex or PeriodArray
        msg = f"Cannot cast {type(index).__name__.rstrip('Index')}.*? to "

        with pytest.raises(TypeError, match=msg):
            Series(index, dtype=float)

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        with tm.assert_produces_warning(FutureWarning):
            # asype(np.int64) deprecated, use .view(np.int64) instead
            result = Series(index, dtype=np.int64)
            expected = Series(index.astype(np.int64))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),
            timedelta_range("1 day", periods=10),
            period_range("2000-Q1", periods=10, freq="Q"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_constructor_cast_object(self, index):
        s = Series(index, dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(Index(index, dtype=object), dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(index.astype(object), dtype=object)
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    def test_constructor_generic_timestamp_no_frequency(self, dtype, request):
        # see gh-15524, gh-15987
        msg = "dtype has no unit. Please pass in"

        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.node.add_marker(mark)

        with pytest.raises(ValueError, match=msg):
            Series([], dtype=dtype)

    @pytest.mark.parametrize(
        "dtype,msg",
        [
            ("m8[ps]", "cannot convert timedeltalike"),
            ("M8[ps]", "cannot convert datetimelike"),
        ],
    )
    def test_constructor_generic_timestamp_bad_frequency(self, dtype, msg):
        # see gh-15524, gh-15987

        with pytest.raises(TypeError, match=msg):
            Series([], dtype=dtype)

    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype):
        # GH 16804
        expected = Series([0, 1, 2, 3, 4], dtype=dtype or "int64")
        result = Series(range(5), dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_constructor_range_overflows(self):
        # GH#30173 range objects that overflow int64
        rng = range(2 ** 63, 2 ** 63 + 4)
        ser = Series(rng)
        expected = Series(list(rng))
        tm.assert_series_equal(ser, expected)
        assert list(ser) == list(rng)
        assert ser.dtype == np.uint64

        rng2 = range(2 ** 63 + 4, 2 ** 63, -1)
        ser2 = Series(rng2)
        expected2 = Series(list(rng2))
        tm.assert_series_equal(ser2, expected2)
        assert list(ser2) == list(rng2)
        assert ser2.dtype == np.uint64

        rng3 = range(-(2 ** 63), -(2 ** 63) - 4, -1)
        ser3 = Series(rng3)
        expected3 = Series(list(rng3))
        tm.assert_series_equal(ser3, expected3)
        assert list(ser3) == list(rng3)
        assert ser3.dtype == object

        rng4 = range(2 ** 73, 2 ** 73 + 4)
        ser4 = Series(rng4)
        expected4 = Series(list(rng4))
        tm.assert_series_equal(ser4, expected4)
        assert list(ser4) == list(rng4)
        assert ser4.dtype == object

    def test_constructor_tz_mixed_data(self):
        # GH 13051
        dt_list = [
            Timestamp("2016-05-01 02:03:37"),
            Timestamp("2016-04-30 19:03:37-0700", tz="US/Pacific"),
        ]
        result = Series(dt_list)
        expected = Series(dt_list, dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # GH#25843, GH#41555, GH#33401
        tz = tz_aware_fixture
        ts = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()
        ts_naive = Timestamp("2019")

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = Series([ts], dtype="datetime64[ns]")
        expected = Series([ts_naive])
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            result = Series(np.array([ts], dtype=object), dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning):
            result = Series({0: ts}, dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning):
            result = Series(ts, index=[0], dtype="datetime64[ns]")
        tm.assert_series_equal(result, expected)

    def test_constructor_datetime64(self):
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        series = Series(dates)
        assert np.issubdtype(series.dtype, np.dtype("M8[ns]"))

    def test_constructor_datetimelike_scalar_to_string_dtype(
        self, nullable_string_dtype
    ):
        # https://github.com/pandas-dev/pandas/pull/33846
        result = Series("M", index=[1, 2, 3], dtype=nullable_string_dtype)
        expected = Series(["M", "M", "M"], index=[1, 2, 3], dtype=nullable_string_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            [np.datetime64("2012-01-01"), np.datetime64("2013-01-01")],
            ["2012-01-01", "2013-01-01"],
        ],
    )
    def test_constructor_sparse_datetime64(self, values):
        # https://github.com/pandas-dev/pandas/issues/35762
        dtype = pd.SparseDtype("datetime64[ns]")
        result = Series(values, dtype=dtype)
        arr = pd.arrays.SparseArray(values, dtype=dtype)
        expected = Series(arr)
        tm.assert_series_equal(result, expected)

    def test_construction_from_ordered_collection(self):
        # https://github.com/pandas-dev/pandas/issues/36044
        result = Series({"a": 1, "b": 2}.keys())
        expected = Series(["a", "b"])
        tm.assert_series_equal(result, expected)

        result = Series({"a": 1, "b": 2}.values())
        expected = Series([1, 2])
        tm.assert_series_equal(result, expected)

    def test_construction_from_large_int_scalar_no_overflow(self):
        # https://github.com/pandas-dev/pandas/issues/36291
        n = 1_000_000_000_000_000_000_000
        result = Series(n, index=[0])
        expected = Series(n)
        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_periods_infers_period_dtype(self):
        series = Series(list(period_range("2000-01-01", periods=10, freq="D")))
        assert series.dtype == "Period[D]"

        series = Series(
            [Period("2011-01-01", freq="D"), Period("2011-02-01", freq="D")]
        )
        assert series.dtype == "Period[D]"

    def test_constructor_subclass_dict(self, dict_subclass):
        data = dict_subclass((x, 10.0 * x) for x in range(10))
        series = Series(data)
        expected = Series(dict(data.items()))
        tm.assert_series_equal(series, expected)

    def test_constructor_ordereddict(self):
        # GH3283
        data = OrderedDict((f"col{i}", np.random.random()) for i in range(12))

        series = Series(data)
        expected = Series(list(data.values()), list(data.keys()))
        tm.assert_series_equal(series, expected)

        # Test with subclass
        class A(OrderedDict):
            pass

        series = Series(A(data))
        tm.assert_series_equal(series, expected)

    def test_constructor_dict_multiindex(self):
        d = {("a", "a"): 0.0, ("b", "a"): 1.0, ("b", "c"): 2.0}
        _d = sorted(d.items())
        result = Series(d)
        expected = Series(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        )
        tm.assert_series_equal(result, expected)

        d["z"] = 111.0
        _d.insert(0, ("z", d["z"]))
        result = Series(d)
        expected = Series(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        )
        result = result.reindex(index=expected.index)
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_multiindex_reindex_flat(self):
        # construction involves reindexing with a MultiIndex corner case
        data = {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2, "j": np.nan}
        expected = Series(data)

        result = Series(expected[:-1].to_dict(), index=expected.index)
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_timedelta_index(self):
        # GH #12169 : Resample category data with timedelta index
        # construct Series from dict as data and TimedeltaIndex as index
        # will result NaN in result Series data
        expected = Series(
            data=["A", "B", "C"], index=pd.to_timedelta([0, 10, 20], unit="s")
        )

        result = Series(
            data={
                pd.to_timedelta(0, unit="s"): "A",
                pd.to_timedelta(10, unit="s"): "B",
                pd.to_timedelta(20, unit="s"): "C",
            },
            index=pd.to_timedelta([0, 10, 20], unit="s"),
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_infer_index_tz(self):
        values = [188.5, 328.25]
        tzinfo = tzoffset(None, 7200)
        index = [
            datetime(2012, 5, 11, 11, tzinfo=tzinfo),
            datetime(2012, 5, 11, 12, tzinfo=tzinfo),
        ]
        series = Series(data=values, index=index)

        assert series.index.tz == tzinfo

        # it works! GH#2443
        repr(series.index[0])

    def test_constructor_with_pandas_dtype(self):
        # going through 2D->1D path
        vals = [(1,), (2,), (3,)]
        ser = Series(vals)
        dtype = ser.array.dtype  # PandasDtype
        ser2 = Series(vals, dtype=dtype)
        tm.assert_series_equal(ser, ser2)


class TestSeriesConstructorIndexCoercion:
    def test_series_constructor_datetimelike_index_coercion(self):
        idx = tm.makeDateIndex(10000)
        ser = Series(np.random.randn(len(idx)), idx.astype(object))
        with tm.assert_produces_warning(FutureWarning):
            assert ser.index.is_all_dates
        assert isinstance(ser.index, DatetimeIndex)

    def test_series_constructor_infer_multiindex(self):
        index_lists = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]

        multi = Series(1.0, index=[np.array(x) for x in index_lists])
        assert isinstance(multi.index, MultiIndex)

        multi = Series(1.0, index=index_lists)
        assert isinstance(multi.index, MultiIndex)

        multi = Series(range(4), index=index_lists)
        assert isinstance(multi.index, MultiIndex)


class TestSeriesConstructorInternals:
    def test_constructor_no_pandas_array(self, using_array_manager):
        ser = Series([1, 2, 3])
        result = Series(ser.array)
        tm.assert_series_equal(ser, result)
        if not using_array_manager:
            assert isinstance(result._mgr.blocks[0], NumericBlock)

    @td.skip_array_manager_invalid_test
    def test_from_array(self):
        result = Series(pd.array(["1H", "2H"], dtype="timedelta64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

        result = Series(pd.array(["2015"], dtype="datetime64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

    @td.skip_array_manager_invalid_test
    def test_from_list_dtype(self):
        result = Series(["1H", "2H"], dtype="timedelta64[ns]")
        assert result._mgr.blocks[0].is_extension is False

        result = Series(["2015"], dtype="datetime64[ns]")
        assert result._mgr.blocks[0].is_extension is False


def test_constructor(rand_series_with_duplicate_datetimeindex):
    dups = rand_series_with_duplicate_datetimeindex
    assert isinstance(dups, Series)
    assert isinstance(dups.index, DatetimeIndex)
