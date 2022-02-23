import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_arm

import pandas as pd
from pandas import (
    Float64Index,
    Index,
    Int64Index,
    Series,
    UInt64Index,
)
import pandas._testing as tm
from pandas.tests.indexes.common import NumericBase


class TestFloat64Index(NumericBase):
    _index_cls = Float64Index

    @pytest.fixture(params=[np.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(
        params=["int64", "uint64", "category", "datetime64", "object"],
    )
    def invalid_dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        values = np.arange(5, dtype=dtype)
        return self._index_cls(values)

    @pytest.fixture(
        params=[
            [1.5, 2, 3, 4, 5],
            [0.0, 2.5, 5.0, 7.5, 10.0],
            [5, 4, 3, 2, 1.5],
            [10.0, 7.5, 5.0, 2.5, 0.0],
        ],
        ids=["mixed", "float", "mixed_dec", "float_dec"],
    )
    def index(self, request, dtype):
        return self._index_cls(request.param, dtype=dtype)

    @pytest.fixture
    def mixed_index(self, dtype):
        return self._index_cls([1.5, 2, 3, 4, 5], dtype=dtype)

    @pytest.fixture
    def float_index(self, dtype):
        return self._index_cls([0.0, 2.5, 5.0, 7.5, 10.0], dtype=dtype)

    def test_repr_roundtrip(self, index):
        tm.assert_index_equal(eval(repr(index)), index)

    def check_is_index(self, idx):
        assert isinstance(idx, Index)
        assert not isinstance(idx, self._index_cls)

    def check_coerce(self, a, b, is_float_index=True):
        assert a.equals(b)
        tm.assert_index_equal(a, b, exact=False)
        if is_float_index:
            assert isinstance(b, self._index_cls)
        else:
            self.check_is_index(b)

    def test_constructor(self, dtype):
        index_cls = self._index_cls

        # explicit construction
        index = index_cls([1, 2, 3, 4, 5], dtype=dtype)

        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        expected = np.array([1, 2, 3, 4, 5], dtype=dtype)
        tm.assert_numpy_array_equal(index.values, expected)

        index = index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=np.float32)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        index = index_cls(np.array([1, 2, 3, 4, 5]), dtype=np.float32)
        assert isinstance(index, index_cls)
        assert index.dtype == dtype

        # nan handling
        result = index_cls([np.nan, np.nan], dtype=dtype)
        assert pd.isna(result.values).all()

        result = index_cls(np.array([np.nan]), dtype=dtype)
        assert pd.isna(result.values).all()

        result = Index(np.array([np.nan], dtype=dtype))
        assert isinstance(result, index_cls)
        assert result.dtype == dtype
        assert pd.isna(result.values).all()

    def test_constructor_invalid(self):
        index_cls = self._index_cls
        cls_name = index_cls.__name__

        # invalid
        msg = (
            rf"{cls_name}\(\.\.\.\) must be called with a collection of "
            r"some kind, 0\.0 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            index_cls(0.0)

        # 2021-02-1 we get ValueError in numpy 1.20, but not on all builds
        msg = "|".join(
            [
                "String dtype not supported, you may need to explicitly cast ",
                "could not convert string to float: 'a'",
            ]
        )
        with pytest.raises((TypeError, ValueError), match=msg):
            index_cls(["a", "b", 0.0])

        msg = (
            r"float\(\) argument must be a string or a( real)? number, not 'Timestamp'"
        )
        with pytest.raises(TypeError, match=msg):
            index_cls([Timestamp("20130101")])

    def test_constructor_coerce(self, mixed_index, float_index):

        self.check_coerce(mixed_index, Index([1.5, 2, 3, 4, 5]))
        self.check_coerce(float_index, Index(np.arange(5) * 2.5))
        self.check_coerce(
            float_index, Index(np.array(np.arange(5) * 2.5, dtype=object))
        )

    def test_constructor_explicit(self, mixed_index, float_index):

        # these don't auto convert
        self.check_coerce(
            float_index, Index((np.arange(5) * 2.5), dtype=object), is_float_index=False
        )
        self.check_coerce(
            mixed_index, Index([1.5, 2, 3, 4, 5], dtype=object), is_float_index=False
        )

    def test_type_coercion_fail(self, any_int_dtype):
        # see gh-15832
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3.5], dtype=any_int_dtype)

    def test_type_coercion_valid(self, float_dtype):
        # There is no Float32Index, so we always
        # generate Float64Index.
        idx = Index([1, 2, 3.5], dtype=float_dtype)
        tm.assert_index_equal(idx, Index([1, 2, 3.5]))

    def test_equals_numeric(self):
        index_cls = self._index_cls

        idx = index_cls([1.0, 2.0])
        assert idx.equals(idx)
        assert idx.identical(idx)

        idx2 = index_cls([1.0, 2.0])
        assert idx.equals(idx2)

        idx = index_cls([1.0, np.nan])
        assert idx.equals(idx)
        assert idx.identical(idx)

        idx2 = index_cls([1.0, np.nan])
        assert idx.equals(idx2)

    @pytest.mark.parametrize(
        "other",
        (
            Int64Index([1, 2]),
            Index([1.0, 2.0], dtype=object),
            Index([1, 2], dtype=object),
        ),
    )
    def test_equals_numeric_other_index_type(self, other):
        idx = self._index_cls([1.0, 2.0])
        assert idx.equals(other)
        assert other.equals(idx)

    @pytest.mark.parametrize(
        "vals",
        [
            pd.date_range("2016-01-01", periods=3),
            pd.timedelta_range("1 Day", periods=3),
        ],
    )
    def test_lookups_datetimelike_values(self, vals, dtype):

        # If we have datetime64 or timedelta64 values, make sure they are
        #  wrappped correctly  GH#31163
        ser = Series(vals, index=range(3, 6))
        ser.index = ser.index.astype(dtype)

        expected = vals[1]

        with tm.assert_produces_warning(FutureWarning):
            result = ser.index.get_value(ser, 4.0)
        assert isinstance(result, type(expected)) and result == expected
        with tm.assert_produces_warning(FutureWarning):
            result = ser.index.get_value(ser, 4)
        assert isinstance(result, type(expected)) and result == expected

        result = ser[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser[4]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.loc[4.0]
        assert isinstance(result, type(expected)) and result == expected
        result = ser.loc[4]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.at[4.0]
        assert isinstance(result, type(expected)) and result == expected
        # GH#31329 .at[4] should cast to 4.0, matching .loc behavior
        result = ser.at[4]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.iloc[1]
        assert isinstance(result, type(expected)) and result == expected

        result = ser.iat[1]
        assert isinstance(result, type(expected)) and result == expected

    def test_doesnt_contain_all_the_things(self):
        idx = self._index_cls([np.nan])
        assert not idx.isin([0]).item()
        assert not idx.isin([1]).item()
        assert idx.isin([np.nan]).item()

    def test_nan_multiple_containment(self):
        index_cls = self._index_cls

        idx = index_cls([1.0, np.nan])
        tm.assert_numpy_array_equal(idx.isin([1.0]), np.array([True, False]))
        tm.assert_numpy_array_equal(idx.isin([2.0, np.pi]), np.array([False, False]))
        tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, True]))
        tm.assert_numpy_array_equal(idx.isin([1.0, np.nan]), np.array([True, True]))
        idx = index_cls([1.0, 2.0])
        tm.assert_numpy_array_equal(idx.isin([np.nan]), np.array([False, False]))

    def test_fillna_float64(self):
        # GH 11343
        idx = Index([1.0, np.nan, 3.0], dtype=float, name="x")
        # can't downcast
        exp = Index([1.0, 0.1, 3.0], name="x")
        tm.assert_index_equal(idx.fillna(0.1), exp)

        # downcast
        exp = self._index_cls([1.0, 2.0, 3.0], name="x")
        tm.assert_index_equal(idx.fillna(2), exp)

        # object
        exp = Index([1.0, "obj", 3.0], name="x")
        tm.assert_index_equal(idx.fillna("obj"), exp)


class NumericInt(NumericBase):
    def test_view(self, dtype):
        index_cls = self._index_cls

        idx = index_cls([], dtype=dtype, name="Foo")
        idx_view = idx.view()
        assert idx_view.name == "Foo"

        idx_view = idx.view(dtype)
        tm.assert_index_equal(idx, index_cls(idx_view, name="Foo"))

        idx_view = idx.view(index_cls)
        tm.assert_index_equal(idx, index_cls(idx_view, name="Foo"))

    def test_is_monotonic(self):
        index_cls = self._index_cls

        index = index_cls([1, 2, 3, 4])
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_decreasing is False

        index = index_cls([4, 3, 2, 1])
        assert index.is_monotonic is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is True

        index = index_cls([1])
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    def test_is_strictly_monotonic(self):
        index_cls = self._index_cls

        index = index_cls([1, 1, 2, 3])
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is False

        index = index_cls([3, 2, 1, 1])
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is False

        index = index_cls([1, 1])
        assert index.is_monotonic_increasing
        assert index.is_monotonic_decreasing
        assert not index._is_strictly_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index, dtype):
        index = simple_index

        idx = Index(index.copy())
        assert idx.identical(index)

        same_values_different_type = Index(idx, dtype=object)
        assert not idx.identical(same_values_different_type)

        idx = index.astype(dtype=object)
        idx = idx.rename("foo")
        same_values = Index(idx, dtype=object)
        assert same_values.identical(idx)

        assert not idx.identical(index)
        assert Index(same_values, name="foo", dtype=object).identical(idx)

        assert not index.astype(dtype=object).identical(index.astype(dtype=dtype))

    def test_cant_or_shouldnt_cast(self):
        msg = (
            "String dtype not supported, "
            "you may need to explicitly cast to a numeric type"
        )
        # can't
        data = ["foo", "bar", "baz"]
        with pytest.raises(TypeError, match=msg):
            self._index_cls(data)

        # shouldn't
        data = ["0", "1", "2"]
        with pytest.raises(TypeError, match=msg):
            self._index_cls(data)

    def test_view_index(self, simple_index):
        index = simple_index
        index.view(Index)

    def test_prevent_casting(self, simple_index):
        index = simple_index
        result = index.astype("O")
        assert result.dtype == np.object_


class TestInt64Index(NumericInt):
    _index_cls = Int64Index

    @pytest.fixture(params=[np.int64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(
        params=["uint64", "float64", "category", "datetime64", "object"],
    )
    def invalid_dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        return self._index_cls(range(0, 20, 2), dtype=dtype)

    @pytest.fixture(
        params=[range(0, 20, 2), range(19, -1, -1)], ids=["index_inc", "index_dec"]
    )
    def index(self, request, dtype):
        return self._index_cls(request.param, dtype=dtype)

    def test_constructor(self, dtype):
        index_cls = self._index_cls

        # pass list, coerce fine
        index = index_cls([-5, 0, 1, 2], dtype=dtype)
        expected = Index([-5, 0, 1, 2], dtype=dtype)
        tm.assert_index_equal(index, expected)

        # from iterable
        index = index_cls(iter([-5, 0, 1, 2]))
        tm.assert_index_equal(index, expected)

        # scalar raise Exception
        msg = (
            rf"{index_cls.__name__}\(\.\.\.\) must be called with a collection of some "
            "kind, 5 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            index_cls(5)

        # copy
        arr = index.values
        new_index = index_cls(arr, copy=True)
        tm.assert_index_equal(new_index, index)
        val = arr[0] + 3000

        # this should not change index
        arr[0] = val
        assert new_index[0] != val

        # interpret list-like
        expected = index_cls([5, 0])
        for cls in [Index, index_cls]:
            for idx in [
                cls([5, 0], dtype=dtype),
                cls(np.array([5, 0]), dtype=dtype),
                cls(Series([5, 0]), dtype=dtype),
            ]:
                tm.assert_index_equal(idx, expected)

    def test_constructor_corner(self, dtype):
        index_cls = self._index_cls

        arr = np.array([1, 2, 3, 4], dtype=object)
        index = index_cls(arr)
        assert index.values.dtype == dtype
        tm.assert_index_equal(index, Index(arr))

        # preventing casting
        arr = np.array([1, "2", 3, "4"], dtype=object)
        with pytest.raises(TypeError, match="casting"):
            index_cls(arr)

        arr_with_floats = [0, 2, 3, 4, 5, 1.25, 3, -1]
        with pytest.raises(TypeError, match="casting"):
            index_cls(arr_with_floats)

    def test_constructor_coercion_signed_to_unsigned(self, uint_dtype):

        # see gh-15832
        msg = "Trying to coerce negative values to unsigned integers"

        with pytest.raises(OverflowError, match=msg):
            Index([-1], dtype=uint_dtype)

    def test_coerce_list(self):
        # coerce things
        arr = Index([1, 2, 3, 4])
        assert isinstance(arr, self._index_cls)

        # but not if explicit dtype passed
        arr = Index([1, 2, 3, 4], dtype=object)
        assert type(arr) is Index


class TestUInt64Index(NumericInt):

    _index_cls = UInt64Index

    @pytest.fixture
    def dtype(self):
        return np.uint64

    @pytest.fixture(
        params=["int64", "float64", "category", "datetime64", "object"],
    )
    def invalid_dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self, dtype):
        # compat with shared Int64/Float64 tests
        return self._index_cls(np.arange(5, dtype=dtype))

    @pytest.fixture(
        params=[
            [2 ** 63, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20, 2 ** 63 + 25],
            [2 ** 63 + 25, 2 ** 63 + 20, 2 ** 63 + 15, 2 ** 63 + 10, 2 ** 63],
        ],
        ids=["index_inc", "index_dec"],
    )
    def index(self, request):
        return self._index_cls(request.param)

    def test_constructor(self, dtype):
        index_cls = self._index_cls

        idx = index_cls([1, 2, 3])
        res = Index([1, 2, 3], dtype=dtype)
        tm.assert_index_equal(res, idx)

        idx = index_cls([1, 2 ** 63])
        res = Index([1, 2 ** 63], dtype=dtype)
        tm.assert_index_equal(res, idx)

        idx = index_cls([1, 2 ** 63])
        res = Index([1, 2 ** 63])
        tm.assert_index_equal(res, idx)

        idx = Index([-1, 2 ** 63], dtype=object)
        res = Index(np.array([-1, 2 ** 63], dtype=object))
        tm.assert_index_equal(res, idx)

        # https://github.com/pandas-dev/pandas/issues/29526
        idx = index_cls([1, 2 ** 63 + 1], dtype=dtype)
        res = Index([1, 2 ** 63 + 1], dtype=dtype)
        tm.assert_index_equal(res, idx)

    @pytest.mark.xfail(
        not is_platform_arm(),
        reason="https://github.com/numpy/numpy/issues/19146",
    )
    def test_constructor_does_not_cast_to_float(self):
        # https://github.com/numpy/numpy/issues/19146
        values = [0, np.iinfo(np.uint64).max]

        result = UInt64Index(values)
        assert list(result) == values


@pytest.mark.parametrize(
    "box",
    [list, lambda x: np.array(x, dtype=object), lambda x: Index(x, dtype=object)],
)
def test_uint_index_does_not_convert_to_float64(box):
    # https://github.com/pandas-dev/pandas/issues/28279
    # https://github.com/pandas-dev/pandas/issues/28023
    series = Series(
        [0, 1, 2, 3, 4, 5],
        index=[
            7606741985629028552,
            17876870360202815256,
            17876870360202815256,
            13106359306506049338,
            8991270399732411471,
            8991270399732411472,
        ],
    )

    result = series.loc[box([7606741985629028552, 17876870360202815256])]

    expected = UInt64Index(
        [7606741985629028552, 17876870360202815256, 17876870360202815256],
        dtype="uint64",
    )
    tm.assert_index_equal(result.index, expected)

    tm.assert_equal(result, series[:3])


def test_float64_index_equals():
    # https://github.com/pandas-dev/pandas/issues/35217
    float_index = Index([1.0, 2, 3])
    string_index = Index(["1", "2", "3"])

    result = float_index.equals(string_index)
    assert result is False

    result = string_index.equals(float_index)
    assert result is False
