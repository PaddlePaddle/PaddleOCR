import numpy as np
import pytest

from pandas.core.dtypes.common import ensure_platform_int

import pandas as pd
from pandas import (
    Float64Index,
    Index,
    Int64Index,
    RangeIndex,
)
import pandas._testing as tm
from pandas.tests.indexes.common import NumericBase

# aliases to make some tests easier to read
RI = RangeIndex
I64 = Int64Index
F64 = Float64Index
OI = Index


class TestRangeIndex(NumericBase):
    _index_cls = RangeIndex

    @pytest.fixture
    def dtype(self):
        return np.int64

    @pytest.fixture(
        params=["uint64", "float64", "category", "datetime64", "object"],
    )
    def invalid_dtype(self, request):
        return request.param

    @pytest.fixture
    def simple_index(self) -> Index:
        return self._index_cls(start=0, stop=20, step=2)

    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2, name="foo"),
            RangeIndex(start=18, stop=-1, step=-2, name="bar"),
        ],
        ids=["index_inc", "index_dec"],
    )
    def index(self, request):
        return request.param

    def test_constructor_unwraps_index(self, dtype):
        result = self._index_cls(1, 3)
        expected = np.array([1, 2], dtype=dtype)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_too_many_names(self, simple_index):
        index = simple_index
        with pytest.raises(ValueError, match="^Length"):
            index.names = ["roger", "harold"]

    @pytest.mark.parametrize(
        "index, start, stop, step",
        [
            (RangeIndex(5), 0, 5, 1),
            (RangeIndex(0, 5), 0, 5, 1),
            (RangeIndex(5, step=2), 0, 5, 2),
            (RangeIndex(1, 5, 2), 1, 5, 2),
        ],
    )
    def test_start_stop_step_attrs(self, index, start, stop, step):
        # GH 25710
        assert index.start == start
        assert index.stop == stop
        assert index.step == step

    @pytest.mark.parametrize("attr_name", ["_start", "_stop", "_step"])
    def test_deprecated_start_stop_step_attrs(self, attr_name, simple_index):
        # GH 26581
        idx = simple_index
        with tm.assert_produces_warning(FutureWarning):
            getattr(idx, attr_name)

    def test_copy(self):
        i = RangeIndex(5, name="Foo")
        i_copy = i.copy()
        assert i_copy is not i
        assert i_copy.identical(i)
        assert i_copy._range == range(0, 5, 1)
        assert i_copy.name == "Foo"

    def test_repr(self):
        i = RangeIndex(5, name="Foo")
        result = repr(i)
        expected = "RangeIndex(start=0, stop=5, step=1, name='Foo')"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

        i = RangeIndex(5, 0, -1)
        result = repr(i)
        expected = "RangeIndex(start=5, stop=0, step=-1)"
        assert result == expected

        result = eval(result)
        tm.assert_index_equal(result, i, exact=True)

    def test_insert(self):

        idx = RangeIndex(5, name="Foo")
        result = idx[1:4]

        # test 0th element
        tm.assert_index_equal(idx[0:4], result.insert(0, idx[0]))

        # GH 18295 (test missing)
        expected = Float64Index([0, np.nan, 1, 2, 3, 4])
        for na in [np.nan, None, pd.NA]:
            result = RangeIndex(5).insert(1, na)
            tm.assert_index_equal(result, expected)

        result = RangeIndex(5).insert(1, pd.NaT)
        expected = Index([0, pd.NaT, 1, 2, 3, 4], dtype=object)
        tm.assert_index_equal(result, expected)

    def test_delete(self):

        idx = RangeIndex(5, name="Foo")
        expected = idx[1:].astype(int)
        result = idx.delete(0)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        expected = idx[:-1].astype(int)
        result = idx.delete(-1)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name

        msg = "index 5 is out of bounds for axis 0 with size 5"
        with pytest.raises((IndexError, ValueError), match=msg):
            # either depending on numpy version
            result = idx.delete(len(idx))

    def test_view(self):
        i = RangeIndex(0, name="Foo")
        i_view = i.view()
        assert i_view.name == "Foo"

        i_view = i.view("i8")
        tm.assert_numpy_array_equal(i.values, i_view)

        i_view = i.view(RangeIndex)
        tm.assert_index_equal(i, i_view)

    def test_dtype(self, simple_index):
        index = simple_index
        assert index.dtype == np.int64

    def test_cache(self):
        # GH 26565, GH26617, GH35432
        # This test checks whether _cache has been set.
        # Calling RangeIndex._cache["_data"] creates an int64 array of the same length
        # as the RangeIndex and stores it in _cache.
        idx = RangeIndex(0, 100, 10)

        assert idx._cache == {}

        repr(idx)
        assert idx._cache == {}

        str(idx)
        assert idx._cache == {}

        idx.get_loc(20)
        assert idx._cache == {}

        90 in idx  # True
        assert idx._cache == {}

        91 in idx  # False
        assert idx._cache == {}

        idx.all()
        assert idx._cache == {}

        idx.any()
        assert idx._cache == {}

        for _ in idx:
            pass
        assert idx._cache == {}

        idx.format()
        assert idx._cache == {}

        df = pd.DataFrame({"a": range(10)}, index=idx)

        str(df)
        assert idx._cache == {}

        df.loc[50]
        assert idx._cache == {}

        with pytest.raises(KeyError, match="51"):
            df.loc[51]
        assert idx._cache == {}

        df.loc[10:50]
        assert idx._cache == {}

        df.iloc[5:10]
        assert idx._cache == {}

        # idx._cache should contain a _data entry after call to idx._data
        idx._data
        assert isinstance(idx._data, np.ndarray)
        assert idx._data is idx._data  # check cached value is reused
        assert len(idx._cache) == 1
        expected = np.arange(0, 100, 10, dtype="int64")
        tm.assert_numpy_array_equal(idx._cache["_data"], expected)

    def test_is_monotonic(self):
        index = RangeIndex(0, 20, 2)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is False

        index = RangeIndex(4, 0, -1)
        assert index.is_monotonic is False
        assert index._is_strictly_monotonic_increasing is False
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 2)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(2, 1)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

        index = RangeIndex(1, 1)
        assert index.is_monotonic is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    def test_equals_range(self):
        equiv_pairs = [
            (RangeIndex(0, 9, 2), RangeIndex(0, 10, 2)),
            (RangeIndex(0), RangeIndex(1, -1, 3)),
            (RangeIndex(1, 2, 3), RangeIndex(1, 3, 4)),
            (RangeIndex(0, -9, -2), RangeIndex(0, -10, -2)),
        ]
        for left, right in equiv_pairs:
            assert left.equals(right)
            assert right.equals(left)

    def test_logical_compat(self, simple_index):
        idx = simple_index
        assert idx.all() == idx.values.all()
        assert idx.any() == idx.values.any()

    def test_identical(self, simple_index):
        index = simple_index
        i = Index(index.copy())
        assert i.identical(index)

        # we don't allow object dtype for RangeIndex
        if isinstance(index, RangeIndex):
            return

        same_values_different_type = Index(i, dtype=object)
        assert not i.identical(same_values_different_type)

        i = index.copy(dtype=object)
        i = i.rename("foo")
        same_values = Index(i, dtype=object)
        assert same_values.identical(index.copy(dtype=object))

        assert not i.identical(index)
        assert Index(same_values, name="foo", dtype=object).identical(i)

        assert not index.copy(dtype=object).identical(index.copy(dtype="int64"))

    def test_nbytes(self):

        # memory savings vs int index
        i = RangeIndex(0, 1000)
        assert i.nbytes < i._int64index.nbytes / 10

        # constant memory usage
        i2 = RangeIndex(0, 10)
        assert i.nbytes == i2.nbytes

    @pytest.mark.parametrize(
        "start,stop,step",
        [
            # can't
            ("foo", "bar", "baz"),
            # shouldn't
            ("0", "1", "2"),
        ],
    )
    def test_cant_or_shouldnt_cast(self, start, stop, step):
        msg = f"Wrong type {type(start)} for value {start}"
        with pytest.raises(TypeError, match=msg):
            RangeIndex(start, stop, step)

    def test_view_index(self, simple_index):
        index = simple_index
        index.view(Index)

    def test_prevent_casting(self, simple_index):
        index = simple_index
        result = index.astype("O")
        assert result.dtype == np.object_

    def test_repr_roundtrip(self, simple_index):
        index = simple_index
        tm.assert_index_equal(eval(repr(index)), index)

    def test_slice_keep_name(self):
        idx = RangeIndex(1, 2, name="asdf")
        assert idx.name == idx[1:].name

    def test_has_duplicates(self, index):
        assert index.is_unique
        assert not index.has_duplicates

    def test_extended_gcd(self, simple_index):
        index = simple_index
        result = index._extended_gcd(6, 10)
        assert result[0] == result[1] * 6 + result[2] * 10
        assert 2 == result[0]

        result = index._extended_gcd(10, 6)
        assert 2 == result[1] * 10 + result[2] * 6
        assert 2 == result[0]

    def test_min_fitting_element(self):
        result = RangeIndex(0, 20, 2)._min_fitting_element(1)
        assert 2 == result

        result = RangeIndex(1, 6)._min_fitting_element(1)
        assert 1 == result

        result = RangeIndex(18, -2, -2)._min_fitting_element(1)
        assert 2 == result

        result = RangeIndex(5, 0, -1)._min_fitting_element(1)
        assert 1 == result

        big_num = 500000000000000000000000

        result = RangeIndex(5, big_num * 2, 1)._min_fitting_element(big_num)
        assert big_num == result

    def test_pickle_compat_construction(self):
        # RangeIndex() is a valid constructor
        pass

    def test_slice_specialised(self, simple_index):
        index = simple_index
        index.name = "foo"

        # scalar indexing
        res = index[1]
        expected = 2
        assert res == expected

        res = index[-1]
        expected = 18
        assert res == expected

        # slicing
        # slice value completion
        index_slice = index[:]
        expected = index
        tm.assert_index_equal(index_slice, expected)

        # positive slice values
        index_slice = index[7:10:2]
        expected = Index(np.array([14, 18]), name="foo")
        tm.assert_index_equal(index_slice, expected)

        # negative slice values
        index_slice = index[-1:-5:-2]
        expected = Index(np.array([18, 14]), name="foo")
        tm.assert_index_equal(index_slice, expected)

        # stop overshoot
        index_slice = index[2:100:4]
        expected = Index(np.array([4, 12]), name="foo")
        tm.assert_index_equal(index_slice, expected)

        # reverse
        index_slice = index[::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected)

        index_slice = index[-8::-1]
        expected = Index(np.array([4, 2, 0]), name="foo")
        tm.assert_index_equal(index_slice, expected)

        index_slice = index[-40::-1]
        expected = Index(np.array([], dtype=np.int64), name="foo")
        tm.assert_index_equal(index_slice, expected)

        index_slice = index[40::-1]
        expected = Index(index.values[40::-1], name="foo")
        tm.assert_index_equal(index_slice, expected)

        index_slice = index[10::-1]
        expected = Index(index.values[::-1], name="foo")
        tm.assert_index_equal(index_slice, expected)

    @pytest.mark.parametrize("step", set(range(-5, 6)) - {0})
    def test_len_specialised(self, step):
        # make sure that our len is the same as np.arange calc
        start, stop = (0, 5) if step > 0 else (5, 0)

        arr = np.arange(start, stop, step)
        index = RangeIndex(start, stop, step)
        assert len(index) == len(arr)

        index = RangeIndex(stop, start, step)
        assert len(index) == 0

    @pytest.fixture(
        params=[
            ([RI(1, 12, 5)], RI(1, 12, 5)),
            ([RI(0, 6, 4)], RI(0, 6, 4)),
            ([RI(1, 3), RI(3, 7)], RI(1, 7)),
            ([RI(1, 5, 2), RI(5, 6)], RI(1, 6, 2)),
            ([RI(1, 3, 2), RI(4, 7, 3)], RI(1, 7, 3)),
            ([RI(-4, 3, 2), RI(4, 7, 2)], RI(-4, 7, 2)),
            ([RI(-4, -8), RI(-8, -12)], RI(0, 0)),
            ([RI(-4, -8), RI(3, -4)], RI(0, 0)),
            ([RI(-4, -8), RI(3, 5)], RI(3, 5)),
            ([RI(-4, -2), RI(3, 5)], I64([-4, -3, 3, 4])),
            ([RI(-2), RI(3, 5)], RI(3, 5)),
            ([RI(2), RI(2)], I64([0, 1, 0, 1])),
            ([RI(2), RI(2, 5), RI(5, 8, 4)], RI(0, 6)),
            ([RI(2), RI(3, 5), RI(5, 8, 4)], I64([0, 1, 3, 4, 5])),
            ([RI(-2, 2), RI(2, 5), RI(5, 8, 4)], RI(-2, 6)),
            ([RI(3), I64([-1, 3, 15])], I64([0, 1, 2, -1, 3, 15])),
            ([RI(3), F64([-1, 3.1, 15.0])], F64([0, 1, 2, -1, 3.1, 15.0])),
            ([RI(3), OI(["a", None, 14])], OI([0, 1, 2, "a", None, 14])),
            ([RI(3, 1), OI(["a", None, 14])], OI(["a", None, 14])),
        ]
    )
    def appends(self, request):
        """Inputs and expected outputs for RangeIndex.append test"""
        return request.param

    def test_append(self, appends):
        # GH16212

        indices, expected = appends

        result = indices[0].append(indices[1:])
        tm.assert_index_equal(result, expected, exact=True)

        if len(indices) == 2:
            # Append single item rather than list
            result2 = indices[0].append(indices[1])
            tm.assert_index_equal(result2, expected, exact=True)

    def test_engineless_lookup(self):
        # GH 16685
        # Standard lookup on RangeIndex should not require the engine to be
        # created
        idx = RangeIndex(2, 10, 3)

        assert idx.get_loc(5) == 1
        tm.assert_numpy_array_equal(
            idx.get_indexer([2, 8]), ensure_platform_int(np.array([0, 2]))
        )
        with pytest.raises(KeyError, match="3"):
            idx.get_loc(3)

        assert "_engine" not in idx._cache

        # Different types of scalars can be excluded immediately, no need to
        #  use the _engine
        with pytest.raises(KeyError, match="'a'"):
            idx.get_loc("a")

        assert "_engine" not in idx._cache

    def test_format_empty(self):
        # GH35712
        empty_idx = self._index_cls(0)
        assert empty_idx.format() == []
        assert empty_idx.format(name=True) == [""]

    @pytest.mark.parametrize(
        "RI",
        [
            RangeIndex(0, -1, -1),
            RangeIndex(0, 1, 1),
            RangeIndex(1, 3, 2),
            RangeIndex(0, -1, -2),
            RangeIndex(-3, -5, -2),
        ],
    )
    def test_append_len_one(self, RI):
        # GH39401
        result = RI.append([])
        tm.assert_index_equal(result, RI, exact=True)

    @pytest.mark.parametrize("base", [RangeIndex(0, 2), Index([0, 1])])
    def test_isin_range(self, base):
        # GH#41151
        values = RangeIndex(0, 1)
        result = base.isin(values)
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
