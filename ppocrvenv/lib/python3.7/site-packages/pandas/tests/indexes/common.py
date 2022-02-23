from __future__ import annotations

from datetime import datetime
import gc

import numpy as np
import pytest

from pandas._libs import iNaT
from pandas._libs.tslibs import Timestamp

from pandas.core.dtypes.common import is_datetime64tz_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    Float64Index,
    Index,
    Int64Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    UInt64Index,
    isna,
)
import pandas._testing as tm
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin


class Base:
    """
    Base class for index sub-class tests.
    """

    _index_cls: type[Index]

    @pytest.fixture
    def simple_index(self):
        raise NotImplementedError("Method not implemented")

    def create_index(self) -> Index:
        raise NotImplementedError("Method not implemented")

    def test_pickle_compat_construction(self):
        # need an object to create with
        msg = "|".join(
            [
                r"Index\(\.\.\.\) must be called with a collection of some "
                r"kind, None was passed",
                r"DatetimeIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"TimedeltaIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"__new__\(\) missing 1 required positional argument: 'data'",
                r"__new__\(\) takes at least 2 arguments \(1 given\)",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            self._index_cls()

    @pytest.mark.parametrize("name", [None, "new_name"])
    def test_to_frame(self, name, simple_index):
        # see GH-15230, GH-22580
        idx = simple_index

        if name:
            idx_name = name
        else:
            idx_name = idx.name or 0

        df = idx.to_frame(name=idx_name)

        assert df.index is idx
        assert len(df.columns) == 1
        assert df.columns[0] == idx_name
        assert df[idx_name].values is not idx.values

        df = idx.to_frame(index=False, name=idx_name)
        assert df.index is not idx

    def test_shift(self, simple_index):

        # GH8083 test the base class for shift
        idx = simple_index
        msg = (
            f"This method is only implemented for DatetimeIndex, PeriodIndex and "
            f"TimedeltaIndex; Got type {type(idx).__name__}"
        )
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1)
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1, 2)

    def test_constructor_name_unhashable(self, simple_index):
        # GH#29069 check that name is hashable
        # See also same-named test in tests.series.test_constructors
        idx = simple_index
        with pytest.raises(TypeError, match="Index.name must be a hashable type"):
            type(idx)(idx, name=[])

    def test_create_index_existing_name(self, simple_index):

        # GH11193, when an existing index is passed, and a new name is not
        # specified, the new index should inherit the previous object name
        expected = simple_index
        if not isinstance(expected, MultiIndex):
            expected.name = "foo"
            result = Index(expected)
            tm.assert_index_equal(result, expected)

            result = Index(expected, name="bar")
            expected.name = "bar"
            tm.assert_index_equal(result, expected)
        else:
            expected.names = ["foo", "bar"]
            result = Index(expected)
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["foo", "bar"],
                ),
            )

            result = Index(expected, names=["A", "B"])
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["A", "B"],
                ),
            )

    def test_numeric_compat(self, simple_index):

        idx = simple_index
        # Check that this doesn't cover MultiIndex case, if/when it does,
        #  we can remove multi.test_compat.test_numeric_compat
        assert not isinstance(idx, MultiIndex)
        if type(idx) is Index:
            return

        typ = type(idx._data).__name__
        lmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typ}' and 'int'",
                "cannot perform (__mul__|__truediv__|__floordiv__) with "
                f"this index type: {typ}",
            ]
        )
        with pytest.raises(TypeError, match=lmsg):
            idx * 1
        rmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: 'int' and '{typ}'",
                "cannot perform (__rmul__|__rtruediv__|__rfloordiv__) with "
                f"this index type: {typ}",
            ]
        )
        with pytest.raises(TypeError, match=rmsg):
            1 * idx

        div_err = lmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            idx / 1
        div_err = rmsg.replace("*", "/")
        with pytest.raises(TypeError, match=div_err):
            1 / idx

        floordiv_err = lmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            idx // 1
        floordiv_err = rmsg.replace("*", "//")
        with pytest.raises(TypeError, match=floordiv_err):
            1 // idx

    def test_logical_compat(self, simple_index):
        idx = simple_index
        with pytest.raises(TypeError, match="cannot perform all"):
            idx.all()
        with pytest.raises(TypeError, match="cannot perform any"):
            idx.any()

    def test_repr_roundtrip(self, simple_index):

        idx = simple_index
        tm.assert_index_equal(eval(repr(idx)), idx)

    def test_repr_max_seq_item_setting(self, simple_index):
        # GH10182
        idx = simple_index
        idx = idx.repeat(50)
        with pd.option_context("display.max_seq_items", None):
            repr(idx)
            assert "..." not in str(idx)

    def test_copy_name(self, index):
        # gh-12309: Check that the "name" argument
        # passed at initialization is honored.
        if isinstance(index, MultiIndex):
            return

        first = type(index)(index, copy=True, name="mario")
        second = type(first)(first, copy=False)

        # Even though "copy=False", we want a new object.
        assert first is not second

        # Not using tm.assert_index_equal() since names differ.
        assert index.equals(first)

        assert first.name == "mario"
        assert second.name == "mario"

        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])

        if not isinstance(index, CategoricalIndex):
            # See gh-13365
            s3 = s1 * s2
            assert s3.index.name == "mario"

    def test_copy_name2(self, index):
        # gh-35592
        if isinstance(index, MultiIndex):
            return

        assert index.copy(name="mario").name == "mario"

        with pytest.raises(ValueError, match="Length of new names must be 1, got 2"):
            index.copy(name=["mario", "luigi"])

        msg = f"{type(index).__name__}.name must be a hashable type"
        with pytest.raises(TypeError, match=msg):
            index.copy(name=[["mario"]])

    def test_ensure_copied_data(self, index):
        # Check the "copy" argument of each Index.__new__ is honoured
        # GH12309
        init_kwargs = {}
        if isinstance(index, PeriodIndex):
            # Needs "freq" specification:
            init_kwargs["freq"] = index.freq
        elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
            # RangeIndex cannot be initialized from data
            # MultiIndex and CategoricalIndex are tested separately
            return

        index_type = type(index)
        result = index_type(index.values, copy=True, **init_kwargs)
        if is_datetime64tz_dtype(index.dtype):
            result = result.tz_localize("UTC").tz_convert(index.tz)
        if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
            index = index._with_freq(None)

        tm.assert_index_equal(index, result)

        if isinstance(index, PeriodIndex):
            # .values an object array of Period, thus copied
            result = index_type(ordinal=index.asi8, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same="same")
        elif isinstance(index, IntervalIndex):
            # checked in test_interval.py
            pass
        else:
            result = index_type(index.values, copy=False, **init_kwargs)
            tm.assert_numpy_array_equal(index.values, result.values, check_same="same")

    def test_memory_usage(self, index):
        index._engine.clear_mapping()
        result = index.memory_usage()
        if index.empty:
            # we report 0 for no-length
            assert result == 0
            return

        # non-zero length
        index.get_loc(index[0])
        result2 = index.memory_usage()
        result3 = index.memory_usage(deep=True)

        # RangeIndex, IntervalIndex
        # don't have engines
        if not isinstance(index, (RangeIndex, IntervalIndex)):
            assert result2 > result

        if index.inferred_type == "object":
            assert result3 > result2

    def test_argsort(self, request, index):
        # separately tested
        if isinstance(index, CategoricalIndex):
            return

        result = index.argsort()
        expected = np.array(index).argsort()
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    def test_numpy_argsort(self, index):
        result = np.argsort(index)
        expected = index.argsort()
        tm.assert_numpy_array_equal(result, expected)

        # these are the only two types that perform
        # pandas compatibility input validation - the
        # rest already perform separate (or no) such
        # validation via their 'values' attribute as
        # defined in pandas.core.indexes/base.py - they
        # cannot be changed at the moment due to
        # backwards compatibility concerns
        if isinstance(type(index), (CategoricalIndex, RangeIndex)):
            # TODO: why type(index)?
            msg = "the 'axis' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, axis=1)

            msg = "the 'kind' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, kind="mergesort")

            msg = "the 'order' parameter is not supported"
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, order=("a", "b"))

    def test_repeat(self, simple_index):
        rep = 2
        idx = simple_index.copy()
        expected = Index(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

        idx = simple_index
        rep = np.arange(len(idx))
        expected = Index(idx.values.repeat(rep), name=idx.name)
        tm.assert_index_equal(idx.repeat(rep), expected)

    def test_numpy_repeat(self, simple_index):
        rep = 2
        idx = simple_index
        expected = idx.repeat(rep)
        tm.assert_index_equal(np.repeat(idx, rep), expected)

        msg = "the 'axis' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.repeat(idx, rep, axis=0)

    @pytest.mark.parametrize("klass", [list, tuple, np.array, Series])
    def test_where(self, klass, simple_index):
        idx = simple_index
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex)):
            # where does not preserve freq
            idx = idx._with_freq(None)

        cond = [True] * len(idx)
        result = idx.where(klass(cond))
        expected = idx
        tm.assert_index_equal(result, expected)

        cond = [False] + [True] * len(idx[1:])
        expected = Index([idx._na_value] + idx[1:].tolist(), dtype=idx.dtype)
        result = idx.where(klass(cond))
        tm.assert_index_equal(result, expected)

    def test_insert_base(self, index):
        result = index[1:4]

        if not len(index):
            return

        # test 0th element
        assert index[0:4].equals(result.insert(0, index[0]))

    def test_delete_base(self, index):
        if not len(index):
            return

        if isinstance(index, RangeIndex):
            # tested in class
            return

        expected = index[1:]
        result = index.delete(0)
        assert result.equals(expected)
        assert result.name == expected.name

        expected = index[:-1]
        result = index.delete(-1)
        assert result.equals(expected)
        assert result.name == expected.name

        length = len(index)
        msg = f"index {length} is out of bounds for axis 0 with size {length}"
        with pytest.raises(IndexError, match=msg):
            index.delete(length)

    def test_equals(self, index):
        if isinstance(index, IntervalIndex):
            # IntervalIndex tested separately, the index.equals(index.astype(object))
            #  fails for IntervalIndex
            return

        assert index.equals(index)
        assert index.equals(index.copy())
        assert index.equals(index.astype(object))

        assert not index.equals(list(index))
        assert not index.equals(np.array(index))

        # Cannot pass in non-int64 dtype to RangeIndex
        if not isinstance(index, RangeIndex):
            same_values = Index(index, dtype=object)
            assert index.equals(same_values)
            assert same_values.equals(index)

        if index.nlevels == 1:
            # do not test MultiIndex
            assert not index.equals(Series(index))

    def test_equals_op(self, simple_index):
        # GH9947, GH10637
        index_a = simple_index

        n = len(index_a)
        index_b = index_a[0:-1]
        index_c = index_a[0:-1].append(index_a[-2:-1])
        index_d = index_a[0:1]

        msg = "Lengths must match|could not be broadcast"
        with pytest.raises(ValueError, match=msg):
            index_a == index_b
        expected1 = np.array([True] * n)
        expected2 = np.array([True] * (n - 1) + [False])
        tm.assert_numpy_array_equal(index_a == index_a, expected1)
        tm.assert_numpy_array_equal(index_a == index_c, expected2)

        # test comparisons with numpy arrays
        array_a = np.array(index_a)
        array_b = np.array(index_a[0:-1])
        array_c = np.array(index_a[0:-1].append(index_a[-2:-1]))
        array_d = np.array(index_a[0:1])
        with pytest.raises(ValueError, match=msg):
            index_a == array_b
        tm.assert_numpy_array_equal(index_a == array_a, expected1)
        tm.assert_numpy_array_equal(index_a == array_c, expected2)

        # test comparisons with Series
        series_a = Series(array_a)
        series_b = Series(array_b)
        series_c = Series(array_c)
        series_d = Series(array_d)
        with pytest.raises(ValueError, match=msg):
            index_a == series_b

        tm.assert_numpy_array_equal(index_a == series_a, expected1)
        tm.assert_numpy_array_equal(index_a == series_c, expected2)

        # cases where length is 1 for one of them
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == index_d
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == series_d
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == array_d
        msg = "Can only compare identically-labeled Series objects"
        with pytest.raises(ValueError, match=msg):
            series_a == series_d
        with pytest.raises(ValueError, match="Lengths must match"):
            series_a == array_d

        # comparing with a scalar should broadcast; note that we are excluding
        # MultiIndex because in this case each item in the index is a tuple of
        # length 2, and therefore is considered an array of length 2 in the
        # comparison instead of a scalar
        if not isinstance(index_a, MultiIndex):
            expected3 = np.array([False] * (len(index_a) - 2) + [True, False])
            # assuming the 2nd to last item is unique in the data
            item = index_a[-2]
            tm.assert_numpy_array_equal(index_a == item, expected3)
            # For RangeIndex we can convert to Int64Index
            tm.assert_series_equal(series_a == item, Series(expected3))

    def test_format(self, simple_index):
        # GH35439
        idx = simple_index
        expected = [str(x) for x in idx]
        assert idx.format() == expected

    def test_format_empty(self):
        # GH35712
        empty_idx = self._index_cls([])
        assert empty_idx.format() == []
        assert empty_idx.format(name=True) == [""]

    def test_hasnans_isnans(self, index_flat):
        # GH 11343, added tests for hasnans / isnans
        index = index_flat

        # cases in indices doesn't include NaN
        idx = index.copy(deep=True)
        expected = np.array([False] * len(idx), dtype=bool)
        tm.assert_numpy_array_equal(idx._isnan, expected)
        assert idx.hasnans is False

        idx = index.copy(deep=True)
        values = np.asarray(idx.values)

        if len(index) == 0:
            return
        elif isinstance(index, DatetimeIndexOpsMixin):
            values[1] = iNaT
        elif isinstance(index, (Int64Index, UInt64Index, RangeIndex)):
            return
        else:
            values[1] = np.nan

        if isinstance(index, PeriodIndex):
            idx = type(index)(values, freq=index.freq)
        else:
            idx = type(index)(values)

            expected = np.array([False] * len(idx), dtype=bool)
            expected[1] = True
            tm.assert_numpy_array_equal(idx._isnan, expected)
            assert idx.hasnans is True

    def test_fillna(self, index):
        # GH 11343
        if len(index) == 0:
            pass
        elif isinstance(index, MultiIndex):
            idx = index.copy(deep=True)
            msg = "isna is not defined for MultiIndex"
            with pytest.raises(NotImplementedError, match=msg):
                idx.fillna(idx[0])
        else:
            idx = index.copy(deep=True)
            result = idx.fillna(idx[0])
            tm.assert_index_equal(result, idx)
            assert result is not idx

            msg = "'value' must be a scalar, passed: "
            with pytest.raises(TypeError, match=msg):
                idx.fillna([idx[0]])

            idx = index.copy(deep=True)
            values = np.asarray(idx.values)

            if isinstance(index, DatetimeIndexOpsMixin):
                values[1] = iNaT
            elif isinstance(index, (Int64Index, UInt64Index, RangeIndex)):
                return
            else:
                values[1] = np.nan

            if isinstance(index, PeriodIndex):
                idx = type(index)(values, freq=index.freq)
            else:
                idx = type(index)(values)

            expected = np.array([False] * len(idx), dtype=bool)
            expected[1] = True
            tm.assert_numpy_array_equal(idx._isnan, expected)
            assert idx.hasnans is True

    def test_nulls(self, index):
        # this is really a smoke test for the methods
        # as these are adequately tested for function elsewhere
        if len(index) == 0:
            tm.assert_numpy_array_equal(index.isna(), np.array([], dtype=bool))
        elif isinstance(index, MultiIndex):
            idx = index.copy()
            msg = "isna is not defined for MultiIndex"
            with pytest.raises(NotImplementedError, match=msg):
                idx.isna()
        elif not index.hasnans:
            tm.assert_numpy_array_equal(index.isna(), np.zeros(len(index), dtype=bool))
            tm.assert_numpy_array_equal(index.notna(), np.ones(len(index), dtype=bool))
        else:
            result = isna(index)
            tm.assert_numpy_array_equal(index.isna(), result)
            tm.assert_numpy_array_equal(index.notna(), ~result)

    def test_empty(self, simple_index):
        # GH 15270
        idx = simple_index
        assert not idx.empty
        assert idx[:0].empty

    def test_join_self_unique(self, join_type, simple_index):
        idx = simple_index
        if idx.is_unique:
            joined = idx.join(idx, how=join_type)
            assert (idx == joined).all()

    def test_map(self, simple_index):
        # callable
        idx = simple_index

        # we don't infer UInt64
        if isinstance(idx, UInt64Index):
            expected = idx.astype("int64")
        else:
            expected = idx

        result = idx.map(lambda x: x)
        # For RangeIndex we convert to Int64Index
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    def test_map_dictlike(self, mapper, simple_index):

        idx = simple_index
        if isinstance(idx, CategoricalIndex):
            pytest.skip(f"skipping tests for {type(idx)}")

        identity = mapper(idx.values, idx)

        # we don't infer to UInt64 for a dict
        if isinstance(idx, UInt64Index) and isinstance(identity, dict):
            expected = idx.astype("int64")
        else:
            expected = idx

        result = idx.map(identity)
        # For RangeIndex we convert to Int64Index
        tm.assert_index_equal(result, expected)

        # empty mappable
        expected = Index([np.nan] * len(idx))
        result = idx.map(mapper(expected, idx))
        tm.assert_index_equal(result, expected)

    def test_map_str(self, simple_index):
        # GH 31202
        idx = simple_index
        result = idx.map(str)
        expected = Index([str(x) for x in idx], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("ordered", [True, False])
    def test_astype_category(self, copy, name, ordered, simple_index):
        # GH 18630
        idx = simple_index
        if name:
            idx = idx.rename(name)

        # standard categories
        dtype = CategoricalDtype(ordered=ordered)
        result = idx.astype(dtype, copy=copy)
        expected = CategoricalIndex(idx, name=name, ordered=ordered)
        tm.assert_index_equal(result, expected, exact=True)

        # non-standard categories
        dtype = CategoricalDtype(idx.unique().tolist()[:-1], ordered)
        result = idx.astype(dtype, copy=copy)
        expected = CategoricalIndex(idx, name=name, dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        if ordered is False:
            # dtype='category' defaults to ordered=False, so only test once
            result = idx.astype("category", copy=copy)
            expected = CategoricalIndex(idx, name=name)
            tm.assert_index_equal(result, expected, exact=True)

    def test_is_unique(self, simple_index):
        # initialize a unique index
        index = simple_index.drop_duplicates()
        assert index.is_unique is True

        # empty index should be unique
        index_empty = index[:0]
        assert index_empty.is_unique is True

        # test basic dupes
        index_dup = index.insert(0, index[0])
        assert index_dup.is_unique is False

        # single NA should be unique
        index_na = index.insert(0, np.nan)
        assert index_na.is_unique is True

        # multiple NA should not be unique
        index_na_dup = index_na.insert(0, np.nan)
        assert index_na_dup.is_unique is False

    @pytest.mark.arm_slow
    def test_engine_reference_cycle(self, simple_index):
        # GH27585
        index = simple_index
        nrefs_pre = len(gc.get_referrers(index))
        index._engine
        assert len(gc.get_referrers(index)) == nrefs_pre

    def test_getitem_2d_deprecated(self, simple_index):
        # GH#30588
        idx = simple_index
        msg = "Support for multi-dimensional indexing"
        check = not isinstance(idx, (RangeIndex, CategoricalIndex))
        with tm.assert_produces_warning(
            FutureWarning, match=msg, check_stacklevel=check
        ):
            res = idx[:, None]

        assert isinstance(res, np.ndarray), type(res)

    def test_copy_shares_cache(self, simple_index):
        # GH32898, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # populates the _cache.
        copy = idx.copy()

        assert copy._cache is idx._cache

    def test_shallow_copy_shares_cache(self, simple_index):
        # GH32669, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # populates the _cache.
        shallow_copy = idx._view()

        assert shallow_copy._cache is idx._cache

        shallow_copy = idx._shallow_copy(idx._data)
        assert shallow_copy._cache is not idx._cache
        assert shallow_copy._cache == {}

    def test_index_groupby(self, simple_index):
        idx = simple_index[:5]
        to_groupby = np.array([1, 2, np.nan, 2, 1])
        tm.assert_dict_equal(
            idx.groupby(to_groupby), {1.0: idx[[0, 4]], 2.0: idx[[1, 3]]}
        )

        to_groupby = DatetimeIndex(
            [
                datetime(2011, 11, 1),
                datetime(2011, 12, 1),
                pd.NaT,
                datetime(2011, 12, 1),
                datetime(2011, 11, 1),
            ],
            tz="UTC",
        ).values

        ex_keys = [Timestamp("2011-11-01"), Timestamp("2011-12-01")]
        expected = {ex_keys[0]: idx[[0, 4]], ex_keys[1]: idx[[1, 3]]}
        tm.assert_dict_equal(idx.groupby(to_groupby), expected)


class NumericBase(Base):
    """
    Base class for numeric index (incl. RangeIndex) sub-class tests.
    """

    def test_constructor_unwraps_index(self, dtype):
        idx = Index([1, 2], dtype=dtype)
        result = self._index_cls(idx)
        expected = np.array([1, 2], dtype=dtype)
        tm.assert_numpy_array_equal(result._data, expected)

    def test_where(self):
        # Tested in numeric.test_indexing
        pass

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False

    def test_format(self, simple_index):
        # GH35439
        idx = simple_index
        max_width = max(len(str(x)) for x in idx)
        expected = [str(x).ljust(max_width) for x in idx]
        assert idx.format() == expected

    def test_numeric_compat(self):
        pass  # override Base method

    def test_insert_na(self, nulls_fixture, simple_index):
        # GH 18295 (test missing)
        index = simple_index
        na_val = nulls_fixture

        if na_val is pd.NaT:
            expected = Index([index[0], pd.NaT] + list(index[1:]), dtype=object)
        else:
            expected = Float64Index([index[0], np.nan] + list(index[1:]))

        result = index.insert(1, na_val)
        tm.assert_index_equal(result, expected)

    def test_arithmetic_explicit_conversions(self):
        # GH 8608
        # add/sub are overridden explicitly for Float/Int Index
        index_cls = self._index_cls
        if index_cls is RangeIndex:
            idx = RangeIndex(5)
        else:
            idx = index_cls(np.arange(5, dtype="int64"))

        # float conversions
        arr = np.arange(5, dtype="int64") * 3.2
        expected = Float64Index(arr)
        fidx = idx * 3.2
        tm.assert_index_equal(fidx, expected)
        fidx = 3.2 * idx
        tm.assert_index_equal(fidx, expected)

        # interops with numpy arrays
        expected = Float64Index(arr)
        a = np.zeros(5, dtype="float64")
        result = fidx - a
        tm.assert_index_equal(result, expected)

        expected = Float64Index(-arr)
        a = np.zeros(5, dtype="float64")
        result = a - fidx
        tm.assert_index_equal(result, expected)

    def test_invalid_dtype(self, invalid_dtype):
        # GH 29539
        dtype = invalid_dtype
        msg = fr"Incorrect `dtype` passed: expected \w+(?: \w+)?, received {dtype}"
        with pytest.raises(ValueError, match=msg):
            self._index_cls([1, 2, 3], dtype=dtype)
