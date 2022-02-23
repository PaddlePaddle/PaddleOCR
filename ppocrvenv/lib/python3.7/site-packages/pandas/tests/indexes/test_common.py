"""
Collection of tests asserting things that should be true for
any index subclass. Makes use of the `indices` fixture defined
in pandas/tests/indexes/conftest.py.
"""
import re

import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas.compat import IS64

from pandas.core.dtypes.common import (
    is_period_dtype,
    needs_i8_conversion,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    TimedeltaIndex,
)
import pandas._testing as tm


class TestCommon:
    def test_droplevel(self, index):
        # GH 21115
        if isinstance(index, MultiIndex):
            # Tested separately in test_multi.py
            return

        assert index.droplevel([]).equals(index)

        for level in index.name, [index.name]:
            if isinstance(index.name, tuple) and level is index.name:
                # GH 21121 : droplevel with tuple name
                continue
            msg = (
                "Cannot remove 1 levels from an index with 1 levels: at least one "
                "level must be left."
            )
            with pytest.raises(ValueError, match=msg):
                index.droplevel(level)

        for level in "wrong", ["wrong"]:
            with pytest.raises(
                KeyError,
                match=r"'Requested level \(wrong\) does not match index name \(None\)'",
            ):
                index.droplevel(level)

    def test_constructor_non_hashable_name(self, index_flat):
        # GH 20527
        index = index_flat

        message = "Index.name must be a hashable type"
        renamed = [["1"]]

        # With .rename()
        with pytest.raises(TypeError, match=message):
            index.rename(name=renamed)

        # With .set_names()
        with pytest.raises(TypeError, match=message):
            index.set_names(names=renamed)

    def test_constructor_unwraps_index(self, index_flat):
        a = index_flat
        b = type(a)(a)
        tm.assert_equal(a._data, b._data)

    def test_to_flat_index(self, index_flat):
        # 22866
        index = index_flat

        result = index.to_flat_index()
        tm.assert_index_equal(result, index)

    def test_set_name_methods(self, index_flat):
        # MultiIndex tested separately
        index = index_flat
        new_name = "This is the new name for this index"

        original_name = index.name
        new_ind = index.set_names([new_name])
        assert new_ind.name == new_name
        assert index.name == original_name
        res = index.rename(new_name, inplace=True)

        # should return None
        assert res is None
        assert index.name == new_name
        assert index.names == [new_name]
        # FIXME: dont leave commented-out
        # with pytest.raises(TypeError, match="list-like"):
        #    # should still fail even if it would be the right length
        #    ind.set_names("a")
        with pytest.raises(ValueError, match="Level must be None"):
            index.set_names("a", level=0)

        # rename in place just leaves tuples and other containers alone
        name = ("A", "B")
        index.rename(name, inplace=True)
        assert index.name == name
        assert index.names == [name]

    def test_copy_and_deepcopy(self, index_flat):
        from copy import (
            copy,
            deepcopy,
        )

        index = index_flat

        for func in (copy, deepcopy):
            idx_copy = func(index)
            assert idx_copy is not index
            assert idx_copy.equals(index)

        new_copy = index.copy(deep=True, name="banana")
        assert new_copy.name == "banana"

    def test_unique(self, index_flat):
        # don't test a MultiIndex here (as its tested separated)
        index = index_flat

        # GH 17896
        expected = index.drop_duplicates()
        for level in 0, index.name, None:
            result = index.unique(level=level)
            tm.assert_index_equal(result, expected)

        msg = "Too many levels: Index has only 1 level, not 4"
        with pytest.raises(IndexError, match=msg):
            index.unique(level=3)

        msg = (
            fr"Requested level \(wrong\) does not match index name "
            fr"\({re.escape(index.name.__repr__())}\)"
        )
        with pytest.raises(KeyError, match=msg):
            index.unique(level="wrong")

    def test_get_unique_index(self, index_flat):
        # MultiIndex tested separately
        index = index_flat
        if not len(index):
            pytest.skip("Skip check for empty Index and MultiIndex")

        idx = index[[0] * 5]
        idx_unique = index[[0]]

        # We test against `idx_unique`, so first we make sure it's unique
        # and doesn't contain nans.
        assert idx_unique.is_unique is True
        try:
            assert idx_unique.hasnans is False
        except NotImplementedError:
            pass

        result = idx._get_unique_index()
        tm.assert_index_equal(result, idx_unique)

        # nans:
        if not index._can_hold_na:
            pytest.skip("Skip na-check if index cannot hold na")

        if is_period_dtype(index.dtype):
            vals = index[[0] * 5]._data
            vals[0] = pd.NaT
        elif needs_i8_conversion(index.dtype):
            vals = index._data._ndarray[[0] * 5]
            vals[0] = iNaT
        else:
            vals = index.values[[0] * 5]
            vals[0] = np.nan

        vals_unique = vals[:2]
        if index.dtype.kind in ["m", "M"]:
            # i.e. needs_i8_conversion but not period_dtype, as above
            vals = type(index._data)(vals, dtype=index.dtype)
            vals_unique = type(index._data)._simple_new(vals_unique, dtype=index.dtype)
        idx_nan = index._shallow_copy(vals)
        idx_unique_nan = index._shallow_copy(vals_unique)
        assert idx_unique_nan.is_unique is True

        assert idx_nan.dtype == index.dtype
        assert idx_unique_nan.dtype == index.dtype

        expected = idx_unique_nan
        for i in [idx_nan, idx_unique_nan]:
            result = i._get_unique_index()
            tm.assert_index_equal(result, expected)

    def test_searchsorted_monotonic(self, index_flat):
        # GH17271
        index = index_flat
        # not implemented for tuple searches in MultiIndex
        # or Intervals searches in IntervalIndex
        if isinstance(index, pd.IntervalIndex):
            pytest.skip("Skip check for MultiIndex/IntervalIndex")

        # nothing to test if the index is empty
        if index.empty:
            pytest.skip("Skip check for empty Index")
        value = index[0]

        # determine the expected results (handle dupes for 'right')
        expected_left, expected_right = 0, (index == value).argmin()
        if expected_right == 0:
            # all values are the same, expected_right should be length
            expected_right = len(index)

        # test _searchsorted_monotonic in all cases
        # test searchsorted only for increasing
        if index.is_monotonic_increasing:
            ssm_left = index._searchsorted_monotonic(value, side="left")
            assert expected_left == ssm_left

            ssm_right = index._searchsorted_monotonic(value, side="right")
            assert expected_right == ssm_right

            ss_left = index.searchsorted(value, side="left")
            assert expected_left == ss_left

            ss_right = index.searchsorted(value, side="right")
            assert expected_right == ss_right

        elif index.is_monotonic_decreasing:
            ssm_left = index._searchsorted_monotonic(value, side="left")
            assert expected_left == ssm_left

            ssm_right = index._searchsorted_monotonic(value, side="right")
            assert expected_right == ssm_right
        else:
            # non-monotonic should raise.
            msg = "index must be monotonic increasing or decreasing"
            with pytest.raises(ValueError, match=msg):
                index._searchsorted_monotonic(value, side="left")

    def test_drop_duplicates(self, index_flat, keep):
        # MultiIndex is tested separately
        index = index_flat
        if isinstance(index, RangeIndex):
            pytest.skip(
                "RangeIndex is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )
        if len(index) == 0:
            pytest.skip(
                "empty index is tested in test_drop_duplicates_no_duplicates "
                "as it cannot hold duplicates"
            )

        # make unique index
        holder = type(index)
        unique_values = list(set(index))
        unique_idx = holder(unique_values)

        # make duplicated index
        n = len(unique_idx)
        duplicated_selection = np.random.choice(n, int(n * 1.5))
        idx = holder(unique_idx.values[duplicated_selection])

        # Series.duplicated is tested separately
        expected_duplicated = (
            pd.Series(duplicated_selection).duplicated(keep=keep).values
        )
        tm.assert_numpy_array_equal(idx.duplicated(keep=keep), expected_duplicated)

        # Series.drop_duplicates is tested separately
        expected_dropped = holder(pd.Series(idx).drop_duplicates(keep=keep))
        tm.assert_index_equal(idx.drop_duplicates(keep=keep), expected_dropped)

    def test_drop_duplicates_no_duplicates(self, index_flat):
        # MultiIndex is tested separately
        index = index_flat

        # make unique index
        if isinstance(index, RangeIndex):
            # RangeIndex cannot have duplicates
            unique_idx = index
        else:
            holder = type(index)
            unique_values = list(set(index))
            unique_idx = holder(unique_values)

        # check on unique index
        expected_duplicated = np.array([False] * len(unique_idx), dtype="bool")
        tm.assert_numpy_array_equal(unique_idx.duplicated(), expected_duplicated)
        result_dropped = unique_idx.drop_duplicates()
        tm.assert_index_equal(result_dropped, unique_idx)
        # validate shallow copy
        assert result_dropped is not unique_idx

    def test_drop_duplicates_inplace(self, index):
        msg = r"drop_duplicates\(\) got an unexpected keyword argument"
        with pytest.raises(TypeError, match=msg):
            index.drop_duplicates(inplace=True)

    def test_has_duplicates(self, index_flat):
        # MultiIndex tested separately in:
        #   tests/indexes/multi/test_unique_and_duplicates.
        index = index_flat
        holder = type(index)
        if not len(index) or isinstance(index, RangeIndex):
            # MultiIndex tested separately in:
            #   tests/indexes/multi/test_unique_and_duplicates.
            # RangeIndex is unique by definition.
            pytest.skip("Skip check for empty Index, MultiIndex, and RangeIndex")

        idx = holder([index[0]] * 5)
        assert idx.is_unique is False
        assert idx.has_duplicates is True

    @pytest.mark.parametrize(
        "dtype",
        ["int64", "uint64", "float64", "category", "datetime64[ns]", "timedelta64[ns]"],
    )
    def test_astype_preserves_name(self, index, dtype):
        # https://github.com/pandas-dev/pandas/issues/32013
        if isinstance(index, MultiIndex):
            index.names = ["idx" + str(i) for i in range(index.nlevels)]
        else:
            index.name = "idx"

        warn = None
        if dtype in ["int64", "uint64"]:
            if needs_i8_conversion(index.dtype):
                warn = FutureWarning
        elif (
            isinstance(index, DatetimeIndex)
            and index.tz is not None
            and dtype == "datetime64[ns]"
        ):
            # This astype is deprecated in favor of tz_localize
            warn = FutureWarning
        try:
            # Some of these conversions cannot succeed so we use a try / except
            with tm.assert_produces_warning(warn):
                result = index.astype(dtype)
        except (ValueError, TypeError, NotImplementedError, SystemError):
            return

        if isinstance(index, MultiIndex):
            assert result.names == index.names
        else:
            assert result.name == index.name

    def test_asi8_deprecation(self, index):
        # GH#37877
        if isinstance(index, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            warn = None
        else:
            warn = FutureWarning

        with tm.assert_produces_warning(warn):
            index.asi8


@pytest.mark.parametrize("na_position", [None, "middle"])
def test_sort_values_invalid_na_position(index_with_missing, na_position):

    with pytest.raises(ValueError, match=f"invalid na_position: {na_position}"):
        index_with_missing.sort_values(na_position=na_position)


@pytest.mark.parametrize("na_position", ["first", "last"])
def test_sort_values_with_missing(index_with_missing, na_position):
    # GH 35584. Test that sort_values works with missing values,
    # sort non-missing and place missing according to na_position

    if isinstance(index_with_missing, CategoricalIndex):
        pytest.skip("missing value sorting order not well-defined")

    missing_count = np.sum(index_with_missing.isna())
    not_na_vals = index_with_missing[index_with_missing.notna()].values
    sorted_values = np.sort(not_na_vals)
    if na_position == "first":
        sorted_values = np.concatenate([[None] * missing_count, sorted_values])
    else:
        sorted_values = np.concatenate([sorted_values, [None] * missing_count])
    expected = type(index_with_missing)(sorted_values)

    result = index_with_missing.sort_values(na_position=na_position)
    tm.assert_index_equal(result, expected)


def test_ndarray_compat_properties(index):
    if isinstance(index, PeriodIndex) and not IS64:
        pytest.skip("Overflow")
    idx = index
    assert idx.T.equals(idx)
    assert idx.transpose().equals(idx)

    values = idx.values

    assert idx.shape == values.shape
    assert idx.ndim == values.ndim
    assert idx.size == values.size

    if not isinstance(index, (RangeIndex, MultiIndex)):
        # These two are not backed by an ndarray
        assert idx.nbytes == values.nbytes

    # test for validity
    idx.nbytes
    idx.values.nbytes
