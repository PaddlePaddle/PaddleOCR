from __future__ import annotations

import re

import numpy as np
import pytest

from pandas._libs import (
    NaT,
    OutOfBoundsDatetime,
    Timestamp,
)
from pandas.compat import np_version_under1p18
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    PandasArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimes import sequence_to_dt64ns
from pandas.core.arrays.timedeltas import sequence_to_td64ns


# TODO: more freq variants
@pytest.fixture(params=["D", "B", "W", "M", "Q", "Y"])
def freqstr(request):
    return request.param


@pytest.fixture
def period_index(freqstr):
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates
    pi = pd.period_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return pi


@pytest.fixture
def datetime_index(freqstr):
    """
    A fixture to provide DatetimeIndex objects with different frequencies.

    Most DatetimeArray behavior is already tested in DatetimeIndex tests,
    so here we just test that the DatetimeArray behavior matches
    the DatetimeIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates, timezones
    dti = pd.date_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return dti


@pytest.fixture
def timedelta_index():
    """
    A fixture to provide TimedeltaIndex objects with different frequencies.
     Most TimedeltaArray behavior is already tested in TimedeltaIndex tests,
    so here we just test that the TimedeltaArray behavior matches
    the TimedeltaIndex behavior.
    """
    # TODO: flesh this out
    return TimedeltaIndex(["1 Day", "3 Hours", "NaT"])


class SharedTests:
    index_cls: type[DatetimeIndex | PeriodIndex | TimedeltaIndex]

    @pytest.fixture
    def arr1d(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")
        return arr

    def test_compare_len1_raises(self, arr1d):
        # make sure we raise when comparing with different lengths, specific
        #  to the case where one has length-1, which numpy would broadcast
        arr = arr1d
        idx = self.index_cls(arr)

        with pytest.raises(ValueError, match="Lengths must match"):
            arr == arr[:1]

        # test the index classes while we're at it, GH#23078
        with pytest.raises(ValueError, match="Lengths must match"):
            idx <= idx[[0]]

    @pytest.mark.parametrize(
        "result",
        [
            pd.date_range("2020", periods=3),
            pd.date_range("2020", periods=3, tz="UTC"),
            pd.timedelta_range("0 days", periods=3),
            pd.period_range("2020Q1", periods=3, freq="Q"),
        ],
    )
    def test_compare_with_Categorical(self, result):
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(self, arr1d, as_index, reverse, ordered):
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)

        left, right = arr1d, other
        if reverse:
            left, right = right, left

        ones = np.ones(arr1d.shape, dtype=bool)
        zeros = ~ones

        result = left == right
        tm.assert_numpy_array_equal(result, ones)

        result = left != right
        tm.assert_numpy_array_equal(result, zeros)

        if not reverse and not as_index:
            # Otherwise Categorical raises TypeError bc it is not ordered
            # TODO: we should probably get the same behavior regardless?
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)

            result = left <= right
            tm.assert_numpy_array_equal(result, ones)

            result = left > right
            tm.assert_numpy_array_equal(result, zeros)

            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self):
        data = np.arange(100, dtype="i8") * 24 * 3600 * 10 ** 9
        np.random.shuffle(data)

        freq = None if self.array_cls is not PeriodArray else "D"

        arr = self.array_cls(data, freq=freq)
        idx = self.index_cls._simple_new(arr)

        takers = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

    @pytest.mark.parametrize("fill_value", [2, 2.0, Timestamp.now().time])
    def test_take_fill_raises(self, fill_value):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9

        arr = self.array_cls(data, freq="D")

        msg = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr.take([0, 1], allow_fill=True, fill_value=fill_value)

    def test_take_fill(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9

        arr = self.array_cls(data, freq="D")

        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT

        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    def test_take_fill_str(self, arr1d):
        # Cast str fill_value matching other fill_value-taking methods
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        tm.assert_equal(result, expected)

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr1d.take([-1, 1], allow_fill=True, fill_value="foo")

    def test_concat_same_type(self, arr1d):
        arr = arr1d
        idx = self.index_cls(arr)
        idx = idx.insert(0, NaT)
        arr = self.array_cls(idx)

        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        arr2 = arr.astype(object)
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]), None)

        tm.assert_index_equal(self.index_cls(result), expected)

    def test_unbox_scalar(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")
        result = arr._unbox_scalar(arr[0])
        expected = arr._data.dtype.type
        assert isinstance(result, expected)

        result = arr._unbox_scalar(NaT)
        assert isinstance(result, expected)

        msg = f"'value' should be a {self.scalar_type.__name__}."
        with pytest.raises(ValueError, match=msg):
            arr._unbox_scalar("foo")

    def test_check_compatible_with(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")

        arr._check_compatible_with(arr[0])
        arr._check_compatible_with(arr[:1])
        arr._check_compatible_with(NaT)

    def test_scalar_from_string(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")
        result = arr._scalar_from_string(str(arr[0]))
        assert result == arr[0]

    def test_reduce_invalid(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")

        msg = f"'{type(arr).__name__}' does not implement reduction 'not a method'"
        with pytest.raises(TypeError, match=msg):
            arr._reduce("not a method")

    @pytest.mark.parametrize("method", ["pad", "backfill"])
    def test_fillna_method_doesnt_change_orig(self, method):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")
        arr[4] = NaT

        fill_value = arr[3] if method == "pad" else arr[5]

        result = arr.fillna(method=method)
        assert result[4] == fill_value

        # check that the original was not changed
        assert arr[4] is NaT

    def test_searchsorted(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")

        # scalar
        result = arr.searchsorted(arr[1])
        assert result == 1

        result = arr.searchsorted(arr[2], side="right")
        assert result == 3

        # own-type
        result = arr.searchsorted(arr[1:3])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        result = arr.searchsorted(arr[1:3], side="right")
        expected = np.array([2, 3], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        # GH#29884 match numpy convention on whether NaT goes
        #  at the end or the beginning
        result = arr.searchsorted(NaT)
        if np_version_under1p18:
            # Following numpy convention, NaT goes at the beginning
            #  (unlike NaN which goes at the end)
            assert result == 0
        else:
            assert result == 10

    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(self, arr1d, box, request, string_storage):
        if isinstance(arr1d, DatetimeArray):
            tz = arr1d.tz
            ts1, ts2 = arr1d[1:3]
            if tz is not None and ts1.tz.tzname(ts1) != ts2.tz.tzname(ts2):
                # If we have e.g. tzutc(), when we cast to string and parse
                #  back we get pytz.UTC, and then consider them different timezones
                #  so incorrectly raise.
                mark = pytest.mark.xfail(
                    raises=TypeError, reason="timezone comparisons inconsistent"
                )
                request.node.add_marker(mark)

        arr = arr1d
        if box is None:
            pass
        elif box == "index":
            # Test the equivalent Index.searchsorted method while we're here
            arr = self.index_cls(arr)
        else:
            # Test the equivalent Series.searchsorted method while we're here
            arr = pd.Series(arr)

        # scalar
        result = arr.searchsorted(str(arr[1]))
        assert result == 1

        result = arr.searchsorted(str(arr[2]), side="right")
        assert result == 3

        result = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)

        with pytest.raises(
            TypeError,
            match=re.escape(
                f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                "or array of those. Got 'str' instead."
            ),
        ):
            arr.searchsorted("foo")

        arr_type = "StringArray" if string_storage == "python" else "ArrowStringArray"

        with pd.option_context("string_storage", string_storage):
            with pytest.raises(
                TypeError,
                match=re.escape(
                    f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                    f"or array of those. Got '{arr_type}' instead."
                ),
            ):
                arr.searchsorted([str(arr[1]), "baz"])

    def test_getitem_near_implementation_bounds(self):
        # We only check tz-naive for DTA bc the bounds are slightly different
        #  for other tzs
        i8vals = np.asarray([NaT.value + n for n in range(1, 5)], dtype="i8")
        arr = self.array_cls(i8vals, freq="ns")
        arr[0]  # should not raise OutOfBoundsDatetime

        index = pd.Index(arr)
        index[0]  # should not raise OutOfBoundsDatetime

        ser = pd.Series(arr)
        ser[0]  # should not raise OutOfBoundsDatetime

    def test_getitem_2d(self, arr1d):
        # 2d slicing on a 1D array
        expected = type(arr1d)(arr1d._data[:, np.newaxis], dtype=arr1d.dtype)
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)

        # Lookup on a 2D array
        arr2d = expected
        expected = type(arr2d)(arr2d._data[:3, 0], dtype=arr2d.dtype)
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)

        # Scalar lookup
        result = arr2d[-1, 0]
        expected = arr1d[-1]
        assert result == expected

    def test_iter_2d(self, arr1d):
        data2d = arr1d._data[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    def test_repr_2d(self, arr1d):
        data2d = arr1d._data[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)

        result = repr(arr2d)

        if isinstance(arr2d, TimedeltaArray):
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]._repr_base()}'],\n"
                f"['{arr1d[1]._repr_base()}'],\n"
                f"['{arr1d[2]._repr_base()}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )
        else:
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]}'],\n"
                f"['{arr1d[1]}'],\n"
                f"['{arr1d[2]}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )

        assert result == expected

    def test_setitem(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")

        arr[0] = arr[1]
        expected = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        expected[0] = expected[1]

        tm.assert_numpy_array_equal(arr.asi8, expected)

        arr[:2] = arr[-2:]
        expected[:2] = expected[-2:]
        tm.assert_numpy_array_equal(arr.asi8, expected)

    @pytest.mark.parametrize(
        "box",
        [
            pd.Index,
            pd.Series,
            np.array,
            list,
            PandasArray,
        ],
    )
    def test_setitem_object_dtype(self, box, arr1d):

        expected = arr1d.copy()[::-1]
        if expected.dtype.kind in ["m", "M"]:
            expected = expected._with_freq(None)

        vals = expected
        if box is list:
            vals = list(vals)
        elif box is np.array:
            # if we do np.array(x).astype(object) then dt64 and td64 cast to ints
            vals = np.array(vals.astype(object))
        elif box is PandasArray:
            vals = box(np.asarray(vals, dtype=object))
        else:
            vals = box(vals).astype(object)

        arr1d[:] = vals

        tm.assert_equal(arr1d, expected)

    def test_setitem_strs(self, arr1d, request):
        # Check that we parse strs in both scalar and listlike
        if isinstance(arr1d, DatetimeArray):
            tz = arr1d.tz
            ts1, ts2 = arr1d[-2:]
            if tz is not None and ts1.tz.tzname(ts1) != ts2.tz.tzname(ts2):
                # If we have e.g. tzutc(), when we cast to string and parse
                #  back we get pytz.UTC, and then consider them different timezones
                #  so incorrectly raise.
                mark = pytest.mark.xfail(
                    raises=TypeError, reason="timezone comparisons inconsistent"
                )
                request.node.add_marker(mark)

        # Setting list-like of strs
        expected = arr1d.copy()
        expected[[0, 1]] = arr1d[-2:]

        result = arr1d.copy()
        result[:2] = [str(x) for x in arr1d[-2:]]
        tm.assert_equal(result, expected)

        # Same thing but now for just a scalar str
        expected = arr1d.copy()
        expected[0] = arr1d[-1]

        result = arr1d.copy()
        result[0] = str(arr1d[-1])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("as_index", [True, False])
    def test_setitem_categorical(self, arr1d, as_index):
        expected = arr1d.copy()[::-1]
        if not isinstance(expected, PeriodArray):
            expected = expected._with_freq(None)

        cat = pd.Categorical(arr1d)
        if as_index:
            cat = pd.CategoricalIndex(cat)

        arr1d[:] = cat[::-1]

        tm.assert_equal(arr1d, expected)

    def test_setitem_raises(self):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")
        val = arr[0]

        with pytest.raises(IndexError, match="index 12 is out of bounds"):
            arr[12] = val

        with pytest.raises(TypeError, match="value should be a.* 'object'"):
            arr[0] = object()

        msg = "cannot set using a list-like indexer with a different length"
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[[]] = [arr[1]]

        msg = "cannot set using a slice indexer with a different length than"
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[1:1] = arr[:3]

    @pytest.mark.parametrize("box", [list, np.array, pd.Index, pd.Series])
    def test_setitem_numeric_raises(self, arr1d, box):
        # We dont case e.g. int64 to our own dtype for setitem

        msg = (
            f"value should be a '{arr1d._scalar_type.__name__}', "
            "'NaT', or array of those. Got"
        )
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0, 1])

        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0.0, 1.0])

    def test_inplace_arithmetic(self):
        # GH#24115 check that iadd and isub are actually in-place
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")

        expected = arr + pd.Timedelta(days=1)
        arr += pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

        expected = arr - pd.Timedelta(days=1)
        arr -= pd.Timedelta(days=1)
        tm.assert_equal(arr, expected)

    def test_shift_fill_int_deprecated(self):
        # GH#31971
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = self.array_cls(data, freq="D")

        msg = "Passing <class 'int'> to shift"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = arr.shift(1, fill_value=1)

        expected = arr.copy()
        if self.array_cls is PeriodArray:
            fill_val = PeriodArray._scalar_type._from_ordinal(1, freq=arr.freq)
        else:
            fill_val = arr._scalar_type(1)
        expected[0] = fill_val
        expected[1:] = arr[:-1]
        tm.assert_equal(result, expected)

    def test_median(self, arr1d):
        arr = arr1d
        if len(arr) % 2 == 0:
            # make it easier to define `expected`
            arr = arr[:-1]

        expected = arr[len(arr) // 2]

        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected

        arr[len(arr) // 2] = NaT
        if not isinstance(expected, Period):
            expected = arr[len(arr) // 2 - 1 : len(arr) // 2 + 2].mean()

        assert arr.median(skipna=False) is NaT

        result = arr.median()
        assert type(result) is type(expected)
        assert result == expected

        assert arr[:0].median() is NaT
        assert arr[:0].median(skipna=False) is NaT

        # 2d Case
        arr2 = arr.reshape(-1, 1)

        result = arr2.median(axis=None)
        assert type(result) is type(expected)
        assert result == expected

        assert arr2.median(axis=None, skipna=False) is NaT

        result = arr2.median(axis=0)
        expected2 = type(arr)._from_sequence([expected], dtype=arr.dtype)
        tm.assert_equal(result, expected2)

        result = arr2.median(axis=0, skipna=False)
        expected2 = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        tm.assert_equal(result, expected2)

        result = arr2.median(axis=1)
        tm.assert_equal(result, arr)

        result = arr2.median(axis=1, skipna=False)
        tm.assert_equal(result, arr)

    def test_from_integer_array(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        expected = self.array_cls(arr, dtype=self.example_dtype)

        data = pd.array(arr, dtype="Int64")
        result = self.array_cls(data, dtype=self.example_dtype)

        tm.assert_extension_array_equal(result, expected)


class TestDatetimeArray(SharedTests):
    index_cls = DatetimeIndex
    array_cls = DatetimeArray
    scalar_type = Timestamp
    example_dtype = "M8[ns]"

    @pytest.fixture
    def arr1d(self, tz_naive_fixture, freqstr):
        tz = tz_naive_fixture
        dti = pd.date_range("2016-01-01 01:01:00", periods=5, freq=freqstr, tz=tz)
        dta = dti._data
        return dta

    def test_round(self, arr1d):
        # GH#24064
        dti = self.index_cls(arr1d)

        result = dti.round(freq="2T")
        expected = dti - pd.Timedelta(minutes=1)
        expected = expected._with_freq(None)
        tm.assert_index_equal(result, expected)

        dta = dti._data
        result = dta.round(freq="2T")
        expected = expected._data._with_freq(None)
        tm.assert_datetime_array_equal(result, expected)

    def test_array_interface(self, datetime_index):
        arr = DatetimeArray(datetime_index)

        # default asarray gives the same underlying data (for tz naive)
        result = np.asarray(arr)
        expected = arr._data
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # specifying M8[ns] gives the same result as default
        result = np.asarray(arr, dtype="datetime64[ns]")
        expected = arr._data
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="datetime64[ns]", copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="datetime64[ns]")
        assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to other dtype always copies
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # other dtypes handled by numpy
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_array_object_dtype(self, arr1d):
        # GH#23524
        arr = arr1d
        dti = self.index_cls(arr1d)

        expected = np.array(list(dti))

        result = np.array(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # also test the DatetimeIndex method while we're at it
        result = np.array(dti, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_tz(self, arr1d):
        # GH#23524
        arr = arr1d
        dti = self.index_cls(arr1d)

        expected = dti.asi8.view("M8[ns]")
        result = np.array(arr, dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        result = np.array(arr, dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # check that we are not making copies when setting copy=False
        result = np.array(arr, dtype="M8[ns]", copy=False)
        assert result.base is expected.base
        assert result.base is not None
        result = np.array(arr, dtype="datetime64[ns]", copy=False)
        assert result.base is expected.base
        assert result.base is not None

    def test_array_i8_dtype(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)

        expected = dti.asi8
        result = np.array(arr, dtype="i8")
        tm.assert_numpy_array_equal(result, expected)

        result = np.array(arr, dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        # check that we are still making copies when setting copy=False
        result = np.array(arr, dtype="i8", copy=False)
        assert result.base is not expected.base
        assert result.base is None

    def test_from_array_keeps_base(self):
        # Ensure that DatetimeArray._data.base isn't lost.
        arr = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        dta = DatetimeArray(arr)

        assert dta._data is arr
        dta = DatetimeArray(arr[:0])
        assert dta._data.base is arr

    def test_from_dti(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)
        assert list(dti) == list(arr)

        # Check that Index.__new__ knows what to do with DatetimeArray
        dti2 = pd.Index(arr)
        assert isinstance(dti2, DatetimeIndex)
        assert list(dti2) == list(arr)

    def test_astype_object(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)

        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(dti)

    def test_to_perioddelta(self, datetime_index, freqstr):
        # GH#23113
        dti = datetime_index
        arr = DatetimeArray(dti)

        msg = "to_perioddelta is deprecated and will be removed"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # Deprecation GH#34853
            expected = dti.to_perioddelta(freq=freqstr)
        with tm.assert_produces_warning(
            FutureWarning, match=msg, check_stacklevel=False
        ):
            # stacklevel is chosen to be "correct" for DatetimeIndex, not
            #  DatetimeArray
            result = arr.to_perioddelta(freq=freqstr)
        assert isinstance(result, TimedeltaArray)

        # placeholder until these become actual EA subclasses and we can use
        #  an EA-specific tm.assert_ function
        tm.assert_index_equal(pd.Index(result), pd.Index(expected))

    def test_to_period(self, datetime_index, freqstr):
        dti = datetime_index
        arr = DatetimeArray(dti)

        expected = dti.to_period(freq=freqstr)
        result = arr.to_period(freq=freqstr)
        assert isinstance(result, PeriodArray)

        # placeholder until these become actual EA subclasses and we can use
        #  an EA-specific tm.assert_ function
        tm.assert_index_equal(pd.Index(result), pd.Index(expected))

    def test_to_period_2d(self, arr1d):
        arr2d = arr1d.reshape(1, -1)

        warn = None if arr1d.tz is None else UserWarning
        with tm.assert_produces_warning(warn):
            result = arr2d.to_period("D")
            expected = arr1d.to_period("D").reshape(1, -1)
        tm.assert_period_array_equal(result, expected)

    @pytest.mark.parametrize("propname", DatetimeIndex._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # in this case _bool_ops is just `is_leap_year`
        dti = self.index_cls(arr1d)
        arr = arr1d
        assert dti.freq == arr.freq

        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", DatetimeIndex._field_ops)
    def test_int_properties(self, arr1d, propname):
        if propname in ["week", "weekofyear"]:
            # GH#33595 Deprecate week and weekofyear
            return
        dti = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, arr1d):
        arr = arr1d
        dti = self.index_cls(arr1d)

        now = Timestamp.now().tz_localize(dti.tz)
        result = arr.take([-1, 1], allow_fill=True, fill_value=now)
        assert result[0] == now

        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # fill_value Timedelta invalid
            arr.take([-1, 1], allow_fill=True, fill_value=now - now)

        with pytest.raises(TypeError, match=msg):
            # fill_value Period invalid
            arr.take([-1, 1], allow_fill=True, fill_value=Period("2014Q1"))

        tz = None if dti.tz is not None else "US/Eastern"
        now = Timestamp.now().tz_localize(tz)
        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            # Timestamp with mismatched tz-awareness
            arr.take([-1, 1], allow_fill=True, fill_value=now)

        value = NaT.value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # require NaT, not iNaT, as it could be confused with an integer
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have a NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        if arr.tz is not None:
            # GH#37356
            # Assuming here that arr1d fixture does not include Australia/Melbourne
            value = Timestamp.now().tz_localize("Australia/Melbourne")
            msg = "Timezones don't match. .* != 'Australia/Melbourne'"
            with pytest.raises(ValueError, match=msg):
                # require tz match, not just tzawareness match
                arr.take([-1, 1], allow_fill=True, fill_value=value)

    def test_concat_same_type_invalid(self, arr1d):
        # different timezones
        arr = arr1d

        if arr.tz is None:
            other = arr.tz_localize("UTC")
        else:
            other = arr.tz_localize(None)

        with pytest.raises(ValueError, match="to_concat must have the same"):
            arr._concat_same_type([arr, other])

    def test_concat_same_type_different_freq(self):
        # we *can* concatenate DTI with different freqs.
        a = DatetimeArray(pd.date_range("2000", periods=2, freq="D", tz="US/Central"))
        b = DatetimeArray(pd.date_range("2000", periods=2, freq="H", tz="US/Central"))
        result = DatetimeArray._concat_same_type([a, b])
        expected = DatetimeArray(
            pd.to_datetime(
                [
                    "2000-01-01 00:00:00",
                    "2000-01-02 00:00:00",
                    "2000-01-01 00:00:00",
                    "2000-01-01 01:00:00",
                ]
            ).tz_localize("US/Central")
        )

        tm.assert_datetime_array_equal(result, expected)

    def test_strftime(self, arr1d):
        arr = arr1d

        result = arr.strftime("%Y %b")
        expected = np.array([ts.strftime("%Y %b") for ts in arr], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime_nat(self):
        # GH 29578
        arr = DatetimeArray(DatetimeIndex(["2019-01-01", NaT]))

        result = arr.strftime("%Y-%m-%d")
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


class TestTimedeltaArray(SharedTests):
    index_cls = TimedeltaIndex
    array_cls = TimedeltaArray
    scalar_type = pd.Timedelta
    example_dtype = "m8[ns]"

    def test_from_tdi(self):
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        arr = TimedeltaArray(tdi)
        assert list(arr) == list(tdi)

        # Check that Index.__new__ knows what to do with TimedeltaArray
        tdi2 = pd.Index(arr)
        assert isinstance(tdi2, TimedeltaIndex)
        assert list(tdi2) == list(arr)

    def test_astype_object(self):
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        arr = TimedeltaArray(tdi)
        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(tdi)

    def test_to_pytimedelta(self, timedelta_index):
        tdi = timedelta_index
        arr = TimedeltaArray(tdi)

        expected = tdi.to_pytimedelta()
        result = arr.to_pytimedelta()

        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, timedelta_index):
        tdi = timedelta_index
        arr = TimedeltaArray(tdi)

        expected = tdi.total_seconds()
        result = arr.total_seconds()

        tm.assert_numpy_array_equal(result, expected.values)

    @pytest.mark.parametrize("propname", TimedeltaIndex._field_ops)
    def test_int_properties(self, timedelta_index, propname):
        tdi = timedelta_index
        arr = TimedeltaArray(tdi)

        result = getattr(arr, propname)
        expected = np.array(getattr(tdi, propname), dtype=result.dtype)

        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, timedelta_index):
        arr = TimedeltaArray(timedelta_index)

        # default asarray gives the same underlying data
        result = np.asarray(arr)
        expected = arr._data
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # specifying m8[ns] gives the same result as default
        result = np.asarray(arr, dtype="timedelta64[ns]")
        expected = arr._data
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[ns]", copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[ns]")
        assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to other dtype always copies
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # other dtypes handled by numpy
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, timedelta_index):
        tdi = timedelta_index
        arr = TimedeltaArray(tdi)

        td1 = pd.Timedelta(days=1)
        result = arr.take([-1, 1], allow_fill=True, fill_value=td1)
        assert result[0] == td1

        now = Timestamp.now()
        value = now
        msg = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # fill_value Timestamp invalid
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = now.to_period("D")
        with pytest.raises(TypeError, match=msg):
            # fill_value Period invalid
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = np.datetime64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have a NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)


class TestPeriodArray(SharedTests):
    index_cls = PeriodIndex
    array_cls = PeriodArray
    scalar_type = Period
    example_dtype = PeriodIndex([], freq="W").dtype

    @pytest.fixture
    def arr1d(self, period_index):
        return period_index._data

    def test_from_pi(self, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        assert list(arr) == list(pi)

        # Check that Index.__new__ knows what to do with PeriodArray
        pi2 = pd.Index(arr)
        assert isinstance(pi2, PeriodIndex)
        assert list(pi2) == list(arr)

    def test_astype_object(self, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d
        asobj = arr.astype("O")
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(pi)

    def test_take_fill_valid(self, arr1d):
        arr = arr1d

        value = NaT.value
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # require NaT, not iNaT, as it could be confused with an integer
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # require appropriate-dtype if we have a NA value
            arr.take([-1, 1], allow_fill=True, fill_value=value)

    @pytest.mark.parametrize("how", ["S", "E"])
    def test_to_timestamp(self, how, arr1d):
        pi = self.index_cls(arr1d)
        arr = arr1d

        expected = DatetimeArray(pi.to_timestamp(how=how))
        result = arr.to_timestamp(how=how)
        assert isinstance(result, DatetimeArray)

        # placeholder until these become actual EA subclasses and we can use
        #  an EA-specific tm.assert_ function
        tm.assert_index_equal(pd.Index(result), pd.Index(expected))

    def test_to_timestamp_out_of_bounds(self):
        # GH#19643 previously overflowed silently
        pi = pd.period_range("1500", freq="Y", periods=3)
        msg = "Out of bounds nanosecond timestamp: 1500-01-01 00:00:00"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi.to_timestamp()

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi._data.to_timestamp()

    @pytest.mark.parametrize("propname", PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # in this case _bool_ops is just `is_leap_year`
        pi = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))

        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", PeriodArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        pi = self.index_cls(arr1d)
        arr = arr1d

        result = getattr(arr, propname)
        expected = np.array(getattr(pi, propname))

        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, arr1d):
        arr = arr1d

        # default asarray gives objects
        result = np.asarray(arr)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # to object dtype (same as default)
        result = np.asarray(arr, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(arr, dtype="int64")
        tm.assert_numpy_array_equal(result, arr.asi8)

        # to other dtypes
        msg = r"float\(\) argument must be a string or a( real)? number, not 'Period'"
        with pytest.raises(TypeError, match=msg):
            np.asarray(arr, dtype="float64")

        result = np.asarray(arr, dtype="S20")
        expected = np.asarray(arr).astype("S20")
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime(self, arr1d):
        arr = arr1d

        result = arr.strftime("%Y")
        expected = np.array([per.strftime("%Y") for per in arr], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime_nat(self):
        # GH 29578
        arr = PeriodArray(PeriodIndex(["2019-01-01", NaT], dtype="period[D]"))

        result = arr.strftime("%Y-%m-%d")
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "arr,casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (NaT, np.timedelta64("NaT", "ns")),
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (NaT, np.datetime64("NaT", "ns")),
        ),
        (pd.period_range("2000-01-01", periods=3, freq="D")._data, (NaT,)),
    ],
    ids=lambda x: type(x).__name__,
)
def test_casting_nat_setitem_array(arr, casting_nats):
    expected = type(arr)._from_sequence([NaT, arr[1], arr[2]])

    for nat in casting_nats:
        arr = arr.copy()
        arr[0] = nat
        tm.assert_equal(arr, expected)


@pytest.mark.parametrize(
    "arr,non_casting_nats",
    [
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (np.datetime64("NaT", "ns"), NaT.value),
        ),
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (np.timedelta64("NaT", "ns"), NaT.value),
        ),
        (
            pd.period_range("2000-01-01", periods=3, freq="D")._data,
            (np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"), NaT.value),
        ),
    ],
    ids=lambda x: type(x).__name__,
)
def test_invalid_nat_setitem_array(arr, non_casting_nats):
    msg = (
        "value should be a '(Timestamp|Timedelta|Period)', 'NaT', or array of those. "
        "Got '(timedelta64|datetime64|int)' instead."
    )

    for nat in non_casting_nats:
        with pytest.raises(TypeError, match=msg):
            arr[0] = nat


@pytest.mark.parametrize(
    "arr",
    [
        pd.date_range("2000", periods=4).array,
        pd.timedelta_range("2000", periods=4).array,
    ],
)
def test_to_numpy_extra(arr):
    if np_version_under1p18:
        # np.isnan(NaT) raises, so use pandas'
        isnan = pd.isna
    else:
        isnan = np.isnan

    arr[0] = NaT
    original = arr.copy()

    result = arr.to_numpy()
    assert isnan(result[0])

    result = arr.to_numpy(dtype="int64")
    assert result[0] == -9223372036854775808

    result = arr.to_numpy(dtype="int64", na_value=0)
    assert result[0] == 0

    result = arr.to_numpy(na_value=arr[1].to_numpy())
    assert result[0] == result[1]

    result = arr.to_numpy(na_value=arr[1].to_numpy(copy=False))
    assert result[0] == result[1]

    tm.assert_equal(arr, original)


@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        TimedeltaIndex([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "klass",
    [
        list,
        np.array,
        pd.array,
        pd.Series,
        pd.Index,
        pd.Categorical,
        pd.CategoricalIndex,
    ],
)
def test_searchsorted_datetimelike_with_listlike(values, klass, as_index):
    # https://github.com/pandas-dev/pandas/issues/32762
    if not as_index:
        values = values._data

    result = values.searchsorted(klass(values))
    expected = np.array([0, 1], dtype=result.dtype)

    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        TimedeltaIndex([1, 2], unit="D"),
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "arg", [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]
)
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(values, arg):
    # https://github.com/pandas-dev/pandas/issues/32762
    msg = "[Unexpected type|Cannot compare]"
    with pytest.raises(TypeError, match=msg):
        values.searchsorted(arg)


@pytest.mark.parametrize("klass", [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass):
    # https://github.com/pandas-dev/pandas/issues/26109
    strings = ["2020Q1", "2020Q2"] * 2
    data = klass(strings)
    result = PeriodIndex(data, freq="Q")
    expected = PeriodIndex([Period(s) for s in strings])
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_pandas_array(dtype):
    # GH#24615
    data = np.array([1, 2, 3], dtype=dtype)
    arr = PandasArray(data)

    cls = {"M8[ns]": DatetimeArray, "m8[ns]": TimedeltaArray}[dtype]

    result = cls(arr)
    expected = cls(data)
    tm.assert_extension_array_equal(result, expected)

    result = cls._from_sequence(arr)
    expected = cls._from_sequence(data)
    tm.assert_extension_array_equal(result, expected)

    func = {"M8[ns]": sequence_to_dt64ns, "m8[ns]": sequence_to_td64ns}[dtype]
    result = func(arr)[0]
    expected = func(data)[0]
    tm.assert_equal(result, expected)

    func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]
    result = func(arr).array
    expected = func(data).array
    tm.assert_equal(result, expected)

    # Let's check the Indexes while we're here
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)


@pytest.fixture(
    params=[
        "memoryview",
        "array",
        pytest.param("dask", marks=td.skip_if_no("dask.array")),
        pytest.param("xarray", marks=td.skip_if_no("xarray")),
    ]
)
def array_likes(request):
    # GH#24539 recognize e.g xarray, dask, ...
    arr = np.array([1, 2, 3], dtype=np.int64)

    name = request.param
    if name == "memoryview":
        data = memoryview(arr)
    elif name == "array":
        # stdlib array
        import array

        data = array.array("i", arr)
    elif name == "dask":
        import dask.array

        data = dask.array.array(arr)
    elif name == "xarray":
        import xarray as xr

        data = xr.DataArray(arr)

    return arr, data


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_obscure_array(dtype, array_likes):
    # GH#24539 recognize e.g xarray, dask, ...
    # Note: we dont do this for PeriodArray bc _from_sequence won't accept
    #  an array of integers
    # TODO: could check with arraylike of Period objects
    arr, data = array_likes

    cls = {"M8[ns]": DatetimeArray, "m8[ns]": TimedeltaArray}[dtype]

    expected = cls(arr)
    result = cls._from_sequence(data)
    tm.assert_extension_array_equal(result, expected)

    func = {"M8[ns]": sequence_to_dt64ns, "m8[ns]": sequence_to_td64ns}[dtype]
    result = func(arr)[0]
    expected = func(data)[0]
    tm.assert_equal(result, expected)

    # FIXME: dask and memoryview both break on these
    # func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]
    # result = func(arr).array
    # expected = func(data).array
    # tm.assert_equal(result, expected)

    # Let's check the Indexes while we're here
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)
