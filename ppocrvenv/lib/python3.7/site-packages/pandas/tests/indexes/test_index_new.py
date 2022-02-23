"""
Tests for the Index constructor conducting inference.
"""
from decimal import Decimal

import numpy as np
import pytest

from pandas.core.dtypes.common import is_unsigned_integer_dtype

from pandas import (
    NA,
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    Index,
    Int64Index,
    IntervalIndex,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    UInt64Index,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


class TestIndexConstructorInference:
    @pytest.mark.parametrize("na_value", [None, np.nan])
    @pytest.mark.parametrize("vtype", [list, tuple, iter])
    def test_construction_list_tuples_nan(self, na_value, vtype):
        # GH#18505 : valid tuples containing NaN
        values = [(1, "two"), (3.0, na_value)]
        result = Index(vtype(values))
        expected = MultiIndex.from_tuples(values)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [int, "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"],
    )
    def test_constructor_int_dtype_float(self, dtype):
        # GH#18400
        if is_unsigned_integer_dtype(dtype):
            index_type = UInt64Index
        else:
            index_type = Int64Index

        expected = index_type([0, 1, 2, 3])
        result = Index([0.0, 1.0, 2.0, 3.0], dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("cast_index", [True, False])
    @pytest.mark.parametrize(
        "vals", [[True, False, True], np.array([True, False, True], dtype=bool)]
    )
    def test_constructor_dtypes_to_object(self, cast_index, vals):
        if cast_index:
            index = Index(vals, dtype=bool)
        else:
            index = Index(vals)

        assert type(index) is Index
        assert index.dtype == object

    def test_constructor_categorical_to_object(self):
        # GH#32167 Categorical data and dtype=object should return object-dtype
        ci = CategoricalIndex(range(5))
        result = Index(ci, dtype=object)
        assert not isinstance(result, CategoricalIndex)

    def test_constructor_infer_periodindex(self):
        xp = period_range("2012-1-1", freq="M", periods=3)
        rs = Index(xp)
        tm.assert_index_equal(rs, xp)
        assert isinstance(rs, PeriodIndex)

    def test_from_list_of_periods(self):
        rng = period_range("1/1/2000", periods=20, freq="D")
        periods = list(rng)

        result = Index(periods)
        assert isinstance(result, PeriodIndex)

    @pytest.mark.parametrize("pos", [0, 1])
    @pytest.mark.parametrize(
        "klass,dtype,ctor",
        [
            (DatetimeIndex, "datetime64[ns]", np.datetime64("nat")),
            (TimedeltaIndex, "timedelta64[ns]", np.timedelta64("nat")),
        ],
    )
    def test_constructor_infer_nat_dt_like(
        self, pos, klass, dtype, ctor, nulls_fixture, request
    ):
        if isinstance(nulls_fixture, Decimal):
            # We dont cast these to datetime64/timedelta64
            return

        expected = klass([NaT, NaT])
        assert expected.dtype == dtype
        data = [ctor]
        data.insert(pos, nulls_fixture)

        warn = None
        if nulls_fixture is NA:
            expected = Index([NA, NaT])
            mark = pytest.mark.xfail(reason="Broken with np.NaT ctor; see GH 31884")
            request.node.add_marker(mark)
            # GH#35942 numpy will emit a DeprecationWarning within the
            #  assert_index_equal calls.  Since we can't do anything
            #  about it until GH#31884 is fixed, we suppress that warning.
            warn = DeprecationWarning

        result = Index(data)

        with tm.assert_produces_warning(warn):
            tm.assert_index_equal(result, expected)

        result = Index(np.array(data, dtype=object))

        with tm.assert_produces_warning(warn):
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("swap_objs", [True, False])
    def test_constructor_mixed_nat_objs_infers_object(self, swap_objs):
        # mixed np.datetime64/timedelta64 nat results in object
        data = [np.datetime64("nat"), np.timedelta64("nat")]
        if swap_objs:
            data = data[::-1]

        expected = Index(data, dtype=object)
        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)

    @pytest.mark.parametrize("swap_objs", [True, False])
    def test_constructor_datetime_and_datetime64(self, swap_objs):
        data = [Timestamp(2021, 6, 8, 9, 42), np.datetime64("now")]
        if swap_objs:
            data = data[::-1]
        expected = DatetimeIndex(data)

        tm.assert_index_equal(Index(data), expected)
        tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)


class TestDtypeEnforced:
    # check we don't silently ignore the dtype keyword

    @pytest.mark.parametrize("dtype", [object, "float64", "uint64", "category"])
    def test_constructor_range_values_mismatched_dtype(self, dtype):
        rng = Index(range(5))

        result = Index(rng, dtype=dtype)
        assert result.dtype == dtype

        result = Index(range(5), dtype=dtype)
        assert result.dtype == dtype

    @pytest.mark.parametrize("dtype", [object, "float64", "uint64", "category"])
    def test_constructor_categorical_values_mismatched_non_ea_dtype(self, dtype):
        cat = Categorical([1, 2, 3])

        result = Index(cat, dtype=dtype)
        assert result.dtype == dtype

    def test_constructor_categorical_values_mismatched_dtype(self):
        dti = date_range("2016-01-01", periods=3)
        cat = Categorical(dti)
        result = Index(cat, dti.dtype)
        tm.assert_index_equal(result, dti)

        dti2 = dti.tz_localize("Asia/Tokyo")
        cat2 = Categorical(dti2)
        result = Index(cat2, dti2.dtype)
        tm.assert_index_equal(result, dti2)

        ii = IntervalIndex.from_breaks(range(5))
        cat3 = Categorical(ii)
        result = Index(cat3, dtype=ii.dtype)
        tm.assert_index_equal(result, ii)

    def test_constructor_ea_values_mismatched_categorical_dtype(self):
        dti = date_range("2016-01-01", periods=3)
        result = Index(dti, dtype="category")
        expected = CategoricalIndex(dti)
        tm.assert_index_equal(result, expected)

        dti2 = date_range("2016-01-01", periods=3, tz="US/Pacific")
        result = Index(dti2, dtype="category")
        expected = CategoricalIndex(dti2)
        tm.assert_index_equal(result, expected)

    def test_constructor_period_values_mismatched_dtype(self):
        pi = period_range("2016-01-01", periods=3, freq="D")
        result = Index(pi, dtype="category")
        expected = CategoricalIndex(pi)
        tm.assert_index_equal(result, expected)

    def test_constructor_timedelta64_values_mismatched_dtype(self):
        # check we don't silently ignore the dtype keyword
        tdi = timedelta_range("4 Days", periods=5)
        result = Index(tdi, dtype="category")
        expected = CategoricalIndex(tdi)
        tm.assert_index_equal(result, expected)

    def test_constructor_interval_values_mismatched_dtype(self):
        dti = date_range("2016-01-01", periods=3)
        ii = IntervalIndex.from_breaks(dti)
        result = Index(ii, dtype="category")
        expected = CategoricalIndex(ii)
        tm.assert_index_equal(result, expected)

    def test_constructor_datetime64_values_mismatched_period_dtype(self):
        dti = date_range("2016-01-01", periods=3)
        result = Index(dti, dtype="Period[D]")
        expected = dti.to_period("D")
        tm.assert_index_equal(result, expected)


class TestIndexConstructorUnwrapping:
    # Test passing different arraylike values to pd.Index

    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    def test_constructor_from_series_dt64(self, klass):
        stamps = [Timestamp("20110101"), Timestamp("20120101"), Timestamp("20130101")]
        expected = DatetimeIndex(stamps)
        ser = Series(stamps)
        result = klass(ser)
        tm.assert_index_equal(result, expected)
