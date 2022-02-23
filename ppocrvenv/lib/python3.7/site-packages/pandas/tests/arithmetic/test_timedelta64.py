# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas.errors import (
    OutOfBoundsDatetime,
    PerformanceWarning,
)

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    offsets,
    timedelta_range,
)
import pandas._testing as tm
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)


def assert_dtype(obj, expected_dtype):
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    """
    dtype = tm.get_dtype(obj)

    assert dtype == expected_dtype


def get_expected_name(box, names):
    if box is DataFrame:
        # Since we are operating with a DataFrame and a non-DataFrame,
        # the non-DataFrame is cast to Series and its name ignored.
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Comparisons


class TestTimedelta64ArrayLikeComparisons:
    # Comparison tests for timedelta64[ns] vectors fully parametrized over
    #  DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all comparison
    #  tests will eventually end up here.

    def test_compare_timedelta64_zerodim(self, box_with_array):
        # GH#26689 should unbox when comparing with zerodim array
        box = box_with_array
        xbox = (
            box_with_array if box_with_array not in [pd.Index, pd.array] else np.ndarray
        )

        tdi = timedelta_range("2H", periods=4)
        other = np.array(tdi.to_numpy()[0])

        tdi = tm.box_expected(tdi, box)
        res = tdi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

        msg = "Invalid comparison between dtype"
        with pytest.raises(TypeError, match=msg):
            # zero-dim of wrong dtype should still raise
            tdi >= np.array(4)

    @pytest.mark.parametrize(
        "td_scalar",
        [
            timedelta(days=1),
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            offsets.Hour(24),
        ],
    )
    def test_compare_timedeltalike_scalar(self, box_with_array, td_scalar):
        # regression test for GH#5963
        box = box_with_array
        xbox = box if box not in [pd.Index, pd.array] else np.ndarray

        ser = Series([timedelta(days=1), timedelta(days=2)])
        ser = tm.box_expected(ser, box)
        actual = ser > td_scalar
        expected = Series([False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "invalid",
        [
            345600000000000,
            "a",
            Timestamp.now(),
            Timestamp.now("UTC"),
            Timestamp.now().to_datetime64(),
            Timestamp.now().to_pydatetime(),
            Timestamp.now().date(),
        ],
    )
    def test_td64_comparisons_invalid(self, box_with_array, invalid):
        # GH#13624 for str
        box = box_with_array

        rng = timedelta_range("1 days", periods=10)
        obj = tm.box_expected(rng, box)

        assert_invalid_comparison(obj, invalid, box)

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.date_range("1970-01-01", periods=10, tz="UTC").array,
            np.array(pd.date_range("1970-01-01", periods=10)),
            list(pd.date_range("1970-01-01", periods=10)),
            pd.date_range("1970-01-01", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_td64arr_cmp_arraylike_invalid(self, other):
        # We don't parametrize this over box_with_array because listlike
        #  other plays poorly with assert_invalid_comparison reversed checks

        rng = timedelta_range("1 days", periods=10)._data
        assert_invalid_comparison(rng, other, tm.to_array)

    def test_td64arr_cmp_mixed_invalid(self):
        rng = timedelta_range("1 days", periods=5)._data

        other = np.array([0, 1, 2, rng[3], Timestamp.now()])
        result = rng == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = rng != other
        tm.assert_numpy_array_equal(result, ~expected)

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            rng < other
        with pytest.raises(TypeError, match=msg):
            rng > other
        with pytest.raises(TypeError, match=msg):
            rng <= other
        with pytest.raises(TypeError, match=msg):
            rng >= other


class TestTimedelta64ArrayComparisons:
    # TODO: All of these need to be parametrized over box

    @pytest.mark.parametrize("dtype", [None, object])
    def test_comp_nat(self, dtype):
        left = TimedeltaIndex([Timedelta("1 days"), NaT, Timedelta("3 days")])
        right = TimedeltaIndex([NaT, NaT, Timedelta("3 days")])

        lhs, rhs = left, right
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)

        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)

        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)

        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    def test_comparisons_nat(self):
        tdidx1 = TimedeltaIndex(
            [
                "1 day",
                NaT,
                "1 day 00:00:01",
                NaT,
                "1 day 00:00:01",
                "5 day 00:00:03",
            ]
        )
        tdidx2 = TimedeltaIndex(
            ["2 day", "2 day", NaT, NaT, "1 day 00:00:02", "5 days 00:00:03"]
        )
        tdarr = np.array(
            [
                np.timedelta64(2, "D"),
                np.timedelta64(2, "D"),
                np.timedelta64("nat"),
                np.timedelta64("nat"),
                np.timedelta64(1, "D") + np.timedelta64(2, "s"),
                np.timedelta64(5, "D") + np.timedelta64(3, "s"),
            ]
        )

        cases = [(tdidx1, tdidx2), (tdidx1, tdarr)]

        # Check pd.NaT is handles as the same as np.nan
        for idx1, idx2 in cases:

            result = idx1 < idx2
            expected = np.array([True, False, False, False, True, False])
            tm.assert_numpy_array_equal(result, expected)

            result = idx2 > idx1
            expected = np.array([True, False, False, False, True, False])
            tm.assert_numpy_array_equal(result, expected)

            result = idx1 <= idx2
            expected = np.array([True, False, False, False, True, True])
            tm.assert_numpy_array_equal(result, expected)

            result = idx2 >= idx1
            expected = np.array([True, False, False, False, True, True])
            tm.assert_numpy_array_equal(result, expected)

            result = idx1 == idx2
            expected = np.array([False, False, False, False, False, True])
            tm.assert_numpy_array_equal(result, expected)

            result = idx1 != idx2
            expected = np.array([True, True, True, True, True, False])
            tm.assert_numpy_array_equal(result, expected)

    # TODO: better name
    def test_comparisons_coverage(self):
        rng = timedelta_range("1 days", periods=10)

        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)

        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Arithmetic Operations


class TestTimedelta64ArithmeticUnsorted:
    # Tests moved from type-specific test files but not
    #  yet sorted/parametrized/de-duplicated

    def test_ufunc_coercions(self):
        # normal ops are also tested in tseries/test_timedeltas.py
        idx = TimedeltaIndex(["2H", "4H", "6H", "8H", "10H"], freq="2H", name="x")

        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["4H", "8H", "12H", "16H", "20H"], freq="4H", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "4H"

        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["1H", "2H", "3H", "4H", "5H"], freq="H", name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq == "H"

        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(
                ["-2H", "-4H", "-6H", "-8H", "-10H"], freq="-2H", name="x"
            )
            tm.assert_index_equal(result, exp)
            assert result.freq == "-2H"

        idx = TimedeltaIndex(["-2H", "-1H", "0H", "1H", "2H"], freq="H", name="x")
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(["2H", "1H", "0H", "1H", "2H"], freq=None, name="x")
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self):
        # with datetimes/timedelta and tdi/dti
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")

        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        msg = r"unsupported operand type\(s\) for -"
        with pytest.raises(TypeError, match=msg):
            td - dt

        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            td - dti

        result = dt - dti
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"], name="bar")
        tm.assert_index_equal(result, expected)

        result = dti - dt
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"], name="bar")
        tm.assert_index_equal(result, expected)

        result = tdi - td
        expected = TimedeltaIndex(["0 days", NaT, "1 days"], name="foo")
        tm.assert_index_equal(result, expected, check_names=False)

        result = td - tdi
        expected = TimedeltaIndex(["0 days", NaT, "-1 days"], name="foo")
        tm.assert_index_equal(result, expected, check_names=False)

        result = dti - td
        expected = DatetimeIndex(
            ["20121231", "20130101", "20130102"], freq="D", name="bar"
        )
        tm.assert_index_equal(result, expected, check_names=False)

        result = dt - tdi
        expected = DatetimeIndex(["20121231", NaT, "20121230"], name="foo")
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self):

        # check that dt/dti subtraction ops with tz are validated
        dti = pd.date_range("20130101", periods=3)
        ts = Timestamp("20130101")
        dt = ts.to_pydatetime()
        dti_tz = pd.date_range("20130101", periods=3).tz_localize("US/Eastern")
        ts_tz = Timestamp("20130101").tz_localize("US/Eastern")
        ts_tz2 = Timestamp("20130101").tz_localize("CET")
        dt_tz = ts_tz.to_pydatetime()
        td = Timedelta("1 days")

        def _check(result, expected):
            assert result == expected
            assert isinstance(result, Timedelta)

        # scalars
        result = ts - ts
        expected = Timedelta("0 days")
        _check(result, expected)

        result = dt_tz - ts_tz
        expected = Timedelta("0 days")
        _check(result, expected)

        result = ts_tz - dt_tz
        expected = Timedelta("0 days")
        _check(result, expected)

        # tz mismatches
        msg = "Timestamp subtraction must have the same timezones or no timezones"
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg = "Timestamp subtraction must have the same timezones or no timezones"
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts_tz2
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg = "Timestamp subtraction must have the same timezones or no timezones"
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt
        with pytest.raises(TypeError, match=msg):
            ts_tz - ts_tz2

        # with dti
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts_tz2

        result = dti_tz - dt_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        tm.assert_index_equal(result, expected)

        result = dt_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        tm.assert_index_equal(result, expected)

        result = dti_tz - ts_tz
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"])
        tm.assert_index_equal(result, expected)

        result = ts_tz - dti_tz
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"])
        tm.assert_index_equal(result, expected)

        result = td - td
        expected = Timedelta("0 days")
        _check(result, expected)

        result = dti_tz - td
        expected = DatetimeIndex(["20121231", "20130101", "20130102"], tz="US/Eastern")
        tm.assert_index_equal(result, expected)

    def test_dti_tdi_numeric_ops(self):
        # These are normally union/diff set-like ops
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")

        result = tdi - tdi
        expected = TimedeltaIndex(["0 days", NaT, "0 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = tdi + tdi
        expected = TimedeltaIndex(["2 days", NaT, "4 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = dti - tdi  # name will be reset
        expected = DatetimeIndex(["20121231", NaT, "20130101"])
        tm.assert_index_equal(result, expected)

    def test_addition_ops(self):
        # with datetimes/timedelta and tdi/dti
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        dti = pd.date_range("20130101", periods=3, name="bar")
        td = Timedelta("1 days")
        dt = Timestamp("20130101")

        result = tdi + dt
        expected = DatetimeIndex(["20130102", NaT, "20130103"], name="foo")
        tm.assert_index_equal(result, expected)

        result = dt + tdi
        expected = DatetimeIndex(["20130102", NaT, "20130103"], name="foo")
        tm.assert_index_equal(result, expected)

        result = td + tdi
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)

        result = tdi + td
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        tm.assert_index_equal(result, expected)

        # unequal length
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti

        # random indexes
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            tdi + pd.Int64Index([1, 2, 3])

        # this is a union!
        # pytest.raises(TypeError, lambda : pd.Int64Index([1,2,3]) + tdi)

        result = tdi + dti  # name will be reset
        expected = DatetimeIndex(["20130102", NaT, "20130105"])
        tm.assert_index_equal(result, expected)

        result = dti + tdi  # name will be reset
        expected = DatetimeIndex(["20130102", NaT, "20130105"])
        tm.assert_index_equal(result, expected)

        result = dt + td
        expected = Timestamp("20130102")
        assert result == expected

        result = td + dt
        expected = Timestamp("20130102")
        assert result == expected

    # TODO: Needs more informative name, probably split up into
    # more targeted tests
    @pytest.mark.parametrize("freq", ["D", "B"])
    def test_timedelta(self, freq):
        index = pd.date_range("1/1/2000", periods=50, freq=freq)

        shifted = index + timedelta(1)
        back = shifted + timedelta(-1)
        back = back._with_freq("infer")
        tm.assert_index_equal(index, back)

        if freq == "D":
            expected = pd.tseries.offsets.Day(1)
            assert index.freq == expected
            assert shifted.freq == expected
            assert back.freq == expected
        else:  # freq == 'B'
            assert index.freq == pd.tseries.offsets.BusinessDay(1)
            assert shifted.freq is None
            assert back.freq == pd.tseries.offsets.BusinessDay(1)

        result = index - timedelta(1)
        expected = index + timedelta(-1)
        tm.assert_index_equal(result, expected)

        # GH#4134, buggy with timedeltas
        rng = pd.date_range("2013", "2014")
        s = Series(rng)
        result1 = rng - offsets.Hour(1)
        result2 = DatetimeIndex(s - np.timedelta64(100000000))
        result3 = rng - np.timedelta64(100000000)
        result4 = DatetimeIndex(s - offsets.Hour(1))

        assert result1.freq == rng.freq
        result1 = result1._with_freq(None)
        tm.assert_index_equal(result1, result4)

        assert result3.freq == rng.freq
        result3 = result3._with_freq(None)
        tm.assert_index_equal(result2, result3)

    def test_tda_add_sub_index(self):
        # Check that TimedeltaArray defers to Index on arithmetic ops
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"])
        tda = tdi.array

        dti = pd.date_range("1999-12-31", periods=3, freq="D")

        result = tda + dti
        expected = tdi + dti
        tm.assert_index_equal(result, expected)

        result = tda + tdi
        expected = tdi + tdi
        tm.assert_index_equal(result, expected)

        result = tda - tdi
        expected = tdi - tdi
        tm.assert_index_equal(result, expected)

    def test_tda_add_dt64_object_array(self, box_with_array, tz_naive_fixture):
        # Result should be cast back to DatetimeArray
        box = box_with_array

        dti = pd.date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        dti = dti._with_freq(None)
        tdi = dti - dti

        obj = tm.box_expected(tdi, box)
        other = tm.box_expected(dti, box)

        with tm.assert_produces_warning(PerformanceWarning):
            result = obj + other.astype(object)
        tm.assert_equal(result, other)

    # -------------------------------------------------------------
    # Binary operations TimedeltaIndex and timedelta-like

    def test_tdi_iadd_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as + is now numeric
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        orig_rng = rng
        rng += two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not pd.Index:
            # Check that operation is actually inplace
            tm.assert_equal(orig_rng, expected)

    def test_tdi_isub_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as - is now numeric
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        orig_rng = rng
        rng -= two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not pd.Index:
            # Check that operation is actually inplace
            tm.assert_equal(orig_rng, expected)

    # -------------------------------------------------------------

    def test_tdi_ops_attributes(self):
        rng = timedelta_range("2 days", periods=5, freq="2D", name="x")

        result = rng + 1 * rng.freq
        exp = timedelta_range("4 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"

        result = rng - 2 * rng.freq
        exp = timedelta_range("-2 days", periods=5, freq="2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "2D"

        result = rng * 2
        exp = timedelta_range("4 days", periods=5, freq="4D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "4D"

        result = rng / 2
        exp = timedelta_range("1 days", periods=5, freq="D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "D"

        result = -rng
        exp = timedelta_range("-2 days", periods=5, freq="-2D", name="x")
        tm.assert_index_equal(result, exp)
        assert result.freq == "-2D"

        rng = timedelta_range("-2 days", periods=5, freq="D", name="x")

        result = abs(rng)
        exp = TimedeltaIndex(
            ["2 days", "1 days", "0 days", "1 days", "2 days"], name="x"
        )
        tm.assert_index_equal(result, exp)
        assert result.freq is None


class TestAddSubNaTMasking:
    # TODO: parametrize over boxes

    def test_tdi_add_timestamp_nat_masking(self):
        # GH#17991 checking for overflow-masking with NaT
        tdinat = pd.to_timedelta(["24658 days 11:15:00", "NaT"])

        tsneg = Timestamp("1950-01-01")
        ts_neg_variants = [
            tsneg,
            tsneg.to_pydatetime(),
            tsneg.to_datetime64().astype("datetime64[ns]"),
            tsneg.to_datetime64().astype("datetime64[D]"),
        ]

        tspos = Timestamp("1980-01-01")
        ts_pos_variants = [
            tspos,
            tspos.to_pydatetime(),
            tspos.to_datetime64().astype("datetime64[ns]"),
            tspos.to_datetime64().astype("datetime64[D]"),
        ]

        for variant in ts_neg_variants + ts_pos_variants:
            res = tdinat + variant
            assert res[1] is NaT

    def test_tdi_add_overflow(self):
        # See GH#14068
        # preliminary test scalar analogue of vectorized tests below
        # TODO: Make raised error message more informative and test
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            pd.to_timedelta(106580, "D") + Timestamp("2000")
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            Timestamp("2000") + pd.to_timedelta(106580, "D")

        _NaT = NaT.value + 1
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([106580], "D") + Timestamp("2000")
        with pytest.raises(OverflowError, match=msg):
            Timestamp("2000") + pd.to_timedelta([106580], "D")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta(["5 days", _NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            (
                pd.to_timedelta([_NaT, "5 days", "1 hours"])
                - pd.to_timedelta(["7 seconds", _NaT, "4 hours"])
            )

        # These should not overflow!
        exp = TimedeltaIndex([NaT])
        result = pd.to_timedelta([NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex(["4 days", NaT])
        result = pd.to_timedelta(["5 days", NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex([NaT, NaT, "5 hours"])
        result = pd.to_timedelta([NaT, "5 days", "1 hours"]) + pd.to_timedelta(
            ["7 seconds", NaT, "4 hours"]
        )
        tm.assert_index_equal(result, exp)


class TestTimedeltaArraylikeAddSubOps:
    # Tests for timedelta64[ns] __add__, __sub__, __radd__, __rsub__

    # TODO: moved from tests.indexes.timedeltas.test_arithmetic; needs
    #  parametrization+de-duplication
    def test_timedelta_ops_with_missing_values(self):
        # setup
        s1 = pd.to_timedelta(Series(["00:00:01"]))
        s2 = pd.to_timedelta(Series(["00:00:02"]))

        msg = r"dtype datetime64\[ns\] cannot be converted to timedelta64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            # Passing datetime64-dtype data to TimedeltaIndex is no longer
            #  supported GH#29794
            pd.to_timedelta(Series([NaT]))

        sn = pd.to_timedelta(Series([NaT], dtype="m8[ns]"))

        df1 = DataFrame(["00:00:01"]).apply(pd.to_timedelta)
        df2 = DataFrame(["00:00:02"]).apply(pd.to_timedelta)
        with pytest.raises(TypeError, match=msg):
            # Passing datetime64-dtype data to TimedeltaIndex is no longer
            #  supported GH#29794
            DataFrame([NaT]).apply(pd.to_timedelta)

        dfn = DataFrame([NaT.value]).apply(pd.to_timedelta)

        scalar1 = pd.to_timedelta("00:00:01")
        scalar2 = pd.to_timedelta("00:00:02")
        timedelta_NaT = pd.to_timedelta("NaT")

        actual = scalar1 + scalar1
        assert actual == scalar2
        actual = scalar2 - scalar1
        assert actual == scalar1

        actual = s1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - s1
        tm.assert_series_equal(actual, s1)

        actual = s1 + scalar1
        tm.assert_series_equal(actual, s2)
        actual = scalar1 + s1
        tm.assert_series_equal(actual, s2)
        actual = s2 - scalar1
        tm.assert_series_equal(actual, s1)
        actual = -scalar1 + s2
        tm.assert_series_equal(actual, s1)

        actual = s1 + timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        actual = s1 - timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual = -timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            s1 + np.nan
        with pytest.raises(TypeError, match=msg):
            np.nan + s1
        with pytest.raises(TypeError, match=msg):
            s1 - np.nan
        with pytest.raises(TypeError, match=msg):
            -np.nan + s1

        actual = s1 + NaT
        tm.assert_series_equal(actual, sn)
        actual = s2 - NaT
        tm.assert_series_equal(actual, sn)

        actual = s1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = s2 - df1
        tm.assert_frame_equal(actual, df1)
        actual = df1 + s1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - s1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + df1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - df1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + scalar1
        tm.assert_frame_equal(actual, df2)
        actual = df2 - scalar1
        tm.assert_frame_equal(actual, df1)

        actual = df1 + timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - timedelta_NaT
        tm.assert_frame_equal(actual, dfn)

        msg = "cannot subtract a datelike from|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            df1 + np.nan
        with pytest.raises(TypeError, match=msg):
            df1 - np.nan

        actual = df1 + NaT  # NaT is datetime, not timedelta
        tm.assert_frame_equal(actual, dfn)
        actual = df1 - NaT
        tm.assert_frame_equal(actual, dfn)

    # TODO: moved from tests.series.test_operators, needs splitting, cleanup,
    # de-duplication, box-parametrization...
    def test_operators_timedelta64(self):
        # series ops
        v1 = pd.date_range("2012-1-1", periods=3, freq="D")
        v2 = pd.date_range("2012-1-2", periods=3, freq="D")
        rs = Series(v2) - Series(v1)
        xp = Series(1e9 * 3600 * 24, rs.index).astype("int64").astype("timedelta64[ns]")
        tm.assert_series_equal(rs, xp)
        assert rs.dtype == "timedelta64[ns]"

        df = DataFrame({"A": v1})
        td = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == "timedelta64[ns]"

        # series on the rhs
        result = df["A"] - df["A"].shift()
        assert result.dtype == "timedelta64[ns]"

        result = df["A"] + td
        assert result.dtype == "M8[ns]"

        # scalar Timestamp on rhs
        maxa = df["A"].max()
        assert isinstance(maxa, Timestamp)

        resultb = df["A"] - df["A"].max()
        assert resultb.dtype == "timedelta64[ns]"

        # timestamp on lhs
        result = resultb + df["A"]
        values = [Timestamp("20111230"), Timestamp("20120101"), Timestamp("20120103")]
        expected = Series(values, name="A")
        tm.assert_series_equal(result, expected)

        # datetimes on rhs
        result = df["A"] - datetime(2001, 1, 1)
        expected = Series([timedelta(days=4017 + i) for i in range(3)], name="A")
        tm.assert_series_equal(result, expected)
        assert result.dtype == "m8[ns]"

        d = datetime(2001, 1, 1, 3, 4)
        resulta = df["A"] - d
        assert resulta.dtype == "m8[ns]"

        # roundtrip
        resultb = resulta + d
        tm.assert_series_equal(df["A"], resultb)

        # timedeltas on rhs
        td = timedelta(days=1)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(resultb, df["A"])
        assert resultb.dtype == "M8[ns]"

        # roundtrip
        td = timedelta(minutes=5, seconds=3)
        resulta = df["A"] + td
        resultb = resulta - td
        tm.assert_series_equal(df["A"], resultb)
        assert resultb.dtype == "M8[ns]"

        # inplace
        value = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        assert rs[2] == value

    def test_timedelta64_ops_nat(self):
        # GH 11349
        timedelta_series = Series([NaT, Timedelta("1s")])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[ns]")
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[ns]")

        # subtraction
        tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)

        tm.assert_series_equal(
            timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta
        )

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_timedelta,
            nat_series_dtype_timedelta,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timedelta,
            nat_series_dtype_timedelta,
        )

        tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)

        tm.assert_series_equal(
            timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_timedelta,
            nat_series_dtype_timedelta,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timedelta,
            nat_series_dtype_timedelta,
        )

        # multiplication
        tm.assert_series_equal(
            nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta
        )
        tm.assert_series_equal(
            1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(timedelta_series * 1, timedelta_series)
        tm.assert_series_equal(1 * timedelta_series, timedelta_series)

        tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta("1.5s")]))
        tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta("1.5s")]))

        tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
        tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)

        # division
        tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta("0.5s")]))
        tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)

    # -------------------------------------------------------------
    # Binary operations td64 arraylike and datetime-like

    def test_td64arr_sub_timestamp_raises(self, box_with_array):
        idx = TimedeltaIndex(["1 day", "2 day"])
        idx = tm.box_expected(idx, box_with_array)

        msg = (
            "cannot subtract a datelike from|"
            "Could not operate|"
            "cannot perform operation"
        )
        with pytest.raises(TypeError, match=msg):
            idx - Timestamp("2011-01-01")

    def test_td64arr_add_timestamp(self, box_with_array, tz_naive_fixture):
        # GH#23215

        # TODO: parametrize over scalar datetime types?
        tz = tz_naive_fixture
        other = Timestamp("2011-01-01", tz=tz)

        idx = TimedeltaIndex(["1 day", "2 day"])
        expected = DatetimeIndex(["2011-01-02", "2011-01-03"], tz=tz)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx + other
        tm.assert_equal(result, expected)

        result = other + idx
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2012-01-01"),
            Timestamp("2012-01-01").to_pydatetime(),
            Timestamp("2012-01-01").to_datetime64(),
        ],
    )
    def test_td64arr_add_sub_datetimelike_scalar(self, ts, box_with_array):
        # GH#11925, GH#29558
        tdi = timedelta_range("1 day", periods=3)
        expected = pd.date_range("2012-01-02", periods=3)

        tdarr = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        tm.assert_equal(ts + tdarr, expected)
        tm.assert_equal(tdarr + ts, expected)

        expected2 = pd.date_range("2011-12-31", periods=3, freq="-1D")
        expected2 = tm.box_expected(expected2, box_with_array)

        tm.assert_equal(ts - tdarr, expected2)
        tm.assert_equal(ts + (-tdarr), expected2)

        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    def test_tdi_sub_dt64_array(self, box_with_array):
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) - tdi

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        msg = "cannot subtract a datelike from"
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr

        # TimedeltaIndex.__rsub__
        result = dtarr - tdi
        tm.assert_equal(result, expected)

    def test_tdi_add_dt64_array(self, box_with_array):
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        dtarr = dti.values
        expected = DatetimeIndex(dtarr) + tdi

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdi + dtarr
        tm.assert_equal(result, expected)
        result = dtarr + tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_datetime64_nat(self, box_with_array):
        # GH#23215
        other = np.datetime64("NaT")

        tdi = timedelta_range("1 day", periods=3)
        expected = DatetimeIndex(["NaT", "NaT", "NaT"])

        tdser = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        tm.assert_equal(tdser + other, expected)
        tm.assert_equal(other + tdser, expected)

    # ------------------------------------------------------------------
    # Invalid __add__/__sub__ operations

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "H"])
    @pytest.mark.parametrize("tdi_freq", [None, "H"])
    def test_td64arr_sub_periodlike(self, box_with_array, tdi_freq, pi_freq):
        # GH#20049 subtracting PeriodIndex should raise TypeError
        tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
        dti = Timestamp("2018-03-07 17:16:40") + tdi
        pi = dti.to_period(pi_freq)

        # TODO: parametrize over box for pi?
        tdi = tm.box_expected(tdi, box_with_array)
        msg = "cannot subtract|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            tdi - pi

        # GH#13078 subtraction of Period scalar not supported
        with pytest.raises(TypeError, match=msg):
            tdi - pi[0]

    @pytest.mark.parametrize(
        "other",
        [
            # GH#12624 for str case
            "a",
            # GH#19123
            1,
            1.5,
            np.array(2),
        ],
    )
    def test_td64arr_addsub_numeric_scalar_invalid(self, box_with_array, other):
        # vector-like others are tested in test_td64arr_add_sub_numeric_arr_invalid
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)

        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize(
        "vec",
        [
            np.array([1, 2, 3]),
            pd.Index([1, 2, 3]),
            Series([1, 2, 3]),
            DataFrame([[1, 2, 3]]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_addsub_numeric_arr_invalid(
        self, box_with_array, vec, any_real_dtype
    ):
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        tdarr = tm.box_expected(tdser, box_with_array)

        vector = vec.astype(any_real_dtype)
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(self, box_with_array, one):
        # Variants of `one` for #19012, deprecated GH#22535
        rng = timedelta_range("1 days 09:00:00", freq="H", periods=10)
        tdarr = tm.box_expected(rng, box_with_array)

        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, one, msg)

        # TODO: get inplace ops into assert_invalid_addsub_type
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array):
        # GH#19959, deprecated GH#22535
        # GH#22696 for DataFrame case, check that we don't dispatch to numpy
        #  implementation, which treats int64 as m8[ns]
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days 09:00:00", freq="H", periods=3)
        tdarr = tm.box_expected(rng, box)
        other = tm.box_expected([4, 3, 2], xbox)

        msg = "Addition/subtraction of integers and integer-arrays"
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_addsub_integer_array_no_freq(self, box_with_array):
        # GH#19959
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = TimedeltaIndex(["1 Day", "NaT", "3 Hours"])
        tdarr = tm.box_expected(tdi, box)
        other = tm.box_expected([14, -1, 16], xbox)

        msg = "Addition/subtraction of integers"
        assert_invalid_addsub_type(tdarr, other, msg)

    # ------------------------------------------------------------------
    # Operations with timedelta-like others

    def test_td64arr_add_td64_array(self, box_with_array):
        box = box_with_array
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        tdarr = tdi.values

        expected = 2 * tdi
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = tdi + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + tdi
        tm.assert_equal(result, expected)

    def test_td64arr_sub_td64_array(self, box_with_array):
        box = box_with_array
        dti = pd.date_range("2016-01-01", periods=3)
        tdi = dti - dti.shift(1)
        tdarr = tdi.values

        expected = 0 * tdi
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = tdi - tdarr
        tm.assert_equal(result, expected)
        result = tdarr - tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_sub_tdi(self, box_with_array, names):
        # GH#17250 make sure result dtype is correct
        # GH#19043 make sure names are propagated correctly
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["0 days", "1 day"], name=names[1])
        tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
        ser = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
        expected = Series([Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname)

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        result = tdi + ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        result = ser + tdi
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        expected = Series(
            [Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname
        )
        expected = tm.box_expected(expected, box)

        result = tdi - ser
        tm.assert_equal(result, expected)
        assert_dtype(result, "timedelta64[ns]")

        result = ser - tdi
        tm.assert_equal(result, -expected)
        assert_dtype(result, "timedelta64[ns]")

    def test_td64arr_add_sub_td64_nat(self, box_with_array):
        # GH#23320 special handling for timedelta64("NaT")
        box = box_with_array
        tdi = TimedeltaIndex([NaT, Timedelta("1s")])
        other = np.timedelta64("NaT")
        expected = TimedeltaIndex(["NaT"] * 2)

        obj = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        result = other - obj
        tm.assert_equal(result, expected)

    def test_td64arr_sub_NaT(self, box_with_array):
        # GH#18808
        box = box_with_array
        ser = Series([NaT, Timedelta("1s")])
        expected = Series([NaT, NaT], dtype="timedelta64[ns]")

        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        res = ser - NaT
        tm.assert_equal(res, expected)

    def test_td64arr_add_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as + is now numeric
        # GH#10699 for Tick cases
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)

        result = rng + two_hours
        tm.assert_equal(result, expected)

        result = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as - is now numeric
        # GH#10699 for Tick cases
        box = box_with_array
        rng = timedelta_range("1 days", "10 days")
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, box)

        result = rng - two_hours
        tm.assert_equal(result, expected)

        result = two_hours - rng
        tm.assert_equal(result, -expected)

    # ------------------------------------------------------------------
    # __add__/__sub__ with DateOffsets and arrays of DateOffsets

    def test_td64arr_add_offset_index(self, names, box_with_array):
        # GH#18849, GH#19744
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = pd.Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        other = np.array(other) if box in [tm.to_array, pd.array] else other

        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer", name=exname
        )
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi + other
        tm.assert_equal(res, expected)

        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)

    # TODO: combine with test_td64arr_add_offset_index by parametrizing
    # over second box?
    def test_td64arr_add_offset_array(self, box_with_array):
        # GH#18849
        box = box_with_array
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])

        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer"
        )

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi + other
        tm.assert_equal(res, expected)

        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + tdi
        tm.assert_equal(res2, expected)

    def test_td64arr_sub_offset_index(self, names, box_with_array):
        # GH#18824, GH#19744
        box = box_with_array
        xbox = box if box not in [tm.to_array, pd.array] else pd.Index
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = pd.Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])

        expected = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer", name=exname
        )

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, xbox)

        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi - other
        tm.assert_equal(res, expected)

    def test_td64arr_sub_offset_array(self, box_with_array):
        # GH#18824
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])

        expected = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer"
        )

        tdi = tm.box_expected(tdi, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        with tm.assert_produces_warning(PerformanceWarning):
            res = tdi - other
        tm.assert_equal(res, expected)

    def test_td64arr_with_offset_series(self, names, box_with_array):
        # GH#18849
        box = box_with_array
        box2 = Series if box in [pd.Index, tm.to_array, pd.array] else box
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])

        expected_add = Series([tdi[n] + other[n] for n in range(len(tdi))], name=exname)
        obj = tm.box_expected(tdi, box)
        expected_add = tm.box_expected(expected_add, box2)

        with tm.assert_produces_warning(PerformanceWarning):
            res = obj + other
        tm.assert_equal(res, expected_add)

        with tm.assert_produces_warning(PerformanceWarning):
            res2 = other + obj
        tm.assert_equal(res2, expected_add)

        expected_sub = Series([tdi[n] - other[n] for n in range(len(tdi))], name=exname)
        expected_sub = tm.box_expected(expected_sub, box2)

        with tm.assert_produces_warning(PerformanceWarning):
            res3 = obj - other
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize("obox", [np.array, pd.Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(self, obox, box_with_array):
        # GH#18824
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        tdi = tm.box_expected(tdi, box_with_array)

        anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])

        # addition/subtraction ops with anchored offsets should issue
        # a PerformanceWarning and _then_ raise a TypeError.
        msg = "has incorrect type|cannot add the type MonthEnd"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                anchored - tdi

    # ------------------------------------------------------------------
    # Unsorted

    def test_td64arr_add_sub_object_array(self, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = timedelta_range("1 day", periods=3, freq="D")
        tdarr = tm.box_expected(tdi, box)

        other = np.array([Timedelta(days=1), offsets.Day(2), Timestamp("2000-01-04")])

        with tm.assert_produces_warning(PerformanceWarning):
            result = tdarr + other

        expected = pd.Index(
            [Timedelta(days=2), Timedelta(days=4), Timestamp("2000-01-07")]
        )
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

        msg = "unsupported operand type|cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(PerformanceWarning):
                tdarr - other

        with tm.assert_produces_warning(PerformanceWarning):
            result = other - tdarr

        expected = pd.Index([Timedelta(0), Timedelta(0), Timestamp("2000-01-01")])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)


class TestTimedeltaArraylikeMulDivOps:
    # Tests for timedelta64[ns]
    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__

    # ------------------------------------------------------------------
    # Multiplication
    # organized with scalar others first, then array-like

    def test_td64arr_mul_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)

        result = idx * 1
        tm.assert_equal(result, idx)

        result = 1 * idx
        tm.assert_equal(result, idx)

    def test_td64arr_mul_tdlike_scalar_raises(self, two_hours, box_with_array):
        rng = timedelta_range("1 days", "10 days", name="foo")
        rng = tm.box_expected(rng, box_with_array)
        msg = "argument must be an integer|cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            rng * two_hours

    def test_tdi_mul_int_array_zerodim(self, box_with_array):
        rng5 = np.arange(5, dtype="int64")
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5 * 5)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx * np.array(5, dtype="int64")
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_array(self, box_with_array):
        rng5 = np.arange(5, dtype="int64")
        idx = TimedeltaIndex(rng5)
        expected = TimedeltaIndex(rng5 ** 2)

        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = idx * rng5
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [pd.Index, tm.to_array, pd.array] else box

        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        expected = TimedeltaIndex(np.arange(5, dtype="int64") ** 2)

        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, xbox)

        result = idx * Series(np.arange(5, dtype="int64"))
        tm.assert_equal(result, expected)

    def test_tdi_mul_float_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [pd.Index, tm.to_array, pd.array] else box

        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box)

        rng5f = np.arange(5, dtype="float64")
        expected = TimedeltaIndex(rng5f * (rng5f + 1.0))
        expected = tm.box_expected(expected, xbox)

        result = idx * Series(rng5f + 1.0)
        tm.assert_equal(result, expected)

    # TODO: Put Series/DataFrame in others?
    @pytest.mark.parametrize(
        "other",
        [
            np.arange(1, 11),
            pd.Int64Index(range(1, 11)),
            pd.UInt64Index(range(1, 11)),
            pd.Float64Index(range(1, 11)),
            pd.RangeIndex(1, 11),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_tdi_rmul_arraylike(self, other, box_with_array):
        box = box_with_array
        xbox = get_upcast_box(box, other)

        tdi = TimedeltaIndex(["1 Day"] * 10)
        expected = timedelta_range("1 days", "10 days")
        expected._data.freq = None

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, xbox)

        result = other * tdi
        tm.assert_equal(result, expected)
        commute = tdi * other
        tm.assert_equal(commute, expected)

    # ------------------------------------------------------------------
    # __div__, __rdiv__

    def test_td64arr_div_nat_invalid(self, box_with_array):
        # don't allow division by NaT (maybe could in the future)
        rng = timedelta_range("1 days", "10 days", name="foo")
        rng = tm.box_expected(rng, box_with_array)

        with pytest.raises(TypeError, match="unsupported operand type"):
            rng / NaT
        with pytest.raises(TypeError, match="Cannot divide NaTType by"):
            NaT / rng

    def test_td64arr_div_td64nat(self, box_with_array):
        # GH#23829
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days")
        rng = tm.box_expected(rng, box)

        other = np.timedelta64("NaT")

        expected = np.array([np.nan] * 10)
        expected = tm.box_expected(expected, xbox)

        result = rng / other
        tm.assert_equal(result, expected)

        result = other / rng
        tm.assert_equal(result, expected)

    def test_td64arr_div_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)

        result = idx / 1
        tm.assert_equal(result, idx)

        with pytest.raises(TypeError, match="Cannot divide"):
            # GH#23829
            1 / idx

    def test_td64arr_div_tdlike_scalar(self, two_hours, box_with_array):
        # GH#20088, GH#22163 ensure DataFrame returns correct dtype
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days", name="foo")
        expected = pd.Float64Index((np.arange(10) + 1) * 12, name="foo")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        result = rng / two_hours
        tm.assert_equal(result, expected)

        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("m", [1, 3, 10])
    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_td64arr_div_td64_scalar(self, m, unit, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        startdate = Series(pd.date_range("2013-01-01", "2013-01-03"))
        enddate = Series(pd.date_range("2013-03-01", "2013-03-03"))

        ser = enddate - startdate
        ser[2] = np.nan
        flat = ser
        ser = tm.box_expected(ser, box)

        # op
        expected = Series([x / np.timedelta64(m, unit) for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = ser / np.timedelta64(m, unit)
        tm.assert_equal(result, expected)

        # reverse op
        expected = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
        expected = tm.box_expected(expected, xbox)
        result = np.timedelta64(m, unit) / ser
        tm.assert_equal(result, expected)

    def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        expected = pd.Float64Index([12, np.nan, 24], name="foo")

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        result = rng / two_hours
        tm.assert_equal(result, expected)

        result = two_hours / rng
        expected = 1 / expected
        tm.assert_equal(result, expected)

    def test_td64arr_div_td64_ndarray(self, box_with_array):
        # GH#22631
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        expected = pd.Float64Index([12, np.nan, 24])

        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        other = np.array([2, 4, 2], dtype="m8[h]")
        result = rng / other
        tm.assert_equal(result, expected)

        result = rng / tm.box_expected(other, box)
        tm.assert_equal(result, expected)

        result = rng / other.astype(object)
        tm.assert_equal(result, expected)

        result = rng / list(other)
        tm.assert_equal(result, expected)

        # reversed op
        expected = 1 / expected
        result = other / rng
        tm.assert_equal(result, expected)

        result = tm.box_expected(other, box) / rng
        tm.assert_equal(result, expected)

        result = other.astype(object) / rng
        tm.assert_equal(result, expected)

        result = list(other) / rng
        tm.assert_equal(result, expected)

    def test_tdarr_div_length_mismatch(self, box_with_array):
        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        mismatched = [1, 2, 3, 4]

        rng = tm.box_expected(rng, box_with_array)
        msg = "Cannot divide vectors|Unable to coerce to Series"
        for obj in [mismatched, mismatched[:2]]:
            # one shorter, one longer
            for other in [obj, np.array(obj), pd.Index(obj)]:
                with pytest.raises(ValueError, match=msg):
                    rng / other
                with pytest.raises(ValueError, match=msg):
                    other / rng

    # ------------------------------------------------------------------
    # __floordiv__, __rfloordiv__

    def test_td64arr_floordiv_td64arr_with_nat(
        self, box_with_array, using_array_manager
    ):
        # GH#35529
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        left = Series([1000, 222330, 30], dtype="timedelta64[ns]")
        right = Series([1000, 222330, None], dtype="timedelta64[ns]")

        left = tm.box_expected(left, box)
        right = tm.box_expected(right, box)

        expected = np.array([1.0, 1.0, np.nan], dtype=np.float64)
        expected = tm.box_expected(expected, xbox)
        if box is DataFrame and using_array_manager:
            # INFO(ArrayManager) floorfiv returns integer, and ArrayManager
            # performs ops column-wise and thus preserves int64 dtype for
            # columns without missing values
            expected[[0, 1]] = expected[[0, 1]].astype("int64")

        result = left // right

        tm.assert_equal(result, expected)

        # case that goes through __rfloordiv__ with arraylike
        result = np.asarray(left) // right
        tm.assert_equal(result, expected)

    def test_td64arr_floordiv_tdscalar(self, box_with_array, scalar_td):
        # GH#18831
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        expected = Series([0, 0, np.nan])

        td1 = tm.box_expected(td1, box, transpose=False)
        expected = tm.box_expected(expected, xbox, transpose=False)

        result = td1 // scalar_td
        tm.assert_equal(result, expected)

    def test_td64arr_rfloordiv_tdscalar(self, box_with_array, scalar_td):
        # GH#18831
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        expected = Series([1, 1, np.nan])

        td1 = tm.box_expected(td1, box, transpose=False)
        expected = tm.box_expected(expected, xbox, transpose=False)

        result = scalar_td // td1
        tm.assert_equal(result, expected)

    def test_td64arr_rfloordiv_tdscalar_explicit(self, box_with_array, scalar_td):
        # GH#18831
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        expected = Series([1, 1, np.nan])

        td1 = tm.box_expected(td1, box, transpose=False)
        expected = tm.box_expected(expected, xbox, transpose=False)

        # We can test __rfloordiv__ using this syntax,
        # see `test_timedelta_rfloordiv`
        result = td1.__rfloordiv__(scalar_td)
        tm.assert_equal(result, expected)

    def test_td64arr_floordiv_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        result = idx // 1
        tm.assert_equal(result, idx)

        pattern = "floor_divide cannot use operands|Cannot divide int by Timedelta*"
        with pytest.raises(TypeError, match=pattern):
            1 // idx

    def test_td64arr_floordiv_tdlike_scalar(self, two_hours, box_with_array):
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        tdi = timedelta_range("1 days", "10 days", name="foo")
        expected = pd.Int64Index((np.arange(10) + 1) * 12, name="foo")

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, xbox)

        result = tdi // two_hours
        tm.assert_equal(result, expected)

    # TODO: Is this redundant with test_td64arr_floordiv_tdlike_scalar?
    @pytest.mark.parametrize(
        "scalar_td",
        [
            timedelta(minutes=10, seconds=7),
            Timedelta("10m7s"),
            Timedelta("10m7s").to_timedelta64(),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_rfloordiv_tdlike_scalar(self, scalar_td, box_with_array):
        # GH#19125
        box = box_with_array
        xbox = np.ndarray if box_with_array is pd.array else box_with_array

        tdi = TimedeltaIndex(["00:05:03", "00:05:03", NaT], freq=None)
        expected = pd.Index([2.0, 2.0, np.nan])

        tdi = tm.box_expected(tdi, box, transpose=False)
        expected = tm.box_expected(expected, xbox, transpose=False)

        res = tdi.__rfloordiv__(scalar_td)
        tm.assert_equal(res, expected)

        expected = pd.Index([0.0, 0.0, np.nan])
        expected = tm.box_expected(expected, xbox, transpose=False)

        res = tdi // (scalar_td)
        tm.assert_equal(res, expected)

    # ------------------------------------------------------------------
    # mod, divmod
    # TODO: operations with timedelta-like arrays, numeric arrays,
    #  reversed ops

    def test_td64arr_mod_tdscalar(self, box_with_array, three_days):
        tdi = timedelta_range("1 Day", "9 days")
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = TimedeltaIndex(["1 Day", "2 Days", "0 Days"] * 3)
        expected = tm.box_expected(expected, box_with_array)

        result = tdarr % three_days
        tm.assert_equal(result, expected)

        warn = None
        if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
            warn = PerformanceWarning

        with tm.assert_produces_warning(warn):
            result = divmod(tdarr, three_days)

        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // three_days)

    def test_td64arr_mod_int(self, box_with_array):
        tdi = timedelta_range("1 ns", "10 ns", periods=10)
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = TimedeltaIndex(["1 ns", "0 ns"] * 5)
        expected = tm.box_expected(expected, box_with_array)

        result = tdarr % 2
        tm.assert_equal(result, expected)

        msg = "Cannot divide int by"
        with pytest.raises(TypeError, match=msg):
            2 % tdarr

        result = divmod(tdarr, 2)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // 2)

    def test_td64arr_rmod_tdscalar(self, box_with_array, three_days):
        tdi = timedelta_range("1 Day", "9 days")
        tdarr = tm.box_expected(tdi, box_with_array)

        expected = ["0 Days", "1 Day", "0 Days"] + ["3 Days"] * 6
        expected = TimedeltaIndex(expected)
        expected = tm.box_expected(expected, box_with_array)

        result = three_days % tdarr
        tm.assert_equal(result, expected)

        result = divmod(three_days, tdarr)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], three_days // tdarr)

    # ------------------------------------------------------------------
    # Operations with invalid others

    def test_td64arr_mul_tdscalar_invalid(self, box_with_array, scalar_td):
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        td1 = tm.box_expected(td1, box_with_array)

        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            td1 * scalar_td
        with pytest.raises(TypeError, match=pattern):
            scalar_td * td1

    def test_td64arr_mul_too_short_raises(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        msg = (
            "cannot use operands with types dtype|"
            "Cannot multiply with unequal lengths|"
            "Unable to coerce to Series"
        )
        with pytest.raises(TypeError, match=msg):
            # length check before dtype check
            idx * idx[:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    def test_td64arr_mul_td64arr_raises(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        idx = tm.box_expected(idx, box_with_array)
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            idx * idx

    # ------------------------------------------------------------------
    # Operations with numeric others

    def test_td64arr_mul_numeric_scalar(self, box_with_array, one):
        # GH#4521
        # divide/multiply by integers
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["-59 Days", "-59 Days", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser * (-one)
        tm.assert_equal(result, expected)
        result = (-one) * tdser
        tm.assert_equal(result, expected)

        expected = Series(["118 Days", "118 Days", "NaT"], dtype="timedelta64[ns]")
        expected = tm.box_expected(expected, box_with_array)

        result = tdser * (2 * one)
        tm.assert_equal(result, expected)
        result = (2 * one) * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(self, box_with_array, two):
        # GH#4521
        # divide/multiply by integers
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = tdser / two
        tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match="Cannot divide"):
            two / tdser

    @pytest.mark.parametrize(
        "vector",
        [np.array([20, 30, 40]), pd.Index([20, 30, 40]), Series([20, 30, 40])],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_rmul_numeric_array(self, box_with_array, vector, any_real_dtype):
        # GH#4521
        # divide/multiply by integers
        xbox = get_upcast_box(box_with_array, vector)

        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        vector = vector.astype(any_real_dtype)

        expected = Series(["1180 Days", "1770 Days", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, xbox)

        result = tdser * vector
        tm.assert_equal(result, expected)

        result = vector * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "vector",
        [np.array([20, 30, 40]), pd.Index([20, 30, 40]), Series([20, 30, 40])],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_div_numeric_array(
        self, box_with_array, vector, any_real_dtype, using_array_manager
    ):
        # GH#4521
        # divide/multiply by integers
        xbox = get_upcast_box(box_with_array, vector)

        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        vector = vector.astype(any_real_dtype)

        expected = Series(["2.95D", "1D 23H 12m", "NaT"], dtype="timedelta64[ns]")

        tdser = tm.box_expected(tdser, box_with_array)
        expected = tm.box_expected(expected, xbox)

        result = tdser / vector
        tm.assert_equal(result, expected)

        pattern = (
            "true_divide'? cannot use operands|"
            "cannot perform __div__|"
            "cannot perform __truediv__|"
            "unsupported operand|"
            "Cannot divide"
        )
        with pytest.raises(TypeError, match=pattern):
            vector / tdser

        if not isinstance(vector, pd.Index):
            # Index.__rdiv__ won't try to operate elementwise, just raises
            result = tdser / vector.astype(object)
            if box_with_array is DataFrame:
                expected = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
            else:
                expected = [tdser[n] / vector[n] for n in range(len(tdser))]
            expected = pd.Index(expected)  # do dtype inference
            expected = tm.box_expected(expected, xbox)
            assert tm.get_dtype(expected) == "m8[ns]"

            if using_array_manager and box_with_array is DataFrame:
                # TODO the behaviour is buggy here (third column with all-NaT
                # as result doesn't get preserved as timedelta64 dtype).
                # Reported at https://github.com/pandas-dev/pandas/issues/39750
                # Changing the expected instead of xfailing to continue to test
                # the correct behaviour for the other columns
                expected[2] = Series([NaT, NaT], dtype=object)

            tm.assert_equal(result, expected)

        with pytest.raises(TypeError, match=pattern):
            vector.astype(object) / tdser

    def test_td64arr_mul_int_series(self, box_with_array, names, request):
        # GH#19042 test for correct name attachment
        box = box_with_array
        exname = get_expected_name(box, names)

        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        # TODO: Should we be parametrizing over types for `ser` too?
        ser = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])

        expected = Series(
            ["0days", "1day", "4days", "9days", "16days"],
            dtype="timedelta64[ns]",
            name=exname,
        )

        tdi = tm.box_expected(tdi, box)
        xbox = get_upcast_box(box, ser)

        expected = tm.box_expected(expected, xbox)

        result = ser * tdi
        tm.assert_equal(result, expected)

        # The direct operation tdi * ser still needs to be fixed.
        result = ser.__rmul__(tdi)
        if box is DataFrame:
            assert result is NotImplemented
        else:
            tm.assert_equal(result, expected)

    # TODO: Should we be parametrizing over types for `ser` too?
    def test_float_series_rdiv_td64arr(self, box_with_array, names):
        # GH#19042 test for correct name attachment
        # TODO: the direct operation TimedeltaIndex / Series still
        # needs to be fixed.
        box = box_with_array
        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        ser = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])

        xname = names[2] if box not in [tm.to_array, pd.array] else names[1]
        expected = Series(
            [tdi[n] / ser[n] for n in range(len(ser))],
            dtype="timedelta64[ns]",
            name=xname,
        )

        xbox = get_upcast_box(box, ser)

        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, xbox)

        result = ser.__rtruediv__(tdi)
        if box is DataFrame:
            # TODO: Should we skip this case sooner or test something else?
            assert result is NotImplemented
        else:
            tm.assert_equal(result, expected)


class TestTimedelta64ArrayLikeArithmetic:
    # Arithmetic tests for timedelta64[ns] vectors fully parametrized over
    #  DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all arithmetic
    #  tests will eventually end up here.

    def test_td64arr_pow_invalid(self, scalar_td, box_with_array):
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        td1 = tm.box_expected(td1, box_with_array)

        # check that we are getting a TypeError
        # with 'operate' (from core/ops.py) for the ops that are not
        # defined
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            scalar_td ** td1

        with pytest.raises(TypeError, match=pattern):
            td1 ** scalar_td


def test_add_timestamp_to_timedelta():
    # GH: 35897
    timestamp = Timestamp.now()
    result = timestamp + timedelta_range("0s", "1s", periods=31)
    expected = DatetimeIndex(
        [
            timestamp
            + (
                pd.to_timedelta("0.033333333s") * i
                + pd.to_timedelta("0.000000001s") * divmod(i, 3)[0]
            )
            for i in range(31)
        ]
    )
    tm.assert_index_equal(result, expected)
