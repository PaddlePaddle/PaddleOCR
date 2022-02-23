# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for datetime64 and datetime64tz dtypes
from datetime import (
    datetime,
    time,
    timedelta,
)
from itertools import (
    product,
    starmap,
)
import operator
import warnings

import numpy as np
import pytest
import pytz

from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.compat import np_datetime64_compat
from pandas.errors import PerformanceWarning

import pandas as pd
from pandas import (
    DateOffset,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
from pandas.core.ops import roperator
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)

# ------------------------------------------------------------------
# Comparisons


class TestDatetime64ArrayLikeComparisons:
    # Comparison tests for datetime64 vectors fully parametrized over
    #  DataFrame/Series/DatetimeIndex/DatetimeArray.  Ideally all comparison
    #  tests will eventually end up here.

    def test_compare_zerodim(self, tz_naive_fixture, box_with_array):
        # Test comparison with zero-dimensional array is unboxed
        tz = tz_naive_fixture
        box = box_with_array
        xbox = (
            box_with_array if box_with_array not in [pd.Index, pd.array] else np.ndarray
        )
        dti = date_range("20130101", periods=3, tz=tz)

        other = np.array(dti.to_numpy()[0])

        dtarr = tm.box_expected(dti, box)
        result = dtarr <= other
        expected = np.array([True, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            "foo",
            -1,
            99,
            4.0,
            object(),
            timedelta(days=2),
            # GH#19800, GH#19301 datetime.date comparison raises to
            #  match DatetimeIndex/Timestamp.  This also matches the behavior
            #  of stdlib datetime.datetime
            datetime(2001, 1, 1).date(),
            # GH#19301 None and NaN are *not* cast to NaT for comparisons
            None,
            np.nan,
        ],
    )
    def test_dt64arr_cmp_scalar_invalid(self, other, tz_naive_fixture, box_with_array):
        # GH#22074, GH#15966
        tz = tz_naive_fixture

        rng = date_range("1/1/2000", periods=10, tz=tz)
        dtarr = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),
            np.arange(10),
            np.arange(10).astype(np.float32),
            np.arange(10).astype(object),
            pd.timedelta_range("1ns", periods=10).array,
            np.array(pd.timedelta_range("1ns", periods=10)),
            list(pd.timedelta_range("1ns", periods=10)),
            pd.timedelta_range("1 Day", periods=10).astype(object),
            pd.period_range("1971-01-01", freq="D", periods=10).array,
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),
        ],
    )
    def test_dt64arr_cmp_arraylike_invalid(self, other, tz_naive_fixture):
        # We don't parametrize this over box_with_array because listlike
        #  other plays poorly with assert_invalid_comparison reversed checks
        tz = tz_naive_fixture

        dta = date_range("1970-01-01", freq="ns", periods=10, tz=tz)._data
        assert_invalid_comparison(dta, other, tm.to_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture):
        tz = tz_naive_fixture

        dta = date_range("1970-01-01", freq="h", periods=5, tz=tz)._data

        other = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result = dta == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture, box_with_array):
        # GH#22242, GH#22163 DataFrame considered NaT == ts incorrectly
        tz = tz_naive_fixture
        box = box_with_array
        xbox = box if box not in [pd.Index, pd.array] else np.ndarray

        ts = Timestamp.now(tz)
        ser = Series([ts, NaT])

        obj = tm.box_expected(ser, box)

        expected = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)

        result = obj == ts
        tm.assert_equal(result, expected)


class TestDatetime64SeriesComparison:
    # TODO: moved from tests.series.test_operators; needs cleanup

    @pytest.mark.parametrize(
        "pair",
        [
            (
                [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
                [NaT, NaT, Timestamp("2011-01-03")],
            ),
            (
                [Timedelta("1 days"), NaT, Timedelta("3 days")],
                [NaT, NaT, Timedelta("3 days")],
            ),
            (
                [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
                [NaT, NaT, Period("2011-03", freq="M")],
            ),
        ],
    )
    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("dtype", [None, object])
    @pytest.mark.parametrize(
        "op, expected",
        [
            (operator.eq, Series([False, False, True])),
            (operator.ne, Series([True, True, False])),
            (operator.lt, Series([False, False, False])),
            (operator.gt, Series([False, False, False])),
            (operator.ge, Series([False, False, True])),
            (operator.le, Series([False, False, True])),
        ],
    )
    def test_nat_comparisons(
        self,
        dtype,
        index_or_series,
        reverse,
        pair,
        op,
        expected,
    ):
        box = index_or_series
        l, r = pair
        if reverse:
            # add lhs / rhs switched data
            l, r = r, l

        left = Series(l, dtype=dtype)
        right = box(r, dtype=dtype)

        result = op(left, right)

        tm.assert_series_equal(result, expected)

    def test_comparison_invalid(self, tz_naive_fixture, box_with_array):
        # GH#4968
        # invalid date/int comparisons
        tz = tz_naive_fixture
        ser = Series(range(5))
        ser2 = Series(date_range("20010101", periods=5, tz=tz))

        ser = tm.box_expected(ser, box_with_array)
        ser2 = tm.box_expected(ser2, box_with_array)

        assert_invalid_comparison(ser, ser2, box_with_array)

    @pytest.mark.parametrize(
        "data",
        [
            [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],
            [Timedelta("1 days"), NaT, Timedelta("3 days")],
            [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_nat_comparisons_scalar(self, dtype, data, box_with_array):
        box = box_with_array
        if box_with_array is tm.to_array and dtype is object:
            # dont bother testing ndarray comparison methods as this fails
            #  on older numpys (since they check object identity)
            return

        xbox = box if box not in [pd.Index, pd.array] else np.ndarray

        left = Series(data, dtype=dtype)
        left = tm.box_expected(left, box)

        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")

        tm.assert_equal(left == NaT, expected)
        tm.assert_equal(NaT == left, expected)

        expected = [True, True, True]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        tm.assert_equal(left != NaT, expected)
        tm.assert_equal(NaT != left, expected)

        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        tm.assert_equal(left < NaT, expected)
        tm.assert_equal(NaT > left, expected)
        tm.assert_equal(left <= NaT, expected)
        tm.assert_equal(NaT >= left, expected)

        tm.assert_equal(left > NaT, expected)
        tm.assert_equal(NaT < left, expected)
        tm.assert_equal(left >= NaT, expected)
        tm.assert_equal(NaT <= left, expected)

    @pytest.mark.parametrize("val", [datetime(2000, 1, 4), datetime(2000, 1, 5)])
    def test_series_comparison_scalars(self, val):
        series = Series(date_range("1/1/2000", periods=10))

        result = series > val
        expected = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "left,right", [("lt", "gt"), ("le", "ge"), ("eq", "eq"), ("ne", "ne")]
    )
    def test_timestamp_compare_series(self, left, right):
        # see gh-4982
        # Make sure we can compare Timestamps on the right AND left hand side.
        ser = Series(date_range("20010101", periods=10), name="dates")
        s_nat = ser.copy(deep=True)

        ser[0] = Timestamp("nat")
        ser[3] = Timestamp("nat")

        left_f = getattr(operator, left)
        right_f = getattr(operator, right)

        # No NaT
        expected = left_f(ser, Timestamp("20010109"))
        result = right_f(Timestamp("20010109"), ser)
        tm.assert_series_equal(result, expected)

        # NaT
        expected = left_f(ser, Timestamp("nat"))
        result = right_f(Timestamp("nat"), ser)
        tm.assert_series_equal(result, expected)

        # Compare to Timestamp with series containing NaT
        expected = left_f(s_nat, Timestamp("20010109"))
        result = right_f(Timestamp("20010109"), s_nat)
        tm.assert_series_equal(result, expected)

        # Compare to NaT with series containing NaT
        expected = left_f(s_nat, Timestamp("nat"))
        result = right_f(Timestamp("nat"), s_nat)
        tm.assert_series_equal(result, expected)

    def test_dt64arr_timestamp_equality(self, box_with_array):
        # GH#11034
        xbox = (
            box_with_array if box_with_array not in [pd.Index, pd.array] else np.ndarray
        )

        ser = Series([Timestamp("2000-01-29 01:59:00"), Timestamp("2000-01-30"), NaT])
        ser = tm.box_expected(ser, box_with_array)

        result = ser != ser
        expected = tm.box_expected([False, False, True], xbox)
        tm.assert_equal(result, expected)

        warn = FutureWarning if box_with_array is pd.DataFrame else None
        with tm.assert_produces_warning(warn):
            # alignment for frame vs series comparisons deprecated
            result = ser != ser[0]
        expected = tm.box_expected([False, True, True], xbox)
        tm.assert_equal(result, expected)

        with tm.assert_produces_warning(warn):
            # alignment for frame vs series comparisons deprecated
            result = ser != ser[2]
        expected = tm.box_expected([True, True, True], xbox)
        tm.assert_equal(result, expected)

        result = ser == ser
        expected = tm.box_expected([True, True, False], xbox)
        tm.assert_equal(result, expected)

        with tm.assert_produces_warning(warn):
            # alignment for frame vs series comparisons deprecated
            result = ser == ser[0]
        expected = tm.box_expected([True, False, False], xbox)
        tm.assert_equal(result, expected)

        with tm.assert_produces_warning(warn):
            # alignment for frame vs series comparisons deprecated
            result = ser == ser[2]
        expected = tm.box_expected([False, False, False], xbox)
        tm.assert_equal(result, expected)


class TestDatetimeIndexComparisons:

    # TODO: moved from tests.indexes.test_base; parametrize and de-duplicate
    @pytest.mark.parametrize(
        "op",
        [operator.eq, operator.ne, operator.gt, operator.lt, operator.ge, operator.le],
    )
    def test_comparators(self, op):
        index = tm.makeDateIndex(100)
        element = index[len(index) // 2]
        element = Timestamp(element).to_datetime64()

        arr = np.array(index)
        arr_result = op(arr, element)
        index_result = op(index, element)

        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    def test_dti_cmp_datetimelike(self, other, tz_naive_fixture):
        tz = tz_naive_fixture
        dti = date_range("2016-01-01", periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                # no tzaware version available
                return
            other = localize_pydatetime(other, dti.tzinfo)

        result = dti == other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dti > other
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)

        result = dti >= other
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        result = dti < other
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

        result = dti <= other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_dti_cmp_nat(self, dtype, box_with_array):
        if box_with_array is tm.to_array and dtype is object:
            # dont bother testing ndarray comparison methods as this fails
            #  on older numpys (since they check object identity)
            return

        xbox = (
            box_with_array if box_with_array not in [pd.Index, pd.array] else np.ndarray
        )

        left = DatetimeIndex([Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")])
        right = DatetimeIndex([NaT, NaT, Timestamp("2011-01-03")])

        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)

        lhs, rhs = left, right
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)

        result = rhs == lhs
        expected = np.array([False, False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

        result = lhs != rhs
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs == NaT, expected)
        tm.assert_equal(NaT == rhs, expected)

        expected = np.array([True, True, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs != NaT, expected)
        tm.assert_equal(NaT != lhs, expected)

        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs < NaT, expected)
        tm.assert_equal(NaT > lhs, expected)

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self):
        fidx1 = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        fidx2 = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])

        didx1 = DatetimeIndex(
            ["2014-01-01", NaT, "2014-03-01", NaT, "2014-05-01", "2014-07-01"]
        )
        didx2 = DatetimeIndex(
            ["2014-02-01", "2014-03-01", NaT, NaT, "2014-06-01", "2014-07-01"]
        )
        darr = np.array(
            [
                np_datetime64_compat("2014-02-01 00:00Z"),
                np_datetime64_compat("2014-03-01 00:00Z"),
                np_datetime64_compat("nat"),
                np.datetime64("nat"),
                np_datetime64_compat("2014-06-01 00:00Z"),
                np_datetime64_compat("2014-07-01 00:00Z"),
            ]
        )

        cases = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]

        # Check pd.NaT is handles as the same as np.nan
        with tm.assert_produces_warning(None):
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

        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, np.nan), (didx1, NaT)]:
                result = idx1 < val
                expected = np.array([False, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 <= val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 == val
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 != val
                expected = np.array([True, True, True, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

        # Check pd.NaT is handles as the same as np.nan
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, 3), (didx1, datetime(2014, 3, 1))]:
                result = idx1 < val
                expected = np.array([True, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                expected = np.array([False, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 <= val
                expected = np.array([True, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                expected = np.array([False, False, True, False, True, True])
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 == val
                expected = np.array([False, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)

                result = idx1 != val
                expected = np.array([True, True, False, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "op",
        [operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le],
    )
    def test_comparison_tzawareness_compat(self, op, box_with_array):
        # GH#18162
        box = box_with_array

        dr = date_range("2016-01-01", periods=6)
        dz = dr.tz_localize("US/Pacific")

        dr = tm.box_expected(dr, box)
        dz = tm.box_expected(dz, box)

        if box is pd.DataFrame:
            tolist = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist = list

        if op not in [operator.eq, operator.ne]:
            msg = (
                r"Invalid comparison between dtype=datetime64\[ns.*\] "
                "and (Timestamp|DatetimeArray|list|ndarray)"
            )
            with pytest.raises(TypeError, match=msg):
                op(dr, dz)

            with pytest.raises(TypeError, match=msg):
                op(dr, tolist(dz))
            with pytest.raises(TypeError, match=msg):
                op(dr, np.array(tolist(dz), dtype=object))
            with pytest.raises(TypeError, match=msg):
                op(dz, dr)

            with pytest.raises(TypeError, match=msg):
                op(dz, tolist(dr))
            with pytest.raises(TypeError, match=msg):
                op(dz, np.array(tolist(dr), dtype=object))

        # The aware==aware and naive==naive comparisons should *not* raise
        assert np.all(dr == dr)
        assert np.all(dr == tolist(dr))
        assert np.all(tolist(dr) == dr)
        assert np.all(np.array(tolist(dr), dtype=object) == dr)
        assert np.all(dr == np.array(tolist(dr), dtype=object))

        assert np.all(dz == dz)
        assert np.all(dz == tolist(dz))
        assert np.all(tolist(dz) == dz)
        assert np.all(np.array(tolist(dz), dtype=object) == dz)
        assert np.all(dz == np.array(tolist(dz), dtype=object))

    @pytest.mark.parametrize(
        "op",
        [operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le],
    )
    def test_comparison_tzawareness_compat_scalars(self, op, box_with_array):
        # GH#18162
        dr = date_range("2016-01-01", periods=6)
        dz = dr.tz_localize("US/Pacific")

        dr = tm.box_expected(dr, box_with_array)
        dz = tm.box_expected(dz, box_with_array)

        # Check comparisons against scalar Timestamps
        ts = Timestamp("2000-03-14 01:59")
        ts_tz = Timestamp("2000-03-14 01:59", tz="Europe/Amsterdam")

        assert np.all(dr > ts)
        msg = r"Invalid comparison between dtype=datetime64\[ns.*\] and Timestamp"
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dr, ts_tz)

        assert np.all(dz > ts_tz)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dz, ts)

        if op not in [operator.eq, operator.ne]:
            # GH#12601: Check comparison against Timestamps and DatetimeIndex
            with pytest.raises(TypeError, match=msg):
                op(ts, dz)

    @pytest.mark.parametrize(
        "op",
        [operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le],
    )
    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    # Bug in NumPy? https://github.com/numpy/numpy/issues/13841
    # Raising in __eq__ will fallback to NumPy, which warns, fails,
    # then re-raises the original exception. So we just need to ignore.
    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:Converting timezone-aware:FutureWarning")
    def test_scalar_comparison_tzawareness(
        self, op, other, tz_aware_fixture, box_with_array
    ):
        box = box_with_array
        tz = tz_aware_fixture
        dti = date_range("2016-01-01", periods=2, tz=tz)
        xbox = box if box not in [pd.Index, pd.array] else np.ndarray

        dtarr = tm.box_expected(dti, box_with_array)
        if op in [operator.eq, operator.ne]:
            exbool = op is operator.ne
            expected = np.array([exbool, exbool], dtype=bool)
            expected = tm.box_expected(expected, xbox)

            result = op(dtarr, other)
            tm.assert_equal(result, expected)

            result = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            msg = (
                r"Invalid comparison between dtype=datetime64\[ns, .*\] "
                f"and {type(other).__name__}"
            )
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    @pytest.mark.parametrize(
        "op",
        [operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le],
    )
    def test_nat_comparison_tzawareness(self, op):
        # GH#19276
        # tzaware DatetimeIndex should not raise when compared to NaT
        dti = DatetimeIndex(
            ["2014-01-01", NaT, "2014-03-01", NaT, "2014-05-01", "2014-07-01"]
        )
        expected = np.array([op == operator.ne] * len(dti))
        result = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)

        result = op(dti.tz_localize("US/Pacific"), NaT)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_str(self, tz_naive_fixture):
        # GH#22074
        # regardless of tz, we expect these comparisons are valid
        tz = tz_naive_fixture
        rng = date_range("1/1/2000", periods=10, tz=tz)
        other = "1/1/2000"

        result = rng == other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng != other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng < other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)

        result = rng <= other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng > other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)

        result = rng >= other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_list(self):
        rng = date_range("1/1/2000", periods=10)

        result = rng == list(rng)
        expected = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "other",
        [
            pd.timedelta_range("1D", periods=10),
            pd.timedelta_range("1D", periods=10).to_series(),
            pd.timedelta_range("1D", periods=10).asi8.view("m8[ns]"),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_dti_cmp_tdi_tzawareness(self, other):
        # GH#22074
        # reversion test that we _don't_ call _assert_tzawareness_compat
        # when comparing against TimedeltaIndex
        dti = date_range("2000-01-01", periods=10, tz="Asia/Tokyo")

        result = dti == other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)

        result = dti != other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        msg = "Invalid comparison between"
        with pytest.raises(TypeError, match=msg):
            dti < other
        with pytest.raises(TypeError, match=msg):
            dti <= other
        with pytest.raises(TypeError, match=msg):
            dti > other
        with pytest.raises(TypeError, match=msg):
            dti >= other

    def test_dti_cmp_object_dtype(self):
        # GH#22074
        dti = date_range("2000-01-01", periods=10, tz="Asia/Tokyo")

        other = dti.astype("O")

        result = dti == other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

        other = dti.tz_localize(None)
        result = dti != other
        tm.assert_numpy_array_equal(result, expected)

        other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        result = dti == other
        expected = np.array([True] * 5 + [False] * 5)
        tm.assert_numpy_array_equal(result, expected)
        msg = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            dti >= other


# ------------------------------------------------------------------
# Arithmetic


class TestDatetime64Arithmetic:
    # This class is intended for "finished" tests that are fully parametrized
    #  over DataFrame/Series/Index/DatetimeArray

    # -------------------------------------------------------------
    # Addition/Subtraction of timedelta-like

    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        # GH#22005, GH#22163 check DataFrame doesn't raise TypeError
        tz = tz_naive_fixture

        rng = date_range("2000-01-01", "2000-02-01", tz=tz)
        expected = date_range("2000-01-01 02:00", "2000-02-01 02:00", tz=tz)

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = rng + two_hours
        tm.assert_equal(result, expected)

    def test_dt64arr_iadd_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        tz = tz_naive_fixture

        rng = date_range("2000-01-01", "2000-02-01", tz=tz)
        expected = date_range("2000-01-01 02:00", "2000-02-01 02:00", tz=tz)

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        rng += two_hours
        tm.assert_equal(rng, expected)

    def test_dt64arr_sub_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        tz = tz_naive_fixture

        rng = date_range("2000-01-01", "2000-02-01", tz=tz)
        expected = date_range("1999-12-31 22:00", "2000-01-31 22:00", tz=tz)

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = rng - two_hours
        tm.assert_equal(result, expected)

    def test_dt64arr_isub_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        tz = tz_naive_fixture

        rng = date_range("2000-01-01", "2000-02-01", tz=tz)
        expected = date_range("1999-12-31 22:00", "2000-01-31 22:00", tz=tz)

        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        rng -= two_hours
        tm.assert_equal(rng, expected)

    # TODO: redundant with test_dt64arr_add_timedeltalike_scalar
    def test_dt64arr_add_td64_scalar(self, box_with_array):
        # scalar timedeltas/np.timedelta64 objects
        # operate with np.timedelta64 correctly
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])

        expected = Series(
            [Timestamp("20130101 9:01:01"), Timestamp("20130101 9:02:01")]
        )

        dtarr = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr + np.timedelta64(1, "s")
        tm.assert_equal(result, expected)
        result = np.timedelta64(1, "s") + dtarr
        tm.assert_equal(result, expected)

        expected = Series(
            [Timestamp("20130101 9:01:00.005"), Timestamp("20130101 9:02:00.005")]
        )
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr + np.timedelta64(5, "ms")
        tm.assert_equal(result, expected)
        result = np.timedelta64(5, "ms") + dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_add_sub_td64_nat(self, box_with_array, tz_naive_fixture):
        # GH#23320 special handling for timedelta64("NaT")
        tz = tz_naive_fixture

        dti = date_range("1994-04-01", periods=9, tz=tz, freq="QS")
        other = np.timedelta64("NaT")
        expected = DatetimeIndex(["NaT"] * 9, tz=tz)

        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg = "cannot subtract"
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture, box_with_array):

        tz = tz_naive_fixture
        dti = date_range("2016-01-01", periods=3, tz=tz)
        tdi = TimedeltaIndex(["-1 Day", "-1 Day", "-1 Day"])
        tdarr = tdi.values

        expected = date_range("2015-12-31", "2016-01-02", periods=3, tz=tz)

        dtarr = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + dtarr
        tm.assert_equal(result, expected)

        expected = date_range("2016-01-02", "2016-01-04", periods=3, tz=tz)
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr - tdarr
        tm.assert_equal(result, expected)
        msg = "cannot subtract|(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr

    # -----------------------------------------------------------------
    # Subtraction of datetime-like scalars

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-01"),
            Timestamp("2013-01-01").to_pydatetime(),
            Timestamp("2013-01-01").to_datetime64(),
        ],
    )
    def test_dt64arr_sub_dtscalar(self, box_with_array, ts):
        # GH#8554, GH#22163 DataFrame op should _not_ return dt64 dtype
        idx = date_range("2013-01-01", periods=3)._with_freq(None)
        idx = tm.box_expected(idx, box_with_array)

        expected = TimedeltaIndex(["0 Days", "1 Day", "2 Days"])
        expected = tm.box_expected(expected, box_with_array)

        result = idx - ts
        tm.assert_equal(result, expected)

    def test_dt64arr_sub_datetime64_not_ns(self, box_with_array):
        # GH#7996, GH#22163 ensure non-nano datetime64 is converted to nano
        #  for DataFrame operation
        dt64 = np.datetime64("2013-01-01")
        assert dt64.dtype == "datetime64[D]"

        dti = date_range("20130101", periods=3)._with_freq(None)
        dtarr = tm.box_expected(dti, box_with_array)

        expected = TimedeltaIndex(["0 Days", "1 Day", "2 Days"])
        expected = tm.box_expected(expected, box_with_array)

        result = dtarr - dt64
        tm.assert_equal(result, expected)

        result = dt64 - dtarr
        tm.assert_equal(result, -expected)

    def test_dt64arr_sub_timestamp(self, box_with_array):
        ser = date_range("2014-03-17", periods=2, freq="D", tz="US/Eastern")
        ser = ser._with_freq(None)
        ts = ser[0]

        ser = tm.box_expected(ser, box_with_array)

        delta_series = Series([np.timedelta64(0, "D"), np.timedelta64(1, "D")])
        expected = tm.box_expected(delta_series, box_with_array)

        tm.assert_equal(ser - ts, expected)
        tm.assert_equal(ts - ser, -expected)

    def test_dt64arr_sub_NaT(self, box_with_array):
        # GH#18808
        dti = DatetimeIndex([NaT, Timestamp("19900315")])
        ser = tm.box_expected(dti, box_with_array)

        result = ser - NaT
        expected = Series([NaT, NaT], dtype="timedelta64[ns]")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        dti_tz = dti.tz_localize("Asia/Tokyo")
        ser_tz = tm.box_expected(dti_tz, box_with_array)

        result = ser_tz - NaT
        expected = Series([NaT, NaT], dtype="timedelta64[ns]")
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    # -------------------------------------------------------------
    # Subtraction of datetime-like array-like

    def test_dt64arr_sub_dt64object_array(self, box_with_array, tz_naive_fixture):
        dti = date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        expected = dti - dti

        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        with tm.assert_produces_warning(PerformanceWarning):
            result = obj - obj.astype(object)
        tm.assert_equal(result, expected)

    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array):
        dti = date_range("2016-01-01", periods=3, tz=None)
        dt64vals = dti.values

        dtarr = tm.box_expected(dti, box_with_array)

        expected = dtarr - dtarr
        result = dtarr - dt64vals
        tm.assert_equal(result, expected)
        result = dt64vals - dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_aware_sub_dt64ndarray_raises(
        self, tz_aware_fixture, box_with_array
    ):

        tz = tz_aware_fixture
        dti = date_range("2016-01-01", periods=3, tz=tz)
        dt64vals = dti.values

        dtarr = tm.box_expected(dti, box_with_array)
        msg = "subtraction must have the same timezones or"
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    # -------------------------------------------------------------
    # Addition of datetime-like others (invalid)

    def test_dt64arr_add_dt64ndarray_raises(self, tz_naive_fixture, box_with_array):

        tz = tz_naive_fixture
        dti = date_range("2016-01-01", periods=3, tz=tz)
        dt64vals = dti.values

        dtarr = tm.box_expected(dti, box_with_array)
        msg = "cannot add"
        with pytest.raises(TypeError, match=msg):
            dtarr + dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals + dtarr

    def test_dt64arr_add_timestamp_raises(self, box_with_array):
        # GH#22163 ensure DataFrame doesn't cast Timestamp to i8
        idx = DatetimeIndex(["2011-01-01", "2011-01-02"])
        idx = tm.box_expected(idx, box_with_array)
        msg = "cannot add"
        with pytest.raises(TypeError, match=msg):
            idx + Timestamp("2011-01-01")
        with pytest.raises(TypeError, match=msg):
            Timestamp("2011-01-01") + idx

    # -------------------------------------------------------------
    # Other Invalid Addition/Subtraction

    @pytest.mark.parametrize(
        "other",
        [
            3.14,
            np.array([2.0, 3.0]),
            # GH#13078 datetime +/- Period is invalid
            Period("2011-01-01", freq="D"),
            # https://github.com/pandas-dev/pandas/issues/10329
            time(1, 2, 3),
        ],
    )
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_invalid(self, dti_freq, other, box_with_array):
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"], freq=dti_freq)
        dtarr = tm.box_expected(dti, box_with_array)
        msg = "|".join(
            [
                "unsupported operand type",
                "cannot (add|subtract)",
                "cannot use operands with types",
                "ufunc '?(add|subtract)'? cannot use operands with types",
                "Concatenation operation is not implemented for NumPy arrays",
            ]
        )
        assert_invalid_addsub_type(dtarr, other, msg)

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "H"])
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_parr(
        self, dti_freq, pi_freq, box_with_array, box_with_array2
    ):
        # GH#20049 subtracting PeriodIndex should raise TypeError
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"], freq=dti_freq)
        pi = dti.to_period(pi_freq)

        dtarr = tm.box_expected(dti, box_with_array)
        parr = tm.box_expected(pi, box_with_array2)
        msg = "|".join(
            [
                "cannot (add|subtract)",
                "unsupported operand",
                "descriptor.*requires",
                "ufunc.*cannot use operands",
            ]
        )
        assert_invalid_addsub_type(dtarr, parr, msg)

    def test_dt64arr_addsub_time_objects_raises(self, box_with_array, tz_naive_fixture):
        # https://github.com/pandas-dev/pandas/issues/10329

        tz = tz_naive_fixture

        obj1 = date_range("2012-01-01", periods=3, tz=tz)
        obj2 = [time(i, i, i) for i in range(3)]

        obj1 = tm.box_expected(obj1, box_with_array)
        obj2 = tm.box_expected(obj2, box_with_array)

        with warnings.catch_warnings(record=True):
            # pandas.errors.PerformanceWarning: Non-vectorized DateOffset being
            # applied to Series or DatetimeIndex
            # we aren't testing that here, so ignore.
            warnings.simplefilter("ignore", PerformanceWarning)

            # If `x + y` raises, then `y + x` should raise here as well

            msg = (
                r"unsupported operand type\(s\) for -: "
                "'(Timestamp|DatetimeArray)' and 'datetime.time'"
            )
            with pytest.raises(TypeError, match=msg):
                obj1 - obj2

            msg = "|".join(
                [
                    "cannot subtract DatetimeArray from ndarray",
                    "ufunc (subtract|'subtract') cannot use operands with types "
                    r"dtype\('O'\) and dtype\('<M8\[ns\]'\)",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                obj2 - obj1

            msg = (
                r"unsupported operand type\(s\) for \+: "
                "'(Timestamp|DatetimeArray)' and 'datetime.time'"
            )
            with pytest.raises(TypeError, match=msg):
                obj1 + obj2

            msg = "|".join(
                [
                    r"unsupported operand type\(s\) for \+: "
                    "'(Timestamp|DatetimeArray)' and 'datetime.time'",
                    "ufunc (add|'add') cannot use operands with types "
                    r"dtype\('O'\) and dtype\('<M8\[ns\]'\)",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                obj2 + obj1


class TestDatetime64DateOffsetArithmetic:

    # -------------------------------------------------------------
    # Tick DateOffsets

    # TODO: parametrize over timezone?
    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array):
        # GH#4532
        # operate with pd.offsets
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        expected = Series(
            [Timestamp("20130101 9:01:05"), Timestamp("20130101 9:02:05")]
        )

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected)

        result2 = pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)

    def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array):
        # GH#4532
        # operate with pd.offsets
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        expected = Series(
            [Timestamp("20130101 9:00:55"), Timestamp("20130101 9:01:55")]
        )

        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        result = ser - pd.offsets.Second(5)
        tm.assert_equal(result, expected)

        result2 = -pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)
        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            pd.offsets.Second(5) - ser

    @pytest.mark.parametrize(
        "cls_name", ["Day", "Hour", "Minute", "Second", "Milli", "Micro", "Nano"]
    )
    def test_dt64arr_add_sub_tick_DateOffset_smoke(self, cls_name, box_with_array):
        # GH#4532
        # smoke tests for valid DateOffsets
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        ser = tm.box_expected(ser, box_with_array)

        offset_cls = getattr(pd.offsets, cls_name)
        ser + offset_cls(5)
        offset_cls(5) + ser
        ser - offset_cls(5)

    def test_dti_add_tick_tzaware(self, tz_aware_fixture, box_with_array):
        # GH#21610, GH#22163 ensure DataFrame doesn't return object-dtype
        tz = tz_aware_fixture
        if tz == "US/Pacific":
            dates = date_range("2012-11-01", periods=3, tz=tz)
            offset = dates + pd.offsets.Hour(5)
            assert dates[0] + pd.offsets.Hour(5) == offset[0]

        dates = date_range("2010-11-01 00:00", periods=3, tz=tz, freq="H")
        expected = DatetimeIndex(
            ["2010-11-01 05:00", "2010-11-01 06:00", "2010-11-01 07:00"],
            freq="H",
            tz=tz,
        )

        dates = tm.box_expected(dates, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # TODO: parametrize over the scalar being added?  radd?  sub?
        offset = dates + pd.offsets.Hour(5)
        tm.assert_equal(offset, expected)
        offset = dates + np.timedelta64(5, "h")
        tm.assert_equal(offset, expected)
        offset = dates + timedelta(hours=5)
        tm.assert_equal(offset, expected)

    # -------------------------------------------------------------
    # RelativeDelta DateOffsets

    def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array):
        # GH#10699
        vec = DatetimeIndex(
            [
                Timestamp("2000-01-05 00:15:00"),
                Timestamp("2000-01-31 00:23:00"),
                Timestamp("2000-01-01"),
                Timestamp("2000-03-31"),
                Timestamp("2000-02-29"),
                Timestamp("2000-12-31"),
                Timestamp("2000-05-15"),
                Timestamp("2001-06-15"),
            ]
        )
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec

        # DateOffset relativedelta fastpath
        relative_kwargs = [
            ("years", 2),
            ("months", 5),
            ("days", 3),
            ("hours", 5),
            ("minutes", 10),
            ("seconds", 2),
            ("microseconds", 5),
        ]
        for i, (unit, value) in enumerate(relative_kwargs):
            off = DateOffset(**{unit: value})

            expected = DatetimeIndex([x + off for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)

            expected = DatetimeIndex([x - off for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)

            off = DateOffset(**dict(relative_kwargs[: i + 1]))

            expected = DatetimeIndex([x + off for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)

            expected = DatetimeIndex([x - off for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            msg = "(bad|unsupported) operand type for unary"
            with pytest.raises(TypeError, match=msg):
                off - vec

    # -------------------------------------------------------------
    # Non-Tick, Non-RelativeDelta DateOffsets

    # TODO: redundant with test_dt64arr_add_sub_DateOffset?  that includes
    #  tz-aware cases which this does not
    @pytest.mark.parametrize(
        "cls_and_kwargs",
        [
            "YearBegin",
            ("YearBegin", {"month": 5}),
            "YearEnd",
            ("YearEnd", {"month": 5}),
            "MonthBegin",
            "MonthEnd",
            "SemiMonthEnd",
            "SemiMonthBegin",
            "Week",
            ("Week", {"weekday": 3}),
            "Week",
            ("Week", {"weekday": 6}),
            "BusinessDay",
            "BDay",
            "QuarterEnd",
            "QuarterBegin",
            "CustomBusinessDay",
            "CDay",
            "CBMonthEnd",
            "CBMonthBegin",
            "BMonthBegin",
            "BMonthEnd",
            "BusinessHour",
            "BYearBegin",
            "BYearEnd",
            "BQuarterBegin",
            ("LastWeekOfMonth", {"weekday": 2}),
            (
                "FY5253Quarter",
                {
                    "qtr_with_extra_week": 1,
                    "startingMonth": 1,
                    "weekday": 2,
                    "variation": "nearest",
                },
            ),
            ("FY5253", {"weekday": 0, "startingMonth": 2, "variation": "nearest"}),
            ("WeekOfMonth", {"weekday": 2, "week": 2}),
            "Easter",
            ("DateOffset", {"day": 4}),
            ("DateOffset", {"month": 5}),
        ],
    )
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("n", [0, 5])
    def test_dt64arr_add_sub_DateOffsets(
        self, box_with_array, n, normalize, cls_and_kwargs
    ):
        # GH#10699
        # assert vectorized operation matches pointwise operations

        if isinstance(cls_and_kwargs, tuple):
            # If cls_name param is a tuple, then 2nd entry is kwargs for
            # the offset constructor
            cls_name, kwargs = cls_and_kwargs
        else:
            cls_name = cls_and_kwargs
            kwargs = {}

        if n == 0 and cls_name in [
            "WeekOfMonth",
            "LastWeekOfMonth",
            "FY5253Quarter",
            "FY5253",
        ]:
            # passing n = 0 is invalid for these offset classes
            return

        vec = DatetimeIndex(
            [
                Timestamp("2000-01-05 00:15:00"),
                Timestamp("2000-01-31 00:23:00"),
                Timestamp("2000-01-01"),
                Timestamp("2000-03-31"),
                Timestamp("2000-02-29"),
                Timestamp("2000-12-31"),
                Timestamp("2000-05-15"),
                Timestamp("2001-06-15"),
            ]
        )
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec

        offset_cls = getattr(pd.offsets, cls_name)

        with warnings.catch_warnings(record=True):
            # pandas.errors.PerformanceWarning: Non-vectorized DateOffset being
            # applied to Series or DatetimeIndex
            # we aren't testing that here, so ignore.
            warnings.simplefilter("ignore", PerformanceWarning)

            offset = offset_cls(n, normalize=normalize, **kwargs)

            expected = DatetimeIndex([x + offset for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + offset)

            expected = DatetimeIndex([x - offset for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - offset)

            expected = DatetimeIndex([offset + x for x in vec_items])
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, offset + vec)
            msg = "(bad|unsupported) operand type for unary"
            with pytest.raises(TypeError, match=msg):
                offset - vec

    def test_dt64arr_add_sub_DateOffset(self, box_with_array):
        # GH#10699
        s = date_range("2000-01-01", "2000-01-31", name="a")
        s = tm.box_expected(s, box_with_array)
        result = s + DateOffset(years=1)
        result2 = DateOffset(years=1) + s
        exp = date_range("2001-01-01", "2001-01-31", name="a")._with_freq(None)
        exp = tm.box_expected(exp, box_with_array)
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        result = s - DateOffset(years=1)
        exp = date_range("1999-01-01", "1999-01-31", name="a")._with_freq(None)
        exp = tm.box_expected(exp, box_with_array)
        tm.assert_equal(result, exp)

        s = DatetimeIndex(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        s = tm.box_expected(s, box_with_array)
        result = s + pd.offsets.Day()
        result2 = pd.offsets.Day() + s
        exp = DatetimeIndex(
            [
                Timestamp("2000-01-16 00:15:00", tz="US/Central"),
                Timestamp("2000-02-16", tz="US/Central"),
            ],
            name="a",
        )
        exp = tm.box_expected(exp, box_with_array)
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

        s = DatetimeIndex(
            [
                Timestamp("2000-01-15 00:15:00", tz="US/Central"),
                Timestamp("2000-02-15", tz="US/Central"),
            ],
            name="a",
        )
        s = tm.box_expected(s, box_with_array)
        result = s + pd.offsets.MonthEnd()
        result2 = pd.offsets.MonthEnd() + s
        exp = DatetimeIndex(
            [
                Timestamp("2000-01-31 00:15:00", tz="US/Central"),
                Timestamp("2000-02-29", tz="US/Central"),
            ],
            name="a",
        )
        exp = tm.box_expected(exp, box_with_array)
        tm.assert_equal(result, exp)
        tm.assert_equal(result2, exp)

    @pytest.mark.parametrize(
        "other",
        [
            np.array([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)]),
            np.array([pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()]),
            np.array(  # matching offsets
                [pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)]
            ),
        ],
    )
    @pytest.mark.parametrize("op", [operator.add, roperator.radd, operator.sub])
    @pytest.mark.parametrize("box_other", [True, False])
    def test_dt64arr_add_sub_offset_array(
        self, tz_naive_fixture, box_with_array, box_other, op, other
    ):
        # GH#18849
        # GH#10699 array of offsets

        tz = tz_naive_fixture
        dti = date_range("2017-01-01", periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)

        other = np.array([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)])
        expected = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
        expected = tm.box_expected(expected, box_with_array)

        if box_other:
            other = tm.box_expected(other, box_with_array)

        with tm.assert_produces_warning(PerformanceWarning):
            res = op(dtarr, other)

        tm.assert_equal(res, expected)

    @pytest.mark.parametrize(
        "op, offset, exp, exp_freq",
        [
            (
                "__add__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2014-04-11"),
                    Timestamp("2015-04-11"),
                    Timestamp("2016-04-11"),
                    Timestamp("2017-04-11"),
                ],
                None,
            ),
            (
                "__add__",
                DateOffset(months=3),
                [
                    Timestamp("2014-04-01"),
                    Timestamp("2015-04-01"),
                    Timestamp("2016-04-01"),
                    Timestamp("2017-04-01"),
                ],
                "AS-APR",
            ),
            (
                "__sub__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2013-09-21"),
                    Timestamp("2014-09-21"),
                    Timestamp("2015-09-21"),
                    Timestamp("2016-09-21"),
                ],
                None,
            ),
            (
                "__sub__",
                DateOffset(months=3),
                [
                    Timestamp("2013-10-01"),
                    Timestamp("2014-10-01"),
                    Timestamp("2015-10-01"),
                    Timestamp("2016-10-01"),
                ],
                "AS-OCT",
            ),
        ],
    )
    def test_dti_add_sub_nonzero_mth_offset(
        self, op, offset, exp, exp_freq, tz_aware_fixture, box_with_array
    ):
        # GH 26258
        tz = tz_aware_fixture
        date = date_range(start="01 Jan 2014", end="01 Jan 2017", freq="AS", tz=tz)
        date = tm.box_expected(date, box_with_array, False)
        mth = getattr(date, op)
        result = mth(offset)

        expected = DatetimeIndex(exp, tz=tz)
        expected = tm.box_expected(expected, box_with_array, False)
        tm.assert_equal(result, expected)


class TestDatetime64OverflowHandling:
    # TODO: box + de-duplicate

    def test_dt64_overflow_masking(self, box_with_array):
        # GH#25317
        left = Series([Timestamp("1969-12-31")])
        right = Series([NaT])

        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)

        expected = TimedeltaIndex([NaT])
        expected = tm.box_expected(expected, box_with_array)

        result = left - right
        tm.assert_equal(result, expected)

    def test_dt64_series_arith_overflow(self):
        # GH#12534, fixed by GH#19024
        dt = Timestamp("1700-01-31")
        td = Timedelta("20000 Days")
        dti = date_range("1949-09-30", freq="100Y", periods=4)
        ser = Series(dti)
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            ser - dt
        with pytest.raises(OverflowError, match=msg):
            dt - ser
        with pytest.raises(OverflowError, match=msg):
            ser + td
        with pytest.raises(OverflowError, match=msg):
            td + ser

        ser.iloc[-1] = NaT
        expected = Series(
            ["2004-10-03", "2104-10-04", "2204-10-04", "NaT"], dtype="datetime64[ns]"
        )
        res = ser + td
        tm.assert_series_equal(res, expected)
        res = td + ser
        tm.assert_series_equal(res, expected)

        ser.iloc[1:] = NaT
        expected = Series(["91279 Days", "NaT", "NaT", "NaT"], dtype="timedelta64[ns]")
        res = ser - dt
        tm.assert_series_equal(res, expected)
        res = dt - ser
        tm.assert_series_equal(res, -expected)

    def test_datetimeindex_sub_timestamp_overflow(self):
        dtimax = pd.to_datetime(["now", Timestamp.max])
        dtimin = pd.to_datetime(["now", Timestamp.min])

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
        msg = "Overflow in int64 addition"
        for variant in ts_neg_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimax - variant

        expected = Timestamp.max.value - tspos.value
        for variant in ts_pos_variants:
            res = dtimax - variant
            assert res[1].value == expected

        expected = Timestamp.min.value - tsneg.value
        for variant in ts_neg_variants:
            res = dtimin - variant
            assert res[1].value == expected

        for variant in ts_pos_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimin - variant

    def test_datetimeindex_sub_datetimeindex_overflow(self):
        # GH#22492, GH#22508
        dtimax = pd.to_datetime(["now", Timestamp.max])
        dtimin = pd.to_datetime(["now", Timestamp.min])

        ts_neg = pd.to_datetime(["1950-01-01", "1950-01-01"])
        ts_pos = pd.to_datetime(["1980-01-01", "1980-01-01"])

        # General tests
        expected = Timestamp.max.value - ts_pos[1].value
        result = dtimax - ts_pos
        assert result[1].value == expected

        expected = Timestamp.min.value - ts_neg[1].value
        result = dtimin - ts_neg
        assert result[1].value == expected
        msg = "Overflow in int64 addition"
        with pytest.raises(OverflowError, match=msg):
            dtimax - ts_neg

        with pytest.raises(OverflowError, match=msg):
            dtimin - ts_pos

        # Edge cases
        tmin = pd.to_datetime([Timestamp.min])
        t1 = tmin + Timedelta.max + Timedelta("1us")
        with pytest.raises(OverflowError, match=msg):
            t1 - tmin

        tmax = pd.to_datetime([Timestamp.max])
        t2 = tmax + Timedelta.min - Timedelta("1us")
        with pytest.raises(OverflowError, match=msg):
            tmax - t2


class TestTimestampSeriesArithmetic:
    def test_empty_series_add_sub(self):
        # GH#13844
        a = Series(dtype="M8[ns]")
        b = Series(dtype="m8[ns]")
        tm.assert_series_equal(a, a + b)
        tm.assert_series_equal(a, a - b)
        tm.assert_series_equal(a, b + a)
        msg = "cannot subtract"
        with pytest.raises(TypeError, match=msg):
            b - a

    def test_operators_datetimelike(self):

        # ## timedelta64 ###
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        # ## datetime64 ###
        dt1 = Series(
            [
                Timestamp("20111230"),
                Timestamp("20120101"),
                Timestamp("20120103"),
            ]
        )
        dt1.iloc[2] = np.nan
        dt2 = Series(
            [
                Timestamp("20111231"),
                Timestamp("20120102"),
                Timestamp("20120104"),
            ]
        )
        dt1 - dt2
        dt2 - dt1

        # datetime64 with timetimedelta
        dt1 + td1
        td1 + dt1
        dt1 - td1

        # timetimedelta with datetime64
        td1 + dt1
        dt1 + td1

    def test_dt64ser_sub_datetime_dtype(self):
        ts = Timestamp(datetime(1993, 1, 7, 13, 30, 00))
        dt = datetime(1993, 6, 22, 13, 30)
        ser = Series([ts])
        result = pd.to_timedelta(np.abs(ser - dt))
        assert result.dtype == "timedelta64[ns]"

    # -------------------------------------------------------------
    # TODO: This next block of tests came from tests.series.test_operators,
    # needs to be de-duplicated and parametrized over `box` classes

    def test_operators_datetimelike_invalid(self, all_arithmetic_operators):
        # these are all TypeEror ops
        op_str = all_arithmetic_operators

        def check(get_ser, test_ser):

            # check that we are getting a TypeError
            # with 'operate' (from core/ops.py) for the ops that are not
            # defined
            op = getattr(get_ser, op_str, None)
            # Previously, _validate_for_numeric_binop in core/indexes/base.py
            # did this for us.
            with pytest.raises(
                TypeError, match="operate|[cC]annot|unsupported operand"
            ):
                op(test_ser)

        # ## timedelta64 ###
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan

        # ## datetime64 ###
        dt1 = Series(
            [Timestamp("20111230"), Timestamp("20120101"), Timestamp("20120103")]
        )
        dt1.iloc[2] = np.nan
        dt2 = Series(
            [Timestamp("20111231"), Timestamp("20120102"), Timestamp("20120104")]
        )
        if op_str not in ["__sub__", "__rsub__"]:
            check(dt1, dt2)

        # ## datetime64 with timetimedelta ###
        # TODO(jreback) __rsub__ should raise?
        if op_str not in ["__add__", "__radd__", "__sub__"]:
            check(dt1, td1)

        # 8260, 10763
        # datetime64 with tz
        tz = "US/Eastern"
        dt1 = Series(date_range("2000-01-01 09:00:00", periods=5, tz=tz), name="foo")
        dt2 = dt1.copy()
        dt2.iloc[2] = np.nan
        td1 = Series(pd.timedelta_range("1 days 1 min", periods=5, freq="H"))
        td2 = td1.copy()
        td2.iloc[1] = np.nan

        if op_str not in ["__add__", "__radd__", "__sub__", "__rsub__"]:
            check(dt2, td2)

    def test_sub_single_tz(self):
        # GH#12290
        s1 = Series([Timestamp("2016-02-10", tz="America/Sao_Paulo")])
        s2 = Series([Timestamp("2016-02-08", tz="America/Sao_Paulo")])
        result = s1 - s2
        expected = Series([Timedelta("2days")])
        tm.assert_series_equal(result, expected)
        result = s2 - s1
        expected = Series([Timedelta("-2days")])
        tm.assert_series_equal(result, expected)

    def test_dt64tz_series_sub_dtitz(self):
        # GH#19071 subtracting tzaware DatetimeIndex from tzaware Series
        # (with same tz) raises, fixed by #19024
        dti = date_range("1999-09-30", periods=10, tz="US/Pacific")
        ser = Series(dti)
        expected = Series(TimedeltaIndex(["0days"] * 10))

        res = dti - ser
        tm.assert_series_equal(res, expected)
        res = ser - dti
        tm.assert_series_equal(res, expected)

    def test_sub_datetime_compat(self):
        # see GH#14088
        s = Series([datetime(2016, 8, 23, 12, tzinfo=pytz.utc), NaT])
        dt = datetime(2016, 8, 22, 12, tzinfo=pytz.utc)
        exp = Series([Timedelta("1 days"), NaT])
        tm.assert_series_equal(s - dt, exp)
        tm.assert_series_equal(s - Timestamp(dt), exp)

    def test_dt64_series_add_mixed_tick_DateOffset(self):
        # GH#4532
        # operate with pd.offsets
        s = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])

        result = s + pd.offsets.Milli(5)
        result2 = pd.offsets.Milli(5) + s
        expected = Series(
            [Timestamp("20130101 9:01:00.005"), Timestamp("20130101 9:02:00.005")]
        )
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        result = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
        expected = Series(
            [Timestamp("20130101 9:06:00.005"), Timestamp("20130101 9:07:00.005")]
        )
        tm.assert_series_equal(result, expected)

    def test_datetime64_ops_nat(self):
        # GH#11349
        datetime_series = Series([NaT, Timestamp("19900315")])
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype="datetime64[ns]")
        single_nat_dtype_datetime = Series([NaT], dtype="datetime64[ns]")

        # subtraction
        tm.assert_series_equal(-NaT + datetime_series, nat_series_dtype_timestamp)
        msg = "Unary negative expects"
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + datetime_series

        tm.assert_series_equal(
            -NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + nat_series_dtype_timestamp

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )

    # -------------------------------------------------------------
    # Invalid Operations
    # TODO: this block also needs to be de-duplicated and parametrized

    @pytest.mark.parametrize(
        "dt64_series",
        [
            Series([Timestamp("19900315"), Timestamp("19900315")]),
            Series([NaT, Timestamp("19900315")]),
            Series([NaT, NaT], dtype="datetime64[ns]"),
        ],
    )
    @pytest.mark.parametrize("one", [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one, dt64_series):
        # multiplication
        msg = "cannot perform .* with this index type"
        with pytest.raises(TypeError, match=msg):
            dt64_series * one
        with pytest.raises(TypeError, match=msg):
            one * dt64_series

        # division
        with pytest.raises(TypeError, match=msg):
            dt64_series / one
        with pytest.raises(TypeError, match=msg):
            one / dt64_series

    # TODO: parametrize over box
    @pytest.mark.parametrize("op", ["__add__", "__radd__", "__sub__", "__rsub__"])
    def test_dt64_series_add_intlike(self, tz_naive_fixture, op):
        # GH#19123
        tz = tz_naive_fixture
        dti = DatetimeIndex(["2016-01-02", "2016-02-03", "NaT"], tz=tz)
        ser = Series(dti)

        other = Series([20, 30, 40], dtype="uint8")

        method = getattr(ser, op)
        msg = "|".join(
            [
                "Addition/subtraction of integers and integer-arrays",
                "cannot subtract .* from ndarray",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            method(1)
        with pytest.raises(TypeError, match=msg):
            method(other)
        with pytest.raises(TypeError, match=msg):
            method(np.array(other))
        with pytest.raises(TypeError, match=msg):
            method(pd.Index(other))

    # -------------------------------------------------------------
    # Timezone-Centric Tests

    def test_operators_datetimelike_with_timezones(self):
        tz = "US/Eastern"
        dt1 = Series(date_range("2000-01-01 09:00:00", periods=5, tz=tz), name="foo")
        dt2 = dt1.copy()
        dt2.iloc[2] = np.nan

        td1 = Series(pd.timedelta_range("1 days 1 min", periods=5, freq="H"))
        td2 = td1.copy()
        td2.iloc[1] = np.nan
        assert td2._values.freq is None

        result = dt1 + td1[0]
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt2 + td2[0]
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        # odd numpy behavior with scalar timedeltas
        result = td1[0] + dt1
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = td2[0] + dt2
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt1 - td1[0]
        exp = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            td1[0] - dt1

        result = dt2 - td2[0]
        exp = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        with pytest.raises(TypeError, match=msg):
            td2[0] - dt2

        result = dt1 + td1
        exp = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt2 + td2
        exp = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt1 - td1
        exp = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)

        result = dt2 - td2
        exp = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = "cannot (add|subtract)"
        with pytest.raises(TypeError, match=msg):
            td1 - dt1
        with pytest.raises(TypeError, match=msg):
            td2 - dt2


class TestDatetimeIndexArithmetic:

    # -------------------------------------------------------------
    # Binary operations DatetimeIndex and int

    def test_dti_addsub_int(self, tz_naive_fixture, one):
        # Variants of `one` for #19012
        tz = tz_naive_fixture
        rng = date_range("2000-01-01 09:00", freq="H", periods=10, tz=tz)
        msg = "Addition/subtraction of integers"

        with pytest.raises(TypeError, match=msg):
            rng + one
        with pytest.raises(TypeError, match=msg):
            rng += one
        with pytest.raises(TypeError, match=msg):
            rng - one
        with pytest.raises(TypeError, match=msg):
            rng -= one

    # -------------------------------------------------------------
    # __add__/__sub__ with integer arrays

    @pytest.mark.parametrize("freq", ["H", "D"])
    @pytest.mark.parametrize("int_holder", [np.array, pd.Index])
    def test_dti_add_intarray_tick(self, int_holder, freq):
        # GH#19959
        dti = date_range("2016-01-01", periods=2, freq=freq)
        other = int_holder([4, -1])

        msg = "Addition/subtraction of integers|cannot subtract DatetimeArray from"
        assert_invalid_addsub_type(dti, other, msg)

    @pytest.mark.parametrize("freq", ["W", "M", "MS", "Q"])
    @pytest.mark.parametrize("int_holder", [np.array, pd.Index])
    def test_dti_add_intarray_non_tick(self, int_holder, freq):
        # GH#19959
        dti = date_range("2016-01-01", periods=2, freq=freq)
        other = int_holder([4, -1])

        msg = "Addition/subtraction of integers|cannot subtract DatetimeArray from"
        assert_invalid_addsub_type(dti, other, msg)

    @pytest.mark.parametrize("int_holder", [np.array, pd.Index])
    def test_dti_add_intarray_no_freq(self, int_holder):
        # GH#19959
        dti = DatetimeIndex(["2016-01-01", "NaT", "2017-04-05 06:07:08"])
        other = int_holder([9, 4, -1])
        msg = "|".join(
            ["cannot subtract DatetimeArray from", "Addition/subtraction of integers"]
        )
        assert_invalid_addsub_type(dti, other, msg)

    # -------------------------------------------------------------
    # Binary operations DatetimeIndex and TimedeltaIndex/array

    def test_dti_add_tdi(self, tz_naive_fixture):
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz)
        expected = expected._with_freq(None)

        # add with TimdeltaIndex
        result = dti + tdi
        tm.assert_index_equal(result, expected)

        result = tdi + dti
        tm.assert_index_equal(result, expected)

        # add with timedelta64 array
        result = dti + tdi.values
        tm.assert_index_equal(result, expected)

        result = tdi.values + dti
        tm.assert_index_equal(result, expected)

    def test_dti_iadd_tdi(self, tz_naive_fixture):
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz)
        expected = expected._with_freq(None)

        # iadd with TimdeltaIndex
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result += tdi
        tm.assert_index_equal(result, expected)

        result = pd.timedelta_range("0 days", periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

        # iadd with timedelta64 array
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result += tdi.values
        tm.assert_index_equal(result, expected)

        result = pd.timedelta_range("0 days", periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

    def test_dti_sub_tdi(self, tz_naive_fixture):
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz, freq="-1D")
        expected = expected._with_freq(None)

        # sub with TimedeltaIndex
        result = dti - tdi
        tm.assert_index_equal(result, expected)

        msg = "cannot subtract .*TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        # sub with timedelta64 array
        result = dti - tdi.values
        tm.assert_index_equal(result, expected)

        msg = "cannot subtract DatetimeArray from"
        with pytest.raises(TypeError, match=msg):
            tdi.values - dti

    def test_dti_isub_tdi(self, tz_naive_fixture):
        # GH#17558
        tz = tz_naive_fixture
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        tdi = pd.timedelta_range("0 days", periods=10)
        expected = date_range("2017-01-01", periods=10, tz=tz, freq="-1D")
        expected = expected._with_freq(None)

        # isub with TimedeltaIndex
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result -= tdi
        tm.assert_index_equal(result, expected)

        msg = "cannot subtract .* from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi -= dti

        # isub with timedelta64 array
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result -= tdi.values
        tm.assert_index_equal(result, expected)

        msg = "|".join(
            [
                "cannot perform __neg__ with this index type:",
                "ufunc subtract cannot use operands with types",
                "cannot subtract DatetimeArray from",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            tdi.values -= dti

    # -------------------------------------------------------------
    # Binary Operations DatetimeIndex and datetime-like
    # TODO: A couple other tests belong in this section.  Move them in
    # A PR where there isn't already a giant diff.

    @pytest.mark.parametrize(
        "addend",
        [
            datetime(2011, 1, 1),
            DatetimeIndex(["2011-01-01", "2011-01-02"]),
            DatetimeIndex(["2011-01-01", "2011-01-02"]).tz_localize("US/Eastern"),
            np.datetime64("2011-01-01"),
            Timestamp("2011-01-01"),
        ],
        ids=lambda x: type(x).__name__,
    )
    @pytest.mark.parametrize("tz", [None, "US/Eastern"])
    def test_add_datetimelike_and_dtarr(self, box_with_array, addend, tz):
        # GH#9631
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"]).tz_localize(tz)
        dtarr = tm.box_expected(dti, box_with_array)
        msg = "cannot add DatetimeArray and"

        with pytest.raises(TypeError, match=msg):
            dtarr + addend
        with pytest.raises(TypeError, match=msg):
            addend + dtarr

    # -------------------------------------------------------------

    def test_dta_add_sub_index(self, tz_naive_fixture):
        # Check that DatetimeArray defers to Index classes
        dti = date_range("20130101", periods=3, tz=tz_naive_fixture)
        dta = dti.array
        result = dta - dti
        expected = dti - dti
        tm.assert_index_equal(result, expected)

        tdi = result
        result = dta + tdi
        expected = dti + tdi
        tm.assert_index_equal(result, expected)

        result = dta - tdi
        expected = dti - tdi
        tm.assert_index_equal(result, expected)

    def test_sub_dti_dti(self):
        # previously performed setop (deprecated in 0.16.0), now changed to
        # return subtraction -> TimeDeltaIndex (GH ...)

        dti = date_range("20130101", periods=3)
        dti_tz = date_range("20130101", periods=3).tz_localize("US/Eastern")
        dti_tz2 = date_range("20130101", periods=3).tz_localize("UTC")
        expected = TimedeltaIndex([0, 0, 0])

        result = dti - dti
        tm.assert_index_equal(result, expected)

        result = dti_tz - dti_tz
        tm.assert_index_equal(result, expected)
        msg = "DatetimeArray subtraction must have the same timezones or"
        with pytest.raises(TypeError, match=msg):
            dti_tz - dti

        with pytest.raises(TypeError, match=msg):
            dti - dti_tz

        with pytest.raises(TypeError, match=msg):
            dti_tz - dti_tz2

        # isub
        dti -= dti
        tm.assert_index_equal(dti, expected)

        # different length raises ValueError
        dti1 = date_range("20130101", periods=3)
        dti2 = date_range("20130101", periods=4)
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            dti1 - dti2

        # NaN propagation
        dti1 = DatetimeIndex(["2012-01-01", np.nan, "2012-01-03"])
        dti2 = DatetimeIndex(["2012-01-02", "2012-01-03", np.nan])
        expected = TimedeltaIndex(["1 days", np.nan, np.nan])
        result = dti2 - dti1
        tm.assert_index_equal(result, expected)

    # -------------------------------------------------------------------
    # TODO: Most of this block is moved from series or frame tests, needs
    # cleanup, box-parametrization, and de-duplication

    @pytest.mark.parametrize("op", [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(self, op):
        ser = Series(
            [
                Timestamp("20130301"),
                Timestamp("20130228 23:00:00"),
                Timestamp("20130228 22:00:00"),
                Timestamp("20130228 21:00:00"),
            ]
        )

        intervals = ["D", "h", "m", "s", "us"]

        def timedelta64(*args):
            # see casting notes in NumPy gh-12927
            return np.sum(list(starmap(np.timedelta64, zip(args, intervals))))

        for d, h, m, s, us in product(*([range(2)] * 5)):
            nptd = timedelta64(d, h, m, s, us)
            pytd = timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=us)
            lhs = op(ser, nptd)
            rhs = op(ser, pytd)

            tm.assert_series_equal(lhs, rhs)

    def test_ops_nat_mixed_datetime64_timedelta64(self):
        # GH#11349
        timedelta_series = Series([NaT, Timedelta("1s")])
        datetime_series = Series([NaT, Timestamp("19900315")])
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[ns]")
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype="datetime64[ns]")
        single_nat_dtype_datetime = Series([NaT], dtype="datetime64[ns]")
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[ns]")

        # subtraction
        tm.assert_series_equal(
            datetime_series - single_nat_dtype_datetime, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            datetime_series - single_nat_dtype_timedelta, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + datetime_series, nat_series_dtype_timestamp
        )

        # without a Series wrapping the NaT, it is ambiguous
        # whether it is a datetime64 or timedelta64
        # defaults to interpreting it as timedelta64
        tm.assert_series_equal(
            nat_series_dtype_timestamp - single_nat_dtype_datetime,
            nat_series_dtype_timedelta,
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp - single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )
        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            timedelta_series - single_nat_dtype_datetime

        # addition
        tm.assert_series_equal(
            nat_series_dtype_timestamp + single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp + single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_datetime,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_datetime + nat_series_dtype_timedelta,
            nat_series_dtype_timestamp,
        )

    def test_ufunc_coercions(self):
        idx = date_range("2011-01-01", periods=3, freq="2D", name="x")

        delta = np.timedelta64(1, "D")
        exp = date_range("2011-01-02", periods=3, freq="2D", name="x")
        for result in [idx + delta, np.add(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == "2D"

        exp = date_range("2010-12-31", periods=3, freq="2D", name="x")

        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == "2D"

        # When adding/subtracting an ndarray (which has no .freq), the result
        #  does not infer freq
        idx = idx._with_freq(None)
        delta = np.array(
            [np.timedelta64(1, "D"), np.timedelta64(2, "D"), np.timedelta64(3, "D")]
        )
        exp = DatetimeIndex(["2011-01-02", "2011-01-05", "2011-01-08"], name="x")

        for result in [idx + delta, np.add(idx, delta)]:
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

        exp = DatetimeIndex(["2010-12-31", "2011-01-01", "2011-01-02"], name="x")
        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

    def test_dti_add_series(self, tz_naive_fixture, names):
        # GH#13905
        tz = tz_naive_fixture
        index = DatetimeIndex(
            ["2016-06-28 05:30", "2016-06-28 05:31"], tz=tz, name=names[0]
        )
        ser = Series([Timedelta(seconds=5)] * 2, index=index, name=names[1])
        expected = Series(index + Timedelta(seconds=5), index=index, name=names[2])

        # passing name arg isn't enough when names[2] is None
        expected.name = names[2]
        assert expected.dtype == index.dtype
        result = ser + index
        tm.assert_series_equal(result, expected)
        result2 = index + ser
        tm.assert_series_equal(result2, expected)

        expected = index + Timedelta(seconds=5)
        result3 = ser.values + index
        tm.assert_index_equal(result3, expected)
        result4 = index + ser.values
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize("op", [operator.add, roperator.radd, operator.sub])
    def test_dti_addsub_offset_arraylike(
        self, tz_naive_fixture, names, op, index_or_series
    ):
        # GH#18849, GH#19744
        box = pd.Index
        other_box = index_or_series

        tz = tz_naive_fixture
        dti = date_range("2017-01-01", periods=2, tz=tz, name=names[0])
        other = other_box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], name=names[1])

        xbox = get_upcast_box(box, other)

        with tm.assert_produces_warning(PerformanceWarning):
            res = op(dti, other)

        expected = DatetimeIndex(
            [op(dti[n], other[n]) for n in range(len(dti))], name=names[2], freq="infer"
        )
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize("other_box", [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(
        self, tz_naive_fixture, box_with_array, other_box
    ):
        tz = tz_naive_fixture

        dti = date_range("2017-01-01", periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        other = other_box([pd.offsets.MonthEnd(), Timedelta(days=4)])
        xbox = get_upcast_box(box_with_array, other)

        expected = DatetimeIndex(["2017-01-31", "2017-01-06"], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox)

        with tm.assert_produces_warning(PerformanceWarning):
            result = dtarr + other
        tm.assert_equal(result, expected)

        expected = DatetimeIndex(["2016-12-31", "2016-12-29"], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox)

        with tm.assert_produces_warning(PerformanceWarning):
            result = dtarr - other
        tm.assert_equal(result, expected)


@pytest.mark.parametrize("years", [-1, 0, 1])
@pytest.mark.parametrize("months", [-2, 0, 2])
def test_shift_months(years, months):
    dti = DatetimeIndex(
        [
            Timestamp("2000-01-05 00:15:00"),
            Timestamp("2000-01-31 00:23:00"),
            Timestamp("2000-01-01"),
            Timestamp("2000-02-29"),
            Timestamp("2000-12-31"),
        ]
    )
    actual = DatetimeIndex(shift_months(dti.asi8, years * 12 + months))

    raw = [x + pd.offsets.DateOffset(years=years, months=months) for x in dti]
    expected = DatetimeIndex(raw)
    tm.assert_index_equal(actual, expected)


def test_dt64arr_addsub_object_dtype_2d():
    # block-wise DataFrame operations will require operating on 2D
    #  DatetimeArray/TimedeltaArray, so check that specifically.
    dti = date_range("1994-02-13", freq="2W", periods=4)
    dta = dti._data.reshape((4, 1))

    other = np.array([[pd.offsets.Day(n)] for n in range(4)])
    assert other.shape == dta.shape

    with tm.assert_produces_warning(PerformanceWarning):
        result = dta + other
    with tm.assert_produces_warning(PerformanceWarning):
        expected = (dta[:, 0] + other[:, 0]).reshape(-1, 1)

    assert isinstance(result, DatetimeArray)
    assert result.freq is None
    tm.assert_numpy_array_equal(result._data, expected._data)

    with tm.assert_produces_warning(PerformanceWarning):
        # Case where we expect to get a TimedeltaArray back
        result2 = dta - dta.astype(object)

    assert isinstance(result2, TimedeltaArray)
    assert result2.shape == (4, 1)
    assert result2.freq is None
    assert (result2.asi8 == 0).all()
