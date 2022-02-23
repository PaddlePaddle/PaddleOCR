from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    offsets,
    to_offset,
)

import pandas._testing as tm


class TestTimestampArithmetic:
    def test_overflow_offset(self):
        # no overflow expected

        stamp = Timestamp("2000/1/1")
        offset_no_overflow = to_offset("D") * 100

        expected = Timestamp("2000/04/10")
        assert stamp + offset_no_overflow == expected

        assert offset_no_overflow + stamp == expected

        expected = Timestamp("1999/09/23")
        assert stamp - offset_no_overflow == expected

    def test_overflow_offset_raises(self):
        # xref https://github.com/statsmodels/statsmodels/issues/3374
        # ends up multiplying really large numbers which overflow

        stamp = Timestamp("2017-01-13 00:00:00")
        offset_overflow = 20169940 * offsets.Day(1)
        msg = (
            "the add operation between "
            r"\<-?\d+ \* Days\> and \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
            "will overflow"
        )
        lmsg = "|".join(
            ["Python int too large to convert to C long", "int too big to convert"]
        )

        with pytest.raises(OverflowError, match=lmsg):
            stamp + offset_overflow

        with pytest.raises(OverflowError, match=msg):
            offset_overflow + stamp

        with pytest.raises(OverflowError, match=lmsg):
            stamp - offset_overflow

        # xref https://github.com/pandas-dev/pandas/issues/14080
        # used to crash, so check for proper overflow exception

        stamp = Timestamp("2000/1/1")
        offset_overflow = to_offset("D") * 100 ** 5

        with pytest.raises(OverflowError, match=lmsg):
            stamp + offset_overflow

        with pytest.raises(OverflowError, match=msg):
            offset_overflow + stamp

        with pytest.raises(OverflowError, match=lmsg):
            stamp - offset_overflow

    def test_overflow_timestamp_raises(self):
        # https://github.com/pandas-dev/pandas/issues/31774
        msg = "Result is too large"
        a = Timestamp("2101-01-01 00:00:00")
        b = Timestamp("1688-01-01 00:00:00")

        with pytest.raises(OutOfBoundsDatetime, match=msg):
            a - b

        # but we're OK for timestamp and datetime.datetime
        assert (a - b.to_pydatetime()) == (a.to_pydatetime() - b)

    def test_delta_preserve_nanos(self):
        val = Timestamp(1337299200000000123)
        result = val + timedelta(1)
        assert result.nanosecond == val.nanosecond

    def test_rsub_dtscalars(self, tz_naive_fixture):
        # In particular, check that datetime64 - Timestamp works GH#28286
        td = Timedelta(1235345642000)
        ts = Timestamp.now(tz_naive_fixture)
        other = ts + td

        assert other - ts == td
        assert other.to_pydatetime() - ts == td
        if tz_naive_fixture is None:
            assert other.to_datetime64() - ts == td
        else:
            msg = "subtraction must have"
            with pytest.raises(TypeError, match=msg):
                other.to_datetime64() - ts

    def test_timestamp_sub_datetime(self):
        dt = datetime(2013, 10, 12)
        ts = Timestamp(datetime(2013, 10, 13))
        assert (ts - dt).days == 1
        assert (dt - ts).days == -1

    def test_addition_subtraction_types(self):
        # Assert on the types resulting from Timestamp +/- various date/time
        # objects
        dt = datetime(2014, 3, 4)
        td = timedelta(seconds=1)
        # build a timestamp with a frequency, since then it supports
        # addition/subtraction of integers
        with tm.assert_produces_warning(FutureWarning, match="The 'freq' argument"):
            # freq deprecated
            ts = Timestamp(dt, freq="D")

        msg = "Addition/subtraction of integers"
        with pytest.raises(TypeError, match=msg):
            # GH#22535 add/sub with integers is deprecated
            ts + 1
        with pytest.raises(TypeError, match=msg):
            ts - 1

        # Timestamp + datetime not supported, though subtraction is supported
        # and yields timedelta more tests in tseries/base/tests/test_base.py
        assert type(ts - dt) == Timedelta
        assert type(ts + td) == Timestamp
        assert type(ts - td) == Timestamp

        # Timestamp +/- datetime64 not supported, so not tested (could possibly
        # assert error raised?)
        td64 = np.timedelta64(1, "D")
        assert type(ts + td64) == Timestamp
        assert type(ts - td64) == Timestamp

    @pytest.mark.parametrize(
        "freq, td, td64",
        [
            ("S", timedelta(seconds=1), np.timedelta64(1, "s")),
            ("min", timedelta(minutes=1), np.timedelta64(1, "m")),
            ("H", timedelta(hours=1), np.timedelta64(1, "h")),
            ("D", timedelta(days=1), np.timedelta64(1, "D")),
            ("W", timedelta(weeks=1), np.timedelta64(1, "W")),
            ("M", None, np.timedelta64(1, "M")),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Timestamp.freq is deprecated:FutureWarning")
    @pytest.mark.filterwarnings("ignore:The 'freq' argument:FutureWarning")
    def test_addition_subtraction_preserve_frequency(self, freq, td, td64):
        ts = Timestamp("2014-03-05 00:00:00", freq=freq)
        original_freq = ts.freq

        assert (ts + 1 * original_freq).freq == original_freq
        assert (ts - 1 * original_freq).freq == original_freq

        if td is not None:
            # timedelta does not support months as unit
            assert (ts + td).freq == original_freq
            assert (ts - td).freq == original_freq

        assert (ts + td64).freq == original_freq
        assert (ts - td64).freq == original_freq

    @pytest.mark.parametrize(
        "td", [Timedelta(hours=3), np.timedelta64(3, "h"), timedelta(hours=3)]
    )
    def test_radd_tdscalar(self, td):
        # GH#24775 timedelta64+Timestamp should not raise
        ts = Timestamp.now()
        assert td + ts == ts + td

    @pytest.mark.parametrize(
        "other,expected_difference",
        [
            (np.timedelta64(-123, "ns"), -123),
            (np.timedelta64(1234567898, "ns"), 1234567898),
            (np.timedelta64(-123, "us"), -123000),
            (np.timedelta64(-123, "ms"), -123000000),
        ],
    )
    def test_timestamp_add_timedelta64_unit(self, other, expected_difference):
        ts = Timestamp(datetime.utcnow())
        result = ts + other
        valdiff = result.value - ts.value
        assert valdiff == expected_difference

    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("1776-07-04"),
            Timestamp("1776-07-04", tz="UTC"),
        ],
    )
    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.uint64),
        ],
    )
    def test_add_int_with_freq(self, ts, other):
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            ts + other
        with pytest.raises(TypeError, match=msg):
            other + ts

        with pytest.raises(TypeError, match=msg):
            ts - other

        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            other - ts

    @pytest.mark.parametrize("shape", [(6,), (2, 3)])
    def test_addsub_m8ndarray(self, shape):
        # GH#33296
        ts = Timestamp("2020-04-04 15:45")
        other = np.arange(6).astype("m8[h]").reshape(shape)

        result = ts + other

        ex_stamps = [ts + Timedelta(hours=n) for n in range(6)]
        expected = np.array([x.asm8 for x in ex_stamps], dtype="M8[ns]").reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        result = other + ts
        tm.assert_numpy_array_equal(result, expected)

        result = ts - other
        ex_stamps = [ts - Timedelta(hours=n) for n in range(6)]
        expected = np.array([x.asm8 for x in ex_stamps], dtype="M8[ns]").reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            other - ts

    @pytest.mark.parametrize("shape", [(6,), (2, 3)])
    def test_addsub_m8ndarray_tzaware(self, shape):
        # GH#33296
        ts = Timestamp("2020-04-04 15:45", tz="US/Pacific")

        other = np.arange(6).astype("m8[h]").reshape(shape)

        result = ts + other

        ex_stamps = [ts + Timedelta(hours=n) for n in range(6)]
        expected = np.array(ex_stamps).reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        result = other + ts
        tm.assert_numpy_array_equal(result, expected)

        result = ts - other
        ex_stamps = [ts - Timedelta(hours=n) for n in range(6)]
        expected = np.array(ex_stamps).reshape(shape)
        tm.assert_numpy_array_equal(result, expected)

        msg = r"unsupported operand type\(s\) for -: 'numpy.ndarray' and 'Timestamp'"
        with pytest.raises(TypeError, match=msg):
            other - ts
