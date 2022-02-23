""" test the scalar Timedelta """
from datetime import timedelta

import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    iNaT,
)

import pandas as pd
from pandas import (
    Timedelta,
    TimedeltaIndex,
    offsets,
    to_timedelta,
)
import pandas._testing as tm


class TestTimedeltaUnaryOps:
    def test_unary_ops(self):
        td = Timedelta(10, unit="d")

        # __neg__, __pos__
        assert -td == Timedelta(-10, unit="d")
        assert -td == Timedelta("-10d")
        assert +td == Timedelta(10, unit="d")

        # __abs__, __abs__(__neg__)
        assert abs(td) == td
        assert abs(-td) == td
        assert abs(-td) == Timedelta("10d")


class TestTimedeltas:
    @pytest.mark.parametrize(
        "unit, value, expected",
        [
            ("us", 9.999, 9999),
            ("ms", 9.999999, 9999999),
            ("s", 9.999999999, 9999999999),
        ],
    )
    def test_rounding_on_int_unit_construction(self, unit, value, expected):
        # GH 12690
        result = Timedelta(value, unit=unit)
        assert result.value == expected
        result = Timedelta(str(value) + unit)
        assert result.value == expected

    def test_total_seconds_scalar(self):
        # see gh-10939
        rng = Timedelta("1 days, 10:11:12.100123456")
        expt = 1 * 86400 + 10 * 3600 + 11 * 60 + 12 + 100123456.0 / 1e9
        tm.assert_almost_equal(rng.total_seconds(), expt)

        rng = Timedelta(np.nan)
        assert np.isnan(rng.total_seconds())

    def test_conversion(self):

        for td in [Timedelta(10, unit="d"), Timedelta("1 days, 10:11:12.012345")]:
            pydt = td.to_pytimedelta()
            assert td == Timedelta(pydt)
            assert td == pydt
            assert isinstance(pydt, timedelta) and not isinstance(pydt, Timedelta)

            assert td == np.timedelta64(td.value, "ns")
            td64 = td.to_timedelta64()

            assert td64 == np.timedelta64(td.value, "ns")
            assert td == td64

            assert isinstance(td64, np.timedelta64)

        # this is NOT equal and cannot be roundtripped (because of the nanos)
        td = Timedelta("1 days, 10:11:12.012345678")
        assert td != td.to_pytimedelta()

    def test_fields(self):
        def check(value):
            # that we are int
            assert isinstance(value, int)

        # compat to datetime.timedelta
        rng = to_timedelta("1 days, 10:11:12")
        assert rng.days == 1
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        assert rng.microseconds == 0
        assert rng.nanoseconds == 0

        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # GH 10050
        check(rng.days)
        check(rng.seconds)
        check(rng.microseconds)
        check(rng.nanoseconds)

        td = Timedelta("-1 days, 10:11:12")
        assert abs(td) == Timedelta("13:48:48")
        assert str(td) == "-1 days +10:11:12"
        assert -td == Timedelta("0 days 13:48:48")
        assert -Timedelta("-1 days, 10:11:12").value == 49728000000000
        assert Timedelta("-1 days, 10:11:12").value == -49728000000000

        rng = to_timedelta("-1 days, 10:11:12.100123456")
        assert rng.days == -1
        assert rng.seconds == 10 * 3600 + 11 * 60 + 12
        assert rng.microseconds == 100 * 1000 + 123
        assert rng.nanoseconds == 456
        msg = "'Timedelta' object has no attribute '{}'"
        with pytest.raises(AttributeError, match=msg.format("hours")):
            rng.hours
        with pytest.raises(AttributeError, match=msg.format("minutes")):
            rng.minutes
        with pytest.raises(AttributeError, match=msg.format("milliseconds")):
            rng.milliseconds

        # components
        tup = to_timedelta(-1, "us").components
        assert tup.days == -1
        assert tup.hours == 23
        assert tup.minutes == 59
        assert tup.seconds == 59
        assert tup.milliseconds == 999
        assert tup.microseconds == 999
        assert tup.nanoseconds == 0

        # GH 10050
        check(tup.days)
        check(tup.hours)
        check(tup.minutes)
        check(tup.seconds)
        check(tup.milliseconds)
        check(tup.microseconds)
        check(tup.nanoseconds)

        tup = Timedelta("-1 days 1 us").components
        assert tup.days == -2
        assert tup.hours == 23
        assert tup.minutes == 59
        assert tup.seconds == 59
        assert tup.milliseconds == 999
        assert tup.microseconds == 999
        assert tup.nanoseconds == 0

    def test_iso_conversion(self):
        # GH #21877
        expected = Timedelta(1, unit="s")
        assert to_timedelta("P0DT0H0M1S") == expected

    def test_nat_converters(self):
        result = to_timedelta("nat").to_numpy()
        assert result.dtype.kind == "M"
        assert result.astype("int64") == iNaT

        result = to_timedelta("nan").to_numpy()
        assert result.dtype.kind == "M"
        assert result.astype("int64") == iNaT

    @pytest.mark.parametrize(
        "unit, np_unit",
        [(value, "W") for value in ["W", "w"]]
        + [(value, "D") for value in ["D", "d", "days", "day", "Days", "Day"]]
        + [
            (value, "m")
            for value in [
                "m",
                "minute",
                "min",
                "minutes",
                "t",
                "Minute",
                "Min",
                "Minutes",
                "T",
            ]
        ]
        + [
            (value, "s")
            for value in [
                "s",
                "seconds",
                "sec",
                "second",
                "S",
                "Seconds",
                "Sec",
                "Second",
            ]
        ]
        + [
            (value, "ms")
            for value in [
                "ms",
                "milliseconds",
                "millisecond",
                "milli",
                "millis",
                "l",
                "MS",
                "Milliseconds",
                "Millisecond",
                "Milli",
                "Millis",
                "L",
            ]
        ]
        + [
            (value, "us")
            for value in [
                "us",
                "microseconds",
                "microsecond",
                "micro",
                "micros",
                "u",
                "US",
                "Microseconds",
                "Microsecond",
                "Micro",
                "Micros",
                "U",
            ]
        ]
        + [
            (value, "ns")
            for value in [
                "ns",
                "nanoseconds",
                "nanosecond",
                "nano",
                "nanos",
                "n",
                "NS",
                "Nanoseconds",
                "Nanosecond",
                "Nano",
                "Nanos",
                "N",
            ]
        ],
    )
    @pytest.mark.parametrize("wrapper", [np.array, list, pd.Index])
    def test_unit_parser(self, unit, np_unit, wrapper):
        # validate all units, GH 6855, GH 21762
        # array-likes
        expected = TimedeltaIndex(
            [np.timedelta64(i, np_unit) for i in np.arange(5).tolist()]
        )
        result = to_timedelta(wrapper(range(5)), unit=unit)
        tm.assert_index_equal(result, expected)
        result = TimedeltaIndex(wrapper(range(5)), unit=unit)
        tm.assert_index_equal(result, expected)

        str_repr = [f"{x}{unit}" for x in np.arange(5)]
        result = to_timedelta(wrapper(str_repr))
        tm.assert_index_equal(result, expected)
        result = to_timedelta(wrapper(str_repr))
        tm.assert_index_equal(result, expected)

        # scalar
        expected = Timedelta(np.timedelta64(2, np_unit).astype("timedelta64[ns]"))
        result = to_timedelta(2, unit=unit)
        assert result == expected
        result = Timedelta(2, unit=unit)
        assert result == expected

        result = to_timedelta(f"2{unit}")
        assert result == expected
        result = Timedelta(f"2{unit}")
        assert result == expected

    @pytest.mark.parametrize("unit", ["Y", "y", "M"])
    def test_unit_m_y_raises(self, unit):
        msg = "Units 'M', 'Y', and 'y' are no longer supported"
        with pytest.raises(ValueError, match=msg):
            Timedelta(10, unit)

        with pytest.raises(ValueError, match=msg):
            to_timedelta(10, unit)

        with pytest.raises(ValueError, match=msg):
            to_timedelta([1, 2], unit)

    def test_numeric_conversions(self):
        assert Timedelta(0) == np.timedelta64(0, "ns")
        assert Timedelta(10) == np.timedelta64(10, "ns")
        assert Timedelta(10, unit="ns") == np.timedelta64(10, "ns")

        assert Timedelta(10, unit="us") == np.timedelta64(10, "us")
        assert Timedelta(10, unit="ms") == np.timedelta64(10, "ms")
        assert Timedelta(10, unit="s") == np.timedelta64(10, "s")
        assert Timedelta(10, unit="d") == np.timedelta64(10, "D")

    def test_timedelta_conversions(self):
        assert Timedelta(timedelta(seconds=1)) == np.timedelta64(1, "s").astype(
            "m8[ns]"
        )
        assert Timedelta(timedelta(microseconds=1)) == np.timedelta64(1, "us").astype(
            "m8[ns]"
        )
        assert Timedelta(timedelta(days=1)) == np.timedelta64(1, "D").astype("m8[ns]")

    def test_to_numpy_alias(self):
        # GH 24653: alias .to_numpy() for scalars
        td = Timedelta("10m7s")
        assert td.to_timedelta64() == td.to_numpy()

    @pytest.mark.parametrize(
        "freq,s1,s2",
        [
            # This first case has s1, s2 being the same as t1,t2 below
            (
                "N",
                Timedelta("1 days 02:34:56.789123456"),
                Timedelta("-1 days 02:34:56.789123456"),
            ),
            (
                "U",
                Timedelta("1 days 02:34:56.789123000"),
                Timedelta("-1 days 02:34:56.789123000"),
            ),
            (
                "L",
                Timedelta("1 days 02:34:56.789000000"),
                Timedelta("-1 days 02:34:56.789000000"),
            ),
            ("S", Timedelta("1 days 02:34:57"), Timedelta("-1 days 02:34:57")),
            ("2S", Timedelta("1 days 02:34:56"), Timedelta("-1 days 02:34:56")),
            ("5S", Timedelta("1 days 02:34:55"), Timedelta("-1 days 02:34:55")),
            ("T", Timedelta("1 days 02:35:00"), Timedelta("-1 days 02:35:00")),
            ("12T", Timedelta("1 days 02:36:00"), Timedelta("-1 days 02:36:00")),
            ("H", Timedelta("1 days 03:00:00"), Timedelta("-1 days 03:00:00")),
            ("d", Timedelta("1 days"), Timedelta("-1 days")),
        ],
    )
    def test_round(self, freq, s1, s2):

        t1 = Timedelta("1 days 02:34:56.789123456")
        t2 = Timedelta("-1 days 02:34:56.789123456")

        r1 = t1.round(freq)
        assert r1 == s1
        r2 = t2.round(freq)
        assert r2 == s2

    def test_round_invalid(self):
        t1 = Timedelta("1 days 02:34:56.789123456")

        for freq, msg in [
            ("Y", "<YearEnd: month=12> is a non-fixed frequency"),
            ("M", "<MonthEnd> is a non-fixed frequency"),
            ("foobar", "Invalid frequency: foobar"),
        ]:
            with pytest.raises(ValueError, match=msg):
                t1.round(freq)

    def test_round_implementation_bounds(self):
        # See also: analogous test for Timestamp
        # GH#38964
        result = Timedelta.min.ceil("s")
        expected = Timedelta.min + Timedelta(seconds=1) - Timedelta(145224193)
        assert result == expected

        result = Timedelta.max.floor("s")
        expected = Timedelta.max - Timedelta(854775807)
        assert result == expected

        with pytest.raises(OverflowError, match="value too large"):
            Timedelta.min.floor("s")

        # the second message here shows up in windows builds
        msg = "|".join(
            ["Python int too large to convert to C long", "int too big to convert"]
        )
        with pytest.raises(OverflowError, match=msg):
            Timedelta.max.ceil("s")

    @pytest.mark.parametrize("n", range(100))
    @pytest.mark.parametrize(
        "method", [Timedelta.round, Timedelta.floor, Timedelta.ceil]
    )
    def test_round_sanity(self, method, n, request):
        val = np.random.randint(iNaT + 1, lib.i8max, dtype=np.int64)
        td = Timedelta(val)

        assert method(td, "ns") == td

        res = method(td, "us")
        nanos = 1000
        assert np.abs((res - td).value) < nanos
        assert res.value % nanos == 0

        res = method(td, "ms")
        nanos = 1_000_000
        assert np.abs((res - td).value) < nanos
        assert res.value % nanos == 0

        res = method(td, "s")
        nanos = 1_000_000_000
        assert np.abs((res - td).value) < nanos
        assert res.value % nanos == 0

        res = method(td, "min")
        nanos = 60 * 1_000_000_000
        assert np.abs((res - td).value) < nanos
        assert res.value % nanos == 0

        res = method(td, "h")
        nanos = 60 * 60 * 1_000_000_000
        assert np.abs((res - td).value) < nanos
        assert res.value % nanos == 0

        res = method(td, "D")
        nanos = 24 * 60 * 60 * 1_000_000_000
        assert np.abs((res - td).value) < nanos
        assert res.value % nanos == 0

    def test_contains(self):
        # Checking for any NaT-like objects
        # GH 13603
        td = to_timedelta(range(5), unit="d") + offsets.Hour(1)
        for v in [NaT, None, float("nan"), np.nan]:
            assert not (v in td)

        td = to_timedelta([NaT])
        for v in [NaT, None, float("nan"), np.nan]:
            assert v in td

    def test_identity(self):

        td = Timedelta(10, unit="d")
        assert isinstance(td, Timedelta)
        assert isinstance(td, timedelta)

    def test_short_format_converters(self):
        def conv(v):
            return v.astype("m8[ns]")

        assert Timedelta("10") == np.timedelta64(10, "ns")
        assert Timedelta("10ns") == np.timedelta64(10, "ns")
        assert Timedelta("100") == np.timedelta64(100, "ns")
        assert Timedelta("100ns") == np.timedelta64(100, "ns")

        assert Timedelta("1000") == np.timedelta64(1000, "ns")
        assert Timedelta("1000ns") == np.timedelta64(1000, "ns")
        assert Timedelta("1000NS") == np.timedelta64(1000, "ns")

        assert Timedelta("10us") == np.timedelta64(10000, "ns")
        assert Timedelta("100us") == np.timedelta64(100000, "ns")
        assert Timedelta("1000us") == np.timedelta64(1000000, "ns")
        assert Timedelta("1000Us") == np.timedelta64(1000000, "ns")
        assert Timedelta("1000uS") == np.timedelta64(1000000, "ns")

        assert Timedelta("1ms") == np.timedelta64(1000000, "ns")
        assert Timedelta("10ms") == np.timedelta64(10000000, "ns")
        assert Timedelta("100ms") == np.timedelta64(100000000, "ns")
        assert Timedelta("1000ms") == np.timedelta64(1000000000, "ns")

        assert Timedelta("-1s") == -np.timedelta64(1000000000, "ns")
        assert Timedelta("1s") == np.timedelta64(1000000000, "ns")
        assert Timedelta("10s") == np.timedelta64(10000000000, "ns")
        assert Timedelta("100s") == np.timedelta64(100000000000, "ns")
        assert Timedelta("1000s") == np.timedelta64(1000000000000, "ns")

        assert Timedelta("1d") == conv(np.timedelta64(1, "D"))
        assert Timedelta("-1d") == -conv(np.timedelta64(1, "D"))
        assert Timedelta("1D") == conv(np.timedelta64(1, "D"))
        assert Timedelta("10D") == conv(np.timedelta64(10, "D"))
        assert Timedelta("100D") == conv(np.timedelta64(100, "D"))
        assert Timedelta("1000D") == conv(np.timedelta64(1000, "D"))
        assert Timedelta("10000D") == conv(np.timedelta64(10000, "D"))

        # space
        assert Timedelta(" 10000D ") == conv(np.timedelta64(10000, "D"))
        assert Timedelta(" - 10000D ") == -conv(np.timedelta64(10000, "D"))

        # invalid
        msg = "invalid unit abbreviation"
        with pytest.raises(ValueError, match=msg):
            Timedelta("1foo")
        msg = "unit abbreviation w/o a number"
        with pytest.raises(ValueError, match=msg):
            Timedelta("foo")

    def test_full_format_converters(self):
        def conv(v):
            return v.astype("m8[ns]")

        d1 = np.timedelta64(1, "D")

        assert Timedelta("1days") == conv(d1)
        assert Timedelta("1days,") == conv(d1)
        assert Timedelta("- 1days,") == -conv(d1)

        assert Timedelta("00:00:01") == conv(np.timedelta64(1, "s"))
        assert Timedelta("06:00:01") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        assert Timedelta("06:00:01.0") == conv(np.timedelta64(6 * 3600 + 1, "s"))
        assert Timedelta("06:00:01.01") == conv(
            np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        assert Timedelta("- 1days, 00:00:01") == conv(-d1 + np.timedelta64(1, "s"))
        assert Timedelta("1days, 06:00:01") == conv(
            d1 + np.timedelta64(6 * 3600 + 1, "s")
        )
        assert Timedelta("1days, 06:00:01.01") == conv(
            d1 + np.timedelta64(1000 * (6 * 3600 + 1) + 10, "ms")
        )

        # invalid
        msg = "have leftover units"
        with pytest.raises(ValueError, match=msg):
            Timedelta("- 1days, 00")

    def test_pickle(self):

        v = Timedelta("1 days 10:11:12.0123456")
        v_p = tm.round_trip_pickle(v)
        assert v == v_p

    def test_timedelta_hash_equality(self):
        # GH 11129
        v = Timedelta(1, "D")
        td = timedelta(days=1)
        assert hash(v) == hash(td)

        d = {td: 2}
        assert d[v] == 2

        tds = [Timedelta(seconds=1) + Timedelta(days=n) for n in range(20)]
        assert all(hash(td) == hash(td.to_pytimedelta()) for td in tds)

        # python timedeltas drop ns resolution
        ns_td = Timedelta(1, "ns")
        assert hash(ns_td) != hash(ns_td.to_pytimedelta())

    def test_implementation_limits(self):
        min_td = Timedelta(Timedelta.min)
        max_td = Timedelta(Timedelta.max)

        # GH 12727
        # timedelta limits correspond to int64 boundaries
        assert min_td.value == iNaT + 1
        assert max_td.value == lib.i8max

        # Beyond lower limit, a NAT before the Overflow
        assert (min_td - Timedelta(1, "ns")) is NaT

        msg = "int too (large|big) to convert"
        with pytest.raises(OverflowError, match=msg):
            min_td - Timedelta(2, "ns")

        with pytest.raises(OverflowError, match=msg):
            max_td + Timedelta(1, "ns")

        # Same tests using the internal nanosecond values
        td = Timedelta(min_td.value - 1, "ns")
        assert td is NaT

        with pytest.raises(OverflowError, match=msg):
            Timedelta(min_td.value - 2, "ns")

        with pytest.raises(OverflowError, match=msg):
            Timedelta(max_td.value + 1, "ns")

    def test_total_seconds_precision(self):
        # GH 19458
        assert Timedelta("30S").total_seconds() == 30.0
        assert Timedelta("0").total_seconds() == 0.0
        assert Timedelta("-2S").total_seconds() == -2.0
        assert Timedelta("5.324S").total_seconds() == 5.324
        assert (Timedelta("30S").total_seconds() - 30.0) < 1e-20
        assert (30.0 - Timedelta("30S").total_seconds()) < 1e-20

    def test_resolution_string(self):
        assert Timedelta(days=1).resolution_string == "D"
        assert Timedelta(days=1, hours=6).resolution_string == "H"
        assert Timedelta(days=1, minutes=6).resolution_string == "T"
        assert Timedelta(days=1, seconds=6).resolution_string == "S"
        assert Timedelta(days=1, milliseconds=6).resolution_string == "L"
        assert Timedelta(days=1, microseconds=6).resolution_string == "U"
        assert Timedelta(days=1, nanoseconds=6).resolution_string == "N"

    def test_resolution_deprecated(self):
        # GH#21344
        td = Timedelta(days=4, hours=3)
        result = td.resolution
        assert result == Timedelta(nanoseconds=1)

        # Check that the attribute is available on the class, mirroring
        #  the stdlib timedelta behavior
        result = Timedelta.resolution
        assert result == Timedelta(nanoseconds=1)


@pytest.mark.parametrize(
    "value, expected",
    [
        (Timedelta("10S"), True),
        (Timedelta("-10S"), True),
        (Timedelta(10, unit="ns"), True),
        (Timedelta(0, unit="ns"), False),
        (Timedelta(-10, unit="ns"), True),
        (Timedelta(None), True),
        (NaT, True),
    ],
)
def test_truthiness(value, expected):
    # https://github.com/pandas-dev/pandas/issues/21484
    assert bool(value) is expected


def test_timedelta_attribute_precision():
    # GH 31354
    td = Timedelta(1552211999999999872, unit="ns")
    result = td.days * 86400
    result += td.seconds
    result *= 1000000
    result += td.microseconds
    result *= 1000
    result += td.nanoseconds
    expected = td.value
    assert result == expected
