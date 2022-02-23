from datetime import (
    datetime,
    timedelta,
    timezone,
)

import numpy as np
import pytest
import pytz

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


class TestSeriesFillNA:
    def test_fillna_nat(self):
        series = Series([0, 1, 2, NaT.value], dtype="M8[ns]")

        filled = series.fillna(method="pad")
        filled2 = series.fillna(value=series.values[2])

        expected = series.copy()
        expected.values[3] = expected.values[2]

        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)

        df = DataFrame({"A": series})
        filled = df.fillna(method="pad")
        filled2 = df.fillna(value=series.values[2])
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

        series = Series([NaT.value, 0, 1, 2], dtype="M8[ns]")

        filled = series.fillna(method="bfill")
        filled2 = series.fillna(value=series[1])

        expected = series.copy()
        expected[0] = expected[1]

        tm.assert_series_equal(filled, expected)
        tm.assert_series_equal(filled2, expected)

        df = DataFrame({"A": series})
        filled = df.fillna(method="bfill")
        filled2 = df.fillna(value=series[1])
        expected = DataFrame({"A": expected})
        tm.assert_frame_equal(filled, expected)
        tm.assert_frame_equal(filled2, expected)

    def test_fillna(self, datetime_series):
        ts = Series([0.0, 1.0, 2.0, 3.0, 4.0], index=tm.makeDateIndex(5))

        tm.assert_series_equal(ts, ts.fillna(method="ffill"))

        ts[2] = np.NaN

        exp = Series([0.0, 1.0, 1.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(method="ffill"), exp)

        exp = Series([0.0, 1.0, 3.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(method="backfill"), exp)

        exp = Series([0.0, 1.0, 5.0, 3.0, 4.0], index=ts.index)
        tm.assert_series_equal(ts.fillna(value=5), exp)

        msg = "Must specify a fill 'value' or 'method'"
        with pytest.raises(ValueError, match=msg):
            ts.fillna()

        msg = "Cannot specify both 'value' and 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_series.fillna(value=0, method="ffill")

        # GH#5703
        s1 = Series([np.nan])
        s2 = Series([1])
        result = s1.fillna(s2)
        expected = Series([1.0])
        tm.assert_series_equal(result, expected)
        result = s1.fillna({})
        tm.assert_series_equal(result, s1)
        result = s1.fillna(Series((), dtype=object))
        tm.assert_series_equal(result, s1)
        result = s2.fillna(s1)
        tm.assert_series_equal(result, s2)
        result = s1.fillna({0: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna({1: 1})
        tm.assert_series_equal(result, Series([np.nan]))
        result = s1.fillna({0: 1, 1: 1})
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}))
        tm.assert_series_equal(result, expected)
        result = s1.fillna(Series({0: 1, 1: 1}, index=[4, 5]))
        tm.assert_series_equal(result, s1)

        s1 = Series([0, 1, 2], list("abc"))
        s2 = Series([0, np.nan, 2], list("bac"))
        result = s2.fillna(s1)
        expected = Series([0, 0, 2.0], list("bac"))
        tm.assert_series_equal(result, expected)

        # limit
        ser = Series(np.nan, index=[0, 1, 2])
        result = ser.fillna(999, limit=1)
        expected = Series([999, np.nan, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        result = ser.fillna(999, limit=2)
        expected = Series([999, 999, np.nan], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        # GH#9043
        # make sure a string representation of int/float values can be filled
        # correctly without raising errors or being converted
        vals = ["0", "1.5", "-0.3"]
        for val in vals:
            ser = Series([0, 1, np.nan, np.nan, 4], dtype="float64")
            result = ser.fillna(val)
            expected = Series([0, 1, val, val, 4], dtype="object")
            tm.assert_series_equal(result, expected)

    def test_fillna_consistency(self):
        # GH#16402
        # fillna with a tz aware to a tz-naive, should result in object

        ser = Series([Timestamp("20130101"), NaT])

        result = ser.fillna(Timestamp("20130101", tz="US/Eastern"))
        expected = Series(
            [Timestamp("20130101"), Timestamp("2013-01-01", tz="US/Eastern")],
            dtype="object",
        )
        tm.assert_series_equal(result, expected)

        # where (we ignore the errors=)
        result = ser.where(
            [True, False], Timestamp("20130101", tz="US/Eastern"), errors="ignore"
        )
        tm.assert_series_equal(result, expected)

        result = ser.where(
            [True, False], Timestamp("20130101", tz="US/Eastern"), errors="ignore"
        )
        tm.assert_series_equal(result, expected)

        # with a non-datetime
        result = ser.fillna("foo")
        expected = Series([Timestamp("20130101"), "foo"])
        tm.assert_series_equal(result, expected)

        # assignment
        ser2 = ser.copy()
        ser2[1] = "foo"
        tm.assert_series_equal(ser2, expected)

    def test_fillna_downcast(self):
        # GH#15277
        # infer int64 from float64
        ser = Series([1.0, np.nan])
        result = ser.fillna(0, downcast="infer")
        expected = Series([1, 0])
        tm.assert_series_equal(result, expected)

        # infer int64 from float64 when fillna value is a dict
        ser = Series([1.0, np.nan])
        result = ser.fillna({1: 0}, downcast="infer")
        expected = Series([1, 0])
        tm.assert_series_equal(result, expected)

    def test_timedelta_fillna(self, frame_or_series):
        # GH#3371
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        td = ser.diff()
        obj = frame_or_series(td)

        # reg fillna
        result = obj.fillna(Timedelta(seconds=0))
        expected = Series(
            [
                timedelta(0),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        # interpreted as seconds, no longer supported
        msg = "value should be a 'Timedelta', 'NaT', or array of those. Got 'int'"
        with pytest.raises(TypeError, match=msg):
            obj.fillna(1)

        result = obj.fillna(Timedelta(seconds=1))
        expected = Series(
            [
                timedelta(seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(timedelta(days=1, seconds=1))
        expected = Series(
            [
                timedelta(days=1, seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(np.timedelta64(10 ** 9))
        expected = Series(
            [
                timedelta(seconds=1),
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(NaT)
        expected = Series(
            [
                NaT,
                timedelta(0),
                timedelta(1),
                timedelta(days=1, seconds=9 * 3600 + 60 + 1),
            ],
            dtype="m8[ns]",
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        # ffill
        td[2] = np.nan
        obj = frame_or_series(td)
        result = obj.ffill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[0] = np.nan
        expected = frame_or_series(expected)

        tm.assert_equal(result, expected)

        # bfill
        td[2] = np.nan
        obj = frame_or_series(td)
        result = obj.bfill()
        expected = td.fillna(Timedelta(seconds=0))
        expected[2] = timedelta(days=1, seconds=9 * 3600 + 60 + 1)
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

    def test_datetime64_fillna(self):

        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan

        # ffill
        result = ser.ffill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

        # bfill
        result = ser.bfill()
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130103 9:01:01"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        tm.assert_series_equal(result, expected)

        # GH#6587
        # make sure that we are treating as integer when filling
        msg = "containing strings is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # this also tests inference of a datetime-like with NaT's
            ser = Series([NaT, NaT, "2013-08-05 15:30:00.000001"])

        expected = Series(
            [
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
                "2013-08-05 15:30:00.000001",
            ],
            dtype="M8[ns]",
        )
        result = ser.fillna(method="backfill")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo"])
    def test_datetime64_tz_fillna(self, tz):
        # DatetimeLikeBlock
        ser = Series(
            [
                Timestamp("2011-01-01 10:00"),
                NaT,
                Timestamp("2011-01-03 10:00"),
                NaT,
            ]
        )
        null_loc = Series([False, True, False, True])

        result = ser.fillna(Timestamp("2011-01-02 10:00"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-02 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        # check s is not changed
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-02 10:00", tz=tz),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna("AAA")
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                "AAA",
                Timestamp("2011-01-03 10:00"),
                "AAA",
            ],
            dtype=object,
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz),
                3: Timestamp("2011-01-04 10:00"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-04 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {1: Timestamp("2011-01-02 10:00"), 3: Timestamp("2011-01-04 10:00")}
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00"),
                Timestamp("2011-01-04 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # DatetimeTZBlock
        idx = DatetimeIndex(["2011-01-01 10:00", NaT, "2011-01-03 10:00", NaT], tz=tz)
        ser = Series(idx)
        assert ser.dtype == f"datetime64[ns, {tz}]"
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00"),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-02 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz))
        idx = DatetimeIndex(
            [
                "2011-01-01 10:00",
                "2011-01-02 10:00",
                "2011-01-03 10:00",
                "2011-01-02 10:00",
            ],
            tz=tz,
        )
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("2011-01-02 10:00", tz=tz).to_pydatetime())
        idx = DatetimeIndex(
            [
                "2011-01-01 10:00",
                "2011-01-02 10:00",
                "2011-01-03 10:00",
                "2011-01-02 10:00",
            ],
            tz=tz,
        )
        expected = Series(idx)
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna("AAA")
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                "AAA",
                Timestamp("2011-01-03 10:00", tz=tz),
                "AAA",
            ],
            dtype=object,
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz),
                3: Timestamp("2011-01-04 10:00"),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-04 10:00"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(
            {
                1: Timestamp("2011-01-02 10:00", tz=tz),
                3: Timestamp("2011-01-04 10:00", tz=tz),
            }
        )
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2011-01-02 10:00", tz=tz),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2011-01-04 10:00", tz=tz),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        # filling with a naive/other zone, coerce to object
        result = ser.fillna(Timestamp("20130101"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2013-01-01"),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2013-01-01"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

        result = ser.fillna(Timestamp("20130101", tz="US/Pacific"))
        expected = Series(
            [
                Timestamp("2011-01-01 10:00", tz=tz),
                Timestamp("2013-01-01", tz="US/Pacific"),
                Timestamp("2011-01-03 10:00", tz=tz),
                Timestamp("2013-01-01", tz="US/Pacific"),
            ]
        )
        tm.assert_series_equal(expected, result)
        tm.assert_series_equal(isna(ser), null_loc)

    def test_fillna_dt64tz_with_method(self):
        # with timezone
        # GH#15855
        ser = Series([Timestamp("2012-11-11 00:00:00+01:00"), NaT])
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00"),
                Timestamp("2012-11-11 00:00:00+01:00"),
            ]
        )
        tm.assert_series_equal(ser.fillna(method="pad"), exp)

        ser = Series([NaT, Timestamp("2012-11-11 00:00:00+01:00")])
        exp = Series(
            [
                Timestamp("2012-11-11 00:00:00+01:00"),
                Timestamp("2012-11-11 00:00:00+01:00"),
            ]
        )
        tm.assert_series_equal(ser.fillna(method="bfill"), exp)

    def test_fillna_pytimedelta(self):
        # GH#8209
        ser = Series([np.nan, Timedelta("1 days")], index=["A", "B"])

        result = ser.fillna(timedelta(1))
        expected = Series(Timedelta("1 days"), index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_fillna_period(self):
        # GH#13737
        ser = Series([Period("2011-01", freq="M"), Period("NaT", freq="M")])

        res = ser.fillna(Period("2012-01", freq="M"))
        exp = Series([Period("2011-01", freq="M"), Period("2012-01", freq="M")])
        tm.assert_series_equal(res, exp)
        assert res.dtype == "Period[M]"

    def test_fillna_dt64_timestamp(self, frame_or_series):
        ser = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130102"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        ser[2] = np.nan
        obj = frame_or_series(ser)

        # reg fillna
        result = obj.fillna(Timestamp("20130104"))
        expected = Series(
            [
                Timestamp("20130101"),
                Timestamp("20130101"),
                Timestamp("20130104"),
                Timestamp("20130103 9:01:01"),
            ]
        )
        expected = frame_or_series(expected)
        tm.assert_equal(result, expected)

        result = obj.fillna(NaT)
        expected = obj
        tm.assert_equal(result, expected)

    def test_fillna_dt64_non_nao(self):
        # GH#27419
        ser = Series([Timestamp("2010-01-01"), NaT, Timestamp("2000-01-01")])
        val = np.datetime64("1975-04-05", "ms")

        result = ser.fillna(val)
        expected = Series(
            [Timestamp("2010-01-01"), Timestamp("1975-04-05"), Timestamp("2000-01-01")]
        )
        tm.assert_series_equal(result, expected)

    def test_fillna_numeric_inplace(self):
        x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        y = x.copy()

        return_value = y.fillna(value=0, inplace=True)
        assert return_value is None

        expected = x.fillna(value=0)
        tm.assert_series_equal(y, expected)

    # ---------------------------------------------------------------
    # CategoricalDtype

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            ("a", ["a", "a", "b", "a", "a"]),
            ({1: "a", 3: "b", 4: "b"}, ["a", "a", "b", "b", "b"]),
            ({1: "a"}, ["a", "a", "b", np.nan, np.nan]),
            ({1: "a", 3: "b"}, ["a", "a", "b", "b", np.nan]),
            (Series("a"), ["a", np.nan, "b", np.nan, np.nan]),
            (Series("a", index=[1]), ["a", "a", "b", np.nan, np.nan]),
            (Series({1: "a", 3: "b"}), ["a", "a", "b", "b", np.nan]),
            (Series(["a", "b"], index=[3, 4]), ["a", np.nan, "b", "a", "b"]),
        ],
    )
    def test_fillna_categorical(self, fill_value, expected_output):
        # GH#17033
        # Test fillna for a Categorical series
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b"]))
        exp = Series(Categorical(expected_output, categories=["a", "b"]))
        result = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "fill_value, expected_output",
        [
            (Series(["a", "b", "c", "d", "e"]), ["a", "b", "b", "d", "e"]),
            (Series(["b", "d", "a", "d", "a"]), ["a", "d", "b", "d", "a"]),
            (
                Series(
                    Categorical(
                        ["b", "d", "a", "d", "a"], categories=["b", "c", "d", "e", "a"]
                    )
                ),
                ["a", "d", "b", "d", "a"],
            ),
        ],
    )
    def test_fillna_categorical_with_new_categories(self, fill_value, expected_output):
        # GH#26215
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b", "c", "d", "e"]))
        exp = Series(Categorical(expected_output, categories=["a", "b", "c", "d", "e"]))
        result = ser.fillna(fill_value)
        tm.assert_series_equal(result, exp)

    def test_fillna_categorical_raises(self):
        data = ["a", np.nan, "b", np.nan, np.nan]
        ser = Series(Categorical(data, categories=["a", "b"]))
        cat = ser._values

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(ValueError, match=msg):
            ser.fillna("d")

        msg2 = "Length of 'value' does not match."
        with pytest.raises(ValueError, match=msg2):
            cat.fillna(Series("d"))

        with pytest.raises(ValueError, match=msg):
            ser.fillna({1: "d", 3: "a"})

        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(["a", "b"])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna(("a", "b"))

        msg = (
            '"value" parameter must be a scalar, dict '
            'or Series, but you passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            ser.fillna(DataFrame({1: ["a"], 3: ["b"]}))

    @pytest.mark.parametrize("dtype", [float, "float32", "float64"])
    @pytest.mark.parametrize("fill_type", tm.ALL_REAL_DTYPES)
    def test_fillna_float_casting(self, dtype, fill_type):
        # GH-43424
        ser = Series([np.nan, 1.2], dtype=dtype)
        fill_values = Series([2, 2], dtype=fill_type)
        result = ser.fillna(fill_values)
        expected = Series([2.0, 1.2], dtype=dtype)
        tm.assert_series_equal(result, expected)

    def test_fillna_f32_upcast_with_dict(self):
        # GH-43424
        ser = Series([np.nan, 1.2], dtype=np.float32)
        result = ser.fillna({0: 1})
        expected = Series([1.0, 1.2], dtype=np.float32)
        tm.assert_series_equal(result, expected)

    # ---------------------------------------------------------------
    # Invalid Usages

    def test_fillna_invalid_method(self, datetime_series):
        try:
            datetime_series.fillna(method="ffil")
        except ValueError as inst:
            assert "ffil" in str(inst)

    def test_fillna_listlike_invalid(self):
        ser = Series(np.random.randint(-100, 100, 50))
        msg = '"value" parameter must be a scalar or dict, but you passed a "list"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna([1, 2])

        msg = '"value" parameter must be a scalar or dict, but you passed a "tuple"'
        with pytest.raises(TypeError, match=msg):
            ser.fillna((1, 2))

    def test_fillna_method_and_limit_invalid(self):

        # related GH#9217, make sure limit is an int and greater than 0
        ser = Series([1, 2, 3, None])
        msg = (
            r"Cannot specify both 'value' and 'method'\.|"
            r"Limit must be greater than 0|"
            "Limit must be an integer"
        )
        for limit in [-1, 0, 1.0, 2.0]:
            for method in ["backfill", "bfill", "pad", "ffill", None]:
                with pytest.raises(ValueError, match=msg):
                    ser.fillna(1, limit=limit, method=method)

    def test_fillna_datetime64_with_timezone_tzinfo(self):
        # https://github.com/pandas-dev/pandas/issues/38851
        # different tzinfos representing UTC treated as equal
        ser = Series(date_range("2020", periods=3, tz="UTC"))
        expected = ser.copy()
        ser[1] = NaT
        result = ser.fillna(datetime(2020, 1, 2, tzinfo=timezone.utc))
        tm.assert_series_equal(result, expected)

        # but we dont (yet) consider distinct tzinfos for non-UTC tz equivalent
        ts = Timestamp("2000-01-01", tz="US/Pacific")
        ser2 = Series(ser._values.tz_convert("dateutil/US/Pacific"))
        result = ser2.fillna(ts)
        expected = Series([ser[0], ts, ser[2]], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_fillna_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        srs = Series([1, 2, 3, np.nan], dtype=float)
        msg = (
            r"In a future version of pandas all arguments of Series.fillna "
            r"except for the argument 'value' will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = srs.fillna(0, None, None)
        expected = Series([1, 2, 3, 0], dtype=float)
        tm.assert_series_equal(result, expected)


class TestFillnaPad:
    def test_fillna_bug(self):
        ser = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"])
        filled = ser.fillna(method="ffill")
        expected = Series([np.nan, 1.0, 1.0, 3.0, 3.0], ser.index)
        tm.assert_series_equal(filled, expected)

        filled = ser.fillna(method="bfill")
        expected = Series([1.0, 1.0, 3.0, 3.0, np.nan], ser.index)
        tm.assert_series_equal(filled, expected)

    def test_ffill(self):
        ts = Series([0.0, 1.0, 2.0, 3.0, 4.0], index=tm.makeDateIndex(5))
        ts[2] = np.NaN
        tm.assert_series_equal(ts.ffill(), ts.fillna(method="ffill"))

    def test_ffill_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        ser = Series([1, 2, 3])
        msg = (
            r"In a future version of pandas all arguments of Series.ffill "
            r"will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.ffill(0)
        expected = Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_ffill_mixed_dtypes_without_missing_data(self):
        # GH#14956
        series = Series([datetime(2015, 1, 1, tzinfo=pytz.utc), 1])
        result = series.ffill()
        tm.assert_series_equal(series, result)

    def test_bfill(self):
        ts = Series([0.0, 1.0, 2.0, 3.0, 4.0], index=tm.makeDateIndex(5))
        ts[2] = np.NaN
        tm.assert_series_equal(ts.bfill(), ts.fillna(method="bfill"))

    def test_bfill_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        ser = Series([1, 2, 3])
        msg = (
            r"In a future version of pandas all arguments of Series.bfill "
            r"will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = ser.bfill(0)
        expected = Series([1, 2, 3])
        tm.assert_series_equal(result, expected)

    def test_pad_nan(self):
        x = Series(
            [np.nan, 1.0, np.nan, 3.0, np.nan], ["z", "a", "b", "c", "d"], dtype=float
        )

        return_value = x.fillna(method="pad", inplace=True)
        assert return_value is None

        expected = Series(
            [np.nan, 1.0, 1.0, 3.0, 3.0], ["z", "a", "b", "c", "d"], dtype=float
        )
        tm.assert_series_equal(x[1:], expected[1:])
        assert np.isnan(x[0]), np.isnan(expected[0])

    def test_series_fillna_limit(self):
        index = np.arange(10)
        s = Series(np.random.randn(10), index=index)

        result = s[:2].reindex(index)
        result = result.fillna(method="pad", limit=5)

        expected = s[:2].reindex(index).fillna(method="pad")
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        result = s[-2:].reindex(index)
        result = result.fillna(method="bfill", limit=5)

        expected = s[-2:].reindex(index).fillna(method="backfill")
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_series_pad_backfill_limit(self):
        index = np.arange(10)
        s = Series(np.random.randn(10), index=index)

        result = s[:2].reindex(index, method="pad", limit=5)

        expected = s[:2].reindex(index).fillna(method="pad")
        expected[-3:] = np.nan
        tm.assert_series_equal(result, expected)

        result = s[-2:].reindex(index, method="backfill", limit=5)

        expected = s[-2:].reindex(index).fillna(method="backfill")
        expected[:3] = np.nan
        tm.assert_series_equal(result, expected)

    def test_fillna_int(self):
        ser = Series(np.random.randint(-100, 100, 50))
        return_value = ser.fillna(method="ffill", inplace=True)
        assert return_value is None
        tm.assert_series_equal(ser.fillna(method="ffill", inplace=False), ser)

    def test_datetime64tz_fillna_round_issue(self):
        # GH#14872

        data = Series(
            [NaT, NaT, datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc)]
        )

        filled = data.fillna(method="bfill")

        expected = Series(
            [
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc),
                datetime(2016, 12, 12, 22, 24, 6, 100001, tzinfo=pytz.utc),
            ]
        )

        tm.assert_series_equal(filled, expected)
