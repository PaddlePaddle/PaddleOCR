import numpy as np
import pytest

from pandas.errors import NullFrequencyError

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    NaT,
    Series,
    TimedeltaIndex,
    date_range,
    offsets,
)
import pandas._testing as tm

from pandas.tseries.offsets import BDay


class TestShift:
    @pytest.mark.parametrize(
        "ser",
        [
            Series([np.arange(5)]),
            date_range("1/1/2011", periods=24, freq="H"),
            Series(range(5), index=date_range("2017", periods=5)),
        ],
    )
    @pytest.mark.parametrize("shift_size", [0, 1, 2])
    def test_shift_always_copy(self, ser, shift_size):
        # GH22397
        assert ser.shift(shift_size) is not ser

    @pytest.mark.parametrize("move_by_freq", [pd.Timedelta("1D"), pd.Timedelta("1min")])
    def test_datetime_shift_always_copy(self, move_by_freq):
        # GH#22397
        ser = Series(range(5), index=date_range("2017", periods=5))
        assert ser.shift(freq=move_by_freq) is not ser

    def test_shift(self, datetime_series):
        shifted = datetime_series.shift(1)
        unshifted = shifted.shift(-1)

        tm.assert_index_equal(shifted.index, datetime_series.index)
        tm.assert_index_equal(unshifted.index, datetime_series.index)
        tm.assert_numpy_array_equal(
            unshifted.dropna().values, datetime_series.values[:-1]
        )

        offset = BDay()
        shifted = datetime_series.shift(1, freq=offset)
        unshifted = shifted.shift(-1, freq=offset)

        tm.assert_series_equal(unshifted, datetime_series)

        unshifted = datetime_series.shift(0, freq=offset)
        tm.assert_series_equal(unshifted, datetime_series)

        shifted = datetime_series.shift(1, freq="B")
        unshifted = shifted.shift(-1, freq="B")

        tm.assert_series_equal(unshifted, datetime_series)

        # corner case
        unshifted = datetime_series.shift(0)
        tm.assert_series_equal(unshifted, datetime_series)

        # Shifting with PeriodIndex
        ps = tm.makePeriodSeries()
        shifted = ps.shift(1)
        unshifted = shifted.shift(-1)
        tm.assert_index_equal(shifted.index, ps.index)
        tm.assert_index_equal(unshifted.index, ps.index)
        tm.assert_numpy_array_equal(unshifted.dropna().values, ps.values[:-1])

        shifted2 = ps.shift(1, "B")
        shifted3 = ps.shift(1, BDay())
        tm.assert_series_equal(shifted2, shifted3)
        tm.assert_series_equal(ps, shifted2.shift(-1, "B"))

        msg = "Given freq D does not match PeriodIndex freq B"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="D")

        # legacy support
        shifted4 = ps.shift(1, freq="B")
        tm.assert_series_equal(shifted2, shifted4)

        shifted5 = ps.shift(1, freq=BDay())
        tm.assert_series_equal(shifted5, shifted4)

        # 32-bit taking
        # GH#8129
        index = date_range("2000-01-01", periods=5)
        for dtype in ["int32", "int64"]:
            s1 = Series(np.arange(5, dtype=dtype), index=index)
            p = s1.iloc[1]
            result = s1.shift(periods=p)
            expected = Series([np.nan, 0, 1, 2, 3], index=index)
            tm.assert_series_equal(result, expected)

        # GH#8260
        # with tz
        s = Series(
            date_range("2000-01-01 09:00:00", periods=5, tz="US/Eastern"), name="foo"
        )
        result = s - s.shift()

        exp = Series(TimedeltaIndex(["NaT"] + ["1 days"] * 4), name="foo")
        tm.assert_series_equal(result, exp)

        # incompat tz
        s2 = Series(date_range("2000-01-01 09:00:00", periods=5, tz="CET"), name="foo")
        msg = "DatetimeArray subtraction must have the same timezones or no timezones"
        with pytest.raises(TypeError, match=msg):
            s - s2

    def test_shift2(self):
        ts = Series(
            np.random.randn(5), index=date_range("1/1/2000", periods=5, freq="H")
        )

        result = ts.shift(1, freq="5T")
        exp_index = ts.index.shift(1, freq="5T")
        tm.assert_index_equal(result.index, exp_index)

        # GH#1063, multiple of same base
        result = ts.shift(1, freq="4H")
        exp_index = ts.index + offsets.Hour(4)
        tm.assert_index_equal(result.index, exp_index)

        idx = DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"])
        msg = "Cannot shift with no freq"
        with pytest.raises(NullFrequencyError, match=msg):
            idx.shift(1)

    def test_shift_fill_value(self):
        # GH#24128
        ts = Series(
            [1.0, 2.0, 3.0, 4.0, 5.0], index=date_range("1/1/2000", periods=5, freq="H")
        )

        exp = Series(
            [0.0, 1.0, 2.0, 3.0, 4.0], index=date_range("1/1/2000", periods=5, freq="H")
        )
        # check that fill value works
        result = ts.shift(1, fill_value=0.0)
        tm.assert_series_equal(result, exp)

        exp = Series(
            [0.0, 0.0, 1.0, 2.0, 3.0], index=date_range("1/1/2000", periods=5, freq="H")
        )
        result = ts.shift(2, fill_value=0.0)
        tm.assert_series_equal(result, exp)

        ts = Series([1, 2, 3])
        res = ts.shift(2, fill_value=0)
        assert res.dtype == ts.dtype

    def test_shift_categorical_fill_value(self):
        ts = Series(["a", "b", "c", "d"], dtype="category")
        res = ts.shift(1, fill_value="a")
        expected = Series(
            pd.Categorical(
                ["a", "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
            )
        )
        tm.assert_equal(res, expected)

        # check for incorrect fill_value
        msg = "'fill_value=f' is not present in this Categorical's categories"
        with pytest.raises(TypeError, match=msg):
            ts.shift(1, fill_value="f")

    def test_shift_dst(self):
        # GH#13926
        dates = date_range("2016-11-06", freq="H", periods=10, tz="US/Eastern")
        s = Series(dates)

        res = s.shift(0)
        tm.assert_series_equal(res, s)
        assert res.dtype == "datetime64[ns, US/Eastern]"

        res = s.shift(1)
        exp_vals = [NaT] + dates.astype(object).values.tolist()[:9]
        exp = Series(exp_vals)
        tm.assert_series_equal(res, exp)
        assert res.dtype == "datetime64[ns, US/Eastern]"

        res = s.shift(-2)
        exp_vals = dates.astype(object).values.tolist()[2:] + [NaT, NaT]
        exp = Series(exp_vals)
        tm.assert_series_equal(res, exp)
        assert res.dtype == "datetime64[ns, US/Eastern]"

        for ex in [10, -10, 20, -20]:
            res = s.shift(ex)
            exp = Series([NaT] * 10, dtype="datetime64[ns, US/Eastern]")
            tm.assert_series_equal(res, exp)
            assert res.dtype == "datetime64[ns, US/Eastern]"

    @pytest.mark.filterwarnings("ignore:tshift is deprecated:FutureWarning")
    def test_tshift(self, datetime_series):
        # TODO: remove this test when tshift deprecation is enforced

        # PeriodIndex
        ps = tm.makePeriodSeries()
        shifted = ps.tshift(1)
        unshifted = shifted.tshift(-1)

        tm.assert_series_equal(unshifted, ps)

        shifted2 = ps.tshift(freq="B")
        tm.assert_series_equal(shifted, shifted2)

        shifted3 = ps.tshift(freq=BDay())
        tm.assert_series_equal(shifted, shifted3)

        msg = "Given freq M does not match PeriodIndex freq B"
        with pytest.raises(ValueError, match=msg):
            ps.tshift(freq="M")

        # DatetimeIndex
        shifted = datetime_series.tshift(1)
        unshifted = shifted.tshift(-1)

        tm.assert_series_equal(datetime_series, unshifted)

        shifted2 = datetime_series.tshift(freq=datetime_series.index.freq)
        tm.assert_series_equal(shifted, shifted2)

        inferred_ts = Series(
            datetime_series.values, Index(np.asarray(datetime_series.index)), name="ts"
        )
        shifted = inferred_ts.tshift(1)
        expected = datetime_series.tshift(1)
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(shifted, expected)

        unshifted = shifted.tshift(-1)
        tm.assert_series_equal(unshifted, inferred_ts)

        no_freq = datetime_series[[0, 5, 7]]
        msg = "Freq was not set in the index hence cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            no_freq.tshift()

    def test_tshift_deprecated(self, datetime_series):
        # GH#11631
        with tm.assert_produces_warning(FutureWarning):
            datetime_series.tshift()

    def test_period_index_series_shift_with_freq(self):
        ps = tm.makePeriodSeries()

        shifted = ps.shift(1, freq="infer")
        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_series_equal(unshifted, ps)

        shifted2 = ps.shift(freq="B")
        tm.assert_series_equal(shifted, shifted2)

        shifted3 = ps.shift(freq=BDay())
        tm.assert_series_equal(shifted, shifted3)

    def test_datetime_series_shift_with_freq(self, datetime_series):
        shifted = datetime_series.shift(1, freq="infer")
        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_series_equal(datetime_series, unshifted)

        shifted2 = datetime_series.shift(freq=datetime_series.index.freq)
        tm.assert_series_equal(shifted, shifted2)

        inferred_ts = Series(
            datetime_series.values, Index(np.asarray(datetime_series.index)), name="ts"
        )
        shifted = inferred_ts.shift(1, freq="infer")
        expected = datetime_series.shift(1, freq="infer")
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(shifted, expected)

        unshifted = shifted.shift(-1, freq="infer")
        tm.assert_series_equal(unshifted, inferred_ts)

    def test_period_index_series_shift_with_freq_error(self):
        ps = tm.makePeriodSeries()
        msg = "Given freq M does not match PeriodIndex freq B"
        with pytest.raises(ValueError, match=msg):
            ps.shift(freq="M")

    def test_datetime_series_shift_with_freq_error(self, datetime_series):
        no_freq = datetime_series[[0, 5, 7]]
        msg = "Freq was not set in the index hence cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            no_freq.shift(freq="infer")

    def test_shift_int(self, datetime_series):
        ts = datetime_series.astype(int)
        shifted = ts.shift(1)
        expected = ts.astype(float).shift(1)
        tm.assert_series_equal(shifted, expected)

    def test_shift_object_non_scalar_fill(self):
        # shift requires scalar fill_value except for object dtype
        ser = Series(range(3))
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            ser.shift(1, fill_value=[])

        df = ser.to_frame()
        with pytest.raises(ValueError, match="fill_value must be a scalar"):
            df.shift(1, fill_value=np.arange(3))

        obj_ser = ser.astype(object)
        result = obj_ser.shift(1, fill_value={})
        assert result[0] == {}

        obj_df = obj_ser.to_frame()
        result = obj_df.shift(1, fill_value={})
        assert result.iloc[0, 0] == {}

    def test_shift_categorical(self):
        # GH#9416
        s = Series(["a", "b", "c", "d"], dtype="category")

        tm.assert_series_equal(s.iloc[:-1], s.shift(1).shift(-1).dropna())

        sp1 = s.shift(1)
        tm.assert_index_equal(s.index, sp1.index)
        assert np.all(sp1.values.codes[:1] == -1)
        assert np.all(s.values.codes[:-1] == sp1.values.codes[1:])

        sn2 = s.shift(-2)
        tm.assert_index_equal(s.index, sn2.index)
        assert np.all(sn2.values.codes[-2:] == -1)
        assert np.all(s.values.codes[2:] == sn2.values.codes[:-2])

        tm.assert_index_equal(s.values.categories, sp1.values.categories)
        tm.assert_index_equal(s.values.categories, sn2.values.categories)

    def test_shift_dt64values_int_fill_deprecated(self):
        # GH#31971
        ser = Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")])

        with tm.assert_produces_warning(FutureWarning):
            result = ser.shift(1, fill_value=0)

        expected = Series([pd.Timestamp(0), ser[0]])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("periods", [1, 2, 3, 4])
    def test_shift_preserve_freqstr(self, periods):
        # GH#21275
        ser = Series(
            range(periods),
            index=date_range("2016-1-1 00:00:00", periods=periods, freq="H"),
        )

        result = ser.shift(1, "2H")

        expected = Series(
            range(periods),
            index=date_range("2016-1-1 02:00:00", periods=periods, freq="H"),
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "input_data, output_data",
        [(np.empty(shape=(0,)), []), (np.ones(shape=(2,)), [np.nan, 1.0])],
    )
    def test_shift_non_writable_array(self, input_data, output_data):
        # GH21049 Verify whether non writable numpy array is shiftable
        input_data.setflags(write=False)

        result = Series(input_data).shift(1)
        expected = Series(output_data, dtype="float64")

        tm.assert_series_equal(result, expected)
