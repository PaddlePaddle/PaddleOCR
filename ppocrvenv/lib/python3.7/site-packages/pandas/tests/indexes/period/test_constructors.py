import numpy as np
import pytest

from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
    Series,
    date_range,
    offsets,
    period_range,
)
import pandas._testing as tm
from pandas.core.arrays import PeriodArray


class TestPeriodIndex:
    def test_construction_base_constructor(self):
        # GH 13664
        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")]
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        arr = [np.nan, NaT, Period("2011-03", freq="M")]
        tm.assert_index_equal(Index(arr), PeriodIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), PeriodIndex(np.array(arr)))

        arr = [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="D")]
        tm.assert_index_equal(Index(arr), Index(arr, dtype=object))

        tm.assert_index_equal(Index(np.array(arr)), Index(np.array(arr), dtype=object))

    def test_base_constructor_with_period_dtype(self):
        dtype = PeriodDtype("D")
        values = ["2011-01-01", "2012-03-04", "2014-05-01"]
        result = Index(values, dtype=dtype)

        expected = PeriodIndex(values, dtype=dtype)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "values_constructor", [list, np.array, PeriodIndex, PeriodArray._from_sequence]
    )
    def test_index_object_dtype(self, values_constructor):
        # Index(periods, dtype=object) is an Index (not an PeriodIndex)
        periods = [
            Period("2011-01", freq="M"),
            NaT,
            Period("2011-03", freq="M"),
        ]
        values = values_constructor(periods)
        result = Index(values, dtype=object)

        assert type(result) is Index
        tm.assert_numpy_array_equal(result.values, np.array(values))

    def test_constructor_use_start_freq(self):
        # GH #1118
        p = Period("4/2/2012", freq="B")
        expected = period_range(start="4/2/2012", periods=10, freq="B")

        index = period_range(start=p, periods=10)
        tm.assert_index_equal(index, expected)

    def test_constructor_field_arrays(self):
        # GH #1264

        years = np.arange(1990, 2010).repeat(4)[2:-2]
        quarters = np.tile(np.arange(1, 5), 20)[2:-2]

        index = PeriodIndex(year=years, quarter=quarters, freq="Q-DEC")
        expected = period_range("1990Q3", "2009Q2", freq="Q-DEC")
        tm.assert_index_equal(index, expected)

        index2 = PeriodIndex(year=years, quarter=quarters, freq="2Q-DEC")
        tm.assert_numpy_array_equal(index.asi8, index2.asi8)

        index = PeriodIndex(year=years, quarter=quarters)
        tm.assert_index_equal(index, expected)

        years = [2007, 2007, 2007]
        months = [1, 2]

        msg = "Mismatched Period array lengths"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(year=years, month=months, freq="M")
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(year=years, month=months, freq="2M")

        years = [2007, 2007, 2007]
        months = [1, 2, 3]
        idx = PeriodIndex(year=years, month=months, freq="M")
        exp = period_range("2007-01", periods=3, freq="M")
        tm.assert_index_equal(idx, exp)

    def test_constructor_U(self):
        # U was used as undefined period
        with pytest.raises(ValueError, match="Invalid frequency: X"):
            period_range("2007-1-1", periods=500, freq="X")

    def test_constructor_nano(self):
        idx = period_range(
            start=Period(ordinal=1, freq="N"), end=Period(ordinal=4, freq="N"), freq="N"
        )
        exp = PeriodIndex(
            [
                Period(ordinal=1, freq="N"),
                Period(ordinal=2, freq="N"),
                Period(ordinal=3, freq="N"),
                Period(ordinal=4, freq="N"),
            ],
            freq="N",
        )
        tm.assert_index_equal(idx, exp)

    def test_constructor_arrays_negative_year(self):
        years = np.arange(1960, 2000, dtype=np.int64).repeat(4)
        quarters = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 40)

        pindex = PeriodIndex(year=years, quarter=quarters)

        tm.assert_index_equal(pindex.year, Index(years))
        tm.assert_index_equal(pindex.quarter, Index(quarters))

    def test_constructor_invalid_quarters(self):
        msg = "Quarter must be 1 <= q <= 4"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(year=range(2000, 2004), quarter=list(range(4)), freq="Q-DEC")

    def test_constructor_corner(self):
        result = period_range("2007-01", periods=10.5, freq="M")
        exp = period_range("2007-01", periods=10, freq="M")
        tm.assert_index_equal(result, exp)

    def test_constructor_fromarraylike(self):
        idx = period_range("2007-01", periods=20, freq="M")

        # values is an array of Period, thus can retrieve freq
        tm.assert_index_equal(PeriodIndex(idx.values), idx)
        tm.assert_index_equal(PeriodIndex(list(idx.values)), idx)

        msg = "freq not specified and cannot be inferred"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(idx.asi8)
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(list(idx.asi8))

        msg = "'Period' object is not iterable"
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(data=Period("2007", freq="A"))

        result = PeriodIndex(iter(idx))
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx)
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(result, idx)

        result = PeriodIndex(idx, freq=offsets.MonthEnd())
        tm.assert_index_equal(result, idx)
        assert result.freq == "M"

        result = PeriodIndex(idx, freq="2M")
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2M"

        result = PeriodIndex(idx, freq=offsets.MonthEnd(2))
        tm.assert_index_equal(result, idx.asfreq("2M"))
        assert result.freq == "2M"

        result = PeriodIndex(idx, freq="D")
        exp = idx.asfreq("D", "e")
        tm.assert_index_equal(result, exp)

    def test_constructor_datetime64arr(self):
        vals = np.arange(100000, 100000 + 10000, 100, dtype=np.int64)
        vals = vals.view(np.dtype("M8[us]"))

        msg = r"Wrong dtype: datetime64\[us\]"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(vals, freq="D")

    @pytest.mark.parametrize("box", [None, "series", "index"])
    def test_constructor_datetime64arr_ok(self, box):
        # https://github.com/pandas-dev/pandas/issues/23438
        data = date_range("2017", periods=4, freq="M")
        if box is None:
            data = data._values
        elif box == "series":
            data = Series(data)

        result = PeriodIndex(data, freq="D")
        expected = PeriodIndex(
            ["2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30"], freq="D"
        )
        tm.assert_index_equal(result, expected)

    def test_constructor_dtype(self):
        # passing a dtype with a tz should localize
        idx = PeriodIndex(["2013-01", "2013-03"], dtype="period[M]")
        exp = PeriodIndex(["2013-01", "2013-03"], freq="M")
        tm.assert_index_equal(idx, exp)
        assert idx.dtype == "period[M]"

        idx = PeriodIndex(["2013-01-05", "2013-03-05"], dtype="period[3D]")
        exp = PeriodIndex(["2013-01-05", "2013-03-05"], freq="3D")
        tm.assert_index_equal(idx, exp)
        assert idx.dtype == "period[3D]"

        # if we already have a freq and its not the same, then asfreq
        # (not changed)
        idx = PeriodIndex(["2013-01-01", "2013-01-02"], freq="D")

        res = PeriodIndex(idx, dtype="period[M]")
        exp = PeriodIndex(["2013-01", "2013-01"], freq="M")
        tm.assert_index_equal(res, exp)
        assert res.dtype == "period[M]"

        res = PeriodIndex(idx, freq="M")
        tm.assert_index_equal(res, exp)
        assert res.dtype == "period[M]"

        msg = "specified freq and dtype are different"
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(["2011-01"], freq="M", dtype="period[D]")

    def test_constructor_empty(self):
        idx = PeriodIndex([], freq="M")
        assert isinstance(idx, PeriodIndex)
        assert len(idx) == 0
        assert idx.freq == "M"

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([])

    def test_constructor_pi_nat(self):
        idx = PeriodIndex(
            [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")]
        )
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            np.array([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="M")])
        )
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
        )
        exp = PeriodIndex(["NaT", "NaT", "2011-01", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(
            np.array(
                [NaT, NaT, Period("2011-01", freq="M"), Period("2011-01", freq="M")]
            )
        )
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex([NaT, NaT, "2011-01", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex([NaT, NaT])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array([NaT, NaT]))

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(["NaT", "NaT"])

        with pytest.raises(ValueError, match="freq not specified"):
            PeriodIndex(np.array(["NaT", "NaT"]))

    def test_constructor_incompat_freq(self):
        msg = "Input has different freq=D from PeriodIndex\\(freq=M\\)"

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")])

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [Period("2011-01", freq="M"), NaT, Period("2011-01", freq="D")]
                )
            )

        # first element is NaT
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex([NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")])

        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(
                np.array(
                    [NaT, Period("2011-01", freq="M"), Period("2011-01", freq="D")]
                )
            )

    def test_constructor_mixed(self):
        idx = PeriodIndex(["2011-01", NaT, Period("2011-01", freq="M")])
        exp = PeriodIndex(["2011-01", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex(["NaT", NaT, Period("2011-01", freq="M")])
        exp = PeriodIndex(["NaT", "NaT", "2011-01"], freq="M")
        tm.assert_index_equal(idx, exp)

        idx = PeriodIndex([Period("2011-01-01", freq="D"), NaT, "2012-01-01"])
        exp = PeriodIndex(["2011-01-01", "NaT", "2012-01-01"], freq="D")
        tm.assert_index_equal(idx, exp)

    def test_constructor_simple_new(self):
        idx = period_range("2007-01", name="p", periods=2, freq="M")

        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

        msg = "Should be numpy array of type i8"
        with pytest.raises(AssertionError, match=msg):
            # Need ndarray, not Int64Index
            type(idx._data)._simple_new(Index(idx.asi8), freq=idx.freq)

        arr = type(idx._data)._simple_new(idx.asi8, freq=idx.freq)
        result = idx._simple_new(arr, name="p")
        tm.assert_index_equal(result, idx)

    def test_constructor_simple_new_empty(self):
        # GH13079
        idx = PeriodIndex([], freq="M", name="p")
        with pytest.raises(AssertionError, match="<class .*PeriodIndex'>"):
            idx._simple_new(idx, name="p")

        result = idx._simple_new(idx._data, name="p")
        tm.assert_index_equal(result, idx)

    @pytest.mark.parametrize("floats", [[1.1, 2.1], np.array([1.1, 2.1])])
    def test_constructor_floats(self, floats):
        with pytest.raises(AssertionError, match="<class "):
            PeriodIndex._simple_new(floats)

        msg = "PeriodIndex does not allow floating point in construction"
        with pytest.raises(TypeError, match=msg):
            PeriodIndex(floats)

    def test_constructor_nat(self):
        msg = "start and end must not be NaT"
        with pytest.raises(ValueError, match=msg):
            period_range(start="NaT", end="2011-01-01", freq="M")
        with pytest.raises(ValueError, match=msg):
            period_range(start="2011-01-01", end="NaT", freq="M")

    def test_constructor_year_and_quarter(self):
        year = Series([2001, 2002, 2003])
        quarter = year - 2000
        idx = PeriodIndex(year=year, quarter=quarter)
        strs = [f"{t[0]:d}Q{t[1]:d}" for t in zip(quarter, year)]
        lops = list(map(Period, strs))
        p = PeriodIndex(lops)
        tm.assert_index_equal(p, idx)

    def test_constructor_freq_mult(self):
        # GH #7811
        pidx = period_range(start="2014-01", freq="2M", periods=4)
        expected = PeriodIndex(["2014-01", "2014-03", "2014-05", "2014-07"], freq="2M")
        tm.assert_index_equal(pidx, expected)

        pidx = period_range(start="2014-01-02", end="2014-01-15", freq="3D")
        expected = PeriodIndex(
            ["2014-01-02", "2014-01-05", "2014-01-08", "2014-01-11", "2014-01-14"],
            freq="3D",
        )
        tm.assert_index_equal(pidx, expected)

        pidx = period_range(end="2014-01-01 17:00", freq="4H", periods=3)
        expected = PeriodIndex(
            ["2014-01-01 09:00", "2014-01-01 13:00", "2014-01-01 17:00"], freq="4H"
        )
        tm.assert_index_equal(pidx, expected)

        msg = "Frequency must be positive, because it represents span: -1M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="-1M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(["2011-01"], freq="0M")

        msg = "Frequency must be positive, because it represents span: 0M"
        with pytest.raises(ValueError, match=msg):
            period_range("2011-01", periods=3, freq="0M")

    @pytest.mark.parametrize("freq", ["A", "M", "D", "T", "S"])
    @pytest.mark.parametrize("mult", [1, 2, 3, 4, 5])
    def test_constructor_freq_mult_dti_compat(self, mult, freq):
        freqstr = str(mult) + freq
        pidx = period_range(start="2014-04-01", freq=freqstr, periods=10)
        expected = date_range(start="2014-04-01", freq=freqstr, periods=10).to_period(
            freqstr
        )
        tm.assert_index_equal(pidx, expected)

    def test_constructor_freq_combined(self):
        for freq in ["1D1H", "1H1D"]:
            pidx = PeriodIndex(["2016-01-01", "2016-01-02"], freq=freq)
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 00:00"], freq="25H")
        for freq in ["1D1H", "1H1D"]:
            pidx = period_range(start="2016-01-01", periods=2, freq=freq)
            expected = PeriodIndex(["2016-01-01 00:00", "2016-01-02 01:00"], freq="25H")
            tm.assert_index_equal(pidx, expected)

    def test_constructor(self):
        pi = period_range(freq="A", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 9

        pi = period_range(freq="Q", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 4 * 9

        pi = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 12 * 9

        pi = period_range(freq="D", start="1/1/2001", end="12/31/2009")
        assert len(pi) == 365 * 9 + 2

        pi = period_range(freq="B", start="1/1/2001", end="12/31/2009")
        assert len(pi) == 261 * 9

        pi = period_range(freq="H", start="1/1/2001", end="12/31/2001 23:00")
        assert len(pi) == 365 * 24

        pi = period_range(freq="Min", start="1/1/2001", end="1/1/2001 23:59")
        assert len(pi) == 24 * 60

        pi = period_range(freq="S", start="1/1/2001", end="1/1/2001 23:59:59")
        assert len(pi) == 24 * 60 * 60

        start = Period("02-Apr-2005", "B")
        i1 = period_range(start=start, periods=20)
        assert len(i1) == 20
        assert i1.freq == start.freq
        assert i1[0] == start

        end_intv = Period("2006-12-31", "W")
        i1 = period_range(end=end_intv, periods=10)
        assert len(i1) == 10
        assert i1.freq == end_intv.freq
        assert i1[-1] == end_intv

        end_intv = Period("2006-12-31", "1w")
        i2 = period_range(end=end_intv, periods=10)
        assert len(i1) == len(i2)
        assert (i1 == i2).all()
        assert i1.freq == i2.freq

        end_intv = Period("2005-05-01", "B")
        i1 = period_range(start=start, end=end_intv)

        # infer freq from first element
        i2 = PeriodIndex([end_intv, Period("2005-05-05", "B")])
        assert len(i2) == 2
        assert i2[0] == end_intv

        i2 = PeriodIndex(np.array([end_intv, Period("2005-05-05", "B")]))
        assert len(i2) == 2
        assert i2[0] == end_intv

        # Mixed freq should fail
        vals = [end_intv, Period("2006-12-31", "w")]
        msg = r"Input has different freq=W-SUN from PeriodIndex\(freq=B\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(vals)
        vals = np.array(vals)
        with pytest.raises(IncompatibleFrequency, match=msg):
            PeriodIndex(vals)

        # tuple freq disallowed GH#34703
        with pytest.raises(TypeError, match="pass as a string instead"):
            Period("2006-12-31", ("w", 1))

    @pytest.mark.parametrize(
        "freq", ["M", "Q", "A", "D", "B", "T", "S", "L", "U", "N", "H"]
    )
    def test_recreate_from_data(self, freq):
        org = period_range(start="2001/04/01", freq=freq, periods=1)
        idx = PeriodIndex(org.values, freq=freq)
        tm.assert_index_equal(idx, org)

    def test_map_with_string_constructor(self):
        raw = [2005, 2007, 2009]
        index = PeriodIndex(raw, freq="A")

        expected = Index([str(num) for num in raw])
        res = index.map(str)

        # should return an Index
        assert isinstance(res, Index)

        # preserve element types
        assert all(isinstance(resi, str) for resi in res)

        # lastly, values should compare equal
        tm.assert_index_equal(res, expected)


class TestShallowCopy:
    def test_shallow_copy_empty(self):
        # GH#13067
        idx = PeriodIndex([], freq="M")
        result = idx._view()
        expected = idx

        tm.assert_index_equal(result, expected)

    def test_shallow_copy_disallow_i8(self):
        # GH#24391
        pi = period_range("2018-01-01", periods=3, freq="2D")
        with pytest.raises(AssertionError, match="ndarray"):
            pi._shallow_copy(pi.asi8)

    def test_shallow_copy_requires_disallow_period_index(self):
        pi = period_range("2018-01-01", periods=3, freq="2D")
        with pytest.raises(AssertionError, match="PeriodIndex"):
            pi._shallow_copy(pi)


class TestSeriesPeriod:
    def setup_method(self, method):
        self.series = Series(period_range("2000-01-01", periods=10, freq="D"))

    def test_constructor_cant_cast_period(self):
        msg = "Cannot cast PeriodIndex to dtype float64"
        with pytest.raises(TypeError, match=msg):
            Series(period_range("2000-01-01", periods=10, freq="D"), dtype=float)

    def test_constructor_cast_object(self):
        s = Series(period_range("1/1/2000", periods=10), dtype=PeriodDtype("D"))
        exp = Series(period_range("1/1/2000", periods=10))
        tm.assert_series_equal(s, exp)
