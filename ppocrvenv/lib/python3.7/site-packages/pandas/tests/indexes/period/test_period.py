import numpy as np
import pytest

from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas import (
    DatetimeIndex,
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
from pandas.tests.indexes.datetimelike import DatetimeLike


class TestPeriodIndex(DatetimeLike):
    _index_cls = PeriodIndex

    @pytest.fixture
    def simple_index(self) -> Index:
        return period_range("20130101", periods=5, freq="D")

    @pytest.fixture(
        params=[
            tm.makePeriodIndex(10),
            period_range("20130101", periods=10, freq="D")[::-1],
        ],
        ids=["index_inc", "index_dec"],
    )
    def index(self, request):
        return request.param

    @pytest.mark.xfail(reason="Goes through a generate_range path")
    def test_pickle_compat_construction(self):
        super().test_pickle_compat_construction()

    @pytest.mark.parametrize("freq", ["D", "M", "A"])
    def test_pickle_round_trip(self, freq):
        idx = PeriodIndex(["2016-05-16", "NaT", NaT, np.NaN], freq=freq)
        result = tm.round_trip_pickle(idx)
        tm.assert_index_equal(result, idx)

    def test_where(self):
        # This is handled in test_indexing
        pass

    def test_no_millisecond_field(self):
        msg = "type object 'DatetimeIndex' has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex.millisecond

        msg = "'DatetimeIndex' object has no attribute 'millisecond'"
        with pytest.raises(AttributeError, match=msg):
            DatetimeIndex([]).millisecond

    def test_make_time_series(self):
        index = period_range(freq="A", start="1/1/2001", end="12/1/2009")
        series = Series(1, index=index)
        assert isinstance(series, Series)

    def test_view_asi8(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.view("i8"), exp)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([492, -9223372036854775808], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.view("i8"), exp)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        exp = np.array([14975, -9223372036854775808], dtype=np.int64)
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")
        tm.assert_numpy_array_equal(idx.view("i8"), exp)
        tm.assert_numpy_array_equal(idx.asi8, exp)

    def test_values(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.values, exp)
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)

        exp = np.array([], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.values, exp)
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)
        exp = np.array([492, -9223372036854775808], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.asi8, exp)

        idx = PeriodIndex(["2011-01-01", NaT], freq="D")

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.values, exp)
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)
        exp = np.array([14975, -9223372036854775808], dtype=np.int64)
        tm.assert_numpy_array_equal(idx.asi8, exp)

    def test_period_index_length(self):
        pi = period_range(freq="A", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 9

        pi = period_range(freq="Q", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 4 * 9

        pi = period_range(freq="M", start="1/1/2001", end="12/1/2009")
        assert len(pi) == 12 * 9

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

        msg = "start and end must have same freq"
        with pytest.raises(ValueError, match=msg):
            period_range(start=start, end=end_intv)

        end_intv = Period("2005-05-01", "B")
        i1 = period_range(start=start, end=end_intv)

        msg = (
            "Of the three parameters: start, end, and periods, exactly two "
            "must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            period_range(start=start)

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
        with pytest.raises(ValueError, match=msg):
            PeriodIndex(vals)

    def test_fields(self):
        # year, month, day, hour, minute
        # second, weekofyear, week, dayofweek, weekday, dayofyear, quarter
        # qyear
        pi = period_range(freq="A", start="1/1/2001", end="12/1/2005")
        self._check_all_fields(pi)

        pi = period_range(freq="Q", start="1/1/2001", end="12/1/2002")
        self._check_all_fields(pi)

        pi = period_range(freq="M", start="1/1/2001", end="1/1/2002")
        self._check_all_fields(pi)

        pi = period_range(freq="D", start="12/1/2001", end="6/1/2001")
        self._check_all_fields(pi)

        pi = period_range(freq="B", start="12/1/2001", end="6/1/2001")
        self._check_all_fields(pi)

        pi = period_range(freq="H", start="12/31/2001", end="1/1/2002 23:00")
        self._check_all_fields(pi)

        pi = period_range(freq="Min", start="12/31/2001", end="1/1/2002 00:20")
        self._check_all_fields(pi)

        pi = period_range(
            freq="S", start="12/31/2001 00:00:00", end="12/31/2001 00:05:00"
        )
        self._check_all_fields(pi)

        end_intv = Period("2006-12-31", "W")
        i1 = period_range(end=end_intv, periods=10)
        self._check_all_fields(i1)

    def _check_all_fields(self, periodindex):
        fields = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekofyear",
            "week",
            "dayofweek",
            "day_of_week",
            "dayofyear",
            "day_of_year",
            "quarter",
            "qyear",
            "days_in_month",
        ]

        periods = list(periodindex)
        s = Series(periodindex)

        for field in fields:
            field_idx = getattr(periodindex, field)
            assert len(periodindex) == len(field_idx)
            for x, val in zip(periods, field_idx):
                assert getattr(x, field) == val

            if len(s) == 0:
                continue

            field_s = getattr(s.dt, field)
            assert len(periodindex) == len(field_s)
            for x, val in zip(periods, field_s):
                assert getattr(x, field) == val

    def test_is_(self):
        create_index = lambda: period_range(freq="A", start="1/1/2001", end="12/1/2009")
        index = create_index()
        assert index.is_(index)
        assert not index.is_(create_index())
        assert index.is_(index.view())
        assert index.is_(index.view().view().view().view().view())
        assert index.view().is_(index)
        ind2 = index.view()
        index.name = "Apple"
        assert ind2.is_(index)
        assert not index.is_(index[:])
        assert not index.is_(index.asfreq("M"))
        assert not index.is_(index.asfreq("A"))

        assert not index.is_(index - 2)
        assert not index.is_(index - 0)

    def test_index_duplicate_periods(self):
        # monotonic
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq="A-JUN")
        ts = Series(np.random.randn(len(idx)), index=idx)

        result = ts["2007"]
        expected = ts[1:3]
        tm.assert_series_equal(result, expected)
        result[:] = 1
        assert (ts[1:3] == 1).all()

        # not monotonic
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2007], freq="A-JUN")
        ts = Series(np.random.randn(len(idx)), index=idx)

        result = ts["2007"]
        expected = ts[idx == "2007"]
        tm.assert_series_equal(result, expected)

    def test_index_unique(self):
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq="A-JUN")
        expected = PeriodIndex([2000, 2007, 2009], freq="A-JUN")
        tm.assert_index_equal(idx.unique(), expected)
        assert idx.nunique() == 3

    def test_shift(self):
        # This is tested in test_arithmetic
        pass

    def test_negative_ordinals(self):
        Period(ordinal=-1000, freq="A")
        Period(ordinal=0, freq="A")

        idx1 = PeriodIndex(ordinal=[-1, 0, 1], freq="A")
        idx2 = PeriodIndex(ordinal=np.array([-1, 0, 1]), freq="A")
        tm.assert_index_equal(idx1, idx2)

    def test_pindex_fieldaccessor_nat(self):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2012-03", "2012-04"], freq="D", name="name"
        )

        exp = Index([2011, 2011, -1, 2012, 2012], dtype=np.int64, name="name")
        tm.assert_index_equal(idx.year, exp)
        exp = Index([1, 2, -1, 3, 4], dtype=np.int64, name="name")
        tm.assert_index_equal(idx.month, exp)

    def test_pindex_qaccess(self):
        pi = PeriodIndex(["2Q05", "3Q05", "4Q05", "1Q06", "2Q06"], freq="Q")
        s = Series(np.random.rand(len(pi)), index=pi).cumsum()
        # Todo: fix these accessors!
        assert s["05Q4"] == s[2]

    def test_pindex_multiples(self):
        expected = PeriodIndex(
            ["2011-01", "2011-03", "2011-05", "2011-07", "2011-09", "2011-11"],
            freq="2M",
        )

        pi = period_range(start="1/1/11", end="12/31/11", freq="2M")
        tm.assert_index_equal(pi, expected)
        assert pi.freq == offsets.MonthEnd(2)
        assert pi.freqstr == "2M"

        pi = period_range(start="1/1/11", periods=6, freq="2M")
        tm.assert_index_equal(pi, expected)
        assert pi.freq == offsets.MonthEnd(2)
        assert pi.freqstr == "2M"

    def test_iteration(self):
        index = period_range(start="1/1/10", periods=4, freq="B")

        result = list(index)
        assert isinstance(result[0], Period)
        assert result[0].freq == index.freq

    def test_with_multi_index(self):
        # #1705
        index = date_range("1/1/2012", periods=4, freq="12H")
        index_as_arrays = [index.to_period(freq="D"), index.hour]

        s = Series([0, 1, 2, 3], index_as_arrays)

        assert isinstance(s.index.levels[0], PeriodIndex)

        assert isinstance(s.index.values[0][0], Period)

    def test_pickle_freq(self):
        # GH2891
        prng = period_range("1/1/2011", "1/1/2012", freq="M")
        new_prng = tm.round_trip_pickle(prng)
        assert new_prng.freq == offsets.MonthEnd()
        assert new_prng.freqstr == "M"

    def test_map(self):
        # test_map_dictlike generally tests

        index = PeriodIndex([2005, 2007, 2009], freq="A")
        result = index.map(lambda x: x.ordinal)
        exp = Index([x.ordinal for x in index])
        tm.assert_index_equal(result, exp)

    def test_format_empty(self):
        # GH35712
        empty_idx = self._index_cls([], freq="A")
        assert empty_idx.format() == []
        assert empty_idx.format(name=True) == [""]


def test_maybe_convert_timedelta():
    pi = PeriodIndex(["2000", "2001"], freq="D")
    offset = offsets.Day(2)
    assert pi._maybe_convert_timedelta(offset) == 2
    assert pi._maybe_convert_timedelta(2) == 2

    offset = offsets.BusinessDay()
    msg = r"Input has different freq=B from PeriodIndex\(freq=D\)"
    with pytest.raises(ValueError, match=msg):
        pi._maybe_convert_timedelta(offset)


def test_is_monotonic_with_nat():
    # GH#31437
    # PeriodIndex.is_monotonic should behave analogously to DatetimeIndex,
    #  in particular never be monotonic when we have NaT
    dti = date_range("2016-01-01", periods=3)
    pi = dti.to_period("D")
    tdi = Index(dti.view("timedelta64[ns]"))

    for obj in [pi, pi._engine, dti, dti._engine, tdi, tdi._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert obj.is_monotonic
        assert obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

    dti1 = dti.insert(0, NaT)
    pi1 = dti1.to_period("D")
    tdi1 = Index(dti1.view("timedelta64[ns]"))

    for obj in [pi1, pi1._engine, dti1, dti1._engine, tdi1, tdi1._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert not obj.is_monotonic
        assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique

    dti2 = dti.insert(3, NaT)
    pi2 = dti2.to_period("H")
    tdi2 = Index(dti2.view("timedelta64[ns]"))

    for obj in [pi2, pi2._engine, dti2, dti2._engine, tdi2, tdi2._engine]:
        if isinstance(obj, Index):
            # i.e. not Engines
            assert not obj.is_monotonic
        assert not obj.is_monotonic_increasing
        assert not obj.is_monotonic_decreasing
        assert obj.is_unique


@pytest.mark.parametrize("array", [True, False])
def test_dunder_array(array):
    obj = PeriodIndex(["2000-01-01", "2001-01-01"], freq="D")
    if array:
        obj = obj._data

    expected = np.array([obj[0], obj[1]], dtype=object)
    result = np.array(obj)
    tm.assert_numpy_array_equal(result, expected)

    result = np.asarray(obj)
    tm.assert_numpy_array_equal(result, expected)

    expected = obj.asi8
    for dtype in ["i8", "int64", np.int64]:
        result = np.array(obj, dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)

        result = np.asarray(obj, dtype=dtype)
        tm.assert_numpy_array_equal(result, expected)

    for dtype in ["float64", "int32", "uint64"]:
        msg = "argument must be"
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=dtype)
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=getattr(np, dtype))
