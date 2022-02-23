from datetime import datetime

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Int64Index,
    Series,
    bdate_range,
    date_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    BMonthEnd,
    Minute,
    MonthEnd,
)

START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)


class TestDatetimeIndexSetOps:
    tz = [
        None,
        "UTC",
        "Asia/Tokyo",
        "US/Eastern",
        "dateutil/Asia/Singapore",
        "dateutil/US/Pacific",
    ]

    # TODO: moved from test_datetimelike; dedup with version below
    def test_union2(self, sort):
        everything = tm.makeDateIndex(10)
        first = everything[:5]
        second = everything[5:]
        union = first.union(second, sort=sort)
        tm.assert_index_equal(union, everything)

    @pytest.mark.parametrize("box", [np.array, Series, list])
    def test_union3(self, sort, box):
        everything = tm.makeDateIndex(10)
        first = everything[:5]
        second = everything[5:]

        # GH 10149 support listlike inputs other than Index objects
        expected = first.union(second, sort=sort)
        case = box(second.values)
        result = first.union(case, sort=sort)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("tz", tz)
    def test_union(self, tz, sort):
        rng1 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)
        expected1 = date_range("1/1/2000", freq="D", periods=10, tz=tz)
        expected1_notsorted = DatetimeIndex(list(other1) + list(rng1))

        rng2 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)
        expected2 = date_range("1/1/2000", freq="D", periods=8, tz=tz)
        expected2_notsorted = DatetimeIndex(list(other2) + list(rng2[:3]))

        rng3 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        other3 = DatetimeIndex([], tz=tz)
        expected3 = date_range("1/1/2000", freq="D", periods=5, tz=tz)
        expected3_notsorted = rng3

        for rng, other, exp, exp_notsorted in [
            (rng1, other1, expected1, expected1_notsorted),
            (rng2, other2, expected2, expected2_notsorted),
            (rng3, other3, expected3, expected3_notsorted),
        ]:

            result_union = rng.union(other, sort=sort)
            tm.assert_index_equal(result_union, exp)

            result_union = other.union(rng, sort=sort)
            if sort is None:
                tm.assert_index_equal(result_union, exp)
            else:
                tm.assert_index_equal(result_union, exp_notsorted)

    def test_union_coverage(self, sort):
        idx = DatetimeIndex(["2000-01-03", "2000-01-01", "2000-01-02"])
        ordered = DatetimeIndex(idx.sort_values(), freq="infer")
        result = ordered.union(idx, sort=sort)
        tm.assert_index_equal(result, ordered)

        result = ordered[:0].union(ordered, sort=sort)
        tm.assert_index_equal(result, ordered)
        assert result.freq == ordered.freq

    def test_union_bug_1730(self, sort):
        rng_a = date_range("1/1/2012", periods=4, freq="3H")
        rng_b = date_range("1/1/2012", periods=4, freq="4H")

        result = rng_a.union(rng_b, sort=sort)
        exp = list(rng_a) + list(rng_b[1:])
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_bug_1745(self, sort):
        left = DatetimeIndex(["2012-05-11 15:19:49.695000"])
        right = DatetimeIndex(
            [
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )

        result = left.union(right, sort=sort)
        exp = DatetimeIndex(
            [
                "2012-05-11 15:19:49.695000",
                "2012-05-29 13:04:21.322000",
                "2012-05-11 15:27:24.873000",
                "2012-05-11 15:31:05.350000",
            ]
        )
        if sort is None:
            exp = exp.sort_values()
        tm.assert_index_equal(result, exp)

    def test_union_bug_4564(self, sort):
        from pandas import DateOffset

        left = date_range("2013-01-01", "2013-02-01")
        right = left + DateOffset(minutes=15)

        result = left.union(right, sort=sort)
        exp = list(left) + list(right)
        if sort is None:
            exp = DatetimeIndex(sorted(exp))
        else:
            exp = DatetimeIndex(exp)
        tm.assert_index_equal(result, exp)

    def test_union_freq_both_none(self, sort):
        # GH11086
        expected = bdate_range("20150101", periods=10)
        expected._data.freq = None

        result = expected.union(expected, sort=sort)
        tm.assert_index_equal(result, expected)
        assert result.freq is None

    def test_union_freq_infer(self):
        # When taking the union of two DatetimeIndexes, we infer
        #  a freq even if the arguments don't have freq.  This matches
        #  TimedeltaIndex behavior.
        dti = date_range("2016-01-01", periods=5)
        left = dti[[0, 1, 3, 4]]
        right = dti[[2, 3, 1]]

        assert left.freq is None
        assert right.freq is None

        result = left.union(right)
        tm.assert_index_equal(result, dti)
        assert result.freq == "D"

    def test_union_dataframe_index(self):
        rng1 = date_range("1/1/1999", "1/1/2012", freq="MS")
        s1 = Series(np.random.randn(len(rng1)), rng1)

        rng2 = date_range("1/1/1980", "12/1/2001", freq="MS")
        s2 = Series(np.random.randn(len(rng2)), rng2)
        df = DataFrame({"s1": s1, "s2": s2})

        exp = date_range("1/1/1980", "1/1/2012", freq="MS")
        tm.assert_index_equal(df.index, exp)

    def test_union_with_DatetimeIndex(self, sort):
        i1 = Int64Index(np.arange(0, 20, 2))
        i2 = date_range(start="2012-01-03 00:00:00", periods=10, freq="D")
        # Works
        i1.union(i2, sort=sort)
        # Fails with "AttributeError: can't set attribute"
        i2.union(i1, sort=sort)

    # TODO: moved from test_datetimelike; de-duplicate with version below
    def test_intersection2(self):
        first = tm.makeDateIndex(10)
        second = first[5:]
        intersect = first.intersection(second)
        assert tm.equalContents(intersect, second)

        # GH 10149
        cases = [klass(second.values) for klass in [np.array, Series, list]]
        for case in cases:
            result = first.intersection(case)
            assert tm.equalContents(result, second)

        third = Index(["a", "b", "c"])
        result = first.intersection(third)
        expected = Index([], dtype=object)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "tz", [None, "Asia/Tokyo", "US/Eastern", "dateutil/US/Pacific"]
    )
    def test_intersection(self, tz, sort):
        # GH 4690 (with tz)
        base = date_range("6/1/2000", "6/30/2000", freq="D", name="idx")

        # if target has the same name, it is preserved
        rng2 = date_range("5/15/2000", "6/20/2000", freq="D", name="idx")
        expected2 = date_range("6/1/2000", "6/20/2000", freq="D", name="idx")

        # if target name is different, it will be reset
        rng3 = date_range("5/15/2000", "6/20/2000", freq="D", name="other")
        expected3 = date_range("6/1/2000", "6/20/2000", freq="D", name=None)

        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", name="idx")
        expected4 = DatetimeIndex([], freq="D", name="idx")

        for (rng, expected) in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng)
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

        # non-monotonic
        base = DatetimeIndex(
            ["2011-01-05", "2011-01-04", "2011-01-02", "2011-01-03"], tz=tz, name="idx"
        )

        rng2 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"], tz=tz, name="idx"
        )
        expected2 = DatetimeIndex(["2011-01-04", "2011-01-02"], tz=tz, name="idx")

        rng3 = DatetimeIndex(
            ["2011-01-04", "2011-01-02", "2011-02-02", "2011-02-03"],
            tz=tz,
            name="other",
        )
        expected3 = DatetimeIndex(["2011-01-04", "2011-01-02"], tz=tz, name=None)

        # GH 7880
        rng4 = date_range("7/1/2000", "7/31/2000", freq="D", tz=tz, name="idx")
        expected4 = DatetimeIndex([], tz=tz, name="idx")
        assert expected4.freq is None

        for (rng, expected) in [
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            result = base.intersection(rng, sort=sort)
            if sort is None:
                expected = expected.sort_values()
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

    # parametrize over both anchored and non-anchored freqs, as they
    #  have different code paths
    @pytest.mark.parametrize("freq", ["T", "B"])
    def test_intersection_empty(self, tz_aware_fixture, freq):
        # empty same freq GH2129
        tz = tz_aware_fixture
        rng = date_range("6/1/2000", "6/15/2000", freq=freq, tz=tz)
        result = rng[0:0].intersection(rng)
        assert len(result) == 0
        assert result.freq == rng.freq

        result = rng.intersection(rng[0:0])
        assert len(result) == 0
        assert result.freq == rng.freq

        # no overlap GH#33604
        check_freq = freq != "T"  # We don't preserve freq on non-anchored offsets
        result = rng[:3].intersection(rng[-3:])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # We don't preserve freq on non-anchored offsets
            assert result.freq == rng.freq

        # swapped left and right
        result = rng[-3:].intersection(rng[:3])
        tm.assert_index_equal(result, rng[:0])
        if check_freq:
            # We don't preserve freq on non-anchored offsets
            assert result.freq == rng.freq

    def test_intersection_bug_1708(self):
        from pandas import DateOffset

        index_1 = date_range("1/1/2012", periods=4, freq="12H")
        index_2 = index_1 + DateOffset(hours=1)

        with tm.assert_produces_warning(FutureWarning):
            result = index_1 & index_2
        assert len(result) == 0

    @pytest.mark.parametrize("tz", tz)
    def test_difference(self, tz, sort):
        rng_dates = ["1/2/2000", "1/3/2000", "1/1/2000", "1/4/2000", "1/5/2000"]

        rng1 = DatetimeIndex(rng_dates, tz=tz)
        other1 = date_range("1/6/2000", freq="D", periods=5, tz=tz)
        expected1 = DatetimeIndex(rng_dates, tz=tz)

        rng2 = DatetimeIndex(rng_dates, tz=tz)
        other2 = date_range("1/4/2000", freq="D", periods=5, tz=tz)
        expected2 = DatetimeIndex(rng_dates[:3], tz=tz)

        rng3 = DatetimeIndex(rng_dates, tz=tz)
        other3 = DatetimeIndex([], tz=tz)
        expected3 = DatetimeIndex(rng_dates, tz=tz)

        for rng, other, expected in [
            (rng1, other1, expected1),
            (rng2, other2, expected2),
            (rng3, other3, expected3),
        ]:
            result_diff = rng.difference(other, sort)
            if sort is None and len(other):
                # We dont sort (yet?) when empty GH#24959
                expected = expected.sort_values()
            tm.assert_index_equal(result_diff, expected)

    def test_difference_freq(self, sort):
        # GH14323: difference of DatetimeIndex should not preserve frequency

        index = date_range("20160920", "20160925", freq="D")
        other = date_range("20160921", "20160924", freq="D")
        expected = DatetimeIndex(["20160920", "20160925"], freq=None)
        idx_diff = index.difference(other, sort)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

        other = date_range("20160922", "20160925", freq="D")
        idx_diff = index.difference(other, sort)
        expected = DatetimeIndex(["20160920", "20160921"], freq=None)
        tm.assert_index_equal(idx_diff, expected)
        tm.assert_attr_equal("freq", idx_diff, expected)

    def test_datetimeindex_diff(self, sort):
        dti1 = date_range(freq="Q-JAN", start=datetime(1997, 12, 31), periods=100)
        dti2 = date_range(freq="Q-JAN", start=datetime(1997, 12, 31), periods=98)
        assert len(dti1.difference(dti2, sort)) == 2

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Eastern"])
    def test_setops_preserve_freq(self, tz):
        rng = date_range("1/1/2000", "1/1/2002", name="idx", tz=tz)

        result = rng[:50].union(rng[50:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[30:100])
        assert result.name == rng.name
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].union(rng[60:100])
        assert result.name == rng.name
        assert result.freq is None
        assert result.tz == rng.tz

        result = rng[:50].intersection(rng[25:75])
        assert result.name == rng.name
        assert result.freqstr == "D"
        assert result.tz == rng.tz

        nofreq = DatetimeIndex(list(rng[25:75]), name="other")
        result = rng[:50].union(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

        result = rng[:50].intersection(nofreq)
        assert result.name is None
        assert result.freq == rng.freq
        assert result.tz == rng.tz

    def test_intersection_non_tick_no_fastpath(self):
        # GH#42104
        dti = DatetimeIndex(
            [
                "2018-12-31",
                "2019-03-31",
                "2019-06-30",
                "2019-09-30",
                "2019-12-31",
                "2020-03-31",
            ],
            freq="Q-DEC",
        )
        result = dti[::2].intersection(dti[1::2])
        expected = dti[:0]
        tm.assert_index_equal(result, expected)


class TestBusinessDatetimeIndex:
    def setup_method(self, method):
        self.rng = bdate_range(START, END)

    def test_union(self, sort):
        # overlapping
        left = self.rng[:10]
        right = self.rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # non-overlapping, gap in middle
        left = self.rng[:5]
        right = self.rng[10:]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, Index)

        # non-overlapping, no gap
        left = self.rng[:5]
        right = self.rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # order does not matter
        if sort is None:
            tm.assert_index_equal(right.union(left, sort=sort), the_union)
        else:
            expected = DatetimeIndex(list(right) + list(left))
            tm.assert_index_equal(right.union(left, sort=sort), expected)

        # overlapping, but different offset
        rng = date_range(START, END, freq=BMonthEnd())

        the_union = self.rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_union_not_cacheable(self, sort):
        rng = date_range("1/1/2000", periods=50, freq=Minute())
        rng1 = rng[10:]
        rng2 = rng[:25]
        the_union = rng1.union(rng2, sort=sort)
        if sort is None:
            tm.assert_index_equal(the_union, rng)
        else:
            expected = DatetimeIndex(list(rng[10:]) + list(rng[:10]))
            tm.assert_index_equal(the_union, expected)

        rng1 = rng[10:]
        rng2 = rng[15:35]
        the_union = rng1.union(rng2, sort=sort)
        expected = rng[10:]
        tm.assert_index_equal(the_union, expected)

    def test_intersection(self):
        rng = date_range("1/1/2000", periods=50, freq=Minute())
        rng1 = rng[10:]
        rng2 = rng[:25]
        the_int = rng1.intersection(rng2)
        expected = rng[10:25]
        tm.assert_index_equal(the_int, expected)
        assert isinstance(the_int, DatetimeIndex)
        assert the_int.freq == rng.freq

        the_int = rng1.intersection(rng2.view(DatetimeIndex))
        tm.assert_index_equal(the_int, expected)

        # non-overlapping
        the_int = rng[:10].intersection(rng[10:])
        expected = DatetimeIndex([])
        tm.assert_index_equal(the_int, expected)

    def test_intersection_bug(self):
        # GH #771
        a = bdate_range("11/30/2011", "12/31/2011")
        b = bdate_range("12/10/2011", "12/20/2011")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq

    def test_intersection_list(self):
        # GH#35876
        # values is not an Index -> no name -> retain "a"
        values = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
        idx = DatetimeIndex(values, name="a")
        res = idx.intersection(values)
        tm.assert_index_equal(res, idx)

    def test_month_range_union_tz_pytz(self, sort):
        from pytz import timezone

        tz = timezone("US/Eastern")

        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        early_dr.union(late_dr, sort=sort)

    @td.skip_if_windows_python_3
    def test_month_range_union_tz_dateutil(self, sort):
        from pandas._libs.tslibs.timezones import dateutil_gettz

        tz = dateutil_gettz("US/Eastern")

        early_start = datetime(2011, 1, 1)
        early_end = datetime(2011, 3, 1)

        late_start = datetime(2011, 3, 1)
        late_end = datetime(2011, 5, 1)

        early_dr = date_range(start=early_start, end=early_end, tz=tz, freq=MonthEnd())
        late_dr = date_range(start=late_start, end=late_end, tz=tz, freq=MonthEnd())

        early_dr.union(late_dr, sort=sort)

    @pytest.mark.parametrize("sort", [False, None])
    def test_intersection_duplicates(self, sort):
        # GH#38196
        idx1 = Index(
            [
                pd.Timestamp("2019-12-13"),
                pd.Timestamp("2019-12-12"),
                pd.Timestamp("2019-12-12"),
            ]
        )
        result = idx1.intersection(idx1, sort=sort)
        expected = Index([pd.Timestamp("2019-12-13"), pd.Timestamp("2019-12-12")])
        tm.assert_index_equal(result, expected)


class TestCustomDatetimeIndex:
    def setup_method(self, method):
        self.rng = bdate_range(START, END, freq="C")

    def test_union(self, sort):
        # overlapping
        left = self.rng[:10]
        right = self.rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # non-overlapping, gap in middle
        left = self.rng[:5]
        right = self.rng[10:]

        the_union = left.union(right, sort)
        assert isinstance(the_union, Index)

        # non-overlapping, no gap
        left = self.rng[:5]
        right = self.rng[5:10]

        the_union = left.union(right, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

        # order does not matter
        if sort is None:
            tm.assert_index_equal(right.union(left, sort=sort), the_union)

        # overlapping, but different offset
        rng = date_range(START, END, freq=BMonthEnd())

        the_union = self.rng.union(rng, sort=sort)
        assert isinstance(the_union, DatetimeIndex)

    def test_intersection_bug(self):
        # GH #771
        a = bdate_range("11/30/2011", "12/31/2011", freq="C")
        b = bdate_range("12/10/2011", "12/20/2011", freq="C")
        result = a.intersection(b)
        tm.assert_index_equal(result, b)
        assert result.freq == b.freq
