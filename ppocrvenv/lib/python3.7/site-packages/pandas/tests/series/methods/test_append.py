import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


class TestSeriesAppend:
    def test_append_preserve_name(self, datetime_series):
        result = datetime_series[:5].append(datetime_series[5:])
        assert result.name == datetime_series.name

    def test_append(self, datetime_series, string_series, object_series):
        appended_series = string_series.append(object_series)
        for idx, value in appended_series.items():
            if idx in string_series.index:
                assert value == string_series[idx]
            elif idx in object_series.index:
                assert value == object_series[idx]
            else:
                raise AssertionError("orphaned index!")

        msg = "Indexes have overlapping values:"
        with pytest.raises(ValueError, match=msg):
            datetime_series.append(datetime_series, verify_integrity=True)

    def test_append_many(self, datetime_series):
        pieces = [datetime_series[:5], datetime_series[5:10], datetime_series[10:]]

        result = pieces[0].append(pieces[1:])
        tm.assert_series_equal(result, datetime_series)

    def test_append_duplicates(self):
        # GH 13677
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        exp = Series([1, 2, 3, 4, 5, 6], index=[0, 1, 2, 0, 1, 2])
        tm.assert_series_equal(s1.append(s2), exp)
        tm.assert_series_equal(pd.concat([s1, s2]), exp)

        # the result must have RangeIndex
        exp = Series([1, 2, 3, 4, 5, 6])
        tm.assert_series_equal(
            s1.append(s2, ignore_index=True), exp, check_index_type=True
        )
        tm.assert_series_equal(
            pd.concat([s1, s2], ignore_index=True), exp, check_index_type=True
        )

        msg = "Indexes have overlapping values:"
        with pytest.raises(ValueError, match=msg):
            s1.append(s2, verify_integrity=True)
        with pytest.raises(ValueError, match=msg):
            pd.concat([s1, s2], verify_integrity=True)

    def test_append_tuples(self):
        # GH 28410
        s = Series([1, 2, 3])
        list_input = [s, s]
        tuple_input = (s, s)

        expected = s.append(list_input)
        result = s.append(tuple_input)

        tm.assert_series_equal(expected, result)

    def test_append_dataframe_raises(self):
        # GH 31413
        df = DataFrame({"A": [1, 2], "B": [3, 4]})

        msg = "to_append should be a Series or list/tuple of Series, got DataFrame"
        with pytest.raises(TypeError, match=msg):
            df.A.append(df)
        with pytest.raises(TypeError, match=msg):
            df.A.append([df])


class TestSeriesAppendWithDatetimeIndex:
    def test_append(self):
        rng = date_range("5/8/2012 1:45", periods=10, freq="5T")
        ts = Series(np.random.randn(len(rng)), rng)
        df = DataFrame(np.random.randn(len(rng), 4), index=rng)

        result = ts.append(ts)
        result_df = df.append(df)
        ex_index = DatetimeIndex(np.tile(rng.values, 2))
        tm.assert_index_equal(result.index, ex_index)
        tm.assert_index_equal(result_df.index, ex_index)

        appended = rng.append(rng)
        tm.assert_index_equal(appended, ex_index)

        appended = rng.append([rng, rng])
        ex_index = DatetimeIndex(np.tile(rng.values, 3))
        tm.assert_index_equal(appended, ex_index)

        # different index names
        rng1 = rng.copy()
        rng2 = rng.copy()
        rng1.name = "foo"
        rng2.name = "bar"
        assert rng1.append(rng1).name == "foo"
        assert rng1.append(rng2).name is None

    def test_append_tz(self):
        # see gh-2938
        rng = date_range("5/8/2012 1:45", periods=10, freq="5T", tz="US/Eastern")
        rng2 = date_range("5/8/2012 2:35", periods=10, freq="5T", tz="US/Eastern")
        rng3 = date_range("5/8/2012 1:45", periods=20, freq="5T", tz="US/Eastern")
        ts = Series(np.random.randn(len(rng)), rng)
        df = DataFrame(np.random.randn(len(rng), 4), index=rng)
        ts2 = Series(np.random.randn(len(rng2)), rng2)
        df2 = DataFrame(np.random.randn(len(rng2), 4), index=rng2)

        result = ts.append(ts2)
        result_df = df.append(df2)
        tm.assert_index_equal(result.index, rng3)
        tm.assert_index_equal(result_df.index, rng3)

        appended = rng.append(rng2)
        tm.assert_index_equal(appended, rng3)

    def test_append_tz_explicit_pytz(self):
        # see gh-2938
        from pytz import timezone as timezone

        rng = date_range(
            "5/8/2012 1:45", periods=10, freq="5T", tz=timezone("US/Eastern")
        )
        rng2 = date_range(
            "5/8/2012 2:35", periods=10, freq="5T", tz=timezone("US/Eastern")
        )
        rng3 = date_range(
            "5/8/2012 1:45", periods=20, freq="5T", tz=timezone("US/Eastern")
        )
        ts = Series(np.random.randn(len(rng)), rng)
        df = DataFrame(np.random.randn(len(rng), 4), index=rng)
        ts2 = Series(np.random.randn(len(rng2)), rng2)
        df2 = DataFrame(np.random.randn(len(rng2), 4), index=rng2)

        result = ts.append(ts2)
        result_df = df.append(df2)
        tm.assert_index_equal(result.index, rng3)
        tm.assert_index_equal(result_df.index, rng3)

        appended = rng.append(rng2)
        tm.assert_index_equal(appended, rng3)

    def test_append_tz_dateutil(self):
        # see gh-2938
        rng = date_range(
            "5/8/2012 1:45", periods=10, freq="5T", tz="dateutil/US/Eastern"
        )
        rng2 = date_range(
            "5/8/2012 2:35", periods=10, freq="5T", tz="dateutil/US/Eastern"
        )
        rng3 = date_range(
            "5/8/2012 1:45", periods=20, freq="5T", tz="dateutil/US/Eastern"
        )
        ts = Series(np.random.randn(len(rng)), rng)
        df = DataFrame(np.random.randn(len(rng), 4), index=rng)
        ts2 = Series(np.random.randn(len(rng2)), rng2)
        df2 = DataFrame(np.random.randn(len(rng2), 4), index=rng2)

        result = ts.append(ts2)
        result_df = df.append(df2)
        tm.assert_index_equal(result.index, rng3)
        tm.assert_index_equal(result_df.index, rng3)

        appended = rng.append(rng2)
        tm.assert_index_equal(appended, rng3)

    def test_series_append_aware(self):
        rng1 = date_range("1/1/2011 01:00", periods=1, freq="H", tz="US/Eastern")
        rng2 = date_range("1/1/2011 02:00", periods=1, freq="H", tz="US/Eastern")
        ser1 = Series([1], index=rng1)
        ser2 = Series([2], index=rng2)
        ts_result = ser1.append(ser2)

        exp_index = DatetimeIndex(
            ["2011-01-01 01:00", "2011-01-01 02:00"], tz="US/Eastern", freq="H"
        )
        exp = Series([1, 2], index=exp_index)
        tm.assert_series_equal(ts_result, exp)
        assert ts_result.index.tz == rng1.tz

        rng1 = date_range("1/1/2011 01:00", periods=1, freq="H", tz="UTC")
        rng2 = date_range("1/1/2011 02:00", periods=1, freq="H", tz="UTC")
        ser1 = Series([1], index=rng1)
        ser2 = Series([2], index=rng2)
        ts_result = ser1.append(ser2)

        exp_index = DatetimeIndex(
            ["2011-01-01 01:00", "2011-01-01 02:00"], tz="UTC", freq="H"
        )
        exp = Series([1, 2], index=exp_index)
        tm.assert_series_equal(ts_result, exp)
        utc = rng1.tz
        assert utc == ts_result.index.tz

        # GH#7795
        # different tz coerces to object dtype, not UTC
        rng1 = date_range("1/1/2011 01:00", periods=1, freq="H", tz="US/Eastern")
        rng2 = date_range("1/1/2011 02:00", periods=1, freq="H", tz="US/Central")
        ser1 = Series([1], index=rng1)
        ser2 = Series([2], index=rng2)
        ts_result = ser1.append(ser2)
        exp_index = Index(
            [
                Timestamp("1/1/2011 01:00", tz="US/Eastern"),
                Timestamp("1/1/2011 02:00", tz="US/Central"),
            ]
        )
        exp = Series([1, 2], index=exp_index)
        tm.assert_series_equal(ts_result, exp)

    def test_series_append_aware_naive(self):
        rng1 = date_range("1/1/2011 01:00", periods=1, freq="H")
        rng2 = date_range("1/1/2011 02:00", periods=1, freq="H", tz="US/Eastern")
        ser1 = Series(np.random.randn(len(rng1)), index=rng1)
        ser2 = Series(np.random.randn(len(rng2)), index=rng2)
        ts_result = ser1.append(ser2)

        expected = ser1.index.astype(object).append(ser2.index.astype(object))
        assert ts_result.index.equals(expected)

        # mixed
        rng1 = date_range("1/1/2011 01:00", periods=1, freq="H")
        rng2 = range(100)
        ser1 = Series(np.random.randn(len(rng1)), index=rng1)
        ser2 = Series(np.random.randn(len(rng2)), index=rng2)
        ts_result = ser1.append(ser2)

        expected = ser1.index.astype(object).append(ser2.index)
        assert ts_result.index.equals(expected)

    def test_series_append_dst(self):
        rng1 = date_range("1/1/2016 01:00", periods=3, freq="H", tz="US/Eastern")
        rng2 = date_range("8/1/2016 01:00", periods=3, freq="H", tz="US/Eastern")
        ser1 = Series([1, 2, 3], index=rng1)
        ser2 = Series([10, 11, 12], index=rng2)
        ts_result = ser1.append(ser2)

        exp_index = DatetimeIndex(
            [
                "2016-01-01 01:00",
                "2016-01-01 02:00",
                "2016-01-01 03:00",
                "2016-08-01 01:00",
                "2016-08-01 02:00",
                "2016-08-01 03:00",
            ],
            tz="US/Eastern",
        )
        exp = Series([1, 2, 3, 10, 11, 12], index=exp_index)
        tm.assert_series_equal(ts_result, exp)
        assert ts_result.index.tz == rng1.tz
