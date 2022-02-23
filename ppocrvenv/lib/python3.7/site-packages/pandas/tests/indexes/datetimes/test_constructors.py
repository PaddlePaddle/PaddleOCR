from datetime import (
    datetime,
    timedelta,
    timezone,
)
from functools import partial
from operator import attrgetter

import dateutil
import numpy as np
import pytest
import pytz

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    conversion,
)

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Timestamp,
    date_range,
    offsets,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    period_array,
)


class TestDatetimeIndex:
    @pytest.mark.parametrize(
        "dt_cls", [DatetimeIndex, DatetimeArray._from_sequence_not_strict]
    )
    def test_freq_validation_with_nat(self, dt_cls):
        # GH#11587 make sure we get a useful error message when generate_range
        #  raises
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            dt_cls([pd.NaT, Timestamp("2011-01-01")], freq="D")
        with pytest.raises(ValueError, match=msg):
            dt_cls([pd.NaT, Timestamp("2011-01-01").value], freq="D")

    # TODO: better place for tests shared by DTI/TDI?
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2016-01-01", periods=5, tz="US/Pacific"),
            pd.timedelta_range("1 Day", periods=5),
        ],
    )
    def test_shallow_copy_inherits_array_freq(self, index):
        # If we pass a DTA/TDA to shallow_copy and dont specify a freq,
        #  we should inherit the array's freq, not our own.
        array = index._data

        arr = array[[0, 3, 2, 4, 1]]
        assert arr.freq is None

        result = index._shallow_copy(arr)
        assert result.freq is None

    def test_categorical_preserves_tz(self):
        # GH#18664 retain tz when going DTI-->Categorical-->DTI
        # TODO: parametrize over DatetimeIndex/DatetimeArray
        #  once pd.CategoricalIndex(DTA) works

        dti = DatetimeIndex(
            [pd.NaT, "2015-01-01", "1999-04-06 15:14:13", "2015-01-01"], tz="US/Eastern"
        )

        ci = pd.CategoricalIndex(dti)
        carr = pd.Categorical(dti)
        cser = pd.Series(ci)

        for obj in [ci, carr, cser]:
            result = DatetimeIndex(obj)
            tm.assert_index_equal(result, dti)

    def test_dti_with_period_data_raises(self):
        # GH#23675
        data = pd.PeriodIndex(["2016Q1", "2016Q2"], freq="Q")

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(data)

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            DatetimeIndex(period_array(data))

        with pytest.raises(TypeError, match="PeriodDtype data is invalid"):
            to_datetime(period_array(data))

    def test_dti_with_timedelta64_data_raises(self):
        # GH#23675 deprecated, enforrced in GH#29794
        data = np.array([0], dtype="m8[ns]")
        msg = r"timedelta64\[ns\] cannot be converted to datetime64"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(data)

        with pytest.raises(TypeError, match=msg):
            to_datetime(data)

        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(pd.TimedeltaIndex(data))

        with pytest.raises(TypeError, match=msg):
            to_datetime(pd.TimedeltaIndex(data))

    def test_constructor_from_sparse_array(self):
        # https://github.com/pandas-dev/pandas/issues/35843
        values = [
            Timestamp("2012-05-01T01:00:00.000000"),
            Timestamp("2016-05-01T01:00:00.000000"),
        ]
        arr = pd.arrays.SparseArray(values)
        result = Index(arr)
        expected = DatetimeIndex(values)
        tm.assert_index_equal(result, expected)

    def test_construction_caching(self):

        df = pd.DataFrame(
            {
                "dt": date_range("20130101", periods=3),
                "dttz": date_range("20130101", periods=3, tz="US/Eastern"),
                "dt_with_null": [
                    Timestamp("20130101"),
                    pd.NaT,
                    Timestamp("20130103"),
                ],
                "dtns": date_range("20130101", periods=3, freq="ns"),
            }
        )
        assert df.dttz.dtype.tz.zone == "US/Eastern"

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt(self, kwargs, tz_aware_fixture):
        tz = tz_aware_fixture
        i = date_range("20130101", periods=5, freq="H", tz=tz)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
        result = DatetimeIndex(i, **kwargs)
        tm.assert_index_equal(i, result)

    @pytest.mark.parametrize(
        "kwargs",
        [{"tz": "dtype.tz"}, {"dtype": "dtype"}, {"dtype": "dtype", "tz": "dtype.tz"}],
    )
    def test_construction_with_alt_tz_localize(self, kwargs, tz_aware_fixture):
        tz = tz_aware_fixture
        i = date_range("20130101", periods=5, freq="H", tz=tz)
        i = i._with_freq(None)
        kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}

        if "tz" in kwargs:
            result = DatetimeIndex(i.asi8, tz="UTC").tz_convert(kwargs["tz"])

            expected = DatetimeIndex(i, **kwargs)
            tm.assert_index_equal(result, expected)

        # localize into the provided tz
        i2 = DatetimeIndex(i.tz_localize(None).asi8, tz="UTC")
        expected = i.tz_localize(None).tz_localize("UTC")
        tm.assert_index_equal(i2, expected)

        # incompat tz/dtype
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(i.tz_localize(None).asi8, dtype=i.dtype, tz="US/Pacific")

    def test_construction_index_with_mixed_timezones(self):
        # gh-11488: no tz results in DatetimeIndex
        result = Index([Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx")
        exp = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # same tz results in DatetimeIndex
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-02 10:00")],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # Different tz results in Index(dtype=object)
        result = Index(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                Timestamp("2011-01-01 10:00"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        # length = 1
        result = Index([Timestamp("2011-01-01")], name="idx")
        exp = DatetimeIndex([Timestamp("2011-01-01")], name="idx")
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # length = 1 with tz
        result = Index([Timestamp("2011-01-01 10:00", tz="Asia/Tokyo")], name="idx")
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00")], tz="Asia/Tokyo", name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_index_with_mixed_timezones_with_NaT(self):
        # see gh-11488
        result = Index(
            [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
            name="idx",
        )
        exp = DatetimeIndex(
            [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # Same tz results in DatetimeIndex
        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00"),
            ],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # same tz results in DatetimeIndex (DST)
        result = Index(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                pd.NaT,
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), pd.NaT, Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

        # different tz results in Index(dtype=object)
        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        result = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = Index(
            [
                pd.NaT,
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                pd.NaT,
                Timestamp("2011-01-02 10:00", tz="US/Eastern"),
            ],
            dtype="object",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert not isinstance(result, DatetimeIndex)

        # all NaT
        result = Index([pd.NaT, pd.NaT], name="idx")
        exp = DatetimeIndex([pd.NaT, pd.NaT], name="idx")
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is None

        # all NaT with tz
        with tm.assert_produces_warning(FutureWarning):
            # subclass-specific kwargs to pd.Index
            result = Index([pd.NaT, pd.NaT], tz="Asia/Tokyo", name="idx")
        exp = DatetimeIndex([pd.NaT, pd.NaT], tz="Asia/Tokyo", name="idx")

        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)
        assert result.tz is not None
        assert result.tz == exp.tz

    def test_construction_dti_with_mixed_timezones(self):
        # GH 11488 (not changed, added explicit tests)

        # no tz results in DatetimeIndex
        result = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01"), Timestamp("2011-01-02")], name="idx"
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex
        result = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-01-02 10:00")],
            tz="Asia/Tokyo",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # same tz results in DatetimeIndex (DST)
        result = DatetimeIndex(
            [
                Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                Timestamp("2011-08-01 10:00", tz="US/Eastern"),
            ],
            name="idx",
        )
        exp = DatetimeIndex(
            [Timestamp("2011-01-01 10:00"), Timestamp("2011-08-01 10:00")],
            tz="US/Eastern",
            name="idx",
        )
        tm.assert_index_equal(result, exp, exact=True)
        assert isinstance(result, DatetimeIndex)

        # tz mismatch affecting to tz-aware raises TypeError/ValueError

        msg = "cannot be converted to datetime64"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                [
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )

        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                [
                    Timestamp("2011-01-01 10:00"),
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                tz="Asia/Tokyo",
                name="idx",
            )

        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(
                [
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                tz="US/Eastern",
                name="idx",
            )

        with pytest.raises(ValueError, match=msg):
            # passing tz should results in DatetimeIndex, then mismatch raises
            # TypeError
            with tm.assert_produces_warning(FutureWarning):
                # subclass-specific kwargs to pd.Index
                Index(
                    [
                        pd.NaT,
                        Timestamp("2011-01-01 10:00"),
                        pd.NaT,
                        Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                    ],
                    tz="Asia/Tokyo",
                    name="idx",
                )

    def test_construction_base_constructor(self):
        arr = [Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-03")]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

        arr = [np.nan, pd.NaT, Timestamp("2011-01-03")]
        tm.assert_index_equal(Index(arr), DatetimeIndex(arr))
        tm.assert_index_equal(Index(np.array(arr)), DatetimeIndex(np.array(arr)))

    def test_construction_outofbounds(self):
        # GH 13663
        dates = [
            datetime(3000, 1, 1),
            datetime(4000, 1, 1),
            datetime(5000, 1, 1),
            datetime(6000, 1, 1),
        ]
        exp = Index(dates, dtype=object)
        # coerces to object
        tm.assert_index_equal(Index(dates), exp)

        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            # can't create DatetimeIndex
            DatetimeIndex(dates)

    def test_construction_with_ndarray(self):
        # GH 5152
        dates = [datetime(2013, 10, 7), datetime(2013, 10, 8), datetime(2013, 10, 9)]
        data = DatetimeIndex(dates, freq=offsets.BDay()).values
        result = DatetimeIndex(data, freq=offsets.BDay())
        expected = DatetimeIndex(["2013-10-07", "2013-10-08", "2013-10-09"], freq="B")
        tm.assert_index_equal(result, expected)

    def test_integer_values_and_tz_interpreted_as_utc(self):
        # GH-24559
        val = np.datetime64("2000-01-01 00:00:00", "ns")
        values = np.array([val.view("i8")])

        result = DatetimeIndex(values).tz_localize("US/Central")

        expected = DatetimeIndex(["2000-01-01T00:00:00"], tz="US/Central")
        tm.assert_index_equal(result, expected)

        # but UTC is *not* deprecated.
        with tm.assert_produces_warning(None):
            result = DatetimeIndex(values, tz="UTC")
        expected = DatetimeIndex(["2000-01-01T00:00:00"], tz="US/Central")

    def test_constructor_coverage(self):
        rng = date_range("1/1/2000", periods=10.5)
        exp = date_range("1/1/2000", periods=10)
        tm.assert_index_equal(rng, exp)

        msg = "periods must be a number, got foo"
        with pytest.raises(TypeError, match=msg):
            date_range(start="1/1/2000", periods="foo", freq="D")

        msg = r"DatetimeIndex\(\.\.\.\) must be called with a collection"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex("1/1/2000")

        # generator expression
        gen = (datetime(2000, 1, 1) + timedelta(i) for i in range(10))
        result = DatetimeIndex(gen)
        expected = DatetimeIndex(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(10)]
        )
        tm.assert_index_equal(result, expected)

        # NumPy string array
        strings = np.array(["2000-01-01", "2000-01-02", "2000-01-03"])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype("O"))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.asi8)
        tm.assert_index_equal(from_ints, expected)

        # string with NaT
        strings = np.array(["2000-01-01", "2000-01-02", "NaT"])
        result = DatetimeIndex(strings)
        expected = DatetimeIndex(strings.astype("O"))
        tm.assert_index_equal(result, expected)

        from_ints = DatetimeIndex(expected.asi8)
        tm.assert_index_equal(from_ints, expected)

        # non-conforming
        msg = (
            "Inferred frequency None from passed values does not conform "
            "to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-04"], freq="D")

        msg = (
            "Of the four parameters: start, end, periods, and freq, exactly "
            "three must be specified"
        )
        with pytest.raises(ValueError, match=msg):
            date_range(start="2011-01-01", freq="b")
        with pytest.raises(ValueError, match=msg):
            date_range(end="2011-01-01", freq="B")
        with pytest.raises(ValueError, match=msg):
            date_range(periods=10, freq="D")

    @pytest.mark.parametrize("freq", ["AS", "W-SUN"])
    def test_constructor_datetime64_tzformat(self, freq):
        # see GH#6572: ISO 8601 format results in pytz.FixedOffset
        idx = date_range(
            "2013-01-01T00:00:00-05:00", "2016-01-01T23:59:59-05:00", freq=freq
        )
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=pytz.FixedOffset(-300),
        )
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range(
            "2013-01-01T00:00:00+09:00", "2016-01-01T23:59:59+09:00", freq=freq
        )
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=pytz.FixedOffset(540),
        )
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        # Non ISO 8601 format results in dateutil.tz.tzoffset
        idx = date_range("2013/1/1 0:00:00-5:00", "2016/1/1 23:59:59-5:00", freq=freq)
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=pytz.FixedOffset(-300),
        )
        tm.assert_index_equal(idx, expected)
        # Unable to use `US/Eastern` because of DST
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="America/Lima"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

        idx = date_range("2013/1/1 0:00:00+9:00", "2016/1/1 23:59:59+09:00", freq=freq)
        expected = date_range(
            "2013-01-01T00:00:00",
            "2016-01-01T23:59:59",
            freq=freq,
            tz=pytz.FixedOffset(540),
        )
        tm.assert_index_equal(idx, expected)
        expected_i8 = date_range(
            "2013-01-01T00:00:00", "2016-01-01T23:59:59", freq=freq, tz="Asia/Tokyo"
        )
        tm.assert_numpy_array_equal(idx.asi8, expected_i8.asi8)

    def test_constructor_dtype(self):

        # passing a dtype with a tz should localize
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )
        expected = DatetimeIndex(["2013-01-01", "2013-01-02"]).tz_localize("US/Eastern")
        tm.assert_index_equal(idx, expected)

        idx = DatetimeIndex(["2013-01-01", "2013-01-02"], tz="US/Eastern")
        tm.assert_index_equal(idx, expected)

        # if we already have a tz and its not the same, then raise
        idx = DatetimeIndex(
            ["2013-01-01", "2013-01-02"], dtype="datetime64[ns, US/Eastern]"
        )

        msg = (
            "cannot supply both a tz and a timezone-naive dtype "
            r"\(i\.e\. datetime64\[ns\]\)"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, dtype="datetime64[ns]")

        # this is effectively trying to convert tz's
        msg = "data is already tz-aware US/Eastern, unable to set specified tz: CET"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(idx, dtype="datetime64[ns, CET]")
        msg = "cannot supply both a tz and a dtype with a tz"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(idx, tz="CET", dtype="datetime64[ns, US/Eastern]")

        result = DatetimeIndex(idx, dtype="datetime64[ns, US/Eastern]")
        tm.assert_index_equal(idx, result)

    @pytest.mark.parametrize("dtype", [object, np.int32, np.int64])
    def test_constructor_invalid_dtype_raises(self, dtype):
        # GH 23986
        msg = "Unexpected value for 'dtype'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex([1, 2], dtype=dtype)

    def test_constructor_name(self):
        idx = date_range(start="2000-01-01", periods=1, freq="A", name="TEST")
        assert idx.name == "TEST"

    def test_000constructor_resolution(self):
        # 2252
        t1 = Timestamp((1352934390 * 1000000000) + 1000000 + 1000 + 1)
        idx = DatetimeIndex([t1])

        assert idx.nanosecond[0] == t1.nanosecond

    def test_disallow_setting_tz(self):
        # GH 3746
        dti = DatetimeIndex(["2010"], tz="UTC")
        msg = "Cannot directly set timezone"
        with pytest.raises(AttributeError, match=msg):
            dti.tz = pytz.timezone("US/Pacific")

    @pytest.mark.parametrize(
        "tz",
        [
            None,
            "America/Los_Angeles",
            pytz.timezone("America/Los_Angeles"),
            Timestamp("2000", tz="America/Los_Angeles").tz,
        ],
    )
    def test_constructor_start_end_with_tz(self, tz):
        # GH 18595
        start = Timestamp("2013-01-01 06:00:00", tz="America/Los_Angeles")
        end = Timestamp("2013-01-02 06:00:00", tz="America/Los_Angeles")
        result = date_range(freq="D", start=start, end=end, tz=tz)
        expected = DatetimeIndex(
            ["2013-01-01 06:00:00", "2013-01-02 06:00:00"],
            tz="America/Los_Angeles",
            freq="D",
        )
        tm.assert_index_equal(result, expected)
        # Especially assert that the timezone is consistent for pytz
        assert pytz.timezone("America/Los_Angeles") is result.tz

    @pytest.mark.parametrize("tz", ["US/Pacific", "US/Eastern", "Asia/Tokyo"])
    def test_constructor_with_non_normalized_pytz(self, tz):
        # GH 18595
        non_norm_tz = Timestamp("2010", tz=tz).tz
        result = DatetimeIndex(["2010"], tz=non_norm_tz)
        assert pytz.timezone(tz) is result.tz

    def test_constructor_timestamp_near_dst(self):
        # GH 20854
        ts = [
            Timestamp("2016-10-30 03:00:00+0300", tz="Europe/Helsinki"),
            Timestamp("2016-10-30 03:00:00+0200", tz="Europe/Helsinki"),
        ]
        result = DatetimeIndex(ts)
        expected = DatetimeIndex([ts[0].to_pydatetime(), ts[1].to_pydatetime()])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    @pytest.mark.parametrize("box", [np.array, partial(np.array, dtype=object), list])
    @pytest.mark.parametrize(
        "tz, dtype",
        [("US/Pacific", "datetime64[ns, US/Pacific]"), (None, "datetime64[ns]")],
    )
    def test_constructor_with_int_tz(self, klass, box, tz, dtype):
        # GH 20997, 20964
        ts = Timestamp("2018-01-01", tz=tz)
        result = klass(box([ts.value]), dtype=dtype)
        expected = klass([ts])
        assert result == expected

    def test_construction_int_rountrip(self, tz_naive_fixture):
        # GH 12619, GH#24559
        tz = tz_naive_fixture

        result = 1293858000000000000
        expected = DatetimeIndex([result], tz=tz).asi8[0]
        assert result == expected

    def test_construction_from_replaced_timestamps_with_dst(self):
        # GH 18785
        index = date_range(
            Timestamp(2000, 1, 1),
            Timestamp(2005, 1, 1),
            freq="MS",
            tz="Australia/Melbourne",
        )
        test = pd.DataFrame({"data": range(len(index))}, index=index)
        test = test.resample("Y").mean()
        result = DatetimeIndex([x.replace(month=6, day=1) for x in test.index])
        expected = DatetimeIndex(
            [
                "2000-06-01 00:00:00",
                "2001-06-01 00:00:00",
                "2002-06-01 00:00:00",
                "2003-06-01 00:00:00",
                "2004-06-01 00:00:00",
                "2005-06-01 00:00:00",
            ],
            tz="Australia/Melbourne",
        )
        tm.assert_index_equal(result, expected)

    def test_construction_with_tz_and_tz_aware_dti(self):
        # GH 23579
        dti = date_range("2016-01-01", periods=3, tz="US/Central")
        msg = "data is already tz-aware US/Central, unable to set specified tz"
        with pytest.raises(TypeError, match=msg):
            DatetimeIndex(dti, tz="Asia/Tokyo")

    def test_construction_with_nat_and_tzlocal(self):
        tz = dateutil.tz.tzlocal()
        result = DatetimeIndex(["2018", "NaT"], tz=tz)
        expected = DatetimeIndex([Timestamp("2018", tz=tz), pd.NaT])
        tm.assert_index_equal(result, expected)

    def test_constructor_with_ambiguous_keyword_arg(self):
        # GH 35297

        expected = DatetimeIndex(
            ["2020-11-01 01:00:00", "2020-11-02 01:00:00"],
            dtype="datetime64[ns, America/New_York]",
            freq="D",
            ambiguous=False,
        )

        # ambiguous keyword in start
        timezone = "America/New_York"
        start = Timestamp(year=2020, month=11, day=1, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        result = date_range(start=start, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

        # ambiguous keyword in end
        timezone = "America/New_York"
        end = Timestamp(year=2020, month=11, day=2, hour=1).tz_localize(
            timezone, ambiguous=False
        )
        result = date_range(end=end, periods=2, ambiguous=False)
        tm.assert_index_equal(result, expected)

    def test_constructor_with_nonexistent_keyword_arg(self):
        # GH 35297

        timezone = "Europe/Warsaw"

        # nonexistent keyword in start
        start = Timestamp("2015-03-29 02:30:00").tz_localize(
            timezone, nonexistent="shift_forward"
        )
        result = date_range(start=start, periods=2, freq="H")
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
                Timestamp("2015-03-29 04:00:00+02:00", tz=timezone),
            ]
        )

        tm.assert_index_equal(result, expected)

        # nonexistent keyword in end
        end = Timestamp("2015-03-29 02:30:00").tz_localize(
            timezone, nonexistent="shift_forward"
        )
        result = date_range(end=end, periods=2, freq="H")
        expected = DatetimeIndex(
            [
                Timestamp("2015-03-29 01:00:00+01:00", tz=timezone),
                Timestamp("2015-03-29 03:00:00+02:00", tz=timezone),
            ]
        )

        tm.assert_index_equal(result, expected)

    def test_constructor_no_precision_raises(self):
        # GH-24753, GH-24739

        msg = "with no precision is not allowed"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000"], dtype="datetime64")

        with pytest.raises(ValueError, match=msg):
            Index(["2000"], dtype="datetime64")

    def test_constructor_wrong_precision_raises(self):
        msg = "Unexpected value for 'dtype': 'datetime64\\[us\\]'"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(["2000"], dtype="datetime64[us]")

    def test_index_constructor_with_numpy_object_array_and_timestamp_tz_with_nan(self):
        # GH 27011
        result = Index(np.array([Timestamp("2019", tz="UTC"), np.nan], dtype=object))
        expected = DatetimeIndex([Timestamp("2019", tz="UTC"), pd.NaT])
        tm.assert_index_equal(result, expected)


class TestTimeSeries:
    def test_dti_constructor_preserve_dti_freq(self):
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        rng2 = DatetimeIndex(rng)
        assert rng.freq == rng2.freq

    def test_explicit_none_freq(self):
        # Explicitly passing freq=None is respected
        rng = date_range("1/1/2000", "1/2/2000", freq="5min")

        result = DatetimeIndex(rng, freq=None)
        assert result.freq is None

        result = DatetimeIndex(rng._data, freq=None)
        assert result.freq is None

    def test_dti_constructor_years_only(self, tz_naive_fixture):
        tz = tz_naive_fixture
        # GH 6961
        rng1 = date_range("2014", "2015", freq="M", tz=tz)
        expected1 = date_range("2014-01-31", "2014-12-31", freq="M", tz=tz)

        rng2 = date_range("2014", "2015", freq="MS", tz=tz)
        expected2 = date_range("2014-01-01", "2015-01-01", freq="MS", tz=tz)

        rng3 = date_range("2014", "2020", freq="A", tz=tz)
        expected3 = date_range("2014-12-31", "2019-12-31", freq="A", tz=tz)

        rng4 = date_range("2014", "2020", freq="AS", tz=tz)
        expected4 = date_range("2014-01-01", "2020-01-01", freq="AS", tz=tz)

        for rng, expected in [
            (rng1, expected1),
            (rng2, expected2),
            (rng3, expected3),
            (rng4, expected4),
        ]:
            tm.assert_index_equal(rng, expected)

    def test_dti_constructor_small_int(self, any_int_dtype):
        # see gh-13721
        exp = DatetimeIndex(
            [
                "1970-01-01 00:00:00.00000000",
                "1970-01-01 00:00:00.00000001",
                "1970-01-01 00:00:00.00000002",
            ]
        )

        arr = np.array([0, 10, 20], dtype=any_int_dtype)
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    def test_ctor_str_intraday(self):
        rng = DatetimeIndex(["1-1-2000 00:00:01"])
        assert rng[0].second == 1

    def test_is_(self):
        dti = date_range(start="1/1/2005", end="12/1/2005", freq="M")
        assert dti.is_(dti)
        assert dti.is_(dti.view())
        assert not dti.is_(dti.copy())

    def test_index_cast_datetime64_other_units(self):
        arr = np.arange(0, 100, 10, dtype=np.int64).view("M8[D]")
        idx = Index(arr)

        assert (idx.values == conversion.ensure_datetime64ns(arr)).all()

    def test_constructor_int64_nocopy(self):
        # GH#1624
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr)

        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()

        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=True)

        arr[50:100] = -1
        assert (index.asi8[50:100] != -1).all()

    @pytest.mark.parametrize(
        "freq", ["M", "Q", "A", "D", "B", "BH", "T", "S", "L", "U", "H", "N", "C"]
    )
    def test_from_freq_recreate_from_data(self, freq):
        org = date_range(start="2001/02/01 09:00", freq=freq, periods=1)
        idx = DatetimeIndex(org, freq=freq)
        tm.assert_index_equal(idx, org)

        org = date_range(
            start="2001/02/01 09:00", freq=freq, tz="US/Pacific", periods=1
        )
        idx = DatetimeIndex(org, freq=freq, tz="US/Pacific")
        tm.assert_index_equal(idx, org)

    def test_datetimeindex_constructor_misc(self):
        arr = ["1/1/2005", "1/2/2005", "Jn 3, 2005", "2005-01-04"]
        msg = r"(\(')?Unknown string format(:', 'Jn 3, 2005'\))?"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)

        arr = ["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"]
        idx1 = DatetimeIndex(arr)

        arr = [datetime(2005, 1, 1), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx2 = DatetimeIndex(arr)

        arr = [Timestamp(datetime(2005, 1, 1)), "1/2/2005", "1/3/2005", "2005-01-04"]
        idx3 = DatetimeIndex(arr)

        arr = np.array(["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"], dtype="O")
        idx4 = DatetimeIndex(arr)

        arr = to_datetime(["1/1/2005", "1/2/2005", "1/3/2005", "2005-01-04"])
        idx5 = DatetimeIndex(arr)

        arr = to_datetime(["1/1/2005", "1/2/2005", "Jan 3, 2005", "2005-01-04"])
        idx6 = DatetimeIndex(arr)

        idx7 = DatetimeIndex(["12/05/2007", "25/01/2008"], dayfirst=True)
        idx8 = DatetimeIndex(
            ["2007/05/12", "2008/01/25"], dayfirst=False, yearfirst=True
        )
        tm.assert_index_equal(idx7, idx8)

        for other in [idx2, idx3, idx4, idx5, idx6]:
            assert (idx1.values == other.values).all()

        sdate = datetime(1999, 12, 25)
        edate = datetime(2000, 1, 1)
        idx = date_range(start=sdate, freq="1B", periods=20)
        assert len(idx) == 20
        assert idx[0] == sdate + 0 * offsets.BDay()
        assert idx.freq == "B"

        idx1 = date_range(start=sdate, end=edate, freq="W-SUN")
        idx2 = date_range(start=sdate, end=edate, freq=offsets.Week(weekday=6))
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq

        idx1 = date_range(start=sdate, end=edate, freq="QS")
        idx2 = date_range(
            start=sdate, end=edate, freq=offsets.QuarterBegin(startingMonth=1)
        )
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq

        idx1 = date_range(start=sdate, end=edate, freq="BQ")
        idx2 = date_range(
            start=sdate, end=edate, freq=offsets.BQuarterEnd(startingMonth=12)
        )
        assert len(idx1) == len(idx2)
        assert idx1.freq == idx2.freq

    def test_pass_datetimeindex_to_index(self):
        # Bugs in #1396
        rng = date_range("1/1/2000", "3/1/2000")
        idx = Index(rng, dtype=object)

        expected = Index(rng.to_pydatetime(), dtype=object)

        tm.assert_numpy_array_equal(idx.values, expected.values)

    def test_date_range_tuple_freq_raises(self):
        # GH#34703
        edate = datetime(2000, 1, 1)
        with pytest.raises(TypeError, match="pass as a string instead"):
            date_range(end=edate, freq=("D", 5), periods=20)


def test_timestamp_constructor_invalid_fold_raise():
    # Test for #25057
    # Valid fold values are only [None, 0, 1]
    msg = "Valid values for the fold argument are None, 0, or 1."
    with pytest.raises(ValueError, match=msg):
        Timestamp(123, fold=2)


def test_timestamp_constructor_pytz_fold_raise():
    # Test for #25057
    # pytz doesn't support fold. Check that we raise
    # if fold is passed with pytz
    msg = "pytz timezones do not support fold. Please use dateutil timezones."
    tz = pytz.timezone("Europe/London")
    with pytest.raises(ValueError, match=msg):
        Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)


@pytest.mark.parametrize("fold", [0, 1])
@pytest.mark.parametrize(
    "ts_input",
    [
        1572136200000000000,
        1572136200000000000.0,
        np.datetime64(1572136200000000000, "ns"),
        "2019-10-27 01:30:00+01:00",
        datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc),
    ],
)
def test_timestamp_constructor_fold_conflict(ts_input, fold):
    # Test for #25057
    # Check that we raise on fold conflict
    msg = (
        "Cannot pass fold with possibly unambiguous input: int, float, "
        "numpy.datetime64, str, or timezone-aware datetime-like. "
        "Pass naive datetime-like or build Timestamp from components."
    )
    with pytest.raises(ValueError, match=msg):
        Timestamp(ts_input=ts_input, fold=fold)


@pytest.mark.parametrize("tz", ["dateutil/Europe/London", None])
@pytest.mark.parametrize("fold", [0, 1])
def test_timestamp_constructor_retain_fold(tz, fold):
    # Test for #25057
    # Check that we retain fold
    ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
    result = ts.fold
    expected = fold
    assert result == expected


@pytest.mark.parametrize("tz", ["dateutil/Europe/London"])
@pytest.mark.parametrize(
    "ts_input,fold_out",
    [
        (1572136200000000000, 0),
        (1572139800000000000, 1),
        ("2019-10-27 01:30:00+01:00", 0),
        ("2019-10-27 01:30:00+00:00", 1),
        (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0),
        (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1),
    ],
)
def test_timestamp_constructor_infer_fold_from_value(tz, ts_input, fold_out):
    # Test for #25057
    # Check that we infer fold correctly based on timestamps since utc
    # or strings
    ts = Timestamp(ts_input, tz=tz)
    result = ts.fold
    expected = fold_out
    assert result == expected


@pytest.mark.parametrize("tz", ["dateutil/Europe/London"])
@pytest.mark.parametrize(
    "ts_input,fold,value_out",
    [
        (datetime(2019, 10, 27, 1, 30, 0, 0), 0, 1572136200000000000),
        (datetime(2019, 10, 27, 1, 30, 0, 0), 1, 1572139800000000000),
    ],
)
def test_timestamp_constructor_adjust_value_for_fold(tz, ts_input, fold, value_out):
    # Test for #25057
    # Check that we adjust value for fold correctly
    # based on timestamps since utc
    ts = Timestamp(ts_input, tz=tz, fold=fold)
    result = ts.value
    expected = value_out
    assert result == expected
