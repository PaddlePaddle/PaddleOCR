from datetime import datetime

import dateutil.tz
import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm


def test_to_native_types_method_deprecated():
    index = pd.date_range(freq="1D", periods=3, start="2017-01-01")
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype=object)

    with tm.assert_produces_warning(FutureWarning):
        result = index.to_native_types()

    tm.assert_numpy_array_equal(result, expected)

    # Make sure slicing works
    expected = np.array(["2017-01-01", "2017-01-03"], dtype=object)

    with tm.assert_produces_warning(FutureWarning):
        result = index.to_native_types([0, 2])

    tm.assert_numpy_array_equal(result, expected)


def test_to_native_types():
    index = pd.date_range(freq="1D", periods=3, start="2017-01-01")

    # First, with no arguments.
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype=object)

    result = index._format_native_types()
    tm.assert_numpy_array_equal(result, expected)

    # No NaN values, so na_rep has no effect
    result = index._format_native_types(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # Make sure date formatting works
    expected = np.array(["01-2017-01", "01-2017-02", "01-2017-03"], dtype=object)

    result = index._format_native_types(date_format="%m-%Y-%d")
    tm.assert_numpy_array_equal(result, expected)

    # NULL object handling should work
    index = DatetimeIndex(["2017-01-01", pd.NaT, "2017-01-03"])
    expected = np.array(["2017-01-01", "NaT", "2017-01-03"], dtype=object)

    result = index._format_native_types()
    tm.assert_numpy_array_equal(result, expected)

    expected = np.array(["2017-01-01", "pandas", "2017-01-03"], dtype=object)

    result = index._format_native_types(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)


class TestDatetimeIndexRendering:
    def test_dti_repr_short(self):
        dr = pd.date_range(start="1/1/2012", periods=1)
        repr(dr)

        dr = pd.date_range(start="1/1/2012", periods=2)
        repr(dr)

        dr = pd.date_range(start="1/1/2012", periods=3)
        repr(dr)

    @pytest.mark.parametrize("method", ["__repr__", "__str__"])
    def test_dti_representation(self, method):
        idxs = []
        idxs.append(DatetimeIndex([], freq="D"))
        idxs.append(DatetimeIndex(["2011-01-01"], freq="D"))
        idxs.append(DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D"))
        idxs.append(DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D"))
        idxs.append(
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
                freq="H",
                tz="Asia/Tokyo",
            )
        )
        idxs.append(
            DatetimeIndex(
                ["2011-01-01 09:00", "2011-01-01 10:00", pd.NaT], tz="US/Eastern"
            )
        )
        idxs.append(
            DatetimeIndex(["2011-01-01 09:00", "2011-01-01 10:00", pd.NaT], tz="UTC")
        )

        exp = []
        exp.append("DatetimeIndex([], dtype='datetime64[ns]', freq='D')")
        exp.append("DatetimeIndex(['2011-01-01'], dtype='datetime64[ns]', freq='D')")
        exp.append(
            "DatetimeIndex(['2011-01-01', '2011-01-02'], "
            "dtype='datetime64[ns]', freq='D')"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], "
            "dtype='datetime64[ns]', freq='D')"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00+09:00', "
            "'2011-01-01 10:00:00+09:00', '2011-01-01 11:00:00+09:00']"
            ", dtype='datetime64[ns, Asia/Tokyo]', freq='H')"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00-05:00', "
            "'2011-01-01 10:00:00-05:00', 'NaT'], "
            "dtype='datetime64[ns, US/Eastern]', freq=None)"
        )
        exp.append(
            "DatetimeIndex(['2011-01-01 09:00:00+00:00', "
            "'2011-01-01 10:00:00+00:00', 'NaT'], "
            "dtype='datetime64[ns, UTC]', freq=None)"
            ""
        )

        with pd.option_context("display.width", 300):
            for indx, expected in zip(idxs, exp):
                result = getattr(indx, method)()
                assert result == expected

    def test_dti_representation_to_series(self):
        idx1 = DatetimeIndex([], freq="D")
        idx2 = DatetimeIndex(["2011-01-01"], freq="D")
        idx3 = DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
            freq="H",
            tz="Asia/Tokyo",
        )
        idx6 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", pd.NaT], tz="US/Eastern"
        )
        idx7 = DatetimeIndex(["2011-01-01 09:00", "2011-01-02 10:15"])

        exp1 = """Series([], dtype: datetime64[ns])"""

        exp2 = "0   2011-01-01\ndtype: datetime64[ns]"

        exp3 = "0   2011-01-01\n1   2011-01-02\ndtype: datetime64[ns]"

        exp4 = (
            "0   2011-01-01\n"
            "1   2011-01-02\n"
            "2   2011-01-03\n"
            "dtype: datetime64[ns]"
        )

        exp5 = (
            "0   2011-01-01 09:00:00+09:00\n"
            "1   2011-01-01 10:00:00+09:00\n"
            "2   2011-01-01 11:00:00+09:00\n"
            "dtype: datetime64[ns, Asia/Tokyo]"
        )

        exp6 = (
            "0   2011-01-01 09:00:00-05:00\n"
            "1   2011-01-01 10:00:00-05:00\n"
            "2                         NaT\n"
            "dtype: datetime64[ns, US/Eastern]"
        )

        exp7 = (
            "0   2011-01-01 09:00:00\n"
            "1   2011-01-02 10:15:00\n"
            "dtype: datetime64[ns]"
        )

        with pd.option_context("display.width", 300):
            for idx, expected in zip(
                [idx1, idx2, idx3, idx4, idx5, idx6, idx7],
                [exp1, exp2, exp3, exp4, exp5, exp6, exp7],
            ):
                result = repr(Series(idx))
                assert result == expected

    def test_dti_summary(self):
        # GH#9116
        idx1 = DatetimeIndex([], freq="D")
        idx2 = DatetimeIndex(["2011-01-01"], freq="D")
        idx3 = DatetimeIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"],
            freq="H",
            tz="Asia/Tokyo",
        )
        idx6 = DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", pd.NaT], tz="US/Eastern"
        )

        exp1 = "DatetimeIndex: 0 entries\nFreq: D"

        exp2 = "DatetimeIndex: 1 entries, 2011-01-01 to 2011-01-01\nFreq: D"

        exp3 = "DatetimeIndex: 2 entries, 2011-01-01 to 2011-01-02\nFreq: D"

        exp4 = "DatetimeIndex: 3 entries, 2011-01-01 to 2011-01-03\nFreq: D"

        exp5 = (
            "DatetimeIndex: 3 entries, 2011-01-01 09:00:00+09:00 "
            "to 2011-01-01 11:00:00+09:00\n"
            "Freq: H"
        )

        exp6 = """DatetimeIndex: 3 entries, 2011-01-01 09:00:00-05:00 to NaT"""

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6], [exp1, exp2, exp3, exp4, exp5, exp6]
        ):
            result = idx._summary()
            assert result == expected

    def test_dti_business_repr(self):
        # only really care that it works
        repr(pd.bdate_range(datetime(2009, 1, 1), datetime(2010, 1, 1)))

    def test_dti_business_summary(self):
        rng = pd.bdate_range(datetime(2009, 1, 1), datetime(2010, 1, 1))
        rng._summary()
        rng[2:2]._summary()

    def test_dti_business_summary_pytz(self):
        pd.bdate_range("1/1/2005", "1/1/2009", tz=pytz.utc)._summary()

    def test_dti_business_summary_dateutil(self):
        pd.bdate_range("1/1/2005", "1/1/2009", tz=dateutil.tz.tzutc())._summary()

    def test_dti_custom_business_repr(self):
        # only really care that it works
        repr(pd.bdate_range(datetime(2009, 1, 1), datetime(2010, 1, 1), freq="C"))

    def test_dti_custom_business_summary(self):
        rng = pd.bdate_range(datetime(2009, 1, 1), datetime(2010, 1, 1), freq="C")
        rng._summary()
        rng[2:2]._summary()

    def test_dti_custom_business_summary_pytz(self):
        pd.bdate_range("1/1/2005", "1/1/2009", freq="C", tz=pytz.utc)._summary()

    def test_dti_custom_business_summary_dateutil(self):
        pd.bdate_range(
            "1/1/2005", "1/1/2009", freq="C", tz=dateutil.tz.tzutc()
        )._summary()
