import numpy as np
import pytest

import pandas as pd
from pandas import (
    PeriodIndex,
    Series,
)
import pandas._testing as tm


def test_to_native_types():
    index = PeriodIndex(["2017-01-01", "2017-01-02", "2017-01-03"], freq="D")

    # First, with no arguments.
    expected = np.array(["2017-01-01", "2017-01-02", "2017-01-03"], dtype="=U10")

    result = index._format_native_types()
    tm.assert_numpy_array_equal(result, expected)

    # No NaN values, so na_rep has no effect
    result = index._format_native_types(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)

    # Make sure date formatting works
    expected = np.array(["01-2017-01", "01-2017-02", "01-2017-03"], dtype="=U10")

    result = index._format_native_types(date_format="%m-%Y-%d")
    tm.assert_numpy_array_equal(result, expected)

    # NULL object handling should work
    index = PeriodIndex(["2017-01-01", pd.NaT, "2017-01-03"], freq="D")
    expected = np.array(["2017-01-01", "NaT", "2017-01-03"], dtype=object)

    result = index._format_native_types()
    tm.assert_numpy_array_equal(result, expected)

    expected = np.array(["2017-01-01", "pandas", "2017-01-03"], dtype=object)

    result = index._format_native_types(na_rep="pandas")
    tm.assert_numpy_array_equal(result, expected)


class TestPeriodIndexRendering:
    def test_frame_repr(self):
        df = pd.DataFrame({"A": [1, 2, 3]}, index=pd.date_range("2000", periods=3))
        result = repr(df)
        expected = "            A\n2000-01-01  1\n2000-01-02  2\n2000-01-03  3"
        assert result == expected

    @pytest.mark.parametrize("method", ["__repr__", "__str__"])
    def test_representation(self, method):
        # GH#7601
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="A")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="H")
        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")
        idx10 = PeriodIndex(["2011-01-01", "2011-02-01"], freq="3D")

        exp1 = "PeriodIndex([], dtype='period[D]')"

        exp2 = "PeriodIndex(['2011-01-01'], dtype='period[D]')"

        exp3 = "PeriodIndex(['2011-01-01', '2011-01-02'], dtype='period[D]')"

        exp4 = (
            "PeriodIndex(['2011-01-01', '2011-01-02', '2011-01-03'], "
            "dtype='period[D]')"
        )

        exp5 = "PeriodIndex(['2011', '2012', '2013'], dtype='period[A-DEC]')"

        exp6 = (
            "PeriodIndex(['2011-01-01 09:00', '2012-02-01 10:00', 'NaT'], "
            "dtype='period[H]')"
        )

        exp7 = "PeriodIndex(['2013Q1'], dtype='period[Q-DEC]')"

        exp8 = "PeriodIndex(['2013Q1', '2013Q2'], dtype='period[Q-DEC]')"

        exp9 = "PeriodIndex(['2013Q1', '2013Q2', '2013Q3'], dtype='period[Q-DEC]')"

        exp10 = "PeriodIndex(['2011-01-01', '2011-02-01'], dtype='period[3D]')"

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10],
        ):
            result = getattr(idx, method)()
            assert result == expected

    def test_representation_to_series(self):
        # GH#10971
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="A")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="H")

        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")

        exp1 = """Series([], dtype: period[D])"""

        exp2 = """0    2011-01-01
dtype: period[D]"""

        exp3 = """0    2011-01-01
1    2011-01-02
dtype: period[D]"""

        exp4 = """0    2011-01-01
1    2011-01-02
2    2011-01-03
dtype: period[D]"""

        exp5 = """0    2011
1    2012
2    2013
dtype: period[A-DEC]"""

        exp6 = """0    2011-01-01 09:00
1    2012-02-01 10:00
2                 NaT
dtype: period[H]"""

        exp7 = """0    2013Q1
dtype: period[Q-DEC]"""

        exp8 = """0    2013Q1
1    2013Q2
dtype: period[Q-DEC]"""

        exp9 = """0    2013Q1
1    2013Q2
2    2013Q3
dtype: period[Q-DEC]"""

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9],
        ):
            result = repr(Series(idx))
            assert result == expected

    def test_summary(self):
        # GH#9116
        idx1 = PeriodIndex([], freq="D")
        idx2 = PeriodIndex(["2011-01-01"], freq="D")
        idx3 = PeriodIndex(["2011-01-01", "2011-01-02"], freq="D")
        idx4 = PeriodIndex(["2011-01-01", "2011-01-02", "2011-01-03"], freq="D")
        idx5 = PeriodIndex(["2011", "2012", "2013"], freq="A")
        idx6 = PeriodIndex(["2011-01-01 09:00", "2012-02-01 10:00", "NaT"], freq="H")

        idx7 = pd.period_range("2013Q1", periods=1, freq="Q")
        idx8 = pd.period_range("2013Q1", periods=2, freq="Q")
        idx9 = pd.period_range("2013Q1", periods=3, freq="Q")

        exp1 = """PeriodIndex: 0 entries
Freq: D"""

        exp2 = """PeriodIndex: 1 entries, 2011-01-01 to 2011-01-01
Freq: D"""

        exp3 = """PeriodIndex: 2 entries, 2011-01-01 to 2011-01-02
Freq: D"""

        exp4 = """PeriodIndex: 3 entries, 2011-01-01 to 2011-01-03
Freq: D"""

        exp5 = """PeriodIndex: 3 entries, 2011 to 2013
Freq: A-DEC"""

        exp6 = """PeriodIndex: 3 entries, 2011-01-01 09:00 to NaT
Freq: H"""

        exp7 = """PeriodIndex: 1 entries, 2013Q1 to 2013Q1
Freq: Q-DEC"""

        exp8 = """PeriodIndex: 2 entries, 2013Q1 to 2013Q2
Freq: Q-DEC"""

        exp9 = """PeriodIndex: 3 entries, 2013Q1 to 2013Q3
Freq: Q-DEC"""

        for idx, expected in zip(
            [idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9],
            [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9],
        ):
            result = idx._summary()
            assert result == expected
