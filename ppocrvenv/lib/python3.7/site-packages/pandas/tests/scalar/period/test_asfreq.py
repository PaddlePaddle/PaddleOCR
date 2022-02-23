import pytest

from pandas._libs.tslibs.dtypes import _period_code_map
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime

from pandas import (
    Period,
    Timestamp,
    offsets,
)


class TestFreqConversion:
    """Test frequency conversion of date objects"""

    @pytest.mark.parametrize("freq", ["A", "Q", "M", "W", "B", "D"])
    def test_asfreq_near_zero(self, freq):
        # GH#19643, GH#19650
        per = Period("0001-01-01", freq=freq)
        tup1 = (per.year, per.hour, per.day)

        prev = per - 1
        assert prev.ordinal == per.ordinal - 1
        tup2 = (prev.year, prev.month, prev.day)
        assert tup2 < tup1

    def test_asfreq_near_zero_weekly(self):
        # GH#19834
        per1 = Period("0001-01-01", "D") + 6
        per2 = Period("0001-01-01", "D") - 6
        week1 = per1.asfreq("W")
        week2 = per2.asfreq("W")
        assert week1 != week2
        assert week1.asfreq("D", "E") >= per1
        assert week2.asfreq("D", "S") <= per2

    def test_to_timestamp_out_of_bounds(self):
        # GH#19643, used to incorrectly give Timestamp in 1754
        per = Period("0001-01-01", freq="B")
        msg = "Out of bounds nanosecond timestamp"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per.to_timestamp()

    def test_asfreq_corner(self):
        val = Period(freq="A", year=2007)
        result1 = val.asfreq("5t")
        result2 = val.asfreq("t")
        expected = Period("2007-12-31 23:59", freq="t")
        assert result1.ordinal == expected.ordinal
        assert result1.freqstr == "5T"
        assert result2.ordinal == expected.ordinal
        assert result2.freqstr == "T"

    def test_conv_annual(self):
        # frequency conversion tests: from Annual Frequency

        ival_A = Period(freq="A", year=2007)

        ival_AJAN = Period(freq="A-JAN", year=2007)
        ival_AJUN = Period(freq="A-JUN", year=2007)
        ival_ANOV = Period(freq="A-NOV", year=2007)

        ival_A_to_Q_start = Period(freq="Q", year=2007, quarter=1)
        ival_A_to_Q_end = Period(freq="Q", year=2007, quarter=4)
        ival_A_to_M_start = Period(freq="M", year=2007, month=1)
        ival_A_to_M_end = Period(freq="M", year=2007, month=12)
        ival_A_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        ival_A_to_W_end = Period(freq="W", year=2007, month=12, day=31)
        ival_A_to_B_start = Period(freq="B", year=2007, month=1, day=1)
        ival_A_to_B_end = Period(freq="B", year=2007, month=12, day=31)
        ival_A_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_A_to_D_end = Period(freq="D", year=2007, month=12, day=31)
        ival_A_to_H_start = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_A_to_H_end = Period(freq="H", year=2007, month=12, day=31, hour=23)
        ival_A_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_A_to_T_end = Period(
            freq="Min", year=2007, month=12, day=31, hour=23, minute=59
        )
        ival_A_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_A_to_S_end = Period(
            freq="S", year=2007, month=12, day=31, hour=23, minute=59, second=59
        )

        ival_AJAN_to_D_end = Period(freq="D", year=2007, month=1, day=31)
        ival_AJAN_to_D_start = Period(freq="D", year=2006, month=2, day=1)
        ival_AJUN_to_D_end = Period(freq="D", year=2007, month=6, day=30)
        ival_AJUN_to_D_start = Period(freq="D", year=2006, month=7, day=1)
        ival_ANOV_to_D_end = Period(freq="D", year=2007, month=11, day=30)
        ival_ANOV_to_D_start = Period(freq="D", year=2006, month=12, day=1)

        assert ival_A.asfreq("Q", "S") == ival_A_to_Q_start
        assert ival_A.asfreq("Q", "e") == ival_A_to_Q_end
        assert ival_A.asfreq("M", "s") == ival_A_to_M_start
        assert ival_A.asfreq("M", "E") == ival_A_to_M_end
        assert ival_A.asfreq("W", "S") == ival_A_to_W_start
        assert ival_A.asfreq("W", "E") == ival_A_to_W_end
        assert ival_A.asfreq("B", "S") == ival_A_to_B_start
        assert ival_A.asfreq("B", "E") == ival_A_to_B_end
        assert ival_A.asfreq("D", "S") == ival_A_to_D_start
        assert ival_A.asfreq("D", "E") == ival_A_to_D_end
        assert ival_A.asfreq("H", "S") == ival_A_to_H_start
        assert ival_A.asfreq("H", "E") == ival_A_to_H_end
        assert ival_A.asfreq("min", "S") == ival_A_to_T_start
        assert ival_A.asfreq("min", "E") == ival_A_to_T_end
        assert ival_A.asfreq("T", "S") == ival_A_to_T_start
        assert ival_A.asfreq("T", "E") == ival_A_to_T_end
        assert ival_A.asfreq("S", "S") == ival_A_to_S_start
        assert ival_A.asfreq("S", "E") == ival_A_to_S_end

        assert ival_AJAN.asfreq("D", "S") == ival_AJAN_to_D_start
        assert ival_AJAN.asfreq("D", "E") == ival_AJAN_to_D_end

        assert ival_AJUN.asfreq("D", "S") == ival_AJUN_to_D_start
        assert ival_AJUN.asfreq("D", "E") == ival_AJUN_to_D_end

        assert ival_ANOV.asfreq("D", "S") == ival_ANOV_to_D_start
        assert ival_ANOV.asfreq("D", "E") == ival_ANOV_to_D_end

        assert ival_A.asfreq("A") == ival_A

    def test_conv_quarterly(self):
        # frequency conversion tests: from Quarterly Frequency

        ival_Q = Period(freq="Q", year=2007, quarter=1)
        ival_Q_end_of_year = Period(freq="Q", year=2007, quarter=4)

        ival_QEJAN = Period(freq="Q-JAN", year=2007, quarter=1)
        ival_QEJUN = Period(freq="Q-JUN", year=2007, quarter=1)

        ival_Q_to_A = Period(freq="A", year=2007)
        ival_Q_to_M_start = Period(freq="M", year=2007, month=1)
        ival_Q_to_M_end = Period(freq="M", year=2007, month=3)
        ival_Q_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        ival_Q_to_W_end = Period(freq="W", year=2007, month=3, day=31)
        ival_Q_to_B_start = Period(freq="B", year=2007, month=1, day=1)
        ival_Q_to_B_end = Period(freq="B", year=2007, month=3, day=30)
        ival_Q_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_Q_to_D_end = Period(freq="D", year=2007, month=3, day=31)
        ival_Q_to_H_start = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_Q_to_H_end = Period(freq="H", year=2007, month=3, day=31, hour=23)
        ival_Q_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_Q_to_T_end = Period(
            freq="Min", year=2007, month=3, day=31, hour=23, minute=59
        )
        ival_Q_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_Q_to_S_end = Period(
            freq="S", year=2007, month=3, day=31, hour=23, minute=59, second=59
        )

        ival_QEJAN_to_D_start = Period(freq="D", year=2006, month=2, day=1)
        ival_QEJAN_to_D_end = Period(freq="D", year=2006, month=4, day=30)

        ival_QEJUN_to_D_start = Period(freq="D", year=2006, month=7, day=1)
        ival_QEJUN_to_D_end = Period(freq="D", year=2006, month=9, day=30)

        assert ival_Q.asfreq("A") == ival_Q_to_A
        assert ival_Q_end_of_year.asfreq("A") == ival_Q_to_A

        assert ival_Q.asfreq("M", "S") == ival_Q_to_M_start
        assert ival_Q.asfreq("M", "E") == ival_Q_to_M_end
        assert ival_Q.asfreq("W", "S") == ival_Q_to_W_start
        assert ival_Q.asfreq("W", "E") == ival_Q_to_W_end
        assert ival_Q.asfreq("B", "S") == ival_Q_to_B_start
        assert ival_Q.asfreq("B", "E") == ival_Q_to_B_end
        assert ival_Q.asfreq("D", "S") == ival_Q_to_D_start
        assert ival_Q.asfreq("D", "E") == ival_Q_to_D_end
        assert ival_Q.asfreq("H", "S") == ival_Q_to_H_start
        assert ival_Q.asfreq("H", "E") == ival_Q_to_H_end
        assert ival_Q.asfreq("Min", "S") == ival_Q_to_T_start
        assert ival_Q.asfreq("Min", "E") == ival_Q_to_T_end
        assert ival_Q.asfreq("S", "S") == ival_Q_to_S_start
        assert ival_Q.asfreq("S", "E") == ival_Q_to_S_end

        assert ival_QEJAN.asfreq("D", "S") == ival_QEJAN_to_D_start
        assert ival_QEJAN.asfreq("D", "E") == ival_QEJAN_to_D_end
        assert ival_QEJUN.asfreq("D", "S") == ival_QEJUN_to_D_start
        assert ival_QEJUN.asfreq("D", "E") == ival_QEJUN_to_D_end

        assert ival_Q.asfreq("Q") == ival_Q

    def test_conv_monthly(self):
        # frequency conversion tests: from Monthly Frequency

        ival_M = Period(freq="M", year=2007, month=1)
        ival_M_end_of_year = Period(freq="M", year=2007, month=12)
        ival_M_end_of_quarter = Period(freq="M", year=2007, month=3)
        ival_M_to_A = Period(freq="A", year=2007)
        ival_M_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_M_to_W_start = Period(freq="W", year=2007, month=1, day=1)
        ival_M_to_W_end = Period(freq="W", year=2007, month=1, day=31)
        ival_M_to_B_start = Period(freq="B", year=2007, month=1, day=1)
        ival_M_to_B_end = Period(freq="B", year=2007, month=1, day=31)
        ival_M_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_M_to_D_end = Period(freq="D", year=2007, month=1, day=31)
        ival_M_to_H_start = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_M_to_H_end = Period(freq="H", year=2007, month=1, day=31, hour=23)
        ival_M_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_M_to_T_end = Period(
            freq="Min", year=2007, month=1, day=31, hour=23, minute=59
        )
        ival_M_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_M_to_S_end = Period(
            freq="S", year=2007, month=1, day=31, hour=23, minute=59, second=59
        )

        assert ival_M.asfreq("A") == ival_M_to_A
        assert ival_M_end_of_year.asfreq("A") == ival_M_to_A
        assert ival_M.asfreq("Q") == ival_M_to_Q
        assert ival_M_end_of_quarter.asfreq("Q") == ival_M_to_Q

        assert ival_M.asfreq("W", "S") == ival_M_to_W_start
        assert ival_M.asfreq("W", "E") == ival_M_to_W_end
        assert ival_M.asfreq("B", "S") == ival_M_to_B_start
        assert ival_M.asfreq("B", "E") == ival_M_to_B_end
        assert ival_M.asfreq("D", "S") == ival_M_to_D_start
        assert ival_M.asfreq("D", "E") == ival_M_to_D_end
        assert ival_M.asfreq("H", "S") == ival_M_to_H_start
        assert ival_M.asfreq("H", "E") == ival_M_to_H_end
        assert ival_M.asfreq("Min", "S") == ival_M_to_T_start
        assert ival_M.asfreq("Min", "E") == ival_M_to_T_end
        assert ival_M.asfreq("S", "S") == ival_M_to_S_start
        assert ival_M.asfreq("S", "E") == ival_M_to_S_end

        assert ival_M.asfreq("M") == ival_M

    def test_conv_weekly(self):
        # frequency conversion tests: from Weekly Frequency
        ival_W = Period(freq="W", year=2007, month=1, day=1)

        ival_WSUN = Period(freq="W", year=2007, month=1, day=7)
        ival_WSAT = Period(freq="W-SAT", year=2007, month=1, day=6)
        ival_WFRI = Period(freq="W-FRI", year=2007, month=1, day=5)
        ival_WTHU = Period(freq="W-THU", year=2007, month=1, day=4)
        ival_WWED = Period(freq="W-WED", year=2007, month=1, day=3)
        ival_WTUE = Period(freq="W-TUE", year=2007, month=1, day=2)
        ival_WMON = Period(freq="W-MON", year=2007, month=1, day=1)

        ival_WSUN_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_WSUN_to_D_end = Period(freq="D", year=2007, month=1, day=7)
        ival_WSAT_to_D_start = Period(freq="D", year=2006, month=12, day=31)
        ival_WSAT_to_D_end = Period(freq="D", year=2007, month=1, day=6)
        ival_WFRI_to_D_start = Period(freq="D", year=2006, month=12, day=30)
        ival_WFRI_to_D_end = Period(freq="D", year=2007, month=1, day=5)
        ival_WTHU_to_D_start = Period(freq="D", year=2006, month=12, day=29)
        ival_WTHU_to_D_end = Period(freq="D", year=2007, month=1, day=4)
        ival_WWED_to_D_start = Period(freq="D", year=2006, month=12, day=28)
        ival_WWED_to_D_end = Period(freq="D", year=2007, month=1, day=3)
        ival_WTUE_to_D_start = Period(freq="D", year=2006, month=12, day=27)
        ival_WTUE_to_D_end = Period(freq="D", year=2007, month=1, day=2)
        ival_WMON_to_D_start = Period(freq="D", year=2006, month=12, day=26)
        ival_WMON_to_D_end = Period(freq="D", year=2007, month=1, day=1)

        ival_W_end_of_year = Period(freq="W", year=2007, month=12, day=31)
        ival_W_end_of_quarter = Period(freq="W", year=2007, month=3, day=31)
        ival_W_end_of_month = Period(freq="W", year=2007, month=1, day=31)
        ival_W_to_A = Period(freq="A", year=2007)
        ival_W_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_W_to_M = Period(freq="M", year=2007, month=1)

        if Period(freq="D", year=2007, month=12, day=31).weekday == 6:
            ival_W_to_A_end_of_year = Period(freq="A", year=2007)
        else:
            ival_W_to_A_end_of_year = Period(freq="A", year=2008)

        if Period(freq="D", year=2007, month=3, day=31).weekday == 6:
            ival_W_to_Q_end_of_quarter = Period(freq="Q", year=2007, quarter=1)
        else:
            ival_W_to_Q_end_of_quarter = Period(freq="Q", year=2007, quarter=2)

        if Period(freq="D", year=2007, month=1, day=31).weekday == 6:
            ival_W_to_M_end_of_month = Period(freq="M", year=2007, month=1)
        else:
            ival_W_to_M_end_of_month = Period(freq="M", year=2007, month=2)

        ival_W_to_B_start = Period(freq="B", year=2007, month=1, day=1)
        ival_W_to_B_end = Period(freq="B", year=2007, month=1, day=5)
        ival_W_to_D_start = Period(freq="D", year=2007, month=1, day=1)
        ival_W_to_D_end = Period(freq="D", year=2007, month=1, day=7)
        ival_W_to_H_start = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_W_to_H_end = Period(freq="H", year=2007, month=1, day=7, hour=23)
        ival_W_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_W_to_T_end = Period(
            freq="Min", year=2007, month=1, day=7, hour=23, minute=59
        )
        ival_W_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_W_to_S_end = Period(
            freq="S", year=2007, month=1, day=7, hour=23, minute=59, second=59
        )

        assert ival_W.asfreq("A") == ival_W_to_A
        assert ival_W_end_of_year.asfreq("A") == ival_W_to_A_end_of_year

        assert ival_W.asfreq("Q") == ival_W_to_Q
        assert ival_W_end_of_quarter.asfreq("Q") == ival_W_to_Q_end_of_quarter

        assert ival_W.asfreq("M") == ival_W_to_M
        assert ival_W_end_of_month.asfreq("M") == ival_W_to_M_end_of_month

        assert ival_W.asfreq("B", "S") == ival_W_to_B_start
        assert ival_W.asfreq("B", "E") == ival_W_to_B_end

        assert ival_W.asfreq("D", "S") == ival_W_to_D_start
        assert ival_W.asfreq("D", "E") == ival_W_to_D_end

        assert ival_WSUN.asfreq("D", "S") == ival_WSUN_to_D_start
        assert ival_WSUN.asfreq("D", "E") == ival_WSUN_to_D_end
        assert ival_WSAT.asfreq("D", "S") == ival_WSAT_to_D_start
        assert ival_WSAT.asfreq("D", "E") == ival_WSAT_to_D_end
        assert ival_WFRI.asfreq("D", "S") == ival_WFRI_to_D_start
        assert ival_WFRI.asfreq("D", "E") == ival_WFRI_to_D_end
        assert ival_WTHU.asfreq("D", "S") == ival_WTHU_to_D_start
        assert ival_WTHU.asfreq("D", "E") == ival_WTHU_to_D_end
        assert ival_WWED.asfreq("D", "S") == ival_WWED_to_D_start
        assert ival_WWED.asfreq("D", "E") == ival_WWED_to_D_end
        assert ival_WTUE.asfreq("D", "S") == ival_WTUE_to_D_start
        assert ival_WTUE.asfreq("D", "E") == ival_WTUE_to_D_end
        assert ival_WMON.asfreq("D", "S") == ival_WMON_to_D_start
        assert ival_WMON.asfreq("D", "E") == ival_WMON_to_D_end

        assert ival_W.asfreq("H", "S") == ival_W_to_H_start
        assert ival_W.asfreq("H", "E") == ival_W_to_H_end
        assert ival_W.asfreq("Min", "S") == ival_W_to_T_start
        assert ival_W.asfreq("Min", "E") == ival_W_to_T_end
        assert ival_W.asfreq("S", "S") == ival_W_to_S_start
        assert ival_W.asfreq("S", "E") == ival_W_to_S_end

        assert ival_W.asfreq("W") == ival_W

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            ival_W.asfreq("WK")

    def test_conv_weekly_legacy(self):
        # frequency conversion tests: from Weekly Frequency
        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK", year=2007, month=1, day=1)

        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-SAT", year=2007, month=1, day=6)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-FRI", year=2007, month=1, day=5)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-THU", year=2007, month=1, day=4)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-WED", year=2007, month=1, day=3)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-TUE", year=2007, month=1, day=2)
        with pytest.raises(ValueError, match=msg):
            Period(freq="WK-MON", year=2007, month=1, day=1)

    def test_conv_business(self):
        # frequency conversion tests: from Business Frequency"

        ival_B = Period(freq="B", year=2007, month=1, day=1)
        ival_B_end_of_year = Period(freq="B", year=2007, month=12, day=31)
        ival_B_end_of_quarter = Period(freq="B", year=2007, month=3, day=30)
        ival_B_end_of_month = Period(freq="B", year=2007, month=1, day=31)
        ival_B_end_of_week = Period(freq="B", year=2007, month=1, day=5)

        ival_B_to_A = Period(freq="A", year=2007)
        ival_B_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_B_to_M = Period(freq="M", year=2007, month=1)
        ival_B_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_B_to_D = Period(freq="D", year=2007, month=1, day=1)
        ival_B_to_H_start = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_B_to_H_end = Period(freq="H", year=2007, month=1, day=1, hour=23)
        ival_B_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_B_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_B_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_B_to_S_end = Period(
            freq="S", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )

        assert ival_B.asfreq("A") == ival_B_to_A
        assert ival_B_end_of_year.asfreq("A") == ival_B_to_A
        assert ival_B.asfreq("Q") == ival_B_to_Q
        assert ival_B_end_of_quarter.asfreq("Q") == ival_B_to_Q
        assert ival_B.asfreq("M") == ival_B_to_M
        assert ival_B_end_of_month.asfreq("M") == ival_B_to_M
        assert ival_B.asfreq("W") == ival_B_to_W
        assert ival_B_end_of_week.asfreq("W") == ival_B_to_W

        assert ival_B.asfreq("D") == ival_B_to_D

        assert ival_B.asfreq("H", "S") == ival_B_to_H_start
        assert ival_B.asfreq("H", "E") == ival_B_to_H_end
        assert ival_B.asfreq("Min", "S") == ival_B_to_T_start
        assert ival_B.asfreq("Min", "E") == ival_B_to_T_end
        assert ival_B.asfreq("S", "S") == ival_B_to_S_start
        assert ival_B.asfreq("S", "E") == ival_B_to_S_end

        assert ival_B.asfreq("B") == ival_B

    def test_conv_daily(self):
        # frequency conversion tests: from Business Frequency"

        ival_D = Period(freq="D", year=2007, month=1, day=1)
        ival_D_end_of_year = Period(freq="D", year=2007, month=12, day=31)
        ival_D_end_of_quarter = Period(freq="D", year=2007, month=3, day=31)
        ival_D_end_of_month = Period(freq="D", year=2007, month=1, day=31)
        ival_D_end_of_week = Period(freq="D", year=2007, month=1, day=7)

        ival_D_friday = Period(freq="D", year=2007, month=1, day=5)
        ival_D_saturday = Period(freq="D", year=2007, month=1, day=6)
        ival_D_sunday = Period(freq="D", year=2007, month=1, day=7)

        # TODO: unused?
        # ival_D_monday = Period(freq='D', year=2007, month=1, day=8)

        ival_B_friday = Period(freq="B", year=2007, month=1, day=5)
        ival_B_monday = Period(freq="B", year=2007, month=1, day=8)

        ival_D_to_A = Period(freq="A", year=2007)

        ival_Deoq_to_AJAN = Period(freq="A-JAN", year=2008)
        ival_Deoq_to_AJUN = Period(freq="A-JUN", year=2007)
        ival_Deoq_to_ADEC = Period(freq="A-DEC", year=2007)

        ival_D_to_QEJAN = Period(freq="Q-JAN", year=2007, quarter=4)
        ival_D_to_QEJUN = Period(freq="Q-JUN", year=2007, quarter=3)
        ival_D_to_QEDEC = Period(freq="Q-DEC", year=2007, quarter=1)

        ival_D_to_M = Period(freq="M", year=2007, month=1)
        ival_D_to_W = Period(freq="W", year=2007, month=1, day=7)

        ival_D_to_H_start = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_D_to_H_end = Period(freq="H", year=2007, month=1, day=1, hour=23)
        ival_D_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_D_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_D_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_D_to_S_end = Period(
            freq="S", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )

        assert ival_D.asfreq("A") == ival_D_to_A

        assert ival_D_end_of_quarter.asfreq("A-JAN") == ival_Deoq_to_AJAN
        assert ival_D_end_of_quarter.asfreq("A-JUN") == ival_Deoq_to_AJUN
        assert ival_D_end_of_quarter.asfreq("A-DEC") == ival_Deoq_to_ADEC

        assert ival_D_end_of_year.asfreq("A") == ival_D_to_A
        assert ival_D_end_of_quarter.asfreq("Q") == ival_D_to_QEDEC
        assert ival_D.asfreq("Q-JAN") == ival_D_to_QEJAN
        assert ival_D.asfreq("Q-JUN") == ival_D_to_QEJUN
        assert ival_D.asfreq("Q-DEC") == ival_D_to_QEDEC
        assert ival_D.asfreq("M") == ival_D_to_M
        assert ival_D_end_of_month.asfreq("M") == ival_D_to_M
        assert ival_D.asfreq("W") == ival_D_to_W
        assert ival_D_end_of_week.asfreq("W") == ival_D_to_W

        assert ival_D_friday.asfreq("B") == ival_B_friday
        assert ival_D_saturday.asfreq("B", "S") == ival_B_friday
        assert ival_D_saturday.asfreq("B", "E") == ival_B_monday
        assert ival_D_sunday.asfreq("B", "S") == ival_B_friday
        assert ival_D_sunday.asfreq("B", "E") == ival_B_monday

        assert ival_D.asfreq("H", "S") == ival_D_to_H_start
        assert ival_D.asfreq("H", "E") == ival_D_to_H_end
        assert ival_D.asfreq("Min", "S") == ival_D_to_T_start
        assert ival_D.asfreq("Min", "E") == ival_D_to_T_end
        assert ival_D.asfreq("S", "S") == ival_D_to_S_start
        assert ival_D.asfreq("S", "E") == ival_D_to_S_end

        assert ival_D.asfreq("D") == ival_D

    def test_conv_hourly(self):
        # frequency conversion tests: from Hourly Frequency"

        ival_H = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_H_end_of_year = Period(freq="H", year=2007, month=12, day=31, hour=23)
        ival_H_end_of_quarter = Period(freq="H", year=2007, month=3, day=31, hour=23)
        ival_H_end_of_month = Period(freq="H", year=2007, month=1, day=31, hour=23)
        ival_H_end_of_week = Period(freq="H", year=2007, month=1, day=7, hour=23)
        ival_H_end_of_day = Period(freq="H", year=2007, month=1, day=1, hour=23)
        ival_H_end_of_bus = Period(freq="H", year=2007, month=1, day=1, hour=23)

        ival_H_to_A = Period(freq="A", year=2007)
        ival_H_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_H_to_M = Period(freq="M", year=2007, month=1)
        ival_H_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_H_to_D = Period(freq="D", year=2007, month=1, day=1)
        ival_H_to_B = Period(freq="B", year=2007, month=1, day=1)

        ival_H_to_T_start = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=0
        )
        ival_H_to_T_end = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=59
        )
        ival_H_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_H_to_S_end = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=59, second=59
        )

        assert ival_H.asfreq("A") == ival_H_to_A
        assert ival_H_end_of_year.asfreq("A") == ival_H_to_A
        assert ival_H.asfreq("Q") == ival_H_to_Q
        assert ival_H_end_of_quarter.asfreq("Q") == ival_H_to_Q
        assert ival_H.asfreq("M") == ival_H_to_M
        assert ival_H_end_of_month.asfreq("M") == ival_H_to_M
        assert ival_H.asfreq("W") == ival_H_to_W
        assert ival_H_end_of_week.asfreq("W") == ival_H_to_W
        assert ival_H.asfreq("D") == ival_H_to_D
        assert ival_H_end_of_day.asfreq("D") == ival_H_to_D
        assert ival_H.asfreq("B") == ival_H_to_B
        assert ival_H_end_of_bus.asfreq("B") == ival_H_to_B

        assert ival_H.asfreq("Min", "S") == ival_H_to_T_start
        assert ival_H.asfreq("Min", "E") == ival_H_to_T_end
        assert ival_H.asfreq("S", "S") == ival_H_to_S_start
        assert ival_H.asfreq("S", "E") == ival_H_to_S_end

        assert ival_H.asfreq("H") == ival_H

    def test_conv_minutely(self):
        # frequency conversion tests: from Minutely Frequency"

        ival_T = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)
        ival_T_end_of_year = Period(
            freq="Min", year=2007, month=12, day=31, hour=23, minute=59
        )
        ival_T_end_of_quarter = Period(
            freq="Min", year=2007, month=3, day=31, hour=23, minute=59
        )
        ival_T_end_of_month = Period(
            freq="Min", year=2007, month=1, day=31, hour=23, minute=59
        )
        ival_T_end_of_week = Period(
            freq="Min", year=2007, month=1, day=7, hour=23, minute=59
        )
        ival_T_end_of_day = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_T_end_of_bus = Period(
            freq="Min", year=2007, month=1, day=1, hour=23, minute=59
        )
        ival_T_end_of_hour = Period(
            freq="Min", year=2007, month=1, day=1, hour=0, minute=59
        )

        ival_T_to_A = Period(freq="A", year=2007)
        ival_T_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_T_to_M = Period(freq="M", year=2007, month=1)
        ival_T_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_T_to_D = Period(freq="D", year=2007, month=1, day=1)
        ival_T_to_B = Period(freq="B", year=2007, month=1, day=1)
        ival_T_to_H = Period(freq="H", year=2007, month=1, day=1, hour=0)

        ival_T_to_S_start = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0
        )
        ival_T_to_S_end = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=59
        )

        assert ival_T.asfreq("A") == ival_T_to_A
        assert ival_T_end_of_year.asfreq("A") == ival_T_to_A
        assert ival_T.asfreq("Q") == ival_T_to_Q
        assert ival_T_end_of_quarter.asfreq("Q") == ival_T_to_Q
        assert ival_T.asfreq("M") == ival_T_to_M
        assert ival_T_end_of_month.asfreq("M") == ival_T_to_M
        assert ival_T.asfreq("W") == ival_T_to_W
        assert ival_T_end_of_week.asfreq("W") == ival_T_to_W
        assert ival_T.asfreq("D") == ival_T_to_D
        assert ival_T_end_of_day.asfreq("D") == ival_T_to_D
        assert ival_T.asfreq("B") == ival_T_to_B
        assert ival_T_end_of_bus.asfreq("B") == ival_T_to_B
        assert ival_T.asfreq("H") == ival_T_to_H
        assert ival_T_end_of_hour.asfreq("H") == ival_T_to_H

        assert ival_T.asfreq("S", "S") == ival_T_to_S_start
        assert ival_T.asfreq("S", "E") == ival_T_to_S_end

        assert ival_T.asfreq("Min") == ival_T

    def test_conv_secondly(self):
        # frequency conversion tests: from Secondly Frequency"

        ival_S = Period(freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=0)
        ival_S_end_of_year = Period(
            freq="S", year=2007, month=12, day=31, hour=23, minute=59, second=59
        )
        ival_S_end_of_quarter = Period(
            freq="S", year=2007, month=3, day=31, hour=23, minute=59, second=59
        )
        ival_S_end_of_month = Period(
            freq="S", year=2007, month=1, day=31, hour=23, minute=59, second=59
        )
        ival_S_end_of_week = Period(
            freq="S", year=2007, month=1, day=7, hour=23, minute=59, second=59
        )
        ival_S_end_of_day = Period(
            freq="S", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )
        ival_S_end_of_bus = Period(
            freq="S", year=2007, month=1, day=1, hour=23, minute=59, second=59
        )
        ival_S_end_of_hour = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=59, second=59
        )
        ival_S_end_of_minute = Period(
            freq="S", year=2007, month=1, day=1, hour=0, minute=0, second=59
        )

        ival_S_to_A = Period(freq="A", year=2007)
        ival_S_to_Q = Period(freq="Q", year=2007, quarter=1)
        ival_S_to_M = Period(freq="M", year=2007, month=1)
        ival_S_to_W = Period(freq="W", year=2007, month=1, day=7)
        ival_S_to_D = Period(freq="D", year=2007, month=1, day=1)
        ival_S_to_B = Period(freq="B", year=2007, month=1, day=1)
        ival_S_to_H = Period(freq="H", year=2007, month=1, day=1, hour=0)
        ival_S_to_T = Period(freq="Min", year=2007, month=1, day=1, hour=0, minute=0)

        assert ival_S.asfreq("A") == ival_S_to_A
        assert ival_S_end_of_year.asfreq("A") == ival_S_to_A
        assert ival_S.asfreq("Q") == ival_S_to_Q
        assert ival_S_end_of_quarter.asfreq("Q") == ival_S_to_Q
        assert ival_S.asfreq("M") == ival_S_to_M
        assert ival_S_end_of_month.asfreq("M") == ival_S_to_M
        assert ival_S.asfreq("W") == ival_S_to_W
        assert ival_S_end_of_week.asfreq("W") == ival_S_to_W
        assert ival_S.asfreq("D") == ival_S_to_D
        assert ival_S_end_of_day.asfreq("D") == ival_S_to_D
        assert ival_S.asfreq("B") == ival_S_to_B
        assert ival_S_end_of_bus.asfreq("B") == ival_S_to_B
        assert ival_S.asfreq("H") == ival_S_to_H
        assert ival_S_end_of_hour.asfreq("H") == ival_S_to_H
        assert ival_S.asfreq("Min") == ival_S_to_T
        assert ival_S_end_of_minute.asfreq("Min") == ival_S_to_T

        assert ival_S.asfreq("S") == ival_S

    def test_conv_microsecond(self):
        # GH#31475 Avoid floating point errors dropping the start_time to
        #  before the beginning of the Period
        per = Period("2020-01-30 15:57:27.576166", freq="U")
        assert per.ordinal == 1580399847576166

        start = per.start_time
        expected = Timestamp("2020-01-30 15:57:27.576166")
        assert start == expected
        assert start.value == per.ordinal * 1000

        per2 = Period("2300-01-01", "us")
        msg = "2300-01-01"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per2.start_time
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            per2.end_time

    def test_asfreq_mult(self):
        # normal freq to mult freq
        p = Period(freq="A", year=2007)
        # ordinal will not change
        for freq in ["3A", offsets.YearEnd(3)]:
            result = p.asfreq(freq)
            expected = Period("2007", freq="3A")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        # ordinal will not change
        for freq in ["3A", offsets.YearEnd(3)]:
            result = p.asfreq(freq, how="S")
            expected = Period("2007", freq="3A")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        # mult freq to normal freq
        p = Period(freq="3A", year=2007)
        # ordinal will change because how=E is the default
        for freq in ["A", offsets.YearEnd()]:
            result = p.asfreq(freq)
            expected = Period("2009", freq="A")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        # ordinal will not change
        for freq in ["A", offsets.YearEnd()]:
            result = p.asfreq(freq, how="S")
            expected = Period("2007", freq="A")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        p = Period(freq="A", year=2007)
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq)
            expected = Period("2007-12", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq, how="S")
            expected = Period("2007-01", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        p = Period(freq="3A", year=2007)
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq)
            expected = Period("2009-12", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq
        for freq in ["2M", offsets.MonthEnd(2)]:
            result = p.asfreq(freq, how="S")
            expected = Period("2007-01", freq="2M")

            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

    def test_asfreq_combined(self):
        # normal freq to combined freq
        p = Period("2007", freq="H")

        # ordinal will not change
        expected = Period("2007", freq="25H")
        for freq, how in zip(["1D1H", "1H1D"], ["E", "S"]):
            result = p.asfreq(freq, how=how)
            assert result == expected
            assert result.ordinal == expected.ordinal
            assert result.freq == expected.freq

        # combined freq to normal freq
        p1 = Period(freq="1D1H", year=2007)
        p2 = Period(freq="1H1D", year=2007)

        # ordinal will change because how=E is the default
        result1 = p1.asfreq("H")
        result2 = p2.asfreq("H")
        expected = Period("2007-01-02", freq="H")
        assert result1 == expected
        assert result1.ordinal == expected.ordinal
        assert result1.freq == expected.freq
        assert result2 == expected
        assert result2.ordinal == expected.ordinal
        assert result2.freq == expected.freq

        # ordinal will not change
        result1 = p1.asfreq("H", how="S")
        result2 = p2.asfreq("H", how="S")
        expected = Period("2007-01-01", freq="H")
        assert result1 == expected
        assert result1.ordinal == expected.ordinal
        assert result1.freq == expected.freq
        assert result2 == expected
        assert result2.ordinal == expected.ordinal
        assert result2.freq == expected.freq

    def test_asfreq_MS(self):
        initial = Period("2013")

        assert initial.asfreq(freq="M", how="S") == Period("2013-01", "M")

        msg = INVALID_FREQ_ERR_MSG
        with pytest.raises(ValueError, match=msg):
            initial.asfreq(freq="MS", how="S")

        with pytest.raises(ValueError, match=msg):
            Period("2013-01", "MS")

        assert _period_code_map.get("MS") is None
