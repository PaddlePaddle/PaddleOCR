from pandas import (
    Index,
    NaT,
    Period,
    PeriodIndex,
)
import pandas._testing as tm


class TestFillNA:
    def test_fillna_period(self):
        # GH#11343
        idx = PeriodIndex(["2011-01-01 09:00", NaT, "2011-01-01 11:00"], freq="H")

        exp = PeriodIndex(
            ["2011-01-01 09:00", "2011-01-01 10:00", "2011-01-01 11:00"], freq="H"
        )
        result = idx.fillna(Period("2011-01-01 10:00", freq="H"))
        tm.assert_index_equal(result, exp)

        exp = Index(
            [
                Period("2011-01-01 09:00", freq="H"),
                "x",
                Period("2011-01-01 11:00", freq="H"),
            ],
            dtype=object,
        )
        result = idx.fillna("x")
        tm.assert_index_equal(result, exp)

        exp = Index(
            [
                Period("2011-01-01 09:00", freq="H"),
                Period("2011-01-01", freq="D"),
                Period("2011-01-01 11:00", freq="H"),
            ],
            dtype=object,
        )
        result = idx.fillna(Period("2011-01-01", freq="D"))
        tm.assert_index_equal(result, exp)
