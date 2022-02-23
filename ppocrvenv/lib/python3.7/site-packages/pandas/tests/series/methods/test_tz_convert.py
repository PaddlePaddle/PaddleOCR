import numpy as np

from pandas import (
    DatetimeIndex,
    Series,
)
import pandas._testing as tm


class TestTZConvert:
    def test_series_tz_convert_to_utc(self):
        base = DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"], tz="UTC")
        idx1 = base.tz_convert("Asia/Tokyo")[:2]
        idx2 = base.tz_convert("US/Eastern")[1:]

        res = Series([1, 2], index=idx1) + Series([1, 1], index=idx2)
        tm.assert_series_equal(res, Series([np.nan, 3, np.nan], index=base))
