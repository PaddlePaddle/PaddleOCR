from pandas import (
    DataFrame,
    date_range,
)
import pandas._testing as tm


class TestToFrame:
    def test_to_frame_datetime_tz(self):
        # GH#25809
        idx = date_range(start="2019-01-01", end="2019-01-30", freq="D", tz="UTC")
        result = idx.to_frame()
        expected = DataFrame(idx, index=idx)
        tm.assert_frame_equal(result, expected)
