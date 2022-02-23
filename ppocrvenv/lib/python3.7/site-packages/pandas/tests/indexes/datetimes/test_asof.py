from pandas import (
    Index,
    Timestamp,
    date_range,
)


class TestAsOf:
    def test_asof_partial(self):
        index = date_range("2010-01-01", periods=2, freq="m")
        expected = Timestamp("2010-02-28")
        result = index.asof("2010-02")
        assert result == expected
        assert not isinstance(result, Index)
