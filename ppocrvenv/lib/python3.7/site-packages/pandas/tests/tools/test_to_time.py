from datetime import time

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import Series
import pandas._testing as tm
from pandas.core.tools.datetimes import to_time as to_time_alias
from pandas.core.tools.times import to_time


class TestToTime:
    @td.skip_if_has_locale
    def test_parsers_time(self):
        # GH#11818
        strings = [
            "14:15",
            "1415",
            "2:15pm",
            "0215pm",
            "14:15:00",
            "141500",
            "2:15:00pm",
            "021500pm",
            time(14, 15),
        ]
        expected = time(14, 15)

        for time_string in strings:
            assert to_time(time_string) == expected

        new_string = "14.15"
        msg = r"Cannot convert arg \['14\.15'\] to a time"
        with pytest.raises(ValueError, match=msg):
            to_time(new_string)
        assert to_time(new_string, format="%H.%M") == expected

        arg = ["14:15", "20:20"]
        expected_arr = [time(14, 15), time(20, 20)]
        assert to_time(arg) == expected_arr
        assert to_time(arg, format="%H:%M") == expected_arr
        assert to_time(arg, infer_time_format=True) == expected_arr
        assert to_time(arg, format="%I:%M%p", errors="coerce") == [None, None]

        res = to_time(arg, format="%I:%M%p", errors="ignore")
        tm.assert_numpy_array_equal(res, np.array(arg, dtype=np.object_))

        msg = "Cannot convert.+to a time with given format"
        with pytest.raises(ValueError, match=msg):
            to_time(arg, format="%I:%M%p", errors="raise")

        tm.assert_series_equal(
            to_time(Series(arg, name="test")), Series(expected_arr, name="test")
        )

        res = to_time(np.array(arg))
        assert isinstance(res, list)
        assert res == expected_arr


def test_to_time_alias():
    expected = time(14, 15)

    with tm.assert_produces_warning(FutureWarning):
        result = to_time_alias(expected)

    assert result == expected
