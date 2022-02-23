import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestConvert:
    def test_convert_objects(self, float_string_frame):

        oops = float_string_frame.T.T
        converted = oops._convert(datetime=True)
        tm.assert_frame_equal(converted, float_string_frame)
        assert converted["A"].dtype == np.float64

        # force numeric conversion
        float_string_frame["H"] = "1."
        float_string_frame["I"] = "1"

        # add in some items that will be nan
        length = len(float_string_frame)
        float_string_frame["J"] = "1."
        float_string_frame["K"] = "1"
        float_string_frame.loc[float_string_frame.index[0:5], ["J", "K"]] = "garbled"
        converted = float_string_frame._convert(datetime=True, numeric=True)
        assert converted["H"].dtype == "float64"
        assert converted["I"].dtype == "int64"
        assert converted["J"].dtype == "float64"
        assert converted["K"].dtype == "float64"
        assert len(converted["J"].dropna()) == length - 5
        assert len(converted["K"].dropna()) == length - 5

        # via astype
        converted = float_string_frame.copy()
        converted["H"] = converted["H"].astype("float64")
        converted["I"] = converted["I"].astype("int64")
        assert converted["H"].dtype == "float64"
        assert converted["I"].dtype == "int64"

        # via astype, but errors
        converted = float_string_frame.copy()
        with pytest.raises(ValueError, match="invalid literal"):
            converted["H"].astype("int32")

        # mixed in a single column
        df = DataFrame({"s": Series([1, "na", 3, 4])})
        result = df._convert(datetime=True, numeric=True)
        expected = DataFrame({"s": Series([1, np.nan, 3, 4])})
        tm.assert_frame_equal(result, expected)

    def test_convert_objects_no_conversion(self):
        mixed1 = DataFrame({"a": [1, 2, 3], "b": [4.0, 5, 6], "c": ["x", "y", "z"]})
        mixed2 = mixed1._convert(datetime=True)
        tm.assert_frame_equal(mixed1, mixed2)
