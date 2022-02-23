import numpy as np

from pandas import Series
import pandas._testing as tm


class TestInferObjects:
    def test_infer_objects_series(self):
        # GH#11221
        actual = Series(np.array([1, 2, 3], dtype="O")).infer_objects()
        expected = Series([1, 2, 3])
        tm.assert_series_equal(actual, expected)

        actual = Series(np.array([1, 2, 3, None], dtype="O")).infer_objects()
        expected = Series([1.0, 2.0, 3.0, np.nan])
        tm.assert_series_equal(actual, expected)

        # only soft conversions, unconvertable pass thru unchanged
        actual = Series(np.array([1, 2, 3, None, "a"], dtype="O")).infer_objects()
        expected = Series([1, 2, 3, None, "a"])

        assert actual.dtype == "object"
        tm.assert_series_equal(actual, expected)
