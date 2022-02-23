import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameInterpolate:
    def test_interp_basic(self):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )
        expected = DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [1.0, 4.0, 9.0, 9.0],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )
        result = df.interpolate()
        tm.assert_frame_equal(result, expected)

        result = df.set_index("C").interpolate()
        expected = df.set_index("C")
        expected.loc[3, "A"] = 3
        expected.loc[5, "B"] = 9
        tm.assert_frame_equal(result, expected)

    def test_interp_empty(self):
        # https://github.com/pandas-dev/pandas/issues/35598
        df = DataFrame()
        result = df.interpolate()
        assert result is not df
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_interp_bad_method(self):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )
        msg = (
            r"method must be one of \['linear', 'time', 'index', 'values', "
            r"'nearest', 'zero', 'slinear', 'quadratic', 'cubic', "
            r"'barycentric', 'krogh', 'spline', 'polynomial', "
            r"'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', "
            r"'cubicspline'\]. Got 'not_a_method' instead."
        )
        with pytest.raises(ValueError, match=msg):
            df.interpolate(method="not_a_method")

    def test_interp_combo(self):
        df = DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0],
                "B": [1, 4, 9, np.nan],
                "C": [1, 2, 3, 5],
                "D": list("abcd"),
            }
        )

        result = df["A"].interpolate()
        expected = Series([1.0, 2.0, 3.0, 4.0], name="A")
        tm.assert_series_equal(result, expected)

        result = df["A"].interpolate(downcast="infer")
        expected = Series([1, 2, 3, 4], name="A")
        tm.assert_series_equal(result, expected)

    def test_interp_nan_idx(self):
        df = DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
        df = df.set_index("A")
        msg = (
            "Interpolation with NaNs in the index has not been implemented. "
            "Try filling those NaNs before interpolating."
        )
        with pytest.raises(NotImplementedError, match=msg):
            df.interpolate(method="values")

    @td.skip_if_no_scipy
    def test_interp_various(self):
        df = DataFrame(
            {"A": [1, 2, np.nan, 4, 5, np.nan, 7], "C": [1, 2, 3, 5, 8, 13, 21]}
        )
        df = df.set_index("C")
        expected = df.copy()
        result = df.interpolate(method="polynomial", order=1)

        expected.A.loc[3] = 2.66666667
        expected.A.loc[13] = 5.76923076
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="cubic")
        # GH #15662.
        expected.A.loc[3] = 2.81547781
        expected.A.loc[13] = 5.52964175
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="nearest")
        expected.A.loc[3] = 2
        expected.A.loc[13] = 5
        tm.assert_frame_equal(result, expected, check_dtype=False)

        result = df.interpolate(method="quadratic")
        expected.A.loc[3] = 2.82150771
        expected.A.loc[13] = 6.12648668
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="slinear")
        expected.A.loc[3] = 2.66666667
        expected.A.loc[13] = 5.76923077
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="zero")
        expected.A.loc[3] = 2.0
        expected.A.loc[13] = 5
        tm.assert_frame_equal(result, expected, check_dtype=False)

    @td.skip_if_no_scipy
    def test_interp_alt_scipy(self):
        df = DataFrame(
            {"A": [1, 2, np.nan, 4, 5, np.nan, 7], "C": [1, 2, 3, 5, 8, 13, 21]}
        )
        result = df.interpolate(method="barycentric")
        expected = df.copy()
        expected.loc[2, "A"] = 3
        expected.loc[5, "A"] = 6
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(method="barycentric", downcast="infer")
        tm.assert_frame_equal(result, expected.astype(np.int64))

        result = df.interpolate(method="krogh")
        expectedk = df.copy()
        expectedk["A"] = expected["A"]
        tm.assert_frame_equal(result, expectedk)

        result = df.interpolate(method="pchip")
        expected.loc[2, "A"] = 3
        expected.loc[5, "A"] = 6.0

        tm.assert_frame_equal(result, expected)

    def test_interp_rowwise(self):
        df = DataFrame(
            {
                0: [1, 2, np.nan, 4],
                1: [2, 3, 4, np.nan],
                2: [np.nan, 4, 5, 6],
                3: [4, np.nan, 6, 7],
                4: [1, 2, 3, 4],
            }
        )
        result = df.interpolate(axis=1)
        expected = df.copy()
        expected.loc[3, 1] = 5
        expected.loc[0, 2] = 3
        expected.loc[1, 3] = 3
        expected[4] = expected[4].astype(np.float64)
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(axis=1, method="values")
        tm.assert_frame_equal(result, expected)

        result = df.interpolate(axis=0)
        expected = df.interpolate()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "axis_name, axis_number",
        [
            pytest.param("rows", 0, id="rows_0"),
            pytest.param("index", 0, id="index_0"),
            pytest.param("columns", 1, id="columns_1"),
        ],
    )
    def test_interp_axis_names(self, axis_name, axis_number):
        # GH 29132: test axis names
        data = {0: [0, np.nan, 6], 1: [1, np.nan, 7], 2: [2, 5, 8]}

        df = DataFrame(data, dtype=np.float64)
        result = df.interpolate(axis=axis_name, method="linear")
        expected = df.interpolate(axis=axis_number, method="linear")
        tm.assert_frame_equal(result, expected)

    def test_rowwise_alt(self):
        df = DataFrame(
            {
                0: [0, 0.5, 1.0, np.nan, 4, 8, np.nan, np.nan, 64],
                1: [1, 2, 3, 4, 3, 2, 1, 0, -1],
            }
        )
        df.interpolate(axis=0)
        # TODO: assert something?

    @pytest.mark.parametrize(
        "check_scipy", [False, pytest.param(True, marks=td.skip_if_no_scipy)]
    )
    def test_interp_leading_nans(self, check_scipy):
        df = DataFrame(
            {"A": [np.nan, np.nan, 0.5, 0.25, 0], "B": [np.nan, -3, -3.5, np.nan, -4]}
        )
        result = df.interpolate()
        expected = df.copy()
        expected["B"].loc[3] = -3.75
        tm.assert_frame_equal(result, expected)

        if check_scipy:
            result = df.interpolate(method="polynomial", order=1)
            tm.assert_frame_equal(result, expected)

    def test_interp_raise_on_only_mixed(self, axis):
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": ["a", "b", "c", "d"],
                "C": [np.nan, 2, 5, 7],
                "D": [np.nan, np.nan, 9, 9],
                "E": [1, 2, 3, 4],
            }
        )
        msg = (
            "Cannot interpolate with all object-dtype columns "
            "in the DataFrame. Try setting at least one "
            "column to a numeric dtype."
        )
        with pytest.raises(TypeError, match=msg):
            df.astype("object").interpolate(axis=axis)

    def test_interp_raise_on_all_object_dtype(self):
        # GH 22985
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, dtype="object")
        msg = (
            "Cannot interpolate with all object-dtype columns "
            "in the DataFrame. Try setting at least one "
            "column to a numeric dtype."
        )
        with pytest.raises(TypeError, match=msg):
            df.interpolate()

    def test_interp_inplace(self):
        df = DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        expected = DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = df.copy()
        return_value = result["a"].interpolate(inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result["a"].interpolate(inplace=True, downcast="infer")
        assert return_value is None
        tm.assert_frame_equal(result, expected.astype("int64"))

    def test_interp_inplace_row(self):
        # GH 10395
        result = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [np.nan, 2.0, 3.0, 4.0], "c": [3, 2, 2, 2]}
        )
        expected = result.interpolate(method="linear", axis=1, inplace=False)
        return_value = result.interpolate(method="linear", axis=1, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_interp_ignore_all_good(self):
        # GH
        df = DataFrame(
            {
                "A": [1, 2, np.nan, 4],
                "B": [1, 2, 3, 4],
                "C": [1.0, 2.0, np.nan, 4.0],
                "D": [1.0, 2.0, 3.0, 4.0],
            }
        )
        expected = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="float64"),
                "B": np.array([1, 2, 3, 4], dtype="int64"),
                "C": np.array([1.0, 2.0, 3, 4.0], dtype="float64"),
                "D": np.array([1.0, 2.0, 3.0, 4.0], dtype="float64"),
            }
        )

        result = df.interpolate(downcast=None)
        tm.assert_frame_equal(result, expected)

        # all good
        result = df[["B", "D"]].interpolate(downcast=None)
        tm.assert_frame_equal(result, df[["B", "D"]])

    def test_interp_time_inplace_axis(self, axis):
        # GH 9687
        periods = 5
        idx = date_range(start="2014-01-01", periods=periods)
        data = np.random.rand(periods, periods)
        data[data < 0.5] = np.nan
        expected = DataFrame(index=idx, columns=idx, data=data)

        result = expected.interpolate(axis=0, method="time")
        return_value = expected.interpolate(axis=0, method="time", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("axis_name, axis_number", [("index", 0), ("columns", 1)])
    def test_interp_string_axis(self, axis_name, axis_number):
        # https://github.com/pandas-dev/pandas/issues/25190
        x = np.linspace(0, 100, 1000)
        y = np.sin(x)
        df = DataFrame(
            data=np.tile(y, (10, 1)), index=np.arange(10), columns=x
        ).reindex(columns=x * 1.005)
        result = df.interpolate(method="linear", axis=axis_name)
        expected = df.interpolate(method="linear", axis=axis_number)
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) support axis=1
    @pytest.mark.parametrize("method", ["ffill", "bfill", "pad"])
    def test_interp_fillna_methods(self, axis, method):
        # GH 12918
        df = DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, np.nan, 5.0],
                "B": [2.0, 4.0, 6.0, np.nan, 8.0, 10.0],
                "C": [3.0, 6.0, 9.0, np.nan, np.nan, 30.0],
            }
        )
        expected = df.fillna(axis=axis, method=method)
        result = df.interpolate(method=method, axis=axis)
        tm.assert_frame_equal(result, expected)

    def test_interpolate_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            r"In a future version of pandas all arguments of DataFrame.interpolate "
            r"except for the argument 'method' will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.interpolate("pad", 0)
        expected = DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)
