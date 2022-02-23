import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
)
import pandas._testing as tm


class TestLookup:
    def test_lookup_float(self, float_frame):
        df = float_frame
        rows = list(df.index) * len(df.columns)
        cols = list(df.columns) * len(df.index)
        with tm.assert_produces_warning(FutureWarning):
            result = df.lookup(rows, cols)

        expected = np.array([df.loc[r, c] for r, c in zip(rows, cols)])
        tm.assert_numpy_array_equal(result, expected)

    def test_lookup_mixed(self, float_string_frame):
        df = float_string_frame
        rows = list(df.index) * len(df.columns)
        cols = list(df.columns) * len(df.index)
        with tm.assert_produces_warning(FutureWarning):
            result = df.lookup(rows, cols)

        expected = np.array(
            [df.loc[r, c] for r, c in zip(rows, cols)], dtype=np.object_
        )
        tm.assert_almost_equal(result, expected)

    def test_lookup_bool(self):
        df = DataFrame(
            {
                "label": ["a", "b", "a", "c"],
                "mask_a": [True, True, False, True],
                "mask_b": [True, False, False, False],
                "mask_c": [False, True, False, True],
            }
        )
        with tm.assert_produces_warning(FutureWarning):
            df["mask"] = df.lookup(df.index, "mask_" + df["label"])

        exp_mask = np.array(
            [df.loc[r, c] for r, c in zip(df.index, "mask_" + df["label"])]
        )

        tm.assert_series_equal(df["mask"], Series(exp_mask, name="mask"))
        assert df["mask"].dtype == np.bool_

    def test_lookup_raises(self, float_frame):
        with pytest.raises(KeyError, match="'One or more row labels was not found'"):
            with tm.assert_produces_warning(FutureWarning):
                float_frame.lookup(["xyz"], ["A"])

        with pytest.raises(KeyError, match="'One or more column labels was not found'"):
            with tm.assert_produces_warning(FutureWarning):
                float_frame.lookup([float_frame.index[0]], ["xyz"])

        with pytest.raises(ValueError, match="same size"):
            with tm.assert_produces_warning(FutureWarning):
                float_frame.lookup(["a", "b", "c"], ["a"])

    def test_lookup_requires_unique_axes(self):
        # GH#33041 raise with a helpful error message
        df = DataFrame(np.random.randn(6).reshape(3, 2), columns=["A", "A"])

        rows = [0, 1]
        cols = ["A", "A"]

        # homogeneous-dtype case
        with pytest.raises(ValueError, match="requires unique index and columns"):
            with tm.assert_produces_warning(FutureWarning):
                df.lookup(rows, cols)
        with pytest.raises(ValueError, match="requires unique index and columns"):
            with tm.assert_produces_warning(FutureWarning):
                df.T.lookup(cols, rows)

        # heterogeneous dtype
        df["B"] = 0
        with pytest.raises(ValueError, match="requires unique index and columns"):
            with tm.assert_produces_warning(FutureWarning):
                df.lookup(rows, cols)


def test_lookup_deprecated():
    # GH#18262
    df = DataFrame(
        {"col": ["A", "A", "B", "B"], "A": [80, 23, np.nan, 22], "B": [80, 55, 76, 67]}
    )
    with tm.assert_produces_warning(FutureWarning):
        df.lookup(df.index, df["col"])
