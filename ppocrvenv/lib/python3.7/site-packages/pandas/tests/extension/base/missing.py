import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_sparse
from pandas.tests.extension.base.base import BaseExtensionTests


class BaseMissingTests(BaseExtensionTests):
    def test_isna(self, data_missing):
        expected = np.array([True, False])

        result = pd.isna(data_missing)
        tm.assert_numpy_array_equal(result, expected)

        result = pd.Series(data_missing).isna()
        expected = pd.Series(expected)
        self.assert_series_equal(result, expected)

        # GH 21189
        result = pd.Series(data_missing).drop([0, 1]).isna()
        expected = pd.Series([], dtype=bool)
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_func", ["isna", "notna"])
    def test_isna_returns_copy(self, data_missing, na_func):
        result = pd.Series(data_missing)
        expected = result.copy()
        mask = getattr(result, na_func)()
        if is_sparse(mask):
            mask = np.array(mask)

        mask[:] = True
        self.assert_series_equal(result, expected)

    def test_dropna_array(self, data_missing):
        result = data_missing.dropna()
        expected = data_missing[[1]]
        self.assert_extension_array_equal(result, expected)

    def test_dropna_series(self, data_missing):
        ser = pd.Series(data_missing)
        result = ser.dropna()
        expected = ser.iloc[[1]]
        self.assert_series_equal(result, expected)

    def test_dropna_frame(self, data_missing):
        df = pd.DataFrame({"A": data_missing})

        # defaults
        result = df.dropna()
        expected = df.iloc[[1]]
        self.assert_frame_equal(result, expected)

        # axis = 1
        result = df.dropna(axis="columns")
        expected = pd.DataFrame(index=[0, 1])
        self.assert_frame_equal(result, expected)

        # multiple
        df = pd.DataFrame({"A": data_missing, "B": [1, np.nan]})
        result = df.dropna()
        expected = df.iloc[:0]
        self.assert_frame_equal(result, expected)

    def test_fillna_scalar(self, data_missing):
        valid = data_missing[1]
        result = data_missing.fillna(valid)
        expected = data_missing.fillna(valid)
        self.assert_extension_array_equal(result, expected)

    def test_fillna_limit_pad(self, data_missing):
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).fillna(method="ffill", limit=2)
        expected = pd.Series(data_missing.take([1, 1, 1, 0, 1]))
        self.assert_series_equal(result, expected)

    def test_fillna_limit_backfill(self, data_missing):
        arr = data_missing.take([1, 0, 0, 0, 1])
        result = pd.Series(arr).fillna(method="backfill", limit=2)
        expected = pd.Series(data_missing.take([1, 0, 1, 1, 1]))
        self.assert_series_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        self.assert_extension_array_equal(result, data)

        result = data.fillna(method="backfill")
        assert result is not data
        self.assert_extension_array_equal(result, data)

    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )
        self.assert_series_equal(result, expected)

        # Fill with a series
        result = ser.fillna(expected)
        self.assert_series_equal(result, expected)

        # Fill with a series not affecting the missing values
        result = ser.fillna(ser)
        self.assert_series_equal(result, ser)

    def test_fillna_series_method(self, data_missing, fillna_method):
        fill_value = data_missing[1]

        if fillna_method == "ffill":
            data_missing = data_missing[::-1]

        result = pd.Series(data_missing).fillna(method=fillna_method)
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )

        self.assert_series_equal(result, expected)

    def test_fillna_frame(self, data_missing):
        fill_value = data_missing[1]

        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence(
                    [fill_value, fill_value], dtype=data_missing.dtype
                ),
                "B": [1, 2],
            }
        )

        self.assert_frame_equal(result, expected)

    def test_fillna_fill_other(self, data):
        result = pd.DataFrame({"A": data, "B": [np.nan] * len(data)}).fillna({"B": 0.0})

        expected = pd.DataFrame({"A": data, "B": [0.0] * len(result)})

        self.assert_frame_equal(result, expected)

    def test_use_inf_as_na_no_effect(self, data_missing):
        ser = pd.Series(data_missing)
        expected = ser.isna()
        with pd.option_context("mode.use_inf_as_na", True):
            result = ser.isna()
        self.assert_series_equal(result, expected)
