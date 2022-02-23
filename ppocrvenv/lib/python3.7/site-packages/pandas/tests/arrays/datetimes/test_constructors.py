import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays.datetimes import sequence_to_dt64ns


class TestDatetimeArrayConstructor:
    def test_from_sequence_invalid_type(self):
        mi = pd.MultiIndex.from_product([np.arange(5), np.arange(5)])
        with pytest.raises(TypeError, match="Cannot create a DatetimeArray"):
            DatetimeArray._from_sequence(mi)

    def test_only_1dim_accepted(self):
        arr = np.array([0, 1, 2, 3], dtype="M8[h]").astype("M8[ns]")

        with pytest.raises(ValueError, match="Only 1-dimensional"):
            # 3-dim, we allow 2D to sneak in for ops purposes GH#29853
            DatetimeArray(arr.reshape(2, 2, 1))

        with pytest.raises(ValueError, match="Only 1-dimensional"):
            # 0-dim
            DatetimeArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        # GH#24623 check that invalid instances cannot be created with the
        #  public constructor
        arr = np.arange(5, dtype=np.int64) * 3600 * 10 ** 9

        msg = (
            "Inferred frequency H from passed values does not "
            "conform to passed frequency W-SUN"
        )
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr, freq="W")

    @pytest.mark.parametrize(
        "meth",
        [
            DatetimeArray._from_sequence,
            sequence_to_dt64ns,
            pd.to_datetime,
            pd.DatetimeIndex,
        ],
    )
    def test_mixing_naive_tzaware_raises(self, meth):
        # GH#24569
        arr = np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")])

        msg = (
            "Cannot mix tz-aware with tz-naive values|"
            "Tz-aware datetime.datetime cannot be converted "
            "to datetime64 unless utc=True"
        )

        for obj in [arr, arr[::-1]]:
            # check that we raise regardless of whether naive is found
            #  before aware or vice-versa
            with pytest.raises(ValueError, match=msg):
                meth(obj)

    def test_from_pandas_array(self):
        arr = pd.array(np.arange(5, dtype=np.int64)) * 3600 * 10 ** 9

        result = DatetimeArray._from_sequence(arr)._with_freq("infer")

        expected = pd.date_range("1970-01-01", periods=5, freq="H")._data
        tm.assert_datetime_array_equal(result, expected)

    def test_mismatched_timezone_raises(self):
        arr = DatetimeArray(
            np.array(["2000-01-01T06:00:00"], dtype="M8[ns]"),
            dtype=DatetimeTZDtype(tz="US/Central"),
        )
        dtype = DatetimeTZDtype(tz="US/Eastern")
        with pytest.raises(TypeError, match="Timezone of the array"):
            DatetimeArray(arr, dtype=dtype)

    def test_non_array_raises(self):
        with pytest.raises(ValueError, match="list"):
            DatetimeArray([1, 2, 3])

    def test_bool_dtype_raises(self):
        arr = np.array([1, 2, 3], dtype="bool")

        with pytest.raises(
            ValueError, match="The dtype of 'values' is incorrect.*bool"
        ):
            DatetimeArray(arr)

        msg = r"dtype bool cannot be converted to datetime64\[ns\]"
        with pytest.raises(TypeError, match=msg):
            DatetimeArray._from_sequence(arr)

        with pytest.raises(TypeError, match=msg):
            sequence_to_dt64ns(arr)

        with pytest.raises(TypeError, match=msg):
            pd.DatetimeIndex(arr)

        with pytest.raises(TypeError, match=msg):
            pd.to_datetime(arr)

    def test_incorrect_dtype_raises(self):
        with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
            DatetimeArray(np.array([1, 2, 3], dtype="i8"), dtype="category")

    def test_freq_infer_raises(self):
        with pytest.raises(ValueError, match="Frequency inference"):
            DatetimeArray(np.array([1, 2, 3], dtype="i8"), freq="infer")

    def test_copy(self):
        data = np.array([1, 2, 3], dtype="M8[ns]")
        arr = DatetimeArray(data, copy=False)
        assert arr._data is data

        arr = DatetimeArray(data, copy=True)
        assert arr._data is not data


class TestSequenceToDT64NS:
    def test_tz_dtype_mismatch_raises(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        with pytest.raises(TypeError, match="data is already tz-aware"):
            sequence_to_dt64ns(arr, dtype=DatetimeTZDtype(tz="UTC"))

    def test_tz_dtype_matches(self):
        arr = DatetimeArray._from_sequence(
            ["2000"], dtype=DatetimeTZDtype(tz="US/Central")
        )
        result, _, _ = sequence_to_dt64ns(arr, dtype=DatetimeTZDtype(tz="US/Central"))
        tm.assert_numpy_array_equal(arr._data, result)

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_2d(self, order):
        dti = pd.date_range("2016-01-01", periods=6, tz="US/Pacific")
        arr = np.array(dti, dtype=object).reshape(3, 2)
        if order == "F":
            arr = arr.T

        res = sequence_to_dt64ns(arr)
        expected = sequence_to_dt64ns(arr.ravel())

        tm.assert_numpy_array_equal(res[0].ravel(), expected[0])
        assert res[1] == expected[1]
        assert res[2] == expected[2]

        res = DatetimeArray._from_sequence(arr)
        expected = DatetimeArray._from_sequence(arr.ravel()).reshape(arr.shape)
        tm.assert_datetime_array_equal(res, expected)
