import numpy as np
import pytest

import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray


class TestTimedeltaArray:
    @pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
    def test_astype_int(self, dtype):
        arr = TimedeltaArray._from_sequence([Timedelta("1H"), Timedelta("2H")])
        with tm.assert_produces_warning(FutureWarning):
            # astype(int..) deprecated
            result = arr.astype(dtype)

        if np.dtype(dtype).kind == "u":
            expected_dtype = np.dtype("uint64")
        else:
            expected_dtype = np.dtype("int64")

        with tm.assert_produces_warning(FutureWarning):
            # astype(int..) deprecated
            expected = arr.astype(expected_dtype)

        assert result.dtype == expected_dtype
        tm.assert_numpy_array_equal(result, expected)

    def test_setitem_clears_freq(self):
        a = TimedeltaArray(pd.timedelta_range("1H", periods=2, freq="H"))
        a[0] = Timedelta("1H")
        assert a.freq is None

    @pytest.mark.parametrize(
        "obj",
        [
            Timedelta(seconds=1),
            Timedelta(seconds=1).to_timedelta64(),
            Timedelta(seconds=1).to_pytimedelta(),
        ],
    )
    def test_setitem_objects(self, obj):
        # make sure we accept timedelta64 and timedelta in addition to Timedelta
        tdi = pd.timedelta_range("2 Days", periods=4, freq="H")
        arr = TimedeltaArray(tdi, freq=tdi.freq)

        arr[0] = obj
        assert arr[0] == Timedelta(seconds=1)

    @pytest.mark.parametrize(
        "other",
        [
            1,
            np.int64(1),
            1.0,
            np.datetime64("NaT"),
            pd.Timestamp.now(),
            "invalid",
            np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9,
            (np.arange(10) * 24 * 3600 * 10 ** 9).view("datetime64[ns]"),
            pd.Timestamp.now().to_period("D"),
        ],
    )
    @pytest.mark.parametrize("index", [True, False])
    def test_searchsorted_invalid_types(self, other, index):
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10 ** 9
        arr = TimedeltaArray(data, freq="D")
        if index:
            arr = pd.Index(arr)

        msg = "|".join(
            [
                "searchsorted requires compatible dtype or scalar",
                "value should be a 'Timedelta', 'NaT', or array of those. Got",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            arr.searchsorted(other)


class TestUnaryOps:
    def test_abs(self):
        vals = np.array([-3600 * 10 ** 9, "NaT", 7200 * 10 ** 9], dtype="m8[ns]")
        arr = TimedeltaArray(vals)

        evals = np.array([3600 * 10 ** 9, "NaT", 7200 * 10 ** 9], dtype="m8[ns]")
        expected = TimedeltaArray(evals)

        result = abs(arr)
        tm.assert_timedelta_array_equal(result, expected)

    def test_neg(self):
        vals = np.array([-3600 * 10 ** 9, "NaT", 7200 * 10 ** 9], dtype="m8[ns]")
        arr = TimedeltaArray(vals)

        evals = np.array([3600 * 10 ** 9, "NaT", -7200 * 10 ** 9], dtype="m8[ns]")
        expected = TimedeltaArray(evals)

        result = -arr
        tm.assert_timedelta_array_equal(result, expected)

    def test_neg_freq(self):
        tdi = pd.timedelta_range("2 Days", periods=4, freq="H")
        arr = TimedeltaArray(tdi, freq=tdi.freq)

        expected = TimedeltaArray(-tdi._data, freq=-tdi.freq)

        result = -arr
        tm.assert_timedelta_array_equal(result, expected)
