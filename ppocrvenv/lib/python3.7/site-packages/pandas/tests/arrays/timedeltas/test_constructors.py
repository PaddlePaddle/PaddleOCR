import numpy as np
import pytest

from pandas.core.arrays import TimedeltaArray


class TestTimedeltaArrayConstructor:
    def test_only_1dim_accepted(self):
        # GH#25282
        arr = np.array([0, 1, 2, 3], dtype="m8[h]").astype("m8[ns]")

        with pytest.raises(ValueError, match="Only 1-dimensional"):
            # 3-dim, we allow 2D to sneak in for ops purposes GH#29853
            TimedeltaArray(arr.reshape(2, 2, 1))

        with pytest.raises(ValueError, match="Only 1-dimensional"):
            # 0-dim
            TimedeltaArray(arr[[0]].squeeze())

    def test_freq_validation(self):
        # ensure that the public constructor cannot create an invalid instance
        arr = np.array([0, 0, 1], dtype=np.int64) * 3600 * 10 ** 9

        msg = (
            "Inferred frequency None from passed values does not "
            "conform to passed frequency D"
        )
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray(arr.view("timedelta64[ns]"), freq="D")

    def test_non_array_raises(self):
        with pytest.raises(ValueError, match="list"):
            TimedeltaArray([1, 2, 3])

    def test_other_type_raises(self):
        with pytest.raises(ValueError, match="dtype bool cannot be converted"):
            TimedeltaArray(np.array([1, 2, 3], dtype="bool"))

    def test_incorrect_dtype_raises(self):
        # TODO: why TypeError for 'category' but ValueError for i8?
        with pytest.raises(
            ValueError, match=r"category cannot be converted to timedelta64\[ns\]"
        ):
            TimedeltaArray(np.array([1, 2, 3], dtype="i8"), dtype="category")

        with pytest.raises(
            ValueError, match=r"dtype int64 cannot be converted to timedelta64\[ns\]"
        ):
            TimedeltaArray(np.array([1, 2, 3], dtype="i8"), dtype=np.dtype("int64"))

    def test_copy(self):
        data = np.array([1, 2, 3], dtype="m8[ns]")
        arr = TimedeltaArray(data, copy=False)
        assert arr._data is data

        arr = TimedeltaArray(data, copy=True)
        assert arr._data is not data
        assert arr._data.base is not data

    def test_from_sequence_dtype(self):
        msg = "dtype .*object.* cannot be converted to timedelta64"
        with pytest.raises(ValueError, match=msg):
            TimedeltaArray._from_sequence([], dtype=object)
