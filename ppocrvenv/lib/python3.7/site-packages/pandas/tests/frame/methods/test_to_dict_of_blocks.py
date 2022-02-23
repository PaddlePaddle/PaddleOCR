import numpy as np

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    MultiIndex,
)
import pandas._testing as tm
from pandas.core.arrays import PandasArray

pytestmark = td.skip_array_manager_invalid_test


class TestToDictOfBlocks:
    def test_copy_blocks(self, float_frame):
        # GH#9607
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]

        # use the default copy=True, change a column
        blocks = df._to_dict_of_blocks(copy=True)
        for _df in blocks.values():
            if column in _df:
                _df.loc[:, column] = _df[column] + 1

        # make sure we did not change the original DataFrame
        assert not _df[column].equals(df[column])

    def test_no_copy_blocks(self, float_frame):
        # GH#9607
        df = DataFrame(float_frame, copy=True)
        column = df.columns[0]

        # use the copy=False, change a column
        blocks = df._to_dict_of_blocks(copy=False)
        for _df in blocks.values():
            if column in _df:
                _df.loc[:, column] = _df[column] + 1

        # make sure we did change the original DataFrame
        assert _df[column].equals(df[column])


def test_to_dict_of_blocks_item_cache():
    # Calling to_dict_of_blocks should not poison item_cache
    df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    df["c"] = PandasArray(np.array([1, 2, None, 3], dtype=object))
    mgr = df._mgr
    assert len(mgr.blocks) == 3  # i.e. not consolidated

    ser = df["b"]  # populations item_cache["b"]

    df._to_dict_of_blocks()

    # Check that the to_dict_of_blocks didn't break link between ser and df
    ser.values[0] = "foo"
    assert df.loc[0, "b"] == "foo"

    assert df["b"] is ser


def test_set_change_dtype_slice():
    # GH#8850
    cols = MultiIndex.from_tuples([("1st", "a"), ("2nd", "b"), ("3rd", "c")])
    df = DataFrame([[1.0, 2, 3], [4.0, 5, 6]], columns=cols)
    df["2nd"] = df["2nd"] * 2.0

    blocks = df._to_dict_of_blocks()
    assert sorted(blocks.keys()) == ["float64", "int64"]
    tm.assert_frame_equal(
        blocks["float64"], DataFrame([[1.0, 4.0], [4.0, 10.0]], columns=cols[:2])
    )
    tm.assert_frame_equal(blocks["int64"], DataFrame([[3], [6]], columns=cols[2:]))
