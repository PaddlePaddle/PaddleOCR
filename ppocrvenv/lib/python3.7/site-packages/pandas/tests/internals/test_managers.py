"""
Testing interaction between the different managers (BlockManager, ArrayManager)
"""
from pandas.core.dtypes.missing import array_equivalent

import pandas as pd
import pandas._testing as tm
from pandas.core.internals import (
    ArrayManager,
    BlockManager,
    SingleArrayManager,
    SingleBlockManager,
)


def test_dataframe_creation():

    with pd.option_context("mode.data_manager", "block"):
        df_block = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [4, 5, 6]})
    assert isinstance(df_block._mgr, BlockManager)

    with pd.option_context("mode.data_manager", "array"):
        df_array = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [4, 5, 6]})
    assert isinstance(df_array._mgr, ArrayManager)

    # also ensure both are seen as equal
    tm.assert_frame_equal(df_block, df_array)

    # conversion from one manager to the other
    result = df_block._as_manager("block")
    assert isinstance(result._mgr, BlockManager)
    result = df_block._as_manager("array")
    assert isinstance(result._mgr, ArrayManager)
    tm.assert_frame_equal(result, df_block)
    assert all(
        array_equivalent(left, right)
        for left, right in zip(result._mgr.arrays, df_array._mgr.arrays)
    )

    result = df_array._as_manager("array")
    assert isinstance(result._mgr, ArrayManager)
    result = df_array._as_manager("block")
    assert isinstance(result._mgr, BlockManager)
    tm.assert_frame_equal(result, df_array)
    assert len(result._mgr.blocks) == 2


def test_series_creation():

    with pd.option_context("mode.data_manager", "block"):
        s_block = pd.Series([1, 2, 3], name="A", index=["a", "b", "c"])
    assert isinstance(s_block._mgr, SingleBlockManager)

    with pd.option_context("mode.data_manager", "array"):
        s_array = pd.Series([1, 2, 3], name="A", index=["a", "b", "c"])
    assert isinstance(s_array._mgr, SingleArrayManager)

    # also ensure both are seen as equal
    tm.assert_series_equal(s_block, s_array)

    # conversion from one manager to the other
    result = s_block._as_manager("block")
    assert isinstance(result._mgr, SingleBlockManager)
    result = s_block._as_manager("array")
    assert isinstance(result._mgr, SingleArrayManager)
    tm.assert_series_equal(result, s_block)

    result = s_array._as_manager("array")
    assert isinstance(result._mgr, SingleArrayManager)
    result = s_array._as_manager("block")
    assert isinstance(result._mgr, SingleBlockManager)
    tm.assert_series_equal(result, s_array)
