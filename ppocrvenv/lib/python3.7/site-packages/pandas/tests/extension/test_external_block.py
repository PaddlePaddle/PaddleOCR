import numpy as np
import pytest

from pandas._libs.internals import BlockPlacement
import pandas.util._test_decorators as td

import pandas as pd
from pandas.core.internals import BlockManager
from pandas.core.internals.blocks import ExtensionBlock

pytestmark = td.skip_array_manager_invalid_test


class CustomBlock(ExtensionBlock):

    _holder = np.ndarray

    # Cannot override final attribute "_can_hold_na"
    @property  # type: ignore[misc]
    def _can_hold_na(self) -> bool:
        return False


@pytest.fixture
def df():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    blocks = df1._mgr.blocks
    values = np.arange(3, dtype="int64")
    bp = BlockPlacement(slice(1, 2))
    custom_block = CustomBlock(values, placement=bp, ndim=2)
    blocks = blocks + (custom_block,)
    block_manager = BlockManager(blocks, [pd.Index(["a", "b"]), df1.index])
    return pd.DataFrame(block_manager)


def test_concat_axis1(df):
    # GH17954
    df2 = pd.DataFrame({"c": [0.1, 0.2, 0.3]})
    res = pd.concat([df, df2], axis=1)
    assert isinstance(res._mgr.blocks[1], CustomBlock)
