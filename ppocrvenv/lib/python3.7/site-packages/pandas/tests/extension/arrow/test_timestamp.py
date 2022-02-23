from __future__ import annotations

import datetime

import pytest

from pandas._typing import type_t

import pandas as pd
from pandas.api.extensions import (
    ExtensionDtype,
    register_extension_dtype,
)

pytest.importorskip("pyarrow", minversion="0.13.0")

import pyarrow as pa  # isort:skip

from pandas.tests.extension.arrow.arrays import ArrowExtensionArray  # isort:skip


@register_extension_dtype
class ArrowTimestampUSDtype(ExtensionDtype):

    type = datetime.datetime
    kind = "M"
    name = "arrow_timestamp_us"
    na_value = pa.NULL

    @classmethod
    def construct_array_type(cls) -> type_t[ArrowTimestampUSArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return ArrowTimestampUSArray


class ArrowTimestampUSArray(ArrowExtensionArray):
    def __init__(self, values):
        if not isinstance(values, pa.ChunkedArray):
            raise ValueError

        assert values.type == pa.timestamp("us")
        self._data = values
        self._dtype = ArrowTimestampUSDtype()


def test_constructor_extensionblock():
    # GH 34986
    pd.DataFrame(
        {
            "timestamp": ArrowTimestampUSArray.from_scalars(
                [None, datetime.datetime(2010, 9, 8, 7, 6, 5, 4)]
            )
        }
    )
