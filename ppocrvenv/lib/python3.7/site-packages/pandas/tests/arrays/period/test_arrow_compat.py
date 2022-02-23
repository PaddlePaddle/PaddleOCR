import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    PeriodArray,
    period_array,
)

pyarrow_skip = td.skip_if_no("pyarrow", min_version="0.17.0")


@pyarrow_skip
def test_arrow_extension_type():
    from pandas.core.arrays._arrow_utils import ArrowPeriodType

    p1 = ArrowPeriodType("D")
    p2 = ArrowPeriodType("D")
    p3 = ArrowPeriodType("M")

    assert p1.freq == "D"
    assert p1 == p2
    assert not p1 == p3
    assert hash(p1) == hash(p2)
    assert not hash(p1) == hash(p3)


@pyarrow_skip
@pytest.mark.parametrize(
    "data, freq",
    [
        (pd.date_range("2017", periods=3), "D"),
        (pd.date_range("2017", periods=3, freq="A"), "A-DEC"),
    ],
)
def test_arrow_array(data, freq):
    import pyarrow as pa

    from pandas.core.arrays._arrow_utils import ArrowPeriodType

    periods = period_array(data, freq=freq)
    result = pa.array(periods)
    assert isinstance(result.type, ArrowPeriodType)
    assert result.type.freq == freq
    expected = pa.array(periods.asi8, type="int64")
    assert result.storage.equals(expected)

    # convert to its storage type
    result = pa.array(periods, type=pa.int64())
    assert result.equals(expected)

    # unsupported conversions
    msg = "Not supported to convert PeriodArray to 'double' type"
    with pytest.raises(TypeError, match=msg):
        pa.array(periods, type="float64")

    with pytest.raises(TypeError, match="different 'freq'"):
        pa.array(periods, type=ArrowPeriodType("T"))


@pyarrow_skip
def test_arrow_array_missing():
    import pyarrow as pa

    from pandas.core.arrays._arrow_utils import ArrowPeriodType

    arr = PeriodArray([1, 2, 3], freq="D")
    arr[1] = pd.NaT

    result = pa.array(arr)
    assert isinstance(result.type, ArrowPeriodType)
    assert result.type.freq == "D"
    expected = pa.array([1, None, 3], type="int64")
    assert result.storage.equals(expected)


@pyarrow_skip
def test_arrow_table_roundtrip():
    import pyarrow as pa

    from pandas.core.arrays._arrow_utils import ArrowPeriodType

    arr = PeriodArray([1, 2, 3], freq="D")
    arr[1] = pd.NaT
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    assert isinstance(table.field("a").type, ArrowPeriodType)
    result = table.to_pandas()
    assert isinstance(result["a"].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)

    table2 = pa.concat_tables([table, table])
    result = table2.to_pandas()
    expected = pd.concat([df, df], ignore_index=True)
    tm.assert_frame_equal(result, expected)


@pyarrow_skip
def test_arrow_load_from_zero_chunks():
    # GH-41040
    import pyarrow as pa

    from pandas.core.arrays._arrow_utils import ArrowPeriodType

    arr = PeriodArray([], freq="D")
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    assert isinstance(table.field("a").type, ArrowPeriodType)
    table = pa.table(
        [pa.chunked_array([], type=table.column(0).type)], schema=table.schema
    )
    result = table.to_pandas()
    assert isinstance(result["a"].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)


@pyarrow_skip
def test_arrow_table_roundtrip_without_metadata():
    import pyarrow as pa

    arr = PeriodArray([1, 2, 3], freq="H")
    arr[1] = pd.NaT
    df = pd.DataFrame({"a": arr})

    table = pa.table(df)
    # remove the metadata
    table = table.replace_schema_metadata()
    assert table.schema.metadata is None

    result = table.to_pandas()
    assert isinstance(result["a"].dtype, PeriodDtype)
    tm.assert_frame_equal(result, df)
