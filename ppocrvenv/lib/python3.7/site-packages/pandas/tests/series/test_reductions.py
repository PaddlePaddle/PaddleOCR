import numpy as np
import pytest

import pandas as pd
from pandas import (
    MultiIndex,
    Series,
)
import pandas._testing as tm


def test_reductions_td64_with_nat():
    # GH#8617
    ser = Series([0, pd.NaT], dtype="m8[ns]")
    exp = ser[0]
    assert ser.median() == exp
    assert ser.min() == exp
    assert ser.max() == exp


@pytest.mark.parametrize("skipna", [True, False])
def test_td64_sum_empty(skipna):
    # GH#37151
    ser = Series([], dtype="timedelta64[ns]")

    result = ser.sum(skipna=skipna)
    assert isinstance(result, pd.Timedelta)
    assert result == pd.Timedelta(0)


def test_td64_summation_overflow():
    # GH#9442
    ser = Series(pd.date_range("20130101", periods=100000, freq="H"))
    ser[0] += pd.Timedelta("1s 1ms")

    # mean
    result = (ser - ser.min()).mean()
    expected = pd.Timedelta((pd.TimedeltaIndex(ser - ser.min()).asi8 / len(ser)).sum())

    # the computation is converted to float so
    # might be some loss of precision
    assert np.allclose(result.value / 1000, expected.value / 1000)

    # sum
    msg = "overflow in timedelta operation"
    with pytest.raises(ValueError, match=msg):
        (ser - ser.min()).sum()

    s1 = ser[0:10000]
    with pytest.raises(ValueError, match=msg):
        (s1 - s1.min()).sum()
    s2 = ser[0:1000]
    (s2 - s2.min()).sum()


def test_prod_numpy16_bug():
    ser = Series([1.0, 1.0, 1.0], index=range(3))
    result = ser.prod()

    assert not isinstance(result, Series)


def test_sum_with_level():
    obj = Series([10.0], index=MultiIndex.from_tuples([(2, 3)]))

    with tm.assert_produces_warning(FutureWarning):
        result = obj.sum(level=0)
    expected = Series([10.0], index=[2])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [np.any, np.all])
@pytest.mark.parametrize("kwargs", [{"keepdims": True}, {"out": object()}])
def test_validate_any_all_out_keepdims_raises(kwargs, func):
    ser = Series([1, 2])
    param = list(kwargs)[0]
    name = func.__name__

    msg = (
        f"the '{param}' parameter is not "
        "supported in the pandas "
        fr"implementation of {name}\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        func(ser, **kwargs)


def test_validate_sum_initial():
    ser = Series([1, 2])
    msg = (
        r"the 'initial' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, initial=10)


def test_validate_median_initial():
    ser = Series([1, 2])
    msg = (
        r"the 'overwrite_input' parameter is not "
        r"supported in the pandas "
        r"implementation of median\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        # It seems like np.median doesn't dispatch, so we use the
        # method instead of the ufunc.
        ser.median(overwrite_input=True)


def test_validate_stat_keepdims():
    ser = Series([1, 2])
    msg = (
        r"the 'keepdims' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, keepdims=True)
