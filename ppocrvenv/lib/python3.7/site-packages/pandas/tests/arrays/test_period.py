import numpy as np
import pytest

from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
    PeriodArray,
    period_array,
)

# ----------------------------------------------------------------------------
# Dtype


def test_registered():
    assert PeriodDtype in registry.dtypes
    result = registry.find("Period[D]")
    expected = PeriodDtype("D")
    assert result == expected


# ----------------------------------------------------------------------------
# period_array


def test_asi8():
    result = period_array(["2000", "2001", None], freq="D").asi8
    expected = np.array([10957, 11323, iNaT])
    tm.assert_numpy_array_equal(result, expected)


def test_take_raises():
    arr = period_array(["2000", "2001"], freq="D")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr.take([0, -1], allow_fill=True, fill_value=pd.Period("2000", freq="W"))

    msg = "value should be a 'Period' or 'NaT'. Got 'str' instead"
    with pytest.raises(TypeError, match=msg):
        arr.take([0, -1], allow_fill=True, fill_value="foo")


def test_fillna_raises():
    arr = period_array(["2000", "2001", "2002"], freq="D")
    with pytest.raises(ValueError, match="Length"):
        arr.fillna(arr[:2])


def test_fillna_copies():
    arr = period_array(["2000", "2001", "2002"], freq="D")
    result = arr.fillna(pd.Period("2000", "D"))
    assert result is not arr


# ----------------------------------------------------------------------------
# setitem


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ([0], pd.Period("2000", "D"), [10957, 1, 2]),
        ([0], None, [iNaT, 1, 2]),
        ([0], np.nan, [iNaT, 1, 2]),
        ([0, 1, 2], pd.Period("2000", "D"), [10957] * 3),
        (
            [0, 1, 2],
            [pd.Period("2000", "D"), pd.Period("2001", "D"), pd.Period("2002", "D")],
            [10957, 11323, 11688],
        ),
    ],
)
def test_setitem(key, value, expected):
    arr = PeriodArray(np.arange(3), freq="D")
    expected = PeriodArray(expected, freq="D")
    arr[key] = value
    tm.assert_period_array_equal(arr, expected)


def test_setitem_raises_incompatible_freq():
    arr = PeriodArray(np.arange(3), freq="D")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr[0] = pd.Period("2000", freq="A")

    other = period_array(["2000", "2001"], freq="A")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr[[0, 1]] = other


def test_setitem_raises_length():
    arr = PeriodArray(np.arange(3), freq="D")
    with pytest.raises(ValueError, match="length"):
        arr[[0, 1]] = [pd.Period("2000", freq="D")]


def test_setitem_raises_type():
    arr = PeriodArray(np.arange(3), freq="D")
    with pytest.raises(TypeError, match="int"):
        arr[0] = 1


# ----------------------------------------------------------------------------
# Ops


def test_sub_period():
    arr = period_array(["2000", "2001"], freq="D")
    other = pd.Period("2000", freq="M")
    with pytest.raises(IncompatibleFrequency, match="freq"):
        arr - other


# ----------------------------------------------------------------------------
# Methods


@pytest.mark.parametrize(
    "other",
    [pd.Period("2000", freq="H"), period_array(["2000", "2001", "2000"], freq="H")],
)
def test_where_different_freq_raises(other):
    ser = pd.Series(period_array(["2000", "2001", "2002"], freq="D"))
    cond = np.array([True, False, True])
    with pytest.raises(IncompatibleFrequency, match="freq"):
        ser.where(cond, other)


# ----------------------------------------------------------------------------
# Printing


def test_repr_small():
    arr = period_array(["2000", "2001"], freq="D")
    result = str(arr)
    expected = (
        "<PeriodArray>\n['2000-01-01', '2001-01-01']\nLength: 2, dtype: period[D]"
    )
    assert result == expected


def test_repr_large():
    arr = period_array(["2000", "2001"] * 500, freq="D")
    result = str(arr)
    expected = (
        "<PeriodArray>\n"
        "['2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01',\n"
        " ...\n"
        " '2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01']\n"
        "Length: 1000, dtype: period[D]"
    )
    assert result == expected
