from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas.core.dtypes.cast import astype_nansafe
import pandas.core.dtypes.common as com
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    CategoricalDtypeType,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import isna

import pandas as pd
import pandas._testing as tm
from pandas.arrays import SparseArray


# EA & Actual Dtypes
def to_ea_dtypes(dtypes):
    """convert list of string dtypes to EA dtype"""
    return [getattr(pd, dt + "Dtype") for dt in dtypes]


def to_numpy_dtypes(dtypes):
    """convert list of string dtypes to numpy dtype"""
    return [getattr(np, dt) for dt in dtypes if isinstance(dt, str)]


class TestPandasDtype:

    # Passing invalid dtype, both as a string or object, must raise TypeError
    # Per issue GH15520
    @pytest.mark.parametrize("box", [pd.Timestamp, "pd.Timestamp", list])
    def test_invalid_dtype_error(self, box):
        with pytest.raises(TypeError, match="not understood"):
            com.pandas_dtype(box)

    @pytest.mark.parametrize(
        "dtype",
        [
            object,
            "float64",
            np.object_,
            np.dtype("object"),
            "O",
            np.float64,
            float,
            np.dtype("float64"),
        ],
    )
    def test_pandas_dtype_valid(self, dtype):
        assert com.pandas_dtype(dtype) == dtype

    @pytest.mark.parametrize(
        "dtype", ["M8[ns]", "m8[ns]", "object", "float64", "int64"]
    )
    def test_numpy_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == np.dtype(dtype)

    def test_numpy_string_dtype(self):
        # do not parse freq-like string as period dtype
        assert com.pandas_dtype("U") == np.dtype("U")
        assert com.pandas_dtype("S") == np.dtype("S")

    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[ns, US/Eastern]",
            "datetime64[ns, Asia/Tokyo]",
            "datetime64[ns, UTC]",
            # GH#33885 check that the M8 alias is understood
            "M8[ns, US/Eastern]",
            "M8[ns, Asia/Tokyo]",
            "M8[ns, UTC]",
        ],
    )
    def test_datetimetz_dtype(self, dtype):
        assert com.pandas_dtype(dtype) == DatetimeTZDtype.construct_from_string(dtype)
        assert com.pandas_dtype(dtype) == dtype

    def test_categorical_dtype(self):
        assert com.pandas_dtype("category") == CategoricalDtype()

    @pytest.mark.parametrize(
        "dtype",
        [
            "period[D]",
            "period[3M]",
            "period[U]",
            "Period[D]",
            "Period[3M]",
            "Period[U]",
        ],
    )
    def test_period_dtype(self, dtype):
        assert com.pandas_dtype(dtype) is PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == PeriodDtype(dtype)
        assert com.pandas_dtype(dtype) == dtype


dtypes = {
    "datetime_tz": com.pandas_dtype("datetime64[ns, US/Eastern]"),
    "datetime": com.pandas_dtype("datetime64[ns]"),
    "timedelta": com.pandas_dtype("timedelta64[ns]"),
    "period": PeriodDtype("D"),
    "integer": np.dtype(np.int64),
    "float": np.dtype(np.float64),
    "object": np.dtype(object),
    "category": com.pandas_dtype("category"),
}


@pytest.mark.parametrize("name1,dtype1", list(dtypes.items()), ids=lambda x: str(x))
@pytest.mark.parametrize("name2,dtype2", list(dtypes.items()), ids=lambda x: str(x))
def test_dtype_equal(name1, dtype1, name2, dtype2):

    # match equal to self, but not equal to other
    assert com.is_dtype_equal(dtype1, dtype1)
    if name1 != name2:
        assert not com.is_dtype_equal(dtype1, dtype2)


@pytest.mark.parametrize(
    "dtype1,dtype2",
    [
        (np.int8, np.int64),
        (np.int16, np.int64),
        (np.int32, np.int64),
        (np.float32, np.float64),
        (PeriodDtype("D"), PeriodDtype("2D")),  # PeriodType
        (
            com.pandas_dtype("datetime64[ns, US/Eastern]"),
            com.pandas_dtype("datetime64[ns, CET]"),
        ),  # Datetime
        (None, None),  # gh-15941: no exception should be raised.
    ],
)
def test_dtype_equal_strict(dtype1, dtype2):
    assert not com.is_dtype_equal(dtype1, dtype2)


def get_is_dtype_funcs():
    """
    Get all functions in pandas.core.dtypes.common that
    begin with 'is_' and end with 'dtype'

    """
    fnames = [f for f in dir(com) if (f.startswith("is_") and f.endswith("dtype"))]
    return [getattr(com, fname) for fname in fnames]


@pytest.mark.parametrize("func", get_is_dtype_funcs(), ids=lambda x: x.__name__)
def test_get_dtype_error_catch(func):
    # see gh-15941
    #
    # No exception should be raised.

    assert not func(None)


def test_is_object():
    assert com.is_object_dtype(object)
    assert com.is_object_dtype(np.array([], dtype=object))

    assert not com.is_object_dtype(int)
    assert not com.is_object_dtype(np.array([], dtype=int))
    assert not com.is_object_dtype([1, 2, 3])


@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no_scipy)]
)
def test_is_sparse(check_scipy):
    assert com.is_sparse(SparseArray([1, 2, 3]))

    assert not com.is_sparse(np.array([1, 2, 3]))

    if check_scipy:
        import scipy.sparse

        assert not com.is_sparse(scipy.sparse.bsr_matrix([1, 2, 3]))


@td.skip_if_no_scipy
def test_is_scipy_sparse():
    from scipy.sparse import bsr_matrix

    assert com.is_scipy_sparse(bsr_matrix([1, 2, 3]))

    assert not com.is_scipy_sparse(SparseArray([1, 2, 3]))


def test_is_categorical():
    cat = pd.Categorical([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning):
        assert com.is_categorical(cat)
        assert com.is_categorical(pd.Series(cat))
        assert com.is_categorical(pd.CategoricalIndex([1, 2, 3]))

        assert not com.is_categorical([1, 2, 3])


def test_is_categorical_deprecation():
    # GH#33385
    with tm.assert_produces_warning(FutureWarning):
        com.is_categorical([1, 2, 3])


def test_is_datetime64_dtype():
    assert not com.is_datetime64_dtype(object)
    assert not com.is_datetime64_dtype([1, 2, 3])
    assert not com.is_datetime64_dtype(np.array([], dtype=int))

    assert com.is_datetime64_dtype(np.datetime64)
    assert com.is_datetime64_dtype(np.array([], dtype=np.datetime64))


def test_is_datetime64tz_dtype():
    assert not com.is_datetime64tz_dtype(object)
    assert not com.is_datetime64tz_dtype([1, 2, 3])
    assert not com.is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))
    assert com.is_datetime64tz_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))


def test_is_timedelta64_dtype():
    assert not com.is_timedelta64_dtype(object)
    assert not com.is_timedelta64_dtype(None)
    assert not com.is_timedelta64_dtype([1, 2, 3])
    assert not com.is_timedelta64_dtype(np.array([], dtype=np.datetime64))
    assert not com.is_timedelta64_dtype("0 days")
    assert not com.is_timedelta64_dtype("0 days 00:00:00")
    assert not com.is_timedelta64_dtype(["0 days 00:00:00"])
    assert not com.is_timedelta64_dtype("NO DATE")

    assert com.is_timedelta64_dtype(np.timedelta64)
    assert com.is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    assert com.is_timedelta64_dtype(pd.to_timedelta(["0 days", "1 days"]))


def test_is_period_dtype():
    assert not com.is_period_dtype(object)
    assert not com.is_period_dtype([1, 2, 3])
    assert not com.is_period_dtype(pd.Period("2017-01-01"))

    assert com.is_period_dtype(PeriodDtype(freq="D"))
    assert com.is_period_dtype(pd.PeriodIndex([], freq="A"))


def test_is_interval_dtype():
    assert not com.is_interval_dtype(object)
    assert not com.is_interval_dtype([1, 2, 3])

    assert com.is_interval_dtype(IntervalDtype())

    interval = pd.Interval(1, 2, closed="right")
    assert not com.is_interval_dtype(interval)
    assert com.is_interval_dtype(pd.IntervalIndex([interval]))


def test_is_categorical_dtype():
    assert not com.is_categorical_dtype(object)
    assert not com.is_categorical_dtype([1, 2, 3])

    assert com.is_categorical_dtype(CategoricalDtype())
    assert com.is_categorical_dtype(pd.Categorical([1, 2, 3]))
    assert com.is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))


def test_is_string_dtype():
    assert not com.is_string_dtype(int)
    assert not com.is_string_dtype(pd.Series([1, 2]))

    assert com.is_string_dtype(str)
    assert com.is_string_dtype(object)
    assert com.is_string_dtype(np.array(["a", "b"]))
    assert com.is_string_dtype(pd.StringDtype())


def test_is_string_dtype_nullable(nullable_string_dtype):
    assert com.is_string_dtype(pd.array(["a", "b"], dtype=nullable_string_dtype))


integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    integer_dtypes
    + [pd.Series([1, 2])]
    + tm.ALL_INT_DTYPES
    + to_numpy_dtypes(tm.ALL_INT_DTYPES)
    + tm.ALL_EA_INT_DTYPES
    + to_ea_dtypes(tm.ALL_EA_INT_DTYPES),
)
def test_is_integer_dtype(dtype):
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ],
)
def test_is_not_integer_dtype(dtype):
    assert not com.is_integer_dtype(dtype)


signed_integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    signed_integer_dtypes
    + [pd.Series([1, 2])]
    + tm.SIGNED_INT_DTYPES
    + to_numpy_dtypes(tm.SIGNED_INT_DTYPES)
    + tm.SIGNED_EA_INT_DTYPES
    + to_ea_dtypes(tm.SIGNED_EA_INT_DTYPES),
)
def test_is_signed_integer_dtype(dtype):
    assert com.is_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ]
    + tm.UNSIGNED_INT_DTYPES
    + to_numpy_dtypes(tm.UNSIGNED_INT_DTYPES)
    + tm.UNSIGNED_EA_INT_DTYPES
    + to_ea_dtypes(tm.UNSIGNED_EA_INT_DTYPES),
)
def test_is_not_signed_integer_dtype(dtype):
    assert not com.is_signed_integer_dtype(dtype)


unsigned_integer_dtypes: list = []


@pytest.mark.parametrize(
    "dtype",
    unsigned_integer_dtypes
    + [pd.Series([1, 2], dtype=np.uint32)]
    + tm.UNSIGNED_INT_DTYPES
    + to_numpy_dtypes(tm.UNSIGNED_INT_DTYPES)
    + tm.UNSIGNED_EA_INT_DTYPES
    + to_ea_dtypes(tm.UNSIGNED_EA_INT_DTYPES),
)
def test_is_unsigned_integer_dtype(dtype):
    assert com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.datetime64,
        np.timedelta64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([], dtype=np.timedelta64),
    ]
    + tm.SIGNED_INT_DTYPES
    + to_numpy_dtypes(tm.SIGNED_INT_DTYPES)
    + tm.SIGNED_EA_INT_DTYPES
    + to_ea_dtypes(tm.SIGNED_EA_INT_DTYPES),
)
def test_is_not_unsigned_integer_dtype(dtype):
    assert not com.is_unsigned_integer_dtype(dtype)


@pytest.mark.parametrize(
    "dtype", [np.int64, np.array([1, 2], dtype=np.int64), "Int64", pd.Int64Dtype]
)
def test_is_int64_dtype(dtype):
    assert com.is_int64_dtype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        str,
        float,
        np.int32,
        np.uint64,
        pd.Index([1, 2.0]),
        np.array(["a", "b"]),
        np.array([1, 2], dtype=np.uint32),
        "int8",
        "Int8",
        pd.Int8Dtype,
    ],
)
def test_is_not_int64_dtype(dtype):
    assert not com.is_int64_dtype(dtype)


def test_is_datetime64_any_dtype():
    assert not com.is_datetime64_any_dtype(int)
    assert not com.is_datetime64_any_dtype(str)
    assert not com.is_datetime64_any_dtype(np.array([1, 2]))
    assert not com.is_datetime64_any_dtype(np.array(["a", "b"]))

    assert com.is_datetime64_any_dtype(np.datetime64)
    assert com.is_datetime64_any_dtype(np.array([], dtype=np.datetime64))
    assert com.is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_any_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]")
    )


def test_is_datetime64_ns_dtype():
    assert not com.is_datetime64_ns_dtype(int)
    assert not com.is_datetime64_ns_dtype(str)
    assert not com.is_datetime64_ns_dtype(np.datetime64)
    assert not com.is_datetime64_ns_dtype(np.array([1, 2]))
    assert not com.is_datetime64_ns_dtype(np.array(["a", "b"]))
    assert not com.is_datetime64_ns_dtype(np.array([], dtype=np.datetime64))

    # This datetime array has the wrong unit (ps instead of ns)
    assert not com.is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))

    assert com.is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    assert com.is_datetime64_ns_dtype(
        pd.DatetimeIndex([1, 2, 3], dtype=np.dtype("datetime64[ns]"))
    )


def test_is_timedelta64_ns_dtype():
    assert not com.is_timedelta64_ns_dtype(np.dtype("m8[ps]"))
    assert not com.is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))

    assert com.is_timedelta64_ns_dtype(np.dtype("m8[ns]"))
    assert com.is_timedelta64_ns_dtype(np.array([1, 2], dtype="m8[ns]"))


def test_is_datetime_or_timedelta_dtype():
    assert not com.is_datetime_or_timedelta_dtype(int)
    assert not com.is_datetime_or_timedelta_dtype(str)
    assert not com.is_datetime_or_timedelta_dtype(pd.Series([1, 2]))
    assert not com.is_datetime_or_timedelta_dtype(np.array(["a", "b"]))

    # TODO(jreback), this is slightly suspect
    assert not com.is_datetime_or_timedelta_dtype(DatetimeTZDtype("ns", "US/Eastern"))

    assert com.is_datetime_or_timedelta_dtype(np.datetime64)
    assert com.is_datetime_or_timedelta_dtype(np.timedelta64)
    assert com.is_datetime_or_timedelta_dtype(np.array([], dtype=np.timedelta64))
    assert com.is_datetime_or_timedelta_dtype(np.array([], dtype=np.datetime64))


def test_is_numeric_v_string_like():
    assert not com.is_numeric_v_string_like(np.array([1]), 1)
    assert not com.is_numeric_v_string_like(np.array([1]), np.array([2]))
    assert not com.is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))

    assert com.is_numeric_v_string_like(np.array([1]), "foo")
    assert com.is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    assert com.is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))


def test_is_datetimelike_v_numeric():
    dt = np.datetime64(datetime(2017, 1, 1))

    assert not com.is_datetimelike_v_numeric(1, 1)
    assert not com.is_datetimelike_v_numeric(dt, dt)
    assert not com.is_datetimelike_v_numeric(np.array([1]), np.array([2]))
    assert not com.is_datetimelike_v_numeric(np.array([dt]), np.array([dt]))

    assert com.is_datetimelike_v_numeric(1, dt)
    assert com.is_datetimelike_v_numeric(1, dt)
    assert com.is_datetimelike_v_numeric(np.array([dt]), 1)
    assert com.is_datetimelike_v_numeric(np.array([1]), dt)
    assert com.is_datetimelike_v_numeric(np.array([dt]), np.array([1]))


def test_needs_i8_conversion():
    assert not com.needs_i8_conversion(str)
    assert not com.needs_i8_conversion(np.int64)
    assert not com.needs_i8_conversion(pd.Series([1, 2]))
    assert not com.needs_i8_conversion(np.array(["a", "b"]))

    assert com.needs_i8_conversion(np.datetime64)
    assert com.needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    assert com.needs_i8_conversion(pd.DatetimeIndex(["2000"], tz="US/Eastern"))


def test_is_numeric_dtype():
    assert not com.is_numeric_dtype(str)
    assert not com.is_numeric_dtype(np.datetime64)
    assert not com.is_numeric_dtype(np.timedelta64)
    assert not com.is_numeric_dtype(np.array(["a", "b"]))
    assert not com.is_numeric_dtype(np.array([], dtype=np.timedelta64))

    assert com.is_numeric_dtype(int)
    assert com.is_numeric_dtype(float)
    assert com.is_numeric_dtype(np.uint64)
    assert com.is_numeric_dtype(pd.Series([1, 2]))
    assert com.is_numeric_dtype(pd.Index([1, 2.0]))


def test_is_float_dtype():
    assert not com.is_float_dtype(str)
    assert not com.is_float_dtype(int)
    assert not com.is_float_dtype(pd.Series([1, 2]))
    assert not com.is_float_dtype(np.array(["a", "b"]))

    assert com.is_float_dtype(float)
    assert com.is_float_dtype(pd.Index([1, 2.0]))


def test_is_bool_dtype():
    assert not com.is_bool_dtype(int)
    assert not com.is_bool_dtype(str)
    assert not com.is_bool_dtype(pd.Series([1, 2]))
    assert not com.is_bool_dtype(np.array(["a", "b"]))
    assert not com.is_bool_dtype(pd.Index(["a", "b"]))
    assert not com.is_bool_dtype("Int64")

    assert com.is_bool_dtype(bool)
    assert com.is_bool_dtype(np.bool_)
    assert com.is_bool_dtype(np.array([True, False]))
    assert com.is_bool_dtype(pd.Index([True, False]))

    assert com.is_bool_dtype(pd.BooleanDtype())
    assert com.is_bool_dtype(pd.array([True, False, None], dtype="boolean"))
    assert com.is_bool_dtype("boolean")


def test_is_bool_dtype_numpy_error():
    # GH39010
    assert not com.is_bool_dtype("0 - Name")


@pytest.mark.filterwarnings("ignore:'is_extension_type' is deprecated:FutureWarning")
@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no_scipy)]
)
def test_is_extension_type(check_scipy):
    assert not com.is_extension_type([1, 2, 3])
    assert not com.is_extension_type(np.array([1, 2, 3]))
    assert not com.is_extension_type(pd.DatetimeIndex([1, 2, 3]))

    cat = pd.Categorical([1, 2, 3])
    assert com.is_extension_type(cat)
    assert com.is_extension_type(pd.Series(cat))
    assert com.is_extension_type(SparseArray([1, 2, 3]))
    assert com.is_extension_type(pd.DatetimeIndex(["2000"], tz="US/Eastern"))

    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_extension_type(s)

    if check_scipy:
        import scipy.sparse

        assert not com.is_extension_type(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_extension_type_deprecation():
    with tm.assert_produces_warning(FutureWarning):
        com.is_extension_type([1, 2, 3])


@pytest.mark.parametrize(
    "check_scipy", [False, pytest.param(True, marks=td.skip_if_no_scipy)]
)
def test_is_extension_array_dtype(check_scipy):
    assert not com.is_extension_array_dtype([1, 2, 3])
    assert not com.is_extension_array_dtype(np.array([1, 2, 3]))
    assert not com.is_extension_array_dtype(pd.DatetimeIndex([1, 2, 3]))

    cat = pd.Categorical([1, 2, 3])
    assert com.is_extension_array_dtype(cat)
    assert com.is_extension_array_dtype(pd.Series(cat))
    assert com.is_extension_array_dtype(SparseArray([1, 2, 3]))
    assert com.is_extension_array_dtype(pd.DatetimeIndex(["2000"], tz="US/Eastern"))

    dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    s = pd.Series([], dtype=dtype)
    assert com.is_extension_array_dtype(s)

    if check_scipy:
        import scipy.sparse

        assert not com.is_extension_array_dtype(scipy.sparse.bsr_matrix([1, 2, 3]))


def test_is_complex_dtype():
    assert not com.is_complex_dtype(int)
    assert not com.is_complex_dtype(str)
    assert not com.is_complex_dtype(pd.Series([1, 2]))
    assert not com.is_complex_dtype(np.array(["a", "b"]))

    assert com.is_complex_dtype(np.complex_)
    assert com.is_complex_dtype(complex)
    assert com.is_complex_dtype(np.array([1 + 1j, 5]))


@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int)),
        ("int32", np.dtype("int32")),
        (float, np.dtype(float)),
        ("float64", np.dtype("float64")),
        (np.dtype("float64"), np.dtype("float64")),
        (str, np.dtype(str)),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.dtype("int16")),
        (pd.Series(["a", "b"]), np.dtype(object)),
        (pd.Index([1, 2]), np.dtype("int64")),
        (pd.Index(["a", "b"]), np.dtype(object)),
        ("category", "category"),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        (pd.Categorical(["a", "b"]), CategoricalDtype(["a", "b"])),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtype(["a", "b"])),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtype(["a", "b"])),
        (CategoricalDtype(), CategoricalDtype()),
        (pd.DatetimeIndex([1, 2]), np.dtype("=M8[ns]")),
        (pd.DatetimeIndex([1, 2]).dtype, np.dtype("=M8[ns]")),
        ("<M8[ns]", np.dtype("<M8[ns]")),
        ("datetime64[ns, Europe/London]", DatetimeTZDtype("ns", "Europe/London")),
        (PeriodDtype(freq="D"), PeriodDtype(freq="D")),
        ("period[D]", PeriodDtype(freq="D")),
        (IntervalDtype(), IntervalDtype()),
    ],
)
def test_get_dtype(input_param, result):
    assert com.get_dtype(input_param) == result


@pytest.mark.parametrize(
    "input_param,expected_error_message",
    [
        (None, "Cannot deduce dtype from null object"),
        (1, "data type not understood"),
        (1.2, "data type not understood"),
        # numpy dev changed from double-quotes to single quotes
        ("random string", "data type [\"']random string[\"'] not understood"),
        (pd.DataFrame([1, 2]), "data type not understood"),
    ],
)
def test_get_dtype_fails(input_param, expected_error_message):
    # python objects
    # 2020-02-02 npdev changed error message
    expected_error_message += f"|Cannot interpret '{input_param}' as a data type"
    with pytest.raises(TypeError, match=expected_error_message):
        com.get_dtype(input_param)


@pytest.mark.parametrize(
    "input_param,result",
    [
        (int, np.dtype(int).type),
        ("int32", np.int32),
        (float, np.dtype(float).type),
        ("float64", np.float64),
        (np.dtype("float64"), np.float64),
        (str, np.dtype(str).type),
        (pd.Series([1, 2], dtype=np.dtype("int16")), np.int16),
        (pd.Series(["a", "b"]), np.object_),
        (pd.Index([1, 2], dtype="int64"), np.int64),
        (pd.Index(["a", "b"]), np.object_),
        ("category", CategoricalDtypeType),
        (pd.Categorical(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.Categorical(["a", "b"]), CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]).dtype, CategoricalDtypeType),
        (pd.CategoricalIndex(["a", "b"]), CategoricalDtypeType),
        (pd.DatetimeIndex([1, 2]), np.datetime64),
        (pd.DatetimeIndex([1, 2]).dtype, np.datetime64),
        ("<M8[ns]", np.datetime64),
        (pd.DatetimeIndex(["2000"], tz="Europe/London"), pd.Timestamp),
        (pd.DatetimeIndex(["2000"], tz="Europe/London").dtype, pd.Timestamp),
        ("datetime64[ns, Europe/London]", pd.Timestamp),
        (PeriodDtype(freq="D"), pd.Period),
        ("period[D]", pd.Period),
        (IntervalDtype(), pd.Interval),
        (None, type(None)),
        (1, type(None)),
        (1.2, type(None)),
        (pd.DataFrame([1, 2]), type(None)),  # composite dtype
    ],
)
def test__is_dtype_type(input_param, result):
    assert com._is_dtype_type(input_param, lambda tipo: tipo == result)


@pytest.mark.parametrize("val", [np.datetime64("NaT"), np.timedelta64("NaT")])
@pytest.mark.parametrize("typ", [np.int64])
def test_astype_nansafe(val, typ):
    arr = np.array([val])

    typ = np.dtype(typ)

    msg = "Cannot convert NaT values to integer"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning):
            # datetimelike astype(int64) deprecated
            astype_nansafe(arr, dtype=typ)


def test_astype_nansafe_copy_false(any_int_dtype):
    # GH#34457 use astype, not view
    arr = np.array([1, 2, 3], dtype=any_int_dtype)

    dtype = np.dtype("float64")
    result = astype_nansafe(arr, dtype, copy=False)

    expected = np.array([1.0, 2.0, 3.0], dtype=dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("from_type", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize(
    "to_type",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
        np.float16,
        np.float32,
    ],
)
def test_astype_datetime64_bad_dtype_raises(from_type, to_type):
    arr = np.array([from_type("2018")])

    to_type = np.dtype(to_type)

    with pytest.raises(TypeError, match="cannot astype"):
        astype_nansafe(arr, dtype=to_type)


@pytest.mark.parametrize("from_type", [np.datetime64, np.timedelta64])
def test_astype_object_preserves_datetime_na(from_type):
    arr = np.array([from_type("NaT", "ns")])
    result = astype_nansafe(arr, dtype=np.dtype("object"))

    assert isna(result)[0]


def test_validate_allhashable():
    assert com.validate_all_hashable(1, "a") is None

    with pytest.raises(TypeError, match="All elements must be hashable"):
        com.validate_all_hashable([])

    with pytest.raises(TypeError, match="list must be a hashable type"):
        com.validate_all_hashable([], error_name="list")
