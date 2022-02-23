"""
Common type operations.
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
)
import warnings

import numpy as np

from pandas._libs import (
    Interval,
    Period,
    algos,
)
from pandas._libs.tslibs import conversion
from pandas._typing import (
    ArrayLike,
    DtypeObj,
    Optional,
)

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategorical,
    ABCIndex,
)
from pandas.core.dtypes.inference import (  # noqa:F401
    is_array_like,
    is_bool,
    is_complex,
    is_dataclass,
    is_decimal,
    is_dict_like,
    is_file_like,
    is_float,
    is_hashable,
    is_integer,
    is_interval,
    is_iterator,
    is_list_like,
    is_named_tuple,
    is_nested_list_like,
    is_number,
    is_re,
    is_re_compilable,
    is_scalar,
    is_sequence,
)

DT64NS_DTYPE = conversion.DT64NS_DTYPE
TD64NS_DTYPE = conversion.TD64NS_DTYPE
INT64_DTYPE = np.dtype(np.int64)

# oh the troubles to reduce import time
_is_scipy_sparse = None

ensure_float64 = algos.ensure_float64
ensure_float32 = algos.ensure_float32


def ensure_float(arr):
    """
    Ensure that an array object has a float dtype if possible.

    Parameters
    ----------
    arr : array-like
        The array whose data type we want to enforce as float.

    Returns
    -------
    float_arr : The original array cast to the float dtype if
                possible. Otherwise, the original array is returned.
    """
    if is_extension_array_dtype(arr.dtype):
        if is_float_dtype(arr.dtype):
            arr = arr.to_numpy(dtype=arr.dtype.numpy_dtype, na_value=np.nan)
        else:
            arr = arr.to_numpy(dtype="float64", na_value=np.nan)
    elif issubclass(arr.dtype.type, (np.integer, np.bool_)):
        arr = arr.astype(float)
    return arr


ensure_uint64 = algos.ensure_uint64
ensure_int64 = algos.ensure_int64
ensure_int32 = algos.ensure_int32
ensure_int16 = algos.ensure_int16
ensure_int8 = algos.ensure_int8
ensure_platform_int = algos.ensure_platform_int
ensure_object = algos.ensure_object


def ensure_str(value: bytes | Any) -> str:
    """
    Ensure that bytes and non-strings get converted into ``str`` objects.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = str(value)
    return value


def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    Parameters
    ----------
    value: int or numpy.integer

    Returns
    -------
    int

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
    if not is_scalar(value):
        raise TypeError(
            f"Value needs to be a scalar value, was type {type(value).__name__}"
        )
    try:
        new_value = int(value)
        assert new_value == value
    except (TypeError, ValueError, AssertionError) as err:
        raise TypeError(f"Wrong type {type(value)} for value {value}") from err
    return new_value


def classes(*klasses) -> Callable:
    """evaluate if the tipo is a subclass of the klasses"""
    return lambda tipo: issubclass(tipo, klasses)


def classes_and_not_datetimelike(*klasses) -> Callable:
    """
    evaluate if the tipo is a subclass of the klasses
    and not a datetimelike
    """
    return lambda tipo: (
        issubclass(tipo, klasses)
        and not issubclass(tipo, (np.datetime64, np.timedelta64))
    )


def is_object_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the object dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the object dtype.

    Examples
    --------
    >>> is_object_dtype(object)
    True
    >>> is_object_dtype(int)
    False
    >>> is_object_dtype(np.array([], dtype=object))
    True
    >>> is_object_dtype(np.array([], dtype=int))
    False
    >>> is_object_dtype([1, 2, 3])
    False
    """
    return _is_dtype_type(arr_or_dtype, classes(np.object_))


def is_sparse(arr) -> bool:
    """
    Check whether an array-like is a 1-D pandas sparse array.

    Check that the one-dimensional array-like is a pandas sparse array.
    Returns True if it is a pandas sparse array, not another type of
    sparse array.

    Parameters
    ----------
    arr : array-like
        Array-like to check.

    Returns
    -------
    bool
        Whether or not the array-like is a pandas sparse array.

    Examples
    --------
    Returns `True` if the parameter is a 1-D pandas sparse array.

    >>> is_sparse(pd.arrays.SparseArray([0, 0, 1, 0]))
    True
    >>> is_sparse(pd.Series(pd.arrays.SparseArray([0, 0, 1, 0])))
    True

    Returns `False` if the parameter is not sparse.

    >>> is_sparse(np.array([0, 0, 1, 0]))
    False
    >>> is_sparse(pd.Series([0, 1, 0, 0]))
    False

    Returns `False` if the parameter is not a pandas sparse array.

    >>> from scipy.sparse import bsr_matrix
    >>> is_sparse(bsr_matrix([0, 1, 0, 0]))
    False

    Returns `False` if the parameter has more than one dimension.
    """
    from pandas.core.arrays.sparse import SparseDtype

    dtype = getattr(arr, "dtype", arr)
    return isinstance(dtype, SparseDtype)


def is_scipy_sparse(arr) -> bool:
    """
    Check whether an array-like is a scipy.sparse.spmatrix instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is a scipy.sparse.spmatrix instance.

    Notes
    -----
    If scipy is not installed, this function will always return False.

    Examples
    --------
    >>> from scipy.sparse import bsr_matrix
    >>> is_scipy_sparse(bsr_matrix([1, 2, 3]))
    True
    >>> is_scipy_sparse(pd.arrays.SparseArray([1, 2, 3]))
    False
    """
    global _is_scipy_sparse

    if _is_scipy_sparse is None:
        try:
            from scipy.sparse import issparse as _is_scipy_sparse
        except ImportError:
            _is_scipy_sparse = lambda _: False

    assert _is_scipy_sparse is not None
    return _is_scipy_sparse(arr)


def is_categorical(arr) -> bool:
    """
    Check whether an array-like is a Categorical instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is of a Categorical instance.

    Examples
    --------
    >>> is_categorical([1, 2, 3])
    False

    Categoricals, Series Categoricals, and CategoricalIndex will return True.

    >>> cat = pd.Categorical([1, 2, 3])
    >>> is_categorical(cat)
    True
    >>> is_categorical(pd.Series(cat))
    True
    >>> is_categorical(pd.CategoricalIndex([1, 2, 3]))
    True
    """
    warnings.warn(
        "is_categorical is deprecated and will be removed in a future version.  "
        "Use is_categorical_dtype instead",
        FutureWarning,
        stacklevel=2,
    )
    return isinstance(arr, ABCCategorical) or is_categorical_dtype(arr)


def is_datetime64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64))


def is_datetime64tz_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of a DatetimeTZDtype dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of a DatetimeTZDtype dtype.

    Examples
    --------
    >>> is_datetime64tz_dtype(object)
    False
    >>> is_datetime64tz_dtype([1, 2, 3])
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))  # tz-naive
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True

    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_datetime64tz_dtype(dtype)
    True
    >>> is_datetime64tz_dtype(s)
    True
    """
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"

    if arr_or_dtype is None:
        return False
    return DatetimeTZDtype.is_dtype(arr_or_dtype)


def is_timedelta64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the timedelta64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the timedelta64 dtype.

    Examples
    --------
    >>> is_timedelta64_dtype(object)
    False
    >>> is_timedelta64_dtype(np.timedelta64)
    True
    >>> is_timedelta64_dtype([1, 2, 3])
    False
    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> is_timedelta64_dtype('0 days')
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "m"

    return _is_dtype_type(arr_or_dtype, classes(np.timedelta64))


def is_period_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Period dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Period dtype.

    Examples
    --------
    >>> is_period_dtype(object)
    False
    >>> is_period_dtype(PeriodDtype(freq="D"))
    True
    >>> is_period_dtype([1, 2, 3])
    False
    >>> is_period_dtype(pd.Period("2017-01-01"))
    False
    >>> is_period_dtype(pd.PeriodIndex([], freq="A"))
    True
    """
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.type is Period

    if arr_or_dtype is None:
        return False
    return PeriodDtype.is_dtype(arr_or_dtype)


def is_interval_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Interval dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Interval dtype.

    Examples
    --------
    >>> is_interval_dtype(object)
    False
    >>> is_interval_dtype(IntervalDtype())
    True
    >>> is_interval_dtype([1, 2, 3])
    False
    >>>
    >>> interval = pd.Interval(1, 2, closed="right")
    >>> is_interval_dtype(interval)
    False
    >>> is_interval_dtype(pd.IntervalIndex([interval]))
    True
    """
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.type is Interval

    if arr_or_dtype is None:
        return False
    return IntervalDtype.is_dtype(arr_or_dtype)


def is_categorical_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Categorical dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Categorical dtype.

    Examples
    --------
    >>> is_categorical_dtype(object)
    False
    >>> is_categorical_dtype(CategoricalDtype())
    True
    >>> is_categorical_dtype([1, 2, 3])
    False
    >>> is_categorical_dtype(pd.Categorical([1, 2, 3]))
    True
    >>> is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))
    True
    """
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.name == "category"

    if arr_or_dtype is None:
        return False
    return CategoricalDtype.is_dtype(arr_or_dtype)


def is_string_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the string dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>>
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    """
    # TODO: gh-15585: consider making the checks stricter.
    def condition(dtype) -> bool:
        return dtype.kind in ("O", "S", "U") and not is_excluded_dtype(dtype)

    def is_excluded_dtype(dtype) -> bool:
        """
        These have kind = "O" but aren't string dtypes so need to be explicitly excluded
        """
        is_excluded_checks = (is_period_dtype, is_interval_dtype, is_categorical_dtype)
        return any(is_excluded(dtype) for is_excluded in is_excluded_checks)

    return _is_dtype(arr_or_dtype, condition)


def is_dtype_equal(source, target) -> bool:
    """
    Check if two dtypes are equal.

    Parameters
    ----------
    source : The first dtype to compare
    target : The second dtype to compare

    Returns
    -------
    boolean
        Whether or not the two dtypes are equal.

    Examples
    --------
    >>> is_dtype_equal(int, float)
    False
    >>> is_dtype_equal("int", int)
    True
    >>> is_dtype_equal(object, "category")
    False
    >>> is_dtype_equal(CategoricalDtype(), "category")
    True
    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")
    False
    """
    if isinstance(target, str):
        if not isinstance(source, str):
            # GH#38516 ensure we get the same behavior from
            #  is_dtype_equal(CDT, "category") and CDT == "category"
            try:
                src = get_dtype(source)
                if isinstance(src, ExtensionDtype):
                    return src == target
            except (TypeError, AttributeError):
                return False
    elif isinstance(source, str):
        return is_dtype_equal(target, source)

    try:
        source = get_dtype(source)
        target = get_dtype(target)
        return source == target
    except (TypeError, AttributeError):

        # invalid comparison
        # object == category will hit this
        return False


def is_any_int_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    In this function, timedelta64 instances are also considered "any-integer"
    type objects and will return True.

    This function is internal and should not be exposed in the public API.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype.

    Examples
    --------
    >>> is_any_int_dtype(str)
    False
    >>> is_any_int_dtype(int)
    True
    >>> is_any_int_dtype(float)
    False
    >>> is_any_int_dtype(np.uint64)
    True
    >>> is_any_int_dtype(np.datetime64)
    False
    >>> is_any_int_dtype(np.timedelta64)
    True
    >>> is_any_int_dtype(np.array(['a', 'b']))
    False
    >>> is_any_int_dtype(pd.Series([1, 2]))
    True
    >>> is_any_int_dtype(np.array([], dtype=np.timedelta64))
    True
    >>> is_any_int_dtype(pd.Index([1, 2.]))  # float
    False
    """
    return _is_dtype_type(arr_or_dtype, classes(np.integer, np.timedelta64))


def is_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    Unlike in `in_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
        not an instance of timedelta64.

    Examples
    --------
    >>> is_integer_dtype(str)
    False
    >>> is_integer_dtype(int)
    True
    >>> is_integer_dtype(float)
    False
    >>> is_integer_dtype(np.uint64)
    True
    >>> is_integer_dtype('int8')
    True
    >>> is_integer_dtype('Int8')
    True
    >>> is_integer_dtype(pd.Int8Dtype)
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(pd.Index([1, 2.]))  # float
    False
    """
    return _is_dtype_type(arr_or_dtype, classes_and_not_datetimelike(np.integer))


def is_signed_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a signed integer dtype.

    Unlike in `in_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a signed integer dtype
        and not an instance of timedelta64.

    Examples
    --------
    >>> is_signed_integer_dtype(str)
    False
    >>> is_signed_integer_dtype(int)
    True
    >>> is_signed_integer_dtype(float)
    False
    >>> is_signed_integer_dtype(np.uint64)  # unsigned
    False
    >>> is_signed_integer_dtype('int8')
    True
    >>> is_signed_integer_dtype('Int8')
    True
    >>> is_signed_integer_dtype(pd.Int8Dtype)
    True
    >>> is_signed_integer_dtype(np.datetime64)
    False
    >>> is_signed_integer_dtype(np.timedelta64)
    False
    >>> is_signed_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_signed_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_signed_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_signed_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_signed_integer_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    return _is_dtype_type(arr_or_dtype, classes_and_not_datetimelike(np.signedinteger))


def is_unsigned_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an unsigned integer dtype.

    The nullable Integer dtypes (e.g. pandas.UInt64Dtype) are also
    considered as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an unsigned integer dtype.

    Examples
    --------
    >>> is_unsigned_integer_dtype(str)
    False
    >>> is_unsigned_integer_dtype(int)  # signed
    False
    >>> is_unsigned_integer_dtype(float)
    False
    >>> is_unsigned_integer_dtype(np.uint64)
    True
    >>> is_unsigned_integer_dtype('uint8')
    True
    >>> is_unsigned_integer_dtype('UInt8')
    True
    >>> is_unsigned_integer_dtype(pd.UInt8Dtype)
    True
    >>> is_unsigned_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_unsigned_integer_dtype(pd.Series([1, 2]))  # signed
    False
    >>> is_unsigned_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_unsigned_integer_dtype(np.array([1, 2], dtype=np.uint32))
    True
    """
    return _is_dtype_type(
        arr_or_dtype, classes_and_not_datetimelike(np.unsignedinteger)
    )


def is_int64_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the int64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the int64 dtype.

    Notes
    -----
    Depending on system architecture, the return value of `is_int64_dtype(
    int)` will be True if the OS uses 64-bit integers and False if the OS
    uses 32-bit integers.

    Examples
    --------
    >>> is_int64_dtype(str)
    False
    >>> is_int64_dtype(np.int32)
    False
    >>> is_int64_dtype(np.int64)
    True
    >>> is_int64_dtype('int8')
    False
    >>> is_int64_dtype('Int8')
    False
    >>> is_int64_dtype(pd.Int64Dtype)
    True
    >>> is_int64_dtype(float)
    False
    >>> is_int64_dtype(np.uint64)  # unsigned
    False
    >>> is_int64_dtype(np.array(['a', 'b']))
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.int64))
    True
    >>> is_int64_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    return _is_dtype_type(arr_or_dtype, classes(np.int64))


def is_datetime64_any_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> is_datetime64_any_dtype(str)
    False
    >>> is_datetime64_any_dtype(int)
    False
    >>> is_datetime64_any_dtype(np.datetime64)  # can be tz-naive
    True
    >>> is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_any_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_any_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_any_dtype(np.array([], dtype="datetime64[ns]"))
    True
    >>> is_datetime64_any_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"

    if arr_or_dtype is None:
        return False
    return is_datetime64_dtype(arr_or_dtype) or is_datetime64tz_dtype(arr_or_dtype)


def is_datetime64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64[ns] dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64[ns] dtype.

    Examples
    --------
    >>> is_datetime64_ns_dtype(str)
    False
    >>> is_datetime64_ns_dtype(int)
    False
    >>> is_datetime64_ns_dtype(np.datetime64)  # no unit
    False
    >>> is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_ns_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_ns_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64"))  # no unit
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))  # wrong unit
    False
    >>> is_datetime64_ns_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    if arr_or_dtype is None:
        return False
    try:
        tipo = get_dtype(arr_or_dtype)
    except TypeError:
        if is_datetime64tz_dtype(arr_or_dtype):
            tipo = get_dtype(arr_or_dtype.dtype)
        else:
            return False
    return tipo == DT64NS_DTYPE or getattr(tipo, "base", None) == DT64NS_DTYPE


def is_timedelta64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the timedelta64[ns] dtype.

    This is a very specific dtype, so generic ones like `np.timedelta64`
    will return False if passed into this function.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the timedelta64[ns] dtype.

    Examples
    --------
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ps]'))  # Wrong frequency
    False
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))
    False
    """
    return _is_dtype(arr_or_dtype, lambda dtype: dtype == TD64NS_DTYPE)


def is_datetime_or_timedelta_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of
    a timedelta64 or datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a timedelta64,
        or datetime64 dtype.

    Examples
    --------
    >>> is_datetime_or_timedelta_dtype(str)
    False
    >>> is_datetime_or_timedelta_dtype(int)
    False
    >>> is_datetime_or_timedelta_dtype(np.datetime64)
    True
    >>> is_datetime_or_timedelta_dtype(np.timedelta64)
    True
    >>> is_datetime_or_timedelta_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime_or_timedelta_dtype(pd.Series([1, 2]))
    False
    >>> is_datetime_or_timedelta_dtype(np.array([], dtype=np.timedelta64))
    True
    >>> is_datetime_or_timedelta_dtype(np.array([], dtype=np.datetime64))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64, np.timedelta64))


# This exists to silence numpy deprecation warnings, see GH#29553
def is_numeric_v_string_like(a: ArrayLike, b):
    """
    Check if we are comparing a string-like object to a numeric ndarray.
    NumPy doesn't like to compare such objects, especially numeric arrays
    and scalar string-likes.

    Parameters
    ----------
    a : array-like
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a string-like object to a numeric array.

    Examples
    --------
    >>> is_numeric_v_string_like(np.array([1]), "foo")
    True
    >>> is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    True
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))
    True
    >>> is_numeric_v_string_like(np.array([1]), np.array([2]))
    False
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))
    False
    """
    is_a_array = isinstance(a, np.ndarray)
    is_b_array = isinstance(b, np.ndarray)

    is_a_numeric_array = is_a_array and a.dtype.kind in ("u", "i", "f", "c", "b")
    is_b_numeric_array = is_b_array and b.dtype.kind in ("u", "i", "f", "c", "b")
    is_a_string_array = is_a_array and a.dtype.kind in ("S", "U")
    is_b_string_array = is_b_array and b.dtype.kind in ("S", "U")

    is_b_scalar_string_like = not is_b_array and isinstance(b, str)

    return (
        (is_a_numeric_array and is_b_scalar_string_like)
        or (is_a_numeric_array and is_b_string_array)
        or (is_b_numeric_array and is_a_string_array)
    )


# This exists to silence numpy deprecation warnings, see GH#29553
def is_datetimelike_v_numeric(a, b):
    """
    Check if we are comparing a datetime-like object to a numeric object.
    By "numeric," we mean an object that is either of an int or float dtype.

    Parameters
    ----------
    a : array-like, scalar
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a datetime-like to a numeric object.

    Examples
    --------
    >>> from datetime import datetime
    >>> dt = np.datetime64(datetime(2017, 1, 1))
    >>>
    >>> is_datetimelike_v_numeric(1, 1)
    False
    >>> is_datetimelike_v_numeric(dt, dt)
    False
    >>> is_datetimelike_v_numeric(1, dt)
    True
    >>> is_datetimelike_v_numeric(dt, 1)  # symmetric check
    True
    >>> is_datetimelike_v_numeric(np.array([dt]), 1)
    True
    >>> is_datetimelike_v_numeric(np.array([1]), dt)
    True
    >>> is_datetimelike_v_numeric(np.array([dt]), np.array([1]))
    True
    >>> is_datetimelike_v_numeric(np.array([1]), np.array([2]))
    False
    >>> is_datetimelike_v_numeric(np.array([dt]), np.array([dt]))
    False
    """
    if not hasattr(a, "dtype"):
        a = np.asarray(a)
    if not hasattr(b, "dtype"):
        b = np.asarray(b)

    def is_numeric(x):
        """
        Check if an object has a numeric dtype (i.e. integer or float).
        """
        return is_integer_dtype(x) or is_float_dtype(x)

    return (needs_i8_conversion(a) and is_numeric(b)) or (
        needs_i8_conversion(b) and is_numeric(a)
    )


def needs_i8_conversion(arr_or_dtype) -> bool:
    """
    Check whether the array or dtype should be converted to int64.

    An array-like or dtype "needs" such a conversion if the array-like
    or dtype is of a datetime-like dtype

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype should be converted to int64.

    Examples
    --------
    >>> needs_i8_conversion(str)
    False
    >>> needs_i8_conversion(np.int64)
    False
    >>> needs_i8_conversion(np.datetime64)
    True
    >>> needs_i8_conversion(np.array(['a', 'b']))
    False
    >>> needs_i8_conversion(pd.Series([1, 2]))
    False
    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True
    """
    if arr_or_dtype is None:
        return False
    if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
        # fastpath
        dtype = arr_or_dtype
        return dtype.kind in ["m", "M"] or dtype.type is Period
    return (
        is_datetime_or_timedelta_dtype(arr_or_dtype)
        or is_datetime64tz_dtype(arr_or_dtype)
        or is_period_dtype(arr_or_dtype)
    )


def is_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a numeric dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a numeric dtype.

    Examples
    --------
    >>> is_numeric_dtype(str)
    False
    >>> is_numeric_dtype(int)
    True
    >>> is_numeric_dtype(float)
    True
    >>> is_numeric_dtype(np.uint64)
    True
    >>> is_numeric_dtype(np.datetime64)
    False
    >>> is_numeric_dtype(np.timedelta64)
    False
    >>> is_numeric_dtype(np.array(['a', 'b']))
    False
    >>> is_numeric_dtype(pd.Series([1, 2]))
    True
    >>> is_numeric_dtype(pd.Index([1, 2.]))
    True
    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))
    False
    """
    return _is_dtype_type(
        arr_or_dtype, classes_and_not_datetimelike(np.number, np.bool_)
    )


def is_float_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a float dtype.

    This function is internal and should not be exposed in the public API.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a float dtype.

    Examples
    --------
    >>> is_float_dtype(str)
    False
    >>> is_float_dtype(int)
    False
    >>> is_float_dtype(float)
    True
    >>> is_float_dtype(np.array(['a', 'b']))
    False
    >>> is_float_dtype(pd.Series([1, 2]))
    False
    >>> is_float_dtype(pd.Index([1, 2.]))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.floating))


def is_bool_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a boolean dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.

    Notes
    -----
    An ExtensionArray is considered boolean when the ``_is_boolean``
    attribute is set to True.

    Examples
    --------
    >>> is_bool_dtype(str)
    False
    >>> is_bool_dtype(int)
    False
    >>> is_bool_dtype(bool)
    True
    >>> is_bool_dtype(np.bool_)
    True
    >>> is_bool_dtype(np.array(['a', 'b']))
    False
    >>> is_bool_dtype(pd.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(pd.Categorical([True, False]))
    True
    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))
    True
    """
    if arr_or_dtype is None:
        return False
    try:
        dtype = get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False

    if isinstance(arr_or_dtype, CategoricalDtype):
        arr_or_dtype = arr_or_dtype.categories
        # now we use the special definition for Index

    if isinstance(arr_or_dtype, ABCIndex):

        # TODO(jreback)
        # we don't have a boolean Index class
        # so its object, we need to infer to
        # guess this
        return arr_or_dtype.is_object() and arr_or_dtype.inferred_type == "boolean"
    elif is_extension_array_dtype(arr_or_dtype):
        return getattr(dtype, "_is_boolean", False)

    return issubclass(dtype.type, np.bool_)


def is_extension_type(arr) -> bool:
    """
    Check whether an array-like is of a pandas extension class instance.

    .. deprecated:: 1.0.0
        Use ``is_extension_array_dtype`` instead.

    Extension classes include categoricals, pandas sparse objects (i.e.
    classes represented within the pandas library and not ones external
    to it like scipy sparse matrices), and datetime-like arrays.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is of a pandas extension class instance.

    Examples
    --------
    >>> is_extension_type([1, 2, 3])
    False
    >>> is_extension_type(np.array([1, 2, 3]))
    False
    >>>
    >>> cat = pd.Categorical([1, 2, 3])
    >>>
    >>> is_extension_type(cat)
    True
    >>> is_extension_type(pd.Series(cat))
    True
    >>> is_extension_type(pd.arrays.SparseArray([1, 2, 3]))
    True
    >>> from scipy.sparse import bsr_matrix
    >>> is_extension_type(bsr_matrix([1, 2, 3]))
    False
    >>> is_extension_type(pd.DatetimeIndex([1, 2, 3]))
    False
    >>> is_extension_type(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True
    >>>
    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_extension_type(s)
    True
    """
    warnings.warn(
        "'is_extension_type' is deprecated and will be removed in a future "
        "version.  Use 'is_extension_array_dtype' instead.",
        FutureWarning,
        stacklevel=2,
    )

    if is_categorical_dtype(arr):
        return True
    elif is_sparse(arr):
        return True
    elif is_datetime64tz_dtype(arr):
        return True
    return False


def is_1d_only_ea_obj(obj: Any) -> bool:
    """
    ExtensionArray that does not support 2D, or more specifically that does
    not use HybridBlock.
    """
    from pandas.core.arrays import (
        DatetimeArray,
        ExtensionArray,
        TimedeltaArray,
    )

    return isinstance(obj, ExtensionArray) and not isinstance(
        obj, (DatetimeArray, TimedeltaArray)
    )


def is_1d_only_ea_dtype(dtype: Optional[DtypeObj]) -> bool:
    """
    Analogue to is_extension_array_dtype but excluding DatetimeTZDtype.
    """
    # Note: if other EA dtypes are ever held in HybridBlock, exclude those
    #  here too.
    # NB: need to check DatetimeTZDtype and not is_datetime64tz_dtype
    #  to exclude ArrowTimestampUSDtype
    return isinstance(dtype, ExtensionDtype) and not isinstance(dtype, DatetimeTZDtype)


def is_extension_array_dtype(arr_or_dtype) -> bool:
    """
    Check if an object is a pandas extension array type.

    See the :ref:`Use Guide <extending.extension-types>` for more.

    Parameters
    ----------
    arr_or_dtype : object
        For array-like input, the ``.dtype`` attribute will
        be extracted.

    Returns
    -------
    bool
        Whether the `arr_or_dtype` is an extension array type.

    Notes
    -----
    This checks whether an object implements the pandas extension
    array interface. In pandas, this includes:

    * Categorical
    * Sparse
    * Interval
    * Period
    * DatetimeArray
    * TimedeltaArray

    Third-party libraries may implement arrays or types satisfying
    this interface as well.

    Examples
    --------
    >>> from pandas.api.types import is_extension_array_dtype
    >>> arr = pd.Categorical(['a', 'b'])
    >>> is_extension_array_dtype(arr)
    True
    >>> is_extension_array_dtype(arr.dtype)
    True

    >>> arr = np.array(['a', 'b'])
    >>> is_extension_array_dtype(arr.dtype)
    False
    """
    dtype = getattr(arr_or_dtype, "dtype", arr_or_dtype)
    if isinstance(dtype, ExtensionDtype):
        return True
    elif isinstance(dtype, np.dtype):
        return False
    else:
        return registry.find(dtype) is not None


def is_ea_or_datetimelike_dtype(dtype: Optional[DtypeObj]) -> bool:
    """
    Check for ExtensionDtype, datetime64 dtype, or timedelta64 dtype.

    Notes
    -----
    Checks only for dtype objects, not dtype-castable strings or types.
    """
    return isinstance(dtype, ExtensionDtype) or (
        isinstance(dtype, np.dtype) and dtype.kind in ["m", "M"]
    )


def is_complex_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a complex dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a complex dtype.

    Examples
    --------
    >>> is_complex_dtype(str)
    False
    >>> is_complex_dtype(int)
    False
    >>> is_complex_dtype(np.complex_)
    True
    >>> is_complex_dtype(np.array(['a', 'b']))
    False
    >>> is_complex_dtype(pd.Series([1, 2]))
    False
    >>> is_complex_dtype(np.array([1 + 1j, 5]))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.complexfloating))


def _is_dtype(arr_or_dtype, condition) -> bool:
    """
    Return a boolean if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like, str, np.dtype, or ExtensionArrayType
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtype]]

    Returns
    -------
    bool

    """
    if arr_or_dtype is None:
        return False
    try:
        dtype = get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False
    return condition(dtype)


def get_dtype(arr_or_dtype) -> DtypeObj:
    """
    Get the dtype instance associated with an array
    or dtype object.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype object whose dtype we want to extract.

    Returns
    -------
    obj_dtype : The extract dtype instance from the
                passed in array or dtype object.

    Raises
    ------
    TypeError : The passed in object is None.
    """
    if arr_or_dtype is None:
        raise TypeError("Cannot deduce dtype from null object")

    # fastpath
    elif isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype
    elif isinstance(arr_or_dtype, type):
        return np.dtype(arr_or_dtype)

    # if we have an array-like
    elif hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    return pandas_dtype(arr_or_dtype)


def _is_dtype_type(arr_or_dtype, condition) -> bool:
    """
    Return a boolean if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtypeType]]

    Returns
    -------
    bool : if the condition is satisfied for the arr_or_dtype
    """
    if arr_or_dtype is None:
        return condition(type(None))

    # fastpath
    if isinstance(arr_or_dtype, np.dtype):
        return condition(arr_or_dtype.type)
    elif isinstance(arr_or_dtype, type):
        if issubclass(arr_or_dtype, ExtensionDtype):
            arr_or_dtype = arr_or_dtype.type
        return condition(np.dtype(arr_or_dtype).type)

    # if we have an array-like
    if hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    # we are not possibly a dtype
    elif is_list_like(arr_or_dtype):
        return condition(type(None))

    try:
        tipo = pandas_dtype(arr_or_dtype).type
    except (TypeError, ValueError):
        if is_scalar(arr_or_dtype):
            return condition(type(None))

        return False

    return condition(tipo)


def infer_dtype_from_object(dtype) -> DtypeObj:
    """
    Get a numpy dtype.type-style object for a dtype object.

    This methods also includes handling of the datetime64[ns] and
    datetime64[ns, TZ] objects.

    If no dtype can be found, we return ``object``.

    Parameters
    ----------
    dtype : dtype, type
        The dtype object whose numpy dtype.type-style
        object we want to extract.

    Returns
    -------
    dtype_object : The extracted numpy dtype.type-style object.
    """
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        # Type object from a dtype

        # error: Incompatible return value type (got "Type[generic]", expected
        # "Union[dtype[Any], ExtensionDtype]")
        return dtype  # type: ignore[return-value]
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        # dtype object
        try:
            _validate_date_like_dtype(dtype)
        except TypeError:
            # Should still pass if we don't have a date-like
            pass
        # error: Incompatible return value type (got "Union[Type[generic], Type[Any]]",
        # expected "Union[dtype[Any], ExtensionDtype]")
        return dtype.type  # type: ignore[return-value]

    try:
        dtype = pandas_dtype(dtype)
    except TypeError:
        pass

    if is_extension_array_dtype(dtype):
        return dtype.type
    elif isinstance(dtype, str):

        # TODO(jreback)
        # should deprecate these
        if dtype in ["datetimetz", "datetime64tz"]:
            # error: Incompatible return value type (got "Type[Any]", expected
            # "Union[dtype[Any], ExtensionDtype]")
            return DatetimeTZDtype.type  # type: ignore[return-value]
        elif dtype in ["period"]:
            raise NotImplementedError

        if dtype in ["datetime", "timedelta"]:
            dtype += "64"
        try:
            return infer_dtype_from_object(getattr(np, dtype))
        except (AttributeError, TypeError):
            # Handles cases like get_dtype(int) i.e.,
            # Python objects that are valid dtypes
            # (unlike user-defined types, in general)
            #
            # TypeError handles the float16 type code of 'e'
            # further handle internal types
            pass

    return infer_dtype_from_object(np.dtype(dtype))


def _validate_date_like_dtype(dtype) -> None:
    """
    Check whether the dtype is a date-like dtype. Raises an error if invalid.

    Parameters
    ----------
    dtype : dtype, type
        The dtype to check.

    Raises
    ------
    TypeError : The dtype could not be casted to a date-like dtype.
    ValueError : The dtype is an illegal date-like dtype (e.g. the
                 frequency provided is too specific)
    """
    try:
        typ = np.datetime_data(dtype)[0]
    except ValueError as e:
        raise TypeError(e) from e
    if typ not in ["generic", "ns"]:
        raise ValueError(
            f"{repr(dtype.name)} is too specific of a frequency, "
            f"try passing {repr(dtype.type.__name__)}"
        )


def validate_all_hashable(*args, error_name: Optional[str] = None) -> None:
    """
    Return None if all args are hashable, else raise a TypeError.

    Parameters
    ----------
    *args
        Arguments to validate.
    error_name : str, optional
        The name to use if error

    Raises
    ------
    TypeError : If an argument is not hashable

    Returns
    -------
    None
    """
    if not all(is_hashable(arg) for arg in args):
        if error_name:
            raise TypeError(f"{error_name} must be a hashable type")
        else:
            raise TypeError("All elements must be hashable")


def pandas_dtype(dtype) -> DtypeObj:
    """
    Convert input into a pandas only dtype object or a numpy dtype object.

    Parameters
    ----------
    dtype : object to be converted

    Returns
    -------
    np.dtype or a pandas dtype

    Raises
    ------
    TypeError if not a dtype
    """
    # short-circuit
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype

    # registered extension types
    result = registry.find(dtype)
    if result is not None:
        # error: Incompatible return value type (got "Type[ExtensionDtype]",
        # expected "Union[dtype, ExtensionDtype]")
        return result  # type: ignore[return-value]

    # try a numpy dtype
    # raise a consistent TypeError if failed
    try:
        npdtype = np.dtype(dtype)
    except SyntaxError as err:
        # np.dtype uses `eval` which can raise SyntaxError
        raise TypeError(f"data type '{dtype}' not understood") from err

    # Any invalid dtype (such as pd.Timestamp) should raise an error.
    # np.dtype(invalid_type).kind = 0 for such objects. However, this will
    # also catch some valid dtypes such as object, np.object_ and 'object'
    # which we safeguard against by catching them earlier and returning
    # np.dtype(valid_dtype) before this condition is evaluated.
    if is_hashable(dtype) and dtype in [object, np.object_, "object", "O"]:
        # check hashability to avoid errors/DeprecationWarning when we get
        # here and `dtype` is an array
        return npdtype
    elif npdtype.kind == "O":
        raise TypeError(f"dtype '{dtype}' not understood")

    return npdtype
