"""
Routines for casting.
"""

from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
)
import functools
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Sized,
    TypeVar,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    Timedelta,
    Timestamp,
    conversion,
)
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas._typing import (
    ArrayLike,
    Dtype,
    DtypeObj,
    Scalar,
)
from pandas.errors import IntCastingNaNError
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    TD64NS_DTYPE,
    ensure_int8,
    ensure_int16,
    ensure_int32,
    ensure_int64,
    ensure_object,
    ensure_str,
    is_bool,
    is_bool_dtype,
    is_complex,
    is_complex_dtype,
    is_datetime64_dtype,
    is_datetime64tz_dtype,
    is_dtype_equal,
    is_extension_array_dtype,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
    notna,
)

if TYPE_CHECKING:
    from pandas.core.arrays import (
        DatetimeArray,
        ExtensionArray,
        IntervalArray,
        PeriodArray,
        TimedeltaArray,
    )

_int8_max = np.iinfo(np.int8).max
_int16_max = np.iinfo(np.int16).max
_int32_max = np.iinfo(np.int32).max
_int64_max = np.iinfo(np.int64).max

NumpyArrayT = TypeVar("NumpyArrayT", bound=np.ndarray)


def maybe_convert_platform(
    values: list | tuple | range | np.ndarray | ExtensionArray,
) -> ArrayLike:
    """try to do platform conversion, allow ndarray or list here"""
    arr: ArrayLike

    if isinstance(values, (list, tuple, range)):
        arr = construct_1d_object_array_from_listlike(values)
    else:
        # The caller is responsible for ensuring that we have np.ndarray
        #  or ExtensionArray here.
        arr = values

    if arr.dtype == object:
        arr = cast(np.ndarray, arr)
        arr = lib.maybe_convert_objects(arr)

    return arr


def is_nested_object(obj) -> bool:
    """
    return a boolean if we have a nested object, e.g. a Series with 1 or
    more Series elements

    This may not be necessarily be performant.

    """
    return bool(
        isinstance(obj, ABCSeries)
        and is_object_dtype(obj.dtype)
        and any(isinstance(v, ABCSeries) for v in obj._values)
    )


def maybe_box_datetimelike(value: Scalar, dtype: Dtype | None = None) -> Scalar:
    """
    Cast scalar to Timestamp or Timedelta if scalar is datetime-like
    and dtype is not object.

    Parameters
    ----------
    value : scalar
    dtype : Dtype, optional

    Returns
    -------
    scalar
    """
    if dtype == object:
        pass
    elif isinstance(value, (np.datetime64, datetime)):
        value = Timestamp(value)
    elif isinstance(value, (np.timedelta64, timedelta)):
        value = Timedelta(value)

    return value


def maybe_box_native(value: Scalar) -> Scalar:
    """
    If passed a scalar cast the scalar to a python native type.

    Parameters
    ----------
    value : scalar or Series

    Returns
    -------
    scalar or Series
    """
    if is_float(value):
        # error: Argument 1 to "float" has incompatible type
        # "Union[Union[str, int, float, bool], Union[Any, Timestamp, Timedelta, Any]]";
        # expected "Union[SupportsFloat, _SupportsIndex, str]"
        value = float(value)  # type: ignore[arg-type]
    elif is_integer(value):
        # error: Argument 1 to "int" has incompatible type
        # "Union[Union[str, int, float, bool], Union[Any, Timestamp, Timedelta, Any]]";
        # expected "Union[str, SupportsInt, _SupportsIndex, _SupportsTrunc]"
        value = int(value)  # type: ignore[arg-type]
    elif is_bool(value):
        value = bool(value)
    elif isinstance(value, (np.datetime64, np.timedelta64)):
        value = maybe_box_datetimelike(value)
    return value


def maybe_unbox_datetimelike(value: Scalar, dtype: DtypeObj) -> Scalar:
    """
    Convert a Timedelta or Timestamp to timedelta64 or datetime64 for setting
    into a numpy array.  Failing to unbox would risk dropping nanoseconds.

    Notes
    -----
    Caller is responsible for checking dtype.kind in ["m", "M"]
    """
    if is_valid_na_for_dtype(value, dtype):
        # GH#36541: can't fill array directly with pd.NaT
        # > np.empty(10, dtype="datetime64[64]").fill(pd.NaT)
        # ValueError: cannot convert float NaN to integer
        value = dtype.type("NaT", "ns")
    elif isinstance(value, Timestamp):
        if value.tz is None:
            value = value.to_datetime64()
        elif not isinstance(dtype, DatetimeTZDtype):
            raise TypeError("Cannot unbox tzaware Timestamp to tznaive dtype")
    elif isinstance(value, Timedelta):
        value = value.to_timedelta64()

    _disallow_mismatched_datetimelike(value, dtype)
    return value


def _disallow_mismatched_datetimelike(value, dtype: DtypeObj):
    """
    numpy allows np.array(dt64values, dtype="timedelta64[ns]") and
    vice-versa, but we do not want to allow this, so we need to
    check explicitly
    """
    vdtype = getattr(value, "dtype", None)
    if vdtype is None:
        return
    elif (vdtype.kind == "m" and dtype.kind == "M") or (
        vdtype.kind == "M" and dtype.kind == "m"
    ):
        raise TypeError(f"Cannot cast {repr(value)} to {dtype}")


def maybe_downcast_to_dtype(result: ArrayLike, dtype: str | np.dtype) -> ArrayLike:
    """
    try to cast to the specified dtype (e.g. convert back to bool/int
    or could be an astype of float64->float32
    """
    do_round = False

    if isinstance(dtype, str):
        if dtype == "infer":
            inferred_type = lib.infer_dtype(ensure_object(result), skipna=False)
            if inferred_type == "boolean":
                dtype = "bool"
            elif inferred_type == "integer":
                dtype = "int64"
            elif inferred_type == "datetime64":
                dtype = "datetime64[ns]"
            elif inferred_type == "timedelta64":
                dtype = "timedelta64[ns]"

            # try to upcast here
            elif inferred_type == "floating":
                dtype = "int64"
                if issubclass(result.dtype.type, np.number):
                    do_round = True

            else:
                # TODO: complex?  what if result is already non-object?
                dtype = "object"

        dtype = np.dtype(dtype)

    if not isinstance(dtype, np.dtype):
        # enforce our signature annotation
        raise TypeError(dtype)  # pragma: no cover

    converted = maybe_downcast_numeric(result, dtype, do_round)
    if converted is not result:
        return converted

    # a datetimelike
    # GH12821, iNaT is cast to float
    if dtype.kind in ["M", "m"] and result.dtype.kind in ["i", "f"]:
        result = result.astype(dtype)

    return result


def maybe_downcast_numeric(
    result: ArrayLike, dtype: DtypeObj, do_round: bool = False
) -> ArrayLike:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.

    Parameters
    ----------
    result : ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    do_round : bool

    Returns
    -------
    ndarray or ExtensionArray
    """
    if not isinstance(dtype, np.dtype) or not isinstance(result.dtype, np.dtype):
        # e.g. SparseDtype has no itemsize attr
        return result

    def trans(x):
        if do_round:
            return x.round()
        return x

    if dtype.kind == result.dtype.kind:
        # don't allow upcasts here (except if empty)
        if result.dtype.itemsize <= dtype.itemsize and result.size:
            return result

    if is_bool_dtype(dtype) or is_integer_dtype(dtype):

        if not result.size:
            # if we don't have any elements, just astype it
            return trans(result).astype(dtype)

        # do a test on the first element, if it fails then we are done
        r = result.ravel()
        arr = np.array([r[0]])

        if isna(arr).any():
            # if we have any nulls, then we are done
            return result

        elif not isinstance(r[0], (np.integer, np.floating, int, float, bool)):
            # a comparable, e.g. a Decimal may slip in here
            return result

        if (
            issubclass(result.dtype.type, (np.object_, np.number))
            and notna(result).all()
        ):
            new_result = trans(result).astype(dtype)
            if new_result.dtype.kind == "O" or result.dtype.kind == "O":
                # np.allclose may raise TypeError on object-dtype
                if (new_result == result).all():
                    return new_result
            else:
                if np.allclose(new_result, result, rtol=0):
                    return new_result

    elif (
        issubclass(dtype.type, np.floating)
        and not is_bool_dtype(result.dtype)
        and not is_string_dtype(result.dtype)
    ):
        return result.astype(dtype)

    return result


def maybe_cast_pointwise_result(
    result: ArrayLike,
    dtype: DtypeObj,
    numeric_only: bool = False,
    same_dtype: bool = True,
) -> ArrayLike:
    """
    Try casting result of a pointwise operation back to the original dtype if
    appropriate.

    Parameters
    ----------
    result : array-like
        Result to cast.
    dtype : np.dtype or ExtensionDtype
        Input Series from which result was calculated.
    numeric_only : bool, default False
        Whether to cast only numerics or datetimes as well.
    same_dtype : bool, default True
        Specify dtype when calling _from_sequence

    Returns
    -------
    result : array-like
        result maybe casted to the dtype.
    """

    assert not is_scalar(result)

    if isinstance(dtype, ExtensionDtype):
        if not isinstance(dtype, (CategoricalDtype, DatetimeTZDtype)):
            # TODO: avoid this special-casing
            # We have to special case categorical so as not to upcast
            # things like counts back to categorical

            cls = dtype.construct_array_type()
            if same_dtype:
                result = maybe_cast_to_extension_array(cls, result, dtype=dtype)
            else:
                result = maybe_cast_to_extension_array(cls, result)

    elif (numeric_only and is_numeric_dtype(dtype)) or not numeric_only:
        result = maybe_downcast_to_dtype(result, dtype)

    return result


def maybe_cast_to_extension_array(
    cls: type[ExtensionArray], obj: ArrayLike, dtype: ExtensionDtype | None = None
) -> ArrayLike:
    """
    Call to `_from_sequence` that returns the object unchanged on Exception.

    Parameters
    ----------
    cls : class, subclass of ExtensionArray
    obj : arraylike
        Values to pass to cls._from_sequence
    dtype : ExtensionDtype, optional

    Returns
    -------
    ExtensionArray or obj
    """
    from pandas.core.arrays.string_ import BaseStringArray

    assert isinstance(cls, type), f"must pass a type: {cls}"
    assertion_msg = f"must pass a subclass of ExtensionArray: {cls}"
    assert issubclass(cls, ABCExtensionArray), assertion_msg

    # Everything can be converted to StringArrays, but we may not want to convert
    if issubclass(cls, BaseStringArray) and lib.infer_dtype(obj) != "string":
        return obj

    try:
        result = cls._from_sequence(obj, dtype=dtype)
    except Exception:
        # We can't predict what downstream EA constructors may raise
        result = obj
    return result


@overload
def ensure_dtype_can_hold_na(dtype: np.dtype) -> np.dtype:
    ...


@overload
def ensure_dtype_can_hold_na(dtype: ExtensionDtype) -> ExtensionDtype:
    ...


def ensure_dtype_can_hold_na(dtype: DtypeObj) -> DtypeObj:
    """
    If we have a dtype that cannot hold NA values, find the best match that can.
    """
    if isinstance(dtype, ExtensionDtype):
        # TODO: ExtensionDtype.can_hold_na?
        return dtype
    elif dtype.kind == "b":
        return np.dtype(object)
    elif dtype.kind in ["i", "u"]:
        return np.dtype(np.float64)
    return dtype


def maybe_promote(dtype: np.dtype, fill_value=np.nan):
    """
    Find the minimal dtype that can hold both the given dtype and fill_value.

    Parameters
    ----------
    dtype : np.dtype
    fill_value : scalar, default np.nan

    Returns
    -------
    dtype
        Upcasted from dtype argument if necessary.
    fill_value
        Upcasted from fill_value argument if necessary.

    Raises
    ------
    ValueError
        If fill_value is a non-scalar and dtype is not object.
    """
    # TODO(2.0): need to directly use the non-cached version as long as we
    # possibly raise a deprecation warning for datetime dtype
    if dtype.kind == "M":
        return _maybe_promote(dtype, fill_value)
    # for performance, we are using a cached version of the actual implementation
    # of the function in _maybe_promote. However, this doesn't always work (in case
    # of non-hashable arguments), so we fallback to the actual implementation if needed
    try:
        # error: Argument 3 to "__call__" of "_lru_cache_wrapper" has incompatible type
        # "Type[Any]"; expected "Hashable"  [arg-type]
        return _maybe_promote_cached(
            dtype, fill_value, type(fill_value)  # type: ignore[arg-type]
        )
    except TypeError:
        # if fill_value is not hashable (required for caching)
        return _maybe_promote(dtype, fill_value)


@functools.lru_cache(maxsize=128)
def _maybe_promote_cached(dtype, fill_value, fill_value_type):
    # The cached version of _maybe_promote below
    # This also use fill_value_type as (unused) argument to use this in the
    # cache lookup -> to differentiate 1 and True
    return _maybe_promote(dtype, fill_value)


def _maybe_promote(dtype: np.dtype, fill_value=np.nan):
    # The actual implementation of the function, use `maybe_promote` above for
    # a cached version.
    if not is_scalar(fill_value):
        # with object dtype there is nothing to promote, and the user can
        #  pass pretty much any weird fill_value they like
        if not is_object_dtype(dtype):
            # with object dtype there is nothing to promote, and the user can
            #  pass pretty much any weird fill_value they like
            raise ValueError("fill_value must be a scalar")
        dtype = np.dtype(object)
        return dtype, fill_value

    kinds = ["i", "u", "f", "c", "m", "M"]
    if is_valid_na_for_dtype(fill_value, dtype) and dtype.kind in kinds:
        dtype = ensure_dtype_can_hold_na(dtype)
        fv = na_value_for_dtype(dtype)
        return dtype, fv

    elif isna(fill_value):
        dtype = np.dtype(object)
        if fill_value is None:
            # but we retain e.g. pd.NA
            fill_value = np.nan
        return dtype, fill_value

    # returns tuple of (dtype, fill_value)
    if issubclass(dtype.type, np.datetime64):
        inferred, fv = infer_dtype_from_scalar(fill_value, pandas_dtype=True)
        if inferred == dtype:
            return dtype, fv

        # TODO(2.0): once this deprecation is enforced, this whole case
        # becomes equivalent to:
        #  dta = DatetimeArray._from_sequence([], dtype="M8[ns]")
        #  try:
        #      fv = dta._validate_setitem_value(fill_value)
        #      return dta.dtype, fv
        #  except (ValueError, TypeError):
        #      return np.dtype(object), fill_value
        if isinstance(fill_value, date) and not isinstance(fill_value, datetime):
            # deprecate casting of date object to match infer_dtype_from_scalar
            #  and DatetimeArray._validate_setitem_value
            try:
                fv = Timestamp(fill_value).to_datetime64()
            except OutOfBoundsDatetime:
                pass
            else:
                warnings.warn(
                    "Using a `date` object for fill_value with `datetime64[ns]` "
                    "dtype is deprecated. In a future version, this will be cast "
                    "to object dtype. Pass `fill_value=Timestamp(date_obj)` instead.",
                    FutureWarning,
                    stacklevel=8,
                )
                return dtype, fv
        elif isinstance(fill_value, str):
            try:
                # explicitly wrap in str to convert np.str_
                fv = Timestamp(str(fill_value))
            except (ValueError, TypeError):
                pass
            else:
                if fv.tz is None:
                    return dtype, fv.asm8

        return np.dtype("object"), fill_value

    elif issubclass(dtype.type, np.timedelta64):
        inferred, fv = infer_dtype_from_scalar(fill_value, pandas_dtype=True)
        if inferred == dtype:
            return dtype, fv

        return np.dtype("object"), fill_value

    elif is_float(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, np.integer):
            dtype = np.dtype(np.float64)

        elif dtype.kind == "f":
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                # e.g. mst is np.float64 and dtype is np.float32
                dtype = mst

        elif dtype.kind == "c":
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)

    elif is_bool(fill_value):
        if not issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

    elif is_integer(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, np.integer):
            if not np.can_cast(fill_value, dtype):
                # upcast to prevent overflow
                mst = np.min_scalar_type(fill_value)
                dtype = np.promote_types(dtype, mst)
                if dtype.kind == "f":
                    # Case where we disagree with numpy
                    dtype = np.dtype(np.object_)

    elif is_complex(fill_value):
        if issubclass(dtype.type, np.bool_):
            dtype = np.dtype(np.object_)

        elif issubclass(dtype.type, (np.integer, np.floating)):
            mst = np.min_scalar_type(fill_value)
            dtype = np.promote_types(dtype, mst)

        elif dtype.kind == "c":
            mst = np.min_scalar_type(fill_value)
            if mst > dtype:
                # e.g. mst is np.complex128 and dtype is np.complex64
                dtype = mst

    else:
        dtype = np.dtype(np.object_)

    # in case we have a string that looked like a number
    if issubclass(dtype.type, (bytes, str)):
        dtype = np.dtype(np.object_)

    fill_value = _ensure_dtype_type(fill_value, dtype)
    return dtype, fill_value


def _ensure_dtype_type(value, dtype: np.dtype):
    """
    Ensure that the given value is an instance of the given dtype.

    e.g. if out dtype is np.complex64_, we should have an instance of that
    as opposed to a python complex object.

    Parameters
    ----------
    value : object
    dtype : np.dtype

    Returns
    -------
    object
    """
    # Start with exceptions in which we do _not_ cast to numpy types

    # error: Non-overlapping equality check (left operand type: "dtype[Any]", right
    # operand type: "Type[object_]")
    if dtype == np.object_:  # type: ignore[comparison-overlap]
        return value

    # Note: before we get here we have already excluded isna(value)
    return dtype.type(value)


def infer_dtype_from(val, pandas_dtype: bool = False) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar or array.

    Parameters
    ----------
    val : object
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, scalar/array belongs to pandas extension types is inferred as
        object
    """
    if not is_list_like(val):
        return infer_dtype_from_scalar(val, pandas_dtype=pandas_dtype)
    return infer_dtype_from_array(val, pandas_dtype=pandas_dtype)


def infer_dtype_from_scalar(val, pandas_dtype: bool = False) -> tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, scalar belongs to pandas extension types is inferred as
        object
    """
    dtype: DtypeObj = np.dtype(object)

    # a 1-element ndarray
    if isinstance(val, np.ndarray):
        if val.ndim != 0:
            msg = "invalid ndarray passed to infer_dtype_from_scalar"
            raise ValueError(msg)

        dtype = val.dtype
        val = lib.item_from_zerodim(val)

    elif isinstance(val, str):

        # If we create an empty array using a string to infer
        # the dtype, NumPy will only allocate one character per entry
        # so this is kind of bad. Alternately we could use np.repeat
        # instead of np.empty (but then you still don't want things
        # coming out as np.str_!

        dtype = np.dtype(object)

    elif isinstance(val, (np.datetime64, datetime)):
        try:
            val = Timestamp(val)
        except OutOfBoundsDatetime:
            return np.dtype(object), val

        # error: Non-overlapping identity check (left operand type: "Timestamp",
        # right operand type: "NaTType")
        if val is NaT or val.tz is None:  # type: ignore[comparison-overlap]
            dtype = np.dtype("M8[ns]")
            val = val.to_datetime64()
        else:
            if pandas_dtype:
                dtype = DatetimeTZDtype(unit="ns", tz=val.tz)
            else:
                # return datetimetz as object
                return np.dtype(object), val

    elif isinstance(val, (np.timedelta64, timedelta)):
        try:
            val = Timedelta(val)
        except (OutOfBoundsTimedelta, OverflowError):
            dtype = np.dtype(object)
        else:
            dtype = np.dtype("m8[ns]")
            val = np.timedelta64(val.value, "ns")

    elif is_bool(val):
        dtype = np.dtype(np.bool_)

    elif is_integer(val):
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)

        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype

    elif is_float(val):
        if isinstance(val, np.floating):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.float64)

    elif is_complex(val):
        dtype = np.dtype(np.complex_)

    elif pandas_dtype:
        if lib.is_period(val):
            dtype = PeriodDtype(freq=val.freq)
        elif lib.is_interval(val):
            subtype = infer_dtype_from_scalar(val.left, pandas_dtype=True)[0]
            dtype = IntervalDtype(subtype=subtype, closed=val.closed)

    return dtype, val


def dict_compat(d: dict[Scalar, Scalar]) -> dict[Scalar, Scalar]:
    """
    Convert datetimelike-keyed dicts to a Timestamp-keyed dict.

    Parameters
    ----------
    d: dict-like object

    Returns
    -------
    dict
    """
    return {maybe_box_datetimelike(key): value for key, value in d.items()}


def infer_dtype_from_array(
    arr, pandas_dtype: bool = False
) -> tuple[DtypeObj, ArrayLike]:
    """
    Infer the dtype from an array.

    Parameters
    ----------
    arr : array
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, array belongs to pandas extension types
        is inferred as object

    Returns
    -------
    tuple (numpy-compat/pandas-compat dtype, array)

    Notes
    -----
    if pandas_dtype=False. these infer to numpy dtypes
    exactly with the exception that mixed / object dtypes
    are not coerced by stringifying or conversion

    if pandas_dtype=True. datetime64tz-aware/categorical
    types will retain there character.

    Examples
    --------
    >>> np.asarray([1, '1'])
    array(['1', '1'], dtype='<U21')

    >>> infer_dtype_from_array([1, '1'])
    (dtype('O'), [1, '1'])
    """
    if isinstance(arr, np.ndarray):
        return arr.dtype, arr

    if not is_list_like(arr):
        raise TypeError("'arr' must be list-like")

    if pandas_dtype and is_extension_array_dtype(arr):
        return arr.dtype, arr

    elif isinstance(arr, ABCSeries):
        return arr.dtype, np.asarray(arr)

    # don't force numpy coerce with nan's
    inferred = lib.infer_dtype(arr, skipna=False)
    if inferred in ["string", "bytes", "mixed", "mixed-integer"]:
        return (np.dtype(np.object_), arr)

    arr = np.asarray(arr)
    return arr.dtype, arr


def maybe_infer_dtype_type(element):
    """
    Try to infer an object's dtype, for use in arithmetic ops.

    Uses `element.dtype` if that's available.
    Objects implementing the iterator protocol are cast to a NumPy array,
    and from there the array's type is used.

    Parameters
    ----------
    element : object
        Possibly has a `.dtype` attribute, and possibly the iterator
        protocol.

    Returns
    -------
    tipo : type

    Examples
    --------
    >>> from collections import namedtuple
    >>> Foo = namedtuple("Foo", "dtype")
    >>> maybe_infer_dtype_type(Foo(np.dtype("i8")))
    dtype('int64')
    """
    tipo = None
    if hasattr(element, "dtype"):
        tipo = element.dtype
    elif is_list_like(element):
        element = np.asarray(element)
        tipo = element.dtype
    return tipo


def maybe_upcast(
    values: NumpyArrayT,
    fill_value: Scalar = np.nan,
    copy: bool = False,
) -> tuple[NumpyArrayT, Scalar]:
    """
    Provide explicit type promotion and coercion.

    Parameters
    ----------
    values : np.ndarray
        The array that we may want to upcast.
    fill_value : what we want to fill with
    copy : bool, default True
        If True always make a copy even if no upcast is required.

    Returns
    -------
    values: np.ndarray
        the original array, possibly upcast
    fill_value:
        the fill value, possibly upcast
    """
    new_dtype, fill_value = maybe_promote(values.dtype, fill_value)
    # We get a copy in all cases _except_ (values.dtype == new_dtype and not copy)
    upcast_values = values.astype(new_dtype, copy=copy)

    return upcast_values, fill_value  # type: ignore[return-value]


def invalidate_string_dtypes(dtype_set: set[DtypeObj]):
    """
    Change string like dtypes to object for
    ``DataFrame.select_dtypes()``.
    """
    # error: Argument 1 to <set> has incompatible type "Type[generic]"; expected
    # "Union[dtype[Any], ExtensionDtype, None]"
    # error: Argument 2 to <set> has incompatible type "Type[generic]"; expected
    # "Union[dtype[Any], ExtensionDtype, None]"
    non_string_dtypes = dtype_set - {
        np.dtype("S").type,  # type: ignore[arg-type]
        np.dtype("<U").type,  # type: ignore[arg-type]
    }
    if non_string_dtypes != dtype_set:
        raise TypeError("string dtypes are not allowed, use 'object' instead")


def coerce_indexer_dtype(indexer, categories):
    """coerce the indexer input array to the smallest dtype possible"""
    length = len(categories)
    if length < _int8_max:
        return ensure_int8(indexer)
    elif length < _int16_max:
        return ensure_int16(indexer)
    elif length < _int32_max:
        return ensure_int32(indexer)
    return ensure_int64(indexer)


def astype_dt64_to_dt64tz(
    values: ArrayLike, dtype: DtypeObj, copy: bool, via_utc: bool = False
) -> DatetimeArray:
    # GH#33401 we have inconsistent behaviors between
    #  Datetimeindex[naive].astype(tzaware)
    #  Series[dt64].astype(tzaware)
    # This collects them in one place to prevent further fragmentation.

    from pandas.core.construction import ensure_wrapped_if_datetimelike

    values = ensure_wrapped_if_datetimelike(values)
    values = cast("DatetimeArray", values)
    aware = isinstance(dtype, DatetimeTZDtype)

    if via_utc:
        # Series.astype behavior

        # caller is responsible for checking this
        assert values.tz is None and aware
        dtype = cast(DatetimeTZDtype, dtype)

        if copy:
            # this should be the only copy
            values = values.copy()

        level = find_stack_level()
        warnings.warn(
            "Using .astype to convert from timezone-naive dtype to "
            "timezone-aware dtype is deprecated and will raise in a "
            "future version.  Use ser.dt.tz_localize instead.",
            FutureWarning,
            stacklevel=level,
        )

        # FIXME: GH#33401 this doesn't match DatetimeArray.astype, which
        #  goes through the `not via_utc` path
        return values.tz_localize("UTC").tz_convert(dtype.tz)

    else:
        # DatetimeArray/DatetimeIndex.astype behavior
        if values.tz is None and aware:
            dtype = cast(DatetimeTZDtype, dtype)
            level = find_stack_level()
            warnings.warn(
                "Using .astype to convert from timezone-naive dtype to "
                "timezone-aware dtype is deprecated and will raise in a "
                "future version.  Use obj.tz_localize instead.",
                FutureWarning,
                stacklevel=level,
            )

            return values.tz_localize(dtype.tz)

        elif aware:
            # GH#18951: datetime64_tz dtype but not equal means different tz
            dtype = cast(DatetimeTZDtype, dtype)
            result = values.tz_convert(dtype.tz)
            if copy:
                result = result.copy()
            return result

        elif values.tz is not None:
            level = find_stack_level()
            warnings.warn(
                "Using .astype to convert from timezone-aware dtype to "
                "timezone-naive dtype is deprecated and will raise in a "
                "future version.  Use obj.tz_localize(None) or "
                "obj.tz_convert('UTC').tz_localize(None) instead",
                FutureWarning,
                stacklevel=level,
            )

            result = values.tz_convert("UTC").tz_localize(None)
            if copy:
                result = result.copy()
            return result

        raise NotImplementedError("dtype_equal case should be handled elsewhere")


def astype_td64_unit_conversion(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    By pandas convention, converting to non-nano timedelta64
    returns an int64-dtyped array with ints representing multiples
    of the desired timedelta unit.  This is essentially division.

    Parameters
    ----------
    values : np.ndarray[timedelta64[ns]]
    dtype : np.dtype
        timedelta64 with unit not-necessarily nano
    copy : bool

    Returns
    -------
    np.ndarray
    """
    if is_dtype_equal(values.dtype, dtype):
        if copy:
            return values.copy()
        return values

    # otherwise we are converting to non-nano
    result = values.astype(dtype, copy=False)  # avoid double-copying
    result = result.astype(np.float64)

    mask = isna(values)
    np.putmask(result, mask, np.nan)
    return result


@overload
def astype_nansafe(
    arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...
) -> np.ndarray:
    ...


@overload
def astype_nansafe(
    arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...
) -> ExtensionArray:
    ...


def astype_nansafe(
    arr: np.ndarray, dtype: DtypeObj, copy: bool = True, skipna: bool = False
) -> ArrayLike:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if
        e.g. the item sizes don't align.
    skipna: bool, default False
        Whether or not we should skip NaN when casting as a string-type.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """
    if arr.ndim > 1:
        flat = arr.ravel()
        result = astype_nansafe(flat, dtype, copy=copy, skipna=skipna)
        # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
        # attribute "reshape"
        return result.reshape(arr.shape)  # type: ignore[union-attr]

    # We get here with 0-dim from sparse
    arr = np.atleast_1d(arr)

    # dispatch on extension dtype if needed
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)

    elif not isinstance(dtype, np.dtype):  # pragma: no cover
        raise ValueError("dtype must be np.dtype or ExtensionDtype")

    if arr.dtype.kind in ["m", "M"] and (
        issubclass(dtype.type, str)
        # error: Non-overlapping equality check (left operand type: "dtype[Any]", right
        # operand type: "Type[object]")
        or dtype == object  # type: ignore[comparison-overlap]
    ):
        from pandas.core.construction import ensure_wrapped_if_datetimelike

        arr = ensure_wrapped_if_datetimelike(arr)
        return arr.astype(dtype, copy=copy)

    if issubclass(dtype.type, str):
        return lib.ensure_string_array(arr, skipna=skipna, convert_na_value=False)

    elif is_datetime64_dtype(arr):
        # Non-overlapping equality check (left operand type: "dtype[Any]", right
        # operand type: "Type[signedinteger[Any]]")
        if dtype == np.int64:  # type: ignore[comparison-overlap]
            warnings.warn(
                f"casting {arr.dtype} values to int64 with .astype(...) "
                "is deprecated and will raise in a future version. "
                "Use .view(...) instead.",
                FutureWarning,
                # stacklevel chosen to be correct when reached via Series.astype
                stacklevel=7,
            )
            if isna(arr).any():
                raise ValueError("Cannot convert NaT values to integer")
            return arr.view(dtype)

        # allow frequency conversions
        if dtype.kind == "M":
            return arr.astype(dtype)

        raise TypeError(f"cannot astype a datetimelike from [{arr.dtype}] to [{dtype}]")

    elif is_timedelta64_dtype(arr):
        # error: Non-overlapping equality check (left operand type: "dtype[Any]", right
        # operand type: "Type[signedinteger[Any]]")
        if dtype == np.int64:  # type: ignore[comparison-overlap]
            warnings.warn(
                f"casting {arr.dtype} values to int64 with .astype(...) "
                "is deprecated and will raise in a future version. "
                "Use .view(...) instead.",
                FutureWarning,
                # stacklevel chosen to be correct when reached via Series.astype
                stacklevel=7,
            )
            if isna(arr).any():
                raise ValueError("Cannot convert NaT values to integer")
            return arr.view(dtype)

        elif dtype.kind == "m":
            return astype_td64_unit_conversion(arr, dtype, copy=copy)

        raise TypeError(f"cannot astype a timedelta from [{arr.dtype}] to [{dtype}]")

    elif np.issubdtype(arr.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        return astype_float_to_int_nansafe(arr, dtype, copy)

    elif is_object_dtype(arr):

        # work around NumPy brokenness, #1987
        if np.issubdtype(dtype.type, np.integer):
            return lib.astype_intsafe(arr, dtype)

        # if we have a datetime/timedelta array of objects
        # then coerce to a proper dtype and recall astype_nansafe

        elif is_datetime64_dtype(dtype):
            from pandas import to_datetime

            return astype_nansafe(
                to_datetime(arr).values,
                dtype,
                copy=copy,
            )
        elif is_timedelta64_dtype(dtype):
            from pandas import to_timedelta

            return astype_nansafe(to_timedelta(arr)._values, dtype, copy=copy)

    if dtype.name in ("datetime64", "timedelta64"):
        msg = (
            f"The '{dtype.name}' dtype has no unit. Please pass in "
            f"'{dtype.name}[ns]' instead."
        )
        raise ValueError(msg)

    if copy or is_object_dtype(arr.dtype) or is_object_dtype(dtype):
        # Explicit copy, or required since NumPy can't view from / to object.
        return arr.astype(dtype, copy=True)

    return arr.astype(dtype, copy=copy)


def astype_float_to_int_nansafe(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    astype with a check preventing converting NaN to an meaningless integer value.
    """
    if not np.isfinite(values).all():
        raise IntCastingNaNError(
            "Cannot convert non-finite values (NA or inf) to integer"
        )
    return values.astype(dtype, copy=copy)


def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = False) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
    if (
        values.dtype.kind in ["m", "M"]
        and dtype.kind in ["i", "u"]
        and isinstance(dtype, np.dtype)
        and dtype.itemsize != 8
    ):
        # TODO(2.0) remove special case once deprecation on DTA/TDA is enforced
        msg = rf"cannot astype a datetimelike from [{values.dtype}] to [{dtype}]"
        raise TypeError(msg)

    if is_datetime64tz_dtype(dtype) and is_datetime64_dtype(values.dtype):
        return astype_dt64_to_dt64tz(values, dtype, copy, via_utc=True)

    if is_dtype_equal(values.dtype, dtype):
        if copy:
            return values.copy()
        return values

    if not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        values = values.astype(dtype, copy=copy)

    else:
        values = astype_nansafe(values, dtype, copy=copy)

    # in pandas we don't store numpy str dtypes, so convert to object
    if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
        values = np.array(values, dtype=object)

    return values


def astype_array_safe(
    values: ArrayLike, dtype, copy: bool = False, errors: str = "raise"
) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
    errors_legal_values = ("raise", "ignore")

    if errors not in errors_legal_values:
        invalid_arg = (
            "Expected value of kwarg 'errors' to be one of "
            f"{list(errors_legal_values)}. Supplied value is '{errors}'"
        )
        raise ValueError(invalid_arg)

    if inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype):
        msg = (
            f"Expected an instance of {dtype.__name__}, "
            "but got the class instead. Try instantiating 'dtype'."
        )
        raise TypeError(msg)

    dtype = pandas_dtype(dtype)

    try:
        new_values = astype_array(values, dtype, copy=copy)
    except (ValueError, TypeError):
        # e.g. astype_nansafe can fail on object-dtype of strings
        #  trying to convert to float
        if errors == "ignore":
            new_values = values
        else:
            raise

    return new_values


def soft_convert_objects(
    values: np.ndarray,
    datetime: bool = True,
    numeric: bool = True,
    timedelta: bool = True,
    period: bool = True,
    copy: bool = True,
) -> ArrayLike:
    """
    Try to coerce datetime, timedelta, and numeric object-dtype columns
    to inferred dtype.

    Parameters
    ----------
    values : np.ndarray[object]
    datetime : bool, default True
    numeric: bool, default True
    timedelta : bool, default True
    period : bool, default True
    copy : bool, default True

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    validate_bool_kwarg(datetime, "datetime")
    validate_bool_kwarg(numeric, "numeric")
    validate_bool_kwarg(timedelta, "timedelta")
    validate_bool_kwarg(copy, "copy")

    conversion_count = sum((datetime, numeric, timedelta))
    if conversion_count == 0:
        raise ValueError("At least one of datetime, numeric or timedelta must be True.")

    # Soft conversions
    if datetime or timedelta:
        # GH 20380, when datetime is beyond year 2262, hence outside
        # bound of nanosecond-resolution 64-bit integers.
        try:
            converted = lib.maybe_convert_objects(
                values,
                convert_datetime=datetime,
                convert_timedelta=timedelta,
                convert_period=period,
            )
        except (OutOfBoundsDatetime, ValueError):
            return values
        if converted is not values:
            return converted

    if numeric and is_object_dtype(values.dtype):
        converted, _ = lib.maybe_convert_numeric(values, set(), coerce_numeric=True)

        # If all NaNs, then do not-alter
        values = converted if not isna(converted).all() else values
        values = values.copy() if copy else values

    return values


def convert_dtypes(
    input_array: ArrayLike,
    convert_string: bool = True,
    convert_integer: bool = True,
    convert_boolean: bool = True,
    convert_floating: bool = True,
) -> DtypeObj:
    """
    Convert objects to best possible type, and optionally,
    to types supporting ``pd.NA``.

    Parameters
    ----------
    input_array : ExtensionArray or np.ndarray
    convert_string : bool, default True
        Whether object dtypes should be converted to ``StringDtype()``.
    convert_integer : bool, default True
        Whether, if possible, conversion can be done to integer extension types.
    convert_boolean : bool, defaults True
        Whether object dtypes should be converted to ``BooleanDtypes()``.
    convert_floating : bool, defaults True
        Whether, if possible, conversion can be done to floating extension types.
        If `convert_integer` is also True, preference will be give to integer
        dtypes if the floats can be faithfully casted to integers.

    Returns
    -------
    np.dtype, or ExtensionDtype
    """
    inferred_dtype: str | DtypeObj

    if (
        convert_string or convert_integer or convert_boolean or convert_floating
    ) and isinstance(input_array, np.ndarray):

        if is_object_dtype(input_array.dtype):
            inferred_dtype = lib.infer_dtype(input_array)
        else:
            inferred_dtype = input_array.dtype

        if is_string_dtype(inferred_dtype):
            if not convert_string or inferred_dtype == "bytes":
                return input_array.dtype
            else:
                return pandas_dtype("string")

        if convert_integer:
            target_int_dtype = pandas_dtype("Int64")

            if is_integer_dtype(input_array.dtype):
                from pandas.core.arrays.integer import INT_STR_TO_DTYPE

                inferred_dtype = INT_STR_TO_DTYPE.get(
                    input_array.dtype.name, target_int_dtype
                )
            elif is_numeric_dtype(input_array.dtype):
                # TODO: de-dup with maybe_cast_to_integer_array?
                arr = input_array[notna(input_array)]
                if (arr.astype(int) == arr).all():
                    inferred_dtype = target_int_dtype
                else:
                    inferred_dtype = input_array.dtype

        if convert_floating:
            if not is_integer_dtype(input_array.dtype) and is_numeric_dtype(
                input_array.dtype
            ):
                from pandas.core.arrays.floating import FLOAT_STR_TO_DTYPE

                inferred_float_dtype: DtypeObj = FLOAT_STR_TO_DTYPE.get(
                    input_array.dtype.name, pandas_dtype("Float64")
                )
                # if we could also convert to integer, check if all floats
                # are actually integers
                if convert_integer:
                    # TODO: de-dup with maybe_cast_to_integer_array?
                    arr = input_array[notna(input_array)]
                    if (arr.astype(int) == arr).all():
                        inferred_dtype = pandas_dtype("Int64")
                    else:
                        inferred_dtype = inferred_float_dtype
                else:
                    inferred_dtype = inferred_float_dtype

        if convert_boolean:
            if is_bool_dtype(input_array.dtype):
                inferred_dtype = pandas_dtype("boolean")
            elif isinstance(inferred_dtype, str) and inferred_dtype == "boolean":
                inferred_dtype = pandas_dtype("boolean")

        if isinstance(inferred_dtype, str):
            # If we couldn't do anything else, then we retain the dtype
            inferred_dtype = input_array.dtype

    else:
        return input_array.dtype

    # error: Incompatible return value type (got "Union[str, Union[dtype[Any],
    # ExtensionDtype]]", expected "Union[dtype[Any], ExtensionDtype]")
    return inferred_dtype  # type: ignore[return-value]


def maybe_infer_to_datetimelike(
    value: np.ndarray,
) -> np.ndarray | DatetimeArray | TimedeltaArray | PeriodArray | IntervalArray:
    """
    we might have a array (or single object) that is datetime like,
    and no dtype is passed don't change the value unless we find a
    datetime/timedelta set

    this is pretty strict in that a datetime/timedelta is REQUIRED
    in addition to possible nulls/string likes

    Parameters
    ----------
    value : np.ndarray[object]

    Returns
    -------
    np.ndarray, DatetimeArray, TimedeltaArray, PeriodArray, or IntervalArray

    """
    if not isinstance(value, np.ndarray) or value.dtype != object:
        # Caller is responsible for passing only ndarray[object]
        raise TypeError(type(value))  # pragma: no cover

    v = np.array(value, copy=False)

    shape = v.shape
    if v.ndim != 1:
        v = v.ravel()

    if not len(v):
        return value

    def try_datetime(v: np.ndarray) -> ArrayLike:
        # Coerce to datetime64, datetime64tz, or in corner cases
        #  object[datetimes]
        from pandas.core.arrays.datetimes import sequence_to_datetimes

        try:
            # GH#19671 we pass require_iso8601 to be relatively strict
            #  when parsing strings.
            dta = sequence_to_datetimes(v, require_iso8601=True, allow_object=True)
        except (ValueError, TypeError):
            # e.g. <class 'numpy.timedelta64'> is not convertible to datetime
            return v.reshape(shape)
        else:
            # GH#19761 we may have mixed timezones, in which cast 'dta' is
            #  an ndarray[object].  Only 1 test
            #  relies on this behavior, see GH#40111
            return dta.reshape(shape)

    def try_timedelta(v: np.ndarray) -> np.ndarray:
        # safe coerce to timedelta64

        # will try first with a string & object conversion
        try:
            # bc we know v.dtype == object, this is equivalent to
            #  `np.asarray(to_timedelta(v))`, but using a lower-level API that
            #  does not require a circular import.
            td_values = array_to_timedelta64(v).view("m8[ns]")
        except (ValueError, OverflowError):
            return v.reshape(shape)
        else:
            return td_values.reshape(shape)

    inferred_type, seen_str = lib.infer_datetimelike_array(ensure_object(v))
    if inferred_type in ["period", "interval"]:
        # Incompatible return value type (got "Union[ExtensionArray, ndarray]",
        # expected "Union[ndarray, DatetimeArray, TimedeltaArray, PeriodArray,
        # IntervalArray]")
        return lib.maybe_convert_objects(  # type: ignore[return-value]
            v, convert_period=True, convert_interval=True
        )

    if inferred_type == "datetime":
        # error: Incompatible types in assignment (expression has type "ExtensionArray",
        # variable has type "Union[ndarray, List[Any]]")
        value = try_datetime(v)  # type: ignore[assignment]
    elif inferred_type == "timedelta":
        value = try_timedelta(v)
    elif inferred_type == "nat":

        # if all NaT, return as datetime
        if isna(v).all():
            # error: Incompatible types in assignment (expression has type
            # "ExtensionArray", variable has type "Union[ndarray, List[Any]]")
            value = try_datetime(v)  # type: ignore[assignment]
        else:

            # We have at least a NaT and a string
            # try timedelta first to avoid spurious datetime conversions
            # e.g. '00:00:01' is a timedelta but technically is also a datetime
            value = try_timedelta(v)
            if lib.infer_dtype(value, skipna=False) in ["mixed"]:
                # cannot skip missing values, as NaT implies that the string
                # is actually a datetime

                # error: Incompatible types in assignment (expression has type
                # "ExtensionArray", variable has type "Union[ndarray, List[Any]]")
                value = try_datetime(v)  # type: ignore[assignment]

    if value.dtype.kind in ["m", "M"] and seen_str:
        warnings.warn(
            f"Inferring {value.dtype} from data containing strings is deprecated "
            "and will be removed in a future version. To retain the old behavior "
            "explicitly pass Series(data, dtype={value.dtype})",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    return value


def maybe_cast_to_datetime(
    value: ExtensionArray | np.ndarray | list, dtype: DtypeObj | None
) -> ExtensionArray | np.ndarray:
    """
    try to cast the array/value to a datetimelike dtype, converting float
    nan to iNaT

    We allow a list *only* when dtype is not None.
    """
    from pandas.core.arrays.datetimes import sequence_to_datetimes
    from pandas.core.arrays.timedeltas import TimedeltaArray

    if not is_list_like(value):
        raise TypeError("value must be listlike")

    if is_timedelta64_dtype(dtype):
        # TODO: _from_sequence would raise ValueError in cases where
        #  ensure_nanosecond_dtype raises TypeError
        dtype = cast(np.dtype, dtype)
        dtype = ensure_nanosecond_dtype(dtype)
        res = TimedeltaArray._from_sequence(value, dtype=dtype)
        return res

    if dtype is not None:
        is_datetime64 = is_datetime64_dtype(dtype)
        is_datetime64tz = is_datetime64tz_dtype(dtype)

        vdtype = getattr(value, "dtype", None)

        if is_datetime64 or is_datetime64tz:
            dtype = ensure_nanosecond_dtype(dtype)

            value = np.array(value, copy=False)

            # we have an array of datetime or timedeltas & nulls
            if value.size or not is_dtype_equal(value.dtype, dtype):
                _disallow_mismatched_datetimelike(value, dtype)

                try:
                    if is_datetime64:
                        dta = sequence_to_datetimes(value, allow_object=False)
                        # GH 25843: Remove tz information since the dtype
                        # didn't specify one

                        if dta.tz is not None:
                            warnings.warn(
                                "Data is timezone-aware. Converting "
                                "timezone-aware data to timezone-naive by "
                                "passing dtype='datetime64[ns]' to "
                                "DataFrame or Series is deprecated and will "
                                "raise in a future version. Use "
                                "`pd.Series(values).dt.tz_localize(None)` "
                                "instead.",
                                FutureWarning,
                                stacklevel=8,
                            )
                            # equiv: dta.view(dtype)
                            # Note: NOT equivalent to dta.astype(dtype)
                            dta = dta.tz_localize(None)

                        value = dta
                    elif is_datetime64tz:
                        dtype = cast(DatetimeTZDtype, dtype)
                        # The string check can be removed once issue #13712
                        # is solved. String data that is passed with a
                        # datetime64tz is assumed to be naive which should
                        # be localized to the timezone.
                        is_dt_string = is_string_dtype(value.dtype)
                        dta = sequence_to_datetimes(value, allow_object=False)
                        if dta.tz is not None:
                            value = dta.astype(dtype, copy=False)
                        elif is_dt_string:
                            # Strings here are naive, so directly localize
                            # equiv: dta.astype(dtype)  # though deprecated

                            value = dta.tz_localize(dtype.tz)
                        else:
                            # Numeric values are UTC at this point,
                            # so localize and convert
                            # equiv: Series(dta).astype(dtype) # though deprecated
                            if getattr(vdtype, "kind", None) == "M":
                                # GH#24559, GH#33401 deprecate behavior inconsistent
                                #  with DatetimeArray/DatetimeIndex
                                warnings.warn(
                                    "In a future version, constructing a Series "
                                    "from datetime64[ns] data and a "
                                    "DatetimeTZDtype will interpret the data "
                                    "as wall-times instead of "
                                    "UTC times, matching the behavior of "
                                    "DatetimeIndex. To treat the data as UTC "
                                    "times, use pd.Series(data).dt"
                                    ".tz_localize('UTC').tz_convert(dtype.tz) "
                                    "or pd.Series(data.view('int64'), dtype=dtype)",
                                    FutureWarning,
                                    stacklevel=5,
                                )

                            value = dta.tz_localize("UTC").tz_convert(dtype.tz)
                except OutOfBoundsDatetime:
                    raise
                except ValueError:
                    # TODO(GH#40048): only catch dateutil's ParserError
                    #  once we can reliably import it in all supported versions
                    pass

        elif getattr(vdtype, "kind", None) in ["m", "M"]:
            # we are already datetimelike and want to coerce to non-datetimelike;
            #  astype_nansafe will raise for anything other than object, then upcast.
            #  see test_datetimelike_values_with_object_dtype
            # error: Argument 2 to "astype_nansafe" has incompatible type
            # "Union[dtype[Any], ExtensionDtype]"; expected "dtype[Any]"
            return astype_nansafe(value, dtype)  # type: ignore[arg-type]

    elif isinstance(value, np.ndarray):
        if value.dtype.kind in ["M", "m"]:
            # catch a datetime/timedelta that is not of ns variety
            # and no coercion specified
            value = sanitize_to_nanoseconds(value)

        elif value.dtype == object:
            value = maybe_infer_to_datetimelike(value)

    elif isinstance(value, list):
        # we only get here with dtype=None, which we do not allow
        raise ValueError(
            "maybe_cast_to_datetime allows a list *only* if dtype is not None"
        )

    # at this point we have converted or raised in all cases where we had a list
    return cast(ArrayLike, value)


def sanitize_to_nanoseconds(values: np.ndarray, copy: bool = False) -> np.ndarray:
    """
    Safely convert non-nanosecond datetime64 or timedelta64 values to nanosecond.
    """
    dtype = values.dtype
    if dtype.kind == "M" and dtype != DT64NS_DTYPE:
        values = conversion.ensure_datetime64ns(values)

    elif dtype.kind == "m" and dtype != TD64NS_DTYPE:
        values = conversion.ensure_timedelta64ns(values)

    elif copy:
        values = values.copy()

    return values


def ensure_nanosecond_dtype(dtype: DtypeObj) -> DtypeObj:
    """
    Convert dtypes with granularity less than nanosecond to nanosecond

    >>> ensure_nanosecond_dtype(np.dtype("M8[s]"))
    dtype('<M8[ns]')

    >>> ensure_nanosecond_dtype(np.dtype("m8[ps]"))
    Traceback (most recent call last):
        ...
    TypeError: cannot convert timedeltalike to dtype [timedelta64[ps]]
    """
    msg = (
        f"The '{dtype.name}' dtype has no unit. "
        f"Please pass in '{dtype.name}[ns]' instead."
    )

    # unpack e.g. SparseDtype
    dtype = getattr(dtype, "subtype", dtype)

    if not isinstance(dtype, np.dtype):
        # i.e. datetime64tz
        pass

    elif dtype.kind == "M" and dtype != DT64NS_DTYPE:
        # pandas supports dtype whose granularity is less than [ns]
        # e.g., [ps], [fs], [as]
        if dtype <= np.dtype("M8[ns]"):
            if dtype.name == "datetime64":
                raise ValueError(msg)
            dtype = DT64NS_DTYPE
        else:
            raise TypeError(f"cannot convert datetimelike to dtype [{dtype}]")

    elif dtype.kind == "m" and dtype != TD64NS_DTYPE:
        # pandas supports dtype whose granularity is less than [ns]
        # e.g., [ps], [fs], [as]
        if dtype <= np.dtype("m8[ns]"):
            if dtype.name == "timedelta64":
                raise ValueError(msg)
            dtype = TD64NS_DTYPE
        else:
            raise TypeError(f"cannot convert timedeltalike to dtype [{dtype}]")
    return dtype


def find_common_type(types: list[DtypeObj]) -> DtypeObj:
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : list of dtypes

    Returns
    -------
    pandas extension or numpy dtype

    See Also
    --------
    numpy.find_common_type

    """
    if not types:
        raise ValueError("no types given")

    first = types[0]

    # workaround for find_common_type([np.dtype('datetime64[ns]')] * 2)
    # => object
    if all(is_dtype_equal(first, t) for t in types[1:]):
        return first

    # get unique types (dict.fromkeys is used as order-preserving set())
    types = list(dict.fromkeys(types).keys())

    if any(isinstance(t, ExtensionDtype) for t in types):
        for t in types:
            if isinstance(t, ExtensionDtype):
                res = t._get_common_dtype(types)
                if res is not None:
                    return res
        return np.dtype("object")

    # take lowest unit
    if all(is_datetime64_dtype(t) for t in types):
        return np.dtype("datetime64[ns]")
    if all(is_timedelta64_dtype(t) for t in types):
        return np.dtype("timedelta64[ns]")

    # don't mix bool / int or float or complex
    # this is different from numpy, which casts bool with float/int as int
    has_bools = any(is_bool_dtype(t) for t in types)
    if has_bools:
        for t in types:
            if is_integer_dtype(t) or is_float_dtype(t) or is_complex_dtype(t):
                return np.dtype("object")

    # error: Argument 1 to "find_common_type" has incompatible type
    # "List[Union[dtype, ExtensionDtype]]"; expected "Sequence[Union[dtype,
    # None, type, _SupportsDtype, str, Tuple[Any, int], Tuple[Any, Union[int,
    # Sequence[int]]], List[Any], _DtypeDict, Tuple[Any, Any]]]"
    return np.find_common_type(types, [])  # type: ignore[arg-type]


def construct_2d_arraylike_from_scalar(
    value: Scalar, length: int, width: int, dtype: np.dtype, copy: bool
) -> np.ndarray:

    shape = (length, width)

    if dtype.kind in ["m", "M"]:
        value = maybe_unbox_datetimelike_tz_deprecation(value, dtype, stacklevel=4)
    # error: Non-overlapping equality check (left operand type: "dtype[Any]", right
    # operand type: "Type[object]")
    elif dtype == object:  # type: ignore[comparison-overlap]
        if isinstance(value, (np.timedelta64, np.datetime64)):
            # calling np.array below would cast to pytimedelta/pydatetime
            out = np.empty(shape, dtype=object)
            out.fill(value)
            return out

    # Attempt to coerce to a numpy array
    try:
        arr = np.array(value, dtype=dtype, copy=copy)
    except (ValueError, TypeError) as err:
        raise TypeError(
            f"DataFrame constructor called with incompatible data and dtype: {err}"
        ) from err

    if arr.ndim != 0:
        raise ValueError("DataFrame constructor not properly called!")

    return np.full(shape, arr)


def construct_1d_arraylike_from_scalar(
    value: Scalar, length: int, dtype: DtypeObj | None
) -> ArrayLike:
    """
    create a np.ndarray / pandas type of specified shape and dtype
    filled with values

    Parameters
    ----------
    value : scalar value
    length : int
    dtype : pandas_dtype or np.dtype

    Returns
    -------
    np.ndarray / pandas type of length, filled with value

    """

    if dtype is None:
        try:
            dtype, value = infer_dtype_from_scalar(value, pandas_dtype=True)
        except OutOfBoundsDatetime:
            dtype = np.dtype(object)

    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        subarr = cls._from_sequence([value] * length, dtype=dtype)

    else:

        if length and is_integer_dtype(dtype) and isna(value):
            # coerce if we have nan for an integer dtype
            dtype = np.dtype("float64")
        elif isinstance(dtype, np.dtype) and dtype.kind in ("U", "S"):
            # we need to coerce to object dtype to avoid
            # to allow numpy to take our string as a scalar value
            dtype = np.dtype("object")
            if not isna(value):
                value = ensure_str(value)
        elif dtype.kind in ["M", "m"]:
            value = maybe_unbox_datetimelike_tz_deprecation(value, dtype)

        subarr = np.empty(length, dtype=dtype)
        subarr.fill(value)

    return subarr


def maybe_unbox_datetimelike_tz_deprecation(
    value: Scalar, dtype: DtypeObj, stacklevel: int = 5
):
    """
    Wrap maybe_unbox_datetimelike with a check for a timezone-aware Timestamp
    along with a timezone-naive datetime64 dtype, which is deprecated.
    """
    # Caller is responsible for checking dtype.kind in ["m", "M"]

    if isinstance(value, datetime):
        # we dont want to box dt64, in particular datetime64("NaT")
        value = maybe_box_datetimelike(value, dtype)

    try:
        value = maybe_unbox_datetimelike(value, dtype)
    except TypeError:
        if (
            isinstance(value, Timestamp)
            and value.tzinfo is not None
            and isinstance(dtype, np.dtype)
            and dtype.kind == "M"
        ):
            warnings.warn(
                "Data is timezone-aware. Converting "
                "timezone-aware data to timezone-naive by "
                "passing dtype='datetime64[ns]' to "
                "DataFrame or Series is deprecated and will "
                "raise in a future version. Use "
                "`pd.Series(values).dt.tz_localize(None)` "
                "instead.",
                FutureWarning,
                stacklevel=stacklevel,
            )
            new_value = value.tz_localize(None)
            return maybe_unbox_datetimelike(new_value, dtype)
        else:
            raise
    return value


def construct_1d_object_array_from_listlike(values: Sized) -> np.ndarray:
    """
    Transform any list-like object in a 1-dimensional numpy array of object
    dtype.

    Parameters
    ----------
    values : any iterable which has a len()

    Raises
    ------
    TypeError
        * If `values` does not have a len()

    Returns
    -------
    1-dimensional numpy array of dtype object
    """
    # numpy will try to interpret nested lists as further dimensions, hence
    # making a 1D array that contains list-likes is a bit tricky:
    result = np.empty(len(values), dtype="object")
    result[:] = values
    return result


def maybe_cast_to_integer_array(
    arr: list | np.ndarray, dtype: np.dtype, copy: bool = False
) -> np.ndarray:
    """
    Takes any dtype and returns the casted version, raising for when data is
    incompatible with integer/unsigned integer dtypes.

    Parameters
    ----------
    arr : np.ndarray or list
        The array to cast.
    dtype : np.dtype
        The integer dtype to cast the array to.
    copy: bool, default False
        Whether to make a copy of the array before returning.

    Returns
    -------
    ndarray
        Array of integer or unsigned integer dtype.

    Raises
    ------
    OverflowError : the dtype is incompatible with the data
    ValueError : loss of precision has occurred during casting

    Examples
    --------
    If you try to coerce negative values to unsigned integers, it raises:

    >>> pd.Series([-1], dtype="uint64")
    Traceback (most recent call last):
        ...
    OverflowError: Trying to coerce negative values to unsigned integers

    Also, if you try to coerce float values to integers, it raises:

    >>> pd.Series([1, 2, 3.5], dtype="int64")
    Traceback (most recent call last):
        ...
    ValueError: Trying to coerce float values to integers
    """
    assert is_integer_dtype(dtype)

    try:
        if not isinstance(arr, np.ndarray):
            casted = np.array(arr, dtype=dtype, copy=copy)
        else:
            casted = arr.astype(dtype, copy=copy)
    except OverflowError as err:
        raise OverflowError(
            "The elements provided in the data cannot all be "
            f"casted to the dtype {dtype}"
        ) from err

    if np.array_equal(arr, casted):
        return casted

    # We do this casting to allow for proper
    # data and dtype checking.
    #
    # We didn't do this earlier because NumPy
    # doesn't handle `uint64` correctly.
    arr = np.asarray(arr)

    if is_unsigned_integer_dtype(dtype) and (arr < 0).any():
        raise OverflowError("Trying to coerce negative values to unsigned integers")

    if is_float_dtype(arr.dtype):
        if not np.isfinite(arr).all():
            raise IntCastingNaNError(
                "Cannot convert non-finite values (NA or inf) to integer"
            )
        raise ValueError("Trying to coerce float values to integers")
    if is_object_dtype(arr.dtype):
        raise ValueError("Trying to coerce float values to integers")

    if casted.dtype < arr.dtype:
        # GH#41734 e.g. [1, 200, 923442] and dtype="int8" -> overflows
        warnings.warn(
            f"Values are too large to be losslessly cast to {dtype}. "
            "In a future version this will raise OverflowError. To retain the "
            f"old behavior, use pd.Series(values).astype({dtype})",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return casted

    if arr.dtype.kind in ["m", "M"]:
        # test_constructor_maskedarray_nonfloat
        warnings.warn(
            f"Constructing Series or DataFrame from {arr.dtype} values and "
            f"dtype={dtype} is deprecated and will raise in a future version. "
            "Use values.view(dtype) instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return casted

    # No known cases that get here, but raising explicitly to cover our bases.
    raise ValueError(f"values cannot be losslessly cast to {dtype}")


def convert_scalar_for_putitemlike(scalar: Scalar, dtype: np.dtype) -> Scalar:
    """
    Convert datetimelike scalar if we are setting into a datetime64
    or timedelta64 ndarray.

    Parameters
    ----------
    scalar : scalar
    dtype : np.dtype

    Returns
    -------
    scalar
    """
    if dtype.kind in ["m", "M"]:
        scalar = maybe_box_datetimelike(scalar, dtype)
        return maybe_unbox_datetimelike(scalar, dtype)
    else:
        validate_numeric_casting(dtype, scalar)
    return scalar


def validate_numeric_casting(dtype: np.dtype, value: Scalar) -> None:
    """
    Check that we can losslessly insert the given value into an array
    with the given dtype.

    Parameters
    ----------
    dtype : np.dtype
    value : scalar

    Raises
    ------
    ValueError
    """
    # error: Argument 1 to "__call__" of "ufunc" has incompatible type
    # "Union[Union[str, int, float, bool], Union[Any, Timestamp, Timedelta, Any]]";
    # expected "Union[Union[int, float, complex, str, bytes, generic],
    # Sequence[Union[int, float, complex, str, bytes, generic]],
    # Sequence[Sequence[Any]], _SupportsArray]"
    if (
        issubclass(dtype.type, (np.integer, np.bool_))
        and is_float(value)
        and np.isnan(value)  # type: ignore[arg-type]
    ):
        raise ValueError("Cannot assign nan to integer series")

    elif dtype.kind in ["i", "u", "f", "c"]:
        if is_bool(value) or isinstance(value, np.timedelta64):
            # numpy will cast td64 to integer if we're not careful
            raise ValueError(
                f"Cannot assign {type(value).__name__} to float/integer series"
            )
    elif dtype.kind == "b":
        if is_scalar(value) and not is_bool(value):
            raise ValueError(f"Cannot assign {type(value).__name__} to bool series")


def can_hold_element(arr: ArrayLike, element: Any) -> bool:
    """
    Can we do an inplace setitem with this element in an array with this dtype?

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
    element : Any

    Returns
    -------
    bool
    """
    dtype = arr.dtype
    if not isinstance(dtype, np.dtype) or dtype.kind in ["m", "M"]:
        if isinstance(dtype, (PeriodDtype, IntervalDtype, DatetimeTZDtype, np.dtype)):
            # np.dtype here catches datetime64ns and timedelta64ns; we assume
            #  in this case that we have DatetimeArray/TimedeltaArray
            arr = cast(
                "PeriodArray | DatetimeArray | TimedeltaArray | IntervalArray", arr
            )
            try:
                arr._validate_setitem_value(element)
                return True
            except (ValueError, TypeError):
                return False

        # This is technically incorrect, but maintains the behavior of
        # ExtensionBlock._can_hold_element
        return True

    # error: Non-overlapping equality check (left operand type: "dtype[Any]", right
    # operand type: "Type[object]")
    if dtype == object:  # type: ignore[comparison-overlap]
        return True

    tipo = maybe_infer_dtype_type(element)

    if dtype.kind in ["i", "u"]:
        if tipo is not None:
            if tipo.kind not in ["i", "u"]:
                if is_float(element) and element.is_integer():
                    return True
                # Anything other than integer we cannot hold
                return False
            elif dtype.itemsize < tipo.itemsize:
                return False
            elif not isinstance(tipo, np.dtype):
                # i.e. nullable IntegerDtype; we can put this into an ndarray
                #  losslessly iff it has no NAs
                return not element._mask.any()
            return True

        # We have not inferred an integer from the dtype
        # check if we have a builtin int or a float equal to an int
        return is_integer(element) or (is_float(element) and element.is_integer())

    elif dtype.kind == "f":
        if tipo is not None:
            # TODO: itemsize check?
            if tipo.kind not in ["f", "i", "u"]:
                # Anything other than float/integer we cannot hold
                return False
            elif not isinstance(tipo, np.dtype):
                # i.e. nullable IntegerDtype or FloatingDtype;
                #  we can put this into an ndarray losslessly iff it has no NAs
                return not element._mask.any()
            return True

        return lib.is_integer(element) or lib.is_float(element)

    elif dtype.kind == "c":
        if tipo is not None:
            return tipo.kind in ["c", "f", "i", "u"]
        return (
            lib.is_integer(element) or lib.is_complex(element) or lib.is_float(element)
        )

    elif dtype.kind == "b":
        if tipo is not None:
            return tipo.kind == "b"
        return lib.is_bool(element)

    elif dtype.kind == "S":
        # TODO: test tests.frame.methods.test_replace tests get here,
        #  need more targeted tests.  xref phofl has a PR about this
        if tipo is not None:
            return tipo.kind == "S" and tipo.itemsize <= dtype.itemsize
        return isinstance(element, bytes) and len(element) <= dtype.itemsize

    raise NotImplementedError(dtype)
