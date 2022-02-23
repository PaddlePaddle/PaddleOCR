from __future__ import annotations

import functools
import itertools
import operator
from typing import (
    Any,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    NaTType,
    Timedelta,
    iNaT,
    lib,
)
from pandas._typing import (
    ArrayLike,
    Dtype,
    DtypeObj,
    F,
    Scalar,
    Shape,
)
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import np_percentile_argname

from pandas.core.dtypes.common import (
    get_dtype,
    is_any_int_dtype,
    is_bool_dtype,
    is_complex,
    is_datetime64_any_dtype,
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_timedelta64_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core.construction import extract_array

bn = import_optional_dependency("bottleneck", errors="warn")
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    # set/unset to use bottleneck
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    def __init__(self, *dtypes):
        super().__init__()
        self.dtypes = tuple(pandas_dtype(dtype).type for dtype in dtypes)

    def check(self, obj) -> bool:
        return hasattr(obj, "dtype") and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        @functools.wraps(f)
        def _f(*args, **kwargs):
            obj_iter = itertools.chain(args, kwargs.values())
            if any(self.check(obj) for obj in obj_iter):
                f_name = f.__name__.replace("nan", "")
                raise TypeError(
                    f"reduction operation '{f_name}' not allowed for this dtype"
                )
            try:
                with np.errstate(invalid="ignore"):
                    return f(*args, **kwargs)
            except ValueError as e:
                # we want to transform an object array
                # ValueError message to the more typical TypeError
                # e.g. this is normally a disallowed function on
                # object arrays that contain strings
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise

        return cast(F, _f)


class bottleneck_switch:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        bn_name = self.name or alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(
            values: np.ndarray,
            *,
            axis: int | None = None,
            skipna: bool = True,
            **kwds,
        ):
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v

            if values.size == 0 and kwds.get("min_count") is None:
                # We are empty, returning NA for our type
                # Only applies for the default `min_count` of None
                # since that affects how empty arrays are handled.
                # TODO(GH-18976) update all the nanops methods to
                # correctly handle empty inputs and remove this check.
                # It *may* just be `var`
                return _na_for_min_count(values, axis)

            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get("mask", None) is None:
                    # `mask` is not recognised by bottleneck, would raise
                    #  TypeError if called
                    kwds.pop("mask", None)
                    result = bn_func(values, axis=axis, **kwds)

                    # prefer to treat inf/-inf as NA, but must compute the func
                    # twice :(
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)

            return result

        return cast(F, f)


def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    # Bottleneck chokes on datetime64, PeriodDtype (or and EA)
    if not is_object_dtype(dtype) and not needs_i8_conversion(dtype):

        # GH 15507
        # bottleneck does not properly upcast during the sum
        # so can overflow

        # GH 9422
        # further we also want to preserve NaN when all elements
        # are NaN, unlike bottleneck/numpy which consider this
        # to be 0
        return name not in ["nansum", "nanprod"]
    return False


def _has_infs(result) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype == "f8":
            return lib.has_infs_f8(result.ravel("K"))
        elif result.dtype == "f4":
            return lib.has_infs_f4(result.ravel("K"))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        # if it doesn't support infs, then it can't have infs
        return False


def _get_fill_value(
    dtype: DtypeObj, fill_value: Scalar | None = None, fill_value_typ=None
):
    """return the correct fill value for the dtype of the values"""
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        else:
            if fill_value_typ == "+inf":
                return np.inf
            else:
                return -np.inf
    else:
        if fill_value_typ == "+inf":
            # need the max int here
            return lib.i8max
        else:
            return iNaT


def _maybe_get_mask(
    values: np.ndarray, skipna: bool, mask: np.ndarray | None
) -> np.ndarray | None:
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray]
    """
    if mask is None:
        if is_bool_dtype(values.dtype) or is_integer_dtype(values.dtype):
            # Boolean data cannot contain nulls, so signal via mask being None
            return None

        if skipna or needs_i8_conversion(values.dtype):
            mask = isna(values)

    return mask


def _get_values(
    values: np.ndarray,
    skipna: bool,
    fill_value: Any = None,
    fill_value_typ: str | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.dtype, np.dtype, Any]:
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    dtype : np.dtype
        dtype for values
    dtype_max : np.dtype
        platform independent dtype
    fill_value : Any
        fill value used
    """
    # In _get_values is only called from within nanops, and in all cases
    #  with scalar fill_value.  This guarantee is important for the
    #  np.where call below
    assert is_scalar(fill_value)
    # error: Incompatible types in assignment (expression has type "Union[Any,
    # Union[ExtensionArray, ndarray]]", variable has type "ndarray")
    values = extract_array(values, extract_numpy=True)  # type: ignore[assignment]

    mask = _maybe_get_mask(values, skipna, mask)

    dtype = values.dtype

    datetimelike = False
    if needs_i8_conversion(values.dtype):
        # changing timedelta64/datetime64 to int64 needs to happen after
        #  finding `mask` above
        values = np.asarray(values.view("i8"))
        datetimelike = True

    dtype_ok = _na_ok_dtype(dtype)

    # get our fill value (in case we need to provide an alternative
    # dtype for it)
    fill_value = _get_fill_value(
        dtype, fill_value=fill_value, fill_value_typ=fill_value_typ
    )

    if skipna and (mask is not None) and (fill_value is not None):
        if mask.any():
            if dtype_ok or datetimelike:
                values = values.copy()
                np.putmask(values, mask, fill_value)
            else:
                # np.where will promote if needed
                values = np.where(~mask, values, fill_value)

    # return a platform independent precision dtype
    dtype_max = dtype
    if is_integer_dtype(dtype) or is_bool_dtype(dtype):
        dtype_max = np.dtype(np.int64)
    elif is_float_dtype(dtype):
        dtype_max = np.dtype(np.float64)

    return values, mask, dtype, dtype_max, fill_value


def _na_ok_dtype(dtype: DtypeObj) -> bool:
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)


def _wrap_results(result, dtype: np.dtype, fill_value=None):
    """wrap our results if needed"""
    if result is NaT:
        pass

    elif is_datetime64_any_dtype(dtype):
        if fill_value is None:
            # GH#24293
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), "Expected non-null fill_value"
            if result == fill_value:
                result = np.nan

            if isna(result):
                result = np.datetime64("NaT", "ns")
            else:
                result = np.int64(result).view("datetime64[ns]")
        else:
            # If we have float dtype, taking a view will give the wrong result
            result = result.astype(dtype)
    elif is_timedelta64_dtype(dtype):
        if not isinstance(result, np.ndarray):
            if result == fill_value:
                result = np.nan

            # raise if we have a timedelta64[ns] which is too large
            if np.fabs(result) > lib.i8max:
                raise ValueError("overflow in timedelta operation")

            result = Timedelta(result, unit="ns")
        else:
            result = result.astype("m8[ns]").view(dtype)

    return result


def _datetimelike_compat(func: F) -> F:
    """
    If we have datetime64 or timedelta64 values, ensure we have a correct
    mask before calling the wrapped function, then cast back afterwards.
    """

    @functools.wraps(func)
    def new_func(
        values: np.ndarray,
        *,
        axis: int | None = None,
        skipna: bool = True,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        orig_values = values

        datetimelike = values.dtype.kind in ["m", "M"]
        if datetimelike and mask is None:
            mask = isna(values)

        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if not skipna:
                assert mask is not None  # checked above
                result = _mask_datetimelike_result(result, axis, mask, orig_values)

        return result

    return cast(F, new_func)


def _na_for_min_count(values: np.ndarray, axis: int | None) -> Scalar | np.ndarray:
    """
    Return the missing value for `values`.

    Parameters
    ----------
    values : ndarray
    axis : int or None
        axis for the reduction, required if values.ndim > 1.

    Returns
    -------
    result : scalar or ndarray
        For 1-D values, returns a scalar of the correct missing type.
        For 2-D values, returns a 1-D array where each element is missing.
    """
    # we either return np.nan or pd.NaT
    if is_numeric_dtype(values):
        values = values.astype("float64")
    fill_value = na_value_for_dtype(values.dtype)

    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1 :]

        return np.full(result_shape, fill_value, dtype=values.dtype)


def nanany(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s)
    True

    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s)
    False
    """
    values, _, _, _, _ = _get_values(values, skipna, fill_value=False, mask=mask)

    # For object type, any won't necessarily return
    # boolean values (numpy/numpy#4352)
    if is_object_dtype(values):
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.any(axis)  # type: ignore[return-value]


def nanall(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> bool:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s)
    True

    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s)
    False
    """
    values, _, _, _, _ = _get_values(values, skipna, fill_value=True, mask=mask)

    # For object type, all won't necessarily return
    # boolean values (numpy/numpy#4352)
    if is_object_dtype(values):
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.all(axis)  # type: ignore[return-value]


@disallow("M8")
@_datetimelike_compat
def nansum(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: np.ndarray | None = None,
) -> float:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s)
    3.0
    """
    values, mask, dtype, dtype_max, _ = _get_values(
        values, skipna, fill_value=0, mask=mask
    )
    dtype_sum = dtype_max
    if is_float_dtype(dtype):
        dtype_sum = dtype
    elif is_timedelta64_dtype(dtype):
        # error: Incompatible types in assignment (expression has type
        # "Type[float64]", variable has type "dtype")
        dtype_sum = np.float64  # type: ignore[assignment]

    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)

    return the_sum


def _mask_datetimelike_result(
    result: np.ndarray | np.datetime64 | np.timedelta64,
    axis: int | None,
    mask: np.ndarray,
    orig_values: np.ndarray,
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    if isinstance(result, np.ndarray):
        # we need to apply the mask
        result = result.astype("i8").view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        # error: Unsupported target for indexed assignment ("Union[ndarray[Any, Any],
        # datetime64, timedelta64]")
        result[axis_mask] = iNaT  # type: ignore[index]
    else:
        if mask.any():
            return NaT
    return result


@disallow(PeriodDtype)
@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s)
    1.5
    """
    values, mask, dtype, dtype_max, _ = _get_values(
        values, skipna, fill_value=0, mask=mask
    )
    dtype_sum = dtype_max
    dtype_count = np.dtype(np.float64)

    # not using needs_i8_conversion because that includes period
    if dtype.kind in ["m", "M"]:
        dtype_sum = np.dtype(np.float64)
    elif is_integer_dtype(dtype):
        dtype_sum = np.dtype(np.float64)
    elif is_float_dtype(dtype):
        dtype_sum = dtype
        dtype_count = dtype

    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = _ensure_numeric(values.sum(axis, dtype=dtype_sum))

    if axis is not None and getattr(the_sum, "ndim", False):
        count = cast(np.ndarray, count)
        with np.errstate(all="ignore"):
            # suppress division by zero warnings
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan

    return the_mean


@bottleneck_switch()
def nanmedian(values, *, axis=None, skipna=True, mask=None):
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s)
    2.0
    """

    def get_median(x):
        mask = notna(x)
        if not skipna and not mask.all():
            return np.nan
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about All-NaN slice
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            res = np.nanmedian(x[mask])
        return res

    values, mask, dtype, _, _ = _get_values(values, skipna, mask=mask)
    if not is_float_dtype(values.dtype):
        try:
            values = values.astype("f8")
        except ValueError as err:
            # e.g. "could not convert string to float: 'a'"
            raise TypeError(str(err)) from err
        if mask is not None:
            values[mask] = np.nan

    if axis is None:
        values = values.ravel("K")

    notempty = values.size

    # an array from a frame
    if values.ndim > 1:

        # there's a non-empty array to apply over otherwise numpy raises
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)

            else:
                # fastpath for the skipna case
                with warnings.catch_warnings():
                    # Suppress RuntimeWarning about All-NaN slice
                    warnings.filterwarnings("ignore", "All-NaN slice encountered")
                    res = np.nanmedian(values, axis)

        else:
            # must return the correct shape, but median is not defined for the
            # empty set so return nans of shape "everything but the passed axis"
            # since "axis" is where the reduction would occur if we had a nonempty
            # array
            res = get_empty_reduction_result(values.shape, axis, np.float_, np.nan)

    else:
        # otherwise return a scalar value
        res = get_median(values) if notempty else np.nan
    return _wrap_results(res, dtype)


def get_empty_reduction_result(
    shape: tuple[int, ...],
    axis: int,
    dtype: np.dtype | type[np.floating],
    fill_value: Any,
) -> np.ndarray:
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int]
    axis : int
    dtype : np.dtype
    fill_value : Any

    Returns
    -------
    np.ndarray
    """
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[dims != axis], dtype=dtype)
    ret.fill(fill_value)
    return ret


def _get_counts_nanvar(
    values_shape: Shape,
    mask: np.ndarray | None,
    axis: int | None,
    ddof: int,
    dtype: Dtype = float,
) -> tuple[int | np.ndarray, int | np.ndarray]:
    """
    Get the count of non-null values along an axis, accounting
    for degrees of freedom.

    Parameters
    ----------
    values_shape : Tuple[int, ...]
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    ddof : int
        degrees of freedom
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : scalar or array
    d : scalar or array
    """
    dtype = get_dtype(dtype)
    count = _get_counts(values_shape, mask, axis, dtype=dtype)
    d = count - dtype.type(ddof)

    # always return NaN, never inf
    if is_scalar(count):
        if count <= ddof:
            count = np.nan
            d = np.nan
    else:
        # error: Incompatible types in assignment (expression has type
        # "Union[bool, Any]", variable has type "ndarray")
        mask2: np.ndarray = count <= ddof  # type: ignore[assignment]
        if mask2.any():
            np.putmask(d, mask2, np.nan)
            np.putmask(count, mask2, np.nan)
    # error: Incompatible return value type (got "Tuple[Union[int, float,
    # ndarray], Any]", expected "Tuple[Union[int, ndarray], Union[int,
    # ndarray]]")
    return count, d  # type: ignore[return-value]


@bottleneck_switch(ddof=1)
def nanstd(values, *, axis=None, skipna=True, ddof=1, mask=None):
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s)
    1.0
    """
    if values.dtype == "M8[ns]":
        values = values.view("m8[ns]")

    orig_dtype = values.dtype
    values, mask, _, _, _ = _get_values(values, skipna, mask=mask)

    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)


@disallow("M8", "m8")
@bottleneck_switch(ddof=1)
def nanvar(values, *, axis=None, skipna=True, ddof=1, mask=None):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s)
    1.0
    """
    values = extract_array(values, extract_numpy=True)
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if is_any_int_dtype(dtype):
        values = values.astype("f8")
        if mask is not None:
            values[mask] = np.nan

    if is_float_dtype(values.dtype):
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    # xref GH10242
    # Compute variance via two-pass algorithm, which is stable against
    # cancellation errors and relatively accurate for small numbers of
    # observations.
    #
    # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)
    sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d

    # Return variance as np.float64 (the datatype used in the accumulator),
    # unless we were dealing with a float array, in which case use the same
    # precision as the original values array.
    if is_float_dtype(dtype):
        result = result.astype(dtype)
    return result


@disallow("M8", "m8")
def nansem(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s)
     0.5773502691896258
    """
    # This checks if non-numeric-like data is passed with numeric_only=False
    # and raises a TypeError otherwise
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    mask = _maybe_get_mask(values, skipna, mask)
    if not is_float_dtype(values.dtype):
        values = values.astype("f8")

    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof)

    return np.sqrt(var) / np.sqrt(count)


def _nanminmax(meth, fill_value_typ):
    @bottleneck_switch(name="nan" + meth)
    @_datetimelike_compat
    def reduction(
        values: np.ndarray,
        *,
        axis: int | None = None,
        skipna: bool = True,
        mask: np.ndarray | None = None,
    ) -> Dtype:

        values, mask, dtype, dtype_max, fill_value = _get_values(
            values, skipna, fill_value_typ=fill_value_typ, mask=mask
        )

        if (axis is not None and values.shape[axis] == 0) or values.size == 0:
            try:
                result = getattr(values, meth)(axis, dtype=dtype_max)
                result.fill(np.nan)
            except (AttributeError, TypeError, ValueError):
                result = np.nan
        else:
            result = getattr(values, meth)(axis)

        result = _maybe_null_out(result, axis, mask, values.shape)
        return result

    return reduction


nanmin = _nanminmax("min", fill_value_typ="+inf")
nanmax = _nanminmax("max", fill_value_typ="-inf")


@disallow("O")
def nanargmax(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    4

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
    values, mask, _, _, _ = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    # error: Need type annotation for 'result'
    result = values.argmax(axis)  # type: ignore[var-annotated]
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result


@disallow("O")
def nanargmin(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    0

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
    values, mask, _, _, _ = _get_values(values, True, fill_value_typ="+inf", mask=mask)
    # error: Need type annotation for 'result'
    result = values.argmin(axis)  # type: ignore[var-annotated]
    result = _maybe_arg_null_out(result, axis, mask, skipna)
    return result


@disallow("M8", "m8")
def nanskew(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s)
    1.7320508075688787
    """
    # error: Incompatible types in assignment (expression has type "Union[Any,
    # Union[ExtensionArray, ndarray]]", variable has type "ndarray")
    values = extract_array(values, extract_numpy=True)  # type: ignore[assignment]
    mask = _maybe_get_mask(values, skipna, mask)
    if not is_float_dtype(values.dtype):
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted ** 2
    adjusted3 = adjusted2 * adjusted
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)

    # floating point error
    #
    # #18044 in _libs/windows.pyx calc_skew follow this behavior
    # to fix the fperr to treat m2 <1e-14 as zero
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2 ** 1.5)

    dtype = values.dtype
    if is_float_dtype(dtype):
        result = result.astype(dtype)

    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)
        result[count < 3] = np.nan
    else:
        result = 0 if m2 == 0 else result
        if count < 3:
            return np.nan

    return result


@disallow("M8", "m8")
def nankurt(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s)
    -1.2892561983471076
    """
    # error: Incompatible types in assignment (expression has type "Union[Any,
    # Union[ExtensionArray, ndarray]]", variable has type "ndarray")
    values = extract_array(values, extract_numpy=True)  # type: ignore[assignment]
    mask = _maybe_get_mask(values, skipna, mask)
    if not is_float_dtype(values.dtype):
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted ** 2
    adjusted4 = adjusted2 ** 2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2 ** 2

    # floating point error
    #
    # #18044 in _libs/windows.pyx calc_kurt follow this behavior
    # to fix the fperr to treat denom <1e-14 as zero
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)

    if not isinstance(denominator, np.ndarray):
        # if ``denom`` is a scalar, check these corner cases first before
        # doing division
        if count < 4:
            return np.nan
        if denominator == 0:
            return 0

    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj

    dtype = values.dtype
    if is_float_dtype(dtype):
        result = result.astype(dtype)

    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan

    return result


@disallow("M8", "m8")
def nanprod(
    values: np.ndarray,
    *,
    axis: int | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: np.ndarray | None = None,
) -> float:
    """
    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)

    Examples
    --------
    >>> import pandas.core.nanops as nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s)
    6.0
    """
    mask = _maybe_get_mask(values, skipna, mask)

    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    # error: Incompatible return value type (got "Union[ndarray, float]", expected
    # "float")
    return _maybe_null_out(  # type: ignore[return-value]
        result, axis, mask, values.shape, min_count=min_count
    )


def _maybe_arg_null_out(
    result: np.ndarray, axis: int | None, mask: np.ndarray | None, skipna: bool
) -> np.ndarray | int:
    # helper function for nanargmin/nanargmax
    if mask is None:
        return result

    if axis is None or not getattr(result, "ndim", False):
        if skipna:
            if mask.all():
                # error: Incompatible types in assignment (expression has type
                # "int", variable has type "ndarray")
                result = -1  # type: ignore[assignment]
        else:
            if mask.any():
                # error: Incompatible types in assignment (expression has type
                # "int", variable has type "ndarray")
                result = -1  # type: ignore[assignment]
    else:
        if skipna:
            na_mask = mask.all(axis)
        else:
            na_mask = mask.any(axis)
        if na_mask.any():
            result[na_mask] = -1
    return result


def _get_counts(
    values_shape: tuple[int, ...],
    mask: np.ndarray | None,
    axis: int | None,
    dtype: Dtype = float,
) -> int | float | np.ndarray:
    """
    Get the count of non-null values along an axis

    Parameters
    ----------
    values_shape : tuple of int
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : scalar or array
    """
    dtype = get_dtype(dtype)
    if axis is None:
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            n = np.prod(values_shape)
        return dtype.type(n)

    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]

    if is_scalar(count):
        return dtype.type(count)
    try:
        return count.astype(dtype)
    except AttributeError:
        # error: Argument "dtype" to "array" has incompatible type
        # "Union[ExtensionDtype, dtype]"; expected "Union[dtype, None, type,
        # _SupportsDtype, str, Tuple[Any, int], Tuple[Any, Union[int,
        # Sequence[int]]], List[Any], _DtypeDict, Tuple[Any, Any]]"
        return np.array(count, dtype=dtype)  # type: ignore[arg-type]


def _maybe_null_out(
    result: np.ndarray | float | NaTType,
    axis: int | None,
    mask: np.ndarray | None,
    shape: tuple[int, ...],
    min_count: int = 1,
) -> np.ndarray | float | NaTType:
    """
    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)
    """
    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        else:
            # we have no nulls, kept mask=None in _maybe_get_mask
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1 :]
            null_mask = np.broadcast_to(below_count, new_shape)

        if np.any(null_mask):
            if is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype("c16")
                else:
                    result = result.astype("f8")
                result[null_mask] = np.nan
            else:
                # GH12941, use None to auto cast null
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result = np.nan

    return result


def check_below_min_count(
    shape: tuple[int, ...], mask: np.ndarray | None, min_count: int
) -> bool:
    """
    Check for the `min_count` keyword. Returns True if below `min_count` (when
    missing value should be returned from the reduction).

    Parameters
    ----------
    shape : tuple
        The shape of the values (`values.shape`).
    mask : ndarray or None
        Boolean numpy array (typically of same shape as `shape`) or None.
    min_count : int
        Keyword passed through from sum/prod call.

    Returns
    -------
    bool
    """
    if min_count > 0:
        if mask is None:
            # no missing values, only check size
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False


def _zero_out_fperr(arg):
    # #18044 reference this behavior to fix rolling skew/kurt issue
    if isinstance(arg, np.ndarray):
        with np.errstate(invalid="ignore"):
            return np.where(np.abs(arg) < 1e-14, 0, arg)
    else:
        return arg.dtype.type(0) if np.abs(arg) < 1e-14 else arg


@disallow("M8", "m8")
def nancorr(
    a: np.ndarray, b: np.ndarray, *, method="pearson", min_periods: int | None = None
):
    """
    a, b: ndarrays
    """
    if len(a) != len(b):
        raise AssertionError("Operands to nancorr must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    f = get_corr_func(method)
    return f(a, b)


def get_corr_func(method):
    if method == "kendall":
        from scipy.stats import kendalltau

        def func(a, b):
            return kendalltau(a, b)[0]

        return func
    elif method == "spearman":
        from scipy.stats import spearmanr

        def func(a, b):
            return spearmanr(a, b)[0]

        return func
    elif method == "pearson":

        def func(a, b):
            return np.corrcoef(a, b)[0, 1]

        return func
    elif callable(method):
        return method

    raise ValueError(
        f"Unknown method '{method}', expected one of "
        "'kendall', 'spearman', 'pearson', or callable"
    )


@disallow("M8", "m8")
def nancov(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_periods: int | None = None,
    ddof: int | None = 1,
):
    if len(a) != len(b):
        raise AssertionError("Operands to nancov must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    return np.cov(a, b, ddof=ddof)[0, 1]


def _ensure_numeric(x):
    if isinstance(x, np.ndarray):
        if is_integer_dtype(x) or is_bool_dtype(x):
            x = x.astype(np.float64)
        elif is_object_dtype(x):
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    # GH#29941 we get here with object arrays containing strs
                    raise TypeError(f"Could not convert {x} to numeric") from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        try:
            x = float(x)
        except (TypeError, ValueError):
            # e.g. "1+1j" or "foo"
            try:
                x = complex(x)
            except ValueError as err:
                # e.g. "foo"
                raise TypeError(f"Could not convert {x} to numeric") from err
    return x


# NA-friendly array comparisons


def make_nancomp(op):
    def f(x, y):
        xmask = isna(x)
        ymask = isna(y)
        mask = xmask | ymask

        with np.errstate(all="ignore"):
            result = op(x, y)

        if mask.any():
            if is_bool_dtype(result):
                result = result.astype("O")
            np.putmask(result, mask, np.nan)

        return result

    return f


nangt = make_nancomp(operator.gt)
nange = make_nancomp(operator.ge)
nanlt = make_nancomp(operator.lt)
nanle = make_nancomp(operator.le)
naneq = make_nancomp(operator.eq)
nanne = make_nancomp(operator.ne)


def _nanpercentile_1d(
    values: np.ndarray, mask: np.ndarray, q: np.ndarray, na_value: Scalar, interpolation
) -> Scalar | np.ndarray:
    """
    Wrapper for np.percentile that skips missing values, specialized to
    1-dimensional case.

    Parameters
    ----------
    values : array over which to find quantiles
    mask : ndarray[bool]
        locations in values that should be considered missing
    q : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """
    # mask is Union[ExtensionArray, ndarray]
    values = values[~mask]

    if len(values) == 0:
        return np.array([na_value] * len(q), dtype=values.dtype)

    return np.percentile(values, q, **{np_percentile_argname: interpolation})


def nanpercentile(
    values: np.ndarray,
    q: np.ndarray,
    *,
    na_value,
    mask: np.ndarray,
    interpolation,
):
    """
    Wrapper for np.percentile that skips missing values.

    Parameters
    ----------
    values : np.ndarray[ndim=2]  over which to find quantiles
    q : np.ndarray[float64] of quantile indices to find
    na_value : scalar
        value to return for empty or all-null values
    mask : ndarray[bool]
        locations in values that should be considered missing
    interpolation : str

    Returns
    -------
    quantiles : scalar or array
    """

    if values.dtype.kind in ["m", "M"]:
        # need to cast to integer to avoid rounding errors in numpy
        result = nanpercentile(
            values.view("i8"),
            q=q,
            na_value=na_value.view("i8"),
            mask=mask,
            interpolation=interpolation,
        )

        # Note: we have to do `astype` and not view because in general we
        #  have float result at this point, not i8
        return result.astype(values.dtype)

    if not lib.is_scalar(mask) and mask.any():
        # Caller is responsible for ensuring mask shape match
        assert mask.shape == values.shape
        result = [
            _nanpercentile_1d(val, m, q, na_value, interpolation=interpolation)
            for (val, m) in zip(list(values), list(mask))
        ]
        result = np.array(result, dtype=values.dtype, copy=False).T
        return result
    else:
        return np.percentile(
            values, q, axis=1, **{np_percentile_argname: interpolation}
        )


def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    mask_a, mask_b = {
        np.cumprod: (1.0, np.nan),
        np.maximum.accumulate: (-np.inf, np.nan),
        np.cumsum: (0.0, np.nan),
        np.minimum.accumulate: (np.inf, np.nan),
    }[accum_func]

    # We will be applying this function to block values
    if values.dtype.kind in ["m", "M"]:
        # GH#30460, GH#29058
        # numpy 1.18 started sorting NaTs at the end instead of beginning,
        #  so we need to work around to maintain backwards-consistency.
        orig_dtype = values.dtype

        # We need to define mask before masking NaTs
        mask = isna(values)

        if accum_func == np.minimum.accumulate:
            # Note: the accum_func comparison fails as an "is" comparison
            y = values.view("i8")
            y[mask] = lib.i8max
            changed = True
        else:
            y = values
            changed = False

        result = accum_func(y.view("i8"), axis=0)
        if skipna:
            result[mask] = iNaT
        elif accum_func == np.minimum.accumulate:
            # Restore NaTs that we masked previously
            nz = (~np.asarray(mask)).nonzero()[0]
            if len(nz):
                # everything up to the first non-na entry stays NaT
                result[: nz[0]] = iNaT

        if changed:
            # restore NaT elements
            y[mask] = iNaT  # TODO: could try/finally for this?

        if isinstance(values.dtype, np.dtype):
            result = result.view(orig_dtype)
        else:
            # DatetimeArray/TimedeltaArray
            # TODO: have this case go through a DTA method?
            # For DatetimeTZDtype, view result as M8[ns]
            npdtype = orig_dtype if isinstance(orig_dtype, np.dtype) else "M8[ns]"
            # Item "type" of "Union[Type[ExtensionArray], Type[ndarray[Any, Any]]]"
            # has no attribute "_simple_new"
            result = type(values)._simple_new(  # type: ignore[union-attr]
                result.view(npdtype), dtype=orig_dtype
            )

    elif skipna and not issubclass(values.dtype.type, (np.integer, np.bool_)):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)

    return result
