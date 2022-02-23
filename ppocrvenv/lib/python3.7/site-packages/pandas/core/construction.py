"""
Constructor functions intended to be shared by pd.array, Series.__init__,
and Index.__new__.

These should not depend on core.internals.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    cast,
)
import warnings

import numpy as np
import numpy.ma as ma

from pandas._libs import lib
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    Dtype,
    DtypeObj,
)
from pandas.errors import IntCastingNaNError

from pandas.core.dtypes.base import (
    ExtensionDtype,
    _registry as registry,
)
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    construct_1d_object_array_from_listlike,
    maybe_cast_to_datetime,
    maybe_cast_to_integer_array,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
    maybe_upcast,
    sanitize_to_nanoseconds,
)
from pandas.core.dtypes.common import (
    is_datetime64_ns_dtype,
    is_extension_array_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_timedelta64_ns_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCPandasArray,
    ABCRangeIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

import pandas.core.common as com

if TYPE_CHECKING:
    from pandas import (
        ExtensionArray,
        Index,
        Series,
    )


def array(
    data: Sequence[object] | AnyArrayLike,
    dtype: Dtype | None = None,
    copy: bool = True,
) -> ExtensionArray:
    """
    Create an array.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It's expected that `data`
        represents a 1-dimensional array of data.

        When `data` is an Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array. This may be a NumPy
        dtype or an extension type registered with pandas using
        :meth:`pandas.api.extensions.register_extension_dtype`.

        If not specified, there are two possibilities:

        1. When `data` is a :class:`Series`, :class:`Index`, or
           :class:`ExtensionArray`, the `dtype` will be taken
           from the data.
        2. Otherwise, pandas will attempt to infer the `dtype`
           from the data.

        Note that when `data` is a NumPy array, ``data.dtype`` is
        *not* used for inferring the array type. This is because
        NumPy cannot represent all the types of data that can be
        held in extension arrays.

        Currently, pandas will infer an extension dtype for sequences of

        ============================== =======================================
        Scalar Type                    Array Type
        ============================== =======================================
        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        :class:`int`                   :class:`pandas.arrays.IntegerArray`
        :class:`float`                 :class:`pandas.arrays.FloatingArray`
        :class:`str`                   :class:`pandas.arrays.StringArray` or
                                       :class:`pandas.arrays.ArrowStringArray`
        :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        ============================== =======================================

        The ExtensionArray created when the scalar type is :class:`str` is determined by
        ``pd.options.mode.string_storage`` if the dtype is not explicitly given.

        For all other cases, NumPy's usual inference rules will be used.

        .. versionchanged:: 1.0.0

           Pandas infers nullable-integer dtype for integer data,
           string dtype for string data, and nullable-boolean dtype
           for boolean data.

        .. versionchanged:: 1.2.0

            Pandas now also infers nullable-floating dtype for float-like
            input data

    copy : bool, default True
        Whether to copy the data, even if not necessary. Depending
        on the type of `data`, creating the new array may require
        copying data, even if ``copy=False``.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.

    See Also
    --------
    numpy.array : Construct a NumPy array.
    Series : Construct a pandas Series.
    Index : Construct a pandas Index.
    arrays.PandasArray : ExtensionArray wrapping a NumPy array.
    Series.array : Extract the array stored within a Series.

    Notes
    -----
    Omitting the `dtype` argument means pandas will attempt to infer the
    best array type from the values in the data. As new array types are
    added by pandas and 3rd party libraries, the "best" array type may
    change. We recommend specifying `dtype` to ensure that

    1. the correct array type for the data is returned
    2. the returned array type doesn't change as new extension types
       are added by pandas and third-party libraries

    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.PandasArray` backed by a NumPy
    array.

    >>> pd.array(['a', 'b'], dtype=str)
    <PandasArray>
    ['a', 'b']
    Length: 2, dtype: str32

    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

    >>> pd.array(['a', 'b'], dtype=np.dtype("<U1"))
    <PandasArray>
    ['a', 'b']
    Length: 2, dtype: str32

    Finally, Pandas has arrays that mostly overlap with NumPy

      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``PandasArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

    >>> pd.array(['2015', '2016'], dtype='datetime64[ns]')
    <DatetimeArray>
    ['2015-01-01 00:00:00', '2016-01-01 00:00:00']
    Length: 2, dtype: datetime64[ns]

    >>> pd.array(["1H", "2H"], dtype='timedelta64[ns]')
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]

    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

    >>> pd.array(["a", None, "c"])
    <StringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    ...
    >>> arr
    <ArrowStringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> pd.array([pd.Period('2000', freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    ['2000-01-01', '2000-01-01']
    Length: 2, dtype: period[D]

    You can use the string alias for `dtype`

    >>> pd.array(['a', 'b', 'a'], dtype='category')
    ['a', 'b', 'a']
    Categories (2, object): ['a', 'b']

    Or specify the actual dtype

    >>> pd.array(['a', 'b', 'a'],
    ...          dtype=pd.CategoricalDtype(['a', 'b', 'c'], ordered=True))
    ['a', 'b', 'a']
    Categories (3, object): ['a' < 'b' < 'c']

    If pandas does not infer a dedicated extension type a
    :class:`arrays.PandasArray` is returned.

    >>> pd.array([1 + 1j, 3 + 2j])
    <PandasArray>
    [(1+1j), (3+2j)]
    Length: 2, dtype: complex128

    As mentioned in the "Notes" section, new extension types may be added
    in the future (by pandas or 3rd party libraries), causing the return
    value to no longer be a :class:`arrays.PandasArray`. Specify the `dtype`
    as a NumPy dtype if you need to ensure there's no future change in
    behavior.

    >>> pd.array([1, 2], dtype=np.dtype("int32"))
    <PandasArray>
    [1, 2]
    Length: 2, dtype: int32

    `data` must be 1-dimensional. A ValueError is raised when the input
    has the wrong dimensionality.

    >>> pd.array(1)
    Traceback (most recent call last):
      ...
    ValueError: Cannot pass scalar '1' to 'pandas.array'.
    """
    from pandas.core.arrays import (
        BooleanArray,
        DatetimeArray,
        FloatingArray,
        IntegerArray,
        IntervalArray,
        PandasArray,
        PeriodArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.string_ import StringDtype

    if lib.is_scalar(data):
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)

    if dtype is None and isinstance(data, (ABCSeries, ABCIndex, ABCExtensionArray)):
        # Note: we exclude np.ndarray here, will do type inference on it
        dtype = data.dtype

    data = extract_array(data, extract_numpy=True)

    # this returns None for not-found dtypes.
    if isinstance(dtype, str):
        dtype = registry.find(dtype) or dtype

    if is_extension_array_dtype(dtype):
        cls = cast(ExtensionDtype, dtype).construct_array_type()
        return cls._from_sequence(data, dtype=dtype, copy=copy)

    if dtype is None:
        inferred_dtype = lib.infer_dtype(data, skipna=True)
        if inferred_dtype == "period":
            return PeriodArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "interval":
            return IntervalArray(data, copy=copy)

        elif inferred_dtype.startswith("datetime"):
            # datetime, datetime64
            try:
                return DatetimeArray._from_sequence(data, copy=copy)
            except ValueError:
                # Mixture of timezones, fall back to PandasArray
                pass

        elif inferred_dtype.startswith("timedelta"):
            # timedelta, timedelta64
            return TimedeltaArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "string":
            # StringArray/ArrowStringArray depending on pd.options.mode.string_storage
            return StringDtype().construct_array_type()._from_sequence(data, copy=copy)

        elif inferred_dtype == "integer":
            return IntegerArray._from_sequence(data, copy=copy)

        elif inferred_dtype in ("floating", "mixed-integer-float"):
            return FloatingArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "boolean":
            return BooleanArray._from_sequence(data, copy=copy)

    # Pandas overrides NumPy for
    #   1. datetime64[ns]
    #   2. timedelta64[ns]
    # so that a DatetimeArray is returned.
    if is_datetime64_ns_dtype(dtype):
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)
    elif is_timedelta64_ns_dtype(dtype):
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)

    return PandasArray._from_sequence(data, dtype=dtype, copy=copy)


def extract_array(
    obj: object, extract_numpy: bool = False, extract_range: bool = False
) -> Any | ArrayLike:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.
        For Numpy-backed ExtensionArrays, the ndarray is extracted.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a PandasArray

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(['a', 'b', 'c'], dtype='category'))
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [1, 2, 3]

    For an ndarray-backed Series / Index a PandasArray is returned.

    >>> extract_array(pd.Series([1, 2, 3]))
    <PandasArray>
    [1, 2, 3]
    Length: 3, dtype: int64

    To extract all the way down to the ndarray, pass ``extract_numpy=True``.

    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)
    array([1, 2, 3])
    """
    if isinstance(obj, (ABCIndex, ABCSeries)):
        if isinstance(obj, ABCRangeIndex):
            if extract_range:
                return obj._values
            return obj

        obj = obj.array

    if extract_numpy and isinstance(obj, ABCPandasArray):
        obj = obj.to_numpy()

    return obj


def ensure_wrapped_if_datetimelike(arr):
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == "M":
            from pandas.core.arrays import DatetimeArray

            return DatetimeArray._from_sequence(arr)

        elif arr.dtype.kind == "m":
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(arr)

    return arr


def sanitize_masked_array(data: ma.MaskedArray) -> np.ndarray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
    mask = ma.getmaskarray(data)
    if mask.any():
        data, fill_value = maybe_upcast(data, copy=True)
        data.soften_mask()  # set hardmask False if it was True
        data[mask] = fill_value
    else:
        data = data.copy()
    return data


def sanitize_array(
    data,
    index: Index | None,
    dtype: DtypeObj | None = None,
    copy: bool = False,
    raise_cast_failure: bool = True,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    raise_cast_failure : bool, default True
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray

    Notes
    -----
    raise_cast_failure=False is only intended to be True when called from the
    DataFrame constructor, as the dtype keyword there may be interpreted as only
    applying to a subset of columns, see GH#24435.
    """
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)

    # extract ndarray or ExtensionArray, ensure we have no PandasArray
    data = extract_array(data, extract_numpy=True)

    if isinstance(data, np.ndarray) and data.ndim == 0:
        if dtype is None:
            dtype = data.dtype
        data = lib.item_from_zerodim(data)
    elif isinstance(data, range):
        # GH#16804
        data = range_to_ndarray(data)
        copy = False

    if not is_list_like(data):
        if index is None:
            raise ValueError("index must be specified when data is not list-like")
        data = construct_1d_arraylike_from_scalar(data, len(index), dtype)
        return data

    # GH#846
    if isinstance(data, np.ndarray):

        if dtype is not None and is_float_dtype(data.dtype) and is_integer_dtype(dtype):
            # possibility of nan -> garbage
            try:
                subarr = _try_cast(data, dtype, copy, True)
            except IntCastingNaNError:
                subarr = np.array(data, copy=copy)
            except ValueError:
                if not raise_cast_failure:
                    # i.e. called via DataFrame constructor
                    warnings.warn(
                        "In a future version, passing float-dtype values and an "
                        "integer dtype to DataFrame will retain floating dtype "
                        "if they cannot be cast losslessly (matching Series behavior). "
                        "To retain the old behavior, use DataFrame(data).astype(dtype)",
                        FutureWarning,
                        stacklevel=4,
                    )
                    # GH#40110 until the deprecation is enforced, we _dont_
                    #  ignore the dtype for DataFrame, and _do_ cast even though
                    #  it is lossy.
                    dtype = cast(np.dtype, dtype)
                    return np.array(data, dtype=dtype, copy=copy)
                subarr = np.array(data, copy=copy)
        else:
            # we will try to copy by-definition here
            subarr = _try_cast(data, dtype, copy, raise_cast_failure)

    elif isinstance(data, ABCExtensionArray):
        # it is already ensured above this is not a PandasArray
        subarr = data

        if dtype is not None:
            subarr = subarr.astype(dtype, copy=copy)
        elif copy:
            subarr = subarr.copy()

    else:
        if isinstance(data, (set, frozenset)):
            # Raise only for unordered sets, e.g., not for dict_keys
            raise TypeError(f"'{type(data).__name__}' type is unordered")

        # materialize e.g. generators, convert e.g. tuples, abc.ValueView
        if hasattr(data, "__array__"):
            # e.g. dask array GH#38645
            data = np.asarray(data)
        else:
            data = list(data)

        if dtype is not None or len(data) == 0:
            subarr = _try_cast(data, dtype, copy, raise_cast_failure)
        else:
            subarr = maybe_convert_platform(data)
            if subarr.dtype == object:
                subarr = cast(np.ndarray, subarr)
                subarr = maybe_infer_to_datetimelike(subarr)

    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)

    if isinstance(subarr, np.ndarray):
        # at this point we should have dtype be None or subarr.dtype == dtype
        dtype = cast(np.dtype, dtype)
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)

    return subarr


def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    # GH#30171 perf avoid realizing range as a list in np.array
    try:
        arr = np.arange(rng.start, rng.stop, rng.step, dtype="int64")
    except OverflowError:
        # GH#30173 handling for ranges that overflow int64
        if (rng.start >= 0 and rng.step > 0) or (rng.stop >= 0 and rng.step < 0):
            try:
                arr = np.arange(rng.start, rng.stop, rng.step, dtype="uint64")
            except OverflowError:
                arr = construct_1d_object_array_from_listlike(list(rng))
        else:
            arr = construct_1d_object_array_from_listlike(list(rng))
    return arr


def _sanitize_ndim(
    result: ArrayLike,
    data,
    dtype: DtypeObj | None,
    index: Index | None,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Ensure we have a 1-dimensional result array.
    """
    if getattr(result, "ndim", 0) == 0:
        raise ValueError("result should be arraylike with ndim > 0")

    elif result.ndim == 1:
        # the result that we want
        result = _maybe_repeat(result, index)

    elif result.ndim > 1:
        if isinstance(data, np.ndarray):
            if allow_2d:
                return result
            raise ValueError("Data must be 1-dimensional")
        if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
            # i.e. PandasDtype("O")

            result = com.asarray_tuplesafe(data, dtype=np.dtype("object"))
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        else:
            # error: Argument "dtype" to "asarray_tuplesafe" has incompatible type
            # "Union[dtype[Any], ExtensionDtype, None]"; expected "Union[str,
            # dtype[Any], None]"
            result = com.asarray_tuplesafe(data, dtype=dtype)  # type: ignore[arg-type]
    return result


def _sanitize_str_dtypes(
    result: np.ndarray, data, dtype: np.dtype | None, copy: bool
) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """

    # This is to prevent mixed-type Series getting all casted to
    # NumPy string type, e.g. NaN --> '-1#IND'.
    if issubclass(result.dtype.type, str):
        # GH#16605
        # If not empty convert the data to dtype
        # GH#19853: If data is a scalar, result has already the result
        if not lib.is_scalar(data):
            if not np.all(isna(data)):
                data = np.array(data, dtype=dtype, copy=False)
            result = np.array(data, dtype=object, copy=copy)
    return result


def _maybe_repeat(arr: ArrayLike, index: Index | None) -> ArrayLike:
    """
    If we have a length-1 array and an index describing how long we expect
    the result to be, repeat the array.
    """
    if index is not None:
        if 1 == len(arr) != len(index):
            arr = arr.repeat(len(index))
    return arr


def _try_cast(
    arr: list | np.ndarray,
    dtype: DtypeObj | None,
    copy: bool,
    raise_cast_failure: bool,
) -> ArrayLike:
    """
    Convert input to numpy ndarray and optionally cast to a given dtype.

    Parameters
    ----------
    arr : ndarray or list
        Excludes: ExtensionArray, Series, Index.
    dtype : np.dtype, ExtensionDtype or None
    copy : bool
        If False, don't copy the data if not needed.
    raise_cast_failure : bool
        If True, and if a dtype is specified, raise errors during casting.
        Otherwise an object array is returned.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    is_ndarray = isinstance(arr, np.ndarray)

    if dtype is None:
        # perf shortcut as this is the most common case
        if is_ndarray:
            arr = cast(np.ndarray, arr)
            if arr.dtype != object:
                return sanitize_to_nanoseconds(arr, copy=copy)

            out = maybe_infer_to_datetimelike(arr)
            if out is arr and copy:
                out = out.copy()
            return out

        else:
            # i.e. list
            varr = np.array(arr, copy=False)
            # filter out cases that we _dont_ want to go through
            #  maybe_infer_to_datetimelike
            if varr.dtype != object or varr.size == 0:
                return varr
            return maybe_infer_to_datetimelike(varr)

    elif isinstance(dtype, ExtensionDtype):
        # create an extension array from its dtype
        if isinstance(dtype, DatetimeTZDtype):
            # We can't go through _from_sequence because it handles dt64naive
            #  data differently; _from_sequence treats naive as wall times,
            #  while maybe_cast_to_datetime treats it as UTC
            #  see test_maybe_promote_any_numpy_dtype_with_datetimetz

            return maybe_cast_to_datetime(arr, dtype)
            # TODO: copy?

        array_type = dtype.construct_array_type()._from_sequence
        subarr = array_type(arr, dtype=dtype, copy=copy)
        return subarr

    elif is_object_dtype(dtype):
        if not is_ndarray:
            subarr = construct_1d_object_array_from_listlike(arr)
            return subarr
        return ensure_wrapped_if_datetimelike(arr).astype(dtype, copy=copy)

    elif dtype.kind == "U":
        # TODO: test cases with arr.dtype.kind in ["m", "M"]
        return lib.ensure_string_array(arr, convert_na_value=False, copy=copy)

    elif dtype.kind in ["m", "M"]:
        return maybe_cast_to_datetime(arr, dtype)

    try:
        # GH#15832: Check if we are requesting a numeric dtype and
        # that we can convert the data to the requested dtype.
        if is_integer_dtype(dtype):
            # this will raise if we have e.g. floats

            subarr = maybe_cast_to_integer_array(arr, dtype)
        else:
            # 4 tests fail if we move this to a try/except/else; see
            #  test_constructor_compound_dtypes, test_constructor_cast_failure
            #  test_constructor_dict_cast2, test_loc_setitem_dtype
            subarr = np.array(arr, dtype=dtype, copy=copy)

    except (ValueError, TypeError):
        if raise_cast_failure:
            raise
        else:
            # we only get here with raise_cast_failure False, which means
            #  called via the DataFrame constructor
            # GH#24435
            warnings.warn(
                f"Could not cast to {dtype}, falling back to object. This "
                "behavior is deprecated. In a future version, when a dtype is "
                "passed to 'DataFrame', either all columns will be cast to that "
                "dtype, or a TypeError will be raised",
                FutureWarning,
                stacklevel=7,
            )
            subarr = np.array(arr, dtype=object, copy=copy)
    return subarr


def is_empty_data(data: Any) -> bool:
    """
    Utility to check if a Series is instantiated with empty data,
    which does not contain dtype information.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series.

    Returns
    -------
    bool
    """
    is_none = data is None
    is_list_like_without_dtype = is_list_like(data) and not hasattr(data, "dtype")
    is_simple_empty = is_list_like_without_dtype and not data
    return is_none or is_simple_empty


def create_series_with_explicit_dtype(
    data: Any = None,
    index: ArrayLike | Index | None = None,
    dtype: Dtype | None = None,
    name: str | None = None,
    copy: bool = False,
    fastpath: bool = False,
    dtype_if_empty: Dtype = object,
) -> Series:
    """
    Helper to pass an explicit dtype when instantiating an empty Series.

    This silences a DeprecationWarning described in GitHub-17261.

    Parameters
    ----------
    data : Mirrored from Series.__init__
    index : Mirrored from Series.__init__
    dtype : Mirrored from Series.__init__
    name : Mirrored from Series.__init__
    copy : Mirrored from Series.__init__
    fastpath : Mirrored from Series.__init__
    dtype_if_empty : str, numpy.dtype, or ExtensionDtype
        This dtype will be passed explicitly if an empty Series will
        be instantiated.

    Returns
    -------
    Series
    """
    from pandas.core.series import Series

    if is_empty_data(data) and dtype is None:
        dtype = dtype_if_empty
    return Series(
        data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
    )
