"""
Generic data algorithms. This module is experimental at the moment and not
intended for public consumption
"""
from __future__ import annotations

import operator
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Union,
    cast,
)
from warnings import warn

import numpy as np

from pandas._libs import (
    algos,
    hashtable as htable,
    iNaT,
    lib,
)
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DtypeObj,
    FrameOrSeriesUnion,
    Scalar,
    final,
)
from pandas.util._decorators import doc

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    infer_dtype_from_array,
    sanitize_to_nanoseconds,
)
from pandas.core.dtypes.common import (
    ensure_float64,
    ensure_object,
    ensure_platform_int,
    is_array_like,
    is_bool_dtype,
    is_categorical_dtype,
    is_complex_dtype,
    is_datetime64_dtype,
    is_extension_array_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_timedelta64_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import PandasDtype
from pandas.core.dtypes.generic import (
    ABCDatetimeArray,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCRangeIndex,
    ABCSeries,
    ABCTimedeltaArray,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
)

from pandas.core.array_algos.take import take_nd
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import validate_indices

if TYPE_CHECKING:
    from typing import Literal

    from pandas import (
        Categorical,
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.arrays import (
        DatetimeArray,
        TimedeltaArray,
    )

_shared_docs: dict[str, str] = {}


# --------------- #
# dtype access    #
# --------------- #
def _ensure_data(values: ArrayLike) -> tuple[np.ndarray, DtypeObj]:
    """
    routine to ensure that our data is of the correct
    input dtype for lower-level routines

    This will coerce:
    - ints -> int64
    - uint -> uint64
    - bool -> uint64 (TODO this should be uint8)
    - datetimelike -> i8
    - datetime64tz -> i8 (in local tz)
    - categorical -> codes

    Parameters
    ----------
    values : array-like

    Returns
    -------
    values : ndarray
    pandas_dtype : np.dtype or ExtensionDtype
    """

    if not isinstance(values, ABCMultiIndex):
        # extract_array would raise
        values = extract_array(values, extract_numpy=True)

    # we check some simple dtypes first
    if is_object_dtype(values.dtype):
        return ensure_object(np.asarray(values)), np.dtype("object")

    elif is_bool_dtype(values.dtype):
        if isinstance(values, np.ndarray):
            # i.e. actually dtype == np.dtype("bool")
            return np.asarray(values).view("uint8"), values.dtype
        else:
            # i.e. all-bool Categorical, BooleanArray
            try:
                return np.asarray(values).astype("uint8", copy=False), values.dtype
            except (TypeError, ValueError):
                # GH#42107 we have pd.NAs present
                return np.asarray(values), values.dtype

    elif is_integer_dtype(values.dtype):
        return np.asarray(values), values.dtype

    elif is_float_dtype(values.dtype):
        # Note: checking `values.dtype == "float128"` raises on Windows and 32bit
        # error: Item "ExtensionDtype" of "Union[Any, ExtensionDtype, dtype[Any]]"
        # has no attribute "itemsize"
        if values.dtype.itemsize in [2, 12, 16]:  # type: ignore[union-attr]
            # we dont (yet) have float128 hashtable support
            return ensure_float64(values), values.dtype
        return np.asarray(values), values.dtype

    elif is_complex_dtype(values.dtype):
        # Incompatible return value type (got "Tuple[Union[Any, ExtensionArray,
        # ndarray[Any, Any]], Union[Any, ExtensionDtype]]", expected
        # "Tuple[ndarray[Any, Any], Union[dtype[Any], ExtensionDtype]]")
        return values, values.dtype  # type: ignore[return-value]

    # datetimelike
    elif needs_i8_conversion(values.dtype):
        if isinstance(values, np.ndarray):
            values = sanitize_to_nanoseconds(values)
        npvalues = values.view("i8")
        npvalues = cast(np.ndarray, npvalues)
        return npvalues, values.dtype

    elif is_categorical_dtype(values.dtype):
        values = cast("Categorical", values)
        values = values.codes
        dtype = pandas_dtype("category")
        return values, dtype

    # we have failed, return object
    values = np.asarray(values, dtype=object)
    return ensure_object(values), np.dtype("object")


def _reconstruct_data(
    values: ArrayLike, dtype: DtypeObj, original: AnyArrayLike
) -> ArrayLike:
    """
    reverse of _ensure_data

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    original : AnyArrayLike

    Returns
    -------
    ExtensionArray or np.ndarray
    """
    if isinstance(values, ABCExtensionArray) and values.dtype == dtype:
        # Catch DatetimeArray/TimedeltaArray
        return values

    if not isinstance(dtype, np.dtype):
        # i.e. ExtensionDtype
        cls = dtype.construct_array_type()
        if isinstance(values, cls) and values.dtype == dtype:
            return values

        values = cls._from_sequence(values)
    elif is_bool_dtype(dtype):
        values = values.astype(dtype, copy=False)

        # we only support object dtypes bool Index
        if isinstance(original, ABCIndex):
            values = values.astype(object, copy=False)
    elif dtype is not None:
        if is_datetime64_dtype(dtype):
            dtype = np.dtype("datetime64[ns]")
        elif is_timedelta64_dtype(dtype):
            dtype = np.dtype("timedelta64[ns]")

        values = values.astype(dtype, copy=False)

    return values


def _ensure_arraylike(values) -> ArrayLike:
    """
    ensure that we are arraylike if not already
    """
    if not is_array_like(values):
        inferred = lib.infer_dtype(values, skipna=False)
        if inferred in ["mixed", "string", "mixed-integer"]:
            # "mixed-integer" to ensure we do not cast ["ss", 42] to str GH#22160
            if isinstance(values, tuple):
                values = list(values)
            values = construct_1d_object_array_from_listlike(values)
        else:
            values = np.asarray(values)
    return values


_hashtables = {
    "complex128": htable.Complex128HashTable,
    "complex64": htable.Complex64HashTable,
    "float64": htable.Float64HashTable,
    "float32": htable.Float32HashTable,
    "uint64": htable.UInt64HashTable,
    "uint32": htable.UInt32HashTable,
    "uint16": htable.UInt16HashTable,
    "uint8": htable.UInt8HashTable,
    "int64": htable.Int64HashTable,
    "int32": htable.Int32HashTable,
    "int16": htable.Int16HashTable,
    "int8": htable.Int8HashTable,
    "string": htable.StringHashTable,
    "object": htable.PyObjectHashTable,
}


def _get_hashtable_algo(values: np.ndarray):
    """
    Parameters
    ----------
    values : np.ndarray

    Returns
    -------
    htable : HashTable subclass
    values : ndarray
    """
    values, _ = _ensure_data(values)

    ndtype = _check_object_for_strings(values)
    htable = _hashtables[ndtype]
    return htable, values


def _get_values_for_rank(values: ArrayLike) -> np.ndarray:
    if is_categorical_dtype(values):
        values = cast("Categorical", values)._values_for_rank()

    values, _ = _ensure_data(values)
    if values.dtype.kind in ["i", "u", "f"]:
        # rank_t includes only object, int64, uint64, float64
        dtype = values.dtype.kind + "8"
        values = values.astype(dtype, copy=False)
    return values


def get_data_algo(values: ArrayLike):
    values = _get_values_for_rank(values)

    ndtype = _check_object_for_strings(values)
    htable = _hashtables.get(ndtype, _hashtables["object"])

    return htable, values


def _check_object_for_strings(values: np.ndarray) -> str:
    """
    Check if we can use string hashtable instead of object hashtable.

    Parameters
    ----------
    values : ndarray

    Returns
    -------
    str
    """
    ndtype = values.dtype.name
    if ndtype == "object":

        # it's cheaper to use a String Hash Table than Object; we infer
        # including nulls because that is the only difference between
        # StringHashTable and ObjectHashtable
        if lib.infer_dtype(values, skipna=False) in ["string"]:
            ndtype = "string"
    return ndtype


# --------------- #
# top-level algos #
# --------------- #


def unique(values):
    """
    Hash table-based unique. Uniques are returned in order
    of appearance. This does NOT sort.

    Significantly faster than numpy.unique for long enough sequences.
    Includes NA values.

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    numpy.ndarray or ExtensionArray

        The return can be:

        * Index : when the input is an Index
        * Categorical : when the input is a Categorical dtype
        * ndarray : when the input is a Series/ndarray

        Return numpy.ndarray or ExtensionArray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> pd.unique(pd.Series([2, 1, 3, 3]))
    array([2, 1, 3])

    >>> pd.unique(pd.Series([2] + [1] * 5))
    array([2, 1])

    >>> pd.unique(pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    array(['2016-01-01T00:00:00.000000000'], dtype='datetime64[ns]')

    >>> pd.unique(
    ...     pd.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    <DatetimeArray>
    ['2016-01-01 00:00:00-05:00']
    Length: 1, dtype: datetime64[ns, US/Eastern]

    >>> pd.unique(
    ...     pd.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00'],
            dtype='datetime64[ns, US/Eastern]',
            freq=None)

    >>> pd.unique(list("baabc"))
    array(['b', 'a', 'c'], dtype=object)

    An unordered Categorical will return categories in the
    order of appearance.

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    >>> pd.unique(pd.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    ['b', 'a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    ['b', 'a', 'c']
    Categories (3, object): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")])
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
    """
    values = _ensure_arraylike(values)

    if is_extension_array_dtype(values):
        # Dispatch to extension dtype's unique.
        return values.unique()

    original = values
    htable, values = _get_hashtable_algo(values)

    table = htable(len(values))
    uniques = table.unique(values)
    uniques = _reconstruct_data(uniques, original.dtype, original)
    return uniques


unique1d = unique


def isin(comps: AnyArrayLike, values: AnyArrayLike) -> np.ndarray:
    """
    Compute the isin boolean array.

    Parameters
    ----------
    comps : array-like
    values : array-like

    Returns
    -------
    ndarray[bool]
        Same length as `comps`.
    """
    if not is_list_like(comps):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a [{type(comps).__name__}]"
        )
    if not is_list_like(values):
        raise TypeError(
            "only list-like objects are allowed to be passed "
            f"to isin(), you passed a [{type(values).__name__}]"
        )

    if not isinstance(values, (ABCIndex, ABCSeries, ABCExtensionArray, np.ndarray)):
        values = _ensure_arraylike(list(values))
    elif isinstance(values, ABCMultiIndex):
        # Avoid raising in extract_array
        values = np.array(values)
    else:
        values = extract_array(values, extract_numpy=True, extract_range=True)

    comps = _ensure_arraylike(comps)
    comps = extract_array(comps, extract_numpy=True)
    if not isinstance(comps, np.ndarray):
        # i.e. Extension Array
        return comps.isin(values)

    elif needs_i8_conversion(comps.dtype):
        # Dispatch to DatetimeLikeArrayMixin.isin
        return pd_array(comps).isin(values)
    elif needs_i8_conversion(values.dtype) and not is_object_dtype(comps.dtype):
        # e.g. comps are integers and values are datetime64s
        return np.zeros(comps.shape, dtype=bool)
        # TODO: not quite right ... Sparse/Categorical
    elif needs_i8_conversion(values.dtype):
        return isin(comps, values.astype(object))

    elif is_extension_array_dtype(values.dtype):
        return isin(np.asarray(comps), np.asarray(values))

    # GH16012
    # Ensure np.in1d doesn't get object types or it *may* throw an exception
    # Albeit hashmap has O(1) look-up (vs. O(logn) in sorted array),
    # in1d is faster for small sizes
    if len(comps) > 1_000_000 and len(values) <= 26 and not is_object_dtype(comps):
        # If the values include nan we need to check for nan explicitly
        # since np.nan it not equal to np.nan
        if isna(values).any():

            def f(c, v):
                return np.logical_or(np.in1d(c, v), np.isnan(c))

        else:
            f = np.in1d

    else:
        # error: List item 0 has incompatible type "Union[Any, dtype[Any],
        # ExtensionDtype]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Tuple[Any, Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any,
        # Any]]"
        # error: List item 1 has incompatible type "Union[Any, ExtensionDtype]";
        # expected "Union[dtype[Any], None, type, _SupportsDType, str, Tuple[Any,
        # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]"
        # error: List item 1 has incompatible type "Union[dtype[Any], ExtensionDtype]";
        # expected "Union[dtype[Any], None, type, _SupportsDType, str, Tuple[Any,
        # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]"
        common = np.find_common_type(
            [values.dtype, comps.dtype], []  # type: ignore[list-item]
        )
        values = values.astype(common, copy=False)
        comps = comps.astype(common, copy=False)
        f = htable.ismember

    return f(comps, values)


def factorize_array(
    values: np.ndarray,
    na_sentinel: int = -1,
    size_hint: int | None = None,
    na_value=None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorize an array-like to codes and uniques.

    This doesn't do any coercion of types or unboxing before factorization.

    Parameters
    ----------
    values : ndarray
    na_sentinel : int, default -1
    size_hint : int, optional
        Passed through to the hashtable's 'get_labels' method
    na_value : object, optional
        A value in `values` to consider missing. Note: only use this
        parameter when you know that you don't have any values pandas would
        consider missing in the array (NaN for float data, iNaT for
        datetimes, etc.).
    mask : ndarray[bool], optional
        If not None, the mask is used as indicator for missing values
        (True = missing, False = valid) instead of `na_value` or
        condition "val != val".

    Returns
    -------
    codes : ndarray[np.intp]
    uniques : ndarray
    """
    hash_klass, values = get_data_algo(values)

    table = hash_klass(size_hint or len(values))
    uniques, codes = table.factorize(
        values, na_sentinel=na_sentinel, na_value=na_value, mask=mask
    )

    codes = ensure_platform_int(codes)
    return codes, uniques


@doc(
    values=dedent(
        """\
    values : sequence
        A 1-D sequence. Sequences that aren't pandas objects are
        coerced to ndarrays before factorization.
    """
    ),
    sort=dedent(
        """\
    sort : bool, default False
        Sort `uniques` and shuffle `codes` to maintain the
        relationship.
    """
    ),
    size_hint=dedent(
        """\
    size_hint : int, optional
        Hint to the hashtable sizer.
    """
    ),
)
def factorize(
    values,
    sort: bool = False,
    na_sentinel: int | None = -1,
    size_hint: int | None = None,
) -> tuple[np.ndarray, np.ndarray | Index]:
    """
    Encode the object as an enumerated type or categorical variable.

    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. `factorize`
    is available as both a top-level function :func:`pandas.factorize`,
    and as a method :meth:`Series.factorize` and :meth:`Index.factorize`.

    Parameters
    ----------
    {values}{sort}
    na_sentinel : int or None, default -1
        Value to mark "not found". If None, will not drop the NaN
        from the uniques of the values.

        .. versionchanged:: 1.1.2
    {size_hint}\

    Returns
    -------
    codes : ndarray
        An integer ndarray that's an indexer into `uniques`.
        ``uniques.take(codes)`` will have the same values as `values`.
    uniques : ndarray, Index, or Categorical
        The unique valid values. When `values` is Categorical, `uniques`
        is a Categorical. When `values` is some other pandas object, an
        `Index` is returned. Otherwise, a 1-D ndarray is returned.

        .. note ::

           Even if there's a missing value in `values`, `uniques` will
           *not* contain an entry for it.

    See Also
    --------
    cut : Discretize continuous-valued array.
    unique : Find the unique value in an array.

    Examples
    --------
    These examples all show factorize as a top-level method like
    ``pd.factorize(values)``. The results are identical for methods like
    :meth:`Series.factorize`.

    >>> codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'])
    >>> codes
    array([0, 0, 1, 2, 0]...)
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    With ``sort=True``, the `uniques` will be sorted, and `codes` will be
    shuffled so that the relationship is the maintained.

    >>> codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'], sort=True)
    >>> codes
    array([1, 1, 0, 2, 1]...)
    >>> uniques
    array(['a', 'b', 'c'], dtype=object)

    Missing values are indicated in `codes` with `na_sentinel`
    (``-1`` by default). Note that missing values are never
    included in `uniques`.

    >>> codes, uniques = pd.factorize(['b', None, 'a', 'c', 'b'])
    >>> codes
    array([ 0, -1,  1,  2,  0]...)
    >>> uniques
    array(['b', 'a', 'c'], dtype=object)

    Thus far, we've only factorized lists (which are internally coerced to
    NumPy arrays). When factorizing pandas objects, the type of `uniques`
    will differ. For Categoricals, a `Categorical` is returned.

    >>> cat = pd.Categorical(['a', 'a', 'c'], categories=['a', 'b', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1]...)
    >>> uniques
    ['a', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Notice that ``'b'`` is in ``uniques.categories``, despite not being
    present in ``cat.values``.

    For all other pandas objects, an Index of the appropriate type is
    returned.

    >>> cat = pd.Series(['a', 'a', 'c'])
    >>> codes, uniques = pd.factorize(cat)
    >>> codes
    array([0, 0, 1]...)
    >>> uniques
    Index(['a', 'c'], dtype='object')

    If NaN is in the values, and we want to include NaN in the uniques of the
    values, it can be achieved by setting ``na_sentinel=None``.

    >>> values = np.array([1, 2, 1, np.nan])
    >>> codes, uniques = pd.factorize(values)  # default: na_sentinel=-1
    >>> codes
    array([ 0,  1,  0, -1])
    >>> uniques
    array([1., 2.])

    >>> codes, uniques = pd.factorize(values, na_sentinel=None)
    >>> codes
    array([0, 1, 0, 2])
    >>> uniques
    array([ 1.,  2., nan])
    """
    # Implementation notes: This method is responsible for 3 things
    # 1.) coercing data to array-like (ndarray, Index, extension array)
    # 2.) factorizing codes and uniques
    # 3.) Maybe boxing the uniques in an Index
    #
    # Step 2 is dispatched to extension types (like Categorical). They are
    # responsible only for factorization. All data coercion, sorting and boxing
    # should happen here.

    if isinstance(values, ABCRangeIndex):
        return values.factorize(sort=sort)

    values = _ensure_arraylike(values)
    original = values
    if not isinstance(values, ABCMultiIndex):
        values = extract_array(values, extract_numpy=True)

    # GH35667, if na_sentinel=None, we will not dropna NaNs from the uniques
    # of values, assign na_sentinel=-1 to replace code value for NaN.
    dropna = True
    if na_sentinel is None:
        na_sentinel = -1
        dropna = False

    if (
        isinstance(values, (ABCDatetimeArray, ABCTimedeltaArray))
        and values.freq is not None
    ):
        codes, uniques = values.factorize(sort=sort)
        if isinstance(original, ABCIndex):
            uniques = original._shallow_copy(uniques, name=None)
        elif isinstance(original, ABCSeries):
            from pandas import Index

            uniques = Index(uniques)
        return codes, uniques

    if not isinstance(values.dtype, np.dtype):
        # i.e. ExtensionDtype
        codes, uniques = values.factorize(na_sentinel=na_sentinel)
        dtype = original.dtype
    else:
        values, dtype = _ensure_data(values)
        na_value: Scalar

        if original.dtype.kind in ["m", "M"]:
            # Note: factorize_array will cast NaT bc it has a __int__
            #  method, but will not cast the more-correct dtype.type("nat")
            na_value = iNaT
        else:
            na_value = None

        codes, uniques = factorize_array(
            values, na_sentinel=na_sentinel, size_hint=size_hint, na_value=na_value
        )

    if sort and len(uniques) > 0:
        uniques, codes = safe_sort(
            uniques, codes, na_sentinel=na_sentinel, assume_unique=True, verify=False
        )

    code_is_na = codes == na_sentinel
    if not dropna and code_is_na.any():
        # na_value is set based on the dtype of uniques, and compat set to False is
        # because we do not want na_value to be 0 for integers
        na_value = na_value_for_dtype(uniques.dtype, compat=False)
        uniques = np.append(uniques, [na_value])
        codes = np.where(code_is_na, len(uniques) - 1, codes)

    uniques = _reconstruct_data(uniques, dtype, original)

    # return original tenor
    if isinstance(original, ABCIndex):
        if original.dtype.kind in ["m", "M"] and isinstance(uniques, np.ndarray):
            original._data = cast(
                "Union[DatetimeArray, TimedeltaArray]", original._data
            )
            uniques = type(original._data)._simple_new(uniques, dtype=original.dtype)
        uniques = original._shallow_copy(uniques, name=None)
    elif isinstance(original, ABCSeries):
        from pandas import Index

        uniques = Index(uniques)

    return codes, uniques


def value_counts(
    values,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins=None,
    dropna: bool = True,
) -> Series:
    """
    Compute a histogram of the counts of non-null values.

    Parameters
    ----------
    values : ndarray (1-d)
    sort : bool, default True
        Sort by values
    ascending : bool, default False
        Sort in ascending order
    normalize: bool, default False
        If True then compute a relative histogram
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        convenience for pd.cut, only works with numeric data
    dropna : bool, default True
        Don't include counts of NaN

    Returns
    -------
    Series
    """
    from pandas.core.series import Series

    name = getattr(values, "name", None)

    if bins is not None:
        from pandas.core.reshape.tile import cut

        values = Series(values)
        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError("bins argument only works with numeric data.") from err

        # count, remove nulls (from the index), and but the bins
        result = ii.value_counts(dropna=dropna)
        result = result[result.index.notna()]
        result.index = result.index.astype("interval")
        result = result.sort_index()

        # if we are dropna and we have NO values
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]

        # normalizing is by len of all (regardless of dropna)
        counts = np.array([len(ii)])

    else:

        if is_extension_array_dtype(values):

            # handle Categorical and sparse,
            result = Series(values)._values.value_counts(dropna=dropna)
            result.name = name
            counts = result._values

        else:
            keys, counts = value_counts_arraylike(values, dropna)

            result = Series(counts, index=keys, name=name)

    if sort:
        result = result.sort_values(ascending=ascending)

    if normalize:
        result = result / counts.sum()

    return result


# Called once from SparseArray, otherwise could be private
def value_counts_arraylike(values, dropna: bool):
    """
    Parameters
    ----------
    values : arraylike
    dropna : bool

    Returns
    -------
    uniques : np.ndarray or ExtensionArray
    counts : np.ndarray
    """
    values = _ensure_arraylike(values)
    original = values
    values, _ = _ensure_data(values)

    # TODO: handle uint8
    keys, counts = htable.value_count(values, dropna)

    if needs_i8_conversion(original.dtype):
        # datetime, timedelta, or period

        if dropna:
            msk = keys != iNaT
            keys, counts = keys[msk], counts[msk]

    res_keys = _reconstruct_data(keys, original.dtype, original)
    return res_keys, counts


def duplicated(
    values: ArrayLike, keep: Literal["first", "last", False] = "first"
) -> np.ndarray:
    """
    Return boolean ndarray denoting duplicate values.

    Parameters
    ----------
    values : ndarray-like
        Array over which to check for duplicate values.
    keep : {'first', 'last', False}, default 'first'
        - ``first`` : Mark duplicates as ``True`` except for the first
          occurrence.
        - ``last`` : Mark duplicates as ``True`` except for the last
          occurrence.
        - False : Mark all duplicates as ``True``.

    Returns
    -------
    duplicated : ndarray[bool]
    """
    values, _ = _ensure_data(values)
    return htable.duplicated(values, keep=keep)


def mode(values, dropna: bool = True) -> Series:
    """
    Returns the mode(s) of an array.

    Parameters
    ----------
    values : array-like
        Array over which to check for duplicate values.
    dropna : bool, default True
        Don't consider counts of NaN/NaT.

    Returns
    -------
    mode : Series
    """
    from pandas import Series
    import pandas.core.indexes.base as ibase

    values = _ensure_arraylike(values)
    original = values

    # categorical is a fast-path
    if is_categorical_dtype(values):
        if isinstance(values, Series):
            # TODO: should we be passing `name` below?
            return Series(values._values.mode(dropna=dropna), name=values.name)
        return values.mode(dropna=dropna)

    if dropna and needs_i8_conversion(values.dtype):
        mask = values.isnull()
        values = values[~mask]

    values, _ = _ensure_data(values)

    npresult = htable.mode(values, dropna=dropna)
    try:
        npresult = np.sort(npresult)
    except TypeError as err:
        warn(f"Unable to sort modes: {err}")

    result = _reconstruct_data(npresult, original.dtype, original)
    # Ensure index is type stable (should always use int index)
    return Series(result, index=ibase.default_index(len(result)))


def rank(
    values: ArrayLike,
    axis: int = 0,
    method: str = "average",
    na_option: str = "keep",
    ascending: bool = True,
    pct: bool = False,
) -> np.ndarray:
    """
    Rank the values along a given axis.

    Parameters
    ----------
    values : array-like
        Array whose values will be ranked. The number of dimensions in this
        array must not exceed 2.
    axis : int, default 0
        Axis over which to perform rankings.
    method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        The method by which tiebreaks are broken during the ranking.
    na_option : {'keep', 'top'}, default 'keep'
        The method by which NaNs are placed in the ranking.
        - ``keep``: rank each NaN value with a NaN ranking
        - ``top``: replace each NaN with either +/- inf so that they
                   there are ranked at the top
    ascending : bool, default True
        Whether or not the elements should be ranked in ascending order.
    pct : bool, default False
        Whether or not to the display the returned rankings in integer form
        (e.g. 1, 2, 3) or in percentile form (e.g. 0.333..., 0.666..., 1).
    """
    is_datetimelike = needs_i8_conversion(values.dtype)
    values = _get_values_for_rank(values)
    if values.ndim == 1:
        ranks = algos.rank_1d(
            values,
            labels=np.zeros(len(values), dtype=np.intp),
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    elif values.ndim == 2:
        ranks = algos.rank_2d(
            values,
            axis=axis,
            is_datetimelike=is_datetimelike,
            ties_method=method,
            ascending=ascending,
            na_option=na_option,
            pct=pct,
        )
    else:
        raise TypeError("Array with ndim > 2 are not supported.")

    return ranks


def checked_add_with_arr(
    arr: np.ndarray,
    b,
    arr_mask: np.ndarray | None = None,
    b_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform array addition that checks for underflow and overflow.

    Performs the addition of an int64 array and an int64 integer (or array)
    but checks that they do not result in overflow first. For elements that
    are indicated to be NaN, whether or not there is overflow for that element
    is automatically ignored.

    Parameters
    ----------
    arr : array addend.
    b : array or scalar addend.
    arr_mask : np.ndarray[bool] or None, default None
        array indicating which elements to exclude from checking
    b_mask : np.ndarray[bool] or None, default None
        array or scalar indicating which element(s) to exclude from checking

    Returns
    -------
    sum : An array for elements x + b for each element x in arr if b is
          a scalar or an array for elements x + y for each element pair
          (x, y) in (arr, b).

    Raises
    ------
    OverflowError if any x + y exceeds the maximum or minimum int64 value.
    """
    # For performance reasons, we broadcast 'b' to the new array 'b2'
    # so that it has the same size as 'arr'.
    b2 = np.broadcast_to(b, arr.shape)
    if b_mask is not None:
        # We do the same broadcasting for b_mask as well.
        b2_mask = np.broadcast_to(b_mask, arr.shape)
    else:
        b2_mask = None

    # For elements that are NaN, regardless of their value, we should
    # ignore whether they overflow or not when doing the checked add.
    if arr_mask is not None and b2_mask is not None:
        not_nan = np.logical_not(arr_mask | b2_mask)
    elif arr_mask is not None:
        not_nan = np.logical_not(arr_mask)
    elif b_mask is not None:
        not_nan = np.logical_not(b2_mask)
    else:
        not_nan = np.empty(arr.shape, dtype=bool)
        not_nan.fill(True)

    # gh-14324: For each element in 'arr' and its corresponding element
    # in 'b2', we check the sign of the element in 'b2'. If it is positive,
    # we then check whether its sum with the element in 'arr' exceeds
    # np.iinfo(np.int64).max. If so, we have an overflow error. If it
    # it is negative, we then check whether its sum with the element in
    # 'arr' exceeds np.iinfo(np.int64).min. If so, we have an overflow
    # error as well.
    i8max = lib.i8max
    i8min = iNaT

    mask1 = b2 > 0
    mask2 = b2 < 0

    if not mask1.any():
        to_raise = ((i8min - b2 > arr) & not_nan).any()
    elif not mask2.any():
        to_raise = ((i8max - b2 < arr) & not_nan).any()
    else:
        to_raise = ((i8max - b2[mask1] < arr[mask1]) & not_nan[mask1]).any() or (
            (i8min - b2[mask2] > arr[mask2]) & not_nan[mask2]
        ).any()

    if to_raise:
        raise OverflowError("Overflow in int64 addition")
    return arr + b


def quantile(x, q, interpolation_method="fraction"):
    """
    Compute sample quantile or quantiles of the input array. For example, q=0.5
    computes the median.

    The `interpolation_method` parameter supports three values, namely
    `fraction` (default), `lower` and `higher`. Interpolation is done only,
    if the desired quantile lies between two data points `i` and `j`. For
    `fraction`, the result is an interpolated value between `i` and `j`;
    for `lower`, the result is `i`, for `higher` the result is `j`.

    Parameters
    ----------
    x : ndarray
        Values from which to extract score.
    q : scalar or array
        Percentile at which to extract score.
    interpolation_method : {'fraction', 'lower', 'higher'}, optional
        This optional parameter specifies the interpolation method to use,
        when the desired quantile lies between two data points `i` and `j`:

        - fraction: `i + (j - i)*fraction`, where `fraction` is the
                    fractional part of the index surrounded by `i` and `j`.
        -lower: `i`.
        - higher: `j`.

    Returns
    -------
    score : float
        Score at percentile.

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(100)
    >>> stats.scoreatpercentile(a, 50)
    49.5

    """
    x = np.asarray(x)
    mask = isna(x)

    x = x[~mask]

    values = np.sort(x)

    def _interpolate(a, b, fraction):
        """
        Returns the point at the given fraction between a and b, where
        'fraction' must be between 0 and 1.
        """
        return a + (b - a) * fraction

    def _get_score(at):
        if len(values) == 0:
            return np.nan

        idx = at * (len(values) - 1)
        if idx % 1 == 0:
            score = values[int(idx)]
        else:
            if interpolation_method == "fraction":
                score = _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1)
            elif interpolation_method == "lower":
                score = values[np.floor(idx)]
            elif interpolation_method == "higher":
                score = values[np.ceil(idx)]
            else:
                raise ValueError(
                    "interpolation_method can only be 'fraction' "
                    ", 'lower' or 'higher'"
                )

        return score

    if is_scalar(q):
        return _get_score(q)
    else:
        q = np.asarray(q, np.float64)
        result = [_get_score(x) for x in q]
        # error: Incompatible types in assignment (expression has type
        # "ndarray", variable has type "List[Any]")
        result = np.array(result, dtype=np.float64)  # type: ignore[assignment]
        return result


# --------------- #
# select n        #
# --------------- #


class SelectN:
    def __init__(self, obj, n: int, keep: str):
        self.obj = obj
        self.n = n
        self.keep = keep

        if self.keep not in ("first", "last", "all"):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> FrameOrSeriesUnion:
        raise NotImplementedError

    @final
    def nlargest(self):
        return self.compute("nlargest")

    @final
    def nsmallest(self):
        return self.compute("nsmallest")

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        return (
            is_numeric_dtype(dtype) and not is_complex_dtype(dtype)
        ) or needs_i8_conversion(dtype)


class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series:

        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

        if n <= 0:
            return self.obj[[]]

        dropped = self.obj.dropna()

        if is_extension_array_dtype(dropped.dtype):
            # GH#41816 bc we have dropped NAs above, MaskedArrays can use the
            #  numpy logic.
            from pandas.core.arrays import BaseMaskedArray

            arr = dropped._values
            if isinstance(arr, BaseMaskedArray):
                ser = type(dropped)(arr._data, index=dropped.index, name=dropped.name)

                result = type(self)(ser, n=self.n, keep=self.keep).compute(method)
                return result.astype(arr.dtype)

        # slow method
        if n >= len(self.obj):
            ascending = method == "nsmallest"
            return dropped.sort_values(ascending=ascending).head(n)

        # fast method
        arr, new_dtype = _ensure_data(dropped.values)
        if method == "nlargest":
            arr = -arr
            if is_integer_dtype(new_dtype):
                # GH 21426: ensure reverse ordering at boundaries
                arr -= 1

            elif is_bool_dtype(new_dtype):
                # GH 26154: ensure False is smaller than True
                arr = 1 - (-arr)

        if self.keep == "last":
            arr = arr[::-1]

        narr = len(arr)
        n = min(n, narr)

        # arr passed into kth_smallest must be contiguous. We copy
        # here because kth_smallest will modify its input
        kth_val = algos.kth_smallest(arr.copy(order="C"), n - 1)
        (ns,) = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind="mergesort")]

        if self.keep != "all":
            inds = inds[:n]

        if self.keep == "last":
            # reverse indices
            inds = narr - 1 - inds

        return dropped.iloc[inds]


class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj, n: int, keep: str, columns):
        super().__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]
        columns = list(columns)
        self.columns = columns

    def compute(self, method: str) -> DataFrame:

        from pandas import Int64Index

        n = self.n
        frame = self.obj
        columns = self.columns

        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(
                    f"Column {repr(column)} has dtype {dtype}, "
                    f"cannot use method {repr(method)} with this dtype"
                )

        def get_indexer(current_indexer, other_indexer):
            """
            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            if method == "nsmallest":
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)

        # Below we save and reset the index in case index contains duplicates
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Int64Index([])

        for i, column in enumerate(columns):
            # For each column we apply method to cur_frame[column].
            # If it's the last column or if we have the number of
            # results desired we are done.
            # Otherwise there are duplicates of the largest/smallest
            # value and we need to look at the rest of the columns
            # to determine which of the rows with the largest/smallest
            # value in the column to keep.
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(
                cur_n, keep=self.keep if is_last_column else "all"
            )

            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break

            # Now find all values which are equal to
            # the (nsmallest: largest)/(nlargest: smallest)
            # from our series.
            border_value = values == values[values.index[-1]]

            # Some of these values are among the top-n
            # some aren't.
            unsafe_values = values[border_value]

            # These values are definitely among the top-n
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)

            # Go on and separate the unsafe_values on the remaining
            # columns.
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)

        frame = frame.take(indexer)

        # Restore the index on frame
        frame.index = original_index.take(indexer)

        # If there is only one column, the frame is already sorted.
        if len(columns) == 1:
            return frame

        ascending = method == "nsmallest"

        return frame.sort_values(columns, ascending=ascending, kind="mergesort")


# ---- #
# take #
# ---- #


def take(
    arr, indices: np.ndarray, axis: int = 0, allow_fill: bool = False, fill_value=None
):
    """
    Take elements from an array.

    Parameters
    ----------
    arr : sequence
        Non array-likes (sequences without a dtype) are coerced
        to an ndarray.
    indices : sequence of integers
        Indices to be taken.
    axis : int, default 0
        The axis over which to select values.
    allow_fill : bool, default False
        How to handle negative values in `indices`.

        * False: negative values in `indices` indicate positional indices
          from the right (the default). This is similar to :func:`numpy.take`.

        * True: negative values in `indices` indicate
          missing values. These values are set to `fill_value`. Any other
          negative values raise a ``ValueError``.

    fill_value : any, optional
        Fill value to use for NA-indices when `allow_fill` is True.
        This may be ``None``, in which case the default NA value for
        the type (``self.dtype.na_value``) is used.

        For multi-dimensional `arr`, each *element* is filled with
        `fill_value`.

    Returns
    -------
    ndarray or ExtensionArray
        Same type as the input.

    Raises
    ------
    IndexError
        When `indices` is out of bounds for the array.
    ValueError
        When the indexer contains negative values other than ``-1``
        and `allow_fill` is True.

    Notes
    -----
    When `allow_fill` is False, `indices` may be whatever dimensionality
    is accepted by NumPy for `arr`.

    When `allow_fill` is True, `indices` should be 1-D.

    See Also
    --------
    numpy.take : Take elements from an array along an axis.

    Examples
    --------
    >>> from pandas.api.extensions import take

    With the default ``allow_fill=False``, negative numbers indicate
    positional indices from the right.

    >>> take(np.array([10, 20, 30]), [0, 0, -1])
    array([10, 10, 30])

    Setting ``allow_fill=True`` will place `fill_value` in those positions.

    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True)
    array([10., 10., nan])

    >>> take(np.array([10, 20, 30]), [0, 0, -1], allow_fill=True,
    ...      fill_value=-10)
    array([ 10,  10, -10])
    """
    if not is_array_like(arr):
        arr = np.asarray(arr)

    indices = np.asarray(indices, dtype=np.intp)

    if allow_fill:
        # Pandas style, -1 means NA
        validate_indices(indices, arr.shape[axis])
        result = take_nd(
            arr, indices, axis=axis, allow_fill=True, fill_value=fill_value
        )
    else:
        # NumPy style
        result = arr.take(indices, axis=axis)
    return result


# ------------ #
# searchsorted #
# ------------ #


def searchsorted(arr, value, side="left", sorter=None) -> np.ndarray:
    """
    Find indices where elements should be inserted to maintain order.

    .. versionadded:: 0.25.0

    Find the indices into a sorted array `arr` (a) such that, if the
    corresponding elements in `value` were inserted before the indices,
    the order of `arr` would be preserved.

    Assuming that `arr` is sorted:

    ======  ================================
    `side`  returned index `i` satisfies
    ======  ================================
    left    ``arr[i-1] < value <= self[i]``
    right   ``arr[i-1] <= value < self[i]``
    ======  ================================

    Parameters
    ----------
    arr: array-like
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    value : array-like
        Values to insert into `arr`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `self`).
    sorter : 1-D array-like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    array of ints
        Array of insertion points with the same shape as `value`.

    See Also
    --------
    numpy.searchsorted : Similar method from NumPy.
    """
    if sorter is not None:
        sorter = ensure_platform_int(sorter)

    if (
        isinstance(arr, np.ndarray)
        and is_integer_dtype(arr.dtype)
        and (is_integer(value) or is_integer_dtype(value))
    ):
        # if `arr` and `value` have different dtypes, `arr` would be
        # recast by numpy, causing a slow search.
        # Before searching below, we therefore try to give `value` the
        # same dtype as `arr`, while guarding against integer overflows.
        iinfo = np.iinfo(arr.dtype.type)
        value_arr = np.array([value]) if is_scalar(value) else np.array(value)
        if (value_arr >= iinfo.min).all() and (value_arr <= iinfo.max).all():
            # value within bounds, so no overflow, so can convert value dtype
            # to dtype of arr
            dtype = arr.dtype
        else:
            dtype = value_arr.dtype

        if is_scalar(value):
            value = dtype.type(value)
        else:
            value = pd_array(value, dtype=dtype)
    elif not (
        is_object_dtype(arr) or is_numeric_dtype(arr) or is_categorical_dtype(arr)
    ):
        # E.g. if `arr` is an array with dtype='datetime64[ns]'
        # and `value` is a pd.Timestamp, we may need to convert value
        arr = ensure_wrapped_if_datetimelike(arr)

    return arr.searchsorted(value, side=side, sorter=sorter)


# ---- #
# diff #
# ---- #

_diff_special = {"float64", "float32", "int64", "int32", "int16", "int8"}


def diff(arr, n: int, axis: int = 0, stacklevel: int = 3):
    """
    difference of n between self,
    analogous to s-s.shift(n)

    Parameters
    ----------
    arr : ndarray or ExtensionArray
    n : int
        number of periods
    axis : {0, 1}
        axis to shift on
    stacklevel : int, default 3
        The stacklevel for the lost dtype warning.

    Returns
    -------
    shifted
    """

    n = int(n)
    na = np.nan
    dtype = arr.dtype

    is_bool = is_bool_dtype(dtype)
    if is_bool:
        op = operator.xor
    else:
        op = operator.sub

    if isinstance(dtype, PandasDtype):
        # PandasArray cannot necessarily hold shifted versions of itself.
        arr = arr.to_numpy()
        dtype = arr.dtype

    if not isinstance(dtype, np.dtype):
        # i.e ExtensionDtype
        if hasattr(arr, f"__{op.__name__}__"):
            if axis != 0:
                raise ValueError(f"cannot diff {type(arr).__name__} on axis={axis}")
            return op(arr, arr.shift(n))
        else:
            warn(
                "dtype lost in 'diff()'. In the future this will raise a "
                "TypeError. Convert to a suitable dtype prior to calling 'diff'.",
                FutureWarning,
                stacklevel=stacklevel,
            )
            arr = np.asarray(arr)
            dtype = arr.dtype

    is_timedelta = False
    if needs_i8_conversion(arr.dtype):
        dtype = np.int64
        arr = arr.view("i8")
        na = iNaT
        is_timedelta = True

    elif is_bool:
        # We have to cast in order to be able to hold np.nan
        dtype = np.object_

    elif is_integer_dtype(dtype):
        # We have to cast in order to be able to hold np.nan

        # int8, int16 are incompatible with float64,
        # see https://github.com/cython/cython/issues/2646
        if arr.dtype.name in ["int8", "int16"]:
            dtype = np.float32
        else:
            dtype = np.float64

    orig_ndim = arr.ndim
    if orig_ndim == 1:
        # reshape so we can always use algos.diff_2d
        arr = arr.reshape(-1, 1)
        # TODO: require axis == 0

    dtype = np.dtype(dtype)
    out_arr = np.empty(arr.shape, dtype=dtype)

    na_indexer = [slice(None)] * 2
    na_indexer[axis] = slice(None, n) if n >= 0 else slice(n, None)
    out_arr[tuple(na_indexer)] = na

    if arr.dtype.name in _diff_special:
        # TODO: can diff_2d dtype specialization troubles be fixed by defining
        #  out_arr inside diff_2d?
        algos.diff_2d(arr, out_arr, n, axis, datetimelike=is_timedelta)
    else:
        # To keep mypy happy, _res_indexer is a list while res_indexer is
        #  a tuple, ditto for lag_indexer.
        _res_indexer = [slice(None)] * 2
        _res_indexer[axis] = slice(n, None) if n >= 0 else slice(None, n)
        res_indexer = tuple(_res_indexer)

        _lag_indexer = [slice(None)] * 2
        _lag_indexer[axis] = slice(None, -n) if n > 0 else slice(-n, None)
        lag_indexer = tuple(_lag_indexer)

        out_arr[res_indexer] = op(arr[res_indexer], arr[lag_indexer])

    if is_timedelta:
        out_arr = out_arr.view("timedelta64[ns]")

    if orig_ndim == 1:
        out_arr = out_arr[:, 0]
    return out_arr


# --------------------------------------------------------------------
# Helper functions

# Note: safe_sort is in algorithms.py instead of sorting.py because it is
#  low-dependency, is used in this module, and used private methods from
#  this module.
def safe_sort(
    values,
    codes=None,
    na_sentinel: int = -1,
    assume_unique: bool = False,
    verify: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Sort ``values`` and reorder corresponding ``codes``.

    ``values`` should be unique if ``codes`` is not None.
    Safe for use with mixed types (int, str), orders ints before strs.

    Parameters
    ----------
    values : list-like
        Sequence; must be unique if ``codes`` is not None.
    codes : list_like, optional
        Indices to ``values``. All out of bound indices are treated as
        "not found" and will be masked with ``na_sentinel``.
    na_sentinel : int, default -1
        Value in ``codes`` to mark "not found".
        Ignored when ``codes`` is None.
    assume_unique : bool, default False
        When True, ``values`` are assumed to be unique, which can speed up
        the calculation. Ignored when ``codes`` is None.
    verify : bool, default True
        Check if codes are out of bound for the values and put out of bound
        codes equal to na_sentinel. If ``verify=False``, it is assumed there
        are no out of bound codes. Ignored when ``codes`` is None.

        .. versionadded:: 0.25.0

    Returns
    -------
    ordered : ndarray
        Sorted ``values``
    new_codes : ndarray
        Reordered ``codes``; returned when ``codes`` is not None.

    Raises
    ------
    TypeError
        * If ``values`` is not list-like or if ``codes`` is neither None
        nor list-like
        * If ``values`` cannot be sorted
    ValueError
        * If ``codes`` is not None and ``values`` contain duplicates.
    """
    if not is_list_like(values):
        raise TypeError(
            "Only list-like objects are allowed to be passed to safe_sort as values"
        )

    if not isinstance(values, (np.ndarray, ABCExtensionArray)):
        # don't convert to string types
        dtype, _ = infer_dtype_from_array(values)
        # error: Argument "dtype" to "asarray" has incompatible type "Union[dtype[Any],
        # ExtensionDtype]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]], List[Any],
        # _DTypeDict, Tuple[Any, Any]]]"
        values = np.asarray(values, dtype=dtype)  # type: ignore[arg-type]

    sorter = None

    if (
        not is_extension_array_dtype(values)
        and lib.infer_dtype(values, skipna=False) == "mixed-integer"
    ):
        ordered = _sort_mixed(values)
    else:
        try:
            sorter = values.argsort()
            ordered = values.take(sorter)
        except TypeError:
            # Previous sorters failed or were not applicable, try `_sort_mixed`
            # which would work, but which fails for special case of 1d arrays
            # with tuples.
            if values.size and isinstance(values[0], tuple):
                ordered = _sort_tuples(values)
            else:
                ordered = _sort_mixed(values)

    # codes:

    if codes is None:
        return ordered

    if not is_list_like(codes):
        raise TypeError(
            "Only list-like objects or None are allowed to "
            "be passed to safe_sort as codes"
        )
    codes = ensure_platform_int(np.asarray(codes))

    if not assume_unique and not len(unique(values)) == len(values):
        raise ValueError("values should be unique if codes is not None")

    if sorter is None:
        # mixed types
        hash_klass, values = get_data_algo(values)
        t = hash_klass(len(values))
        t.map_locations(values)
        sorter = ensure_platform_int(t.lookup(ordered))

    if na_sentinel == -1:
        # take_nd is faster, but only works for na_sentinels of -1
        order2 = sorter.argsort()
        new_codes = take_nd(order2, codes, fill_value=-1)
        if verify:
            mask = (codes < -len(values)) | (codes >= len(values))
        else:
            mask = None
    else:
        reverse_indexer = np.empty(len(sorter), dtype=np.int_)
        reverse_indexer.put(sorter, np.arange(len(sorter)))
        # Out of bound indices will be masked with `na_sentinel` next, so we
        # may deal with them here without performance loss using `mode='wrap'`
        new_codes = reverse_indexer.take(codes, mode="wrap")

        mask = codes == na_sentinel
        if verify:
            mask = mask | (codes < -len(values)) | (codes >= len(values))

    if mask is not None:
        np.putmask(new_codes, mask, na_sentinel)

    return ordered, ensure_platform_int(new_codes)


def _sort_mixed(values) -> np.ndarray:
    """order ints before strings in 1d arrays, safe in py3"""
    str_pos = np.array([isinstance(x, str) for x in values], dtype=bool)
    nums = np.sort(values[~str_pos])
    strs = np.sort(values[str_pos])
    return np.concatenate([nums, np.asarray(strs, dtype=object)])


def _sort_tuples(values: np.ndarray) -> np.ndarray:
    """
    Convert array of tuples (1d) to array or array (2d).
    We need to keep the columns separately as they contain different types and
    nans (can't use `np.sort` as it may fail when str and nan are mixed in a
    column as types cannot be compared).
    """
    from pandas.core.internals.construction import to_arrays
    from pandas.core.sorting import lexsort_indexer

    arrays, _ = to_arrays(values, None)
    indexer = lexsort_indexer(arrays, orders=True)
    return values[indexer]


def union_with_duplicates(lvals: ArrayLike, rvals: ArrayLike) -> ArrayLike:
    """
    Extracts the union from lvals and rvals with respect to duplicates and nans in
    both arrays.

    Parameters
    ----------
    lvals: np.ndarray or ExtensionArray
        left values which is ordered in front.
    rvals: np.ndarray or ExtensionArray
        right values ordered after lvals.

    Returns
    -------
    np.ndarray or ExtensionArray
        Containing the unsorted union of both arrays.
    """
    indexer = []
    l_count = value_counts(lvals, dropna=False)
    r_count = value_counts(rvals, dropna=False)
    l_count, r_count = l_count.align(r_count, fill_value=0)
    unique_array = unique(np.append(lvals, rvals))
    if not isinstance(lvals, np.ndarray):
        # i.e. ExtensionArray
        # Note: we only get here with lvals.dtype == rvals.dtype
        # TODO: are there any cases where union won't be type/dtype preserving?
        unique_array = type(lvals)._from_sequence(unique_array, dtype=lvals.dtype)
    for i, value in enumerate(unique_array):
        indexer += [i] * int(max(l_count[value], r_count[value]))
    return unique_array.take(indexer)
