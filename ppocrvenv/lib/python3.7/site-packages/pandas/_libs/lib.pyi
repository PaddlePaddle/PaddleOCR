# TODO(npdtypes): Many types specified here can be made more specific/accurate;
#  the more specific versions are specified in comments

from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    overload,
)

import numpy as np

from pandas._typing import (
    ArrayLike,
    DtypeObj,
)

# placeholder until we can specify np.ndarray[object, ndim=2]
ndarray_obj_2d = np.ndarray

from enum import Enum

class NoDefault(Enum): ...

no_default: NoDefault

i8max: int
u8max: int

def item_from_zerodim(val: object) -> object: ...
def infer_dtype(value: object, skipna: bool = True) -> str: ...
def is_iterator(obj: object) -> bool: ...
def is_scalar(val: object) -> bool: ...
def is_list_like(obj: object, allow_sets: bool = True) -> bool: ...
def is_period(val: object) -> bool: ...
def is_interval(val: object) -> bool: ...
def is_decimal(val: object) -> bool: ...
def is_complex(val: object) -> bool: ...
def is_bool(val: object) -> bool: ...
def is_integer(val: object) -> bool: ...
def is_float(val: object) -> bool: ...
def is_interval_array(values: np.ndarray) -> bool: ...
def is_datetime64_array(values: np.ndarray) -> bool: ...
def is_timedelta_or_timedelta64_array(values: np.ndarray) -> bool: ...
def is_datetime_with_singletz_array(values: np.ndarray) -> bool: ...
def is_time_array(values: np.ndarray, skipna: bool = False): ...
def is_date_array(values: np.ndarray, skipna: bool = False): ...
def is_datetime_array(values: np.ndarray, skipna: bool = False): ...
def is_string_array(values: np.ndarray, skipna: bool = False): ...
def is_float_array(values: np.ndarray, skipna: bool = False): ...
def is_integer_array(values: np.ndarray, skipna: bool = False): ...
def is_bool_array(values: np.ndarray, skipna: bool = False): ...
def fast_multiget(mapping: dict, keys: np.ndarray, default=np.nan) -> np.ndarray: ...
def fast_unique_multiple_list_gen(gen: Generator, sort: bool = True) -> list: ...
def fast_unique_multiple_list(lists: list, sort: bool = True) -> list: ...
def fast_unique_multiple(arrays: list, sort: bool = True) -> list: ...
def map_infer(
    arr: np.ndarray,
    f: Callable[[Any], Any],
    convert: bool = True,
    ignore_na: bool = False,
) -> np.ndarray: ...
@overload  # both convert_datetime and convert_to_nullable_integer False -> np.ndarray
def maybe_convert_objects(
    objects: np.ndarray,  # np.ndarray[object]
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_datetime: Literal[False] = ...,
    convert_timedelta: bool = ...,
    convert_period: Literal[False] = ...,
    convert_interval: Literal[False] = ...,
    convert_to_nullable_integer: Literal[False] = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> np.ndarray: ...
@overload
def maybe_convert_objects(
    objects: np.ndarray,  # np.ndarray[object]
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_datetime: bool = ...,
    convert_timedelta: bool = ...,
    convert_period: bool = ...,
    convert_interval: bool = ...,
    convert_to_nullable_integer: Literal[True] = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> ArrayLike: ...
@overload
def maybe_convert_objects(
    objects: np.ndarray,  # np.ndarray[object]
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_datetime: Literal[True] = ...,
    convert_timedelta: bool = ...,
    convert_period: bool = ...,
    convert_interval: bool = ...,
    convert_to_nullable_integer: bool = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> ArrayLike: ...
@overload
def maybe_convert_objects(
    objects: np.ndarray,  # np.ndarray[object]
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_datetime: bool = ...,
    convert_timedelta: bool = ...,
    convert_period: Literal[True] = ...,
    convert_interval: bool = ...,
    convert_to_nullable_integer: bool = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> ArrayLike: ...
@overload
def maybe_convert_objects(
    objects: np.ndarray,  # np.ndarray[object]
    *,
    try_float: bool = ...,
    safe: bool = ...,
    convert_datetime: bool = ...,
    convert_timedelta: bool = ...,
    convert_period: bool = ...,
    convert_interval: bool = ...,
    convert_to_nullable_integer: bool = ...,
    dtype_if_all_nat: DtypeObj | None = ...,
) -> ArrayLike: ...
@overload
def maybe_convert_numeric(
    values: np.ndarray,  # np.ndarray[object]
    na_values: set,
    convert_empty: bool = True,
    coerce_numeric: bool = False,
    convert_to_masked_nullable: Literal[False] = ...,
) -> tuple[np.ndarray, None]: ...
@overload
def maybe_convert_numeric(
    values: np.ndarray,  # np.ndarray[object]
    na_values: set,
    convert_empty: bool = True,
    coerce_numeric: bool = False,
    *,
    convert_to_masked_nullable: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...

# TODO: restrict `arr`?
def ensure_string_array(
    arr,
    na_value: object = np.nan,
    convert_na_value: bool = True,
    copy: bool = True,
    skipna: bool = True,
) -> np.ndarray: ...  # np.ndarray[object]
def infer_datetimelike_array(
    arr: np.ndarray,  # np.ndarray[object]
) -> tuple[str, bool]: ...
def astype_intsafe(
    arr: np.ndarray,  # np.ndarray[object]
    new_dtype: np.dtype,
) -> np.ndarray: ...
def fast_zip(ndarrays: list) -> np.ndarray: ...  # np.ndarray[object]

# TODO: can we be more specific about rows?
def to_object_array_tuples(rows: object) -> ndarray_obj_2d: ...
def tuples_to_object_array(
    tuples: np.ndarray,  # np.ndarray[object]
) -> ndarray_obj_2d: ...

# TODO: can we be more specific about rows?
def to_object_array(rows: object, min_width: int = 0) -> ndarray_obj_2d: ...
def dicts_to_array(dicts: list, columns: list) -> ndarray_obj_2d: ...
def maybe_booleans_to_slice(
    mask: np.ndarray,  # ndarray[uint8_t]
) -> slice | np.ndarray: ...  # np.ndarray[np.uint8]
def maybe_indices_to_slice(
    indices: np.ndarray,  # np.ndarray[np.intp]
    max_len: int,
) -> slice | np.ndarray: ...  # np.ndarray[np.uint8]
def is_all_arraylike(obj: list) -> bool: ...

# -----------------------------------------------------------------
# Functions which in reality take memoryviews

def memory_usage_of_objects(arr: np.ndarray) -> int: ...  # object[:]  # np.int64
def map_infer_mask(
    arr: np.ndarray,
    f: Callable[[Any], Any],
    mask: np.ndarray,  # const uint8_t[:]
    convert: bool = ...,
    na_value: Any = ...,
    dtype: np.dtype = ...,
) -> np.ndarray: ...
def indices_fast(
    index: np.ndarray,  # ndarray[intp_t]
    labels: np.ndarray,  # const int64_t[:]
    keys: list,
    sorted_labels: list[np.ndarray],  # list[ndarray[np.int64]]
) -> dict: ...
def generate_slices(
    labels: np.ndarray, ngroups: int  # const intp_t[:]
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.int64]  # np.ndarray[np.int64]
def count_level_2d(
    mask: np.ndarray,  # ndarray[uint8_t, ndim=2, cast=True],
    labels: np.ndarray,  # const intp_t[:]
    max_bin: int,
    axis: int,
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=2]
def get_level_sorter(
    label: np.ndarray,  # const int64_t[:]
    starts: np.ndarray,  # const intp_t[:]
) -> np.ndarray: ...  #  np.ndarray[np.intp, ndim=1]
def generate_bins_dt64(
    values: np.ndarray,  # np.ndarray[np.int64]
    binner: np.ndarray,  # const int64_t[:]
    closed: object = "left",
    hasnans: bool = False,
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=1]
def array_equivalent_object(
    left: np.ndarray,  # object[:]
    right: np.ndarray,  # object[:]
) -> bool: ...
def has_infs_f8(arr: np.ndarray) -> bool: ...  # const float64_t[:]
def has_infs_f4(arr: np.ndarray) -> bool: ...  # const float32_t[:]
def get_reverse_indexer(
    indexer: np.ndarray,  # const intp_t[:]
    length: int,
) -> np.ndarray: ...  # np.ndarray[np.intp]
def is_bool_list(obj: list) -> bool: ...
