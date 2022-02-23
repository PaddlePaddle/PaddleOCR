from typing import Literal

import numpy as np

def group_median_float64(
    out: np.ndarray,  # ndarray[float64_t, ndim=2]
    counts: np.ndarray,  # ndarray[int64_t]
    values: np.ndarray,  # ndarray[float64_t, ndim=2]
    labels: np.ndarray,  # ndarray[int64_t]
    min_count: int = ...,  # Py_ssize_t
) -> None: ...
def group_cumprod_float64(
    out: np.ndarray,  # float64_t[:, ::1]
    values: np.ndarray,  # const float64_t[:, :]
    labels: np.ndarray,  # const int64_t[:]
    ngroups: int,
    is_datetimelike: bool,
    skipna: bool = ...,
) -> None: ...
def group_cumsum(
    out: np.ndarray,  # numeric[:, ::1]
    values: np.ndarray,  # ndarray[numeric, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    ngroups: int,
    is_datetimelike: bool,
    skipna: bool = ...,
) -> None: ...
def group_shift_indexer(
    out: np.ndarray,  # int64_t[::1]
    labels: np.ndarray,  # const int64_t[:]
    ngroups: int,
    periods: int,
) -> None: ...
def group_fillna_indexer(
    out: np.ndarray,  # ndarray[int64_t]
    labels: np.ndarray,  # ndarray[int64_t]
    mask: np.ndarray,  # ndarray[uint8_t]
    direction: Literal["ffill", "bfill"],
    limit: int,  # int64_t
    dropna: bool,
) -> None: ...
def group_any_all(
    out: np.ndarray,  # uint8_t[::1]
    values: np.ndarray,  # const uint8_t[::1]
    labels: np.ndarray,  # const int64_t[:]
    mask: np.ndarray,  # const uint8_t[::1]
    val_test: Literal["any", "all"],
    skipna: bool,
) -> None: ...
def group_add(
    out: np.ndarray,  # complexfloating_t[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[complexfloating_t, ndim=2]
    labels: np.ndarray,  # const intp_t[:]
    min_count: int = ...,
    datetimelike: bool = ...,
) -> None: ...
def group_prod(
    out: np.ndarray,  # floating[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[floating, ndim=2]
    labels: np.ndarray,  # const intp_t[:]
    min_count: int = ...,
) -> None: ...
def group_var(
    out: np.ndarray,  # floating[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[floating, ndim=2]
    labels: np.ndarray,  # const intp_t[:]
    min_count: int = ...,  # Py_ssize_t
    ddof: int = ...,  # int64_t
) -> None: ...
def group_mean(
    out: np.ndarray,  # floating[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[floating, ndim=2]
    labels: np.ndarray,  # const intp_t[:]
    min_count: int = ...,  # Py_ssize_t
    is_datetimelike: bool = ...,  # bint
    mask: np.ndarray | None = ...,
    result_mask: np.ndarray | None = ...,
) -> None: ...
def group_ohlc(
    out: np.ndarray,  # floating[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[floating, ndim=2]
    labels: np.ndarray,  # const intp_t[:]
    min_count: int = ...,
) -> None: ...
def group_quantile(
    out: np.ndarray,  # ndarray[float64_t]
    values: np.ndarray,  # ndarray[numeric, ndim=1]
    labels: np.ndarray,  # ndarray[int64_t]
    mask: np.ndarray,  # ndarray[uint8_t]
    q: float,  # float64_t
    interpolation: Literal["linear", "lower", "higher", "nearest", "midpoint"],
) -> None: ...
def group_last(
    out: np.ndarray,  # rank_t[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[rank_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    min_count: int = ...,  # Py_ssize_t
) -> None: ...
def group_nth(
    out: np.ndarray,  # rank_t[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[rank_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    min_count: int = ...,  # int64_t
    rank: int = ...,  # int64_t
) -> None: ...
def group_rank(
    out: np.ndarray,  # float64_t[:, ::1]
    values: np.ndarray,  # ndarray[rank_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    ngroups: int,
    is_datetimelike: bool,
    ties_method: Literal["aveage", "min", "max", "first", "dense"] = ...,
    ascending: bool = ...,
    pct: bool = ...,
    na_option: Literal["keep", "top", "bottom"] = ...,
) -> None: ...
def group_max(
    out: np.ndarray,  # groupby_t[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[groupby_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    min_count: int = ...,
) -> None: ...
def group_min(
    out: np.ndarray,  # groupby_t[:, ::1]
    counts: np.ndarray,  # int64_t[::1]
    values: np.ndarray,  # ndarray[groupby_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    min_count: int = ...,
) -> None: ...
def group_cummin(
    out: np.ndarray,  # groupby_t[:, ::1]
    values: np.ndarray,  # ndarray[groupby_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    ngroups: int,
    is_datetimelike: bool,
) -> None: ...
def group_cummax(
    out: np.ndarray,  # groupby_t[:, ::1]
    values: np.ndarray,  # ndarray[groupby_t, ndim=2]
    labels: np.ndarray,  # const int64_t[:]
    ngroups: int,
    is_datetimelike: bool,
) -> None: ...
