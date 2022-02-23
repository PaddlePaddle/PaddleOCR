import numpy as np

def inner_join(
    left: np.ndarray,  # const intp_t[:]
    right: np.ndarray,  # const intp_t[:]
    max_groups: int,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def left_outer_join(
    left: np.ndarray,  # const intp_t[:]
    right: np.ndarray,  # const intp_t[:]
    max_groups: int,
    sort: bool = True,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def full_outer_join(
    left: np.ndarray,  # const intp_t[:]
    right: np.ndarray,  # const intp_t[:]
    max_groups: int,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def ffill_indexer(
    indexer: np.ndarray,  # const intp_t[:]
) -> np.ndarray: ...  # np.ndarray[np.intp]
def left_join_indexer_unique(
    left: np.ndarray,  # ndarray[join_t]
    right: np.ndarray,  # ndarray[join_t]
) -> np.ndarray: ...  # np.ndarray[np.intp]
def left_join_indexer(
    left: np.ndarray,  # ndarray[join_t]
    right: np.ndarray,  # ndarray[join_t]
) -> tuple[
    np.ndarray,  # np.ndarray[join_t]
    np.ndarray,  # np.ndarray[np.intp]
    np.ndarray,  # np.ndarray[np.intp]
]: ...
def inner_join_indexer(
    left: np.ndarray,  # ndarray[join_t]
    right: np.ndarray,  # ndarray[join_t]
) -> tuple[
    np.ndarray,  # np.ndarray[join_t]
    np.ndarray,  # np.ndarray[np.intp]
    np.ndarray,  # np.ndarray[np.intp]
]: ...
def outer_join_indexer(
    left: np.ndarray,  # ndarray[join_t]
    right: np.ndarray,  # ndarray[join_t]
) -> tuple[
    np.ndarray,  # np.ndarray[join_t]
    np.ndarray,  # np.ndarray[np.intp]
    np.ndarray,  # np.ndarray[np.intp]
]: ...
def asof_join_backward_on_X_by_Y(
    left_values: np.ndarray,  # asof_t[:]
    right_values: np.ndarray,  # asof_t[:]
    left_by_values: np.ndarray,  # by_t[:]
    right_by_values: np.ndarray,  # by_t[:]
    allow_exact_matches: bool = True,
    tolerance=None,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def asof_join_forward_on_X_by_Y(
    left_values: np.ndarray,  # asof_t[:]
    right_values: np.ndarray,  # asof_t[:]
    left_by_values: np.ndarray,  # by_t[:]
    right_by_values: np.ndarray,  # by_t[:]
    allow_exact_matches: bool = True,
    tolerance=None,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def asof_join_nearest_on_X_by_Y(
    left_values: np.ndarray,  # asof_t[:]
    right_values: np.ndarray,  # asof_t[:]
    left_by_values: np.ndarray,  # by_t[:]
    right_by_values: np.ndarray,  # by_t[:]
    allow_exact_matches: bool = True,
    tolerance=None,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def asof_join_backward(
    left_values: np.ndarray,  # asof_t[:]
    right_values: np.ndarray,  # asof_t[:]
    allow_exact_matches: bool = True,
    tolerance=None,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def asof_join_forward(
    left_values: np.ndarray,  # asof_t[:]
    right_values: np.ndarray,  # asof_t[:]
    allow_exact_matches: bool = True,
    tolerance=None,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
def asof_join_nearest(
    left_values: np.ndarray,  # asof_t[:]
    right_values: np.ndarray,  # asof_t[:]
    allow_exact_matches: bool = True,
    tolerance=None,
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.intp]  # np.ndarray[np.intp]
