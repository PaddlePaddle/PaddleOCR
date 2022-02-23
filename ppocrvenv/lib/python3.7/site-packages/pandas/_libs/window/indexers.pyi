import numpy as np

def calculate_variable_window_bounds(
    num_values: int,  # int64_t
    window_size: int,  # int64_t
    min_periods,
    center: bool,
    closed: str | None,
    index: np.ndarray,  # const int64_t[:]
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[np.int64]  # np.ndarray[np.int64]
