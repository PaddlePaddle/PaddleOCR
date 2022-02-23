import numpy as np

def unstack(
    values: np.ndarray,  # reshape_t[:, :]
    mask: np.ndarray,  # const uint8_t[:]
    stride: int,
    length: int,
    width: int,
    new_values: np.ndarray,  # reshape_t[:, :]
    new_mask: np.ndarray,  # uint8_t[:, :]
) -> None: ...
def explode(
    values: np.ndarray,  # np.ndarray[object]
) -> tuple[np.ndarray, np.ndarray,]: ...  # np.ndarray[object]  # np.ndarray[np.int64]
