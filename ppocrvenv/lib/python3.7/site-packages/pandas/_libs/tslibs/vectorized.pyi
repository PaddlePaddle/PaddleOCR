"""
For cython types that cannot be represented precisely, closest-available
python equivalents are used, and the precise types kept as adjacent comments.
"""
from datetime import tzinfo

import numpy as np

from pandas._libs.tslibs.dtypes import Resolution
from pandas._libs.tslibs.offsets import BaseOffset

def dt64arr_to_periodarr(
    stamps: np.ndarray,  # const int64_t[:]
    freq: int,
    tz: tzinfo | None,
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=1]
def is_date_array_normalized(
    stamps: np.ndarray,  # const int64_t[:]
    tz: tzinfo | None = None,
) -> bool: ...
def normalize_i8_timestamps(
    stamps: np.ndarray,  # const int64_t[:]
    tz: tzinfo | None,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def get_resolution(
    stamps: np.ndarray,  # const int64_t[:]
    tz: tzinfo | None = None,
) -> Resolution: ...
def ints_to_pydatetime(
    arr: np.ndarray,  # const int64_t[:}]
    tz: tzinfo | None = None,
    freq: str | BaseOffset | None = None,
    fold: bool = False,
    box: str = "datetime",
) -> np.ndarray: ...  # np.ndarray[object]
