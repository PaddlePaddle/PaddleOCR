import numpy as np
from scipy._lib._util import IntNumber
from typing_extensions import Literal

def initialize_v(
    v : np.ndarray, 
    dim : IntNumber
) -> None: ...

def _cscramble (
    dim : IntNumber,
    ltm : np.ndarray,
    sv: np.ndarray
) -> None: ...

def _fill_p_cumulative(
    p: np.ndarray,
    p_cumulative: np.ndarray
) -> None: ...

def _draw(
    n : IntNumber,
    num_gen: IntNumber,
    dim: IntNumber,
    sv: np.ndarray,
    quasi: np.ndarray,
    result: np.ndarray
    ) -> None: ...

def _fast_forward(
    n: IntNumber,
    num_gen: IntNumber,
    dim: IntNumber,
    sv: np.ndarray,
    quasi: np.ndarray
    ) -> None: ...

def _categorize(
    draws: np.ndarray,
    p_cumulative: np.ndarray,
    result: np.ndarray
    ) -> None: ...

def initialize_direction_numbers() -> None: ...

_MAXDIM: Literal[21201]
_MAXBIT: Literal[30]
_MAXDEG: Literal[18]

def _test_find_index(
    p_cumulative: np.ndarray, 
    size: int, 
    value: float
    ) -> int: ...