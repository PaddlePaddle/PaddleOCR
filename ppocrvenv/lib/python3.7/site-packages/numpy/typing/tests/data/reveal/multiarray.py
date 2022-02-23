from typing import Any
import numpy as np

AR_f8: np.ndarray[Any, np.dtype[np.float64]]
AR_i8: np.ndarray[Any, np.dtype[np.int64]]

b_f8 = np.broadcast(AR_f8)
b_i8_f8_f8 = np.broadcast(AR_i8, AR_f8, AR_f8)

reveal_type(next(b_f8))  # E: tuple[Any]
reveal_type(next(b_i8_f8_f8))  # E: tuple[Any]

reveal_type(b_f8.reset())  # E: None
reveal_type(b_i8_f8_f8.reset())  # E: None

reveal_type(b_f8.index)  # E: int
reveal_type(b_i8_f8_f8.index)  # E: int

reveal_type(b_f8.iters)  # E: tuple[numpy.flatiter[Any]]
reveal_type(b_i8_f8_f8.iters)  # E: tuple[numpy.flatiter[Any]]

reveal_type(b_f8.nd)  # E: int
reveal_type(b_i8_f8_f8.nd)  # E: int

reveal_type(b_f8.ndim)  # E: int
reveal_type(b_i8_f8_f8.ndim)  # E: int

reveal_type(b_f8.numiter)  # E: int
reveal_type(b_i8_f8_f8.numiter)  # E: int

reveal_type(b_f8.shape)  # E: tuple[builtins.int]
reveal_type(b_i8_f8_f8.shape)  # E: tuple[builtins.int]

reveal_type(b_f8.size)  # E: int
reveal_type(b_i8_f8_f8.size)  # E: int
