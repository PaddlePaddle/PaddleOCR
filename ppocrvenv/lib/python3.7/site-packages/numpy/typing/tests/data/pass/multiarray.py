from __future__ import annotations

from typing import Any
import numpy as np

AR_f8: np.ndarray[Any, np.dtype[np.float64]] = np.array([1.0])
AR_i8: np.ndarray[Any, np.dtype[np.int_]] = np.array([1])

b_f8 = np.broadcast(AR_f8)
b_i8_f8_f8 = np.broadcast(AR_i8, AR_f8, AR_f8)

next(b_f8)
next(b_i8_f8_f8)

b_f8.reset()
b_i8_f8_f8.reset()

b_f8.index
b_i8_f8_f8.index

b_f8.iters
b_i8_f8_f8.iters

b_f8.nd
b_i8_f8_f8.nd

b_f8.ndim
b_i8_f8_f8.ndim

b_f8.numiter
b_i8_f8_f8.numiter

b_f8.shape
b_i8_f8_f8.shape

b_f8.size
b_i8_f8_f8.size
