from typing import Any
import numpy as np

a: np.flatiter[np.ndarray[Any, np.dtype[np.str_]]]

reveal_type(a.base)  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(a.copy())  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(a.coords)  # E: tuple[builtins.int]
reveal_type(a.index)  # E: int
reveal_type(iter(a))  # E: Iterator[numpy.str_]
reveal_type(next(a))  # E: numpy.str_
reveal_type(a[0])  # E: numpy.str_
reveal_type(a[[0, 1, 2]])  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(a[...])  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(a[:])  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(a.__array__())  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
reveal_type(a.__array__(np.dtype(np.float64)))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
