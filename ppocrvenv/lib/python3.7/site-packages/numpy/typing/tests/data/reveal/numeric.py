"""
Tests for :mod:`numpy.core.numeric`.

Does not include tests which fall under ``array_constructors``.

"""

from typing import List
import numpy as np

class SubClass(np.ndarray):
    ...

i8: np.int64

A: np.ndarray
B: List[int]
C: SubClass

reveal_type(np.count_nonzero(i8))  # E: int
reveal_type(np.count_nonzero(A))  # E: int
reveal_type(np.count_nonzero(B))  # E: int
reveal_type(np.count_nonzero(A, keepdims=True))  # E: Any
reveal_type(np.count_nonzero(A, axis=0))  # E: Any

reveal_type(np.isfortran(i8))  # E: bool
reveal_type(np.isfortran(A))  # E: bool

reveal_type(np.argwhere(i8))  # E: numpy.ndarray[Any, Any]
reveal_type(np.argwhere(A))  # E: numpy.ndarray[Any, Any]

reveal_type(np.flatnonzero(i8))  # E: numpy.ndarray[Any, Any]
reveal_type(np.flatnonzero(A))  # E: numpy.ndarray[Any, Any]

reveal_type(np.correlate(B, A, mode="valid"))  # E: numpy.ndarray[Any, Any]
reveal_type(np.correlate(A, A, mode="same"))  # E: numpy.ndarray[Any, Any]

reveal_type(np.convolve(B, A, mode="valid"))  # E: numpy.ndarray[Any, Any]
reveal_type(np.convolve(A, A, mode="same"))  # E: numpy.ndarray[Any, Any]

reveal_type(np.outer(i8, A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.outer(B, A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.outer(A, A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.outer(A, A, out=C))  # E: SubClass

reveal_type(np.tensordot(B, A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.tensordot(A, A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.tensordot(A, A, axes=0))  # E: numpy.ndarray[Any, Any]
reveal_type(np.tensordot(A, A, axes=(0, 1)))  # E: numpy.ndarray[Any, Any]

reveal_type(np.isscalar(i8))  # E: bool
reveal_type(np.isscalar(A))  # E: bool
reveal_type(np.isscalar(B))  # E: bool

reveal_type(np.roll(A, 1))  # E: numpy.ndarray[Any, Any]
reveal_type(np.roll(A, (1, 2)))  # E: numpy.ndarray[Any, Any]
reveal_type(np.roll(B, 1))  # E: numpy.ndarray[Any, Any]

reveal_type(np.rollaxis(A, 0, 1))  # E: numpy.ndarray[Any, Any]

reveal_type(np.moveaxis(A, 0, 1))  # E: numpy.ndarray[Any, Any]
reveal_type(np.moveaxis(A, (0, 1), (1, 2)))  # E: numpy.ndarray[Any, Any]

reveal_type(np.cross(B, A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.cross(A, A))  # E: numpy.ndarray[Any, Any]

reveal_type(np.indices([0, 1, 2]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.indices([0, 1, 2], sparse=False))  # E: numpy.ndarray[Any, Any]
reveal_type(np.indices([0, 1, 2], sparse=True))  # E: tuple[numpy.ndarray[Any, Any]]

reveal_type(np.binary_repr(1))  # E: str

reveal_type(np.base_repr(1))  # E: str

reveal_type(np.allclose(i8, A))  # E: bool
reveal_type(np.allclose(B, A))  # E: bool
reveal_type(np.allclose(A, A))  # E: bool

reveal_type(np.isclose(i8, A))  # E: Any
reveal_type(np.isclose(B, A))  # E: Any
reveal_type(np.isclose(A, A))  # E: Any

reveal_type(np.array_equal(i8, A))  # E: bool
reveal_type(np.array_equal(B, A))  # E: bool
reveal_type(np.array_equal(A, A))  # E: bool

reveal_type(np.array_equiv(i8, A))  # E: bool
reveal_type(np.array_equiv(B, A))  # E: bool
reveal_type(np.array_equiv(A, A))  # E: bool
