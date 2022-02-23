"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

import operator
import ctypes as ct
from typing import Any

import numpy as np

class SubClass(np.ndarray): ...

f8: np.float64
B: SubClass
AR_f8: np.ndarray[Any, np.dtype[np.float64]]
AR_i8: np.ndarray[Any, np.dtype[np.int64]]
AR_U: np.ndarray[Any, np.dtype[np.str_]]

ctypes_obj = AR_f8.ctypes

reveal_type(ctypes_obj.data)  # E: int
reveal_type(ctypes_obj.shape)  # E: ctypes.Array[ctypes.c_int64]
reveal_type(ctypes_obj.strides)  # E: ctypes.Array[ctypes.c_int64]
reveal_type(ctypes_obj._as_parameter_)  # E: ctypes.c_void_p

reveal_type(ctypes_obj.data_as(ct.c_void_p))  # E: ctypes.c_void_p
reveal_type(ctypes_obj.shape_as(ct.c_longlong))  # E: ctypes.Array[ctypes.c_longlong]
reveal_type(ctypes_obj.strides_as(ct.c_ubyte))  # E: ctypes.Array[ctypes.c_ubyte]

reveal_type(f8.all())  # E: numpy.bool_
reveal_type(AR_f8.all())  # E: numpy.bool_
reveal_type(AR_f8.all(axis=0))  # E: Any
reveal_type(AR_f8.all(keepdims=True))  # E: Any
reveal_type(AR_f8.all(out=B))  # E: SubClass

reveal_type(f8.any())  # E: numpy.bool_
reveal_type(AR_f8.any())  # E: numpy.bool_
reveal_type(AR_f8.any(axis=0))  # E: Any
reveal_type(AR_f8.any(keepdims=True))  # E: Any
reveal_type(AR_f8.any(out=B))  # E: SubClass

reveal_type(f8.argmax())  # E: {intp}
reveal_type(AR_f8.argmax())  # E: {intp}
reveal_type(AR_f8.argmax(axis=0))  # E: Any
reveal_type(AR_f8.argmax(out=B))  # E: SubClass

reveal_type(f8.argmin())  # E: {intp}
reveal_type(AR_f8.argmin())  # E: {intp}
reveal_type(AR_f8.argmin(axis=0))  # E: Any
reveal_type(AR_f8.argmin(out=B))  # E: SubClass

reveal_type(f8.argsort())  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.argsort())  # E: numpy.ndarray[Any, Any]

reveal_type(f8.astype(np.int64).choose([()]))  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.choose([0]))  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.choose([0], out=B))  # E: SubClass

reveal_type(f8.clip(1))  # E: Any
reveal_type(AR_f8.clip(1))  # E: Any
reveal_type(AR_f8.clip(None, 1))  # E: Any
reveal_type(AR_f8.clip(1, out=B))  # E: SubClass
reveal_type(AR_f8.clip(None, 1, out=B))  # E: SubClass

reveal_type(f8.compress([0]))  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.compress([0]))  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.compress([0], out=B))  # E: SubClass

reveal_type(f8.conj())  # E: {float64}
reveal_type(AR_f8.conj())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(B.conj())  # E: SubClass

reveal_type(f8.conjugate())  # E: {float64}
reveal_type(AR_f8.conjugate())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(B.conjugate())  # E: SubClass

reveal_type(f8.cumprod())  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.cumprod())  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.cumprod(out=B))  # E: SubClass

reveal_type(f8.cumsum())  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.cumsum())  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.cumsum(out=B))  # E: SubClass

reveal_type(f8.max())  # E: Any
reveal_type(AR_f8.max())  # E: Any
reveal_type(AR_f8.max(axis=0))  # E: Any
reveal_type(AR_f8.max(keepdims=True))  # E: Any
reveal_type(AR_f8.max(out=B))  # E: SubClass

reveal_type(f8.mean())  # E: Any
reveal_type(AR_f8.mean())  # E: Any
reveal_type(AR_f8.mean(axis=0))  # E: Any
reveal_type(AR_f8.mean(keepdims=True))  # E: Any
reveal_type(AR_f8.mean(out=B))  # E: SubClass

reveal_type(f8.min())  # E: Any
reveal_type(AR_f8.min())  # E: Any
reveal_type(AR_f8.min(axis=0))  # E: Any
reveal_type(AR_f8.min(keepdims=True))  # E: Any
reveal_type(AR_f8.min(out=B))  # E: SubClass

reveal_type(f8.newbyteorder())  # E: {float64}
reveal_type(AR_f8.newbyteorder())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(B.newbyteorder('|'))  # E: SubClass

reveal_type(f8.prod())  # E: Any
reveal_type(AR_f8.prod())  # E: Any
reveal_type(AR_f8.prod(axis=0))  # E: Any
reveal_type(AR_f8.prod(keepdims=True))  # E: Any
reveal_type(AR_f8.prod(out=B))  # E: SubClass

reveal_type(f8.ptp())  # E: Any
reveal_type(AR_f8.ptp())  # E: Any
reveal_type(AR_f8.ptp(axis=0))  # E: Any
reveal_type(AR_f8.ptp(keepdims=True))  # E: Any
reveal_type(AR_f8.ptp(out=B))  # E: SubClass

reveal_type(f8.round())  # E: {float64}
reveal_type(AR_f8.round())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(AR_f8.round(out=B))  # E: SubClass

reveal_type(f8.repeat(1))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(AR_f8.repeat(1))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(B.repeat(1))  # E: numpy.ndarray[Any, Any]

reveal_type(f8.std())  # E: Any
reveal_type(AR_f8.std())  # E: Any
reveal_type(AR_f8.std(axis=0))  # E: Any
reveal_type(AR_f8.std(keepdims=True))  # E: Any
reveal_type(AR_f8.std(out=B))  # E: SubClass

reveal_type(f8.sum())  # E: Any
reveal_type(AR_f8.sum())  # E: Any
reveal_type(AR_f8.sum(axis=0))  # E: Any
reveal_type(AR_f8.sum(keepdims=True))  # E: Any
reveal_type(AR_f8.sum(out=B))  # E: SubClass

reveal_type(f8.take(0))  # E: {float64}
reveal_type(AR_f8.take(0))  # E: {float64}
reveal_type(AR_f8.take([0]))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(AR_f8.take(0, out=B))  # E: SubClass
reveal_type(AR_f8.take([0], out=B))  # E: SubClass

reveal_type(f8.var())  # E: Any
reveal_type(AR_f8.var())  # E: Any
reveal_type(AR_f8.var(axis=0))  # E: Any
reveal_type(AR_f8.var(keepdims=True))  # E: Any
reveal_type(AR_f8.var(out=B))  # E: SubClass

reveal_type(AR_f8.argpartition([0]))  # E: numpy.ndarray[Any, numpy.dtype[{intp}]]

reveal_type(AR_f8.diagonal())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]

reveal_type(AR_f8.dot(1))  # E: numpy.ndarray[Any, Any]
reveal_type(AR_f8.dot([1]))  # E: Any
reveal_type(AR_f8.dot(1, out=B))  # E: SubClass

reveal_type(AR_f8.nonzero())  # E: tuple[numpy.ndarray[Any, numpy.dtype[{intp}]]]

reveal_type(AR_f8.searchsorted(1))  # E: {intp}
reveal_type(AR_f8.searchsorted([1]))  # E: numpy.ndarray[Any, numpy.dtype[{intp}]]

reveal_type(AR_f8.trace())  # E: Any
reveal_type(AR_f8.trace(out=B))  # E: SubClass

reveal_type(AR_f8.item())  # E: float
reveal_type(AR_U.item())  # E: str

reveal_type(AR_f8.ravel())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(AR_U.ravel())  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]

reveal_type(AR_f8.flatten())  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(AR_U.flatten())  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]

reveal_type(AR_f8.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(AR_U.reshape(1))  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]

reveal_type(int(AR_f8))  # E: int
reveal_type(int(AR_U))  # E: int

reveal_type(float(AR_f8))  # E: float
reveal_type(float(AR_U))  # E: float

reveal_type(complex(AR_f8))  # E: complex

reveal_type(operator.index(AR_i8))  # E: int
