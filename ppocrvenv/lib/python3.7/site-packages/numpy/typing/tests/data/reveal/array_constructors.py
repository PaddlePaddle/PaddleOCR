from typing import List, Any
import numpy as np

class SubClass(np.ndarray): ...

i8: np.int64

A: np.ndarray
B: SubClass
C: List[int]

def func(i: int, j: int, **kwargs: Any) -> SubClass: ...

reveal_type(np.asarray(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.asarray(B))  # E: numpy.ndarray[Any, Any]
reveal_type(np.asarray(C))  # E: numpy.ndarray[Any, Any]

reveal_type(np.asanyarray(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.asanyarray(B))  # E: SubClass
reveal_type(np.asanyarray(B, dtype=int))  # E: numpy.ndarray[Any, Any]
reveal_type(np.asanyarray(C))  # E: numpy.ndarray[Any, Any]

reveal_type(np.ascontiguousarray(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.ascontiguousarray(B))  # E: numpy.ndarray[Any, Any]
reveal_type(np.ascontiguousarray(C))  # E: numpy.ndarray[Any, Any]

reveal_type(np.asfortranarray(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.asfortranarray(B))  # E: numpy.ndarray[Any, Any]
reveal_type(np.asfortranarray(C))  # E: numpy.ndarray[Any, Any]

reveal_type(np.require(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.require(B))  # E: SubClass
reveal_type(np.require(B, requirements=None))  # E: SubClass
reveal_type(np.require(B, dtype=int))  # E: numpy.ndarray[Any, Any]
reveal_type(np.require(B, requirements="E"))  # E: numpy.ndarray[Any, Any]
reveal_type(np.require(B, requirements=["ENSUREARRAY"]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.require(B, requirements={"F", "E"}))  # E: numpy.ndarray[Any, Any]
reveal_type(np.require(B, requirements=["C", "OWNDATA"]))  # E: SubClass
reveal_type(np.require(B, requirements="W"))  # E: SubClass
reveal_type(np.require(B, requirements="A"))  # E: SubClass
reveal_type(np.require(C))  # E: numpy.ndarray[Any, Any]

reveal_type(np.linspace(0, 10))  # E: numpy.ndarray[Any, Any]
reveal_type(np.linspace(0, 10, retstep=True))  # E: Tuple[numpy.ndarray[Any, Any], Any]
reveal_type(np.logspace(0, 10))  # E: numpy.ndarray[Any, Any]
reveal_type(np.geomspace(1, 10))  # E: numpy.ndarray[Any, Any]

reveal_type(np.zeros_like(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.zeros_like(C))  # E: numpy.ndarray[Any, Any]
reveal_type(np.zeros_like(B))  # E: SubClass
reveal_type(np.zeros_like(B, dtype=np.int64))  # E: numpy.ndarray[Any, Any]

reveal_type(np.ones_like(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.ones_like(C))  # E: numpy.ndarray[Any, Any]
reveal_type(np.ones_like(B))  # E: SubClass
reveal_type(np.ones_like(B, dtype=np.int64))  # E: numpy.ndarray[Any, Any]

reveal_type(np.empty_like(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.empty_like(C))  # E: numpy.ndarray[Any, Any]
reveal_type(np.empty_like(B))  # E: SubClass
reveal_type(np.empty_like(B, dtype=np.int64))  # E: numpy.ndarray[Any, Any]

reveal_type(np.full_like(A, i8))  # E: numpy.ndarray[Any, Any]
reveal_type(np.full_like(C, i8))  # E: numpy.ndarray[Any, Any]
reveal_type(np.full_like(B, i8))  # E: SubClass
reveal_type(np.full_like(B, i8, dtype=np.int64))  # E: numpy.ndarray[Any, Any]

reveal_type(np.ones(1))  # E: numpy.ndarray[Any, Any]
reveal_type(np.ones([1, 1, 1]))  # E: numpy.ndarray[Any, Any]

reveal_type(np.full(1, i8))  # E: numpy.ndarray[Any, Any]
reveal_type(np.full([1, 1, 1], i8))  # E: numpy.ndarray[Any, Any]

reveal_type(np.indices([1, 2, 3]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.indices([1, 2, 3], sparse=True))  # E: tuple[numpy.ndarray[Any, Any]]

reveal_type(np.fromfunction(func, (3, 5)))  # E: SubClass

reveal_type(np.identity(10))  # E: numpy.ndarray[Any, Any]

reveal_type(np.atleast_1d(A))  # E: numpy.ndarray[Any, Any]
reveal_type(np.atleast_1d(C))  # E: numpy.ndarray[Any, Any]
reveal_type(np.atleast_1d(A, A))  # E: list[numpy.ndarray[Any, Any]]
reveal_type(np.atleast_1d(A, C))  # E: list[numpy.ndarray[Any, Any]]
reveal_type(np.atleast_1d(C, C))  # E: list[numpy.ndarray[Any, Any]]

reveal_type(np.atleast_2d(A))  # E: numpy.ndarray[Any, Any]

reveal_type(np.atleast_3d(A))  # E: numpy.ndarray[Any, Any]

reveal_type(np.vstack([A, A]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.vstack([A, C]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.vstack([C, C]))  # E: numpy.ndarray[Any, Any]

reveal_type(np.hstack([A, A]))  # E: numpy.ndarray[Any, Any]

reveal_type(np.stack([A, A]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.stack([A, A], axis=0))  # E: numpy.ndarray[Any, Any]
reveal_type(np.stack([A, A], out=B))  # E: SubClass

reveal_type(np.block([[A, A], [A, A]]))  # E: numpy.ndarray[Any, Any]
reveal_type(np.block(C))  # E: numpy.ndarray[Any, Any]
