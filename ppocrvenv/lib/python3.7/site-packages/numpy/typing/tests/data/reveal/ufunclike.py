from typing import List, Any
import numpy as np

AR_LIKE_b: List[bool]
AR_LIKE_u: List[np.uint32]
AR_LIKE_i: List[int]
AR_LIKE_f: List[float]
AR_LIKE_O: List[np.object_]

AR_U: np.ndarray[Any, np.dtype[np.str_]]

reveal_type(np.fix(AR_LIKE_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.fix(AR_LIKE_u))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.fix(AR_LIKE_i))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.fix(AR_LIKE_f))  # E: numpy.ndarray[Any, numpy.dtype[numpy.floating[Any]]]
reveal_type(np.fix(AR_LIKE_O))  # E: Any
reveal_type(np.fix(AR_LIKE_f, out=AR_U))  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]

reveal_type(np.isposinf(AR_LIKE_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isposinf(AR_LIKE_u))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isposinf(AR_LIKE_i))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isposinf(AR_LIKE_f))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isposinf(AR_LIKE_f, out=AR_U))  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]

reveal_type(np.isneginf(AR_LIKE_b))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isneginf(AR_LIKE_u))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isneginf(AR_LIKE_i))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isneginf(AR_LIKE_f))  # E: numpy.ndarray[Any, numpy.dtype[numpy.bool_]]
reveal_type(np.isneginf(AR_LIKE_f, out=AR_U))  # E: numpy.ndarray[Any, numpy.dtype[numpy.str_]]
