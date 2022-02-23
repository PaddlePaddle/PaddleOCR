from typing import List, Any
import numpy as np

AR_LIKE_b: List[bool]
AR_LIKE_u: List[np.uint32]
AR_LIKE_i: List[int]
AR_LIKE_f: List[float]
AR_LIKE_c: List[complex]
AR_LIKE_U: List[str]

OUT_f: np.ndarray[Any, np.dtype[np.float64]]

reveal_type(np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_b))  # E: Any
reveal_type(np.einsum("i,i->i", AR_LIKE_u, AR_LIKE_u))  # E: Any
reveal_type(np.einsum("i,i->i", AR_LIKE_i, AR_LIKE_i))  # E: Any
reveal_type(np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f))  # E: Any
reveal_type(np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c))  # E: Any
reveal_type(np.einsum("i,i->i", AR_LIKE_b, AR_LIKE_i))  # E: Any
reveal_type(np.einsum("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c))  # E: Any

reveal_type(np.einsum("i,i->i", AR_LIKE_c, AR_LIKE_c, out=OUT_f))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]
reveal_type(np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe", out=OUT_f))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]
reveal_type(np.einsum("i,i->i", AR_LIKE_f, AR_LIKE_f, dtype="c16"))  # E: Any
reveal_type(np.einsum("i,i->i", AR_LIKE_U, AR_LIKE_U, dtype=bool, casting="unsafe"))  # E: Any

reveal_type(np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_b))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_u, AR_LIKE_u))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_i, AR_LIKE_i))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_f, AR_LIKE_f))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_c, AR_LIKE_c))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i->i", AR_LIKE_b, AR_LIKE_i))  # E: Tuple[builtins.list[Any], builtins.str]
reveal_type(np.einsum_path("i,i,i,i->i", AR_LIKE_b, AR_LIKE_u, AR_LIKE_i, AR_LIKE_c))  # E: Tuple[builtins.list[Any], builtins.str]
