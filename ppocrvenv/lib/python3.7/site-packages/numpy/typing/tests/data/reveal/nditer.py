import copy
import numpy as np

nditer_obj: np.nditer

with nditer_obj as context:
    reveal_type(context)   # E: numpy.nditer

reveal_type(len(nditer_obj))  # E: builtins.int
reveal_type(copy.copy(nditer_obj))  # E: numpy.nditer
reveal_type(next(nditer_obj))  # E: Any
reveal_type(iter(nditer_obj))  # E: typing.Iterator[Any]
reveal_type(nditer_obj[1])  # E: Any
reveal_type(nditer_obj[1:5])  # E: Any

nditer_obj[1] = 1
nditer_obj[1:5] = 1
del nditer_obj[1]
del nditer_obj[1:5]
