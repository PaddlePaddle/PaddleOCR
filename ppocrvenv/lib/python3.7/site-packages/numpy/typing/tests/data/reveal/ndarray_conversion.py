import numpy as np
import numpy.typing as npt

nd: npt.NDArray[np.int_] = np.array([[1, 2], [3, 4]])

# item
reveal_type(nd.item())  # E: int
reveal_type(nd.item(1))  # E: int
reveal_type(nd.item(0, 1))  # E: int
reveal_type(nd.item((0, 1)))  # E: int

# tolist
reveal_type(nd.tolist())  # E: Any

# itemset does not return a value
# tostring is pretty simple
# tobytes is pretty simple
# tofile does not return a value
# dump does not return a value
# dumps is pretty simple

# astype
reveal_type(nd.astype("float"))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(nd.astype(float))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(nd.astype(np.float64))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(nd.astype(np.float64, "K"))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(nd.astype(np.float64, "K", "unsafe"))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(nd.astype(np.float64, "K", "unsafe", True))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(nd.astype(np.float64, "K", "unsafe", True, True))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]

# byteswap
reveal_type(nd.byteswap())  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(nd.byteswap(True))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]

# copy
reveal_type(nd.copy())  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(nd.copy("C"))  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]

reveal_type(nd.view())  # E: numpy.ndarray[Any, numpy.dtype[{int_}]]
reveal_type(nd.view(np.float64))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(nd.view(float))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(nd.view(np.float64, np.matrix))  # E: numpy.matrix[Any, Any]

# getfield
reveal_type(nd.getfield("float"))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(nd.getfield(float))  # E: numpy.ndarray[Any, numpy.dtype[Any]]
reveal_type(nd.getfield(np.float64))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]
reveal_type(nd.getfield(np.float64, 8))  # E: numpy.ndarray[Any, numpy.dtype[{float64}]]

# setflags does not return a value
# fill does not return a value
