import ctypes as ct
import numpy as np

dtype_obj: np.dtype[np.str_]
void_dtype_obj: np.dtype[np.void]

reveal_type(np.dtype(np.float64))  # E: numpy.dtype[{float64}]
reveal_type(np.dtype(np.int64))  # E: numpy.dtype[{int64}]

# String aliases
reveal_type(np.dtype("float64"))  # E: numpy.dtype[{float64}]
reveal_type(np.dtype("float32"))  # E: numpy.dtype[{float32}]
reveal_type(np.dtype("int64"))  # E: numpy.dtype[{int64}]
reveal_type(np.dtype("int32"))  # E: numpy.dtype[{int32}]
reveal_type(np.dtype("bool"))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype("bytes"))  # E: numpy.dtype[numpy.bytes_]
reveal_type(np.dtype("str"))  # E: numpy.dtype[numpy.str_]

# Python types
reveal_type(np.dtype(complex))  # E: numpy.dtype[{cdouble}]
reveal_type(np.dtype(float))  # E: numpy.dtype[{double}]
reveal_type(np.dtype(int))  # E: numpy.dtype[{int_}]
reveal_type(np.dtype(bool))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype(str))  # E: numpy.dtype[numpy.str_]
reveal_type(np.dtype(bytes))  # E: numpy.dtype[numpy.bytes_]
reveal_type(np.dtype(object))  # E: numpy.dtype[numpy.object_]

# ctypes
reveal_type(np.dtype(ct.c_double))  # E: numpy.dtype[{double}]
reveal_type(np.dtype(ct.c_longlong))  # E: numpy.dtype[{longlong}]
reveal_type(np.dtype(ct.c_uint32))  # E: numpy.dtype[{uint32}]
reveal_type(np.dtype(ct.c_bool))  # E: numpy.dtype[numpy.bool_]
reveal_type(np.dtype(ct.c_char))  # E: numpy.dtype[numpy.bytes_]
reveal_type(np.dtype(ct.py_object))  # E: numpy.dtype[numpy.object_]

# Special case for None
reveal_type(np.dtype(None))  # E: numpy.dtype[{double}]

# Dtypes of dtypes
reveal_type(np.dtype(np.dtype(np.float64)))  # E: numpy.dtype[{float64}]

# Parameterized dtypes
reveal_type(np.dtype("S8"))  # E: numpy.dtype

# Void
reveal_type(np.dtype(("U", 10)))  # E: numpy.dtype[numpy.void]

# Methods and attributes
reveal_type(dtype_obj.base)  # E: numpy.dtype[numpy.str_]
reveal_type(dtype_obj.subdtype)  # E: Union[Tuple[numpy.dtype[numpy.str_], builtins.tuple[builtins.int]], None]
reveal_type(dtype_obj.newbyteorder())  # E: numpy.dtype[numpy.str_]
reveal_type(dtype_obj.type)  # E: Type[numpy.str_]
reveal_type(dtype_obj.name)  # E: str
reveal_type(dtype_obj.names)  # E: Union[builtins.tuple[builtins.str], None]

reveal_type(dtype_obj * 0)  # E: None
reveal_type(dtype_obj * 1)  # E: numpy.dtype[numpy.str_]
reveal_type(dtype_obj * 2)  # E: numpy.dtype[numpy.void]

reveal_type(0 * dtype_obj)  # E: Any
reveal_type(1 * dtype_obj)  # E: Any
reveal_type(2 * dtype_obj)  # E: Any

reveal_type(void_dtype_obj["f0"])  # E: numpy.dtype[Any]
reveal_type(void_dtype_obj[0])  # E: numpy.dtype[Any]
reveal_type(void_dtype_obj[["f0", "f1"]])  # E: numpy.dtype[numpy.void]
reveal_type(void_dtype_obj[["f0"]])  # E: numpy.dtype[numpy.void]
