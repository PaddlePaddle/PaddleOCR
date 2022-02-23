import numpy as np

# Techincally this works, but probably shouldn't. See
#
# https://github.com/numpy/numpy/issues/16366
#
np.maximum_sctype(1)  # E: incompatible type "int"

np.issubsctype(1, np.int64)  # E: incompatible type "int"

np.issubdtype(1, np.int64)  # E: incompatible type "int"

np.find_common_type(np.int64, np.int64)  # E: incompatible type "Type[signedinteger[Any]]"
