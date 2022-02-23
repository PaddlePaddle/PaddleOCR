import numpy as np

np.AxisError(1.0)  # E: Argument 1 to "AxisError" has incompatible type
np.AxisError(1, ndim=2.0)  # E: Argument "ndim" to "AxisError" has incompatible type
np.AxisError(
    2, msg_prefix=404  # E: Argument "msg_prefix" to "AxisError" has incompatible type
)
