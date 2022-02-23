import numpy as np

a: np.ndarray
generator = (i for i in range(10))

np.require(a, requirements=1)  # E: No overload variant
np.require(a, requirements="TEST")  # E: incompatible type

np.zeros("test")  # E: incompatible type
np.zeros()  # E: Missing positional argument

np.ones("test")  # E: incompatible type
np.ones()  # E: Missing positional argument

np.array(0, float, True)  # E: Too many positional

np.linspace(None, 'bob')  # E: No overload variant
np.linspace(0, 2, num=10.0)  # E: No overload variant
np.linspace(0, 2, endpoint='True')  # E: No overload variant
np.linspace(0, 2, retstep=b'False')  # E: No overload variant
np.linspace(0, 2, dtype=0)  # E: No overload variant
np.linspace(0, 2, axis=None)  # E: No overload variant

np.logspace(None, 'bob')  # E: Argument 1
np.logspace(0, 2, base=None)  # E: Argument "base"

np.geomspace(None, 'bob')  # E: Argument 1

np.stack(generator)  # E: No overload variant
np.hstack({1, 2})  # E: incompatible type
np.vstack(1)  # E: incompatible type
