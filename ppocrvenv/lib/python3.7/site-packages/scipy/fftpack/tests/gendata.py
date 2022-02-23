import numpy as np
from scipy.io import loadmat

m = loadmat('test.mat', squeeze_me=True, struct_as_record=True,
        mat_dtype=True)
np.savez('test.npz', **m)
