import numpy as np

nd = np.array([[1, 2], [3, 4]])

# reshape
reveal_type(nd.reshape())  # E: numpy.ndarray
reveal_type(nd.reshape(4))  # E: numpy.ndarray
reveal_type(nd.reshape(2, 2))  # E: numpy.ndarray
reveal_type(nd.reshape((2, 2)))  # E: numpy.ndarray

reveal_type(nd.reshape((2, 2), order="C"))  # E: numpy.ndarray
reveal_type(nd.reshape(4, order="C"))  # E: numpy.ndarray

# resize does not return a value

# transpose
reveal_type(nd.transpose())  # E: numpy.ndarray
reveal_type(nd.transpose(1, 0))  # E: numpy.ndarray
reveal_type(nd.transpose((1, 0)))  # E: numpy.ndarray

# swapaxes
reveal_type(nd.swapaxes(0, 1))  # E: numpy.ndarray

# flatten
reveal_type(nd.flatten())  # E: numpy.ndarray
reveal_type(nd.flatten("C"))  # E: numpy.ndarray

# ravel
reveal_type(nd.ravel())  # E: numpy.ndarray
reveal_type(nd.ravel("C"))  # E: numpy.ndarray

# squeeze
reveal_type(nd.squeeze())  # E: numpy.ndarray
reveal_type(nd.squeeze(0))  # E: numpy.ndarray
reveal_type(nd.squeeze((0, 2)))  # E: numpy.ndarray
