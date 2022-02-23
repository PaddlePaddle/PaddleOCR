__all__ = []

import sys
import numpy as np
import cv2 as cv

# NumPy documentation: https://numpy.org/doc/stable/user/basics.subclassing.html

class Mat(np.ndarray):
    '''
    cv.Mat wrapper for numpy array.

    Stores extra metadata information how to interpret and process of numpy array for underlying C++ code.
    '''

    def __new__(cls, arr, **kwargs):
        obj = arr.view(Mat)
        return obj

    def __init__(self, arr, **kwargs):
        self.wrap_channels = kwargs.pop('wrap_channels', getattr(arr, 'wrap_channels', False))
        if len(kwargs) > 0:
            raise TypeError('Unknown parameters: {}'.format(repr(kwargs)))

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.wrap_channels = getattr(obj, 'wrap_channels', None)


Mat.__module__ = cv.__name__
cv.Mat = Mat
cv._registerMatType(Mat)
