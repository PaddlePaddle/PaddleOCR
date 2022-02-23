import numpy as np

class _MockFunction:
    def __init__(self, return_value = None):
        self.number_calls = 0
        self.return_value = return_value
        self.last_args = ([], {})

    def __call__(self, *args, **kwargs):
        self.number_calls += 1
        self.last_args = (args, kwargs)
        return self.return_value


fft = _MockFunction(np.random.random(10))
fft2 = _MockFunction(np.random.random(10))
fftn = _MockFunction(np.random.random(10))

ifft = _MockFunction(np.random.random(10))
ifft2 = _MockFunction(np.random.random(10))
ifftn = _MockFunction(np.random.random(10))

rfft = _MockFunction(np.random.random(10))
rfft2 = _MockFunction(np.random.random(10))
rfftn = _MockFunction(np.random.random(10))

irfft = _MockFunction(np.random.random(10))
irfft2 = _MockFunction(np.random.random(10))
irfftn = _MockFunction(np.random.random(10))

hfft = _MockFunction(np.random.random(10))
hfft2 = _MockFunction(np.random.random(10))
hfftn = _MockFunction(np.random.random(10))

ihfft = _MockFunction(np.random.random(10))
ihfft2 = _MockFunction(np.random.random(10))
ihfftn = _MockFunction(np.random.random(10))

dct = _MockFunction(np.random.random(10))
idct = _MockFunction(np.random.random(10))
dctn = _MockFunction(np.random.random(10))
idctn = _MockFunction(np.random.random(10))

dst = _MockFunction(np.random.random(10))
idst = _MockFunction(np.random.random(10))
dstn = _MockFunction(np.random.random(10))
idstn = _MockFunction(np.random.random(10))


__ua_domain__ = "numpy.scipy.fft"


def __ua_function__(method, args, kwargs):
    fn = globals().get(method.__name__)
    return (fn(*args, **kwargs) if fn is not None
            else NotImplemented)
