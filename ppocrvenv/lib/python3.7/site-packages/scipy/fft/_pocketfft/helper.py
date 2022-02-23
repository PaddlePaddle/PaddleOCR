from numbers import Number
import operator
import os
import threading
import contextlib

import numpy as np
# good_size is exposed (and used) from this import
from .pypocketfft import good_size

_config = threading.local()
_cpu_count = os.cpu_count()


def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int

    Parameters
    ----------
    x : value, or sequence of values, convertible to int
    name : str, optional
        Name of the argument being converted, only used in the error message

    Returns
    -------
    y : ``List[int]``
    """
    if isinstance(x, Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or "value"
        raise ValueError("{} must be a scalar or iterable of integers"
                         .format(name)) from e

    return x


def _init_nd_shape_and_axes(x, shape, axes):
    """Handles shape and axes arguments for nd transforms"""
    noshape = shape is None
    noaxes = axes is None

    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]

        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")

    if not noshape:
        shape = _iterable_of_int(shape, 'shape')

        if axes and len(axes) != len(shape):
            raise ValueError("when given, axes and shape arguments"
                             " have to be of the same length")
        if noaxes:
            if len(shape) > x.ndim:
                raise ValueError("shape requires more axes than are present")
            axes = range(x.ndim - len(shape), x.ndim)

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]

    if any(s < 1 for s in shape):
        raise ValueError(
            "invalid number of data points ({0}) specified".format(shape))

    return shape, axes


def _asfarray(x):
    """
    Convert to array with floating or complex dtype.

    float16 values are also promoted to float32.
    """
    if not hasattr(x, "dtype"):
        x = np.asarray(x)

    if x.dtype == np.float16:
        return np.asarray(x, np.float32)
    elif x.dtype.kind not in 'fc':
        return np.asarray(x, np.float64)

    # Require native byte order
    dtype = x.dtype.newbyteorder('=')
    # Always align input
    copy = not x.flags['ALIGNED']
    return np.array(x, dtype=dtype, copy=copy)

def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    if arr is original:
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None


def _fix_shape(x, shape, axes):
    """Internal auxiliary function for _raw_fft, _raw_fftnd."""
    must_copy = False

    # Build an nd slice with the dimensions to be read from x
    index = [slice(None)]*x.ndim
    for n, ax in zip(shape, axes):
        if x.shape[ax] >= n:
            index[ax] = slice(0, n)
        else:
            index[ax] = slice(0, x.shape[ax])
            must_copy = True

    index = tuple(index)

    if not must_copy:
        return x[index], False

    s = list(x.shape)
    for n, axis in zip(shape, axes):
        s[axis] = n

    z = np.zeros(s, x.dtype)
    z[index] = x[index]
    return z, True


def _fix_shape_1d(x, n, axis):
    if n < 1:
        raise ValueError(
            "invalid number of data points ({0}) specified".format(n))

    return _fix_shape(x, (n,), (axis,))


_NORM_MAP = {None: 0, 'backward': 0, 'ortho': 1, 'forward': 2}


def _normalization(norm, forward):
    """Returns the pypocketfft normalization mode from the norm argument"""
    try:
        inorm = _NORM_MAP[norm]
        return inorm if forward else (2 - inorm)
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm!r}, should '
            'be "backward", "ortho" or "forward"') from None


def _workers(workers):
    if workers is None:
        return getattr(_config, 'default_workers', 1)

    if workers < 0:
        if workers >= -_cpu_count:
            workers += 1 + _cpu_count
        else:
            raise ValueError("workers value out of range; got {}, must not be"
                             " less than {}".format(workers, -_cpu_count))
    elif workers == 0:
        raise ValueError("workers must not be zero")

    return workers


@contextlib.contextmanager
def set_workers(workers):
    """Context manager for the default number of workers used in `scipy.fft`

    Parameters
    ----------
    workers : int
        The default number of workers to use

    Examples
    --------
    >>> from scipy import fft, signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal((128, 64))
    >>> with fft.set_workers(4):
    ...     y = signal.fftconvolve(x, x)

    """
    old_workers = get_workers()
    _config.default_workers = _workers(operator.index(workers))
    try:
        yield
    finally:
        _config.default_workers = old_workers


def get_workers():
    """Returns the default number of workers within the current context

    Examples
    --------
    >>> from scipy import fft
    >>> fft.get_workers()
    1
    >>> with fft.set_workers(4):
    ...     fft.get_workers()
    4
    """
    return getattr(_config, 'default_workers', 1)
