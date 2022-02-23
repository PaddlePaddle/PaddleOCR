"""
Discrete Fourier Transforms - basic.py
"""
import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization,
                     _workers)

def c2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """ Return discrete Fourier transform of real or complex sequence. """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        raise ValueError("invalid number of data points ({0}) specified"
                         .format(tmp.shape[axis]))

    out = (tmp if overwrite_x and tmp.dtype.kind == 'c' else None)

    return pfft.c2c(tmp, (axis,), forward, norm, out, workers)


fft = functools.partial(c2c, True)
fft.__name__ = 'fft'
ifft = functools.partial(c2c, False)
ifft.__name__ = 'ifft'


def r2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """
    Discrete Fourier transform of a real sequence.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if not np.isrealobj(tmp):
        raise TypeError("x must be a real sequence")

    if n is not None:
        tmp, _ = _fix_shape_1d(tmp, n, axis)
    elif tmp.shape[axis] < 1:
        raise ValueError("invalid number of data points ({0}) specified"
                         .format(tmp.shape[axis]))

    # Note: overwrite_x is not utilised
    return pfft.r2c(tmp, (axis,), forward, norm, None, workers)


rfft = functools.partial(r2c, True)
rfft.__name__ = 'rfft'
ihfft = functools.partial(r2c, False)
ihfft.__name__ = 'ihfft'


def c2r(forward, x, n=None, axis=-1, norm=None, overwrite_x=False,
        workers=None, *, plan=None):
    """
    Return inverse discrete Fourier transform of real sequence x.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    # TODO: Optimize for hermitian and real?
    if np.isrealobj(tmp):
        tmp = tmp + 0.j

    # Last axis utilizes hermitian symmetry
    if n is None:
        n = (tmp.shape[axis] - 1) * 2
        if n < 1:
            raise ValueError("Invalid number of data points ({0}) specified"
                             .format(n))
    else:
        tmp, _ = _fix_shape_1d(tmp, (n//2) + 1, axis)

    # Note: overwrite_x is not utilized
    return pfft.c2r(tmp, (axis,), n, forward, norm, None, workers)


hfft = functools.partial(c2r, True)
hfft.__name__ = 'hfft'
irfft = functools.partial(c2r, False)
irfft.__name__ = 'irfft'


def fft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
         *, plan=None):
    """
    2-D discrete Fourier transform.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return fftn(x, s, axes, norm, overwrite_x, workers)


def ifft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
          *, plan=None):
    """
    2-D discrete inverse Fourier transform of real or complex sequence.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return ifftn(x, s, axes, norm, overwrite_x, workers)


def rfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
          *, plan=None):
    """
    2-D discrete Fourier transform of a real sequence
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return rfftn(x, s, axes, norm, overwrite_x, workers)


def irfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
           *, plan=None):
    """
    2-D discrete inverse Fourier transform of a real sequence
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return irfftn(x, s, axes, norm, overwrite_x, workers)


def hfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
          *, plan=None):
    """
    2-D discrete Fourier transform of a Hermitian sequence
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return hfftn(x, s, axes, norm, overwrite_x, workers)


def ihfft2(x, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None,
           *, plan=None):
    """
    2-D discrete inverse Fourier transform of a Hermitian sequence
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    return ihfftn(x, s, axes, norm, overwrite_x, workers)


def c2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """
    Return multidimensional discrete Fourier transform.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    workers = _workers(workers)

    if len(axes) == 0:
        return x

    tmp, copied = _fix_shape(tmp, shape, axes)
    overwrite_x = overwrite_x or copied

    norm = _normalization(norm, forward)
    out = (tmp if overwrite_x and tmp.dtype.kind == 'c' else None)

    return pfft.c2c(tmp, axes, forward, norm, out, workers)


fftn = functools.partial(c2cn, True)
fftn.__name__ = 'fftn'
ifftn = functools.partial(c2cn, False)
ifftn.__name__ = 'ifftn'

def r2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """Return multidimensional discrete Fourier transform of real input"""
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    if not np.isrealobj(tmp):
        raise TypeError("x must be a real sequence")

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    tmp, _ = _fix_shape(tmp, shape, axes)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if len(axes) == 0:
        raise ValueError("at least 1 axis must be transformed")

    # Note: overwrite_x is not utilized
    return pfft.r2c(tmp, axes, forward, norm, None, workers)


rfftn = functools.partial(r2cn, True)
rfftn.__name__ = 'rfftn'
ihfftn = functools.partial(r2cn, False)
ihfftn.__name__ = 'ihfftn'


def c2rn(forward, x, s=None, axes=None, norm=None, overwrite_x=False,
         workers=None, *, plan=None):
    """Multidimensional inverse discrete fourier transform with real output"""
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet '
                                  'supported by scipy.fft functions')
    tmp = _asfarray(x)

    # TODO: Optimize for hermitian and real?
    if np.isrealobj(tmp):
        tmp = tmp + 0.j

    noshape = s is None
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)

    if len(axes) == 0:
        raise ValueError("at least 1 axis must be transformed")

    if noshape:
        shape[-1] = (x.shape[axes[-1]] - 1) * 2

    norm = _normalization(norm, forward)
    workers = _workers(workers)

    # Last axis utilizes hermitian symmetry
    lastsize = shape[-1]
    shape[-1] = (shape[-1] // 2) + 1

    tmp, _ = _fix_shape(tmp, shape, axes)

    # Note: overwrite_x is not utilized
    return pfft.c2r(tmp, axes, lastsize, forward, norm, None, workers)


hfftn = functools.partial(c2rn, True)
hfftn.__name__ = 'hfftn'
irfftn = functools.partial(c2rn, False)
irfftn.__name__ = 'irfftn'


def r2r_fftpack(forward, x, n=None, axis=-1, norm=None, overwrite_x=False):
    """FFT of a real sequence, returning fftpack half complex format"""
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(None)

    if tmp.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        raise ValueError("invalid number of data points ({0}) specified"
                         .format(tmp.shape[axis]))

    out = (tmp if overwrite_x else None)

    return pfft.r2r_fftpack(tmp, (axis,), forward, forward, norm, out, workers)


rfft_fftpack = functools.partial(r2r_fftpack, True)
rfft_fftpack.__name__ = 'rfft_fftpack'
irfft_fftpack = functools.partial(r2r_fftpack, False)
irfft_fftpack.__name__ = 'irfft_fftpack'
