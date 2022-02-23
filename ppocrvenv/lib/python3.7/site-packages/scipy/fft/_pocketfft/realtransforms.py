import numpy as np
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
                     _fix_shape, _fix_shape_1d, _normalization, _workers)
import functools


def _r2r(forward, transform, x, type=2, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None):
    """Forward or backward 1-D DCT/DST

    Parameters
    ----------
    forward: bool
        Transform direction (determines type and normalisation)
    transform: {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)

    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2

    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        raise ValueError("invalid number of data points ({0}) specified"
                         .format(tmp.shape[axis]))

    out = (tmp if overwrite_x else None)

    # For complex input, transform real and imaginary components separably
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out
        transform(tmp.real, type, (axis,), norm, out.real, workers)
        transform(tmp.imag, type, (axis,), norm, out.imag, workers)
        return out

    return transform(tmp, type, (axis,), norm, out, workers)


dct = functools.partial(_r2r, True, pfft.dct)
dct.__name__ = 'dct'
idct = functools.partial(_r2r, False, pfft.dct)
idct.__name__ = 'idct'

dst = functools.partial(_r2r, True, pfft.dst)
dst.__name__ = 'dst'
idst = functools.partial(_r2r, False, pfft.dst)
idst.__name__ = 'idst'


def _r2rn(forward, transform, x, type=2, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None):
    """Forward or backward nd DCT/DST

    Parameters
    ----------
    forward: bool
        Transform direction (determines type and normalisation)
    transform: {pypocketfft.dct, pypocketfft.dst}
        The transform to perform
    """
    tmp = _asfarray(x)

    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    overwrite_x = overwrite_x or _datacopied(tmp, x)

    if len(axes) == 0:
        return x

    tmp, copied = _fix_shape(tmp, shape, axes)
    overwrite_x = overwrite_x or copied

    if not forward:
        if type == 2:
            type = 3
        elif type == 3:
            type = 2

    norm = _normalization(norm, forward)
    workers = _workers(workers)
    out = (tmp if overwrite_x else None)

    # For complex input, transform real and imaginary components separably
    if np.iscomplexobj(x):
        out = np.empty_like(tmp) if out is None else out
        transform(tmp.real, type, axes, norm, out.real, workers)
        transform(tmp.imag, type, axes, norm, out.imag, workers)
        return out

    return transform(tmp, type, axes, norm, out, workers)


dctn = functools.partial(_r2rn, True, pfft.dct)
dctn.__name__ = 'dctn'
idctn = functools.partial(_r2rn, False, pfft.dct)
idctn.__name__ = 'idctn'

dstn = functools.partial(_r2rn, True, pfft.dst)
dstn.__name__ = 'dstn'
idstn = functools.partial(_r2rn, False, pfft.dst)
idstn.__name__ = 'idstn'
