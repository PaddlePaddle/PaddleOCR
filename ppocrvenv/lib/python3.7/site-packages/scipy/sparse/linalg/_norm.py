"""Sparse matrix norms.

"""
import numpy as np
from scipy.sparse import issparse

from numpy import Inf, sqrt, abs

__all__ = ['norm']


def _sparse_frobenius_norm(x):
    if np.issubdtype(x.dtype, np.complexfloating):
        sqnorm = abs(x).power(2).sum()
    else:
        sqnorm = x.power(2).sum()
    return sqrt(sqnorm)


def norm(x, ord=None, axis=None):
    """
    Norm of a sparse matrix

    This function is able to return one of seven different matrix norms,
    depending on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : a sparse matrix
        Input sparse matrix.
    ord : {non-zero int, inf, -inf, 'fro'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.

    Returns
    -------
    n : float or ndarray

    Notes
    -----
    Some of the ord are not implemented because some associated functions like,
    _multi_svd_norm, are not yet available for sparse matrix.

    This docstring is modified based on numpy.linalg.norm.
    https://github.com/numpy/numpy/blob/master/numpy/linalg/linalg.py

    The following norms can be calculated:

    =====  ============================
    ord    norm for sparse matrices
    =====  ============================
    None   Frobenius norm
    'fro'  Frobenius norm
    inf    max(sum(abs(x), axis=1))
    -inf   min(sum(abs(x), axis=1))
    0      abs(x).sum(axis=axis)
    1      max(sum(abs(x), axis=0))
    -1     min(sum(abs(x), axis=0))
    2      Not implemented
    -2     Not implemented
    other  Not implemented
    =====  ============================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from scipy.sparse import *
    >>> import numpy as np
    >>> from scipy.sparse.linalg import norm
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1, 0, 1],
           [ 2, 3, 4]])

    >>> b = csr_matrix(b)
    >>> norm(b)
    7.745966692414834
    >>> norm(b, 'fro')
    7.745966692414834
    >>> norm(b, np.inf)
    9
    >>> norm(b, -np.inf)
    2
    >>> norm(b, 1)
    7
    >>> norm(b, -1)
    6

    """
    if not issparse(x):
        raise TypeError("input is not sparse. use numpy.linalg.norm")

    # Check the default case first and handle it immediately.
    if axis is None and ord in (None, 'fro', 'f'):
        return _sparse_frobenius_norm(x)

    # Some norms require functions that are not implemented for all types.
    x = x.tocsr()

    if axis is None:
        axis = (0, 1)
    elif not isinstance(axis, tuple):
        msg = "'axis' must be None, an integer or a tuple of integers"
        try:
            int_axis = int(axis)
        except TypeError as e:
            raise TypeError(msg) from e
        if axis != int_axis:
            raise TypeError(msg)
        axis = (int_axis,)

    nd = 2
    if len(axis) == 2:
        row_axis, col_axis = axis
        if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if row_axis % nd == col_axis % nd:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            raise NotImplementedError
            #return _multi_svd_norm(x, row_axis, col_axis, amax)
        elif ord == -2:
            raise NotImplementedError
            #return _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            return abs(x).sum(axis=row_axis).max(axis=col_axis)[0,0]
        elif ord == Inf:
            return abs(x).sum(axis=col_axis).max(axis=row_axis)[0,0]
        elif ord == -1:
            return abs(x).sum(axis=row_axis).min(axis=col_axis)[0,0]
        elif ord == -Inf:
            return abs(x).sum(axis=col_axis).min(axis=row_axis)[0,0]
        elif ord in (None, 'f', 'fro'):
            # The axis order does not matter for this norm.
            return _sparse_frobenius_norm(x)
        else:
            raise ValueError("Invalid norm order for matrices.")
    elif len(axis) == 1:
        a, = axis
        if not (-nd <= a < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if ord == Inf:
            M = abs(x).max(axis=a)
        elif ord == -Inf:
            M = abs(x).min(axis=a)
        elif ord == 0:
            # Zero norm
            M = (x != 0).sum(axis=a)
        elif ord == 1:
            # special case for speedup
            M = abs(x).sum(axis=a)
        elif ord in (2, None):
            M = sqrt(abs(x).power(2).sum(axis=a))
        else:
            try:
                ord + 1
            except TypeError as e:
                raise ValueError('Invalid norm order for vectors.') from e
            M = np.power(abs(x).power(ord).sum(axis=a), 1 / ord)
        return M.A.ravel()
    else:
        raise ValueError("Improper number of dimensions to norm.")
