"""Dictionary Of Keys based matrix"""

__docformat__ = "restructuredtext en"

__all__ = ['dok_matrix', 'isspmatrix_dok']

import itertools
import numpy as np

from .base import spmatrix, isspmatrix
from ._index import IndexMixin
from .sputils import (isdense, getdtype, isshape, isintlike, isscalarlike,
                      upcast, upcast_scalar, get_index_dtype, check_shape)

try:
    from operator import isSequenceType as _is_sequence
except ImportError:
    def _is_sequence(x):
        return (hasattr(x, '__len__') or hasattr(x, '__next__')
                or hasattr(x, 'next'))


class dok_matrix(spmatrix, IndexMixin, dict):
    """
    Dictionary Of Keys based sparse matrix.

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        dok_matrix(D)
            with a dense matrix, D

        dok_matrix(S)
            with a sparse matrix, S

        dok_matrix((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Allows for efficient O(1) access of individual elements.
    Duplicates are not allowed.
    Can be efficiently converted to a coo_matrix once constructed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import dok_matrix
    >>> S = dok_matrix((5, 5), dtype=np.float32)
    >>> for i in range(5):
    ...     for j in range(5):
    ...         S[i, j] = i + j    # Update element

    """
    format = 'dok'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        dict.__init__(self)
        spmatrix.__init__(self)

        self.dtype = getdtype(dtype, default=float)
        if isinstance(arg1, tuple) and isshape(arg1):  # (M,N)
            M, N = arg1
            self._shape = check_shape((M, N))
        elif isspmatrix(arg1):  # Sparse ctor
            if isspmatrix_dok(arg1) and copy:
                arg1 = arg1.copy()
            else:
                arg1 = arg1.todok()

            if dtype is not None:
                arg1 = arg1.astype(dtype, copy=False)

            dict.update(self, arg1)
            self._shape = check_shape(arg1.shape)
            self.dtype = arg1.dtype
        else:  # Dense ctor
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise TypeError('Invalid input format.') from e

            if len(arg1.shape) != 2:
                raise TypeError('Expected rank <=2 dense array or matrix.')

            from .coo import coo_matrix
            d = coo_matrix(arg1, dtype=dtype).todok()
            dict.update(self, d)
            self._shape = check_shape(arg1.shape)
            self.dtype = d.dtype

    def update(self, val):
        # Prevent direct usage of update
        raise NotImplementedError("Direct modification to dok_matrix element "
                                  "is not allowed.")

    def _update(self, data):
        """An update method for dict data defined for direct access to
        `dok_matrix` data. Main purpose is to be used for effcient conversion
        from other spmatrix classes. Has no checking if `data` is valid."""
        return dict.update(self, data)

    def set_shape(self, shape):
        new_matrix = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_matrix.__dict__
        dict.clear(self)
        dict.update(self, new_matrix)

    shape = property(fget=spmatrix.get_shape, fset=set_shape)

    def getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("getnnz over an axis is not implemented "
                                      "for DOK format.")
        return dict.__len__(self)

    def count_nonzero(self):
        return sum(x != 0 for x in self.values())

    getnnz.__doc__ = spmatrix.getnnz.__doc__
    count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__

    def __len__(self):
        return dict.__len__(self)

    def get(self, key, default=0.):
        """This overrides the dict.get method, providing type checking
        but otherwise equivalent functionality.
        """
        try:
            i, j = key
            assert isintlike(i) and isintlike(j)
        except (AssertionError, TypeError, ValueError) as e:
            raise IndexError('Index must be a pair of integers.') from e
        if (i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]):
            raise IndexError('Index out of bounds.')
        return dict.get(self, key, default)

    def _get_intXint(self, row, col):
        return dict.get(self, (row, col), self.dtype.type(0))

    def _get_intXslice(self, row, col):
        return self._get_sliceXslice(slice(row, row+1), col)

    def _get_sliceXint(self, row, col):
        return self._get_sliceXslice(row, slice(col, col+1))

    def _get_sliceXslice(self, row, col):
        row_start, row_stop, row_step = row.indices(self.shape[0])
        col_start, col_stop, col_step = col.indices(self.shape[1])
        row_range = range(row_start, row_stop, row_step)
        col_range = range(col_start, col_stop, col_step)
        shape = (len(row_range), len(col_range))
        # Switch paths only when advantageous
        # (count the iterations in the loops, adjust for complexity)
        if len(self) >= 2 * shape[0] * shape[1]:
            # O(nr*nc) path: loop over <row x col>
            return self._get_columnXarray(row_range, col_range)
        # O(nnz) path: loop over entries of self
        newdok = dok_matrix(shape, dtype=self.dtype)
        for key in self.keys():
            i, ri = divmod(int(key[0]) - row_start, row_step)
            if ri != 0 or i < 0 or i >= shape[0]:
                continue
            j, rj = divmod(int(key[1]) - col_start, col_step)
            if rj != 0 or j < 0 or j >= shape[1]:
                continue
            x = dict.__getitem__(self, key)
            dict.__setitem__(newdok, (i, j), x)
        return newdok

    def _get_intXarray(self, row, col):
        return self._get_columnXarray([row], col)

    def _get_arrayXint(self, row, col):
        return self._get_columnXarray(row, [col])

    def _get_sliceXarray(self, row, col):
        row = list(range(*row.indices(self.shape[0])))
        return self._get_columnXarray(row, col)

    def _get_arrayXslice(self, row, col):
        col = list(range(*col.indices(self.shape[1])))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        # outer indexing
        newdok = dok_matrix((len(row), len(col)), dtype=self.dtype)

        for i, r in enumerate(row):
            for j, c in enumerate(col):
                v = dict.get(self, (r, c), 0)
                if v:
                    dict.__setitem__(newdok, (i, j), v)
        return newdok

    def _get_arrayXarray(self, row, col):
        # inner indexing
        i, j = map(np.atleast_2d, np.broadcast_arrays(row, col))
        newdok = dok_matrix(i.shape, dtype=self.dtype)

        for key in itertools.product(range(i.shape[0]), range(i.shape[1])):
            v = dict.get(self, (i[key], j[key]), 0)
            if v:
                dict.__setitem__(newdok, key, v)
        return newdok

    def _set_intXint(self, row, col, x):
        key = (row, col)
        if x:
            dict.__setitem__(self, key, x)
        elif dict.__contains__(self, key):
            del self[key]

    def _set_arrayXarray(self, row, col, x):
        row = list(map(int, row.ravel()))
        col = list(map(int, col.ravel()))
        x = x.ravel()
        dict.update(self, zip(zip(row, col), x))

        for i in np.nonzero(x == 0)[0]:
            key = (row[i], col[i])
            if dict.__getitem__(self, key) == 0:
                # may have been superseded by later update
                del self[key]

    def __add__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = dok_matrix(self.shape, dtype=res_dtype)
            # Add this scalar to every element.
            M, N = self.shape
            for key in itertools.product(range(M), range(N)):
                aij = dict.get(self, (key), 0) + other
                if aij:
                    new[key] = aij
            # new.dtype.char = self.dtype.char
        elif isspmatrix_dok(other):
            if other.shape != self.shape:
                raise ValueError("Matrix dimensions are not equal.")
            # We could alternatively set the dimensions to the largest of
            # the two matrices to be summed.  Would this be a good idea?
            res_dtype = upcast(self.dtype, other.dtype)
            new = dok_matrix(self.shape, dtype=res_dtype)
            dict.update(new, self)
            with np.errstate(over='ignore'):
                dict.update(new,
                           ((k, new[k] + other[k]) for k in other.keys()))
        elif isspmatrix(other):
            csc = self.tocsc()
            new = csc + other
        elif isdense(other):
            new = self.todense() + other
        else:
            return NotImplemented
        return new

    def __radd__(self, other):
        if isscalarlike(other):
            new = dok_matrix(self.shape, dtype=self.dtype)
            M, N = self.shape
            for key in itertools.product(range(M), range(N)):
                aij = dict.get(self, (key), 0) + other
                if aij:
                    new[key] = aij
        elif isspmatrix_dok(other):
            if other.shape != self.shape:
                raise ValueError("Matrix dimensions are not equal.")
            new = dok_matrix(self.shape, dtype=self.dtype)
            dict.update(new, self)
            dict.update(new,
                       ((k, self[k] + other[k]) for k in other.keys()))
        elif isspmatrix(other):
            csc = self.tocsc()
            new = csc + other
        elif isdense(other):
            new = other + self.todense()
        else:
            return NotImplemented
        return new

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('Negating a sparse boolean matrix is not'
                                      ' supported.')
        new = dok_matrix(self.shape, dtype=self.dtype)
        dict.update(new, ((k, -self[k]) for k in self.keys()))
        return new

    def _mul_scalar(self, other):
        res_dtype = upcast_scalar(self.dtype, other)
        # Multiply this scalar by every element.
        new = dok_matrix(self.shape, dtype=res_dtype)
        dict.update(new, ((k, v * other) for k, v in self.items()))
        return new

    def _mul_vector(self, other):
        # matrix * vector
        result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))
        for (i, j), v in self.items():
            result[i] += v * other[j]
        return result

    def _mul_multivector(self, other):
        # matrix * multivector
        result_shape = (self.shape[0], other.shape[1])
        result_dtype = upcast(self.dtype, other.dtype)
        result = np.zeros(result_shape, dtype=result_dtype)
        for (i, j), v in self.items():
            result[i,:] += v * other[j,:]
        return result

    def __imul__(self, other):
        if isscalarlike(other):
            dict.update(self, ((k, v * other) for k, v in self.items()))
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = dok_matrix(self.shape, dtype=res_dtype)
            dict.update(new, ((k, v / other) for k, v in self.items()))
            return new
        return self.tocsr() / other

    def __itruediv__(self, other):
        if isscalarlike(other):
            dict.update(self, ((k, v / other) for k, v in self.items()))
            return self
        return NotImplemented

    def __reduce__(self):
        # this approach is necessary because __setstate__ is called after
        # __setitem__ upon unpickling and since __init__ is not called there
        # is no shape attribute hence it is not possible to unpickle it.
        return dict.__reduce__(self)

    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  For now it returns the number
    # of non-zeros.

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'axes' parameter because swapping "
                             "dimensions is the only logical permutation.")

        M, N = self.shape
        new = dok_matrix((N, M), dtype=self.dtype, copy=copy)
        dict.update(new, (((right, left), val)
                          for (left, right), val in self.items()))
        return new

    transpose.__doc__ = spmatrix.transpose.__doc__

    def conjtransp(self):
        """Return the conjugate transpose."""
        M, N = self.shape
        new = dok_matrix((N, M), dtype=self.dtype)
        dict.update(new, (((right, left), np.conj(val))
                          for (left, right), val in self.items()))
        return new

    def copy(self):
        new = dok_matrix(self.shape, dtype=self.dtype)
        dict.update(new, self)
        return new

    copy.__doc__ = spmatrix.copy.__doc__

    def tocoo(self, copy=False):
        from .coo import coo_matrix
        if self.nnz == 0:
            return coo_matrix(self.shape, dtype=self.dtype)

        idx_dtype = get_index_dtype(maxval=max(self.shape))
        data = np.fromiter(self.values(), dtype=self.dtype, count=self.nnz)
        row = np.fromiter((i for i, _ in self.keys()), dtype=idx_dtype, count=self.nnz)
        col = np.fromiter((j for _, j in self.keys()), dtype=idx_dtype, count=self.nnz)
        A = coo_matrix((data, (row, col)), shape=self.shape, dtype=self.dtype)
        A.has_canonical_format = True
        return A

    tocoo.__doc__ = spmatrix.tocoo.__doc__

    def todok(self, copy=False):
        if copy:
            return self.copy()
        return self

    todok.__doc__ = spmatrix.todok.__doc__

    def tocsc(self, copy=False):
        return self.tocoo(copy=False).tocsc(copy=copy)

    tocsc.__doc__ = spmatrix.tocsc.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        newM, newN = shape
        M, N = self.shape
        if newM < M or newN < N:
            # Remove all elements outside new dimensions
            for (i, j) in list(self.keys()):
                if i >= newM or j >= newN:
                    del self[i, j]
        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__


def isspmatrix_dok(x):
    """Is x of dok_matrix type?

    Parameters
    ----------
    x
        object to check for being a dok matrix

    Returns
    -------
    bool
        True if x is a dok matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dok_matrix, isspmatrix_dok
    >>> isspmatrix_dok(dok_matrix([[5]]))
    True

    >>> from scipy.sparse import dok_matrix, csr_matrix, isspmatrix_dok
    >>> isspmatrix_dok(csr_matrix([[5]]))
    False
    """
    return isinstance(x, dok_matrix)
