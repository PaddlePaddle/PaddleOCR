"""List of Lists sparse matrix class
"""

__docformat__ = "restructuredtext en"

__all__ = ['lil_matrix', 'isspmatrix_lil']

from bisect import bisect_left

import numpy as np

from .base import spmatrix, isspmatrix
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from .sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
                      get_index_dtype, check_shape, check_reshape_kwargs,
                      asmatrix)
from . import _csparsetools


class lil_matrix(spmatrix, IndexMixin):
    """Row-based list of lists sparse matrix

    This is a structure for constructing sparse matrices incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct a matrix efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_matrix(D)
            with a dense matrix or rank-2 ndarray D

        lil_matrix(S)
            with another sparse matrix S (equivalent to S.tolil())

        lil_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of stored values, including explicit zeros
    data
        LIL format data array of the matrix
    rows
        LIL format row index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse matrices
        - once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large matrices

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.


    """
    format = 'lil'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        spmatrix.__init__(self)
        self.dtype = getdtype(dtype, arg1, default=float)

        # First get the shape
        if isspmatrix(arg1):
            if isspmatrix_lil(arg1) and copy:
                A = arg1.copy()
            else:
                A = arg1.tolil()

            if dtype is not None:
                A = A.astype(dtype, copy=False)

            self._shape = check_shape(A.shape)
            self.dtype = A.dtype
            self.rows = A.rows
            self.data = A.data
        elif isinstance(arg1,tuple):
            if isshape(arg1):
                if shape is not None:
                    raise ValueError('invalid use of shape parameter')
                M, N = arg1
                self._shape = check_shape((M, N))
                self.rows = np.empty((M,), dtype=object)
                self.data = np.empty((M,), dtype=object)
                for i in range(M):
                    self.rows[i] = []
                    self.data[i] = []
            else:
                raise TypeError('unrecognized lil_matrix constructor usage')
        else:
            # assume A is dense
            try:
                A = asmatrix(arg1)
            except TypeError as e:
                raise TypeError('unsupported matrix type') from e
            else:
                from .csr import csr_matrix
                A = csr_matrix(A, dtype=dtype).tolil()

                self._shape = check_shape(A.shape)
                self.dtype = A.dtype
                self.rows = A.rows
                self.data = A.data

    def __iadd__(self,other):
        self[:,:] = self + other
        return self

    def __isub__(self,other):
        self[:,:] = self - other
        return self

    def __imul__(self,other):
        if isscalarlike(other):
            self[:,:] = self * other
            return self
        else:
            return NotImplemented

    def __itruediv__(self,other):
        if isscalarlike(other):
            self[:,:] = self / other
            return self
        else:
            return NotImplemented

    # Whenever the dimensions change, empty lists should be created for each
    # row

    def getnnz(self, axis=None):
        if axis is None:
            return sum([len(rowvals) for rowvals in self.data])
        if axis < 0:
            axis += 2
        if axis == 0:
            out = np.zeros(self.shape[1], dtype=np.intp)
            for row in self.rows:
                out[row] += 1
            return out
        elif axis == 1:
            return np.array([len(rowvals) for rowvals in self.data], dtype=np.intp)
        else:
            raise ValueError('axis out of bounds')

    def count_nonzero(self):
        return sum(np.count_nonzero(rowvals) for rowvals in self.data)

    getnnz.__doc__ = spmatrix.getnnz.__doc__
    count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__

    def __str__(self):
        val = ''
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                val += "  %s\t%s\n" % (str((i, j)), str(self.data[i][pos]))
        return val[:-1]

    def getrowview(self, i):
        """Returns a view of the 'i'th row (without copying).
        """
        new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i]
        new.data[0] = self.data[i]
        return new

    def getrow(self, i):
        """Returns a copy of the 'i'th row.
        """
        M, N = self.shape
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('row index out of bounds')
        new = lil_matrix((1, N), dtype=self.dtype)
        new.rows[0] = self.rows[i][:]
        new.data[0] = self.data[i][:]
        return new

    def __getitem__(self, key):
        # Fast path for simple (int, int) indexing.
        if (isinstance(key, tuple) and len(key) == 2 and
                isinstance(key[0], INT_TYPES) and
                isinstance(key[1], INT_TYPES)):
            # lil_get1 handles validation for us.
            return self._get_intXint(*key)
        # Everything else takes the normal path.
        return IndexMixin.__getitem__(self, key)

    def _asindices(self, idx, N):
        # LIL routines handle bounds-checking for us, so don't do it here.
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            raise IndexError('invalid index') from e
        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')
        return x

    def _get_intXint(self, row, col):
        v = _csparsetools.lil_get1(self.shape[0], self.shape[1], self.rows,
                                   self.data, row, col)
        return self.dtype.type(v)

    def _get_sliceXint(self, row, col):
        row = range(*row.indices(self.shape[0]))
        return self._get_row_ranges(row, slice(col, col+1))

    def _get_arrayXint(self, row, col):
        return self._get_row_ranges(row, slice(col, col+1))

    def _get_intXslice(self, row, col):
        return self._get_row_ranges((row,), col)

    def _get_sliceXslice(self, row, col):
        row = range(*row.indices(self.shape[0]))
        return self._get_row_ranges(row, col)

    def _get_arrayXslice(self, row, col):
        return self._get_row_ranges(row, col)

    def _get_intXarray(self, row, col):
        row = np.array(row, dtype=col.dtype, ndmin=1)
        return self._get_columnXarray(row, col)

    def _get_sliceXarray(self, row, col):
        row = np.arange(*row.indices(self.shape[0]))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        # outer indexing
        row, col = _broadcast_arrays(row[:,None], col)
        return self._get_arrayXarray(row, col)

    def _get_arrayXarray(self, row, col):
        # inner indexing
        i, j = map(np.atleast_2d, _prepare_index_for_memoryview(row, col))
        new = lil_matrix(i.shape, dtype=self.dtype)
        _csparsetools.lil_fancy_get(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    new.rows, new.data,
                                    i, j)
        return new

    def _get_row_ranges(self, rows, col_slice):
        """
        Fast path for indexing in the case where column index is slice.

        This gains performance improvement over brute force by more
        efficient skipping of zeros, by accessing the elements
        column-wise in order.

        Parameters
        ----------
        rows : sequence or range
            Rows indexed. If range, must be within valid bounds.
        col_slice : slice
            Columns indexed

        """
        j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
        col_range = range(j_start, j_stop, j_stride)
        nj = len(col_range)
        new = lil_matrix((len(rows), nj), dtype=self.dtype)

        _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1],
                                         self.rows, self.data,
                                         new.rows, new.data,
                                         rows,
                                         j_start, j_stop, j_stride, nj)

        return new

    def _set_intXint(self, row, col, x):
        _csparsetools.lil_insert(self.shape[0], self.shape[1], self.rows,
                                 self.data, row, col, x)

    def _set_arrayXarray(self, row, col, x):
        i, j, x = map(np.atleast_2d, _prepare_index_for_memoryview(row, col, x))
        _csparsetools.lil_fancy_set(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # Special case: full matrix assignment
        if (x.shape == self.shape and
                isinstance(row, slice) and row == slice(None) and
                isinstance(col, slice) and col == slice(None)):
            x = lil_matrix(x, dtype=self.dtype)
            self.rows = x.rows
            self.data = x.data
            return
        # Fall back to densifying x
        x = np.asarray(x.toarray(), dtype=self.dtype)
        x, _ = _broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)

    def __setitem__(self, key, x):
        # Fast path for simple (int, int) indexing.
        if (isinstance(key, tuple) and len(key) == 2 and
                isinstance(key[0], INT_TYPES) and
                isinstance(key[1], INT_TYPES)):
            x = self.dtype.type(x)
            if x.size > 1:
                raise ValueError("Trying to assign a sequence to an item")
            return self._set_intXint(key[0], key[1], x)
        # Everything else takes the normal path.
        IndexMixin.__setitem__(self, key, x)

    def _mul_scalar(self, other):
        if other == 0:
            # Multiply by zero: return the zero matrix
            new = lil_matrix(self.shape, dtype=self.dtype)
        else:
            res_dtype = upcast_scalar(self.dtype, other)

            new = self.copy()
            new = new.astype(res_dtype)
            # Multiply this scalar by every element.
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val*other for val in rowvals]
        return new

    def __truediv__(self, other):           # self / other
        if isscalarlike(other):
            new = self.copy()
            # Divide every element by this scalar
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val/other for val in rowvals]
            return new
        else:
            return self.tocsr() / other

    def copy(self):
        M, N = self.shape
        new = lil_matrix(self.shape, dtype=self.dtype)
        # This is ~14x faster than calling deepcopy() on rows and data.
        _csparsetools.lil_get_row_ranges(M, N, self.rows, self.data,
                                         new.rows, new.data, range(M),
                                         0, N, 1, N)
        return new

    copy.__doc__ = spmatrix.copy.__doc__

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        new = lil_matrix(shape, dtype=self.dtype)

        if order == 'C':
            ncols = self.shape[1]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    new_r, new_c = np.unravel_index(i * ncols + j, shape)
                    new[new_r, new_c] = self[i, j]
        elif order == 'F':
            nrows = self.shape[0]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    new_r, new_c = np.unravel_index(i + j * nrows, shape, order)
                    new[new_r, new_c] = self[i, j]
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        return new

    reshape.__doc__ = spmatrix.reshape.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        if new_M < M:
            self.rows = self.rows[:new_M]
            self.data = self.data[:new_M]
        elif new_M > M:
            self.rows = np.resize(self.rows, new_M)
            self.data = np.resize(self.data, new_M)
            for i in range(M, new_M):
                self.rows[i] = []
                self.data[i] = []

        if new_N < N:
            for row, data in zip(self.rows, self.data):
                trunc = bisect_left(row, new_N)
                del row[trunc:]
                del data[trunc:]

        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

    def toarray(self, order=None, out=None):
        d = self._process_toarray_args(order, out)
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                d[i, j] = self.data[i][pos]
        return d

    toarray.__doc__ = spmatrix.toarray.__doc__

    def transpose(self, axes=None, copy=False):
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False).tolil(copy=False)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def tolil(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tolil.__doc__ = spmatrix.tolil.__doc__

    def tocsr(self, copy=False):
        from .csr import csr_matrix

        M, N = self.shape
        if M == 0 or N == 0:
            return csr_matrix((M, N), dtype=self.dtype)

        # construct indptr array
        if M*N <= np.iinfo(np.int32).max:
            # fast path: it is known that 64-bit indexing will not be needed.
            idx_dtype = np.int32
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indptr[0] = 0
            _csparsetools.lil_get_lengths(self.rows, indptr[1:])
            np.cumsum(indptr, out=indptr)
            nnz = indptr[-1]
        else:
            idx_dtype = get_index_dtype(maxval=N)
            lengths = np.empty(M, dtype=idx_dtype)
            _csparsetools.lil_get_lengths(self.rows, lengths)
            nnz = lengths.sum(dtype=np.int64)
            idx_dtype = get_index_dtype(maxval=max(N, nnz))
            indptr = np.empty(M + 1, dtype=idx_dtype)
            indptr[0] = 0
            np.cumsum(lengths, dtype=idx_dtype, out=indptr[1:])

        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=self.dtype)
        _csparsetools.lil_flatten_to_array(self.rows, indices)
        _csparsetools.lil_flatten_to_array(self.data, data)

        # init csr matrix
        return csr_matrix((data, indices, indptr), shape=self.shape)

    tocsr.__doc__ = spmatrix.tocsr.__doc__


def _prepare_index_for_memoryview(i, j, x=None):
    """
    Convert index and data arrays to form suitable for passing to the
    Cython fancy getset routines.

    The conversions are necessary since to (i) ensure the integer
    index arrays are in one of the accepted types, and (ii) to ensure
    the arrays are writable so that Cython memoryview support doesn't
    choke on them.

    Parameters
    ----------
    i, j
        Index arrays
    x : optional
        Data arrays

    Returns
    -------
    i, j, x
        Re-formatted arrays (x is omitted, if input was None)

    """
    if i.dtype > j.dtype:
        j = j.astype(i.dtype)
    elif i.dtype < j.dtype:
        i = i.astype(j.dtype)

    if not i.flags.writeable or i.dtype not in (np.int32, np.int64):
        i = i.astype(np.intp)
    if not j.flags.writeable or j.dtype not in (np.int32, np.int64):
        j = j.astype(np.intp)

    if x is not None:
        if not x.flags.writeable:
            x = x.copy()
        return i, j, x
    else:
        return i, j


def isspmatrix_lil(x):
    """Is x of lil_matrix type?

    Parameters
    ----------
    x
        object to check for being a lil matrix

    Returns
    -------
    bool
        True if x is a lil matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import lil_matrix, isspmatrix_lil
    >>> isspmatrix_lil(lil_matrix([[5]]))
    True

    >>> from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_lil
    >>> isspmatrix_lil(csr_matrix([[5]]))
    False
    """
    return isinstance(x, lil_matrix)
