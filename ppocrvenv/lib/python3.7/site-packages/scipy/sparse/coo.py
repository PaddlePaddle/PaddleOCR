""" A sparse matrix in COOrdinate or 'triplet' format"""

__docformat__ = "restructuredtext en"

__all__ = ['coo_matrix', 'isspmatrix_coo']

from warnings import warn

import numpy as np


from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from .base import isspmatrix, SparseEfficiencyWarning, spmatrix
from .data import _data_matrix, _minmax_mixin
from .sputils import (upcast, upcast_char, to_native, isshape, getdtype,
                      getdata, get_index_dtype, downcast_intp_index,
                      check_shape, check_reshape_kwargs, matrix)

import operator


class coo_matrix(_data_matrix, _minmax_mixin):
    """
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            with a dense matrix D

        coo_matrix(S)
            with another sparse matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

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
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate indices
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate indices are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """
    format = 'coo'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        if isinstance(arg1, tuple):
            if isshape(arg1):
                M, N = arg1
                self._shape = check_shape((M, N))
                idx_dtype = get_index_dtype(maxval=max(M, N))
                data_dtype = getdtype(dtype, default=float)
                self.row = np.array([], dtype=idx_dtype)
                self.col = np.array([], dtype=idx_dtype)
                self.data = np.array([], dtype=data_dtype)
                self.has_canonical_format = True
            else:
                try:
                    obj, (row, col) = arg1
                except (TypeError, ValueError) as e:
                    raise TypeError('invalid input format') from e

                if shape is None:
                    if len(row) == 0 or len(col) == 0:
                        raise ValueError('cannot infer dimensions from zero '
                                         'sized index arrays')
                    M = operator.index(np.max(row)) + 1
                    N = operator.index(np.max(col)) + 1
                    self._shape = check_shape((M, N))
                else:
                    # Use 2 steps to ensure shape has length 2.
                    M, N = shape
                    self._shape = check_shape((M, N))

                idx_dtype = get_index_dtype(maxval=max(self.shape))
                self.row = np.array(row, copy=copy, dtype=idx_dtype)
                self.col = np.array(col, copy=copy, dtype=idx_dtype)
                self.data = getdata(obj, copy=copy, dtype=dtype)
                self.has_canonical_format = False
        else:
            if isspmatrix(arg1):
                if isspmatrix_coo(arg1) and copy:
                    self.row = arg1.row.copy()
                    self.col = arg1.col.copy()
                    self.data = arg1.data.copy()
                    self._shape = check_shape(arg1.shape)
                else:
                    coo = arg1.tocoo()
                    self.row = coo.row
                    self.col = coo.col
                    self.data = coo.data
                    self._shape = check_shape(coo.shape)
                self.has_canonical_format = False
            else:
                #dense argument
                M = np.atleast_2d(np.asarray(arg1))

                if M.ndim != 2:
                    raise TypeError('expected dimension <= 2 array or matrix')

                self._shape = check_shape(M.shape)
                if shape is not None:
                    if check_shape(shape) != self._shape:
                        raise ValueError('inconsistent shapes: %s != %s' %
                                         (shape, self._shape))

                self.row, self.col = M.nonzero()
                self.data = M[self.row, self.col]
                self.has_canonical_format = True

        if dtype is not None:
            self.data = self.data.astype(dtype, copy=False)

        self._check()

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        nrows, ncols = self.shape

        if order == 'C':
            # Upcast to avoid overflows: the coo_matrix constructor
            # below will downcast the results to a smaller dtype, if
            # possible.
            dtype = get_index_dtype(maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))

            flat_indices = np.multiply(ncols, self.row, dtype=dtype) + self.col
            new_row, new_col = divmod(flat_indices, shape[1])
        elif order == 'F':
            dtype = get_index_dtype(maxval=(nrows * max(0, ncols - 1) + max(0, nrows - 1)))

            flat_indices = np.multiply(nrows, self.col, dtype=dtype) + self.row
            new_col, new_row = divmod(flat_indices, shape[0])
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        # Handle copy here rather than passing on to the constructor so that no
        # copy will be made of new_row and new_col regardless
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        return coo_matrix((new_data, (new_row, new_col)),
                          shape=shape, copy=False)

    reshape.__doc__ = spmatrix.reshape.__doc__

    def getnnz(self, axis=None):
        if axis is None:
            nnz = len(self.data)
            if nnz != len(self.row) or nnz != len(self.col):
                raise ValueError('row, column, and data array must all be the '
                                 'same length')

            if self.data.ndim != 1 or self.row.ndim != 1 or \
                    self.col.ndim != 1:
                raise ValueError('row, column, and data arrays must be 1-D')

            return int(nnz)

        if axis < 0:
            axis += 2
        if axis == 0:
            return np.bincount(downcast_intp_index(self.col),
                               minlength=self.shape[1])
        elif axis == 1:
            return np.bincount(downcast_intp_index(self.row),
                               minlength=self.shape[0])
        else:
            raise ValueError('axis out of bounds')

    getnnz.__doc__ = spmatrix.getnnz.__doc__

    def _check(self):
        """ Checks data structure for consistency """

        # index arrays should have integer data types
        if self.row.dtype.kind != 'i':
            warn("row index array has non-integer dtype (%s)  "
                    % self.row.dtype.name)
        if self.col.dtype.kind != 'i':
            warn("col index array has non-integer dtype (%s) "
                    % self.col.dtype.name)

        idx_dtype = get_index_dtype(maxval=max(self.shape))
        self.row = np.asarray(self.row, dtype=idx_dtype)
        self.col = np.asarray(self.col, dtype=idx_dtype)
        self.data = to_native(self.data)

        if self.nnz > 0:
            if self.row.max() >= self.shape[0]:
                raise ValueError('row index exceeds matrix dimensions')
            if self.col.max() >= self.shape[1]:
                raise ValueError('column index exceeds matrix dimensions')
            if self.row.min() < 0:
                raise ValueError('negative row index found')
            if self.col.min() < 0:
                raise ValueError('negative column index found')

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape
        return coo_matrix((self.data, (self.col, self.row)),
                          shape=(N, M), copy=copy)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def resize(self, *shape):
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        if new_M < M or new_N < N:
            mask = np.logical_and(self.row < new_M, self.col < new_N)
            if not mask.all():
                self.row = self.row[mask]
                self.col = self.col[mask]
                self.data = self.data[mask]

        self._shape = shape

    resize.__doc__ = spmatrix.resize.__doc__

    def toarray(self, order=None, out=None):
        """See the docstring for `spmatrix.toarray`."""
        B = self._process_toarray_args(order, out)
        fortran = int(B.flags.f_contiguous)
        if not fortran and not B.flags.c_contiguous:
            raise ValueError("Output array must be C or F contiguous")
        M,N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    B.ravel('A'), fortran)
        return B

    def tocsc(self, copy=False):
        """Convert this matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        from .csc import csc_matrix
        if self.nnz == 0:
            return csc_matrix(self.shape, dtype=self.dtype)
        else:
            M,N = self.shape
            idx_dtype = get_index_dtype((self.col, self.row),
                                        maxval=max(self.nnz, M))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(N + 1, dtype=idx_dtype)
            indices = np.empty_like(row, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(N, M, self.nnz, col, row, self.data,
                      indptr, indices, data)

            x = csc_matrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """Convert this matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_matrix
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        from .csr import csr_matrix
        if self.nnz == 0:
            return csr_matrix(self.shape, dtype=self.dtype)
        else:
            M,N = self.shape
            idx_dtype = get_index_dtype((self.row, self.col),
                                        maxval=max(self.nnz, N))
            row = self.row.astype(idx_dtype, copy=False)
            col = self.col.astype(idx_dtype, copy=False)

            indptr = np.empty(M + 1, dtype=idx_dtype)
            indices = np.empty_like(col, dtype=idx_dtype)
            data = np.empty_like(self.data, dtype=upcast(self.dtype))

            coo_tocsr(M, N, self.nnz, row, col, self.data,
                      indptr, indices, data)

            x = csr_matrix((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocoo(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocoo.__doc__ = spmatrix.tocoo.__doc__

    def todia(self, copy=False):
        from .dia import dia_matrix

        self.sum_duplicates()
        ks = self.col - self.row  # the diagonal for each nonzero
        diags, diag_idx = np.unique(ks, return_inverse=True)

        if len(diags) > 100:
            # probably undesired, should todia() have a maxdiags parameter?
            warn("Constructing a DIA matrix with %d diagonals "
                 "is inefficient" % len(diags), SparseEfficiencyWarning)

        #initialize and fill in data array
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data

        return dia_matrix((data,diags), shape=self.shape)

    todia.__doc__ = spmatrix.todia.__doc__

    def todok(self, copy=False):
        from .dok import dok_matrix

        self.sum_duplicates()
        dok = dok_matrix((self.shape), dtype=self.dtype)
        dok._update(zip(zip(self.row,self.col),self.data))

        return dok

    todok.__doc__ = spmatrix.todok.__doc__

    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                        dtype=self.dtype)
        diag_mask = (self.row + k) == self.col

        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            row, _, data = self._sum_duplicates(self.row[diag_mask],
                                                self.col[diag_mask],
                                                self.data[diag_mask])
        diag[row + min(k, 0)] = data

        return diag

    diagonal.__doc__ = _data_matrix.diagonal.__doc__

    def _setdiag(self, values, k):
        M, N = self.shape
        if values.ndim and not len(values):
            return
        idx_dtype = self.row.dtype

        # Determine which triples to keep and where to put the new ones.
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M+k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N-k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)

        # Define the array of data consisting of the entries to be added.
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values

        # Update the internal structure.
        self.row = np.concatenate((self.row[keep], new_row))
        self.col = np.concatenate((self.col[keep], new_col))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    # needed by _data_matrix
    def _with_data(self,data,copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the index arrays
        (i.e. .row and .col) are copied.
        """
        if copy:
            return coo_matrix((data, (self.row.copy(), self.col.copy())),
                                   shape=self.shape, dtype=data.dtype)
        else:
            return coo_matrix((data, (self.row, self.col)),
                                   shape=self.shape, dtype=data.dtype)

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together

        This is an *in place* operation
        """
        if self.has_canonical_format:
            return
        summed = self._sum_duplicates(self.row, self.col, self.data)
        self.row, self.col, self.data = summed
        self.has_canonical_format = True

    def _sum_duplicates(self, row, col, data):
        # Assumes (data, row, col) not in canonical format.
        if len(data) == 0:
            return row, col, data
        order = np.lexsort((row, col))
        row = row[order]
        col = col[order]
        data = data[order]
        unique_mask = ((row[1:] != row[:-1]) |
                       (col[1:] != col[:-1]))
        unique_mask = np.append(True, unique_mask)
        row = row[unique_mask]
        col = col[unique_mask]
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        return row, col, data

    def eliminate_zeros(self):
        """Remove zero entries from the matrix

        This is an *in place* operation
        """
        mask = self.data != 0
        self.data = self.data[mask]
        self.row = self.row[mask]
        self.col = self.col[mask]

    #######################
    # Arithmetic handlers #
    #######################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError('Incompatible shapes ({} and {})'
                             .format(self.shape, other.shape))
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        M, N = self.shape
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    result.ravel('A'), fortran)
        return matrix(result, copy=False)

    def _mul_vector(self, other):
        #output array
        result = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
                                                            other.dtype.char))
        coo_matvec(self.nnz, self.row, self.col, self.data, other, result)
        return result

    def _mul_multivector(self, other):
        result = np.zeros((other.shape[1], self.shape[0]),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))
        for i, col in enumerate(other.T):
            coo_matvec(self.nnz, self.row, self.col, self.data, col, result[i])
        return result.T.view(type=type(other))


def isspmatrix_coo(x):
    """Is x of coo_matrix type?

    Parameters
    ----------
    x
        object to check for being a coo matrix

    Returns
    -------
    bool
        True if x is a coo matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, isspmatrix_coo
    >>> isspmatrix_coo(coo_matrix([[5]]))
    True

    >>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo
    >>> isspmatrix_coo(csr_matrix([[5]]))
    False
    """
    return isinstance(x, coo_matrix)
