"""Compressed Sparse Row matrix format"""

__docformat__ = "restructuredtext en"

__all__ = ['csr_matrix', 'isspmatrix_csr']

import numpy as np

from .base import spmatrix
from ._sparsetools import (csr_tocsc, csr_tobsr, csr_count_blocks,
                           get_csr_submatrix)
from .sputils import upcast, get_index_dtype

from .compressed import _cs_matrix


class csr_matrix(_cs_matrix):
    """
    Compressed Sparse Row matrix

    This can be instantiated in several ways:
        csr_matrix(D)
            with a dense matrix or rank-2 ndarray D

        csr_matrix(S)
            with another sparse matrix S (equivalent to S.tocsr())

        csr_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the matrix dimensions
            are inferred from the index arrays.

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
        CSR format data array of the matrix
    indices
        CSR format index array of the matrix
    indptr
        CSR format index pointer array of the matrix
    has_sorted_indices
        Whether indices are sorted

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> csr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    Duplicate entries are summed together:

    >>> row = np.array([0, 1, 2, 0])
    >>> col = np.array([0, 1, 1, 0])
    >>> data = np.array([1, 2, 4, 8])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[9, 0, 0],
           [0, 2, 0],
           [0, 4, 0]])

    As an example of how to construct a CSR matrix incrementally,
    the following snippet builds a term-document matrix from texts:

    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    >>> indptr = [0]
    >>> indices = []
    >>> data = []
    >>> vocabulary = {}
    >>> for d in docs:
    ...     for term in d:
    ...         index = vocabulary.setdefault(term, len(vocabulary))
    ...         indices.append(index)
    ...         data.append(1)
    ...     indptr.append(len(indices))
    ...
    >>> csr_matrix((data, indices, indptr), dtype=int).toarray()
    array([[2, 1, 0, 0],
           [0, 1, 1, 1]])

    """
    format = 'csr'

    def transpose(self, axes=None, copy=False):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))

        M, N = self.shape

        from .csc import csc_matrix
        return csc_matrix((self.data, self.indices,
                           self.indptr), shape=(N, M), copy=copy)

    transpose.__doc__ = spmatrix.transpose.__doc__

    def tolil(self, copy=False):
        from .lil import lil_matrix
        lil = lil_matrix(self.shape,dtype=self.dtype)

        self.sum_duplicates()
        ptr,ind,dat = self.indptr,self.indices,self.data
        rows, data = lil.rows, lil.data

        for n in range(self.shape[0]):
            start = ptr[n]
            end = ptr[n+1]
            rows[n] = ind[start:end].tolist()
            data[n] = dat[start:end].tolist()

        return lil

    tolil.__doc__ = spmatrix.tolil.__doc__

    def tocsr(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocsr.__doc__ = spmatrix.tocsr.__doc__

    def tocsc(self, copy=False):
        idx_dtype = get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, self.shape[0]))
        indptr = np.empty(self.shape[1] + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csr_tocsc(self.shape[0], self.shape[1],
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        from .csc import csc_matrix
        A = csc_matrix((data, indices, indptr), shape=self.shape)
        A.has_sorted_indices = True
        return A

    tocsc.__doc__ = spmatrix.tocsc.__doc__

    def tobsr(self, blocksize=None, copy=True):
        from .bsr import bsr_matrix

        if blocksize is None:
            from .spfuncs import estimate_blocksize
            return self.tobsr(blocksize=estimate_blocksize(self))

        elif blocksize == (1,1):
            arg1 = (self.data.reshape(-1,1,1),self.indices,self.indptr)
            return bsr_matrix(arg1, shape=self.shape, copy=copy)

        else:
            R,C = blocksize
            M,N = self.shape

            if R < 1 or C < 1 or M % R != 0 or N % C != 0:
                raise ValueError('invalid blocksize %s' % blocksize)

            blks = csr_count_blocks(M,N,R,C,self.indptr,self.indices)

            idx_dtype = get_index_dtype((self.indptr, self.indices),
                                        maxval=max(N//C, blks))
            indptr = np.empty(M//R+1, dtype=idx_dtype)
            indices = np.empty(blks, dtype=idx_dtype)
            data = np.zeros((blks,R,C), dtype=self.dtype)

            csr_tobsr(M, N, R, C,
                      self.indptr.astype(idx_dtype),
                      self.indices.astype(idx_dtype),
                      self.data,
                      indptr, indices, data.ravel())

            return bsr_matrix((data,indices,indptr), shape=self.shape)

    tobsr.__doc__ = spmatrix.tobsr.__doc__

    # these functions are used by the parent class (_cs_matrix)
    # to remove redundancy between csc_matrix and csr_matrix
    def _swap(self, x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x

    def __iter__(self):
        indptr = np.zeros(2, dtype=self.indptr.dtype)
        shape = (1, self.shape[1])
        i0 = 0
        for i1 in self.indptr[1:]:
            indptr[1] = i1 - i0
            indices = self.indices[i0:i1]
            data = self.data[i0:i1]
            yield csr_matrix((data, indices, indptr), shape=shape, copy=True)
            i0 = i1

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i, i + 1, 0, N)
        return csr_matrix((data, indices, indptr), shape=(1, N),
                          dtype=self.dtype, copy=False)

    def getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, 0, M, i, i + 1)
        return csr_matrix((data, indices, indptr), shape=(M, 1),
                          dtype=self.dtype, copy=False)

    def _get_intXarray(self, row, col):
        return self.getrow(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        # TODO: uncomment this once it's faster:
        # return self.getrow(row)._minor_slice(col)

        M, N = self.shape
        start, stop, stride = col.indices(N)

        ii, jj = self.indptr[row:row+2]
        row_indices = self.indices[ii:jj]
        row_data = self.data[ii:jj]

        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)

        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0

        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = np.array([0, len(row_indices)])

        if stride < 0:
            row_data = row_data[::-1]
            row_indices = abs(row_indices[::-1])

        shape = (1, max(0, int(np.ceil(float(stop - start) / stride))))
        return csr_matrix((row_data, row_indices, row_indptr), shape=shape,
                          dtype=self.dtype, copy=False)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        return self._major_slice(row)._get_submatrix(minor=col)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            col = np.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        return self._major_index_fancy(row)._get_submatrix(minor=col)


def isspmatrix_csr(x):
    """Is x of csr_matrix type?

    Parameters
    ----------
    x
        object to check for being a csr matrix

    Returns
    -------
    bool
        True if x is a csr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, isspmatrix_csr
    >>> isspmatrix_csr(csr_matrix([[5]]))
    True

    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
    >>> isspmatrix_csr(csc_matrix([[5]]))
    False
    """
    return isinstance(x, csr_matrix)
