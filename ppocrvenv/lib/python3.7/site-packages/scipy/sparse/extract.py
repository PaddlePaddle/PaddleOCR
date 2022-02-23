"""Functions to extract parts of sparse matrices
"""

__docformat__ = "restructuredtext en"

__all__ = ['find', 'tril', 'triu']


from .coo import coo_matrix


def find(A):
    """Return the indices and values of the nonzero elements of a matrix

    Parameters
    ----------
    A : dense or sparse matrix
        Matrix whose nonzero elements are desired.

    Returns
    -------
    (I,J,V) : tuple of arrays
        I,J, and V contain the row indices, column indices, and values
        of the nonzero matrix entries.


    Examples
    --------
    >>> from scipy.sparse import csr_matrix, find
    >>> A = csr_matrix([[7.0, 8.0, 0],[0, 0, 9.0]])
    >>> find(A)
    (array([0, 0, 1], dtype=int32), array([0, 1, 2], dtype=int32), array([ 7.,  8.,  9.]))

    """

    A = coo_matrix(A, copy=True)
    A.sum_duplicates()
    # remove explicit zeros
    nz_mask = A.data != 0
    return A.row[nz_mask], A.col[nz_mask], A.data[nz_mask]


def tril(A, k=0, format=None):
    """Return the lower triangular portion of a matrix in sparse format

    Returns the elements on or below the k-th diagonal of the matrix A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse matrix
        Matrix whose lower trianglar portion is desired.
    k : integer : optional
        The top-most diagonal of the lower triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse matrix
        Lower triangular portion of A in sparse format.

    See Also
    --------
    triu : upper triangle in sparse format

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, tril
    >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...                dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> tril(A).toarray()
    array([[1, 0, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 0, 0]])
    >>> tril(A).nnz
    4
    >>> tril(A, k=1).toarray()
    array([[1, 2, 0, 0, 0],
           [4, 5, 0, 0, 0],
           [0, 0, 8, 9, 0]])
    >>> tril(A, k=-1).toarray()
    array([[0, 0, 0, 0, 0],
           [4, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> tril(A, format='csc')
    <3x5 sparse matrix of type '<class 'numpy.int32'>'
            with 4 stored elements in Compressed Sparse Column format>

    """

    # convert to COOrdinate format where things are easy
    A = coo_matrix(A, copy=False)
    mask = A.row + k >= A.col
    return _masked_coo(A, mask).asformat(format)


def triu(A, k=0, format=None):
    """Return the upper triangular portion of a matrix in sparse format

    Returns the elements on or above the k-th diagonal of the matrix A.
        - k = 0 corresponds to the main diagonal
        - k > 0 is above the main diagonal
        - k < 0 is below the main diagonal

    Parameters
    ----------
    A : dense or sparse matrix
        Matrix whose upper trianglar portion is desired.
    k : integer : optional
        The bottom-most diagonal of the upper triangle.
    format : string
        Sparse format of the result, e.g. format="csr", etc.

    Returns
    -------
    L : sparse matrix
        Upper triangular portion of A in sparse format.

    See Also
    --------
    tril : lower triangle in sparse format

    Examples
    --------
    >>> from scipy.sparse import csr_matrix, triu
    >>> A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]],
    ...                dtype='int32')
    >>> A.toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A).toarray()
    array([[1, 2, 0, 0, 3],
           [0, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A).nnz
    8
    >>> triu(A, k=1).toarray()
    array([[0, 2, 0, 0, 3],
           [0, 0, 0, 6, 7],
           [0, 0, 0, 9, 0]])
    >>> triu(A, k=-1).toarray()
    array([[1, 2, 0, 0, 3],
           [4, 5, 0, 6, 7],
           [0, 0, 8, 9, 0]])
    >>> triu(A, format='csc')
    <3x5 sparse matrix of type '<class 'numpy.int32'>'
            with 8 stored elements in Compressed Sparse Column format>

    """

    # convert to COOrdinate format where things are easy
    A = coo_matrix(A, copy=False)
    mask = A.row + k <= A.col
    return _masked_coo(A, mask).asformat(format)


def _masked_coo(A, mask):
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    return coo_matrix((data, (row, col)), shape=A.shape, dtype=A.dtype)
