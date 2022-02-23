"""Functions to construct sparse matrices
"""

__docformat__ = "restructuredtext en"

__all__ = ['spdiags', 'eye', 'identity', 'kron', 'kronsum',
           'hstack', 'vstack', 'bmat', 'rand', 'random', 'diags', 'block_diag']

import numbers
from functools import partial
import numpy as np

from scipy._lib._util import check_random_state, rng_integers
from .sputils import upcast, get_index_dtype, isscalarlike

from .csr import csr_matrix
from .csc import csc_matrix
from .bsr import bsr_matrix
from .coo import coo_matrix
from .dia import dia_matrix

from .base import issparse


def spdiags(data, diags, m, n, format=None):
    """
    Return a sparse matrix from diagonals.

    Parameters
    ----------
    data : array_like
        matrix diagonals stored row-wise
    diags : diagonals to set
        - k = 0  the main diagonal
        - k > 0  the k-th upper diagonal
        - k < 0  the k-th lower diagonal
    m, n : int
        shape of the result
    format : str, optional
        Format of the result. By default (format=None) an appropriate sparse
        matrix format is returned. This choice is subject to change.

    See Also
    --------
    diags : more convenient form of this function
    dia_matrix : the sparse DIAgonal format.

    Examples
    --------
    >>> from scipy.sparse import spdiags
    >>> data = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    >>> diags = np.array([0, -1, 2])
    >>> spdiags(data, diags, 4, 4).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    """
    return dia_matrix((data, diags), shape=(m,n)).asformat(format)


def diags(diagonals, offsets=0, shape=None, format=None, dtype=None):
    """
    Construct a sparse matrix from diagonals.

    Parameters
    ----------
    diagonals : sequence of array_like
        Sequence of arrays containing the matrix diagonals,
        corresponding to `offsets`.
    offsets : sequence of int or an int, optional
        Diagonals to set:
          - k = 0  the main diagonal (default)
          - k > 0  the kth upper diagonal
          - k < 0  the kth lower diagonal
    shape : tuple of int, optional
        Shape of the result. If omitted, a square matrix large enough
        to contain the diagonals is returned.
    format : {"dia", "csr", "csc", "lil", ...}, optional
        Matrix format of the result. By default (format=None) an
        appropriate sparse matrix format is returned. This choice is
        subject to change.
    dtype : dtype, optional
        Data type of the matrix.

    See Also
    --------
    spdiags : construct matrix from diagonals

    Notes
    -----
    This function differs from `spdiags` in the way it handles
    off-diagonals.

    The result from `diags` is the sparse equivalent of::

        np.diag(diagonals[0], offsets[0])
        + ...
        + np.diag(diagonals[k], offsets[k])

    Repeated diagonal offsets are disallowed.

    .. versionadded:: 0.11

    Examples
    --------
    >>> from scipy.sparse import diags
    >>> diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
    >>> diags(diagonals, [0, -1, 2]).toarray()
    array([[1, 0, 1, 0],
           [1, 2, 0, 2],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    Broadcasting of scalars is supported (but shape needs to be
    specified):

    >>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).toarray()
    array([[-2.,  1.,  0.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  0.,  1., -2.]])


    If only one diagonal is wanted (as in `numpy.diag`), the following
    works as well:

    >>> diags([1, 2, 3], 1).toarray()
    array([[ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  2.,  0.],
           [ 0.,  0.,  0.,  3.],
           [ 0.,  0.,  0.,  0.]])
    """
    # if offsets is not a sequence, assume that there's only one diagonal
    if isscalarlike(offsets):
        # now check that there's actually only one diagonal
        if len(diagonals) == 0 or isscalarlike(diagonals[0]):
            diagonals = [np.atleast_1d(diagonals)]
        else:
            raise ValueError("Different number of diagonals and offsets.")
    else:
        diagonals = list(map(np.atleast_1d, diagonals))

    offsets = np.atleast_1d(offsets)

    # Basic check
    if len(diagonals) != len(offsets):
        raise ValueError("Different number of diagonals and offsets.")

    # Determine shape, if omitted
    if shape is None:
        m = len(diagonals[0]) + abs(int(offsets[0]))
        shape = (m, m)

    # Determine data type, if omitted
    if dtype is None:
        dtype = np.common_type(*diagonals)

    # Construct data array
    m, n = shape

    M = max([min(m + offset, n - offset) + max(0, offset)
             for offset in offsets])
    M = max(0, M)
    data_arr = np.zeros((len(offsets), M), dtype=dtype)

    K = min(m, n)

    for j, diagonal in enumerate(diagonals):
        offset = offsets[j]
        k = max(0, offset)
        length = min(m + offset, n - offset, K)
        if length < 0:
            raise ValueError("Offset %d (index %d) out of bounds" % (offset, j))
        try:
            data_arr[j, k:k+length] = diagonal[...,:length]
        except ValueError as e:
            if len(diagonal) != length and len(diagonal) != 1:
                raise ValueError(
                    "Diagonal length (index %d: %d at offset %d) does not "
                    "agree with matrix size (%d, %d)." % (
                    j, len(diagonal), offset, m, n)) from e
            raise

    return dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)


def identity(n, dtype='d', format=None):
    """Identity matrix in sparse format

    Returns an identity matrix with shape (n,n) using a given
    sparse format and dtype.

    Parameters
    ----------
    n : int
        Shape of the identity matrix.
    dtype : dtype, optional
        Data type of the matrix
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> from scipy.sparse import identity
    >>> identity(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> identity(3, dtype='int8', format='dia')
    <3x3 sparse matrix of type '<class 'numpy.int8'>'
            with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    return eye(n, n, dtype=dtype, format=format)


def eye(m, n=None, k=0, dtype=float, format=None):
    """Sparse matrix with ones on diagonal

    Returns a sparse (m x n) matrix where the kth diagonal
    is all ones and everything else is zeros.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.
    n : int, optional
        Number of columns. Default: `m`.
    k : int, optional
        Diagonal to place ones on. Default: 0 (main diagonal).
    dtype : dtype, optional
        Data type of the matrix.
    format : str, optional
        Sparse format of the result, e.g., format="csr", etc.

    Examples
    --------
    >>> from scipy import sparse
    >>> sparse.eye(3).toarray()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> sparse.eye(3, dtype=np.int8)
    <3x3 sparse matrix of type '<class 'numpy.int8'>'
        with 3 stored elements (1 diagonals) in DIAgonal format>

    """
    if n is None:
        n = m
    m,n = int(m),int(n)

    if m == n and k == 0:
        # fast branch for special formats
        if format in ['csr', 'csc']:
            idx_dtype = get_index_dtype(maxval=n)
            indptr = np.arange(n+1, dtype=idx_dtype)
            indices = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            cls = {'csr': csr_matrix, 'csc': csc_matrix}[format]
            return cls((data,indices,indptr),(n,n))
        elif format == 'coo':
            idx_dtype = get_index_dtype(maxval=n)
            row = np.arange(n, dtype=idx_dtype)
            col = np.arange(n, dtype=idx_dtype)
            data = np.ones(n, dtype=dtype)
            return coo_matrix((data,(row,col)),(n,n))

    diags = np.ones((1, max(0, min(m + k, n))), dtype=dtype)
    return spdiags(diags, k, m, n).asformat(format)


def kron(A, B, format=None):
    """kronecker product of sparse matrices A and B

    Parameters
    ----------
    A : sparse or dense matrix
        first matrix of the product
    B : sparse or dense matrix
        second matrix of the product
    format : str, optional
        format of the result (e.g. "csr")

    Returns
    -------
    kronecker product in a sparse matrix format


    Examples
    --------
    >>> from scipy import sparse
    >>> A = sparse.csr_matrix(np.array([[0, 2], [5, 0]]))
    >>> B = sparse.csr_matrix(np.array([[1, 2], [3, 4]]))
    >>> sparse.kron(A, B).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    >>> sparse.kron(A, [[1, 2], [3, 4]]).toarray()
    array([[ 0,  0,  2,  4],
           [ 0,  0,  6,  8],
           [ 5, 10,  0,  0],
           [15, 20,  0,  0]])

    """
    B = coo_matrix(B)

    if (format is None or format == "bsr") and 2*B.nnz >= B.shape[0] * B.shape[1]:
        # B is fairly dense, use BSR
        A = csr_matrix(A,copy=True)

        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_matrix(output_shape).asformat(format)

        B = B.toarray()
        data = A.data.repeat(B.size).reshape(-1,B.shape[0],B.shape[1])
        data = data * B

        return bsr_matrix((data,A.indices,A.indptr), shape=output_shape)
    else:
        # use COO
        A = coo_matrix(A)
        output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return coo_matrix(output_shape).asformat(format)

        # expand entries of a into blocks
        row = A.row.repeat(B.nnz)
        col = A.col.repeat(B.nnz)
        data = A.data.repeat(B.nnz)

        if max(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]) > np.iinfo('int32').max:
            row = row.astype(np.int64)
            col = col.astype(np.int64)

        row *= B.shape[0]
        col *= B.shape[1]

        # increment block indices
        row,col = row.reshape(-1,B.nnz),col.reshape(-1,B.nnz)
        row += B.row
        col += B.col
        row,col = row.reshape(-1),col.reshape(-1)

        # compute block entries
        data = data.reshape(-1,B.nnz) * B.data
        data = data.reshape(-1)

        return coo_matrix((data,(row,col)), shape=output_shape).asformat(format)


def kronsum(A, B, format=None):
    """kronecker sum of sparse matrices A and B

    Kronecker sum of two sparse matrices is a sum of two Kronecker
    products kron(I_n,A) + kron(B,I_m) where A has shape (m,m)
    and B has shape (n,n) and I_m and I_n are identity matrices
    of shape (m,m) and (n,n), respectively.

    Parameters
    ----------
    A
        square matrix
    B
        square matrix
    format : str
        format of the result (e.g. "csr")

    Returns
    -------
    kronecker sum in a sparse matrix format

    Examples
    --------


    """
    A = coo_matrix(A)
    B = coo_matrix(B)

    if A.shape[0] != A.shape[1]:
        raise ValueError('A is not square')

    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square')

    dtype = upcast(A.dtype, B.dtype)

    L = kron(eye(B.shape[0],dtype=dtype), A, format=format)
    R = kron(B, eye(A.shape[0],dtype=dtype), format=format)

    return (L+R).asformat(format)  # since L + R is not always same format


def _compressed_sparse_stack(blocks, axis):
    """
    Stacking fast path for CSR/CSC matrices
    (i) vstack for CSR, (ii) hstack for CSC.
    """
    other_axis = 1 if axis == 0 else 0
    data = np.concatenate([b.data for b in blocks])
    constant_dim = blocks[0].shape[other_axis]
    idx_dtype = get_index_dtype(arrays=[b.indptr for b in blocks],
                                maxval=max(data.size, constant_dim))
    indices = np.empty(data.size, dtype=idx_dtype)
    indptr = np.empty(sum(b.shape[axis] for b in blocks) + 1, dtype=idx_dtype)
    last_indptr = idx_dtype(0)
    sum_dim = 0
    sum_indices = 0
    for b in blocks:
        if b.shape[other_axis] != constant_dim:
            raise ValueError('incompatible dimensions for axis %d' % other_axis)
        indices[sum_indices:sum_indices+b.indices.size] = b.indices
        sum_indices += b.indices.size
        idxs = slice(sum_dim, sum_dim + b.shape[axis])
        indptr[idxs] = b.indptr[:-1]
        indptr[idxs] += last_indptr
        sum_dim += b.shape[axis]
        last_indptr += b.indptr[-1]
    indptr[-1] = last_indptr
    if axis == 0:
        return csr_matrix((data, indices, indptr),
                          shape=(sum_dim, constant_dim))
    else:
        return csc_matrix((data, indices, indptr),
                          shape=(constant_dim, sum_dim))


def hstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices horizontally (column wise)

    Parameters
    ----------
    blocks
        sequence of sparse matrices with compatible shapes
    format : str
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    See Also
    --------
    vstack : stack sparse matrices vertically (row wise)

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, hstack
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> hstack([A,B]).toarray()
    array([[1, 2, 5],
           [3, 4, 6]])

    """
    return bmat([blocks], format=format, dtype=dtype)


def vstack(blocks, format=None, dtype=None):
    """
    Stack sparse matrices vertically (row wise)

    Parameters
    ----------
    blocks
        sequence of sparse matrices with compatible shapes
    format : str, optional
        sparse format of the result (e.g., "csr")
        by default an appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    See Also
    --------
    hstack : stack sparse matrices horizontally (column wise)

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, vstack
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5, 6]])
    >>> vstack([A, B]).toarray()
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    return bmat([[b] for b in blocks], format=format, dtype=dtype)


def bmat(blocks, format=None, dtype=None):
    """
    Build a sparse matrix from sparse sub-blocks

    Parameters
    ----------
    blocks : array_like
        Grid of sparse matrices with compatible shapes.
        An entry of None implies an all-zero matrix.
    format : {'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, optional
        The sparse format of the result (e.g. "csr"). By default an
        appropriate sparse matrix format is returned.
        This choice is subject to change.
    dtype : dtype, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    bmat : sparse matrix

    See Also
    --------
    block_diag, diags

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, bmat
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> C = coo_matrix([[7]])
    >>> bmat([[A, B], [None, C]]).toarray()
    array([[1, 2, 5],
           [3, 4, 6],
           [0, 0, 7]])

    >>> bmat([[A, None], [None, C]]).toarray()
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 7]])

    """

    blocks = np.asarray(blocks, dtype='object')

    if blocks.ndim != 2:
        raise ValueError('blocks must be 2-D')

    M,N = blocks.shape

    # check for fast path cases
    if (N == 1 and format in (None, 'csr') and all(isinstance(b, csr_matrix)
                                                   for b in blocks.flat)):
        A = _compressed_sparse_stack(blocks[:,0], 0)
        if dtype is not None:
            A = A.astype(dtype)
        return A
    elif (M == 1 and format in (None, 'csc')
          and all(isinstance(b, csc_matrix) for b in blocks.flat)):
        A = _compressed_sparse_stack(blocks[0,:], 1)
        if dtype is not None:
            A = A.astype(dtype)
        return A

    block_mask = np.zeros(blocks.shape, dtype=bool)
    brow_lengths = np.zeros(M, dtype=np.int64)
    bcol_lengths = np.zeros(N, dtype=np.int64)

    # convert everything to COO format
    for i in range(M):
        for j in range(N):
            if blocks[i,j] is not None:
                A = coo_matrix(blocks[i,j])
                blocks[i,j] = A
                block_mask[i,j] = True

                if brow_lengths[i] == 0:
                    brow_lengths[i] = A.shape[0]
                elif brow_lengths[i] != A.shape[0]:
                    msg = ('blocks[{i},:] has incompatible row dimensions. '
                           'Got blocks[{i},{j}].shape[0] == {got}, '
                           'expected {exp}.'.format(i=i, j=j,
                                                    exp=brow_lengths[i],
                                                    got=A.shape[0]))
                    raise ValueError(msg)

                if bcol_lengths[j] == 0:
                    bcol_lengths[j] = A.shape[1]
                elif bcol_lengths[j] != A.shape[1]:
                    msg = ('blocks[:,{j}] has incompatible row dimensions. '
                           'Got blocks[{i},{j}].shape[1] == {got}, '
                           'expected {exp}.'.format(i=i, j=j,
                                                    exp=bcol_lengths[j],
                                                    got=A.shape[1]))
                    raise ValueError(msg)

    nnz = sum(block.nnz for block in blocks[block_mask])
    if dtype is None:
        all_dtypes = [blk.dtype for blk in blocks[block_mask]]
        dtype = upcast(*all_dtypes) if all_dtypes else None

    row_offsets = np.append(0, np.cumsum(brow_lengths))
    col_offsets = np.append(0, np.cumsum(bcol_lengths))

    shape = (row_offsets[-1], col_offsets[-1])

    data = np.empty(nnz, dtype=dtype)
    idx_dtype = get_index_dtype(maxval=max(shape))
    row = np.empty(nnz, dtype=idx_dtype)
    col = np.empty(nnz, dtype=idx_dtype)

    nnz = 0
    ii, jj = np.nonzero(block_mask)
    for i, j in zip(ii, jj):
        B = blocks[i, j]
        idx = slice(nnz, nnz + B.nnz)
        data[idx] = B.data
        row[idx] = B.row + row_offsets[i]
        col[idx] = B.col + col_offsets[j]
        nnz += B.nnz

    return coo_matrix((data, (row, col)), shape=shape).asformat(format)


def block_diag(mats, format=None, dtype=None):
    """
    Build a block diagonal sparse matrix from provided matrices.

    Parameters
    ----------
    mats : sequence of matrices
        Input matrices.
    format : str, optional
        The sparse format of the result (e.g., "csr"). If not given, the matrix
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.

    Returns
    -------
    res : sparse matrix

    Notes
    -----

    .. versionadded:: 0.11.0

    See Also
    --------
    bmat, diags

    Examples
    --------
    >>> from scipy.sparse import coo_matrix, block_diag
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> C = coo_matrix([[7]])
    >>> block_diag((A, B, C)).toarray()
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 7]])

    """
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for a in mats:
        if isinstance(a, (list, numbers.Number)):
            a = coo_matrix(a)
        nrows, ncols = a.shape
        if issparse(a):
            a = a.tocoo()
            row.append(a.row + r_idx)
            col.append(a.col + c_idx)
            data.append(a.data)
        else:
            a_row, a_col = np.divmod(np.arange(nrows*ncols), ncols)
            row.append(a_row + r_idx)
            col.append(a_col + c_idx)
            data.append(a.ravel())
        r_idx += nrows
        c_idx += ncols
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)
    return coo_matrix((data, (row, col)),
                      shape=(r_idx, c_idx),
                      dtype=dtype).asformat(format)


def random(m, n, density=0.01, format='coo', dtype=None,
           random_state=None, data_rvs=None):
    """Generate a sparse matrix of the given shape and density with randomly
    distributed values.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : callable, optional
        Samples a requested number of random values.
        This function should take a single argument specifying the length
        of the ndarray that it will return. The structurally nonzero entries
        of the sparse random matrix will be taken from the array sampled
        by this function. By default, uniform [0, 1) random values will be
        sampled using the same random state as is used for sampling
        the sparsity structure.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Only float types are supported for now.

    Examples
    --------
    >>> from scipy.sparse import random
    >>> from scipy import stats
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> rvs = stats.poisson(25, loc=10).rvs
    >>> S = random(3, 4, density=0.25, random_state=rng, data_rvs=rvs)
    >>> S.A
    array([[ 36.,   0.,  33.,   0.],   # random
           [  0.,   0.,   0.,   0.],
           [  0.,   0.,  36.,   0.]])

    >>> from scipy.sparse import random
    >>> from scipy.stats import rv_continuous
    >>> class CustomDistribution(rv_continuous):
    ...     def _rvs(self,  size=None, random_state=None):
    ...         return random_state.standard_normal(size)
    >>> X = CustomDistribution(seed=rng)
    >>> Y = X()  # get a frozen version of the distribution
    >>> S = random(3, 4, density=0.25, random_state=rng, data_rvs=Y.rvs)
    >>> S.A
    array([[ 0.        ,  0.        ,  0.        ,  0.        ],   # random
           [ 0.13569738,  1.9467163 , -0.81205367,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ]])

    """
    if density < 0 or density > 1:
        raise ValueError("density expected to be 0 <= density <= 1")
    dtype = np.dtype(dtype)

    mn = m * n

    tp = np.intc
    if mn > np.iinfo(tp).max:
        tp = np.int64

    if mn > np.iinfo(tp).max:
        msg = """\
Trying to generate a random sparse matrix such as the product of dimensions is
greater than %d - this is not supported on this machine
"""
        raise ValueError(msg % np.iinfo(tp).max)

    # Number of non zero values
    k = int(round(density * m * n))

    random_state = check_random_state(random_state)

    if data_rvs is None:
        if np.issubdtype(dtype, np.integer):
            def data_rvs(n):
                return rng_integers(random_state,
                                    np.iinfo(dtype).min,
                                    np.iinfo(dtype).max,
                                    n,
                                    dtype=dtype)
        elif np.issubdtype(dtype, np.complexfloating):
            def data_rvs(n):
                return (random_state.uniform(size=n) +
                        random_state.uniform(size=n) * 1j)
        else:
            data_rvs = partial(random_state.uniform, 0., 1.)

    ind = random_state.choice(mn, size=k, replace=False)

    j = np.floor(ind * 1. / m).astype(tp, copy=False)
    i = (ind - j * m).astype(tp, copy=False)
    vals = data_rvs(k).astype(dtype, copy=False)
    return coo_matrix((vals, (i, j)), shape=(m, n)).asformat(format,
                                                             copy=False)


def rand(m, n, density=0.01, format="coo", dtype=None, random_state=None):
    """Generate a sparse matrix of the given shape and density with uniformly
    distributed values.

    Parameters
    ----------
    m, n : int
        shape of the matrix
    density : real, optional
        density of the generated matrix: density equal to one means a full
        matrix, density of 0 means a matrix with no non-zero items.
    format : str, optional
        sparse matrix format.
    dtype : dtype, optional
        type of the returned matrix values.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    res : sparse matrix

    Notes
    -----
    Only float types are supported for now.

    See Also
    --------
    scipy.sparse.random : Similar function that allows a user-specified random
        data source.

    Examples
    --------
    >>> from scipy.sparse import rand
    >>> matrix = rand(3, 4, density=0.25, format="csr", random_state=42)
    >>> matrix
    <3x4 sparse matrix of type '<class 'numpy.float64'>'
       with 3 stored elements in Compressed Sparse Row format>
    >>> matrix.todense()
    matrix([[0.05641158, 0.        , 0.        , 0.65088847],
            [0.        , 0.        , 0.        , 0.14286682],
            [0.        , 0.        , 0.        , 0.        ]])

    """
    return random(m, n, density, format, dtype, random_state)
