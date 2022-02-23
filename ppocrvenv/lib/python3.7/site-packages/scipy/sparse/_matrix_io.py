import numpy as np
import scipy.sparse

__all__ = ['save_npz', 'load_npz']


# Make loading safe vs. malicious input
PICKLE_KWARGS = dict(allow_pickle=False)


def save_npz(file, matrix, compressed=True):
    """ Save a sparse matrix to a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already
        there.
    matrix: spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)
        The sparse matrix to save.
    compressed : bool, optional
        Allow compressing the file. Default: True

    See Also
    --------
    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
    numpy.savez: Save several arrays into a ``.npz`` archive.
    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import scipy.sparse
    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)

    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)
    """
    arrays_dict = {}
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        raise NotImplementedError('Save is not implemented for sparse matrix of format {}.'.format(matrix.format))
    arrays_dict.update(
        format=matrix.format.encode('ascii'),
        shape=matrix.shape,
        data=matrix.data
    )
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_npz(file):
    """ Load a sparse matrix from a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be loaded.

    Returns
    -------
    result : csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix
        A sparse matrix containing the loaded data.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.

    See Also
    --------
    scipy.sparse.save_npz: Save a sparse matrix to a file using ``.npz`` format.
    numpy.load: Load several arrays from a ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import scipy.sparse
    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)

    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
        with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.todense()
    matrix([[0, 0, 3],
            [4, 0, 0]], dtype=int64)
    """

    with np.load(file, **PICKLE_KWARGS) as loaded:
        try:
            matrix_format = loaded['format']
        except KeyError as e:
            raise ValueError('The file {} does not contain a sparse matrix.'.format(file)) from e

        matrix_format = matrix_format.item()

        if not isinstance(matrix_format, str):
            # Play safe with Python 2 vs 3 backward compatibility;
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            matrix_format = matrix_format.decode('ascii')

        try:
            cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
        except AttributeError as e:
            raise ValueError('Unknown matrix format "{}"'.format(matrix_format)) from e

        if matrix_format in ('csc', 'csr', 'bsr'):
            return cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        elif matrix_format == 'dia':
            return cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
        elif matrix_format == 'coo':
            return cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
        else:
            raise NotImplementedError('Load is not implemented for '
                                      'sparse matrix of format {}.'.format(matrix_format))
