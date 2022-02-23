from numpy.lib import add_newdoc

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU',
    """
    LU factorization of a sparse matrix.

    Factorization is represented as::

        Pr * A * Pc = L * U

    To construct these `SuperLU` objects, call the `splu` and `spilu`
    functions.

    Attributes
    ----------
    shape
    nnz
    perm_c
    perm_r
    L
    U

    Methods
    -------
    solve

    Notes
    -----

    .. versionadded:: 0.14.0

    Examples
    --------
    The LU decomposition can be used to solve matrix equations. Consider:

    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix, linalg as sla
    >>> A = csc_matrix([[1,2,0,4],[1,0,0,1],[1,0,2,1],[2,2,1,0.]])

    This can be solved for a given right-hand side:

    >>> lu = sla.splu(A)
    >>> b = np.array([1, 2, 3, 4])
    >>> x = lu.solve(b)
    >>> A.dot(x)
    array([ 1.,  2.,  3.,  4.])

    The ``lu`` object also contains an explicit representation of the
    decomposition. The permutations are represented as mappings of
    indices:

    >>> lu.perm_r
    array([0, 2, 1, 3], dtype=int32)
    >>> lu.perm_c
    array([2, 0, 1, 3], dtype=int32)

    The L and U factors are sparse matrices in CSC format:

    >>> lu.L.A
    array([[ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ],
           [ 1. ,  0.5,  0.5,  1. ]])
    >>> lu.U.A
    array([[ 2.,  0.,  1.,  4.],
           [ 0.,  2.,  1.,  1.],
           [ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  0., -5.]])

    The permutation matrices can be constructed:

    >>> Pr = csc_matrix((np.ones(4), (lu.perm_r, np.arange(4))))
    >>> Pc = csc_matrix((np.ones(4), (np.arange(4), lu.perm_c)))

    We can reassemble the original matrix:

    >>> (Pr.T * (lu.L * lu.U) * Pc.T).A
    array([[ 1.,  2.,  0.,  4.],
           [ 1.,  0.,  0.,  1.],
           [ 1.,  0.,  2.,  1.],
           [ 2.,  2.,  1.,  0.]])
    """)

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('solve',
    """
    solve(rhs[, trans])

    Solves linear system of equations with one or several right-hand sides.

    Parameters
    ----------
    rhs : ndarray, shape (n,) or (n, k)
        Right hand side(s) of equation
    trans : {'N', 'T', 'H'}, optional
        Type of system to solve::

            'N':   A   * x == rhs  (default)
            'T':   A^T * x == rhs
            'H':   A^H * x == rhs

        i.e., normal, transposed, and hermitian conjugate.

    Returns
    -------
    x : ndarray, shape ``rhs.shape``
        Solution vector(s)
    """))

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('L',
    """
    Lower triangular factor with unit diagonal as a
    `scipy.sparse.csc_matrix`.

    .. versionadded:: 0.14.0
    """))

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('U',
    """
    Upper triangular factor as a `scipy.sparse.csc_matrix`.

    .. versionadded:: 0.14.0
    """))

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('shape',
    """
    Shape of the original matrix as a tuple of ints.
    """))

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('nnz',
    """
    Number of nonzero elements in the matrix.
    """))

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('perm_c',
    """
    Permutation Pc represented as an array of indices.

    The column permutation matrix can be reconstructed via:

    >>> Pc = np.zeros((n, n))
    >>> Pc[np.arange(n), perm_c] = 1
    """))

add_newdoc('scipy.sparse.linalg.dsolve._superlu', 'SuperLU', ('perm_r',
    """
    Permutation Pr represented as an array of indices.

    The row permutation matrix can be reconstructed via:

    >>> Pr = np.zeros((n, n))
    >>> Pr[perm_r, np.arange(n)] = 1
    """))
