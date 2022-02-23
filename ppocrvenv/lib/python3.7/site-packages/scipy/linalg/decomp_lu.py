"""LU decomposition functions."""

from warnings import warn

from numpy import asarray, asarray_chkfinite

# Local imports
from .misc import _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs
from .flinalg import get_flinalg_funcs

__all__ = ['lu', 'lu_solve', 'lu_factor']


def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a matrix.

    The decomposition is::

        A = P L U

    where P is a permutation matrix, L lower triangular with unit
    diagonal elements, and U upper triangular.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    overwrite_a : bool, optional
        Whether to overwrite data in A (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : (N, N) ndarray
        Matrix containing U in its upper triangle, and L in its lower triangle.
        The unit diagonal elements of L are not stored.
    piv : (N,) ndarray
        Pivot indices representing the permutation matrix P:
        row i of matrix was interchanged with row piv[i].

    See also
    --------
    lu_solve : solve an equation system using the LU factorization of a matrix

    Notes
    -----
    This is a wrapper to the ``*GETRF`` routines from LAPACK.

    Examples
    --------
    >>> from scipy.linalg import lu_factor
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> lu, piv = lu_factor(A)
    >>> piv
    array([2, 2, 3, 3], dtype=int32)

    Convert LAPACK's ``piv`` array to NumPy index and test the permutation

    >>> piv_py = [2, 0, 3, 1]
    >>> L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)
    >>> np.allclose(A[piv_py] - L @ U, np.zeros((4, 4)))
    True
    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    getrf, = get_lapack_funcs(('getrf',), (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('illegal value in %dth argument of '
                         'internal getrf (lu_factor)' % -info)
    if info > 0:
        warn("Diagonal number %d is exactly zero. Singular matrix." % info,
             LinAlgWarning, stacklevel=2)
    return lu, piv


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, a x = b, given the LU factorization of a

    Parameters
    ----------
    (lu, piv)
        Factorization of the coefficient matrix a, as given by lu_factor
    b : array
        Right-hand side
    trans : {0, 1, 2}, optional
        Type of system to solve:

        =====  =========
        trans  system
        =====  =========
        0      a x   = b
        1      a^T x = b
        2      a^H x = b
        =====  =========
    overwrite_b : bool, optional
        Whether to overwrite data in b (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        Solution to the system

    See also
    --------
    lu_factor : LU factorize a matrix

    Examples
    --------
    >>> from scipy.linalg import lu_factor, lu_solve
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> b = np.array([1, 1, 1, 1])
    >>> lu, piv = lu_factor(A)
    >>> x = lu_solve((lu, piv), b)
    >>> np.allclose(A @ x - b, np.zeros((4,)))
    True

    """
    (lu, piv) = lu_and_piv
    if check_finite:
        b1 = asarray_chkfinite(b)
    else:
        b1 = asarray(b)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if lu.shape[0] != b1.shape[0]:
        raise ValueError("Shapes of lu {} and b {} are incompatible"
                         .format(lu.shape, b1.shape))

    getrs, = get_lapack_funcs(('getrs',), (lu, b1))
    x, info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
    if info == 0:
        return x
    raise ValueError('illegal value in %dth argument of internal gesv|posv'
                     % -info)


def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a matrix.

    The decomposition is::

        A = P L U

    where P is a permutation matrix, L lower triangular with unit
    diagonal elements, and U upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Array to decompose
    permute_l : bool, optional
        Perform the multiplication P*L (Default: do not permute)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    **(If permute_l == False)**

    p : (M, M) ndarray
        Permutation matrix
    l : (M, K) ndarray
        Lower triangular or trapezoidal matrix with unit diagonal.
        K = min(M, N)
    u : (K, N) ndarray
        Upper triangular or trapezoidal matrix

    **(If permute_l == True)**

    pl : (M, K) ndarray
        Permuted L matrix.
        K = min(M, N)
    u : (K, N) ndarray
        Upper triangular or trapezoidal matrix

    Notes
    -----
    This is a LU factorization routine written for SciPy.

    Examples
    --------
    >>> from scipy.linalg import lu
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> p, l, u = lu(A)
    >>> np.allclose(A - p @ l @ u, np.zeros((4, 4)))
    True

    """
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    flu, = get_flinalg_funcs(('lu',), (a1,))
    p, l, u, info = flu(a1, permute_l=permute_l, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('illegal value in %dth argument of '
                         'internal lu.getrf' % -info)
    if permute_l:
        return l, u
    return p, l, u
