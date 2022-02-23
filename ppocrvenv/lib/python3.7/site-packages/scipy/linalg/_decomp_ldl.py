from warnings import warn

import numpy as np
from numpy import (atleast_2d, ComplexWarning, arange, zeros_like, imag, diag,
                   iscomplexobj, tril, triu, argsort, empty_like)
from .decomp import _asarray_validated
from .lapack import get_lapack_funcs, _compute_lwork

__all__ = ['ldl']


def ldl(A, lower=True, hermitian=True, overwrite_a=False, check_finite=True):
    """ Computes the LDLt or Bunch-Kaufman factorization of a symmetric/
    hermitian matrix.

    This function returns a block diagonal matrix D consisting blocks of size
    at most 2x2 and also a possibly permuted unit lower triangular matrix
    ``L`` such that the factorization ``A = L D L^H`` or ``A = L D L^T``
    holds. If `lower` is False then (again possibly permuted) upper
    triangular matrices are returned as outer factors.

    The permutation array can be used to triangularize the outer factors
    simply by a row shuffle, i.e., ``lu[perm, :]`` is an upper/lower
    triangular matrix. This is also equivalent to multiplication with a
    permutation matrix ``P.dot(lu)``, where ``P`` is a column-permuted
    identity matrix ``I[:, perm]``.

    Depending on the value of the boolean `lower`, only upper or lower
    triangular part of the input array is referenced. Hence, a triangular
    matrix on entry would give the same result as if the full matrix is
    supplied.

    Parameters
    ----------
    A : array_like
        Square input array
    lower : bool, optional
        This switches between the lower and upper triangular outer factors of
        the factorization. Lower triangular (``lower=True``) is the default.
    hermitian : bool, optional
        For complex-valued arrays, this defines whether ``A = A.conj().T`` or
        ``A = A.T`` is assumed. For real-valued arrays, this switch has no
        effect.
    overwrite_a : bool, optional
        Allow overwriting data in `A` (may enhance performance). The default
        is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : ndarray
        The (possibly) permuted upper/lower triangular outer factor of the
        factorization.
    d : ndarray
        The block diagonal multiplier of the factorization.
    perm : ndarray
        The row-permutation index array that brings lu into triangular form.

    Raises
    ------
    ValueError
        If input array is not square.
    ComplexWarning
        If a complex-valued array with nonzero imaginary parts on the
        diagonal is given and hermitian is set to True.

    Examples
    --------
    Given an upper triangular array ``a`` that represents the full symmetric
    array with its entries, obtain ``l``, 'd' and the permutation vector `perm`:

    >>> import numpy as np
    >>> from scipy.linalg import ldl
    >>> a = np.array([[2, -1, 3], [0, 2, 0], [0, 0, 1]])
    >>> lu, d, perm = ldl(a, lower=0) # Use the upper part
    >>> lu
    array([[ 0. ,  0. ,  1. ],
           [ 0. ,  1. , -0.5],
           [ 1. ,  1. ,  1.5]])
    >>> d
    array([[-5. ,  0. ,  0. ],
           [ 0. ,  1.5,  0. ],
           [ 0. ,  0. ,  2. ]])
    >>> perm
    array([2, 1, 0])
    >>> lu[perm, :]
    array([[ 1. ,  1. ,  1.5],
           [ 0. ,  1. , -0.5],
           [ 0. ,  0. ,  1. ]])
    >>> lu.dot(d).dot(lu.T)
    array([[ 2., -1.,  3.],
           [-1.,  2.,  0.],
           [ 3.,  0.,  1.]])

    Notes
    -----
    This function uses ``?SYTRF`` routines for symmetric matrices and
    ``?HETRF`` routines for Hermitian matrices from LAPACK. See [1]_ for
    the algorithm details.

    Depending on the `lower` keyword value, only lower or upper triangular
    part of the input array is referenced. Moreover, this keyword also defines
    the structure of the outer factors of the factorization.

    .. versionadded:: 1.1.0

    See Also
    --------
    cholesky, lu

    References
    ----------
    .. [1] J.R. Bunch, L. Kaufman, Some stable methods for calculating
       inertia and solving symmetric linear systems, Math. Comput. Vol.31,
       1977. :doi:`10.2307/2005787`

    """
    a = atleast_2d(_asarray_validated(A, check_finite=check_finite))
    if a.shape[0] != a.shape[1]:
        raise ValueError('The input array "a" should be square.')
    # Return empty arrays for empty square input
    if a.size == 0:
        return empty_like(a), empty_like(a), np.array([], dtype=int)

    n = a.shape[0]
    r_or_c = complex if iscomplexobj(a) else float

    # Get the LAPACK routine
    if r_or_c is complex and hermitian:
        s, sl = 'hetrf', 'hetrf_lwork'
        if np.any(imag(diag(a))):
            warn('scipy.linalg.ldl():\nThe imaginary parts of the diagonal'
                 'are ignored. Use "hermitian=False" for factorization of'
                 'complex symmetric arrays.', ComplexWarning, stacklevel=2)
    else:
        s, sl = 'sytrf', 'sytrf_lwork'

    solver, solver_lwork = get_lapack_funcs((s, sl), (a,))
    lwork = _compute_lwork(solver_lwork, n, lower=lower)
    ldu, piv, info = solver(a, lwork=lwork, lower=lower,
                            overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('{} exited with the internal error "illegal value '
                         'in argument number {}". See LAPACK documentation '
                         'for the error codes.'.format(s.upper(), -info))

    swap_arr, pivot_arr = _ldl_sanitize_ipiv(piv, lower=lower)
    d, lu = _ldl_get_d_and_l(ldu, pivot_arr, lower=lower, hermitian=hermitian)
    lu, perm = _ldl_construct_tri_factor(lu, swap_arr, pivot_arr, lower=lower)

    return lu, d, perm


def _ldl_sanitize_ipiv(a, lower=True):
    """
    This helper function takes the rather strangely encoded permutation array
    returned by the LAPACK routines ?(HE/SY)TRF and converts it into
    regularized permutation and diagonal pivot size format.

    Since FORTRAN uses 1-indexing and LAPACK uses different start points for
    upper and lower formats there are certain offsets in the indices used
    below.

    Let's assume a result where the matrix is 6x6 and there are two 2x2
    and two 1x1 blocks reported by the routine. To ease the coding efforts,
    we still populate a 6-sized array and fill zeros as the following ::

        pivots = [2, 0, 2, 0, 1, 1]

    This denotes a diagonal matrix of the form ::

        [x x        ]
        [x x        ]
        [    x x    ]
        [    x x    ]
        [        x  ]
        [          x]

    In other words, we write 2 when the 2x2 block is first encountered and
    automatically write 0 to the next entry and skip the next spin of the
    loop. Thus, a separate counter or array appends to keep track of block
    sizes are avoided. If needed, zeros can be filtered out later without
    losing the block structure.

    Parameters
    ----------
    a : ndarray
        The permutation array ipiv returned by LAPACK
    lower : bool, optional
        The switch to select whether upper or lower triangle is chosen in
        the LAPACK call.

    Returns
    -------
    swap_ : ndarray
        The array that defines the row/column swap operations. For example,
        if row two is swapped with row four, the result is [0, 3, 2, 3].
    pivots : ndarray
        The array that defines the block diagonal structure as given above.

    """
    n = a.size
    swap_ = arange(n)
    pivots = zeros_like(swap_, dtype=int)
    skip_2x2 = False

    # Some upper/lower dependent offset values
    # range (s)tart, r(e)nd, r(i)ncrement
    x, y, rs, re, ri = (1, 0, 0, n, 1) if lower else (-1, -1, n-1, -1, -1)

    for ind in range(rs, re, ri):
        # If previous spin belonged already to a 2x2 block
        if skip_2x2:
            skip_2x2 = False
            continue

        cur_val = a[ind]
        # do we have a 1x1 block or not?
        if cur_val > 0:
            if cur_val != ind+1:
                # Index value != array value --> permutation required
                swap_[ind] = swap_[cur_val-1]
            pivots[ind] = 1
        # Not.
        elif cur_val < 0 and cur_val == a[ind+x]:
            # first neg entry of 2x2 block identifier
            if -cur_val != ind+2:
                # Index value != array value --> permutation required
                swap_[ind+x] = swap_[-cur_val-1]
            pivots[ind+y] = 2
            skip_2x2 = True
        else:  # Doesn't make sense, give up
            raise ValueError('While parsing the permutation array '
                             'in "scipy.linalg.ldl", invalid entries '
                             'found. The array syntax is invalid.')
    return swap_, pivots


def _ldl_get_d_and_l(ldu, pivs, lower=True, hermitian=True):
    """
    Helper function to extract the diagonal and triangular matrices for
    LDL.T factorization.

    Parameters
    ----------
    ldu : ndarray
        The compact output returned by the LAPACK routing
    pivs : ndarray
        The sanitized array of {0, 1, 2} denoting the sizes of the pivots. For
        every 2 there is a succeeding 0.
    lower : bool, optional
        If set to False, upper triangular part is considered.
    hermitian : bool, optional
        If set to False a symmetric complex array is assumed.

    Returns
    -------
    d : ndarray
        The block diagonal matrix.
    lu : ndarray
        The upper/lower triangular matrix
    """
    is_c = iscomplexobj(ldu)
    d = diag(diag(ldu))
    n = d.shape[0]
    blk_i = 0  # block index

    # row/column offsets for selecting sub-, super-diagonal
    x, y = (1, 0) if lower else (0, 1)

    lu = tril(ldu, -1) if lower else triu(ldu, 1)
    diag_inds = arange(n)
    lu[diag_inds, diag_inds] = 1

    for blk in pivs[pivs != 0]:
        # increment the block index and check for 2s
        # if 2 then copy the off diagonals depending on uplo
        inc = blk_i + blk

        if blk == 2:
            d[blk_i+x, blk_i+y] = ldu[blk_i+x, blk_i+y]
            # If Hermitian matrix is factorized, the cross-offdiagonal element
            # should be conjugated.
            if is_c and hermitian:
                d[blk_i+y, blk_i+x] = ldu[blk_i+x, blk_i+y].conj()
            else:
                d[blk_i+y, blk_i+x] = ldu[blk_i+x, blk_i+y]

            lu[blk_i+x, blk_i+y] = 0.
        blk_i = inc

    return d, lu


def _ldl_construct_tri_factor(lu, swap_vec, pivs, lower=True):
    """
    Helper function to construct explicit outer factors of LDL factorization.

    If lower is True the permuted factors are multiplied as L(1)*L(2)*...*L(k).
    Otherwise, the permuted factors are multiplied as L(k)*...*L(2)*L(1). See
    LAPACK documentation for more details.

    Parameters
    ----------
    lu : ndarray
        The triangular array that is extracted from LAPACK routine call with
        ones on the diagonals.
    swap_vec : ndarray
        The array that defines the row swapping indices. If the kth entry is m
        then rows k,m are swapped. Notice that the mth entry is not necessarily
        k to avoid undoing the swapping.
    pivs : ndarray
        The array that defines the block diagonal structure returned by
        _ldl_sanitize_ipiv().
    lower : bool, optional
        The boolean to switch between lower and upper triangular structure.

    Returns
    -------
    lu : ndarray
        The square outer factor which satisfies the L * D * L.T = A
    perm : ndarray
        The permutation vector that brings the lu to the triangular form

    Notes
    -----
    Note that the original argument "lu" is overwritten.

    """
    n = lu.shape[0]
    perm = arange(n)
    # Setup the reading order of the permutation matrix for upper/lower
    rs, re, ri = (n-1, -1, -1) if lower else (0, n, 1)

    for ind in range(rs, re, ri):
        s_ind = swap_vec[ind]
        if s_ind != ind:
            # Column start and end positions
            col_s = ind if lower else 0
            col_e = n if lower else ind+1

            # If we stumble upon a 2x2 block include both cols in the perm.
            if pivs[ind] == (0 if lower else 2):
                col_s += -1 if lower else 0
                col_e += 0 if lower else 1
            lu[[s_ind, ind], col_s:col_e] = lu[[ind, s_ind], col_s:col_e]
            perm[[s_ind, ind]] = perm[[ind, s_ind]]

    return lu, argsort(perm)
