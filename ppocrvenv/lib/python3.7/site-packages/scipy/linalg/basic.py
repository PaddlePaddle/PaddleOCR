#
# Author: Pearu Peterson, March 2002
#
# w/ additions by Travis Oliphant, March 2002
#              and Jake Vanderplas, August 2012

from warnings import warn
import numpy as np
from numpy import atleast_1d, atleast_2d
from .flinalg import get_flinalg_funcs
from .lapack import get_lapack_funcs, _compute_lwork
from .misc import LinAlgError, _datacopied, LinAlgWarning
from .decomp import _asarray_validated
from . import decomp, decomp_svd
from ._solve_toeplitz import levinson

__all__ = ['solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
           'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
           'pinv', 'pinv2', 'pinvh', 'matrix_balance', 'matmul_toeplitz']


# Linear equations
def _solve_check(n, info, lamch=None, rcond=None):
    """ Check arguments during the different steps of the solution phase """
    if info < 0:
        raise ValueError('LAPACK reported an illegal value in {}-th argument'
                         '.'.format(-info))
    elif 0 < info:
        raise LinAlgError('Matrix is singular.')

    if lamch is None:
        return
    E = lamch('E')
    if rcond < E:
        warn('Ill-conditioned matrix (rcond={:.6g}): '
             'result may not be accurate.'.format(rcond),
             LinAlgWarning, stacklevel=3)


def solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
          overwrite_b=False, debug=None, check_finite=True, assume_a='gen',
          transposed=False):
    """
    Solves the linear equation set ``a * x = b`` for the unknown ``x``
    for square ``a`` matrix.

    If the data matrix is known to be a particular type then supplying the
    corresponding string to ``assume_a`` key chooses the dedicated solver.
    The available options are

    ===================  ========
     generic matrix       'gen'
     symmetric            'sym'
     hermitian            'her'
     positive definite    'pos'
    ===================  ========

    If omitted, ``'gen'`` is the default structure.

    The datatype of the arrays define which solver is called regardless
    of the values. In other words, even when the complex array entries have
    precisely zero imaginary parts, the complex solver will be called based
    on the data type of the array.

    Parameters
    ----------
    a : (N, N) array_like
        Square input data
    b : (N, NRHS) array_like
        Input data for the right hand side.
    sym_pos : bool, optional
        Assume `a` is symmetric and positive definite. This key is deprecated
        and assume_a = 'pos' keyword is recommended instead. The functionality
        is the same. It will be removed in the future.
    lower : bool, optional
        If True, only the data contained in the lower triangle of `a`. Default
        is to use upper triangle. (ignored for ``'gen'``)
    overwrite_a : bool, optional
        Allow overwriting data in `a` (may enhance performance).
        Default is False.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance).
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    assume_a : str, optional
        Valid entries are explained above.
    transposed: bool, optional
        If True, ``a^T x = b`` for real matrices, raises `NotImplementedError`
        for complex matrices (only for True).

    Returns
    -------
    x : (N, NRHS) ndarray
        The solution array.

    Raises
    ------
    ValueError
        If size mismatches detected or input a is not square.
    LinAlgError
        If the matrix is singular.
    LinAlgWarning
        If an ill-conditioned input a is detected.
    NotImplementedError
        If transposed is True and input a is a complex matrix.

    Examples
    --------
    Given `a` and `b`, solve for `x`:

    >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    >>> b = np.array([2, 4, -1])
    >>> from scipy import linalg
    >>> x = linalg.solve(a, b)
    >>> x
    array([ 2., -2.,  9.])
    >>> np.dot(a, x) == b
    array([ True,  True,  True], dtype=bool)

    Notes
    -----
    If the input b matrix is a 1-D array with N elements, when supplied
    together with an NxN input a, it is assumed as a valid column vector
    despite the apparent size mismatch. This is compatible with the
    numpy.dot() behavior and the returned result is still 1-D array.

    The generic, symmetric, Hermitian and positive definite solutions are
    obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of
    LAPACK respectively.
    """
    # Flags for 1-D or N-D right-hand side
    b_is_1D = False

    a1 = atleast_2d(_asarray_validated(a, check_finite=check_finite))
    b1 = atleast_1d(_asarray_validated(b, check_finite=check_finite))
    n = a1.shape[0]

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input a needs to be a square matrix.')

    if n != b1.shape[0]:
        # Last chance to catch 1x1 scalar a and 1-D b arrays
        if not (n == 1 and b1.size != 0):
            raise ValueError('Input b has to have same number of rows as '
                             'input a')

    # accommodate empty arrays
    if b1.size == 0:
        return np.asfortranarray(b1.copy())

    # regularize 1-D b arrays to 2D
    if b1.ndim == 1:
        if n == 1:
            b1 = b1[None, :]
        else:
            b1 = b1[:, None]
        b_is_1D = True

    # Backwards compatibility - old keyword.
    if sym_pos:
        assume_a = 'pos'

    if assume_a not in ('gen', 'sym', 'her', 'pos'):
        raise ValueError('{} is not a recognized matrix structure'
                         ''.format(assume_a))

    # for a real matrix, describe it as "symmetric", not "hermitian"
    # (lapack doesn't know what to do with real hermitian matrices)
    if assume_a == 'her' and not np.iscomplexobj(a1):
        assume_a = 'sym'

    # Deprecate keyword "debug"
    if debug is not None:
        warn('Use of the "debug" keyword is deprecated '
             'and this keyword will be removed in future '
             'versions of SciPy.', DeprecationWarning, stacklevel=2)

    # Get the correct lamch function.
    # The LAMCH functions only exists for S and D
    # So for complex values we have to convert to real/double.
    if a1.dtype.char in 'fF':  # single precision
        lamch = get_lapack_funcs('lamch', dtype='f')
    else:
        lamch = get_lapack_funcs('lamch', dtype='d')

    # Currently we do not have the other forms of the norm calculators
    #   lansy, lanpo, lanhe.
    # However, in any case they only reduce computations slightly...
    lange = get_lapack_funcs('lange', (a1,))

    # Since the I-norm and 1-norm are the same for symmetric matrices
    # we can collect them all in this one call
    # Note however, that when issuing 'gen' and form!='none', then
    # the I-norm should be used
    if transposed:
        trans = 1
        norm = 'I'
        if np.iscomplexobj(a1):
            raise NotImplementedError('scipy.linalg.solve can currently '
                                      'not solve a^T x = b or a^H x = b '
                                      'for complex matrices.')
    else:
        trans = 0
        norm = '1'

    anorm = lange(norm, a1)

    # Generalized case 'gesv'
    if assume_a == 'gen':
        gecon, getrf, getrs = get_lapack_funcs(('gecon', 'getrf', 'getrs'),
                                               (a1, b1))
        lu, ipvt, info = getrf(a1, overwrite_a=overwrite_a)
        _solve_check(n, info)
        x, info = getrs(lu, ipvt, b1,
                        trans=trans, overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = gecon(lu, anorm, norm=norm)
    # Hermitian case 'hesv'
    elif assume_a == 'her':
        hecon, hesv, hesv_lw = get_lapack_funcs(('hecon', 'hesv',
                                                 'hesv_lwork'), (a1, b1))
        lwork = _compute_lwork(hesv_lw, n, lower)
        lu, ipvt, x, info = hesv(a1, b1, lwork=lwork,
                                 lower=lower,
                                 overwrite_a=overwrite_a,
                                 overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = hecon(lu, ipvt, anorm)
    # Symmetric case 'sysv'
    elif assume_a == 'sym':
        sycon, sysv, sysv_lw = get_lapack_funcs(('sycon', 'sysv',
                                                 'sysv_lwork'), (a1, b1))
        lwork = _compute_lwork(sysv_lw, n, lower)
        lu, ipvt, x, info = sysv(a1, b1, lwork=lwork,
                                 lower=lower,
                                 overwrite_a=overwrite_a,
                                 overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = sycon(lu, ipvt, anorm)
    # Positive definite case 'posv'
    else:
        pocon, posv = get_lapack_funcs(('pocon', 'posv'),
                                       (a1, b1))
        lu, x, info = posv(a1, b1, lower=lower,
                           overwrite_a=overwrite_a,
                           overwrite_b=overwrite_b)
        _solve_check(n, info)
        rcond, info = pocon(lu, anorm)

    _solve_check(n, info, lamch, rcond)

    if b_is_1D:
        x = x.ravel()

    return x


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, debug=None, check_finite=True):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        Right-hand side matrix in `a x = b`
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x  = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and
        will not be referenced.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system `a x = b`.  Shape of return matches `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular

    Notes
    -----
    .. versionadded:: 0.9.0

    Examples
    --------
    Solve the lower triangular system a x = b, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> from scipy.linalg import solve_triangular
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a.dot(x)  # Check the result
    array([ 4.,  2.,  4.,  2.])

    """

    # Deprecate keyword "debug"
    if debug is not None:
        warn('Use of the "debug" keyword is deprecated '
             'and this keyword will be removed in the future '
             'versions of SciPy.', DeprecationWarning, stacklevel=2)

    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    if a1.shape[0] != b1.shape[0]:
        raise ValueError('shapes of a {} and b {} are incompatible'
                         .format(a1.shape, b1.shape))
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if debug:
        print('solve:overwrite_b=', overwrite_b)
    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
    trtrs, = get_lapack_funcs(('trtrs',), (a1, b1))
    if a1.flags.f_contiguous or trans == 2:
        x, info = trtrs(a1, b1, overwrite_b=overwrite_b, lower=lower,
                        trans=trans, unitdiag=unit_diagonal)
    else:
        # transposed system is solved since trtrs expects Fortran ordering
        x, info = trtrs(a1.T, b1, overwrite_b=overwrite_b, lower=not lower,
                        trans=not trans, unitdiag=unit_diagonal)

    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" %
                          (info-1))
    raise ValueError('illegal value in %dth argument of internal trtrs' %
                     (-info))


def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False,
                 debug=None, check_finite=True):
    """
    Solve the equation a x = b for x, assuming a is banded matrix.

    The matrix a is stored in `ab` using the matrix diagonal ordered form::

        ab[u + i - j, j] == a[i,j]

    Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::

        *    a01  a12  a23  a34  a45
        a00  a11  a22  a33  a44  a55
        a10  a21  a32  a43  a54   *
        a20  a31  a42  a53   *    *

    Parameters
    ----------
    (l, u) : (integer, integer)
        Number of non-zero lower and upper diagonals
    ab : (`l` + `u` + 1, M) array_like
        Banded matrix
    b : (M,) or (M, K) array_like
        Right-hand side
    overwrite_ab : bool, optional
        Discard data in `ab` (may enhance performance)
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system a x = b. Returned shape depends on the
        shape of `b`.

    Examples
    --------
    Solve the banded system a x = b, where::

            [5  2 -1  0  0]       [0]
            [1  4  2 -1  0]       [1]
        a = [0  1  3  2 -1]   b = [2]
            [0  0  1  2  2]       [2]
            [0  0  0  1  1]       [3]

    There is one nonzero diagonal below the main diagonal (l = 1), and
    two above (u = 2). The diagonal banded form of the matrix is::

             [*  * -1 -1 -1]
        ab = [*  2  2  2  2]
             [5  4  3  2  1]
             [1  1  1  1  *]

    >>> from scipy.linalg import solve_banded
    >>> ab = np.array([[0,  0, -1, -1, -1],
    ...                [0,  2,  2,  2,  2],
    ...                [5,  4,  3,  2,  1],
    ...                [1,  1,  1,  1,  0]])
    >>> b = np.array([0, 1, 2, 2, 3])
    >>> x = solve_banded((1, 2), ab, b)
    >>> x
    array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])

    """

    # Deprecate keyword "debug"
    if debug is not None:
        warn('Use of the "debug" keyword is deprecated '
             'and this keyword will be removed in the future '
             'versions of SciPy.', DeprecationWarning, stacklevel=2)

    a1 = _asarray_validated(ab, check_finite=check_finite, as_inexact=True)
    b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)
    # Validate shapes.
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")
    (nlower, nupper) = l_and_u
    if nlower + nupper + 1 != a1.shape[0]:
        raise ValueError("invalid values for the number of lower and upper "
                         "diagonals: l+u+1 (%d) does not equal ab.shape[0] "
                         "(%d)" % (nlower + nupper + 1, ab.shape[0]))

    overwrite_b = overwrite_b or _datacopied(b1, b)
    if a1.shape[-1] == 1:
        b2 = np.array(b1, copy=(not overwrite_b))
        b2 /= a1[1, 0]
        return b2
    if nlower == nupper == 1:
        overwrite_ab = overwrite_ab or _datacopied(a1, ab)
        gtsv, = get_lapack_funcs(('gtsv',), (a1, b1))
        du = a1[0, 1:]
        d = a1[1, :]
        dl = a1[2, :-1]
        du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab,
                                   overwrite_ab, overwrite_b)
    else:
        gbsv, = get_lapack_funcs(('gbsv',), (a1, b1))
        a2 = np.zeros((2*nlower + nupper + 1, a1.shape[1]), dtype=gbsv.dtype)
        a2[nlower:, :] = a1
        lu, piv, x, info = gbsv(nlower, nupper, a2, b1, overwrite_ab=True,
                                overwrite_b=overwrite_b)
    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix")
    raise ValueError('illegal value in %d-th argument of internal '
                     'gbsv/gtsv' % -info)


def solveh_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False,
                  check_finite=True):
    """
    Solve equation a x = b. a is Hermitian positive-definite banded matrix.

    The matrix a is stored in `ab` either in lower diagonal or upper
    diagonal ordered form:

        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

    Example of `ab` (shape of a is (6, 6), `u` =2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Cells marked with * are not used.

    Parameters
    ----------
    ab : (`u` + 1, M) array_like
        Banded matrix
    b : (M,) or (M, K) array_like
        Right-hand side
    overwrite_ab : bool, optional
        Discard data in `ab` (may enhance performance)
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance)
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system a x = b. Shape of return matches shape
        of `b`.

    Examples
    --------
    Solve the banded system A x = b, where::

            [ 4  2 -1  0  0  0]       [1]
            [ 2  5  2 -1  0  0]       [2]
        A = [-1  2  6  2 -1  0]   b = [2]
            [ 0 -1  2  7  2 -1]       [3]
            [ 0  0 -1  2  8  2]       [3]
            [ 0  0  0 -1  2  9]       [3]

    >>> from scipy.linalg import solveh_banded

    `ab` contains the main diagonal and the nonzero diagonals below the
    main diagonal. That is, we use the lower form:

    >>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
    ...                [ 2,  2,  2,  2, 2, 0],
    ...                [-1, -1, -1, -1, 0, 0]])
    >>> b = np.array([1, 2, 2, 3, 3, 3])
    >>> x = solveh_banded(ab, b, lower=True)
    >>> x
    array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
            0.34733894])


    Solve the Hermitian banded system H x = b, where::

            [ 8   2-1j   0     0  ]        [ 1  ]
        H = [2+1j  5     1j    0  ]    b = [1+1j]
            [ 0   -1j    9   -2-1j]        [1-2j]
            [ 0    0   -2+1j   6  ]        [ 0  ]

    In this example, we put the upper diagonals in the array `hb`:

    >>> hb = np.array([[0, 2-1j, 1j, -2-1j],
    ...                [8,  5,    9,   6  ]])
    >>> b = np.array([1, 1+1j, 1-2j, 0])
    >>> x = solveh_banded(hb, b)
    >>> x
    array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
            0.10077984-0.23035393j, -0.00479904-0.09358128j])

    """
    a1 = _asarray_validated(ab, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    # Validate shapes.
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")

    overwrite_b = overwrite_b or _datacopied(b1, b)
    overwrite_ab = overwrite_ab or _datacopied(a1, ab)

    if a1.shape[0] == 2:
        ptsv, = get_lapack_funcs(('ptsv',), (a1, b1))
        if lower:
            d = a1[0, :].real
            e = a1[1, :-1]
        else:
            d = a1[1, :].real
            e = a1[0, 1:].conj()
        d, du, x, info = ptsv(d, e, b1, overwrite_ab, overwrite_ab,
                              overwrite_b)
    else:
        pbsv, = get_lapack_funcs(('pbsv',), (a1, b1))
        c, x, info = pbsv(a1, b1, lower=lower, overwrite_ab=overwrite_ab,
                          overwrite_b=overwrite_b)
    if info > 0:
        raise LinAlgError("%dth leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal '
                         'pbsv' % -info)
    return x


def solve_toeplitz(c_or_cr, b, check_finite=True):
    """Solve a Toeplitz system using Levinson Recursion

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system ``T x = b``. Shape of return matches shape
        of `b`.

    See Also
    --------
    toeplitz : Toeplitz matrix

    Notes
    -----
    The solution is computed using Levinson-Durbin recursion, which is faster
    than generic least-squares methods, but can be less numerically stable.

    Examples
    --------
    Solve the Toeplitz system T x = b, where::

            [ 1 -1 -2 -3]       [1]
        T = [ 3  1 -1 -2]   b = [2]
            [ 6  3  1 -1]       [2]
            [10  6  3  1]       [5]

    To specify the Toeplitz matrix, only the first column and the first
    row are needed.

    >>> c = np.array([1, 3, 6, 10])    # First column of T
    >>> r = np.array([1, -1, -2, -3])  # First row of T
    >>> b = np.array([1, 2, 2, 5])

    >>> from scipy.linalg import solve_toeplitz, toeplitz
    >>> x = solve_toeplitz((c, r), b)
    >>> x
    array([ 1.66666667, -1.        , -2.66666667,  2.33333333])

    Check the result by creating the full Toeplitz matrix and
    multiplying it by `x`.  We should get `b`.

    >>> T = toeplitz(c, r)
    >>> T.dot(x)
    array([ 1.,  2.,  2.,  5.])

    """
    # If numerical stability of this algorithm is a problem, a future
    # developer might consider implementing other O(N^2) Toeplitz solvers,
    # such as GKO (https://www.jstor.org/stable/2153371) or Bareiss.

    r, c, b, dtype, b_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, b, check_finite, keep_b_shape=True)

    # Form a 1-D array of values to be used in the matrix, containing a
    # reversed copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    if b is None:
        raise ValueError('illegal value, `b` is a required argument')

    if b.ndim == 1:
        x, _ = levinson(vals, np.ascontiguousarray(b))
    else:
        x = np.column_stack([levinson(vals, np.ascontiguousarray(b[:, i]))[0]
                             for i in range(b.shape[1])])
        x = x.reshape(*b_shape)

    return x


def _get_axis_len(aname, a, axis):
    ax = axis
    if ax < 0:
        ax += a.ndim
    if 0 <= ax < a.ndim:
        return a.shape[ax]
    raise ValueError("'%saxis' entry is out of bounds" % (aname,))


def solve_circulant(c, b, singular='raise', tol=None,
                    caxis=-1, baxis=0, outaxis=0):
    """Solve C x = b for x, where C is a circulant matrix.

    `C` is the circulant matrix associated with the vector `c`.

    The system is solved by doing division in Fourier space. The
    calculation is::

        x = ifft(fft(b) / fft(c))

    where `fft` and `ifft` are the fast Fourier transform and its inverse,
    respectively. For a large vector `c`, this is *much* faster than
    solving the system with the full circulant matrix.

    Parameters
    ----------
    c : array_like
        The coefficients of the circulant matrix.
    b : array_like
        Right-hand side matrix in ``a x = b``.
    singular : str, optional
        This argument controls how a near singular circulant matrix is
        handled.  If `singular` is "raise" and the circulant matrix is
        near singular, a `LinAlgError` is raised. If `singular` is
        "lstsq", the least squares solution is returned. Default is "raise".
    tol : float, optional
        If any eigenvalue of the circulant matrix has an absolute value
        that is less than or equal to `tol`, the matrix is considered to be
        near singular. If not given, `tol` is set to::

            tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps

        where `abs_eigs` is the array of absolute values of the eigenvalues
        of the circulant matrix.
    caxis : int
        When `c` has dimension greater than 1, it is viewed as a collection
        of circulant vectors. In this case, `caxis` is the axis of `c` that
        holds the vectors of circulant coefficients.
    baxis : int
        When `b` has dimension greater than 1, it is viewed as a collection
        of vectors. In this case, `baxis` is the axis of `b` that holds the
        right-hand side vectors.
    outaxis : int
        When `c` or `b` are multidimensional, the value returned by
        `solve_circulant` is multidimensional. In this case, `outaxis` is
        the axis of the result that holds the solution vectors.

    Returns
    -------
    x : ndarray
        Solution to the system ``C x = b``.

    Raises
    ------
    LinAlgError
        If the circulant matrix associated with `c` is near singular.

    See Also
    --------
    circulant : circulant matrix

    Notes
    -----
    For a 1-D vector `c` with length `m`, and an array `b`
    with shape ``(m, ...)``,

        solve_circulant(c, b)

    returns the same result as

        solve(circulant(c), b)

    where `solve` and `circulant` are from `scipy.linalg`.

    .. versionadded:: 0.16.0

    Examples
    --------
    >>> from scipy.linalg import solve_circulant, solve, circulant, lstsq

    >>> c = np.array([2, 2, 4])
    >>> b = np.array([1, 2, 3])
    >>> solve_circulant(c, b)
    array([ 0.75, -0.25,  0.25])

    Compare that result to solving the system with `scipy.linalg.solve`:

    >>> solve(circulant(c), b)
    array([ 0.75, -0.25,  0.25])

    A singular example:

    >>> c = np.array([1, 1, 0, 0])
    >>> b = np.array([1, 2, 3, 4])

    Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the
    least square solution, use the option ``singular='lstsq'``:

    >>> solve_circulant(c, b, singular='lstsq')
    array([ 0.25,  1.25,  2.25,  1.25])

    Compare to `scipy.linalg.lstsq`:

    >>> x, resid, rnk, s = lstsq(circulant(c), b)
    >>> x
    array([ 0.25,  1.25,  2.25,  1.25])

    A broadcasting example:

    Suppose we have the vectors of two circulant matrices stored in an array
    with shape (2, 5), and three `b` vectors stored in an array with shape
    (3, 5).  For example,

    >>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])
    >>> b = np.arange(15).reshape(-1, 5)

    We want to solve all combinations of circulant matrices and `b` vectors,
    with the result stored in an array with shape (2, 3, 5). When we
    disregard the axes of `c` and `b` that hold the vectors of coefficients,
    the shapes of the collections are (2,) and (3,), respectively, which are
    not compatible for broadcasting. To have a broadcast result with shape
    (2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has
    shape (2, 1, 5). The last dimension holds the coefficients of the
    circulant matrices, so when we call `solve_circulant`, we can use the
    default ``caxis=-1``. The coefficients of the `b` vectors are in the last
    dimension of the array `b`, so we use ``baxis=-1``. If we use the
    default `outaxis`, the result will have shape (5, 2, 3), so we'll use
    ``outaxis=-1`` to put the solution vectors in the last dimension.

    >>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)
    >>> x.shape
    (2, 3, 5)
    >>> np.set_printoptions(precision=3)  # For compact output of numbers.
    >>> x
    array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],
            [ 0.651,  0.989,  2.046,  0.627,  1.072],
            [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],
           [[ 0.401,  0.304,  0.694, -0.867,  0.377],
            [ 0.856,  0.758,  1.149, -0.412,  0.831],
            [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])

    Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):

    >>> solve_circulant(c[1], b[1, :])
    array([ 0.856,  0.758,  1.149, -0.412,  0.831])

    """
    c = np.atleast_1d(c)
    nc = _get_axis_len("c", c, caxis)
    b = np.atleast_1d(b)
    nb = _get_axis_len("b", b, baxis)
    if nc != nb:
        raise ValueError('Shapes of c {} and b {} are incompatible'
                         .format(c.shape, b.shape))

    fc = np.fft.fft(np.rollaxis(c, caxis, c.ndim), axis=-1)
    abs_fc = np.abs(fc)
    if tol is None:
        # This is the same tolerance as used in np.linalg.matrix_rank.
        tol = abs_fc.max(axis=-1) * nc * np.finfo(np.float64).eps
        if tol.shape != ():
            tol.shape = tol.shape + (1,)
        else:
            tol = np.atleast_1d(tol)

    near_zeros = abs_fc <= tol
    is_near_singular = np.any(near_zeros)
    if is_near_singular:
        if singular == 'raise':
            raise LinAlgError("near singular circulant matrix.")
        else:
            # Replace the small values with 1 to avoid errors in the
            # division fb/fc below.
            fc[near_zeros] = 1

    fb = np.fft.fft(np.rollaxis(b, baxis, b.ndim), axis=-1)

    q = fb / fc

    if is_near_singular:
        # `near_zeros` is a boolean array, same shape as `c`, that is
        # True where `fc` is (near) zero. `q` is the broadcasted result
        # of fb / fc, so to set the values of `q` to 0 where `fc` is near
        # zero, we use a mask that is the broadcast result of an array
        # of True values shaped like `b` with `near_zeros`.
        mask = np.ones_like(b, dtype=bool) & near_zeros
        q[mask] = 0

    x = np.fft.ifft(q, axis=-1)
    if not (np.iscomplexobj(c) or np.iscomplexobj(b)):
        x = x.real
    if outaxis != -1:
        x = np.rollaxis(x, -1, outaxis)
    return x


# matrix inversion
def inv(a, overwrite_a=False, check_finite=True):
    """
    Compute the inverse of a matrix.

    Parameters
    ----------
    a : array_like
        Square matrix to be inverted.
    overwrite_a : bool, optional
        Discard data in `a` (may improve performance). Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    ainv : ndarray
        Inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is singular.
    ValueError
        If `a` is not square, or not 2D.

    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> linalg.inv(a)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    >>> np.dot(a, linalg.inv(a))
    array([[ 1.,  0.],
           [ 0.,  1.]])

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or _datacopied(a1, a)
    # XXX: I found no advantage or disadvantage of using finv.
#     finv, = get_flinalg_funcs(('inv',),(a1,))
#     if finv is not None:
#         a_inv,info = finv(a1,overwrite_a=overwrite_a)
#         if info==0:
#             return a_inv
#         if info>0: raise LinAlgError, "singular matrix"
#         if info<0: raise ValueError('illegal value in %d-th argument of '
#                                     'internal inv.getrf|getri'%(-info))
    getrf, getri, getri_lwork = get_lapack_funcs(('getrf', 'getri',
                                                  'getri_lwork'),
                                                 (a1,))
    lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
    if info == 0:
        lwork = _compute_lwork(getri_lwork, a1.shape[0])

        # XXX: the following line fixes curious SEGFAULT when
        # benchmarking 500x500 matrix inverse. This seems to
        # be a bug in LAPACK ?getri routine because if lwork is
        # minimal (when using lwork[0] instead of lwork[1]) then
        # all tests pass. Further investigation is required if
        # more such SEGFAULTs occur.
        lwork = int(1.01 * lwork)
        inv_a, info = getri(lu, piv, lwork=lwork, overwrite_lu=1)
    if info > 0:
        raise LinAlgError("singular matrix")
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'getrf|getri' % -info)
    return inv_a


# Determinant

def det(a, overwrite_a=False, check_finite=True):
    """
    Compute the determinant of a matrix

    The determinant of a square matrix is a value derived arithmetically
    from the coefficients of the matrix.

    The determinant for a 3x3 matrix, for example, is computed as follows::

        a    b    c
        d    e    f = A
        g    h    i

        det(A) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

    Parameters
    ----------
    a : (M, M) array_like
        A square matrix.
    overwrite_a : bool, optional
        Allow overwriting data in a (may enhance performance).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    det : float or complex
        Determinant of `a`.

    Notes
    -----
    The determinant is computed via LU factorization, LAPACK routine z/dgetrf.

    Examples
    --------
    >>> from scipy import linalg
    >>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> linalg.det(a)
    0.0
    >>> a = np.array([[0,2,3], [4,5,6], [7,8,9]])
    >>> linalg.det(a)
    3.0

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or _datacopied(a1, a)
    fdet, = get_flinalg_funcs(('det',), (a1,))
    a_det, info = fdet(a1, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal '
                         'det.getrf' % -info)
    return a_det


# Linear Least Squares
def lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True, lapack_driver=None):
    """
    Compute least-squares solution to equation Ax = b.

    Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.

    Parameters
    ----------
    a : (M, N) array_like
        Left-hand side array
    b : (M,) or (M, K) array_like
        Right hand side array
    cond : float, optional
        Cutoff for 'small' singular values; used to determine effective
        rank of a. Singular values smaller than
        ``rcond * largest_singular_value`` are considered zero.
    overwrite_a : bool, optional
        Discard data in `a` (may enhance performance). Default is False.
    overwrite_b : bool, optional
        Discard data in `b` (may enhance performance). Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : str, optional
        Which LAPACK driver is used to solve the least-squares problem.
        Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
        (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
        faster on many problems.  ``'gelss'`` was used historically.  It is
        generally slow but uses less memory.

        .. versionadded:: 0.17.0

    Returns
    -------
    x : (N,) or (N, K) ndarray
        Least-squares solution.  Return shape matches shape of `b`.
    residues : (K,) ndarray or float
        Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
        ``ndim(A) == n`` (returns a scalar if b is 1-D). Otherwise a
        (0,)-shaped array is returned.
    rank : int
        Effective rank of `a`.
    s : (min(M, N),) ndarray or None
        Singular values of `a`. The condition number of a is
        ``abs(s[0] / s[-1])``.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    ValueError
        When parameters are not compatible.

    See Also
    --------
    scipy.optimize.nnls : linear least squares with non-negativity constraint

    Notes
    -----
    When ``'gelsy'`` is used as a driver, `residues` is set to a (0,)-shaped
    array and `s` is always ``None``.

    Examples
    --------
    >>> from scipy.linalg import lstsq
    >>> import matplotlib.pyplot as plt

    Suppose we have the following data:

    >>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
    >>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

    We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
    to this data.  We first form the "design matrix" M, with a constant
    column of 1s and a column containing ``x**2``:

    >>> M = x[:, np.newaxis]**[0, 2]
    >>> M
    array([[  1.  ,   1.  ],
           [  1.  ,   6.25],
           [  1.  ,  12.25],
           [  1.  ,  16.  ],
           [  1.  ,  25.  ],
           [  1.  ,  49.  ],
           [  1.  ,  72.25]])

    We want to find the least-squares solution to ``M.dot(p) = y``,
    where ``p`` is a vector with length 2 that holds the parameters
    ``a`` and ``b``.

    >>> p, res, rnk, s = lstsq(M, y)
    >>> p
    array([ 0.20925829,  0.12013861])

    Plot the data and the fitted curve.

    >>> plt.plot(x, y, 'o', label='data')
    >>> xx = np.linspace(0, 9, 101)
    >>> yy = p[0] + p[1]*xx**2
    >>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend(framealpha=1, shadow=True)
    >>> plt.grid(alpha=0.25)
    >>> plt.show()

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError('Input array a should be 2D')
    m, n = a1.shape
    if len(b1.shape) == 2:
        nrhs = b1.shape[1]
    else:
        nrhs = 1
    if m != b1.shape[0]:
        raise ValueError('Shape mismatch: a and b should have the same number'
                         ' of rows ({} != {}).'.format(m, b1.shape[0]))
    if m == 0 or n == 0:  # Zero-sized problem, confuses LAPACK
        x = np.zeros((n,) + b1.shape[1:], dtype=np.common_type(a1, b1))
        if n == 0:
            residues = np.linalg.norm(b1, axis=0)**2
        else:
            residues = np.empty((0,))
        return x, residues, 0, np.empty((0,))

    driver = lapack_driver
    if driver is None:
        driver = lstsq.default_lapack_driver
    if driver not in ('gelsd', 'gelsy', 'gelss'):
        raise ValueError('LAPACK driver "%s" is not found' % driver)

    lapack_func, lapack_lwork = get_lapack_funcs((driver,
                                                 '%s_lwork' % driver),
                                                 (a1, b1))
    real_data = True if (lapack_func.dtype.kind == 'f') else False

    if m < n:
        # need to extend b matrix as it will be filled with
        # a larger solution matrix
        if len(b1.shape) == 2:
            b2 = np.zeros((n, nrhs), dtype=lapack_func.dtype)
            b2[:m, :] = b1
        else:
            b2 = np.zeros(n, dtype=lapack_func.dtype)
            b2[:m] = b1
        b1 = b2

    overwrite_a = overwrite_a or _datacopied(a1, a)
    overwrite_b = overwrite_b or _datacopied(b1, b)

    if cond is None:
        cond = np.finfo(lapack_func.dtype).eps

    if driver in ('gelss', 'gelsd'):
        if driver == 'gelss':
            lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
            v, x, s, rank, work, info = lapack_func(a1, b1, cond, lwork,
                                                    overwrite_a=overwrite_a,
                                                    overwrite_b=overwrite_b)

        elif driver == 'gelsd':
            if real_data:
                lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
                x, s, rank, info = lapack_func(a1, b1, lwork,
                                               iwork, cond, False, False)
            else:  # complex data
                lwork, rwork, iwork = _compute_lwork(lapack_lwork, m, n,
                                                     nrhs, cond)
                x, s, rank, info = lapack_func(a1, b1, lwork, rwork, iwork,
                                               cond, False, False)
        if info > 0:
            raise LinAlgError("SVD did not converge in Linear Least Squares")
        if info < 0:
            raise ValueError('illegal value in %d-th argument of internal %s'
                             % (-info, lapack_driver))
        resids = np.asarray([], dtype=x.dtype)
        if m > n:
            x1 = x[:n]
            if rank == n:
                resids = np.sum(np.abs(x[n:])**2, axis=0)
            x = x1
        return x, resids, rank, s

    elif driver == 'gelsy':
        lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
        jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
        v, x, j, rank, info = lapack_func(a1, b1, jptv, cond,
                                          lwork, False, False)
        if info < 0:
            raise ValueError("illegal value in %d-th argument of internal "
                             "gelsy" % -info)
        if m > n:
            x1 = x[:n]
            x = x1
        return x, np.array([], x.dtype), rank, None


lstsq.default_lapack_driver = 'gelsd'


def pinv(a, atol=None, rtol=None, return_rank=False, check_finite=True,
         cond=None, rcond=None):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition ``U @ S @ V`` in the economy mode and picking
    up only the columns/rows that are associated with significant singular
    values.

    If ``s`` is the maximum singular value of ``a``, then the
    significance cut-off value is determined by ``atol + rtol * s``. Any
    singular value below this value is assumed insignificant.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    atol: float, optional
        Absolute threshold term, default value is 0.

        .. versionadded:: 1.7.0

    rtol: float, optional
        Relative threshold term, default value is ``max(M, N) * eps`` where
        ``eps`` is the machine precision value of the datatype of ``a``.

        .. versionadded:: 1.7.0

    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    cond, rcond : float, optional
        In older versions, these values were meant to be used as ``atol`` with
        ``rtol=0``. If both were given ``rcond`` overwrote ``cond`` and hence
        the code was not correct. Thus using these are strongly discouraged and
        the tolerances above are recommended instead. In fact, if provided,
        atol, rtol takes precedence over these keywords.

        .. versionchanged:: 1.7.0
            Deprecated in favor of ``rtol`` and ``atol`` parameters above and
            will be removed in future versions of SciPy.

        .. versionchanged:: 1.3.0
            Previously the default cutoff value was just ``eps*f`` where ``f``
            was ``1e3`` for single precision and ``1e6`` for double precision.

    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix. Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Examples
    --------
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((9, 6))
    >>> B = linalg.pinv(a)
    >>> np.allclose(a, a @ B @ a)
    True
    >>> np.allclose(B, B @ a @ B)
    True

    """
    a = _asarray_validated(a, check_finite=check_finite)
    u, s, vh = decomp_svd.svd(a, full_matrices=False, check_finite=False)
    t = u.dtype.char.lower()
    maxS = np.max(s)

    if rcond or cond:
        warn('Use of the "cond" and "rcond" keywords are deprecated and '
             'will be removed in future versions of SciPy. Use "atol" and '
             '"rtol" keywords instead', DeprecationWarning, stacklevel=2)

    # backwards compatible only atol and rtol are both missing
    if (rcond or cond) and (atol is None) and (rtol is None):
        atol = rcond or cond
        rtol = 0.

    atol = 0. if atol is None else atol
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol

    if (atol < 0.) or (rtol < 0.):
        raise ValueError("atol and rtol values must be positive.")

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    if return_rank:
        return B, rank
    else:
        return B


def pinv2(a, cond=None, rcond=None, return_rank=False, check_finite=True):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    `scipy.linalg.pinv2` is deprecated since SciPy 1.7.0, use
    `scipy.linalg.pinv` instead for better tolerance control.

    Calculate a generalized inverse of a matrix using its
    singular-value decomposition and including all 'large' singular
    values.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    cond, rcond : float or None
        Cutoff for 'small' singular values; singular values smaller than this
        value are considered as zero. If both are omitted, the default value
        ``max(M,N)*largest_singular_value*eps`` is used where ``eps`` is the
        machine precision value of the datatype of ``a``.

        .. versionchanged:: 1.3.0
            Previously the default cutoff value was just ``eps*f`` where ``f``
            was ``1e3`` for single precision and ``1e6`` for double precision.

    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix. Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    """
    # SciPy 1.7.0 2021-04-10
    warn('scipy.linalg.pinv2 is deprecated since SciPy 1.7.0, use '
         'scipy.linalg.pinv instead', DeprecationWarning, stacklevel=2)
    if rcond is not None:
        cond = rcond

    return pinv(a=a, atol=cond, rtol=None, return_rank=return_rank,
                check_finite=check_finite)


def pinvh(a, atol=None, rtol=None, lower=True, return_rank=False,
          check_finite=True, cond=None, rcond=None):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.

    Calculate a generalized inverse of a copmlex Hermitian/real symmetric
    matrix using its eigenvalue decomposition and including all eigenvalues
    with 'large' absolute value.

    Parameters
    ----------
    a : (N, N) array_like
        Real symmetric or complex hermetian matrix to be pseudo-inverted
    atol: float, optional
        Absolute threshold term, default value is 0.

        .. versionadded:: 1.7.0

    rtol: float, optional
        Relative threshold term, default value is ``N * eps`` where
        ``eps`` is the machine precision value of the datatype of ``a``.

        .. versionadded:: 1.7.0

    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of `a`. (Default: lower)
    return_rank : bool, optional
        If True, return the effective rank of the matrix.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    cond, rcond : float, optional
        In older versions, these values were meant to be used as ``atol`` with
        ``rtol=0``. If both were given ``rcond`` overwrote ``cond`` and hence
        the code was not correct. Thus using these are strongly discouraged and
        the tolerances above are recommended instead.  In fact, if provided,
        atol, rtol takes precedence over these keywords.

        .. versionchanged:: 1.7.0
            Deprecated in favor of ``rtol`` and ``atol`` parameters above and
            will be removed in future versions of SciPy.

        .. versionchanged:: 1.3.0
            Previously the default cutoff value was just ``eps*f`` where ``f``
            was ``1e3`` for single precision and ``1e6`` for double precision.

    Returns
    -------
    B : (N, N) ndarray
        The pseudo-inverse of matrix `a`.
    rank : int
        The effective rank of the matrix.  Returned if `return_rank` is True.

    Raises
    ------
    LinAlgError
        If eigenvalue algorithm does not converge.

    Examples
    --------
    >>> from scipy.linalg import pinvh
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((9, 6))
    >>> a = np.dot(a, a.T)
    >>> B = pinvh(a)
    >>> np.allclose(a, a @ B @ a)
    True
    >>> np.allclose(B, B @ a @ B)
    True

    """
    a = _asarray_validated(a, check_finite=check_finite)
    s, u = decomp.eigh(a, lower=lower, check_finite=False)
    t = u.dtype.char.lower()
    maxS = np.max(np.abs(s))

    if rcond or cond:
        warn('Use of the "cond" and "rcond" keywords are deprecated and '
             'will be removed in future versions of SciPy. Use "atol" and '
             '"rtol" keywords instead', DeprecationWarning, stacklevel=2)

    # backwards compatible only atol and rtol are both missing
    if (rcond or cond) and (atol is None) and (rtol is None):
        atol = rcond or cond
        rtol = 0.

    atol = 0. if atol is None else atol
    rtol = max(a.shape) * np.finfo(t).eps if (rtol is None) else rtol

    if (atol < 0.) or (rtol < 0.):
        raise ValueError("atol and rtol values must be positive.")

    val = atol + maxS * rtol
    above_cutoff = (abs(s) > val)

    psigma_diag = 1.0 / s[above_cutoff]
    u = u[:, above_cutoff]

    B = (u * psigma_diag) @ u.conj().T

    if return_rank:
        return B, len(psigma_diag)
    else:
        return B


def matrix_balance(A, permute=True, scale=True, separate=False,
                   overwrite_a=False):
    """
    Compute a diagonal similarity transformation for row/column balancing.

    The balancing tries to equalize the row and column 1-norms by applying
    a similarity transformation such that the magnitude variation of the
    matrix entries is reflected to the scaling matrices.

    Moreover, if enabled, the matrix is first permuted to isolate the upper
    triangular parts of the matrix and, again if scaling is also enabled,
    only the remaining subblocks are subjected to scaling.

    The balanced matrix satisfies the following equality

    .. math::

                        B = T^{-1} A T

    The scaling coefficients are approximated to the nearest power of 2
    to avoid round-off errors.

    Parameters
    ----------
    A : (n, n) array_like
        Square data matrix for the balancing.
    permute : bool, optional
        The selector to define whether permutation of A is also performed
        prior to scaling.
    scale : bool, optional
        The selector to turn on and off the scaling. If False, the matrix
        will not be scaled.
    separate : bool, optional
        This switches from returning a full matrix of the transformation
        to a tuple of two separate 1-D permutation and scaling arrays.
    overwrite_a : bool, optional
        This is passed to xGEBAL directly. Essentially, overwrites the result
        to the data. It might increase the space efficiency. See LAPACK manual
        for details. This is False by default.

    Returns
    -------
    B : (n, n) ndarray
        Balanced matrix
    T : (n, n) ndarray
        A possibly permuted diagonal matrix whose nonzero entries are
        integer powers of 2 to avoid numerical truncation errors.
    scale, perm : (n,) ndarray
        If ``separate`` keyword is set to True then instead of the array
        ``T`` above, the scaling and the permutation vectors are given
        separately as a tuple without allocating the full array ``T``.

    Notes
    -----

    This algorithm is particularly useful for eigenvalue and matrix
    decompositions and in many cases it is already called by various
    LAPACK routines.

    The algorithm is based on the well-known technique of [1]_ and has
    been modified to account for special cases. See [2]_ for details
    which have been implemented since LAPACK v3.5.0. Before this version
    there are corner cases where balancing can actually worsen the
    conditioning. See [3]_ for such examples.

    The code is a wrapper around LAPACK's xGEBAL routine family for matrix
    balancing.

    .. versionadded:: 0.19.0

    Examples
    --------
    >>> from scipy import linalg
    >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])

    >>> y, permscale = linalg.matrix_balance(x)
    >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
    array([ 3.66666667,  0.4995005 ,  0.91312162])

    >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
    array([ 1.2       ,  1.27041742,  0.92658316])  # may vary

    >>> permscale  # only powers of 2 (0.5 == 2^(-1))
    array([[  0.5,   0. ,  0. ],  # may vary
           [  0. ,   1. ,  0. ],
           [  0. ,   0. ,  1. ]])

    References
    ----------
    .. [1] : B.N. Parlett and C. Reinsch, "Balancing a Matrix for
       Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,
       Vol.13(4), 1969, :doi:`10.1007/BF02165404`

    .. [2] : R. James, J. Langou, B.R. Lowery, "On matrix balancing and
       eigenvector computation", 2014, :arxiv:`1401.5766`

    .. [3] :  D.S. Watkins. A case where balancing is harmful.
       Electron. Trans. Numer. Anal, Vol.23, 2006.

    """

    A = np.atleast_2d(_asarray_validated(A, check_finite=True))

    if not np.equal(*A.shape):
        raise ValueError('The data matrix for balancing should be square.')

    gebal = get_lapack_funcs(('gebal'), (A,))
    B, lo, hi, ps, info = gebal(A, scale=scale, permute=permute,
                                overwrite_a=overwrite_a)

    if info < 0:
        raise ValueError('xGEBAL exited with the internal error '
                         '"illegal value in argument number {}.". See '
                         'LAPACK documentation for the xGEBAL error codes.'
                         ''.format(-info))

    # Separate the permutations from the scalings and then convert to int
    scaling = np.ones_like(ps, dtype=float)
    scaling[lo:hi+1] = ps[lo:hi+1]

    # gebal uses 1-indexing
    ps = ps.astype(int, copy=False) - 1
    n = A.shape[0]
    perm = np.arange(n)

    # LAPACK permutes with the ordering n --> hi, then 0--> lo
    if hi < n:
        for ind, x in enumerate(ps[hi+1:][::-1], 1):
            if n-ind == x:
                continue
            perm[[x, n-ind]] = perm[[n-ind, x]]

    if lo > 0:
        for ind, x in enumerate(ps[:lo]):
            if ind == x:
                continue
            perm[[x, ind]] = perm[[ind, x]]

    if separate:
        return B, (scaling, perm)

    # get the inverse permutation
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(n)

    return B, np.diag(scaling)[iperm, :]


def _validate_args_for_toeplitz_ops(c_or_cr, b, check_finite, keep_b_shape,
                                    enforce_square=True):
    """Validate arguments and format inputs for toeplitz functions

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.
    keep_b_shape: bool
        Whether to convert a (M,) dimensional b into a (M, 1) dimensional
        matrix.
    enforce_square: bool, optional
        If True (default), this verifies that the Toeplitz matrix is square.

    Returns
    -------
    r : array
        1d array corresponding to the first row of the Toeplitz matrix.
    c: array
        1d array corresponding to the first column of the Toeplitz matrix.
    b: array
        (M,), (M, 1) or (M, K) dimensional array, post validation,
        corresponding to ``b``.
    dtype: numpy datatype
        ``dtype`` stores the datatype of ``r``, ``c`` and ``b``. If any of
        ``r``, ``c`` or ``b`` are complex, ``dtype`` is ``np.complex128``,
        otherwise, it is ``np.float``.
    b_shape: tuple
        Shape of ``b`` after passing it through ``_asarray_validated``.

    """

    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        c = _asarray_validated(c, check_finite=check_finite).ravel()
        r = _asarray_validated(r, check_finite=check_finite).ravel()
    else:
        c = _asarray_validated(c_or_cr, check_finite=check_finite).ravel()
        r = c.conjugate()

    if b is None:
        raise ValueError('`b` must be an array, not None.')

    b = _asarray_validated(b, check_finite=check_finite)
    b_shape = b.shape

    is_not_square = r.shape[0] != c.shape[0]
    if (enforce_square and is_not_square) or b.shape[0] != r.shape[0]:
        raise ValueError('Incompatible dimensions.')

    is_cmplx = np.iscomplexobj(r) or np.iscomplexobj(c) or np.iscomplexobj(b)
    dtype = np.complex128 if is_cmplx else np.double
    r, c, b = (np.asarray(i, dtype=dtype) for i in (r, c, b))

    if b.ndim == 1 and not keep_b_shape:
        b = b.reshape(-1, 1)
    elif b.ndim != 1:
        b = b.reshape(b.shape[0], -1)

    return r, c, b, dtype, b_shape


def matmul_toeplitz(c_or_cr, x, check_finite=False, workers=None):
    """Efficient Toeplitz Matrix-Matrix Multiplication using FFT

    This function returns the matrix multiplication between a Toeplitz
    matrix and a dense matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
        actual shape of ``c``, it will be converted to a 1-D array. If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``. Whatever the actual shape
        of ``r``, it will be converted to a 1-D array.
    x : (M,) or (M, K) array_like
        Matrix with which to multiply.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.
    workers : int, optional
        To pass to scipy.fft.fft and ifft. Maximum number of workers to use
        for parallel computation. If negative, the value wraps around from
        ``os.cpu_count()``. See scipy.fft.fft for more details.

    Returns
    -------
    T @ x : (M,) or (M, K) ndarray
        The result of the matrix multiplication ``T @ x``. Shape of return
        matches shape of `x`.

    See Also
    --------
    toeplitz : Toeplitz matrix
    solve_toeplitz : Solve a Toeplitz system using Levinson Recursion

    Notes
    -----
    The Toeplitz matrix is embedded in a circulant matrix and the FFT is used
    to efficiently calculate the matrix-matrix product.

    Because the computation is based on the FFT, integer inputs will
    result in floating point outputs.  This is unlike NumPy's `matmul`,
    which preserves the data type of the input.

    This is partly based on the implementation that can be found in [1]_,
    licensed under the MIT license. More information about the method can be
    found in reference [2]_. References [3]_ and [4]_ have more reference
    implementations in Python.

    .. versionadded:: 1.6.0

    References
    ----------
    .. [1] Jacob R Gardner, Geoff Pleiss, David Bindel, Kilian
       Q Weinberger, Andrew Gordon Wilson, "GPyTorch: Blackbox Matrix-Matrix
       Gaussian Process Inference with GPU Acceleration" with contributions
       from Max Balandat and Ruihan Wu. Available online:
       https://github.com/cornellius-gp/gpytorch

    .. [2] J. Demmel, P. Koev, and X. Li, "A Brief Survey of Direct Linear
       Solvers". In Z. Bai, J. Demmel, J. Dongarra, A. Ruhe, and H. van der
       Vorst, editors. Templates for the Solution of Algebraic Eigenvalue
       Problems: A Practical Guide. SIAM, Philadelphia, 2000. Available at:
       http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html

    .. [3] R. Scheibler, E. Bezzam, I. Dokmanic, Pyroomacoustics: A Python
       package for audio room simulations and array processing algorithms,
       Proc. IEEE ICASSP, Calgary, CA, 2018.
       https://github.com/LCAV/pyroomacoustics/blob/pypi-release/
       pyroomacoustics/adaptive/util.py

    .. [4] Marano S, Edwards B, Ferrari G and Fah D (2017), "Fitting
       Earthquake Spectra: Colored Noise and Incomplete Data", Bulletin of
       the Seismological Society of America., January, 2017. Vol. 107(1),
       pp. 276-291.

    Examples
    --------
    Multiply the Toeplitz matrix T with matrix x::

            [ 1 -1 -2 -3]       [1 10]
        T = [ 3  1 -1 -2]   x = [2 11]
            [ 6  3  1 -1]       [2 11]
            [10  6  3  1]       [5 19]

    To specify the Toeplitz matrix, only the first column and the first
    row are needed.

    >>> c = np.array([1, 3, 6, 10])    # First column of T
    >>> r = np.array([1, -1, -2, -3])  # First row of T
    >>> x = np.array([[1, 10], [2, 11], [2, 11], [5, 19]])

    >>> from scipy.linalg import toeplitz, matmul_toeplitz
    >>> matmul_toeplitz((c, r), x)
    array([[-20., -80.],
           [ -7.,  -8.],
           [  9.,  85.],
           [ 33., 218.]])

    Check the result by creating the full Toeplitz matrix and
    multiplying it by ``x``.

    >>> toeplitz(c, r) @ x
    array([[-20, -80],
           [ -7,  -8],
           [  9,  85],
           [ 33, 218]])

    The full matrix is never formed explicitly, so this routine
    is suitable for very large Toeplitz matrices.

    >>> n = 1000000
    >>> matmul_toeplitz([1] + [0]*(n-1), np.ones(n))
    array([1., 1., 1., ..., 1., 1., 1.])

    """

    from ..fft import fft, ifft, rfft, irfft

    r, c, x, dtype, x_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, x, check_finite, keep_b_shape=False, enforce_square=False)
    n, m = x.shape

    T_nrows = len(c)
    T_ncols = len(r)
    p = T_nrows + T_ncols - 1  # equivalent to len(embedded_col)

    embedded_col = np.concatenate((c, r[-1:0:-1]))

    if np.iscomplexobj(embedded_col) or np.iscomplexobj(x):
        fft_mat = fft(embedded_col, axis=0, workers=workers).reshape(-1, 1)
        fft_x = fft(x, n=p, axis=0, workers=workers)

        mat_times_x = ifft(fft_mat*fft_x, axis=0,
                           workers=workers)[:T_nrows, :]
    else:
        # Real inputs; using rfft is faster
        fft_mat = rfft(embedded_col, axis=0, workers=workers).reshape(-1, 1)
        fft_x = rfft(x, n=p, axis=0, workers=workers)

        mat_times_x = irfft(fft_mat*fft_x, axis=0,
                            workers=workers, n=p)[:T_nrows, :]

    return_shape = (T_nrows,) if len(x_shape) == 1 else (T_nrows, m)
    return mat_times_x.reshape(*return_shape)
