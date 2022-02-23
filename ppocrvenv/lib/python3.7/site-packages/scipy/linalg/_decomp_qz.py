import warnings

import numpy as np
from numpy import asarray_chkfinite
from .misc import LinAlgError, _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs


__all__ = ['qz', 'ordqz']

_double_precision = ['i', 'l', 'd']


def _select_function(sort):
    if callable(sort):
        # assume the user knows what they're doing
        sfunction = sort
    elif sort == 'lhp':
        sfunction = _lhp
    elif sort == 'rhp':
        sfunction = _rhp
    elif sort == 'iuc':
        sfunction = _iuc
    elif sort == 'ouc':
        sfunction = _ouc
    else:
        raise ValueError("sort parameter must be None, a callable, or "
                         "one of ('lhp','rhp','iuc','ouc')")

    return sfunction


def _lhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) < 0.0)
    return out


def _rhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) > 0.0)
    return out


def _iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (abs(x[nonzero]/y[nonzero]) < 1.0)
    return out


def _ouc(x, y):
    out = np.empty_like(x, dtype=bool)
    xzero = (x == 0)
    yzero = (y == 0)
    out[xzero & yzero] = False
    out[~xzero & yzero] = True
    out[~yzero] = (abs(x[~yzero]/y[~yzero]) > 1.0)
    return out


def _qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
        overwrite_b=False, check_finite=True):
    if sort is not None:
        # Disabled due to segfaults on win32, see ticket 1717.
        raise ValueError("The 'sort' input of qz() has to be None and will be "
                         "removed in a future release. Use ordqz instead.")

    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if check_finite:
        a1 = asarray_chkfinite(A)
        b1 = asarray_chkfinite(B)
    else:
        a1 = np.asarray(A)
        b1 = np.asarray(B)

    a_m, a_n = a1.shape
    b_m, b_n = b1.shape
    if not (a_m == a_n == b_m == b_n):
        raise ValueError("Array dimensions must be square and agree")

    typa = a1.dtype.char
    if output in ['complex', 'c'] and typa not in ['F', 'D']:
        if typa in _double_precision:
            a1 = a1.astype('D')
            typa = 'D'
        else:
            a1 = a1.astype('F')
            typa = 'F'
    typb = b1.dtype.char
    if output in ['complex', 'c'] and typb not in ['F', 'D']:
        if typb in _double_precision:
            b1 = b1.astype('D')
            typb = 'D'
        else:
            b1 = b1.astype('F')
            typb = 'F'

    overwrite_a = overwrite_a or (_datacopied(a1, A))
    overwrite_b = overwrite_b or (_datacopied(b1, B))

    gges, = get_lapack_funcs(('gges',), (a1, b1))

    if lwork is None or lwork == -1:
        # get optimal work array size
        result = gges(lambda x: None, a1, b1, lwork=-1)
        lwork = result[-2][0].real.astype(np.int_)

    sfunction = lambda x: None
    result = gges(sfunction, a1, b1, lwork=lwork, overwrite_a=overwrite_a,
                  overwrite_b=overwrite_b, sort_t=0)

    info = result[-1]
    if info < 0:
        raise ValueError("Illegal value in argument {} of gges".format(-info))
    elif info > 0 and info <= a_n:
        warnings.warn("The QZ iteration failed. (a,b) are not in Schur "
                      "form, but ALPHAR(j), ALPHAI(j), and BETA(j) should be "
                      "correct for J={},...,N".format(info-1), LinAlgWarning,
                      stacklevel=3)
    elif info == a_n+1:
        raise LinAlgError("Something other than QZ iteration failed")
    elif info == a_n+2:
        raise LinAlgError("After reordering, roundoff changed values of some "
                          "complex eigenvalues so that leading eigenvalues "
                          "in the Generalized Schur form no longer satisfy "
                          "sort=True. This could also be due to scaling.")
    elif info == a_n+3:
        raise LinAlgError("Reordering failed in <s,d,c,z>tgsen")

    return result, gges.typecode


def qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
       overwrite_b=False, check_finite=True):
    """
    QZ decomposition for generalized eigenvalues of a pair of matrices.

    The QZ, or generalized Schur, decomposition for a pair of n-by-n
    matrices (A,B) is::

        (A,B) = (Q @ AA @ Z*, Q @ BB @ Z*)

    where AA, BB is in generalized Schur form if BB is upper-triangular
    with non-negative diagonal and AA is upper-triangular, or for real QZ
    decomposition (``output='real'``) block upper triangular with 1x1
    and 2x2 blocks. In this case, the 1x1 blocks correspond to real
    generalized eigenvalues and 2x2 blocks are 'standardized' by making
    the corresponding elements of BB have the form::

        [ a 0 ]
        [ 0 b ]

    and the pair of corresponding 2x2 blocks in AA and BB will have a complex
    conjugate pair of generalized eigenvalues. If (``output='complex'``) or
    A and B are complex matrices, Z' denotes the conjugate-transpose of Z.
    Q and Z are unitary matrices.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.

        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True). For
        real matrix pairs, the sort function takes three real arguments
        (alphar, alphai, beta). The eigenvalue
        ``x = (alphar + alphai*1j)/beta``. For complex matrix pairs or
        output='complex', the sort function takes two complex arguments
        (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,
        string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    Notes
    -----
    Q is transposed versus the equivalent function in Matlab.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> A = np.arange(9).reshape((3, 3))
    >>> B = rng.standard_normal((3, 3))

    >>> AA, BB, Q, Z = linalg.qz(A, B)
    >>> AA
    array([[ 8.99591445e+00, -1.07917902e+01, -2.18309912e+00],
           [ 0.00000000e+00, -8.60837546e-01,  1.05063006e+00],
           [ 0.00000000e+00,  0.00000000e+00, -1.40584278e-15]])
    >>> BB
    array([[ 0.2058989 , -0.6007898 , -0.5771396 ],
           [ 0.        ,  1.6997737 , -1.12160842],
           [ 0.        ,  0.        ,  1.76304656]])
    >>> Q
    array([[ 0.10356118,  0.90697763, -0.40824829],
           [ 0.48575095,  0.31205664,  0.81649658],
           [ 0.86794072, -0.28286434, -0.40824829]])
    >>> Z
    array([[ 0.78900531,  0.16010775, -0.59315776],
           [-0.21754047, -0.83009894, -0.51343148],
           [ 0.57458399, -0.53413598,  0.62012256]])

    See also
    --------
    ordqz
    """
    # output for real
    # AA, BB, sdim, alphar, alphai, beta, vsl, vsr, work, info
    # output for complex
    # AA, BB, sdim, alpha, beta, vsl, vsr, work, info
    result, _ = _qz(A, B, output=output, lwork=lwork, sort=sort,
                    overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                    check_finite=check_finite)
    return result[0], result[1], result[-4], result[-3]


def ordqz(A, B, sort='lhp', output='real', overwrite_a=False,
          overwrite_b=False, check_finite=True):
    """QZ decomposition for a pair of matrices with reordering.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A
        callable may be passed that, given an ordered pair ``(alpha,
        beta)`` representing the eigenvalue ``x = (alpha/beta)``,
        returns a boolean denoting whether the eigenvalue should be
        sorted to the top-left (True). For the real matrix pairs
        ``beta`` is real while ``alpha`` can be complex, and for
        complex matrix pairs both ``alpha`` and ``beta`` can be
        complex. The callable must be able to accept a NumPy
        array. Alternatively, string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        With the predefined sorting functions, an infinite eigenvalue
        (i.e., ``alpha != 0`` and ``beta = 0``) is considered to lie in
        neither the left-hand nor the right-hand plane, but it is
        considered to lie outside the unit circle. For the eigenvalue
        ``(alpha, beta) = (0, 0)``, the predefined sorting functions
        all return `False`.
    output : str {'real','complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    overwrite_a : bool, optional
        If True, the contents of A are overwritten.
    overwrite_b : bool, optional
        If True, the contents of B are overwritten.
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    alpha : (N,) ndarray
        alpha = alphar + alphai * 1j. See notes.
    beta : (N,) ndarray
        See notes.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    Notes
    -----
    On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
    generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
    ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
    that would result if the 2-by-2 diagonal blocks of the real generalized
    Schur form of (A,B) were further reduced to triangular form using complex
    unitary transformations. If ALPHAI(j) is zero, then the jth eigenvalue is
    real; if positive, then the ``j``th and ``(j+1)``st eigenvalues are a
    complex conjugate pair, with ``ALPHAI(j+1)`` negative.

    .. versionadded:: 0.17.0

    See also
    --------
    qz

    Examples
    --------
    >>> from scipy.linalg import ordqz
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> B = np.array([[0, 6, 0, 0], [5, 0, 2, 1], [5, 2, 6, 6], [4, 7, 7, 7]])
    >>> AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='lhp')

    Since we have sorted for left half plane eigenvalues, negatives come first

    >>> (alpha/beta).real < 0
    array([ True,  True, False, False], dtype=bool)

    """
    (AA, BB, _, *ab, Q, Z, _, _), typ = _qz(A, B, output=output, sort=None,
                                            overwrite_a=overwrite_a,
                                            overwrite_b=overwrite_b,
                                            check_finite=check_finite)

    if typ == 's':
        alpha, beta = ab[0] + ab[1]*np.complex64(1j), ab[2]
    elif typ == 'd':
        alpha, beta = ab[0] + ab[1]*1.j, ab[2]
    else:
        alpha, beta = ab

    sfunction = _select_function(sort)
    select = sfunction(alpha, beta)

    tgsen = get_lapack_funcs('tgsen', (AA, BB))
    # the real case needs 4n + 16 lwork
    lwork = 4*AA.shape[0] + 16 if typ in 'sd' else 1
    AAA, BBB, *ab, QQ, ZZ, _, _, _, _, info = tgsen(select, AA, BB, Q, Z,
                                                    ijob=0,
                                                    lwork=lwork, liwork=1)

    # Once more for tgsen output
    if typ == 's':
        alpha, beta = ab[0] + ab[1]*np.complex64(1j), ab[2]
    elif typ == 'd':
        alpha, beta = ab[0] + ab[1]*1.j, ab[2]
    else:
        alpha, beta = ab

    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of tgsen")
    elif info == 1:
        raise ValueError("Reordering of (A, B) failed because the transformed"
                         " matrix pair (A, B) would be too far from "
                         "generalized Schur form; the problem is very "
                         "ill-conditioned. (A, B) may have been partially "
                         "reordered.")

    return AAA, BBB, alpha, beta, QQ, ZZ
