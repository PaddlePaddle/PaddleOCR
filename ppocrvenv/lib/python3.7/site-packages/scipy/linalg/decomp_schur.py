"""Schur decomposition functions."""
import numpy
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm


# Local imports.
from .misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from .decomp import eigvals

__all__ = ['schur', 'rsf2csf']

_double_precision = ['i', 'l', 'd']


def schur(a, output='real', lwork=None, overwrite_a=False, sort=None,
          check_finite=True):
    """
    Compute Schur decomposition of a matrix.

    The Schur decomposition is::

        A = Z T Z^H

    where Z is unitary and T is either upper-triangular, or for real
    Schur decomposition (output='real'), quasi-upper triangular. In
    the quasi-triangular form, 2x2 blocks describing complex-valued
    eigenvalue pairs may extrude from the diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex Schur decomposition (for real matrices).
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance).
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True).
        Alternatively, string parameters may be used::

            'lhp'   Left-hand plane (x.real < 0.0)
            'rhp'   Right-hand plane (x.real > 0.0)
            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Schur form of A. It is real-valued for the real Schur decomposition.
    Z : (M, M) ndarray
        An unitary Schur transformation matrix for A.
        It is real-valued for the real Schur decomposition.
    sdim : int
        If and only if sorting was requested, a third return value will
        contain the number of eigenvalues satisfying the sort condition.

    Raises
    ------
    LinAlgError
        Error raised under three conditions:

        1. The algorithm failed due to a failure of the QR algorithm to
           compute all eigenvalues.
        2. If eigenvalue sorting was requested, the eigenvalues could not be
           reordered due to a failure to separate eigenvalues, usually because
           of poor conditioning.
        3. If eigenvalue sorting was requested, roundoff errors caused the
           leading eigenvalues to no longer satisfy the sorting condition.

    See also
    --------
    rsf2csf : Convert real Schur form to complex Schur form

    Examples
    --------
    >>> from scipy.linalg import schur, eigvals
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])

    >>> T2, Z2 = schur(A, output='complex')
    >>> T2
    array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j],
           [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
           [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
    >>> eigvals(T2)
    array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])

    An arbitrary custom eig-sorting condition, having positive imaginary part,
    which is satisfied by only one eigenvalue

    >>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)
    >>> sdim
    1

    """
    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    typ = a1.dtype.char
    if output in ['complex', 'c'] and typ not in ['F', 'D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
            typ = 'D'
        else:
            a1 = a1.astype('F')
            typ = 'F'
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    gees, = get_lapack_funcs(('gees',), (a1,))
    if lwork is None or lwork == -1:
        # get optimal work array
        result = gees(lambda x: None, a1, lwork=-1)
        lwork = result[-2][0].real.astype(numpy.int_)

    if sort is None:
        sort_t = 0
        sfunction = lambda x: None
    else:
        sort_t = 1
        if callable(sort):
            sfunction = sort
        elif sort == 'lhp':
            sfunction = lambda x: (x.real < 0.0)
        elif sort == 'rhp':
            sfunction = lambda x: (x.real >= 0.0)
        elif sort == 'iuc':
            sfunction = lambda x: (abs(x) <= 1.0)
        elif sort == 'ouc':
            sfunction = lambda x: (abs(x) > 1.0)
        else:
            raise ValueError("'sort' parameter must either be 'None', or a "
                             "callable, or one of ('lhp','rhp','iuc','ouc')")

    result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a,
                  sort_t=sort_t)

    info = result[-1]
    if info < 0:
        raise ValueError('illegal value in {}-th argument of internal gees'
                         ''.format(-info))
    elif info == a1.shape[0] + 1:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise LinAlgError("Schur form not found. Possibly ill-conditioned.")

    if sort_t == 0:
        return result[0], result[-3]
    else:
        return result[0], result[-3], result[1]


eps = numpy.finfo(float).eps
feps = numpy.finfo(single).eps

_array_kind = {'b': 0, 'h': 0, 'B': 0, 'i': 0, 'l': 0,
               'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]


def _commonType(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]


def _castCopy(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.char == type:
            cast_arrays = cast_arrays + (a.copy(),)
        else:
            cast_arrays = cast_arrays + (a.astype(type),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays


def rsf2csf(T, Z, check_finite=True):
    """
    Convert real Schur form to complex Schur form.

    Convert a quasi-diagonal real-valued Schur form to the upper-triangular
    complex-valued Schur form.

    Parameters
    ----------
    T : (M, M) array_like
        Real Schur form of the original array
    Z : (M, M) array_like
        Schur transformation matrix
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Complex Schur form of the original array
    Z : (M, M) ndarray
        Schur transformation matrix corresponding to the complex form

    See Also
    --------
    schur : Schur decomposition of an array

    Examples
    --------
    >>> from scipy.linalg import schur, rsf2csf
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])
    >>> T2 , Z2 = rsf2csf(T, Z)
    >>> T2
    array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
           [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
           [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
    >>> Z2
    array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
           [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
           [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])

    """
    if check_finite:
        Z, T = map(asarray_chkfinite, (Z, T))
    else:
        Z, T = map(asarray, (Z, T))

    for ind, X in enumerate([Z, T]):
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Input '{}' must be square.".format('ZT'[ind]))

    if T.shape[0] != Z.shape[0]:
        raise ValueError("Input array shapes must match: Z: {} vs. T: {}"
                         "".format(Z.shape, T.shape))
    N = T.shape[0]
    t = _commonType(Z, T, array([3.0], 'F'))
    Z, T = _castCopy(t, Z, T)

    for m in range(N-1, 0, -1):
        if abs(T[m, m-1]) > eps*(abs(T[m-1, m-1]) + abs(T[m, m])):
            mu = eigvals(T[m-1:m+1, m-1:m+1]) - T[m, m]
            r = norm([mu[0], T[m, m-1]])
            c = mu[0] / r
            s = T[m, m-1] / r
            G = array([[c.conj(), s], [-s, c]], dtype=t)

            T[m-1:m+1, m-1:] = G.dot(T[m-1:m+1, m-1:])
            T[:m+1, m-1:m+1] = T[:m+1, m-1:m+1].dot(G.conj().T)
            Z[:, m-1:m+1] = Z[:, m-1:m+1].dot(G.conj().T)

        T[m, m-1] = 0.0
    return T, Z
