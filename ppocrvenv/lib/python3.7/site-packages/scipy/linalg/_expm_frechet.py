"""Frechet derivative of the matrix exponential."""
import numpy as np
import scipy.linalg

__all__ = ['expm_frechet', 'expm_cond']


def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    """
    Frechet derivative of the matrix exponential of A in the direction E.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of which to take the matrix exponential.
    E : (N, N) array_like
        Matrix direction in which to take the Frechet derivative.
    method : str, optional
        Choice of algorithm. Should be one of

        - `SPS` (default)
        - `blockEnlarge`

    compute_expm : bool, optional
        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
        Default is True.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    expm_A : ndarray
        Matrix exponential of A.
    expm_frechet_AE : ndarray
        Frechet derivative of the matrix exponential of A in the direction E.

    For ``compute_expm = False``, only `expm_frechet_AE` is returned.

    See also
    --------
    expm : Compute the exponential of a matrix.

    Notes
    -----
    This section describes the available implementations that can be selected
    by the `method` parameter. The default method is *SPS*.

    Method *blockEnlarge* is a naive algorithm.

    Method *SPS* is Scaling-Pade-Squaring [1]_.
    It is a sophisticated implementation which should take
    only about 3/8 as much time as the naive implementation.
    The asymptotics are the same.

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           Computing the Frechet Derivative of the Matrix Exponential,
           with an application to Condition Number Estimation.
           SIAM Journal On Matrix Analysis and Applications.,
           30 (4). pp. 1639-1657. ISSN 1095-7162

    Examples
    --------
    >>> import scipy.linalg
    >>> rng = np.random.default_rng()
    >>> A = rng.standard_normal((3, 3))
    >>> E = rng.standard_normal((3, 3))
    >>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
    >>> expm_A.shape, expm_frechet_AE.shape
    ((3, 3), (3, 3))

    >>> import scipy.linalg
    >>> rng = np.random.default_rng()
    >>> A = rng.standard_normal((3, 3))
    >>> E = rng.standard_normal((3, 3))
    >>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
    >>> M = np.zeros((6, 6))
    >>> M[:3, :3] = A; M[:3, 3:] = E; M[3:, 3:] = A
    >>> expm_M = scipy.linalg.expm(M)
    >>> np.allclose(expm_A, expm_M[:3, :3])
    True
    >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
    True

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
        E = np.asarray_chkfinite(E)
    else:
        A = np.asarray(A)
        E = np.asarray(E)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    if method is None:
        method = 'SPS'
    if method == 'SPS':
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError('Unknown implementation %s' % method)
    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        return expm_frechet_AE


def expm_frechet_block_enlarge(A, E):
    """
    This is a helper function, mostly for testing and profiling.
    Return expm(A), frechet(A, E)
    """
    n = A.shape[0]
    M = np.vstack([
        np.hstack([A, E]),
        np.hstack([np.zeros_like(A), A])])
    expm_M = scipy.linalg.expm(M)
    return expm_M[:n, :n], expm_M[:n, n:]


"""
Maximal values ell_m of ||2**-s A|| such that the backward error bound
does not exceed 2**-53.
"""
ell_table_61 = (
        None,
        # 1
        2.11e-8,
        3.56e-4,
        1.08e-2,
        6.49e-2,
        2.00e-1,
        4.37e-1,
        7.83e-1,
        1.23e0,
        1.78e0,
        2.42e0,
        # 11
        3.13e0,
        3.90e0,
        4.74e0,
        5.63e0,
        6.56e0,
        7.52e0,
        8.53e0,
        9.56e0,
        1.06e1,
        1.17e1,
        )


# The b vectors and U and V are copypasted
# from scipy.sparse.linalg.matfuncs.py.
# M, Lu, Lv follow (6.11), (6.12), (6.13), (3.3)

def _diff_pade3(A, E, ident):
    b = (120., 60., 12., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    U = A.dot(b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
    Lv = b[2]*M2
    return U, V, Lu, Lv


def _diff_pade5(A, E, ident):
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[5]*M4 + b[3]*M2) +
            E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade7(A, E, ident):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade9(A, E, ident):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    A8 = np.dot(A4, A4)
    M8 = np.dot(A4, M4) + np.dot(M4, A4)
    U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def expm_frechet_algo_64(A, E):
    n = A.shape[0]
    s = None
    ident = np.identity(n)
    A_norm_1 = scipy.linalg.norm(A, 1)
    m_pade_pairs = (
            (3, _diff_pade3),
            (5, _diff_pade5),
            (7, _diff_pade7),
            (9, _diff_pade9))
    for m, pade in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break
    if s is None:
        # scaling
        s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
        A = A * 2.0**-s
        E = E * 2.0**-s
        # pade order 13
        A2 = np.dot(A, A)
        M2 = np.dot(A, E) + np.dot(E, A)
        A4 = np.dot(A2, A2)
        M4 = np.dot(A2, M2) + np.dot(M2, A2)
        A6 = np.dot(A2, A4)
        M6 = np.dot(A4, M2) + np.dot(M4, A2)
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
        W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident
        Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
        Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
        W = np.dot(A6, W1) + W2
        U = np.dot(A, W)
        V = np.dot(A6, Z1) + Z2
        Lw1 = b[13]*M6 + b[11]*M4 + b[9]*M2
        Lw2 = b[7]*M6 + b[5]*M4 + b[3]*M2
        Lz1 = b[12]*M6 + b[10]*M4 + b[8]*M2
        Lz2 = b[6]*M6 + b[4]*M4 + b[2]*M2
        Lw = np.dot(A6, Lw1) + np.dot(M6, W1) + Lw2
        Lu = np.dot(A, Lw) + np.dot(E, W)
        Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
    # factor once and solve twice
    lu_piv = scipy.linalg.lu_factor(-U + V)
    R = scipy.linalg.lu_solve(lu_piv, U + V)
    L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
    # squaring
    for k in range(s):
        L = np.dot(R, L) + np.dot(L, R)
        R = np.dot(R, R)
    return R, L


def vec(M):
    """
    Stack columns of M to construct a single vector.

    This is somewhat standard notation in linear algebra.

    Parameters
    ----------
    M : 2-D array_like
        Input matrix

    Returns
    -------
    v : 1-D ndarray
        Output vector

    """
    return M.T.ravel()


def expm_frechet_kronform(A, method=None, check_finite=True):
    """
    Construct the Kronecker form of the Frechet derivative of expm.

    Parameters
    ----------
    A : array_like with shape (N, N)
        Matrix to be expm'd.
    method : str, optional
        Extra keyword to be passed to expm_frechet.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    K : 2-D ndarray with shape (N*N, N*N)
        Kronecker form of the Frechet derivative of the matrix exponential.

    Notes
    -----
    This function is used to help compute the condition number
    of the matrix exponential.

    See also
    --------
    expm : Compute a matrix exponential.
    expm_frechet : Compute the Frechet derivative of the matrix exponential.
    expm_cond : Compute the relative condition number of the matrix exponential
                in the Frobenius norm.

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    n = A.shape[0]
    ident = np.identity(n)
    cols = []
    for i in range(n):
        for j in range(n):
            E = np.outer(ident[i], ident[j])
            F = expm_frechet(A, E,
                    method=method, compute_expm=False, check_finite=False)
            cols.append(vec(F))
    return np.vstack(cols).T


def expm_cond(A, check_finite=True):
    """
    Relative condition number of the matrix exponential in the Frobenius norm.

    Parameters
    ----------
    A : 2-D array_like
        Square input matrix with shape (N, N).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    kappa : float
        The relative condition number of the matrix exponential
        in the Frobenius norm

    Notes
    -----
    A faster estimate for the condition number in the 1-norm
    has been published but is not yet implemented in SciPy.

    .. versionadded:: 0.14.0

    See also
    --------
    expm : Compute the exponential of a matrix.
    expm_frechet : Compute the Frechet derivative of the matrix exponential.

    Examples
    --------
    >>> from scipy.linalg import expm_cond
    >>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])
    >>> k = expm_cond(A)
    >>> k
    1.7787805864469866

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    X = scipy.linalg.expm(A)
    K = expm_frechet_kronform(A, check_finite=False)

    # The following norm choices are deliberate.
    # The norms of A and X are Frobenius norms,
    # and the norm of K is the induced 2-norm.
    A_norm = scipy.linalg.norm(A, 'fro')
    X_norm = scipy.linalg.norm(X, 'fro')
    K_norm = scipy.linalg.norm(K, 2)

    kappa = (K_norm * A_norm) / X_norm
    return kappa
