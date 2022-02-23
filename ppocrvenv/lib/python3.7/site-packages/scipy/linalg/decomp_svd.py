"""SVD decomposition functions."""
import numpy
from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip

# Local imports.
from .misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs, _compute_lwork
from .decomp import _asarray_validated

__all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space']


def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and
    a 1-D array ``s`` of singular values (real, non-negative) such that
    ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with
    main diagonal ``s``.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    full_matrices : bool, optional
        If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.
        If False, the shapes are ``(M, K)`` and ``(K, N)``, where
        ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether to compute also ``U`` and ``Vh`` in addition to ``s``.
        Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : {'gesdd', 'gesvd'}, optional
        Whether to use the more efficient divide-and-conquer approach
        (``'gesdd'``) or general rectangular approach (``'gesvd'``)
        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
        Default is ``'gesdd'``.

        .. versionadded:: 0.18

    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
    s : ndarray
        The singular values, sorted in non-increasing order.
        Of shape (K,), with ``K = min(M, N)``.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.
        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.

    For ``compute_uv=False``, only ``s`` is returned.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See also
    --------
    svdvals : Compute singular values of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> from scipy import linalg
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> m, n = 9, 6
    >>> a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))
    >>> U, s, Vh = linalg.svd(a)
    >>> U.shape,  s.shape, Vh.shape
    ((9, 9), (6,), (6, 6))

    Reconstruct the original matrix from the decomposition:

    >>> sigma = np.zeros((m, n))
    >>> for i in range(min(m, n)):
    ...     sigma[i, i] = s[i]
    >>> a1 = np.dot(U, np.dot(sigma, Vh))
    >>> np.allclose(a, a1)
    True

    Alternatively, use ``full_matrices=False`` (notice that the shape of
    ``U`` is then ``(m, n)`` instead of ``(m, m)``):

    >>> U, s, Vh = linalg.svd(a, full_matrices=False)
    >>> U.shape, s.shape, Vh.shape
    ((9, 6), (6,), (6, 6))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = linalg.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    m, n = a1.shape
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be a string')
    if lapack_driver not in ('gesdd', 'gesvd'):
        raise ValueError('lapack_driver must be "gesdd" or "gesvd", not "%s"'
                         % (lapack_driver,))
    funcs = (lapack_driver, lapack_driver + '_lwork')
    gesXd, gesXd_lwork = get_lapack_funcs(funcs, (a1,), ilp64='preferred')

    # compute optimal lwork
    lwork = _compute_lwork(gesXd_lwork, a1.shape[0], a1.shape[1],
                           compute_uv=compute_uv, full_matrices=full_matrices)

    # perform decomposition
    u, s, v, info = gesXd(a1, compute_uv=compute_uv, lwork=lwork,
                          full_matrices=full_matrices, overwrite_a=overwrite_a)

    if info > 0:
        raise LinAlgError("SVD did not converge")
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal gesdd'
                         % -info)
    if compute_uv:
        return u, s, v
    else:
        return s


def svdvals(a, overwrite_a=False, check_finite=True):
    """
    Compute singular values of a matrix.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    s : (min(M, N),) ndarray
        The singular values, sorted in decreasing order.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    ``svdvals(a)`` only differs from ``svd(a, compute_uv=False)`` by its
    handling of the edge case of empty ``a``, where it returns an
    empty sequence:

    >>> a = np.empty((0, 2))
    >>> from scipy.linalg import svdvals
    >>> svdvals(a)
    array([], dtype=float64)

    See Also
    --------
    svd : Compute the full singular value decomposition of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> from scipy.linalg import svdvals
    >>> m = np.array([[1.0, 0.0],
    ...               [2.0, 3.0],
    ...               [1.0, 1.0],
    ...               [0.0, 2.0],
    ...               [1.0, 0.0]])
    >>> svdvals(m)
    array([ 4.28091555,  1.63516424])

    We can verify the maximum singular value of `m` by computing the maximum
    length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.
    We approximate "all" the unit vectors with a large sample. Because
    of linearity, we only need the unit vectors with angles in [0, pi].

    >>> t = np.linspace(0, np.pi, 2000)
    >>> u = np.array([np.cos(t), np.sin(t)])
    >>> np.linalg.norm(m.dot(u), axis=0).max()
    4.2809152422538475

    `p` is a projection matrix with rank 1. With exact arithmetic,
    its singular values would be [1, 0, 0, 0].

    >>> v = np.array([0.1, 0.3, 0.9, 0.3])
    >>> p = np.outer(v, v)
    >>> svdvals(p)
    array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,
             8.15115104e-34])

    The singular values of an orthogonal matrix are all 1. Here, we
    create a random orthogonal matrix by using the `rvs()` method of
    `scipy.stats.ortho_group`.

    >>> from scipy.stats import ortho_group
    >>> orth = ortho_group.rvs(4)
    >>> svdvals(orth)
    array([ 1.,  1.,  1.,  1.])

    """
    a = _asarray_validated(a, check_finite=check_finite)
    if a.size:
        return svd(a, compute_uv=0, overwrite_a=overwrite_a,
                   check_finite=False)
    elif len(a.shape) != 2:
        raise ValueError('expected matrix')
    else:
        return numpy.empty(0)


def diagsvd(s, M, N):
    """
    Construct the sigma matrix in SVD from singular values and size M, N.

    Parameters
    ----------
    s : (M,) or (N,) array_like
        Singular values
    M : int
        Size of the matrix whose singular values are `s`.
    N : int
        Size of the matrix whose singular values are `s`.

    Returns
    -------
    S : (M, N) ndarray
        The S-matrix in the singular value decomposition

    See Also
    --------
    svd : Singular value decomposition of a matrix
    svdvals : Compute singular values of a matrix.

    Examples
    --------
    >>> from scipy.linalg import diagsvd
    >>> vals = np.array([1, 2, 3])  # The array representing the computed svd
    >>> diagsvd(vals, 3, 4)
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0]])
    >>> diagsvd(vals, 4, 3)
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3],
           [0, 0, 0]])

    """
    part = diag(s)
    typ = part.dtype.char
    MorN = len(s)
    if MorN == M:
        return r_['-1', part, zeros((M, N-M), typ)]
    elif MorN == N:
        return r_[part, zeros((M-N, N), typ)]
    else:
        raise ValueError("Length of s must be M or N.")


# Orthonormal decomposition

def orth(A, rcond=None):
    """
    Construct an orthonormal basis for the range of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Q : (M, K) ndarray
        Orthonormal basis for the range of A.
        K = effective rank of A, as determined by rcond

    See also
    --------
    svd : Singular value decomposition of a matrix
    null_space : Matrix null space

    Examples
    --------
    >>> from scipy.linalg import orth
    >>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
    >>> orth(A)
    array([[0., 1.],
           [1., 0.]])
    >>> orth(A.T)
    array([[0., 1.],
           [1., 0.],
           [0., 0.]])

    """
    u, s, vh = svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = numpy.finfo(s.dtype).eps * max(M, N)
    tol = numpy.amax(s) * rcond
    num = numpy.sum(s > tol, dtype=int)
    Q = u[:, :num]
    return Q


def null_space(A, rcond=None):
    """
    Construct an orthonormal basis for the null space of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Z : (N, K) ndarray
        Orthonormal basis for the null space of A.
        K = dimension of effective null space, as determined by rcond

    See also
    --------
    svd : Singular value decomposition of a matrix
    orth : Matrix range

    Examples
    --------
    1-D null space:

    >>> from scipy.linalg import null_space
    >>> A = np.array([[1, 1], [1, 1]])
    >>> ns = null_space(A)
    >>> ns * np.sign(ns[0,0])  # Remove the sign ambiguity of the vector
    array([[ 0.70710678],
           [-0.70710678]])

    2-D null space:

    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> B = rng.random((3, 5))
    >>> Z = null_space(B)
    >>> Z.shape
    (5, 2)
    >>> np.allclose(B.dot(Z), 0)
    True

    The basis vectors are orthonormal (up to rounding error):

    >>> Z.T.dot(Z)
    array([[  1.00000000e+00,   6.92087741e-17],
           [  6.92087741e-17,   1.00000000e+00]])

    """
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = numpy.finfo(s.dtype).eps * max(M, N)
    tol = numpy.amax(s) * rcond
    num = numpy.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def subspace_angles(A, B):
    r"""
    Compute the subspace angles between two matrices.

    Parameters
    ----------
    A : (M, N) array_like
        The first input array.
    B : (M, K) array_like
        The second input array.

    Returns
    -------
    angles : ndarray, shape (min(N, K),)
        The subspace angles between the column spaces of `A` and `B` in
        descending order.

    See Also
    --------
    orth
    svd

    Notes
    -----
    This computes the subspace angles according to the formula
    provided in [1]_. For equivalence with MATLAB and Octave behavior,
    use ``angles[0]``.

    .. versionadded:: 1.0

    References
    ----------
    .. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces
           in an A-Based Scalar Product: Algorithms and Perturbation
           Estimates. SIAM J. Sci. Comput. 23:2008-2040.

    Examples
    --------
    An Hadamard matrix, which has orthogonal columns, so we expect that
    the suspace angle to be :math:`\frac{\pi}{2}`:

    >>> from numpy.random import default_rng
    >>> from scipy.linalg import hadamard, subspace_angles
    >>> rng = default_rng()
    >>> H = hadamard(4)
    >>> print(H)
    [[ 1  1  1  1]
     [ 1 -1  1 -1]
     [ 1  1 -1 -1]
     [ 1 -1 -1  1]]
    >>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))
    array([ 90.,  90.])

    And the subspace angle of a matrix to itself should be zero:

    >>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps
    array([ True,  True], dtype=bool)

    The angles between non-orthogonal subspaces are in between these extremes:

    >>> x = rng.standard_normal((4, 3))
    >>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))
    array([ 55.832])  # random
    """
    # Steps here omit the U and V calculation steps from the paper

    # 1. Compute orthonormal bases of column-spaces
    A = _asarray_validated(A, check_finite=True)
    if len(A.shape) != 2:
        raise ValueError('expected 2D array, got shape %s' % (A.shape,))
    QA = orth(A)
    del A

    B = _asarray_validated(B, check_finite=True)
    if len(B.shape) != 2:
        raise ValueError('expected 2D array, got shape %s' % (B.shape,))
    if len(B) != len(QA):
        raise ValueError('A and B must have the same number of rows, got '
                         '%s and %s' % (QA.shape[0], B.shape[0]))
    QB = orth(B)
    del B

    # 2. Compute SVD for cosine
    QA_H_QB = dot(QA.T.conj(), QB)
    sigma = svdvals(QA_H_QB)

    # 3. Compute matrix B
    if QA.shape[1] >= QB.shape[1]:
        B = QB - dot(QA, QA_H_QB)
    else:
        B = QA - dot(QB, QA_H_QB.T.conj())
    del QA, QB, QA_H_QB

    # 4. Compute SVD for sine
    mask = sigma ** 2 >= 0.5
    if mask.any():
        mu_arcsin = arcsin(clip(svdvals(B, overwrite_a=True), -1., 1.))
    else:
        mu_arcsin = 0.

    # 5. Compute the principal angles
    # with reverse ordering of sigma because smallest sigma belongs to largest
    # angle theta
    theta = where(mask, mu_arcsin, arccos(clip(sigma[::-1], -1., 1.)))
    return theta
