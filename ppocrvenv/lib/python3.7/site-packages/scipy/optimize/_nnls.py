from . import __nnls
from numpy import asarray_chkfinite, zeros, double

__all__ = ['nnls']


def nnls(A, b, maxiter=None):
    """
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``. This is a wrapper
    for a FORTRAN non-negative least squares solver.

    Parameters
    ----------
    A : ndarray
        Matrix ``A`` as shown above.
    b : ndarray
        Right-hand side vector.
    maxiter: int, optional
        Maximum number of iterations, optional.
        Default is ``3 * A.shape[1]``.

    Returns
    -------
    x : ndarray
        Solution vector.
    rnorm : float
        The residual, ``|| Ax-b ||_2``.

    See Also
    --------
    lsq_linear : Linear least squares with bounds on the variables

    Notes
    -----
    The FORTRAN code was published in the book below. The algorithm
    is an active set method. It solves the KKT (Karush-Kuhn-Tucker)
    conditions for the non-negative least squares problem.

    References
    ----------
    Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM

     Examples
    --------
    >>> from scipy.optimize import nnls
    ...
    >>> A = np.array([[1, 0], [1, 0], [0, 1]])
    >>> b = np.array([2, 1, 1])
    >>> nnls(A, b)
    (array([1.5, 1. ]), 0.7071067811865475)

    >>> b = np.array([-1, -1, -1])
    >>> nnls(A, b)
    (array([0., 0.]), 1.7320508075688772)

    """

    A, b = map(asarray_chkfinite, (A, b))

    if len(A.shape) != 2:
        raise ValueError("Expected a two-dimensional array (matrix)" +
                         ", but the shape of A is %s" % (A.shape, ))
    if len(b.shape) != 1:
        raise ValueError("Expected a one-dimensional array (vector" +
                         ", but the shape of b is %s" % (b.shape, ))

    m, n = A.shape

    if m != b.shape[0]:
        raise ValueError(
                "Incompatible dimensions. The first dimension of " +
                "A is %s, while the shape of b is %s" % (m, (b.shape[0], )))

    maxiter = -1 if maxiter is None else int(maxiter)

    w = zeros((n,), dtype=double)
    zz = zeros((m,), dtype=double)
    index = zeros((n,), dtype=int)

    x, rnorm, mode = __nnls.nnls(A, m, n, b, w, zz, index, maxiter)
    if mode != 1:
        raise RuntimeError("too many iterations")

    return x, rnorm
