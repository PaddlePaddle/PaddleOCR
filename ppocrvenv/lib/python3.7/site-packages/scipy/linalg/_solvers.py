"""Matrix equation solver routines"""
# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# February 24, 2012

# Modified: Chad Fulton <ChadFulton@gmail.com>
# June 19, 2014

# Modified: Ilhan Polat <ilhanpolat@gmail.com>
# September 13, 2016

import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd

from .basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from .decomp_schur import schur
from .decomp_lu import lu
from .decomp_qr import qr
from ._decomp_qz import ordqz
from .decomp import _asarray_validated
from .special_matrices import kron, block_diag

__all__ = ['solve_sylvester',
           'solve_continuous_lyapunov', 'solve_discrete_lyapunov',
           'solve_lyapunov',
           'solve_continuous_are', 'solve_discrete_are']


def solve_sylvester(a, b, q):
    """
    Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.

    Parameters
    ----------
    a : (M, M) array_like
        Leading matrix of the Sylvester equation
    b : (N, N) array_like
        Trailing matrix of the Sylvester equation
    q : (M, N) array_like
        Right-hand side

    Returns
    -------
    x : (M, N) ndarray
        The solution to the Sylvester equation.

    Raises
    ------
    LinAlgError
        If solution was not found

    Notes
    -----
    Computes a solution to the Sylvester matrix equation via the Bartels-
    Stewart algorithm. The A and B matrices first undergo Schur
    decompositions. The resulting matrices are used to construct an
    alternative Sylvester equation (``RY + YS^T = F``) where the R and S
    matrices are in quasi-triangular form (or, when R, S or F are complex,
    triangular form). The simplified equation is then solved using
    ``*TRSYL`` from LAPACK directly.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a`, `b`, and `q` solve for `x`:

    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
    >>> b = np.array([[1]])
    >>> q = np.array([[1],[2],[3]])
    >>> x = linalg.solve_sylvester(a, b, q)
    >>> x
    array([[ 0.0625],
           [-0.5625],
           [ 0.6875]])
    >>> np.allclose(a.dot(x) + x.dot(b), q)
    True

    """

    # Compute the Schur decomposition form of a
    r, u = schur(a, output='real')

    # Compute the Schur decomposition of b
    s, v = schur(b.conj().transpose(), output='real')

    # Construct f = u'*q*v
    f = np.dot(np.dot(u.conj().transpose(), q), v)

    # Call the Sylvester equation solver
    trsyl, = get_lapack_funcs(('trsyl',), (r, s, f))
    if trsyl is None:
        raise RuntimeError('LAPACK implementation does not contain a proper '
                           'Sylvester equation solver (TRSYL)')
    y, scale, info = trsyl(r, s, f, tranb='C')

    y = scale*y

    if info < 0:
        raise LinAlgError("Illegal value encountered in "
                          "the %d term" % (-info,))

    return np.dot(np.dot(u, y), v.conj().transpose())


def solve_continuous_lyapunov(a, q):
    """
    Solves the continuous Lyapunov equation :math:`AX + XA^H = Q`.

    Uses the Bartels-Stewart algorithm to find :math:`X`.

    Parameters
    ----------
    a : array_like
        A square matrix

    q : array_like
        Right-hand side square matrix

    Returns
    -------
    x : ndarray
        Solution to the continuous Lyapunov equation

    See Also
    --------
    solve_discrete_lyapunov : computes the solution to the discrete-time
        Lyapunov equation
    solve_sylvester : computes the solution to the Sylvester equation

    Notes
    -----
    The continuous Lyapunov equation is a special form of the Sylvester
    equation, hence this solver relies on LAPACK routine ?TRSYL.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 0], [0, -5, -1]])
    >>> b = np.array([2, 4, -1])
    >>> q = np.eye(3)
    >>> x = linalg.solve_continuous_lyapunov(a, q)
    >>> x
    array([[ -0.75  ,   0.875 ,  -3.75  ],
           [  0.875 ,  -1.375 ,   5.3125],
           [ -3.75  ,   5.3125, -27.0625]])
    >>> np.allclose(a.dot(x) + x.dot(a.T), q)
    True
    """

    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))

    r_or_c = float

    for ind, _ in enumerate((a, q)):
        if np.iscomplexobj(_):
            r_or_c = complex

        if not np.equal(*_.shape):
            raise ValueError("Matrix {} should be square.".format("aq"[ind]))

    # Shape consistency check
    if a.shape != q.shape:
        raise ValueError("Matrix a and q should have the same shape.")

    # Compute the Schur decomposition form of a
    r, u = schur(a, output='real')

    # Construct f = u'*q*u
    f = u.conj().T.dot(q.dot(u))

    # Call the Sylvester equation solver
    trsyl = get_lapack_funcs('trsyl', (r, f))

    dtype_string = 'T' if r_or_c == float else 'C'
    y, scale, info = trsyl(r, r, f, tranb=dtype_string)

    if info < 0:
        raise ValueError('?TRSYL exited with the internal error '
                         '"illegal value in argument number {}.". See '
                         'LAPACK documentation for the ?TRSYL error codes.'
                         ''.format(-info))
    elif info == 1:
        warnings.warn('Input "a" has an eigenvalue pair whose sum is '
                      'very close to or exactly zero. The solution is '
                      'obtained via perturbing the coefficients.',
                      RuntimeWarning)
    y *= scale

    return u.dot(y).dot(u.conj().T)


# For backwards compatibility, keep the old name
solve_lyapunov = solve_continuous_lyapunov


def _solve_discrete_lyapunov_direct(a, q):
    """
    Solves the discrete Lyapunov equation directly.

    This function is called by the `solve_discrete_lyapunov` function with
    `method=direct`. It is not supposed to be called directly.
    """

    lhs = kron(a, a.conj())
    lhs = np.eye(lhs.shape[0]) - lhs
    x = solve(lhs, q.flatten())

    return np.reshape(x, q.shape)


def _solve_discrete_lyapunov_bilinear(a, q):
    """
    Solves the discrete Lyapunov equation using a bilinear transformation.

    This function is called by the `solve_discrete_lyapunov` function with
    `method=bilinear`. It is not supposed to be called directly.
    """
    eye = np.eye(a.shape[0])
    aH = a.conj().transpose()
    aHI_inv = inv(aH + eye)
    b = np.dot(aH - eye, aHI_inv)
    c = 2*np.dot(np.dot(inv(a + eye), q), aHI_inv)
    return solve_lyapunov(b.conj().transpose(), -c)


def solve_discrete_lyapunov(a, q, method=None):
    """
    Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.

    Parameters
    ----------
    a, q : (M, M) array_like
        Square matrices corresponding to A and Q in the equation
        above respectively. Must have the same shape.

    method : {'direct', 'bilinear'}, optional
        Type of solver.

        If not given, chosen to be ``direct`` if ``M`` is less than 10 and
        ``bilinear`` otherwise.

    Returns
    -------
    x : ndarray
        Solution to the discrete Lyapunov equation

    See Also
    --------
    solve_continuous_lyapunov : computes the solution to the continuous-time
        Lyapunov equation

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *direct* if ``M`` is less than 10
    and ``bilinear`` otherwise.

    Method *direct* uses a direct analytical solution to the discrete Lyapunov
    equation. The algorithm is given in, for example, [1]_. However, it requires
    the linear solution of a system with dimension :math:`M^2` so that
    performance degrades rapidly for even moderately sized matrices.

    Method *bilinear* uses a bilinear transformation to convert the discrete
    Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`
    where :math:`B=(A-I)(A+I)^{-1}` and
    :math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be
    efficiently solved since it is a special case of a Sylvester equation.
    The transformation algorithm is from Popov (1964) as described in [2]_.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Hamilton, James D. Time Series Analysis, Princeton: Princeton
       University Press, 1994.  265.  Print.
       http://doc1.lbfl.li/aca/FLMF037168.pdf
    .. [2] Gajic, Z., and M.T.J. Qureshi. 2008.
       Lyapunov Matrix Equation in System Stability and Control.
       Dover Books on Engineering Series. Dover Publications.

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> from scipy import linalg
    >>> a = np.array([[0.2, 0.5],[0.7, -0.9]])
    >>> q = np.eye(2)
    >>> x = linalg.solve_discrete_lyapunov(a, q)
    >>> x
    array([[ 0.70872893,  1.43518822],
           [ 1.43518822, -2.4266315 ]])
    >>> np.allclose(a.dot(x).dot(a.T)-x, -q)
    True

    """
    a = np.asarray(a)
    q = np.asarray(q)
    if method is None:
        # Select automatically based on size of matrices
        if a.shape[0] >= 10:
            method = 'bilinear'
        else:
            method = 'direct'

    meth = method.lower()

    if meth == 'direct':
        x = _solve_discrete_lyapunov_direct(a, q)
    elif meth == 'bilinear':
        x = _solve_discrete_lyapunov_bilinear(a, q)
    else:
        raise ValueError('Unknown solver %s' % method)

    return x


def solve_continuous_are(a, b, q, r, e=None, s=None, balanced=True):
    r"""
    Solves the continuous-time algebraic Riccati equation (CARE).

    The CARE is defined as

    .. math::

          X A + A^H X - X B R^{-1} B^H X + Q = 0

    The limitations for a solution to exist are :

        * All eigenvalues of :math:`A` on the right half plane, should be
          controllable.

        * The associated hamiltonian pencil (See Notes), should have
          eigenvalues sufficiently away from the imaginary axis.

    Moreover, if ``e`` or ``s`` is not precisely ``None``, then the
    generalized version of CARE

    .. math::

          E^HXA + A^HXE - (E^HXB + S) R^{-1} (B^HXE + S^H) + Q = 0

    is solved. When omitted, ``e`` is assumed to be the identity and ``s``
    is assumed to be the zero matrix with sizes compatible with ``a`` and
    ``b``, respectively.

    Parameters
    ----------
    a : (M, M) array_like
        Square matrix
    b : (M, N) array_like
        Input
    q : (M, M) array_like
        Input
    r : (N, N) array_like
        Nonsingular square matrix
    e : (M, M) array_like, optional
        Nonsingular square matrix
    s : (M, N) array_like, optional
        Input
    balanced : bool, optional
        The boolean that indicates whether a balancing step is performed
        on the data. The default is set to True.

    Returns
    -------
    x : (M, M) ndarray
        Solution to the continuous-time algebraic Riccati equation.

    Raises
    ------
    LinAlgError
        For cases where the stable subspace of the pencil could not be
        isolated. See Notes section and the references for details.

    See Also
    --------
    solve_discrete_are : Solves the discrete-time algebraic Riccati equation

    Notes
    -----
    The equation is solved by forming the extended hamiltonian matrix pencil,
    as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

        [ A    0    B ]             [ E   0    0 ]
        [-Q  -A^H  -S ] - \lambda * [ 0  E^H   0 ]
        [ S^H B^H   R ]             [ 0   0    0 ]

    and using a QZ decomposition method.

    In this algorithm, the fail conditions are linked to the symmetry
    of the product :math:`U_2 U_1^{-1}` and condition number of
    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
    eigenvectors spanning the stable subspace with 2-m rows and partitioned
    into two m-row matrices. See [1]_ and [2]_ for more details.

    In order to improve the QZ decomposition accuracy, the pencil goes
    through a balancing step where the sum of absolute values of
    :math:`H` and :math:`J` entries (after removing the diagonal entries of
    the sum) is balanced following the recipe given in [3]_.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
       Riccati Equations.", SIAM Journal on Scientific and Statistical
       Computing, Vol.2(2), :doi:`10.1137/0902010`

    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
       Equations.", Massachusetts Institute of Technology. Laboratory for
       Information and Decision Systems. LIDS-R ; 859. Available online :
       http://hdl.handle.net/1721.1/1301

    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`

    Examples
    --------
    Given `a`, `b`, `q`, and `r` solve for `x`:

    >>> from scipy import linalg
    >>> a = np.array([[4, 3], [-4.5, -3.5]])
    >>> b = np.array([[1], [-1]])
    >>> q = np.array([[9, 6], [6, 4.]])
    >>> r = 1
    >>> x = linalg.solve_continuous_are(a, b, q, r)
    >>> x
    array([[ 21.72792206,  14.48528137],
           [ 14.48528137,   9.65685425]])
    >>> np.allclose(a.T.dot(x) + x.dot(a)-x.dot(b).dot(b.T).dot(x), -q)
    True

    """

    # Validate input arguments
    a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
                                                     a, b, q, r, e, s, 'care')

    H = np.empty((2*m+n, 2*m+n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, m:2*m] = 0.
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = -a.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, m:2*m] = b.conj().T
    H[2*m:, 2*m:] = r

    if gen_are and e is not None:
        J = block_diag(e, e.conj().T, np.zeros_like(r, dtype=r_or_c))
    else:
        J = block_diag(np.eye(2*m), np.zeros_like(r, dtype=r_or_c))

    if balanced:
        # xGEBAL does not remove the diagonals before scaling. Also
        # to avoid destroying the Symplectic structure, we follow Ref.3
        M = np.abs(H) + np.abs(J)
        M[np.diag_indices_from(M)] = 0.
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        # do we need to bother?
        if not np.allclose(sca, np.ones_like(sca)):
            # Now impose diag(D,inv(D)) from Benner where D is
            # square root of s_i/s_(n+i) for i=0,....
            sca = np.log2(sca)
            # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
            s = np.round((sca[m:2*m] - sca[:m])/2)
            sca = 2 ** np.r_[s, -s, sca[2*m:]]
            # Elementwise multiplication via broadcasting.
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale

    # Deflate the pencil to 2m x 2m ala Ref.1, eq.(55)
    q, r = qr(H[:, -n:])
    H = q[:, n:].conj().T.dot(H[:, :2*m])
    J = q[:2*m, n:].conj().T.dot(J[:2*m, :2*m])

    # Decide on which output type is needed for QZ
    out_str = 'real' if r_or_c == float else 'complex'

    _, _, _, _, _, u = ordqz(H, J, sort='lhp', overwrite_a=True,
                             overwrite_b=True, check_finite=False,
                             output=out_str)

    # Get the relevant parts of the stable subspace basis
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # Solve via back-substituion after checking the condition of u00
    up, ul, uu = lu(u00)
    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    # Exploit the triangular structure
    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # Check the deviation from symmetry for lack of success
    # See proof of Thm.5 item 3 in [2]
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])

    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated Hamiltonian pencil has eigenvalues '
                          'too close to the imaginary axis')

    return (x + x.conj().T)/2


def solve_discrete_are(a, b, q, r, e=None, s=None, balanced=True):
    r"""
    Solves the discrete-time algebraic Riccati equation (DARE).

    The DARE is defined as

    .. math::

          A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0

    The limitations for a solution to exist are :

        * All eigenvalues of :math:`A` outside the unit disc, should be
          controllable.

        * The associated symplectic pencil (See Notes), should have
          eigenvalues sufficiently away from the unit circle.

    Moreover, if ``e`` and ``s`` are not both precisely ``None``, then the
    generalized version of DARE

    .. math::

          A^HXA - E^HXE - (A^HXB+S) (R+B^HXB)^{-1} (B^HXA+S^H) + Q = 0

    is solved. When omitted, ``e`` is assumed to be the identity and ``s``
    is assumed to be the zero matrix.

    Parameters
    ----------
    a : (M, M) array_like
        Square matrix
    b : (M, N) array_like
        Input
    q : (M, M) array_like
        Input
    r : (N, N) array_like
        Square matrix
    e : (M, M) array_like, optional
        Nonsingular square matrix
    s : (M, N) array_like, optional
        Input
    balanced : bool
        The boolean that indicates whether a balancing step is performed
        on the data. The default is set to True.

    Returns
    -------
    x : (M, M) ndarray
        Solution to the discrete algebraic Riccati equation.

    Raises
    ------
    LinAlgError
        For cases where the stable subspace of the pencil could not be
        isolated. See Notes section and the references for details.

    See Also
    --------
    solve_continuous_are : Solves the continuous algebraic Riccati equation

    Notes
    -----
    The equation is solved by forming the extended symplectic matrix pencil,
    as described in [1]_, :math:`H - \lambda J` given by the block matrices ::

           [  A   0   B ]             [ E   0   B ]
           [ -Q  E^H -S ] - \lambda * [ 0  A^H  0 ]
           [ S^H  0   R ]             [ 0 -B^H  0 ]

    and using a QZ decomposition method.

    In this algorithm, the fail conditions are linked to the symmetry
    of the product :math:`U_2 U_1^{-1}` and condition number of
    :math:`U_1`. Here, :math:`U` is the 2m-by-m matrix that holds the
    eigenvectors spanning the stable subspace with 2-m rows and partitioned
    into two m-row matrices. See [1]_ and [2]_ for more details.

    In order to improve the QZ decomposition accuracy, the pencil goes
    through a balancing step where the sum of absolute values of
    :math:`H` and :math:`J` rows/cols (after removing the diagonal entries)
    is balanced following the recipe given in [3]_. If the data has small
    numerical noise, balancing may amplify their effects and some clean up
    is required.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1]  P. van Dooren , "A Generalized Eigenvalue Approach For Solving
       Riccati Equations.", SIAM Journal on Scientific and Statistical
       Computing, Vol.2(2), :doi:`10.1137/0902010`

    .. [2] A.J. Laub, "A Schur Method for Solving Algebraic Riccati
       Equations.", Massachusetts Institute of Technology. Laboratory for
       Information and Decision Systems. LIDS-R ; 859. Available online :
       http://hdl.handle.net/1721.1/1301

    .. [3] P. Benner, "Symplectic Balancing of Hamiltonian Matrices", 2001,
       SIAM J. Sci. Comput., 2001, Vol.22(5), :doi:`10.1137/S1064827500367993`

    Examples
    --------
    Given `a`, `b`, `q`, and `r` solve for `x`:

    >>> from scipy import linalg as la
    >>> a = np.array([[0, 1], [0, -1]])
    >>> b = np.array([[1, 0], [2, 1]])
    >>> q = np.array([[-4, -4], [-4, 7]])
    >>> r = np.array([[9, 3], [3, 1]])
    >>> x = la.solve_discrete_are(a, b, q, r)
    >>> x
    array([[-4., -4.],
           [-4.,  7.]])
    >>> R = la.solve(r + b.T.dot(x).dot(b), b.T.dot(x).dot(a))
    >>> np.allclose(a.T.dot(x).dot(a) - x - a.T.dot(x).dot(b).dot(R), -q)
    True

    """

    # Validate input arguments
    a, b, q, r, e, s, m, n, r_or_c, gen_are = _are_validate_args(
                                                     a, b, q, r, e, s, 'dare')

    # Form the matrix pencil
    H = np.zeros((2*m+n, 2*m+n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = np.eye(m) if e is None else e.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, 2*m:] = r

    J = np.zeros_like(H, dtype=r_or_c)
    J[:m, :m] = np.eye(m) if e is None else e
    J[m:2*m, m:2*m] = a.conj().T
    J[2*m:, m:2*m] = -b.conj().T

    if balanced:
        # xGEBAL does not remove the diagonals before scaling. Also
        # to avoid destroying the Symplectic structure, we follow Ref.3
        M = np.abs(H) + np.abs(J)
        M[np.diag_indices_from(M)] = 0.
        _, (sca, _) = matrix_balance(M, separate=1, permute=0)
        # do we need to bother?
        if not np.allclose(sca, np.ones_like(sca)):
            # Now impose diag(D,inv(D)) from Benner where D is
            # square root of s_i/s_(n+i) for i=0,....
            sca = np.log2(sca)
            # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
            s = np.round((sca[m:2*m] - sca[:m])/2)
            sca = 2 ** np.r_[s, -s, sca[2*m:]]
            # Elementwise multiplication via broadcasting.
            elwisescale = sca[:, None] * np.reciprocal(sca)
            H *= elwisescale
            J *= elwisescale

    # Deflate the pencil by the R column ala Ref.1
    q_of_qr, _ = qr(H[:, -n:])
    H = q_of_qr[:, n:].conj().T.dot(H[:, :2*m])
    J = q_of_qr[:, n:].conj().T.dot(J[:, :2*m])

    # Decide on which output type is needed for QZ
    out_str = 'real' if r_or_c == float else 'complex'

    _, _, _, _, _, u = ordqz(H, J, sort='iuc',
                             overwrite_a=True,
                             overwrite_b=True,
                             check_finite=False,
                             output=out_str)

    # Get the relevant parts of the stable subspace basis
    if e is not None:
        u, _ = qr(np.vstack((e.dot(u[:m, :m]), u[m:, :m])))
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # Solve via back-substituion after checking the condition of u00
    up, ul, uu = lu(u00)

    if 1/cond(uu) < np.spacing(1.):
        raise LinAlgError('Failed to find a finite solution.')

    # Exploit the triangular structure
    x = solve_triangular(ul.conj().T,
                         solve_triangular(uu.conj().T,
                                          u10.conj().T,
                                          lower=True),
                         unit_diagonal=True,
                         ).conj().T.dot(up.conj().T)
    if balanced:
        x *= sca[:m, None] * sca[:m]

    # Check the deviation from symmetry for lack of success
    # See proof of Thm.5 item 3 in [2]
    u_sym = u00.conj().T.dot(u10)
    n_u_sym = norm(u_sym, 1)
    u_sym = u_sym - u_sym.conj().T
    sym_threshold = np.max([np.spacing(1000.), 0.1*n_u_sym])

    if norm(u_sym, 1) > sym_threshold:
        raise LinAlgError('The associated symplectic pencil has eigenvalues'
                          'too close to the unit circle')

    return (x + x.conj().T)/2


def _are_validate_args(a, b, q, r, e, s, eq_type='care'):
    """
    A helper function to validate the arguments supplied to the
    Riccati equation solvers. Any discrepancy found in the input
    matrices leads to a ``ValueError`` exception.

    Essentially, it performs:

        - a check whether the input is free of NaN and Infs
        - a pass for the data through ``numpy.atleast_2d()``
        - squareness check of the relevant arrays
        - shape consistency check of the arrays
        - singularity check of the relevant arrays
        - symmetricity check of the relevant matrices
        - a check whether the regular or the generalized version is asked.

    This function is used by ``solve_continuous_are`` and
    ``solve_discrete_are``.

    Parameters
    ----------
    a, b, q, r, e, s : array_like
        Input data
    eq_type : str
        Accepted arguments are 'care' and 'dare'.

    Returns
    -------
    a, b, q, r, e, s : ndarray
        Regularized input data
    m, n : int
        shape of the problem
    r_or_c : type
        Data type of the problem, returns float or complex
    gen_or_not : bool
        Type of the equation, True for generalized and False for regular ARE.

    """

    if not eq_type.lower() in ('dare', 'care'):
        raise ValueError("Equation type unknown. "
                         "Only 'care' and 'dare' is understood")

    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    b = np.atleast_2d(_asarray_validated(b, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r = np.atleast_2d(_asarray_validated(r, check_finite=True))

    # Get the correct data types otherwise NumPy complains
    # about pushing complex numbers into real arrays.
    r_or_c = complex if np.iscomplexobj(b) else float

    for ind, mat in enumerate((a, q, r)):
        if np.iscomplexobj(mat):
            r_or_c = complex

        if not np.equal(*mat.shape):
            raise ValueError("Matrix {} should be square.".format("aqr"[ind]))

    # Shape consistency checks
    m, n = b.shape
    if m != a.shape[0]:
        raise ValueError("Matrix a and b should have the same number of rows.")
    if m != q.shape[0]:
        raise ValueError("Matrix a and q should have the same shape.")
    if n != r.shape[0]:
        raise ValueError("Matrix b and r should have the same number of cols.")

    # Check if the data matrices q, r are (sufficiently) hermitian
    for ind, mat in enumerate((q, r)):
        if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1))*100:
            raise ValueError("Matrix {} should be symmetric/hermitian."
                             "".format("qr"[ind]))

    # Continuous time ARE should have a nonsingular r matrix.
    if eq_type == 'care':
        min_sv = svd(r, compute_uv=False)[-1]
        if min_sv == 0. or min_sv < np.spacing(1.)*norm(r, 1):
            raise ValueError('Matrix r is numerically singular.')

    # Check if the generalized case is required with omitted arguments
    # perform late shape checking etc.
    generalized_case = e is not None or s is not None

    if generalized_case:
        if e is not None:
            e = np.atleast_2d(_asarray_validated(e, check_finite=True))
            if not np.equal(*e.shape):
                raise ValueError("Matrix e should be square.")
            if m != e.shape[0]:
                raise ValueError("Matrix a and e should have the same shape.")
            # numpy.linalg.cond doesn't check for exact zeros and
            # emits a runtime warning. Hence the following manual check.
            min_sv = svd(e, compute_uv=False)[-1]
            if min_sv == 0. or min_sv < np.spacing(1.) * norm(e, 1):
                raise ValueError('Matrix e is numerically singular.')
            if np.iscomplexobj(e):
                r_or_c = complex
        if s is not None:
            s = np.atleast_2d(_asarray_validated(s, check_finite=True))
            if s.shape != b.shape:
                raise ValueError("Matrix b and s should have the same shape.")
            if np.iscomplexobj(s):
                r_or_c = complex

    return a, b, q, r, e, s, m, n, r_or_c, generalized_case
