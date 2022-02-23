"""
Matrix functions that use Pade approximation with inverse scaling and squaring.

"""
import warnings

import numpy as np

from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg.decomp_schur import schur, rsf2csf
from scipy.linalg.matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special


class LogmRankWarning(UserWarning):
    pass


class LogmExactlySingularWarning(LogmRankWarning):
    pass


class LogmNearlySingularWarning(LogmRankWarning):
    pass


class LogmError(np.linalg.LinAlgError):
    pass


class FractionalMatrixPowerError(np.linalg.LinAlgError):
    pass


#TODO renovate or move this class when scipy operators are more mature
class _MatrixM1PowerOperator(LinearOperator):
    """
    A representation of the linear operator (A - I)^p.
    """

    def __init__(self, A, p):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0 or p != int(p):
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x) - x
        return x

    def _rmatvec(self, x):
        for i in range(self._p):
            x = x.dot(self._A) - x
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = self._A.dot(X) - X
        return X

    def _adjoint(self):
        return _MatrixM1PowerOperator(self._A.T, self._p)


#TODO renovate or move this function when SciPy operators are more mature
def _onenormest_m1_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Efficiently estimate the 1-norm of (A - I)^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    return onenormest(_MatrixM1PowerOperator(A, p),
            t=t, itmax=itmax, compute_v=compute_v, compute_w=compute_w)


def _unwindk(z):
    """
    Compute the scalar unwinding number.

    Uses Eq. (5.3) in [1]_, and should be equal to (z - log(exp(z)) / (2 pi i).
    Note that this definition differs in sign from the original definition
    in equations (5, 6) in [2]_.  The sign convention is justified in [3]_.

    Parameters
    ----------
    z : complex
        A complex number.

    Returns
    -------
    unwinding_number : integer
        The scalar unwinding number of z.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    .. [2] Robert M. Corless and David J. Jeffrey,
           "The unwinding number." Newsletter ACM SIGSAM Bulletin
           Volume 30, Issue 2, June 1996, Pages 28-35.

    .. [3] Russell Bradford and Robert M. Corless and James H. Davenport and
           David J. Jeffrey and Stephen M. Watt,
           "Reasoning about the elementary functions of complex analysis"
           Annals of Mathematics and Artificial Intelligence,
           36: 303-318, 2002.

    """
    return int(np.ceil((z.imag - np.pi) / (2*np.pi)))


def _briggs_helper_function(a, k):
    """
    Computes r = a^(1 / (2^k)) - 1.

    This is algorithm (2) of [1]_.
    The purpose is to avoid a danger of subtractive cancellation.
    For more computational efficiency it should probably be cythonized.

    Parameters
    ----------
    a : complex
        A complex number.
    k : integer
        A nonnegative integer.

    Returns
    -------
    r : complex
        The value r = a^(1 / (2^k)) - 1 computed with less cancellation.

    Notes
    -----
    The algorithm as formulated in the reference does not handle k=0 or k=1
    correctly, so these are special-cased in this implementation.
    This function is intended to not allow `a` to belong to the closed
    negative real axis, but this constraint is relaxed.

    References
    ----------
    .. [1] Awad H. Al-Mohy (2012)
           "A more accurate Briggs method for the logarithm",
           Numerical Algorithms, 59 : 393--402.

    """
    if k < 0 or int(k) != k:
        raise ValueError('expected a nonnegative integer k')
    if k == 0:
        return a - 1
    elif k == 1:
        return np.sqrt(a) - 1
    else:
        k_hat = k
        if np.angle(a) >= np.pi / 2:
            a = np.sqrt(a)
            k_hat = k - 1
        z0 = a - 1
        a = np.sqrt(a)
        r = 1 + a
        for j in range(1, k_hat):
            a = np.sqrt(a)
            r = r * (1 + a)
        r = z0 / r
        return r


def _fractional_power_superdiag_entry(l1, l2, t12, p):
    """
    Compute a superdiagonal entry of a fractional matrix power.

    This is Eq. (5.6) in [1]_.

    Parameters
    ----------
    l1 : complex
        A diagonal entry of the matrix.
    l2 : complex
        A diagonal entry of the matrix.
    t12 : complex
        A superdiagonal entry of the matrix.
    p : float
        A fractional power.

    Returns
    -------
    f12 : complex
        A superdiagonal entry of the fractional matrix power.

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    if l1 == l2:
        f12 = t12 * p * l1**(p-1)
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * ((l2**p) - (l1**p)) / (l2 - l1)
    else:
        # This is Eq. (5.5) in [1].
        z = (l2 - l1) / (l2 + l1)
        log_l1 = np.log(l1)
        log_l2 = np.log(l2)
        arctanh_z = np.arctanh(z)
        tmp_a = t12 * np.exp((p/2)*(log_l2 + log_l1))
        tmp_u = _unwindk(log_l2 - log_l1)
        if tmp_u:
            tmp_b = p * (arctanh_z + np.pi * 1j * tmp_u)
        else:
            tmp_b = p * arctanh_z
        tmp_c = 2 * np.sinh(tmp_b) / (l2 - l1)
        f12 = tmp_a * tmp_c
    return f12


def _logm_superdiag_entry(l1, l2, t12):
    """
    Compute a superdiagonal entry of a matrix logarithm.

    This is like Eq. (11.28) in [1]_, except the determination of whether
    l1 and l2 are sufficiently far apart has been modified.

    Parameters
    ----------
    l1 : complex
        A diagonal entry of the matrix.
    l2 : complex
        A diagonal entry of the matrix.
    t12 : complex
        A superdiagonal entry of the matrix.

    Returns
    -------
    f12 : complex
        A superdiagonal entry of the matrix logarithm.

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    """
    if l1 == l2:
        f12 = t12 / l1
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * (np.log(l2) - np.log(l1)) / (l2 - l1)
    else:
        z = (l2 - l1) / (l2 + l1)
        u = _unwindk(np.log(l2) - np.log(l1))
        if u:
            f12 = t12 * 2 * (np.arctanh(z) + np.pi*1j*u) / (l2 - l1)
        else:
            f12 = t12 * 2 * np.arctanh(z) / (l2 - l1)
    return f12


def _inverse_squaring_helper(T0, theta):
    """
    A helper function for inverse scaling and squaring for Pade approximation.

    Parameters
    ----------
    T0 : (N, N) array_like upper triangular
        Matrix involved in inverse scaling and squaring.
    theta : indexable
        The values theta[1] .. theta[7] must be available.
        They represent bounds related to Pade approximation, and they depend
        on the matrix function which is being computed.
        For example, different values of theta are required for
        matrix logarithm than for fractional matrix power.

    Returns
    -------
    R : (N, N) array_like upper triangular
        Composition of zero or more matrix square roots of T0, minus I.
    s : non-negative integer
        Number of square roots taken.
    m : positive integer
        The degree of the Pade approximation.

    Notes
    -----
    This subroutine appears as a chunk of lines within
    a couple of published algorithms; for example it appears
    as lines 4--35 in algorithm (3.1) of [1]_, and
    as lines 3--34 in algorithm (4.1) of [2]_.
    The instances of 'goto line 38' in algorithm (3.1) of [1]_
    probably mean 'goto line 36' and have been intepreted accordingly.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    """
    if len(T0.shape) != 2 or T0.shape[0] != T0.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T0.shape
    T = T0

    # Find s0, the smallest s such that the spectral radius
    # of a certain diagonal matrix is at most theta[7].
    # Note that because theta[7] < 1,
    # this search will not terminate if any diagonal entry of T is zero.
    s0 = 0
    tmp_diag = np.diag(T)
    if np.count_nonzero(tmp_diag) != n:
        raise Exception('internal inconsistency')
    while np.max(np.absolute(tmp_diag - 1)) > theta[7]:
        tmp_diag = np.sqrt(tmp_diag)
        s0 += 1

    # Take matrix square roots of T.
    for i in range(s0):
        T = _sqrtm_triu(T)

    # Flow control in this section is a little odd.
    # This is because I am translating algorithm descriptions
    # which have GOTOs in the publication.
    s = s0
    k = 0
    d2 = _onenormest_m1_power(T, 2) ** (1/2)
    d3 = _onenormest_m1_power(T, 3) ** (1/3)
    a2 = max(d2, d3)
    m = None
    for i in (1, 2):
        if a2 <= theta[i]:
            m = i
            break
    while m is None:
        if s > s0:
            d3 = _onenormest_m1_power(T, 3) ** (1/3)
        d4 = _onenormest_m1_power(T, 4) ** (1/4)
        a3 = max(d3, d4)
        if a3 <= theta[7]:
            j1 = min(i for i in (3, 4, 5, 6, 7) if a3 <= theta[i])
            if j1 <= 6:
                m = j1
                break
            elif a3 / 2 <= theta[5] and k < 2:
                k += 1
                T = _sqrtm_triu(T)
                s += 1
                continue
        d5 = _onenormest_m1_power(T, 5) ** (1/5)
        a4 = max(d4, d5)
        eta = min(a3, a4)
        for i in (6, 7):
            if eta <= theta[i]:
                m = i
                break
        if m is not None:
            break
        T = _sqrtm_triu(T)
        s += 1

    # The subtraction of the identity is redundant here,
    # because the diagonal will be replaced for improved numerical accuracy,
    # but this formulation should help clarify the meaning of R.
    R = T - np.identity(n)

    # Replace the diagonal and first superdiagonal of T0^(1/(2^s)) - I
    # using formulas that have less subtractive cancellation.
    # Skip this step if the principal branch
    # does not exist at T0; this happens when a diagonal entry of T0
    # is negative with imaginary part 0.
    has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    if has_principal_branch:
        for j in range(n):
            a = T0[j, j]
            r = _briggs_helper_function(a, s)
            R[j, j] = r
        p = np.exp2(-s)
        for j in range(n-1):
            l1 = T0[j, j]
            l2 = T0[j+1, j+1]
            t12 = T0[j, j+1]
            f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
            R[j, j+1] = f12

    # Return the T-I matrix, the number of square roots, and the Pade degree.
    if not np.array_equal(R, np.triu(R)):
        raise Exception('internal inconsistency')
    return R, s, m


def _fractional_power_pade_constant(i, t):
    # A helper function for matrix fractional power.
    if i < 1:
        raise ValueError('expected a positive integer i')
    if not (-1 < t < 1):
        raise ValueError('expected -1 < t < 1')
    if i == 1:
        return -t
    elif i % 2 == 0:
        j = i // 2
        return (-j + t) / (2 * (2*j - 1))
    elif i % 2 == 1:
        j = (i - 1) // 2
        return (-j - t) / (2 * (2*j + 1))
    else:
        raise Exception('internal error')


def _fractional_power_pade(R, t, m):
    """
    Evaluate the Pade approximation of a fractional matrix power.

    Evaluate the degree-m Pade approximation of R
    to the fractional matrix power t using the continued fraction
    in bottom-up fashion using algorithm (4.1) in [1]_.

    Parameters
    ----------
    R : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.
    m : positive integer
        Degree of Pade approximation.

    Returns
    -------
    U : (N, N) array_like
        The degree-m Pade approximation of R to the fractional power t.
        This matrix will be upper triangular.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    if m < 1 or int(m) != m:
        raise ValueError('expected a positive integer m')
    if not (-1 < t < 1):
        raise ValueError('expected -1 < t < 1')
    R = np.asarray(R)
    if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = R.shape
    ident = np.identity(n)
    Y = R * _fractional_power_pade_constant(2*m, t)
    for j in range(2*m - 1, 0, -1):
        rhs = R * _fractional_power_pade_constant(j, t)
        Y = solve_triangular(ident + Y, rhs)
    U = ident + Y
    if not np.array_equal(U, np.triu(U)):
        raise Exception('internal inconsistency')
    return U


def _remainder_matrix_power_triu(T, t):
    """
    Compute a fractional power of an upper triangular matrix.

    The fractional power is restricted to fractions -1 < t < 1.
    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    m_to_theta = {
            1: 1.51e-5,
            2: 2.24e-3,
            3: 1.88e-2,
            4: 6.04e-2,
            5: 1.24e-1,
            6: 2.00e-1,
            7: 2.79e-1,
            }
    n, n = T.shape
    T0 = T
    T0_diag = np.diag(T0)
    if np.array_equal(T0, np.diag(T0_diag)):
        U = np.diag(T0_diag ** t)
    else:
        R, s, m = _inverse_squaring_helper(T0, m_to_theta)

        # Evaluate the Pade approximation.
        # Note that this function expects the negative of the matrix
        # returned by the inverse squaring helper.
        U = _fractional_power_pade(-R, t, m)

        # Undo the inverse scaling and squaring.
        # Be less clever about this
        # if the principal branch does not exist at T0;
        # this happens when a diagonal entry of T0
        # is negative with imaginary part 0.
        eivals = np.diag(T0)
        has_principal_branch = all(x.real > 0 or x.imag != 0 for x in eivals)
        for i in range(s, -1, -1):
            if i < s:
                U = U.dot(U)
            else:
                if has_principal_branch:
                    p = t * np.exp2(-i)
                    U[np.diag_indices(n)] = T0_diag ** p
                    for j in range(n-1):
                        l1 = T0[j, j]
                        l2 = T0[j+1, j+1]
                        t12 = T0[j, j+1]
                        f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
                        U[j, j+1] = f12
    if not np.array_equal(U, np.triu(U)):
        raise Exception('internal inconsistency')
    return U


def _remainder_matrix_power(A, t):
    """
    Compute the fractional power of a matrix, for fractions -1 < t < 1.

    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    # This code block is copied from numpy.matrix_power().
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('input must be a square array')

    # Get the number of rows and columns.
    n, n = A.shape

    # Triangularize the matrix if necessary,
    # attempting to preserve dtype if possible.
    if np.array_equal(A, np.triu(A)):
        Z = None
        T = A
    else:
        if np.isrealobj(A):
            T, Z = schur(A)
            if not np.array_equal(T, np.triu(T)):
                T, Z = rsf2csf(T, Z)
        else:
            T, Z = schur(A, output='complex')

    # Zeros on the diagonal of the triangular matrix are forbidden,
    # because the inverse scaling and squaring cannot deal with it.
    T_diag = np.diag(T)
    if np.count_nonzero(T_diag) != n:
        raise FractionalMatrixPowerError(
                'cannot use inverse scaling and squaring to find '
                'the fractional matrix power of a singular matrix')

    # If the triangular matrix is real and has a negative
    # entry on the diagonal, then force the matrix to be complex.
    if np.isrealobj(T) and np.min(T_diag) < 0:
        T = T.astype(complex)

    # Get the fractional power of the triangular matrix,
    # and de-triangularize it if necessary.
    U = _remainder_matrix_power_triu(T, t)
    if Z is not None:
        ZH = np.conjugate(Z).T
        return Z.dot(U).dot(ZH)
    else:
        return U


def _fractional_matrix_power(A, p):
    """
    Compute the fractional power of a matrix.

    See the fractional_matrix_power docstring in matfuncs.py for more info.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    if p == int(p):
        return np.linalg.matrix_power(A, int(p))
    # Compute singular values.
    s = svdvals(A)
    # Inverse scaling and squaring cannot deal with a singular matrix,
    # because the process of repeatedly taking square roots
    # would not converge to the identity matrix.
    if s[-1]:
        # Compute the condition number relative to matrix inversion,
        # and use this to decide between floor(p) and ceil(p).
        k2 = s[0] / s[-1]
        p1 = p - np.floor(p)
        p2 = p - np.ceil(p)
        if p1 * k2 ** (1 - p1) <= -p2 * k2:
            a = int(np.floor(p))
            b = p1
        else:
            a = int(np.ceil(p))
            b = p2
        try:
            R = _remainder_matrix_power(A, b)
            Q = np.linalg.matrix_power(A, a)
            return Q.dot(R)
        except np.linalg.LinAlgError:
            pass
    # If p is negative then we are going to give up.
    # If p is non-negative then we can fall back to generic funm.
    if p < 0:
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
    else:
        p1 = p - np.floor(p)
        a = int(np.floor(p))
        b = p1
        R, info = funm(A, lambda x: pow(x, b), disp=False)
        Q = np.linalg.matrix_power(A, a)
        return Q.dot(R)


def _logm_triu(T):
    """
    Compute matrix logarithm of an upper triangular matrix.

    The matrix logarithm is the inverse of
    expm: expm(logm(`T`)) == `T`

    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose logarithm to evaluate

    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `T`

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    T = np.asarray(T)
    if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T.shape

    # Construct T0 with the appropriate type,
    # depending on the dtype and the spectrum of T.
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0
    if keep_it_real:
        T0 = T
    else:
        T0 = T.astype(complex)

    # Define bounds given in Table (2.1).
    theta = (None,
            1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2,
            1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1,
            4.39e-1, 5.03e-1, 5.60e-1, 6.09e-1,
            6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1)

    R, s, m = _inverse_squaring_helper(T0, theta)

    # Evaluate U = 2**s r_m(T - I) using the partial fraction expansion (1.1).
    # This requires the nodes and weights
    # corresponding to degree-m Gauss-Legendre quadrature.
    # These quadrature arrays need to be transformed from the [-1, 1] interval
    # to the [0, 1] interval.
    nodes, weights = scipy.special.p_roots(m)
    nodes = nodes.real
    if nodes.shape != (m,) or weights.shape != (m,):
        raise Exception('internal error')
    nodes = 0.5 + 0.5 * nodes
    weights = 0.5 * weights
    ident = np.identity(n)
    U = np.zeros_like(R)
    for alpha, beta in zip(weights, nodes):
        U += solve_triangular(ident + beta*R, alpha*R)
    U *= np.exp2(s)

    # Skip this step if the principal branch
    # does not exist at T0; this happens when a diagonal entry of T0
    # is negative with imaginary part 0.
    has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    if has_principal_branch:

        # Recompute diagonal entries of U.
        U[np.diag_indices(n)] = np.log(np.diag(T0))

        # Recompute superdiagonal entries of U.
        # This indexing of this code should be renovated
        # when newer np.diagonal() becomes available.
        for i in range(n-1):
            l1 = T0[i, i]
            l2 = T0[i+1, i+1]
            t12 = T0[i, i+1]
            U[i, i+1] = _logm_superdiag_entry(l1, l2, t12)

    # Return the logm of the upper triangular matrix.
    if not np.array_equal(U, np.triu(U)):
        raise Exception('internal inconsistency')
    return U


def _logm_force_nonsingular_triangular_matrix(T, inplace=False):
    # The input matrix should be upper triangular.
    # The eps is ad hoc and is not meant to be machine precision.
    tri_eps = 1e-20
    abs_diag = np.absolute(np.diag(T))
    if np.any(abs_diag == 0):
        exact_singularity_msg = 'The logm input matrix is exactly singular.'
        warnings.warn(exact_singularity_msg, LogmExactlySingularWarning)
        if not inplace:
            T = T.copy()
        n = T.shape[0]
        for i in range(n):
            if not T[i, i]:
                T[i, i] = tri_eps
    elif np.any(abs_diag < tri_eps):
        near_singularity_msg = 'The logm input matrix may be nearly singular.'
        warnings.warn(near_singularity_msg, LogmNearlySingularWarning)
    return T


def _logm(A):
    """
    Compute the matrix logarithm.

    See the logm docstring in matfuncs.py for more info.

    Notes
    -----
    In this function we look at triangular matrices that are similar
    to the input matrix. If any diagonal entry of such a triangular matrix
    is exactly zero then the original matrix is singular.
    The matrix logarithm does not exist for such matrices,
    but in such cases we will pretend that the diagonal entries that are zero
    are actually slightly positive by an ad-hoc amount, in the interest
    of returning something more useful than NaN. This will cause a warning.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # If the input matrix dtype is integer then copy to a float dtype matrix.
    if issubclass(A.dtype.type, np.integer):
        A = np.asarray(A, dtype=float)

    keep_it_real = np.isrealobj(A)
    try:
        if np.array_equal(A, np.triu(A)):
            A = _logm_force_nonsingular_triangular_matrix(A)
            if np.min(np.diag(A)) < 0:
                A = A.astype(complex)
            return _logm_triu(A)
        else:
            if keep_it_real:
                T, Z = schur(A)
                if not np.array_equal(T, np.triu(T)):
                    T, Z = rsf2csf(T, Z)
            else:
                T, Z = schur(A, output='complex')
            T = _logm_force_nonsingular_triangular_matrix(T, inplace=True)
            U = _logm_triu(T)
            ZH = np.conjugate(Z).T
            return Z.dot(U).dot(ZH)
    except (SqrtmError, LogmError):
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
