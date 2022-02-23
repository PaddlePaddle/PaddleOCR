"""
Sparse matrix functions
"""

#
# Authors: Travis Oliphant, March 2002
#          Anthony Scopatz, August 2012 (Sparse Updates)
#          Jake Vanderplas, August 2012 (Sparse Updates)
#

__all__ = ['expm', 'inv']

import numpy as np

import scipy.special
from scipy._lib._util import float_factorial
from scipy.linalg.basic import solve, solve_triangular

from scipy.sparse.base import isspmatrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.sputils import is_pydata_spmatrix

import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.interface import LinearOperator

from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm


UPPER_TRIANGULAR = 'upper_triangular'


def inv(A):
    """
    Compute the inverse of a sparse matrix

    Parameters
    ----------
    A : (M,M) ndarray or sparse matrix
        square matrix to be inverted

    Returns
    -------
    Ainv : (M,M) ndarray or sparse matrix
        inverse of `A`

    Notes
    -----
    This computes the sparse inverse of `A`. If the inverse of `A` is expected
    to be non-sparse, it will likely be faster to convert `A` to dense and use
    scipy.linalg.inv.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import inv
    >>> A = csc_matrix([[1., 0.], [1., 2.]])
    >>> Ainv = inv(A)
    >>> Ainv
    <2x2 sparse matrix of type '<class 'numpy.float64'>'
        with 3 stored elements in Compressed Sparse Column format>
    >>> A.dot(Ainv)
    <2x2 sparse matrix of type '<class 'numpy.float64'>'
        with 2 stored elements in Compressed Sparse Column format>
    >>> A.dot(Ainv).todense()
    matrix([[ 1.,  0.],
            [ 0.,  1.]])

    .. versionadded:: 0.12.0

    """
    #check input
    if not (scipy.sparse.isspmatrix(A) or is_pydata_spmatrix(A)):
        raise TypeError('Input must be a sparse matrix')

    I = _ident_like(A)
    Ainv = spsolve(A, I)
    return Ainv


def _onenorm_matrix_power_nnm(A, p):
    """
    Compute the 1-norm of a non-negative integer power of a non-negative matrix.

    Parameters
    ----------
    A : a square ndarray or matrix or sparse matrix
        Input matrix with non-negative entries.
    p : non-negative integer
        The power to which the matrix is to be raised.

    Returns
    -------
    out : float
        The 1-norm of the matrix power p of A.

    """
    # check input
    if int(p) != p or p < 0:
        raise ValueError('expected non-negative integer p')
    p = int(p)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # Explicitly make a column vector so that this works when A is a
    # numpy matrix (in addition to ndarray and sparse matrix).
    v = np.ones((A.shape[0], 1), dtype=float)
    M = A.T
    for i in range(p):
        v = M.dot(v)
    return np.max(v)


def _is_upper_triangular(A):
    # This function could possibly be of wider interest.
    if isspmatrix(A):
        lower_part = scipy.sparse.tril(A, -1)
        # Check structural upper triangularity,
        # then coincidental upper triangularity if needed.
        return lower_part.nnz == 0 or lower_part.count_nonzero() == 0
    elif is_pydata_spmatrix(A):
        import sparse
        lower_part = sparse.tril(A, -1)
        return lower_part.nnz == 0
    else:
        return not np.tril(A, -1).any()


def _smart_matrix_product(A, B, alpha=None, structure=None):
    """
    A matrix product that knows about sparse and structured matrices.

    Parameters
    ----------
    A : 2d ndarray
        First matrix.
    B : 2d ndarray
        Second matrix.
    alpha : float
        The matrix product will be scaled by this constant.
    structure : str, optional
        A string describing the structure of both matrices `A` and `B`.
        Only `upper_triangular` is currently supported.

    Returns
    -------
    M : 2d ndarray
        Matrix product of A and B.

    """
    if len(A.shape) != 2:
        raise ValueError('expected A to be a rectangular matrix')
    if len(B.shape) != 2:
        raise ValueError('expected B to be a rectangular matrix')
    f = None
    if structure == UPPER_TRIANGULAR:
        if (not isspmatrix(A) and not isspmatrix(B)
                and not is_pydata_spmatrix(A) and not is_pydata_spmatrix(B)):
            f, = scipy.linalg.get_blas_funcs(('trmm',), (A, B))
    if f is not None:
        if alpha is None:
            alpha = 1.
        out = f(alpha, A, B)
    else:
        if alpha is None:
            out = A.dot(B)
        else:
            out = alpha * A.dot(B)
    return out


class MatrixPowerOperator(LinearOperator):

    def __init__(self, A, p, structure=None):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0:
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self._structure = structure
        self.dtype = A.dtype
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x)
        return x

    def _rmatvec(self, x):
        A_T = self._A.T
        x = x.ravel()
        for i in range(self._p):
            x = A_T.dot(x)
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = _smart_matrix_product(self._A, X, structure=self._structure)
        return X

    @property
    def T(self):
        return MatrixPowerOperator(self._A.T, self._p)


class ProductOperator(LinearOperator):
    """
    For now, this is limited to products of multiple square matrices.
    """

    def __init__(self, *args, **kwargs):
        self._structure = kwargs.get('structure', None)
        for A in args:
            if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(
                        'For now, the ProductOperator implementation is '
                        'limited to the product of multiple square matrices.')
        if args:
            n = args[0].shape[0]
            for A in args:
                for d in A.shape:
                    if d != n:
                        raise ValueError(
                                'The square matrices of the ProductOperator '
                                'must all have the same shape.')
            self.shape = (n, n)
            self.ndim = len(self.shape)
        self.dtype = np.find_common_type([x.dtype for x in args], [])
        self._operator_sequence = args

    def _matvec(self, x):
        for A in reversed(self._operator_sequence):
            x = A.dot(x)
        return x

    def _rmatvec(self, x):
        x = x.ravel()
        for A in self._operator_sequence:
            x = A.T.dot(x)
        return x

    def _matmat(self, X):
        for A in reversed(self._operator_sequence):
            X = _smart_matrix_product(A, X, structure=self._structure)
        return X

    @property
    def T(self):
        T_args = [A.T for A in reversed(self._operator_sequence)]
        return ProductOperator(*T_args)


def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of A^p.

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
    return scipy.sparse.linalg.onenormest(
            MatrixPowerOperator(A, p, structure=structure))


def _onenormest_product(operator_seq,
        t=2, itmax=5, compute_v=False, compute_w=False, structure=None):
    """
    Efficiently estimate the 1-norm of the matrix product of the args.

    Parameters
    ----------
    operator_seq : linear operator sequence
        Matrices whose 1-norm of product is to be computed.
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
    structure : str, optional
        A string describing the structure of all operators.
        Only `upper_triangular` is currently supported.

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
    return scipy.sparse.linalg.onenormest(
            ProductOperator(*operator_seq, structure=structure))


class _ExpmPadeHelper:
    """
    Help lazily evaluate a matrix exponential.

    The idea is to not do more work than we need for high expm precision,
    so we lazily compute matrix powers and store or precompute
    other properties of the matrix.

    """
    def __init__(self, A, structure=None, use_exact_onenorm=False):
        """
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        structure : str, optional
            A string describing the structure of matrix `A`.
            Only `upper_triangular` is currently supported.
        use_exact_onenorm : bool, optional
            If True then only the exact one-norm of matrix powers and products
            will be used. Otherwise, the one-norm of powers and products
            may initially be estimated.
        """
        self.A = A
        self._A2 = None
        self._A4 = None
        self._A6 = None
        self._A8 = None
        self._A10 = None
        self._d4_exact = None
        self._d6_exact = None
        self._d8_exact = None
        self._d10_exact = None
        self._d4_approx = None
        self._d6_approx = None
        self._d8_approx = None
        self._d10_approx = None
        self.ident = _ident_like(A)
        self.structure = structure
        self.use_exact_onenorm = use_exact_onenorm

    @property
    def A2(self):
        if self._A2 is None:
            self._A2 = _smart_matrix_product(
                    self.A, self.A, structure=self.structure)
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            self._A4 = _smart_matrix_product(
                    self.A2, self.A2, structure=self.structure)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            self._A6 = _smart_matrix_product(
                    self.A4, self.A2, structure=self.structure)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            self._A8 = _smart_matrix_product(
                    self.A6, self.A2, structure=self.structure)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            self._A10 = _smart_matrix_product(
                    self.A4, self.A6, structure=self.structure)
        return self._A10

    @property
    def d4_tight(self):
        if self._d4_exact is None:
            self._d4_exact = _onenorm(self.A4)**(1/4.)
        return self._d4_exact

    @property
    def d6_tight(self):
        if self._d6_exact is None:
            self._d6_exact = _onenorm(self.A6)**(1/6.)
        return self._d6_exact

    @property
    def d8_tight(self):
        if self._d8_exact is None:
            self._d8_exact = _onenorm(self.A8)**(1/8.)
        return self._d8_exact

    @property
    def d10_tight(self):
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10)**(1/10.)
        return self._d10_exact

    @property
    def d4_loose(self):
        if self.use_exact_onenorm:
            return self.d4_tight
        if self._d4_exact is not None:
            return self._d4_exact
        else:
            if self._d4_approx is None:
                self._d4_approx = _onenormest_matrix_power(self.A2, 2,
                        structure=self.structure)**(1/4.)
            return self._d4_approx

    @property
    def d6_loose(self):
        if self.use_exact_onenorm:
            return self.d6_tight
        if self._d6_exact is not None:
            return self._d6_exact
        else:
            if self._d6_approx is None:
                self._d6_approx = _onenormest_matrix_power(self.A2, 3,
                        structure=self.structure)**(1/6.)
            return self._d6_approx

    @property
    def d8_loose(self):
        if self.use_exact_onenorm:
            return self.d8_tight
        if self._d8_exact is not None:
            return self._d8_exact
        else:
            if self._d8_approx is None:
                self._d8_approx = _onenormest_matrix_power(self.A4, 2,
                        structure=self.structure)**(1/8.)
            return self._d8_approx

    @property
    def d10_loose(self):
        if self.use_exact_onenorm:
            return self.d10_tight
        if self._d10_exact is not None:
            return self._d10_exact
        else:
            if self._d10_approx is None:
                self._d10_approx = _onenormest_product((self.A4, self.A6),
                        structure=self.structure)**(1/10.)
            return self._d10_approx

    def pade3(self):
        b = (120., 60., 12., 1.)
        U = _smart_matrix_product(self.A,
                b[3]*self.A2 + b[1]*self.ident,
                structure=self.structure)
        V = b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade5(self):
        b = (30240., 15120., 3360., 420., 30., 1.)
        U = _smart_matrix_product(self.A,
                b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
                structure=self.structure)
        V = b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade7(self):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = _smart_matrix_product(self.A,
                b[7]*self.A6 + b[5]*self.A4 + b[3]*self.A2 + b[1]*self.ident,
                structure=self.structure)
        V = b[6]*self.A6 + b[4]*self.A4 + b[2]*self.A2 + b[0]*self.ident
        return U, V

    def pade9(self):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                2162160., 110880., 3960., 90., 1.)
        U = _smart_matrix_product(self.A,
                (b[9]*self.A8 + b[7]*self.A6 + b[5]*self.A4 +
                    b[3]*self.A2 + b[1]*self.ident),
                structure=self.structure)
        V = (b[8]*self.A8 + b[6]*self.A6 + b[4]*self.A4 +
                b[2]*self.A2 + b[0]*self.ident)
        return U, V

    def pade13_scaled(self, s):
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        B = self.A * 2**-s
        B2 = self.A2 * 2**(-2*s)
        B4 = self.A4 * 2**(-4*s)
        B6 = self.A6 * 2**(-6*s)
        U2 = _smart_matrix_product(B6,
                b[13]*B6 + b[11]*B4 + b[9]*B2,
                structure=self.structure)
        U = _smart_matrix_product(B,
                (U2 + b[7]*B6 + b[5]*B4 +
                    b[3]*B2 + b[1]*self.ident),
                structure=self.structure)
        V2 = _smart_matrix_product(B6,
                b[12]*B6 + b[10]*B4 + b[8]*B2,
                structure=self.structure)
        V = V2 + b[6]*B6 + b[4]*B4 + b[2]*B2 + b[0]*self.ident
        return U, V


def expm(A):
    """
    Compute the matrix exponential using Pade approximation.

    Parameters
    ----------
    A : (M,M) array_like or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated

    Returns
    -------
    expA : (M,M) ndarray
        Matrix exponential of `A`

    Notes
    -----
    This is algorithm (6.1) which is a simplification of algorithm (5.1).

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           "A New Scaling and Squaring Algorithm for the Matrix Exponential."
           SIAM Journal on Matrix Analysis and Applications.
           31 (3). pp. 970-989. ISSN 1095-7162

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm
    >>> A = csc_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    >>> A.todense()
    matrix([[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]], dtype=int64)
    >>> Aexp = expm(A)
    >>> Aexp
    <3x3 sparse matrix of type '<class 'numpy.float64'>'
        with 3 stored elements in Compressed Sparse Column format>
    >>> Aexp.todense()
    matrix([[  2.71828183,   0.        ,   0.        ],
            [  0.        ,   7.3890561 ,   0.        ],
            [  0.        ,   0.        ,  20.08553692]])
    """
    return _expm(A, use_exact_onenorm='auto')


def _expm(A, use_exact_onenorm):
    # Core of expm, separated to allow testing exact and approximate
    # algorithms.

    # Avoid indiscriminate asarray() to allow sparse or other strange arrays.
    if isinstance(A, (list, tuple, np.matrix)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # gracefully handle size-0 input,
    # carefully handling sparse scenario
    if A.shape == (0, 0):
        out = np.zeros([0, 0], dtype=A.dtype)
        if isspmatrix(A) or is_pydata_spmatrix(A):
            return A.__class__(out)
        return out

    # Trivial case
    if A.shape == (1, 1):
        out = [[np.exp(A[0, 0])]]

        # Avoid indiscriminate casting to ndarray to
        # allow for sparse or other strange arrays
        if isspmatrix(A) or is_pydata_spmatrix(A):
            return A.__class__(out)

        return np.array(out)

    # Ensure input is of float type, to avoid integer overflows etc.
    if ((isinstance(A, np.ndarray) or isspmatrix(A) or is_pydata_spmatrix(A))
            and not np.issubdtype(A.dtype, np.inexact)):
        A = A.astype(float)

    # Detect upper triangularity.
    structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None

    if use_exact_onenorm == "auto":
        # Hardcode a matrix order threshold for exact vs. estimated one-norms.
        use_exact_onenorm = A.shape[0] < 200

    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper(
            A, structure=structure, use_exact_onenorm=use_exact_onenorm)

    # Try Pade order 3.
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
        U, V = h.pade3()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade order 5.
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
        U, V = h.pade5()
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade orders 7 and 9.
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
        U, V = h.pade7()
        return _solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
        U, V = h.pade9()
        return _solve_P_Q(U, V, structure=structure)

    # Use Pade order 13.
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25

    # Choose smallest s>=0 such that 2**(-s) eta_5 <= theta_13
    if eta_5 == 0:
        # Nilpotent special case
        s = 0
    else:
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2**-s * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V, structure=structure)
    if structure == UPPER_TRIANGULAR:
        # Invoke Code Fragment 2.1.
        X = _fragment_2_1(X, h.A, s)
    else:
        # X = r_13(A)^(2^s) by repeated squaring.
        for i in range(s):
            X = X.dot(X)
    return X


def _solve_P_Q(U, V, structure=None):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade numerator.
    V : ndarray
        Pade denominator.
    structure : str, optional
        A string describing the structure of both matrices `U` and `V`.
        Only `upper_triangular` is currently supported.

    Notes
    -----
    The `structure` argument is inspired by similar args
    for theano and cvxopt functions.

    """
    P = U + V
    Q = -U + V
    if isspmatrix(U) or is_pydata_spmatrix(U):
        return spsolve(Q, P)
    elif structure is None:
        return solve(Q, P)
    elif structure == UPPER_TRIANGULAR:
        return solve_triangular(Q, P)
    else:
        raise ValueError('unsupported matrix structure: ' + str(structure))


def _exp_sinch(a, x):
    """
    Stably evaluate exp(a)*sinh(x)/x

    Notes
    -----
    The strategy of falling back to a sixth order Taylor expansion
    was suggested by the Spallation Neutron Source docs
    which was found on the internet by google search.
    http://www.ornl.gov/~t6p/resources/xal/javadoc/gov/sns/tools/math/ElementaryFunction.html
    The details of the cutoff point and the Horner-like evaluation
    was picked without reference to anything in particular.

    Note that sinch is not currently implemented in scipy.special,
    whereas the "engineer's" definition of sinc is implemented.
    The implementation of sinc involves a scaling factor of pi
    that distinguishes it from the "mathematician's" version of sinc.

    """

    # If x is small then use sixth order Taylor expansion.
    # How small is small? I am using the point where the relative error
    # of the approximation is less than 1e-14.
    # If x is large then directly evaluate sinh(x) / x.
    if abs(x) < 0.0135:
        x2 = x*x
        return np.exp(a) * (1 + (x2/6.)*(1 + (x2/20.)*(1 + (x2/42.))))
    else:
        return (np.exp(a + x) - np.exp(a - x)) / (2*x)


def _eq_10_42(lam_1, lam_2, t_12):
    """
    Equation (10.42) of Functions of Matrices: Theory and Computation.

    Notes
    -----
    This is a helper function for _fragment_2_1 of expm_2009.
    Equation (10.42) is on page 251 in the section on Schur algorithms.
    In particular, section 10.4.3 explains the Schur-Parlett algorithm.
    expm([[lam_1, t_12], [0, lam_1])
    =
    [[exp(lam_1), t_12*exp((lam_1 + lam_2)/2)*sinch((lam_1 - lam_2)/2)],
    [0, exp(lam_2)]
    """

    # The plain formula t_12 * (exp(lam_2) - exp(lam_2)) / (lam_2 - lam_1)
    # apparently suffers from cancellation, according to Higham's textbook.
    # A nice implementation of sinch, defined as sinh(x)/x,
    # will apparently work around the cancellation.
    a = 0.5 * (lam_1 + lam_2)
    b = 0.5 * (lam_1 - lam_2)
    return t_12 * _exp_sinch(a, b)


def _fragment_2_1(X, T, s):
    """
    A helper function for expm_2009.

    Notes
    -----
    The argument X is modified in-place, but this modification is not the same
    as the returned value of the function.
    This function also takes pains to do things in ways that are compatible
    with sparse matrices, for example by avoiding fancy indexing
    and by using methods of the matrices whenever possible instead of
    using functions of the numpy or scipy libraries themselves.

    """
    # Form X = r_m(2^-s T)
    # Replace diag(X) by exp(2^-s diag(T)).
    n = X.shape[0]
    diag_T = np.ravel(T.diagonal().copy())

    # Replace diag(X) by exp(2^-s diag(T)).
    scale = 2 ** -s
    exp_diag = np.exp(scale * diag_T)
    for k in range(n):
        X[k, k] = exp_diag[k]

    for i in range(s-1, -1, -1):
        X = X.dot(X)

        # Replace diag(X) by exp(2^-i diag(T)).
        scale = 2 ** -i
        exp_diag = np.exp(scale * diag_T)
        for k in range(n):
            X[k, k] = exp_diag[k]

        # Replace (first) superdiagonal of X by explicit formula
        # for superdiagonal of exp(2^-i T) from Eq (10.42) of
        # the author's 2008 textbook
        # Functions of Matrices: Theory and Computation.
        for k in range(n-1):
            lam_1 = scale * diag_T[k]
            lam_2 = scale * diag_T[k+1]
            t_12 = scale * T[k, k+1]
            value = _eq_10_42(lam_1, lam_2, t_12)
            X[k, k+1] = value

    # Return the updated X matrix.
    return X


def _ell(A, m):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    A : linear operator
        A linear operator whose norm of power we care about.
    m : int
        The power of the linear operator

    Returns
    -------
    value : int
        A value related to a bound.

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
    # They are coefficients of terms of a generating function series expansion.
    choose_2m_m = scipy.special.comb(2*m, m, exact=True)
    abs_c_recip = float(choose_2m_m) * float_factorial(2*m + 1)

    # This is explained after Eq. (1.2) of the 2009 expm paper.
    # It is the "unit roundoff" of IEEE double precision arithmetic.
    u = 2**-53

    # Compute the one-norm of matrix power p of abs(A).
    A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), 2*m + 1)

    # Treat zero norm as a special case.
    if not A_abs_onenorm:
        return 0

    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    log2_alpha_div_u = np.log2(alpha/u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)
