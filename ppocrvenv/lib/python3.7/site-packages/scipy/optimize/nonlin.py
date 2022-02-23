r"""

Nonlinear solvers
-----------------

.. currentmodule:: scipy.optimize

This is a collection of general-purpose nonlinear multidimensional
solvers. These solvers find *x* for which *F(x) = 0*. Both *x*
and *F* can be multidimensional.

Routines
~~~~~~~~

Large-scale nonlinear solvers:

.. autosummary::

   newton_krylov
   anderson

General nonlinear solvers:

.. autosummary::

   broyden1
   broyden2

Simple iterations:

.. autosummary::

   excitingmixing
   linearmixing
   diagbroyden


Examples
~~~~~~~~

**Small problem**

>>> def F(x):
...    return np.cos(x) + x[::-1] - [1, 2, 3, 4]
>>> import scipy.optimize
>>> x = scipy.optimize.broyden1(F, [1,1,1,1], f_tol=1e-14)
>>> x
array([ 4.04674914,  3.91158389,  2.71791677,  1.61756251])
>>> np.cos(x) + x[::-1]
array([ 1.,  2.,  3.,  4.])


**Large problem**

Suppose that we needed to solve the following integrodifferential
equation on the square :math:`[0,1]\times[0,1]`:

.. math::

   \nabla^2 P = 10 \left(\int_0^1\int_0^1\cosh(P)\,dx\,dy\right)^2

with :math:`P(x,1) = 1` and :math:`P=0` elsewhere on the boundary of
the square.

The solution can be found using the `newton_krylov` solver:

.. plot::

   import numpy as np
   from scipy.optimize import newton_krylov
   from numpy import cosh, zeros_like, mgrid, zeros

   # parameters
   nx, ny = 75, 75
   hx, hy = 1./(nx-1), 1./(ny-1)

   P_left, P_right = 0, 0
   P_top, P_bottom = 1, 0

   def residual(P):
       d2x = zeros_like(P)
       d2y = zeros_like(P)

       d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
       d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
       d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

       d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
       d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
       d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

       return d2x + d2y - 10*cosh(P).mean()**2

   # solve
   guess = zeros((nx, ny), float)
   sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
   print('Residual: %g' % abs(residual(sol)).max())

   # visualize
   import matplotlib.pyplot as plt
   x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
   plt.pcolormesh(x, y, sol, shading='gouraud')
   plt.colorbar()
   plt.show()

"""
# Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as SciPy.

import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from .linesearch import scalar_search_wolfe1, scalar_search_armijo


__all__ = [
    'broyden1', 'broyden2', 'anderson', 'linearmixing',
    'diagbroyden', 'excitingmixing', 'newton_krylov']

#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------


class NoConvergence(Exception):
    pass


def maxnorm(x):
    return np.absolute(x).max()


def _as_inexact(x):
    """Return `x` as an array, of either floats or complex floats"""
    x = asarray(x)
    if not np.issubdtype(x.dtype, np.inexact):
        return asarray(x, dtype=np.float_)
    return x


def _array_like(x, x0):
    """Return ndarray `x` as same array subclass and shape as `x0`"""
    x = np.reshape(x, np.shape(x0))
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)
    return wrap(x)


def _safe_norm(v):
    if not np.isfinite(v).all():
        return np.array(np.inf)
    return norm(v)

#------------------------------------------------------------------------------
# Generic nonlinear solver machinery
#------------------------------------------------------------------------------


_doc_parts = dict(
    params_basic="""
    F : function(x) -> f
        Function whose root to find; should take and return an array-like
        object.
    xin : array_like
        Initial guess for the solution
    """.strip(),
    params_extra="""
    iter : int, optional
        Number of iterations to make. If omitted (default), make as many
        as required to meet tolerances.
    verbose : bool, optional
        Print status to stdout on every iteration.
    maxiter : int, optional
        Maximum number of iterations to make. If more are needed to
        meet convergence, `NoConvergence` is raised.
    f_tol : float, optional
        Absolute tolerance (in max-norm) for the residual.
        If omitted, default is 6e-6.
    f_rtol : float, optional
        Relative tolerance for the residual. If omitted, not used.
    x_tol : float, optional
        Absolute minimum step size, as determined from the Jacobian
        approximation. If the step size is smaller than this, optimization
        is terminated as successful. If omitted, not used.
    x_rtol : float, optional
        Relative minimum step size. If omitted, not used.
    tol_norm : function(vector) -> scalar, optional
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        Which type of a line search to use to determine the step size in the
        direction given by the Jacobian approximation. Defaults to 'armijo'.
    callback : function, optional
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual.

    Returns
    -------
    sol : ndarray
        An array (of similar array type as `x0`) containing the final solution.

    Raises
    ------
    NoConvergence
        When a solution was not found.

    """.strip()
)


def _set_doc(obj):
    if obj.__doc__:
        obj.__doc__ = obj.__doc__ % _doc_parts


def nonlin_solve(F, x0, jacobian='krylov', iter=None, verbose=False,
                 maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 tol_norm=None, line_search='armijo', callback=None,
                 full_output=False, raise_exception=True):
    """
    Find a root of a function, in a way suitable for large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    jacobian : Jacobian
        A Jacobian approximation: `Jacobian` object or something that
        `asjacobian` can transform to one. Alternatively, a string specifying
        which of the builtin Jacobian approximations to use:

            krylov, broyden1, broyden2, anderson
            diagbroyden, linearmixing, excitingmixing

    %(params_extra)s
    full_output : bool
        If true, returns a dictionary `info` containing convergence
        information.
    raise_exception : bool
        If True, a `NoConvergence` exception is raise if no solution is found.

    See Also
    --------
    asjacobian, Jacobian

    Notes
    -----
    This algorithm implements the inexact Newton method, with
    backtracking or full line searches. Several Jacobian
    approximations are available, including Krylov and Quasi-Newton
    methods.

    References
    ----------
    .. [KIM] C. T. Kelley, \"Iterative Methods for Linear and Nonlinear
       Equations\". Society for Industrial and Applied Mathematics. (1995)
       https://archive.siam.org/books/kelley/fr16/

    """
    # Can't use default parameters because it's being explicitly passed as None
    # from the calling function, so we need to set it here.
    tol_norm = maxnorm if tol_norm is None else tol_norm
    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol,
                                     x_tol=x_tol, x_rtol=x_rtol,
                                     iter=iter, norm=tol_norm)

    x0 = _as_inexact(x0)
    func = lambda z: _as_inexact(F(_array_like(z, x0))).flatten()
    x = x0.flatten()

    dx = np.full_like(x, np.inf)
    Fx = func(x)
    Fx_norm = norm(Fx)

    jacobian = asjacobian(jacobian)
    jacobian.setup(x.copy(), Fx, func)

    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100*(x.size+1)

    if line_search is True:
        line_search = 'armijo'
    elif line_search is False:
        line_search = None

    if line_search not in (None, 'armijo', 'wolfe'):
        raise ValueError("Invalid line search")

    # Solver tolerance selection
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3

    for n in range(maxiter):
        status = condition.check(Fx, x, dx)
        if status:
            break

        # The tolerance, as computed for scipy.sparse.linalg.* routines
        tol = min(eta, eta*Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)

        if norm(dx) == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        # Line search, or Newton step
        if line_search:
            s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx,
                                                        line_search)
        else:
            s = 1.0
            x = x + dx
            Fx = func(x)
            Fx_norm_new = norm(Fx)

        jacobian.update(x.copy(), Fx)

        if callback:
            callback(x, Fx)

        # Adjust forcing parameters for inexact methods
        eta_A = gamma * Fx_norm_new**2 / Fx_norm**2
        if gamma * eta**2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma*eta**2))

        Fx_norm = Fx_norm_new

        # Print status
        if verbose:
            sys.stdout.write("%d:  |F(x)| = %g; step %g\n" % (
                n, tol_norm(Fx), s))
            sys.stdout.flush()
    else:
        if raise_exception:
            raise NoConvergence(_array_like(x, x0))
        else:
            status = 2

    if full_output:
        info = {'nit': condition.iteration,
                'fun': Fx,
                'status': status,
                'success': status == 1,
                'message': {1: 'A solution was found at the specified '
                               'tolerance.',
                            2: 'The maximum number of iterations allowed '
                               'has been reached.'
                            }[status]
                }
        return _array_like(x, x0), info
    else:
        return _array_like(x, x0)


_set_doc(nonlin_solve)


def _nonlin_line_search(func, x, Fx, dx, search_type='armijo', rdiff=1e-8,
                        smin=1e-2):
    tmp_s = [0]
    tmp_Fx = [Fx]
    tmp_phi = [norm(Fx)**2]
    s_norm = norm(x) / norm(dx)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s*dx
        v = func(xt)
        p = _safe_norm(v)**2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_Fx[0] = v
        return p

    def derphi(s):
        ds = (abs(s) + s_norm + 1) * rdiff
        return (phi(s+ds, store=False) - phi(s)) / ds

    if search_type == 'wolfe':
        s, phi1, phi0 = scalar_search_wolfe1(phi, derphi, tmp_phi[0],
                                             xtol=1e-2, amin=smin)
    elif search_type == 'armijo':
        s, phi1 = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0],
                                       amin=smin)

    if s is None:
        # XXX: No suitable step length found. Take the full Newton step,
        #      and hope for the best.
        s = 1.0

    x = x + s*dx
    if s == tmp_s[0]:
        Fx = tmp_Fx[0]
    else:
        Fx = func(x)
    Fx_norm = norm(Fx)

    return s, x, Fx, Fx_norm


class TerminationCondition:
    """
    Termination condition for an iteration. It is terminated if

    - |F| < f_rtol*|F_0|, AND
    - |F| < f_tol

    AND

    - |dx| < x_rtol*|x|, AND
    - |dx| < x_tol

    """
    def __init__(self, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 iter=None, norm=maxnorm):

        if f_tol is None:
            f_tol = np.finfo(np.float_).eps ** (1./3)
        if f_rtol is None:
            f_rtol = np.inf
        if x_tol is None:
            x_tol = np.inf
        if x_rtol is None:
            x_rtol = np.inf

        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol

        self.norm = norm

        self.iter = iter

        self.f0_norm = None
        self.iteration = 0

    def check(self, f, x, dx):
        self.iteration += 1
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)

        if self.f0_norm is None:
            self.f0_norm = f_norm

        if f_norm == 0:
            return 1

        if self.iter is not None:
            # backwards compatibility with SciPy 0.6.0
            return 2 * (self.iteration > self.iter)

        # NB: condition must succeed for rtol=inf even if norm == 0
        return int((f_norm <= self.f_tol
                    and f_norm/self.f_rtol <= self.f0_norm)
                   and (dx_norm <= self.x_tol
                        and dx_norm/self.x_rtol <= x_norm))


#------------------------------------------------------------------------------
# Generic Jacobian approximation
#------------------------------------------------------------------------------

class Jacobian:
    """
    Common interface for Jacobians or Jacobian approximations.

    The optional methods come useful when implementing trust region
    etc., algorithms that often require evaluating transposes of the
    Jacobian.

    Methods
    -------
    solve
        Returns J^-1 * v
    update
        Updates Jacobian to point `x` (where the function has residual `Fx`)

    matvec : optional
        Returns J * v
    rmatvec : optional
        Returns A^H * v
    rsolve : optional
        Returns A^-H * v
    matmat : optional
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    todense : optional
        Form the dense Jacobian matrix. Necessary for dense trust region
        algorithms, and useful for testing.

    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.
    func : callable, optional
        Function the Jacobian corresponds to

    """

    def __init__(self, **kw):
        names = ["solve", "update", "matvec", "rmatvec", "rsolve",
                 "matmat", "todense", "shape", "dtype"]
        for name, value in kw.items():
            if name not in names:
                raise ValueError("Unknown keyword argument %s" % name)
            if value is not None:
                setattr(self, name, kw[name])

        if hasattr(self, 'todense'):
            self.__array__ = lambda: self.todense()

    def aspreconditioner(self):
        return InverseJacobian(self)

    def solve(self, v, tol=0):
        raise NotImplementedError

    def update(self, x, F):
        pass

    def setup(self, x, F, func):
        self.func = func
        self.shape = (F.size, x.size)
        self.dtype = F.dtype
        if self.__class__.setup is Jacobian.setup:
            # Call on the first point unless overridden
            self.update(x, F)


class InverseJacobian:
    def __init__(self, jacobian):
        self.jacobian = jacobian
        self.matvec = jacobian.solve
        self.update = jacobian.update
        if hasattr(jacobian, 'setup'):
            self.setup = jacobian.setup
        if hasattr(jacobian, 'rsolve'):
            self.rmatvec = jacobian.rsolve

    @property
    def shape(self):
        return self.jacobian.shape

    @property
    def dtype(self):
        return self.jacobian.dtype


def asjacobian(J):
    """
    Convert given object to one suitable for use as a Jacobian.
    """
    spsolve = scipy.sparse.linalg.spsolve
    if isinstance(J, Jacobian):
        return J
    elif inspect.isclass(J) and issubclass(J, Jacobian):
        return J()
    elif isinstance(J, np.ndarray):
        if J.ndim > 2:
            raise ValueError('array must have rank <= 2')
        J = np.atleast_2d(np.asarray(J))
        if J.shape[0] != J.shape[1]:
            raise ValueError('array must be square')

        return Jacobian(matvec=lambda v: dot(J, v),
                        rmatvec=lambda v: dot(J.conj().T, v),
                        solve=lambda v: solve(J, v),
                        rsolve=lambda v: solve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    elif scipy.sparse.isspmatrix(J):
        if J.shape[0] != J.shape[1]:
            raise ValueError('matrix must be square')
        return Jacobian(matvec=lambda v: J*v,
                        rmatvec=lambda v: J.conj().T * v,
                        solve=lambda v: spsolve(J, v),
                        rsolve=lambda v: spsolve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    elif hasattr(J, 'shape') and hasattr(J, 'dtype') and hasattr(J, 'solve'):
        return Jacobian(matvec=getattr(J, 'matvec'),
                        rmatvec=getattr(J, 'rmatvec'),
                        solve=J.solve,
                        rsolve=getattr(J, 'rsolve'),
                        update=getattr(J, 'update'),
                        setup=getattr(J, 'setup'),
                        dtype=J.dtype,
                        shape=J.shape)
    elif callable(J):
        # Assume it's a function J(x) that returns the Jacobian
        class Jac(Jacobian):
            def update(self, x, F):
                self.x = x

            def solve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m, v)
                elif scipy.sparse.isspmatrix(m):
                    return spsolve(m, v)
                else:
                    raise ValueError("Unknown matrix type")

            def matvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m, v)
                elif scipy.sparse.isspmatrix(m):
                    return m*v
                else:
                    raise ValueError("Unknown matrix type")

            def rsolve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m.conj().T, v)
                elif scipy.sparse.isspmatrix(m):
                    return spsolve(m.conj().T, v)
                else:
                    raise ValueError("Unknown matrix type")

            def rmatvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m.conj().T, v)
                elif scipy.sparse.isspmatrix(m):
                    return m.conj().T * v
                else:
                    raise ValueError("Unknown matrix type")
        return Jac()
    elif isinstance(J, str):
        return dict(broyden1=BroydenFirst,
                    broyden2=BroydenSecond,
                    anderson=Anderson,
                    diagbroyden=DiagBroyden,
                    linearmixing=LinearMixing,
                    excitingmixing=ExcitingMixing,
                    krylov=KrylovJacobian)[J]()
    else:
        raise TypeError('Cannot convert object to a Jacobian')


#------------------------------------------------------------------------------
# Broyden
#------------------------------------------------------------------------------

class GenericBroyden(Jacobian):
    def setup(self, x0, f0, func):
        Jacobian.setup(self, x0, f0, func)
        self.last_f = f0
        self.last_x = x0

        if hasattr(self, 'alpha') and self.alpha is None:
            # Autoscale the initial Jacobian parameter
            # unless we have already guessed the solution.
            normf0 = norm(f0)
            if normf0:
                self.alpha = 0.5*max(norm(x0), 1) / normf0
            else:
                self.alpha = 1.0

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        raise NotImplementedError

    def update(self, x, f):
        df = f - self.last_f
        dx = x - self.last_x
        self._update(x, f, dx, df, norm(dx), norm(df))
        self.last_f = f
        self.last_x = x


class LowRankMatrix:
    r"""
    A matrix represented as

    .. math:: \alpha I + \sum_{n=0}^{n=M} c_n d_n^\dagger

    However, if the rank of the matrix reaches the dimension of the vectors,
    full matrix representation will be used thereon.

    """

    def __init__(self, alpha, n, dtype):
        self.alpha = alpha
        self.cs = []
        self.ds = []
        self.n = n
        self.dtype = dtype
        self.collapsed = None

    @staticmethod
    def _matvec(v, alpha, cs, ds):
        axpy, scal, dotc = get_blas_funcs(['axpy', 'scal', 'dotc'],
                                          cs[:1] + [v])
        w = alpha * v
        for c, d in zip(cs, ds):
            a = dotc(d, v)
            w = axpy(c, w, w.size, a)
        return w

    @staticmethod
    def _solve(v, alpha, cs, ds):
        """Evaluate w = M^-1 v"""
        if len(cs) == 0:
            return v/alpha

        # (B + C D^H)^-1 = B^-1 - B^-1 C (I + D^H B^-1 C)^-1 D^H B^-1

        axpy, dotc = get_blas_funcs(['axpy', 'dotc'], cs[:1] + [v])

        c0 = cs[0]
        A = alpha * np.identity(len(cs), dtype=c0.dtype)
        for i, d in enumerate(ds):
            for j, c in enumerate(cs):
                A[i,j] += dotc(d, c)

        q = np.zeros(len(cs), dtype=c0.dtype)
        for j, d in enumerate(ds):
            q[j] = dotc(d, v)
        q /= alpha
        q = solve(A, q)

        w = v/alpha
        for c, qc in zip(cs, q):
            w = axpy(c, w, w.size, -qc)

        return w

    def matvec(self, v):
        """Evaluate w = M v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed, v)
        return LowRankMatrix._matvec(v, self.alpha, self.cs, self.ds)

    def rmatvec(self, v):
        """Evaluate w = M^H v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed.T.conj(), v)
        return LowRankMatrix._matvec(v, np.conj(self.alpha), self.ds, self.cs)

    def solve(self, v, tol=0):
        """Evaluate w = M^-1 v"""
        if self.collapsed is not None:
            return solve(self.collapsed, v)
        return LowRankMatrix._solve(v, self.alpha, self.cs, self.ds)

    def rsolve(self, v, tol=0):
        """Evaluate w = M^-H v"""
        if self.collapsed is not None:
            return solve(self.collapsed.T.conj(), v)
        return LowRankMatrix._solve(v, np.conj(self.alpha), self.ds, self.cs)

    def append(self, c, d):
        if self.collapsed is not None:
            self.collapsed += c[:,None] * d[None,:].conj()
            return

        self.cs.append(c)
        self.ds.append(d)

        if len(self.cs) > c.size:
            self.collapse()

    def __array__(self):
        if self.collapsed is not None:
            return self.collapsed

        Gm = self.alpha*np.identity(self.n, dtype=self.dtype)
        for c, d in zip(self.cs, self.ds):
            Gm += c[:,None]*d[None,:].conj()
        return Gm

    def collapse(self):
        """Collapse the low-rank matrix to a full-rank one."""
        self.collapsed = np.array(self)
        self.cs = None
        self.ds = None
        self.alpha = None

    def restart_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping all vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        if len(self.cs) > rank:
            del self.cs[:]
            del self.ds[:]

    def simple_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping oldest vectors.
        """
        if self.collapsed is not None:
            return
        assert rank > 0
        while len(self.cs) > rank:
            del self.cs[0]
            del self.ds[0]

    def svd_reduce(self, max_rank, to_retain=None):
        """
        Reduce the rank of the matrix by retaining some SVD components.

        This corresponds to the \"Broyden Rank Reduction Inverse\"
        algorithm described in [1]_.

        Note that the SVD decomposition can be done by solving only a
        problem whose size is the effective rank of this matrix, which
        is viable even for large problems.

        Parameters
        ----------
        max_rank : int
            Maximum rank of this matrix after reduction.
        to_retain : int, optional
            Number of SVD components to retain when reduction is done
            (ie. rank > max_rank). Default is ``max_rank - 2``.

        References
        ----------
        .. [1] B.A. van der Rotten, PhD thesis,
           \"A limited memory Broyden method to solve high-dimensional
           systems of nonlinear equations\". Mathematisch Instituut,
           Universiteit Leiden, The Netherlands (2003).

           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

        """
        if self.collapsed is not None:
            return

        p = max_rank
        if to_retain is not None:
            q = to_retain
        else:
            q = p - 2

        if self.cs:
            p = min(p, len(self.cs[0]))
        q = max(0, min(q, p-1))

        m = len(self.cs)
        if m < p:
            # nothing to do
            return

        C = np.array(self.cs).T
        D = np.array(self.ds).T

        D, R = qr(D, mode='economic')
        C = dot(C, R.T.conj())

        U, S, WH = svd(C, full_matrices=False, compute_uv=True)

        C = dot(C, inv(WH))
        D = dot(D, WH.T.conj())

        for k in range(q):
            self.cs[k] = C[:,k].copy()
            self.ds[k] = D[:,k].copy()

        del self.cs[q:]
        del self.ds[q:]


_doc_parts['broyden_params'] = """
    alpha : float, optional
        Initial guess for the Jacobian is ``(-1/alpha)``.
    reduction_method : str or tuple, optional
        Method used in ensuring that the rank of the Broyden matrix
        stays low. Can either be a string giving the name of the method,
        or a tuple of the form ``(method, param1, param2, ...)``
        that gives the name of the method and values for additional parameters.

        Methods available:

            - ``restart``: drop all matrix columns. Has no extra parameters.
            - ``simple``: drop oldest matrix column. Has no extra parameters.
            - ``svd``: keep only the most significant SVD components.
              Takes an extra parameter, ``to_retain``, which determines the
              number of SVD components to retain when rank reduction is done.
              Default is ``max_rank - 2``.

    max_rank : int, optional
        Maximum rank for the Broyden matrix.
        Default is infinity (i.e., no rank reduction).
    """.strip()


class BroydenFirst(GenericBroyden):
    r"""
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as \"Broyden's good method\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='broyden1'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) dx^\dagger H / ( dx^\dagger H df)

    which corresponds to Broyden's first Jacobian update

    .. math:: J_+ = J + (df - J dx) dx^\dagger / dx^\dagger dx


    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       \"A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations\". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden1(fun, [0, 0])
    >>> sol
    array([0.84116396, 0.15883641])

    """

    def __init__(self, alpha=None, reduction_method='restart', max_rank=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.Gm = None

        if max_rank is None:
            max_rank = np.inf
        self.max_rank = max_rank

        if isinstance(reduction_method, str):
            reduce_params = ()
        else:
            reduce_params = reduction_method[1:]
            reduction_method = reduction_method[0]
        reduce_params = (max_rank - 1,) + reduce_params

        if reduction_method == 'svd':
            self._reduce = lambda: self.Gm.svd_reduce(*reduce_params)
        elif reduction_method == 'simple':
            self._reduce = lambda: self.Gm.simple_reduce(*reduce_params)
        elif reduction_method == 'restart':
            self._reduce = lambda: self.Gm.restart_reduce(*reduce_params)
        else:
            raise ValueError("Unknown rank reduction method '%s'" %
                             reduction_method)

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.Gm = LowRankMatrix(-self.alpha, self.shape[0], self.dtype)

    def todense(self):
        return inv(self.Gm)

    def solve(self, f, tol=0):
        r = self.Gm.matvec(f)
        if not np.isfinite(r).all():
            # singular; reset the Jacobian approximation
            self.setup(self.last_x, self.last_f, self.func)
        return self.Gm.matvec(f)

    def matvec(self, f):
        return self.Gm.solve(f)

    def rsolve(self, f, tol=0):
        return self.Gm.rmatvec(f)

    def rmatvec(self, f):
        return self.Gm.rsolve(f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()  # reduce first to preserve secant condition

        v = self.Gm.rmatvec(dx)
        c = dx - self.Gm.matvec(df)
        d = v / vdot(df, v)

        self.Gm.append(c, d)


class BroydenSecond(BroydenFirst):
    """
    Find a root of a function, using Broyden\'s second Jacobian approximation.

    This method is also known as \"Broyden's bad method\".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='broyden2'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) df^\\dagger / ( df^\\dagger df)

    corresponding to Broyden's second method.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       \"A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations\". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden2(fun, [0, 0])
    >>> sol
    array([0.84116365, 0.15883529])

    """

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()  # reduce first to preserve secant condition

        v = df
        c = dx - self.Gm.matvec(df)
        d = v / df_norm**2
        self.Gm.append(c, d)


#------------------------------------------------------------------------------
# Broyden-like (restricted memory)
#------------------------------------------------------------------------------

class Anderson(GenericBroyden):
    """
    Find a root of a function, using (extended) Anderson mixing.

    The Jacobian is formed by for a 'best' solution in the space
    spanned by last `M` vectors. As a result, only a MxM matrix
    inversions and MxN multiplications are required. [Ey]_

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    M : float, optional
        Number of previous vectors to retain. Defaults to 5.
    w0 : float, optional
        Regularization parameter for numerical stability.
        Compared to unity, good values of the order of 0.01.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='anderson'`` in particular.

    References
    ----------
    .. [Ey] V. Eyert, J. Comp. Phys., 124, 271 (1996).

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.anderson(fun, [0, 0])
    >>> sol
    array([0.84116588, 0.15883789])

    """

    # Note:
    #
    # Anderson method maintains a rank M approximation of the inverse Jacobian,
    #
    #     J^-1 v ~ -v*alpha + (dX + alpha dF) A^-1 dF^H v
    #     A      = W + dF^H dF
    #     W      = w0^2 diag(dF^H dF)
    #
    # so that for w0 = 0 the secant condition applies for last M iterates, i.e.,
    #
    #     J^-1 df_j = dx_j
    #
    # for all j = 0 ... M-1.
    #
    # Moreover, (from Sherman-Morrison-Woodbury formula)
    #
    #    J v ~ [ b I - b^2 C (I + b dF^H A^-1 C)^-1 dF^H ] v
    #    C   = (dX + alpha dF) A^-1
    #    b   = -1/alpha
    #
    # and after simplification
    #
    #    J v ~ -v/alpha + (dX/alpha + dF) (dF^H dX - alpha W)^-1 dF^H v
    #

    def __init__(self, alpha=None, w0=0.01, M=5):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.M = M
        self.dx = []
        self.df = []
        self.gamma = None
        self.w0 = w0

    def solve(self, f, tol=0):
        dx = -self.alpha*f

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n, dtype=f.dtype)
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)

        try:
            gamma = solve(self.a, df_f)
        except LinAlgError:
            # singular; reset the Jacobian approximation
            del self.dx[:]
            del self.df[:]
            return dx

        for m in range(n):
            dx += gamma[m]*(self.dx[m] + self.alpha*self.df[m])
        return dx

    def matvec(self, f):
        dx = -f/self.alpha

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = np.empty(n, dtype=f.dtype)
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)

        b = np.empty((n, n), dtype=f.dtype)
        for i in range(n):
            for j in range(n):
                b[i,j] = vdot(self.df[i], self.dx[j])
                if i == j and self.w0 != 0:
                    b[i,j] -= vdot(self.df[i], self.df[i])*self.w0**2*self.alpha
        gamma = solve(b, df_f)

        for m in range(n):
            dx += gamma[m]*(self.df[m] + self.dx[m]/self.alpha)
        return dx

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        if self.M == 0:
            return

        self.dx.append(dx)
        self.df.append(df)

        while len(self.dx) > self.M:
            self.dx.pop(0)
            self.df.pop(0)

        n = len(self.dx)
        a = np.zeros((n, n), dtype=f.dtype)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    wd = self.w0**2
                else:
                    wd = 0
                a[i,j] = (1+wd)*vdot(self.df[i], self.df[j])

        a += np.triu(a, 1).T.conj()
        self.a = a

#------------------------------------------------------------------------------
# Simple iterations
#------------------------------------------------------------------------------


class DiagBroyden(GenericBroyden):
    """
    Find a root of a function, using diagonal Broyden Jacobian approximation.

    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='diagbroyden'`` in particular.

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.diagbroyden(fun, [0, 0])
    >>> sol
    array([0.84116403, 0.15883384])

    """

    def __init__(self, alpha=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.d = np.full((self.shape[0],), 1 / self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f / self.d

    def matvec(self, f):
        return -f * self.d

    def rsolve(self, f, tol=0):
        return -f / self.d.conj()

    def rmatvec(self, f):
        return -f * self.d.conj()

    def todense(self):
        return np.diag(-self.d)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self.d -= (df + self.d*dx)*dx/dx_norm**2


class LinearMixing(GenericBroyden):
    """
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='linearmixing'`` in particular.

    """

    def __init__(self, alpha=None):
        GenericBroyden.__init__(self)
        self.alpha = alpha

    def solve(self, f, tol=0):
        return -f*self.alpha

    def matvec(self, f):
        return -f/self.alpha

    def rsolve(self, f, tol=0):
        return -f*np.conj(self.alpha)

    def rmatvec(self, f):
        return -f/np.conj(self.alpha)

    def todense(self):
        return np.diag(np.full(self.shape[0], -1/self.alpha))

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        pass


class ExcitingMixing(GenericBroyden):
    """
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='excitingmixing'`` in particular.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.
    %(params_extra)s
    """

    def __init__(self, alpha=None, alphamax=1.0):
        GenericBroyden.__init__(self)
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = None

    def setup(self, x, F, func):
        GenericBroyden.setup(self, x, F, func)
        self.beta = np.full((self.shape[0],), self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        return -f*self.beta

    def matvec(self, f):
        return -f/self.beta

    def rsolve(self, f, tol=0):
        return -f*self.beta.conj()

    def rmatvec(self, f):
        return -f/self.beta.conj()

    def todense(self):
        return np.diag(-1/self.beta)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        incr = f*self.last_f > 0
        self.beta[incr] += self.alpha
        self.beta[~incr] = self.alpha
        np.clip(self.beta, 0, self.alphamax, out=self.beta)


#------------------------------------------------------------------------------
# Iterative/Krylov approximated Jacobians
#------------------------------------------------------------------------------

class KrylovJacobian(Jacobian):
    r"""
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : {'lgmres', 'gmres', 'bicgstab', 'cgs', 'minres'} or function
        Krylov method to use to approximate the Jacobian.
        Can be a string, or a function implementing the same interface as
        the iterative solvers in `scipy.sparse.linalg`.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_maxiter : int, optional
        Parameter to pass to the "inner" Krylov solver: maximum number of
        iterations. Iteration will stop after maxiter steps even if the
        specified tolerance has not been achieved.
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,

        >>> from scipy.optimize.nonlin import BroydenFirst, KrylovJacobian
        >>> from scipy.optimize.nonlin import InverseJacobian
        >>> jac = BroydenFirst()
        >>> kjac = KrylovJacobian(inner_M=InverseJacobian(jac))

        If the preconditioner has a method named 'update', it will be called
        as ``update(x, f)`` after each nonlinear step, with ``x`` giving
        the current point, and ``f`` the current function value.
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
    inner_kwargs : kwargs
        Keyword parameters for the "inner" Krylov solver
        (defined with `method`). Parameter names must start with
        the `inner_` prefix which will be stripped before passing on
        the inner method. See, e.g., `scipy.sparse.linalg.gmres` for details.
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method=='krylov'`` in particular.
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are conveniently approximated by a finite difference:

    .. math:: J v \approx (f(x + \omega*v/|v|) - f(x)) / \omega

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    SciPy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [1]_,
    and for the LGMRES sparse inverse method, see [2]_.

    References
    ----------
    .. [1] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2004).
           :doi:`10.1016/j.jcp.2003.08.010`
    .. [2] A.H. Baker and E.R. Jessup and T. Manteuffel,
           SIAM J. Matrix Anal. Appl. 26, 962 (2005).
           :doi:`10.1137/S0895479803422014`

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0] + 0.5 * x[1] - 1.0,
    ...             0.5 * (x[1] - x[0]) ** 2]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.newton_krylov(fun, [0, 0])
    >>> sol
    array([0.66731771, 0.66536458])

    """

    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20,
                 inner_M=None, outer_k=10, **kw):
        self.preconditioner = inner_M
        self.rdiff = rdiff
        self.method = dict(
            bicgstab=scipy.sparse.linalg.bicgstab,
            gmres=scipy.sparse.linalg.gmres,
            lgmres=scipy.sparse.linalg.lgmres,
            cgs=scipy.sparse.linalg.cgs,
            minres=scipy.sparse.linalg.minres,
            ).get(method, method)

        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)

        if self.method is scipy.sparse.linalg.gmres:
            # Replace GMRES's outer iteration with Newton steps
            self.method_kw['restrt'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        elif self.method is scipy.sparse.linalg.gcrotmk:
            self.method_kw.setdefault('atol', 0)
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            # Replace LGMRES's outer iteration with Newton steps
            self.method_kw['maxiter'] = 1
            # Carry LGMRES's `outer_v` vectors across nonlinear iterations
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            # But don't carry the corresponding Jacobian*v products, in case
            # the Jacobian changes a lot in the nonlinear step
            #
            # XXX: some trust-region inspired ideas might be more efficient...
            #      See e.g., Brown & Saad. But needs to be implemented separately
            #      since it's not an inexact Newton method.
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)

        for key, value in kw.items():
            if not key.startswith('inner_'):
                raise ValueError("Unknown parameter %s" % key)
            self.method_kw[key[6:]] = value

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        nv = norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.f0) / sc
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        return r

    def solve(self, rhs, tol=0):
        if 'tol' in self.method_kw:
            sol, info = self.method(self.op, rhs, **self.method_kw)
        else:
            sol, info = self.method(self.op, rhs, tol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()

        # Update also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    def setup(self, x, f, func):
        Jacobian.setup(self, x, f, func)
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)

        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1./2)

        self._update_diff_step()

        # Setup also the preconditioner, if possible
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)


#------------------------------------------------------------------------------
# Wrapper functions
#------------------------------------------------------------------------------

def _nonlin_wrapper(name, jac):
    """
    Construct a solver wrapper with given name and Jacobian approx.

    It inspects the keyword arguments of ``jac.__init__``, and allows to
    use the same arguments in the wrapper function, in addition to the
    keyword arguments of `nonlin_solve`

    """
    signature = _getfullargspec(jac.__init__)
    args, varargs, varkw, defaults, kwonlyargs, kwdefaults, _ = signature
    kwargs = list(zip(args[-len(defaults):], defaults))
    kw_str = ", ".join(["%s=%r" % (k, v) for k, v in kwargs])
    if kw_str:
        kw_str = ", " + kw_str
    kwkw_str = ", ".join(["%s=%s" % (k, k) for k, v in kwargs])
    if kwkw_str:
        kwkw_str = kwkw_str + ", "
    if kwonlyargs:
        raise ValueError('Unexpected signature %s' % signature)

    # Construct the wrapper function so that its keyword arguments
    # are visible in pydoc.help etc.
    wrapper = """
def %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None,
             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
             tol_norm=None, line_search='armijo', callback=None, **kw):
    jac = %(jac)s(%(kwkw)s **kw)
    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,
                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,
                        callback)
"""

    wrapper = wrapper % dict(name=name, kw=kw_str, jac=jac.__name__,
                             kwkw=kwkw_str)
    ns = {}
    ns.update(globals())
    exec(wrapper, ns)
    func = ns[name]
    func.__doc__ = jac.__doc__
    _set_doc(func)
    return func


broyden1 = _nonlin_wrapper('broyden1', BroydenFirst)
broyden2 = _nonlin_wrapper('broyden2', BroydenSecond)
anderson = _nonlin_wrapper('anderson', Anderson)
linearmixing = _nonlin_wrapper('linearmixing', LinearMixing)
diagbroyden = _nonlin_wrapper('diagbroyden', DiagBroyden)
excitingmixing = _nonlin_wrapper('excitingmixing', ExcitingMixing)
newton_krylov = _nonlin_wrapper('newton_krylov', KrylovJacobian)
