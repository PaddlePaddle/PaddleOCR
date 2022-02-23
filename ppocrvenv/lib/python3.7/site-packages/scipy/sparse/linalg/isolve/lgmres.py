# Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as SciPy.

import warnings
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import get_blas_funcs
from .utils import make_system

from ._gcrotmk import _fgmres

__all__ = ['lgmres']


def lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
           prepend_outer_v=False, atol=None):
    """
    Solve a matrix equation using the LGMRES algorithm.

    The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems
    in the convergence in restarted GMRES, and often converges in fewer
    iterations.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0  : {array, matrix}
        Starting guess for the solution.
    tol, atol : float, optional
        Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
        The default for ``atol`` is `tol`.

        .. warning::

           The default value for `atol` will be changed in a future release.
           For future compatibility, specify `atol` explicitly.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, dense matrix, LinearOperator}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    inner_m : int, optional
        Number of inner GMRES iterations per each outer iteration.
    outer_k : int, optional
        Number of vectors to carry between inner GMRES iterations.
        According to [1]_, good values are in the range of 1...3.
        However, note that if you want to use the additional vectors to
        accelerate solving multiple similar problems, larger values may
        be beneficial.
    outer_v : list of tuples, optional
        List containing tuples ``(v, Av)`` of vectors and corresponding
        matrix-vector products, used to augment the Krylov subspace, and
        carried between inner GMRES iterations. The element ``Av`` can
        be `None` if the matrix-vector product should be re-evaluated.
        This parameter is modified in-place by `lgmres`, and can be used
        to pass "guess" vectors in and out of the algorithm when solving
        similar problems.
    store_outer_Av : bool, optional
        Whether LGMRES should store also A*v in addition to vectors `v`
        in the `outer_v` list. Default is True.
    prepend_outer_v : bool, optional 
        Whether to put outer_v augmentation vectors before Krylov iterates.
        In standard LGMRES, prepend_outer_v=False.

    Returns
    -------
    x : array or matrix
        The converged solution.
    info : int
        Provides convergence information:

            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    Notes
    -----
    The LGMRES algorithm [1]_ [2]_ is designed to avoid the
    slowing of convergence in restarted GMRES, due to alternating
    residual vectors. Typically, it often outperforms GMRES(m) of
    comparable memory requirements by some measure, or at least is not
    much worse.

    Another advantage in this algorithm is that you can supply it with
    'guess' vectors in the `outer_v` argument that augment the Krylov
    subspace. If the solution lies close to the span of these vectors,
    the algorithm converges faster. This can be useful if several very
    similar matrices need to be inverted one after another, such as in
    Newton-Krylov iteration where the Jacobian matrix often changes
    little in the nonlinear steps.

    References
    ----------
    .. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, "A Technique for
             Accelerating the Convergence of Restarted GMRES", SIAM J. Matrix
             Anal. Appl. 26, 962 (2005).
    .. [2] A.H. Baker, "On Improving the Performance of the Linear Solver
             restarted GMRES", PhD thesis, University of Colorado (2003).

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import lgmres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = lgmres(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A,M,x,b,postprocess = make_system(A,M,x0,b)

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    if atol is None:
        warnings.warn("scipy.sparse.linalg.lgmres called without specifying `atol`. "
                      "The default value will change in the future. To preserve "
                      "current behavior, set ``atol=tol``.",
                      category=DeprecationWarning, stacklevel=2)
        atol = tol

    matvec = A.matvec
    psolve = M.matvec

    if outer_v is None:
        outer_v = []

    axpy, dot, scal = None, None, None
    nrm2 = get_blas_funcs('nrm2', [b])

    b_norm = nrm2(b)
    ptol_max_factor = 1.0

    for k_outer in range(maxiter):
        r_outer = matvec(x) - b

        # -- callback
        if callback is not None:
            callback(x)

        # -- determine input type routines
        if axpy is None:
            if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
                x = x.astype(r_outer.dtype)
            axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'],
                                                   (x, r_outer))

        # -- check stopping condition
        r_norm = nrm2(r_outer)
        if r_norm <= max(atol, tol * b_norm):
            break

        # -- inner LGMRES iteration
        v0 = -psolve(r_outer)
        inner_res_0 = nrm2(v0)

        if inner_res_0 == 0:
            rnorm = nrm2(r_outer)
            raise RuntimeError("Preconditioner returned a zero vector; "
                               "|v| ~ %.1g, |M v| = 0" % rnorm)

        v0 = scal(1.0/inner_res_0, v0)

        ptol = min(ptol_max_factor, max(atol, tol*b_norm)/r_norm)

        try:
            Q, R, B, vs, zs, y, pres = _fgmres(matvec,
                                               v0,
                                               inner_m,
                                               lpsolve=psolve,
                                               atol=ptol,
                                               outer_v=outer_v,
                                               prepend_outer_v=prepend_outer_v)
            y *= inner_res_0
            if not np.isfinite(y).all():
                # Overflow etc. in computation. There's no way to
                # recover from this, so we have to bail out.
                raise LinAlgError()
        except LinAlgError:
            # Floating point over/underflow, non-finite result from
            # matmul etc. -- report failure.
            return postprocess(x), k_outer + 1

        # Inner loop tolerance control
        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

        # -- GMRES terminated: eval solution
        dx = zs[0]*y[0]
        for w, yc in zip(zs[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc)  # dx += w*yc

        # -- Store LGMRES augmentation vectors
        nx = nrm2(dx)
        if nx > 0:
            if store_outer_Av:
                q = Q.dot(R.dot(y))
                ax = vs[0]*q[0]
                for v, qc in zip(vs[1:], q[1:]):
                    ax = axpy(v, ax, ax.shape[0], qc)
                outer_v.append((dx/nx, ax/nx))
            else:
                outer_v.append((dx/nx, None))

        # -- Retain only a finite number of augmentation vectors
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- Apply step
        x += dx
    else:
        # didn't converge ...
        return postprocess(x), maxiter

    return postprocess(x), 0
