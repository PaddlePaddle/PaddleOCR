# Copyright (C) 2015, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as SciPy.

import warnings
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
from scipy.sparse.linalg.isolve.utils import make_system


__all__ = ['gcrotmk']


def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(),
            prepend_outer_v=False):
    """
    FGMRES Arnoldi process, with optional projection or augmentation

    Parameters
    ----------
    matvec : callable
        Operation A*x
    v0 : ndarray
        Initial vector, normalized to nrm2(v0) == 1
    m : int
        Number of GMRES rounds
    atol : float
        Absolute tolerance for early exit
    lpsolve : callable
        Left preconditioner L
    rpsolve : callable
        Right preconditioner R
    CU : list of (ndarray, ndarray)
        Columns of matrices C and U in GCROT
    outer_v : list of ndarrays
        Augmentation vectors in LGMRES
    prepend_outer_v : bool, optional
        Whether augmentation vectors come before or after 
        Krylov iterates

    Raises
    ------
    LinAlgError
        If nans encountered

    Returns
    -------
    Q, R : ndarray
        QR decomposition of the upper Hessenberg H=QR
    B : ndarray
        Projections corresponding to matrix C
    vs : list of ndarray
        Columns of matrix V
    zs : list of ndarray
        Columns of matrix Z
    y : ndarray
        Solution to ||H y - e_1||_2 = min!
    res : float
        The final (preconditioned) residual norm

    """

    if lpsolve is None:
        lpsolve = lambda x: x
    if rpsolve is None:
        rpsolve = lambda x: x

    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))

    vs = [v0]
    zs = []
    y = None
    res = np.nan

    m = m + len(outer_v)

    # Orthogonal projection coefficients
    B = np.zeros((len(cs), m), dtype=v0.dtype)

    # H is stored in QR factorized form
    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)

    eps = np.finfo(v0.dtype).eps

    breakdown = False

    # FGMRES Arnoldi process
    for j in range(m):
        # L A Z = C B + V H

        if prepend_outer_v and j < len(outer_v):
            z, w = outer_v[j]
        elif prepend_outer_v and j == len(outer_v):
            z = rpsolve(v0)
            w = None
        elif not prepend_outer_v and j >= m - len(outer_v):
            z, w = outer_v[j - (m - len(outer_v))]
        else:
            z = rpsolve(vs[-1])
            w = None

        if w is None:
            w = lpsolve(matvec(z))
        else:
            # w is clobbered below
            w = w.copy()

        w_norm = nrm2(w)

        # GCROT projection: L A -> (1 - C C^H) L A
        # i.e. orthogonalize against C
        for i, c in enumerate(cs):
            alpha = dot(c, w)
            B[i,j] = alpha
            w = axpy(c, w, c.shape[0], -alpha)  # w -= alpha*c

        # Orthogonalize against V
        hcur = np.zeros(j+2, dtype=Q.dtype)
        for i, v in enumerate(vs):
            alpha = dot(v, w)
            hcur[i] = alpha
            w = axpy(v, w, v.shape[0], -alpha)  # w -= alpha*v
        hcur[i+1] = nrm2(w)

        with np.errstate(over='ignore', divide='ignore'):
            # Careful with denormals
            alpha = 1/hcur[-1]

        if np.isfinite(alpha):
            w = scal(alpha, w)

        if not (hcur[-1] > eps * w_norm):
            # w essentially in the span of previous vectors,
            # or we have nans. Bail out after updating the QR
            # solution.
            breakdown = True

        vs.append(w)
        zs.append(z)

        # Arnoldi LSQ problem

        # Add new column to H=Q*R, padding other columns with zeros
        Q2 = np.zeros((j+2, j+2), dtype=Q.dtype, order='F')
        Q2[:j+1,:j+1] = Q
        Q2[j+1,j+1] = 1

        R2 = np.zeros((j+2, j), dtype=R.dtype, order='F')
        R2[:j+1,:] = R

        Q, R = qr_insert(Q2, R2, hcur, j, which='col',
                         overwrite_qru=True, check_finite=False)

        # Transformed least squares problem
        # || Q R y - inner_res_0 * e_1 ||_2 = min!
        # Since R = [R'; 0], solution is y = inner_res_0 (R')^{-1} (Q^H)[:j,0]

        # Residual is immediately known
        res = abs(Q[0,-1])

        # Check for termination
        if res < atol or breakdown:
            break

    if not np.isfinite(R[j,j]):
        # nans encountered, bail out
        raise LinAlgError()

    # -- Get the LSQ problem solution

    # The problem is triangular, but the condition number may be
    # bad (or in case of breakdown the last diagonal entry may be
    # zero), so use lstsq instead of trtrs.
    y, _, _, _, = lstsq(R[:j+1,:j+1], Q[0,:j+1].conj())

    B = B[:,:j+1]

    return Q, R, B, vs, zs, y, res


def gcrotmk(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
            m=20, k=None, CU=None, discard_C=False, truncate='oldest',
            atol=None):
    """
    Solve a matrix equation using flexible GCROT(m,k) algorithm.

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
        inverse of A. gcrotmk is a 'flexible' algorithm and the preconditioner
        can vary from iteration to iteration. Effective preconditioning
        dramatically improves the rate of convergence, which implies that
        fewer iterations are needed to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    m : int, optional
        Number of inner FGMRES iterations per each outer iteration.
        Default: 20
    k : int, optional
        Number of vectors to carry between inner FGMRES iterations.
        According to [2]_, good values are around m.
        Default: m
    CU : list of tuples, optional
        List of tuples ``(c, u)`` which contain the columns of the matrices
        C and U in the GCROT(m,k) algorithm. For details, see [2]_.
        The list given and vectors contained in it are modified in-place.
        If not given, start from empty matrices. The ``c`` elements in the
        tuples can be ``None``, in which case the vectors are recomputed
        via ``c = A u`` on start and orthogonalized as described in [3]_.
    discard_C : bool, optional
        Discard the C-vectors at the end. Useful if recycling Krylov subspaces
        for different linear systems.
    truncate : {'oldest', 'smallest'}, optional
        Truncation scheme to use. Drop: oldest vectors, or vectors with
        smallest singular values using the scheme discussed in [1,2].
        See [2]_ for detailed comparison.
        Default: 'oldest'

    Returns
    -------
    x : array or matrix
        The solution found.
    info : int
        Provides convergence information:

        * 0  : successful exit
        * >0 : convergence to tolerance not achieved, number of iterations

    References
    ----------
    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
           methods'', SIAM J. Numer. Anal. 36, 864 (1999).
    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
           of GCROT for solving nonsymmetric linear systems'',
           SIAM J. Sci. Comput. 32, 172 (2010).
    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
           ''Recycling Krylov subspaces for sequences of linear systems'',
           SIAM J. Sci. Comput. 28, 1651 (2006).

    """
    A,M,x,b,postprocess = make_system(A,M,x0,b)

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    if truncate not in ('oldest', 'smallest'):
        raise ValueError("Invalid value for 'truncate': %r" % (truncate,))

    if atol is None:
        warnings.warn("scipy.sparse.linalg.gcrotmk called without specifying `atol`. "
                      "The default value will change in the future. To preserve "
                      "current behavior, set ``atol=tol``.",
                      category=DeprecationWarning, stacklevel=2)
        atol = tol

    matvec = A.matvec
    psolve = M.matvec

    if CU is None:
        CU = []

    if k is None:
        k = m

    axpy, dot, scal = None, None, None

    r = b - matvec(x)

    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r))

    b_norm = nrm2(b)

    if discard_C:
        CU[:] = [(None, u) for c, u in CU]

    # Reorthogonalize old vectors
    if CU:
        # Sort already existing vectors to the front
        CU.sort(key=lambda cu: cu[0] is not None)

        # Fill-in missing ones
        C = np.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
        us = []
        j = 0
        while CU:
            # More memory-efficient: throw away old vectors as we go
            c, u = CU.pop(0)
            if c is None:
                c = matvec(u)
            C[:,j] = c
            j += 1
            us.append(u)

        # Orthogonalize
        Q, R, P = qr(C, overwrite_a=True, mode='economic', pivoting=True)
        del C

        # C := Q
        cs = list(Q.T)

        # U := U P R^-1,  back-substitution
        new_us = []
        for j in range(len(cs)):
            u = us[P[j]]
            for i in range(j):
                u = axpy(us[P[i]], u, u.shape[0], -R[i,j])
            if abs(R[j,j]) < 1e-12 * abs(R[0,0]):
                # discard rest of the vectors
                break
            u = scal(1.0/R[j,j], u)
            new_us.append(u)

        # Form the new CU lists
        CU[:] = list(zip(cs, new_us))[::-1]

    if CU:
        axpy, dot = get_blas_funcs(['axpy', 'dot'], (r,))

        # Solve first the projection operation with respect to the CU
        # vectors. This corresponds to modifying the initial guess to
        # be
        #
        #     x' = x + U y
        #     y = argmin_y || b - A (x + U y) ||^2
        #
        # The solution is y = C^H (b - A x)
        for c, u in CU:
            yc = dot(c, r)
            x = axpy(u, x, x.shape[0], yc)
            r = axpy(c, r, r.shape[0], -yc)

    # GCROT main iteration
    for j_outer in range(maxiter):
        # -- callback
        if callback is not None:
            callback(x)

        beta = nrm2(r)

        # -- check stopping condition
        beta_tol = max(atol, tol * b_norm)

        if beta <= beta_tol and (j_outer > 0 or CU):
            # recompute residual to avoid rounding error
            r = b - matvec(x)
            beta = nrm2(r)

        if beta <= beta_tol:
            j_outer = -1
            break

        ml = m + max(k - len(CU), 0)

        cs = [c for c, u in CU]

        try:
            Q, R, B, vs, zs, y, pres = _fgmres(matvec,
                                               r/beta,
                                               ml,
                                               rpsolve=psolve,
                                               atol=max(atol, tol*b_norm)/beta,
                                               cs=cs)
            y *= beta
        except LinAlgError:
            # Floating point over/underflow, non-finite result from
            # matmul etc. -- report failure.
            break

        #
        # At this point,
        #
        #     [A U, A Z] = [C, V] G;   G =  [ I  B ]
        #                                   [ 0  H ]
        #
        # where [C, V] has orthonormal columns, and r = beta v_0. Moreover,
        #
        #     || b - A (x + Z y + U q) ||_2 = || r - C B y - V H y - C q ||_2 = min!
        #
        # from which y = argmin_y || beta e_1 - H y ||_2, and q = -B y
        #

        #
        # GCROT(m,k) update
        #

        # Define new outer vectors

        # ux := (Z - U B) y
        ux = zs[0]*y[0]
        for z, yc in zip(zs[1:], y[1:]):
            ux = axpy(z, ux, ux.shape[0], yc)  # ux += z*yc
        by = B.dot(y)
        for cu, byc in zip(CU, by):
            c, u = cu
            ux = axpy(u, ux, ux.shape[0], -byc)  # ux -= u*byc

        # cx := V H y
        hy = Q.dot(R.dot(y))
        cx = vs[0] * hy[0]
        for v, hyc in zip(vs[1:], hy[1:]):
            cx = axpy(v, cx, cx.shape[0], hyc)  # cx += v*hyc

        # Normalize cx, maintaining cx = A ux
        # This new cx is orthogonal to the previous C, by construction
        try:
            alpha = 1/nrm2(cx)
            if not np.isfinite(alpha):
                raise FloatingPointError()
        except (FloatingPointError, ZeroDivisionError):
            # Cannot update, so skip it
            continue

        cx = scal(alpha, cx)
        ux = scal(alpha, ux)

        # Update residual and solution
        gamma = dot(cx, r)
        r = axpy(cx, r, r.shape[0], -gamma)  # r -= gamma*cx
        x = axpy(ux, x, x.shape[0], gamma)  # x += gamma*ux

        # Truncate CU
        if truncate == 'oldest':
            while len(CU) >= k and CU:
                del CU[0]
        elif truncate == 'smallest':
            if len(CU) >= k and CU:
                # cf. [1,2]
                D = solve(R[:-1,:].T, B.T).T
                W, sigma, V = svd(D)

                # C := C W[:,:k-1],  U := U W[:,:k-1]
                new_CU = []
                for j, w in enumerate(W[:,:k-1].T):
                    c, u = CU[0]
                    c = c * w[0]
                    u = u * w[0]
                    for cup, wp in zip(CU[1:], w[1:]):
                        cp, up = cup
                        c = axpy(cp, c, c.shape[0], wp)
                        u = axpy(up, u, u.shape[0], wp)

                    # Reorthogonalize at the same time; not necessary
                    # in exact arithmetic, but floating point error
                    # tends to accumulate here
                    for cp, up in new_CU:
                        alpha = dot(cp, c)
                        c = axpy(cp, c, c.shape[0], -alpha)
                        u = axpy(up, u, u.shape[0], -alpha)
                    alpha = nrm2(c)
                    c = scal(1.0/alpha, c)
                    u = scal(1.0/alpha, u)

                    new_CU.append((c, u))
                CU[:] = new_CU

        # Add new vector to CU
        CU.append((cx, ux))

    # Include the solution vector to the span
    CU.append((None, x.copy()))
    if discard_C:
        CU[:] = [(None, uz) for cz, uz in CU]

    return postprocess(x), j_outer + 1
