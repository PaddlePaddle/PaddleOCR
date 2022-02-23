from numpy import sqrt, inner, zeros, inf, finfo
from numpy.linalg import norm

from .utils import make_system

__all__ = ['minres']


def minres(A, b, x0=None, shift=0.0, tol=1e-5, maxiter=None,
           M=None, callback=None, show=False, check=False):
    """
    Use MINimum RESidual iteration to solve Ax=b

    MINRES minimizes norm(A*x - b) for a real symmetric matrix A.  Unlike
    the Conjugate Gradient method, A can be indefinite or singular.

    If shift != 0 then the method solves (A - shift*I)x = b

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real symmetric N-by-N matrix of the linear system
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : {array, matrix}
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    Other Parameters
    ----------------
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below `tol`.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, dense matrix, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import minres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> A = A + A.T
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = minres(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    References
    ----------
    Solution of sparse indefinite systems of linear equations,
        C. C. Paige and M. A. Saunders (1975),
        SIAM J. Numer. Anal. 12(4), pp. 617-629.
        https://web.stanford.edu/group/SOL/software/minres/

    This file is a translation of the following MATLAB implementation:
        https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    matvec = A.matvec
    psolve = M.matvec

    first = 'Enter minres.   '
    last = 'Exit  minres.   '

    n = A.shape[0]

    if maxiter is None:
        maxiter = 5 * n

    msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ',   # -1
            ' beta1 = 0.  The exact solution is x0          ',   # 0
            ' A solution to Ax = b was found, given rtol        ',   # 1
            ' A least-squares solution was found, given rtol    ',   # 2
            ' Reasonable accuracy achieved, given eps           ',   # 3
            ' x has converged to an eigenvector                 ',   # 4
            ' acond has exceeded 0.1/eps                        ',   # 5
            ' The iteration limit was reached                   ',   # 6
            ' A  does not define a symmetric matrix             ',   # 7
            ' M  does not define a symmetric matrix             ',   # 8
            ' M  does not define a pos-def preconditioner       ']   # 9

    if show:
        print(first + 'Solution of symmetric Ax = b')
        print(first + 'n      =  %3g     shift  =  %23.14e' % (n,shift))
        print(first + 'itnlim =  %3g     rtol   =  %11.2e' % (maxiter,tol))
        print()

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = x.dtype

    eps = finfo(xtype).eps

    # Set up y and v for the first Lanczos vector v1.
    # y  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.

    r1 = b - A*x
    y = psolve(r1)

    beta1 = inner(r1, y)

    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return (postprocess(x), 0)

    beta1 = sqrt(beta1)

    if check:
        # are these too strict?

        # see if A is symmetric
        w = matvec(y)
        r2 = matvec(w)
        s = inner(w,w)
        t = inner(y,r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            raise ValueError('non-symmetric matrix')

        # see if M is symmetric
        r2 = psolve(y)
        s = inner(y,y)
        t = inner(r1,r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            raise ValueError('non-symmetric preconditioner')

    # Initialize other quantities
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = finfo(xtype).max
    cs = -1
    sn = 0
    w = zeros(n, dtype=xtype)
    w2 = zeros(n, dtype=xtype)
    r2 = r1

    if show:
        print()
        print()
        print('   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')

    while itn < maxiter:
        itn += 1

        s = 1.0/beta
        v = s*y

        y = matvec(v)
        y = y - shift * v

        if itn >= 2:
            y = y - (beta/oldb)*r1

        alfa = inner(v,y)
        y = y - (alfa/beta)*r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = inner(r2,y)
        if beta < 0:
            raise ValueError('non-symmetric matrix')
        beta = sqrt(beta)
        tnorm2 += alfa**2 + oldb**2 + beta**2

        if itn == 1:
            if beta/beta1 <= 10*eps:
                istop = -1  # Terminate later

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
        epsln = sn * beta     # epsln2 = 0         epslnk+1
        dbar = - cs * beta   # dbar 2 = beta2     dbar k+1
        root = norm([gbar, dbar])
        Arnorm = phibar * root

        # Compute the next plane rotation Qk

        gamma = norm([gbar, beta])       # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma             # ck
        sn = beta / gamma             # sk
        phi = cs * phibar              # phik
        phibar = sn * phibar              # phibark+1

        # Update  x.

        denom = 1.0/gamma
        w1 = w2
        w2 = w
        w = (v - oldeps*w1 - delta*w2) * denom
        x = x + phi*w

        # Go round again.

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta*z
        rhs2 = - epsln*z

        # Estimate various norms and test for convergence.

        Anorm = sqrt(tnorm2)
        ynorm = norm(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * tol
        diag = gbar

        if diag == 0:
            diag = epsa

        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = inf
        else:
            test1 = rnorm / (Anorm*ynorm)    # ||r||  / (||A|| ||x||)
        if Anorm == 0:
            test2 = inf
        else:
            test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q * H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax/gmin

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        if istop == 0:
            t1 = 1 + test1      # These tests work if tol < eps
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1/eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            # if rnorm <= epsx   : istop = 2
            # if rnorm <= epsr   : istop = 1
            if test2 <= tol:
                istop = 2
            if test1 <= tol:
                istop = 1

        # See if it is time to print something.

        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= maxiter-10:
            prnt = True
        if itn % 10 == 0:
            prnt = True
        if qrnorm <= 10*epsx:
            prnt = True
        if qrnorm <= 10*epsr:
            prnt = True
        if Acond <= 1e-2/eps:
            prnt = True
        if istop != 0:
            prnt = True

        if show and prnt:
            str1 = '%6g %12.5e %10.3e' % (itn, x[0], test1)
            str2 = ' %10.3e' % (test2,)
            str3 = ' %8.1e %8.1e %8.1e' % (Anorm, Acond, gbar/Anorm)

            print(str1 + str2 + str3)

            if itn % 10 == 0:
                print()

        if callback is not None:
            callback(x)

        if istop != 0:
            break  # TODO check this

    if show:
        print()
        print(last + ' istop   =  %3g               itn   =%5g' % (istop,itn))
        print(last + ' Anorm   =  %12.4e      Acond =  %12.4e' % (Anorm,Acond))
        print(last + ' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm,ynorm))
        print(last + ' Arnorm  =  %12.4e' % (Arnorm,))
        print(last + msg[istop+1])

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return (postprocess(x),info)


if __name__ == '__main__':
    from numpy import arange
    from scipy.sparse import spdiags

    n = 10

    residuals = []

    def cb(x):
        residuals.append(norm(b - A*x))

    # A = poisson((10,),format='csr')
    A = spdiags([arange(1,n+1,dtype=float)], [0], n, n, format='csr')
    M = spdiags([1.0/arange(1,n+1,dtype=float)], [0], n, n, format='csr')
    A.psolve = M.matvec
    b = zeros(A.shape[0])
    x = minres(A,b,tol=1e-12,maxiter=None,callback=cb)
    # x = cg(A,b,x0=b,tol=1e-12,maxiter=None,callback=cb)[0]
