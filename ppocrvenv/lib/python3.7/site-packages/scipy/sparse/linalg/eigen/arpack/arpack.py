"""
Find a few eigenvectors and eigenvalues of a matrix.


Uses ARPACK: http://www.caam.rice.edu/software/ARPACK/

"""
# Wrapper implementation notes
#
# ARPACK Entry Points
# -------------------
# The entry points to ARPACK are
# - (s,d)seupd : single and double precision symmetric matrix
# - (s,d,c,z)neupd: single,double,complex,double complex general matrix
# This wrapper puts the *neupd (general matrix) interfaces in eigs()
# and the *seupd (symmetric matrix) in eigsh().
# There is no specialized interface for complex Hermitian matrices.
# To find eigenvalues of a complex Hermitian matrix you
# may use eigsh(), but eigsh() will simply call eigs()
# and return the real part of the eigenvalues thus obtained.

# Number of eigenvalues returned and complex eigenvalues
# ------------------------------------------------------
# The ARPACK nonsymmetric real and double interface (s,d)naupd return
# eigenvalues and eigenvectors in real (float,double) arrays.
# Since the eigenvalues and eigenvectors are, in general, complex
# ARPACK puts the real and imaginary parts in consecutive entries
# in real-valued arrays.   This wrapper puts the real entries
# into complex data types and attempts to return the requested eigenvalues
# and eigenvectors.


# Solver modes
# ------------
# ARPACK and handle shifted and shift-inverse computations
# for eigenvalues by providing a shift (sigma) and a solver.

__docformat__ = "restructuredtext en"

__all__ = ['eigs', 'eigsh', 'svds', 'ArpackError', 'ArpackNoConvergence']

from . import _arpack
arpack_int = _arpack.timing.nbx.dtype

import numpy as np
import warnings
from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse, isspmatrix, isspmatrix_csr
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse.sputils import isdense, is_pydata_spmatrix
from scipy.sparse.linalg import gmres, splu
from scipy.sparse.linalg.eigen.lobpcg import lobpcg
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock


_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}
_ndigits = {'f': 5, 'd': 12, 'F': 5, 'D': 12}

DNAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found. IPARAM(5) "
       "returns the number of wanted converged Ritz values.",
    2: "No longer an informational error. Deprecated starting "
       "with release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the "
       "Implicitly restarted Arnoldi iteration. One possibility "
       "is to increase the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 2 and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: " WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from LAPACK eigenvalue calculation;",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible.",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated."
}

SNAUPD_ERRORS = DNAUPD_ERRORS

ZNAUPD_ERRORS = DNAUPD_ERRORS.copy()
ZNAUPD_ERRORS[-10] = "IPARAM(7) must be 1,2,3."

CNAUPD_ERRORS = ZNAUPD_ERRORS

DSAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found.",
    2: "No longer an informational error. Deprecated starting with "
       "release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the Implicitly "
       "restarted Arnoldi iteration. One possibility is to increase "
       "the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from trid. eigenvalue calculation; "
        "Informational error from LAPACK routine dsteqr .",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible. ",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated.",
}

SSAUPD_ERRORS = DSAUPD_ERRORS

DNEUPD_ERRORS = {
    0: "Normal exit.",
    1: "The Schur form computed by LAPACK routine dlahqr "
       "could not be reordered by LAPACK routine dtrsen. "
       "Re-enter subroutine dneupd  with IPARAM(5)NCV and "
       "increase the size of the arrays DR and DI to have "
       "dimension at least dimension NCV and allocate at least NCV "
       "columns for Z. NOTE: Not necessary if Z and V share "
       "the same space. Please notify the authors if this error"
       "occurs.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 2 and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: "Error return from calculation of a real Schur form. "
        "Informational error from LAPACK routine dlahqr .",
    -9: "Error return from calculation of eigenvectors. "
        "Informational error from LAPACK routine dtrevc.",
    -10: "IPARAM(7) must be 1,2,3,4.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "HOWMNY = 'S' not yet implemented",
    -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
    -14: "DNAUPD  did not find any eigenvalues to sufficient "
         "accuracy.",
    -15: "DNEUPD got a different count of the number of converged "
         "Ritz values than DNAUPD got.  This indicates the user "
         "probably made an error in passing data from DNAUPD to "
         "DNEUPD or that the data was modified before entering "
         "DNEUPD",
}

SNEUPD_ERRORS = DNEUPD_ERRORS.copy()
SNEUPD_ERRORS[1] = ("The Schur form computed by LAPACK routine slahqr "
                    "could not be reordered by LAPACK routine strsen . "
                    "Re-enter subroutine dneupd  with IPARAM(5)=NCV and "
                    "increase the size of the arrays DR and DI to have "
                    "dimension at least dimension NCV and allocate at least "
                    "NCV columns for Z. NOTE: Not necessary if Z and V share "
                    "the same space. Please notify the authors if this error "
                    "occurs.")
SNEUPD_ERRORS[-14] = ("SNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.")
SNEUPD_ERRORS[-15] = ("SNEUPD got a different count of the number of "
                      "converged Ritz values than SNAUPD got.  This indicates "
                      "the user probably made an error in passing data from "
                      "SNAUPD to SNEUPD or that the data was modified before "
                      "entering SNEUPD")

ZNEUPD_ERRORS = {0: "Normal exit.",
                 1: "The Schur form computed by LAPACK routine csheqr "
                    "could not be reordered by LAPACK routine ztrsen. "
                    "Re-enter subroutine zneupd with IPARAM(5)=NCV and "
                    "increase the size of the array D to have "
                    "dimension at least dimension NCV and allocate at least "
                    "NCV columns for Z. NOTE: Not necessary if Z and V share "
                    "the same space. Please notify the authors if this error "
                    "occurs.",
                 -1: "N must be positive.",
                 -2: "NEV must be positive.",
                 -3: "NCV-NEV >= 1 and less than or equal to N.",
                 -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
                 -6: "BMAT must be one of 'I' or 'G'.",
                 -7: "Length of private work WORKL array is not sufficient.",
                 -8: "Error return from LAPACK eigenvalue calculation. "
                     "This should never happened.",
                 -9: "Error return from calculation of eigenvectors. "
                     "Informational error from LAPACK routine ztrevc.",
                 -10: "IPARAM(7) must be 1,2,3",
                 -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
                 -12: "HOWMNY = 'S' not yet implemented",
                 -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
                 -14: "ZNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.",
                 -15: "ZNEUPD got a different count of the number of "
                      "converged Ritz values than ZNAUPD got.  This "
                      "indicates the user probably made an error in passing "
                      "data from ZNAUPD to ZNEUPD or that the data was "
                      "modified before entering ZNEUPD"
                 }

CNEUPD_ERRORS = ZNEUPD_ERRORS.copy()
CNEUPD_ERRORS[-14] = ("CNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.")
CNEUPD_ERRORS[-15] = ("CNEUPD got a different count of the number of "
                      "converged Ritz values than CNAUPD got.  This indicates "
                      "the user probably made an error in passing data from "
                      "CNAUPD to CNEUPD or that the data was modified before "
                      "entering CNEUPD")

DSEUPD_ERRORS = {
    0: "Normal exit.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: ("Error return from trid. eigenvalue calculation; "
         "Information error from LAPACK routine dsteqr."),
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "NEV and WHICH = 'BE' are incompatible.",
    -14: "DSAUPD  did not find any eigenvalues to sufficient accuracy.",
    -15: "HOWMNY must be one of 'A' or 'S' if RVEC = .true.",
    -16: "HOWMNY = 'S' not yet implemented",
    -17: ("DSEUPD  got a different count of the number of converged "
          "Ritz values than DSAUPD  got.  This indicates the user "
          "probably made an error in passing data from DSAUPD  to "
          "DSEUPD  or that the data was modified before entering  "
          "DSEUPD.")
}

SSEUPD_ERRORS = DSEUPD_ERRORS.copy()
SSEUPD_ERRORS[-14] = ("SSAUPD  did not find any eigenvalues "
                      "to sufficient accuracy.")
SSEUPD_ERRORS[-17] = ("SSEUPD  got a different count of the number of "
                      "converged "
                      "Ritz values than SSAUPD  got.  This indicates the user "
                      "probably made an error in passing data from SSAUPD  to "
                      "SSEUPD  or that the data was modified before entering  "
                      "SSEUPD.")

_SAUPD_ERRORS = {'d': DSAUPD_ERRORS,
                 's': SSAUPD_ERRORS}
_NAUPD_ERRORS = {'d': DNAUPD_ERRORS,
                 's': SNAUPD_ERRORS,
                 'z': ZNAUPD_ERRORS,
                 'c': CNAUPD_ERRORS}
_SEUPD_ERRORS = {'d': DSEUPD_ERRORS,
                 's': SSEUPD_ERRORS}
_NEUPD_ERRORS = {'d': DNEUPD_ERRORS,
                 's': SNEUPD_ERRORS,
                 'z': ZNEUPD_ERRORS,
                 'c': CNEUPD_ERRORS}

# accepted values of parameter WHICH in _SEUPD
_SEUPD_WHICH = ['LM', 'SM', 'LA', 'SA', 'BE']

# accepted values of parameter WHICH in _NAUPD
_NEUPD_WHICH = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']


class ArpackError(RuntimeError):
    """
    ARPACK error
    """
    def __init__(self, info, infodict=_NAUPD_ERRORS):
        msg = infodict.get(info, "Unknown error")
        RuntimeError.__init__(self, "ARPACK error %d: %s" % (info, msg))


class ArpackNoConvergence(ArpackError):
    """
    ARPACK iteration did not converge

    Attributes
    ----------
    eigenvalues : ndarray
        Partial result. Converged eigenvalues.
    eigenvectors : ndarray
        Partial result. Converged eigenvectors.

    """
    def __init__(self, msg, eigenvalues, eigenvectors):
        ArpackError.__init__(self, -1, {-1: msg})
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors


def choose_ncv(k):
    """
    Choose number of lanczos vectors based on target number
    of singular/eigen values and vectors to compute, k.
    """
    return max(2 * k + 1, 20)


class _ArpackParams:
    def __init__(self, n, k, tp, mode=1, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        if k <= 0:
            raise ValueError("k must be positive, k=%d" % k)

        if maxiter is None:
            maxiter = n * 10
        if maxiter <= 0:
            raise ValueError("maxiter must be positive, maxiter=%d" % maxiter)

        if tp not in 'fdFD':
            raise ValueError("matrix type must be 'f', 'd', 'F', or 'D'")

        if v0 is not None:
            # ARPACK overwrites its initial resid,  make a copy
            self.resid = np.array(v0, copy=True)
            info = 1
        else:
            # ARPACK will use a random initial vector.
            self.resid = np.zeros(n, tp)
            info = 0

        if sigma is None:
            #sigma not used
            self.sigma = 0
        else:
            self.sigma = sigma

        if ncv is None:
            ncv = choose_ncv(k)
        ncv = min(ncv, n)

        self.v = np.zeros((n, ncv), tp)  # holds Ritz vectors
        self.iparam = np.zeros(11, arpack_int)

        # set solver mode and parameters
        ishfts = 1
        self.mode = mode
        self.iparam[0] = ishfts
        self.iparam[2] = maxiter
        self.iparam[3] = 1
        self.iparam[6] = mode

        self.n = n
        self.tol = tol
        self.k = k
        self.maxiter = maxiter
        self.ncv = ncv
        self.which = which
        self.tp = tp
        self.info = info

        self.converged = False
        self.ido = 0

    def _raise_no_convergence(self):
        msg = "No convergence (%d iterations, %d/%d eigenvectors converged)"
        k_ok = self.iparam[4]
        num_iter = self.iparam[2]
        try:
            ev, vec = self.extract(True)
        except ArpackError as err:
            msg = "%s [%s]" % (msg, err)
            ev = np.zeros((0,))
            vec = np.zeros((self.n, 0))
            k_ok = 0
        raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)


class _SymmetricArpackParams(_ArpackParams):
    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None,
                 Minv_matvec=None, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        # The following modes are supported:
        #  mode = 1:
        #    Solve the standard eigenvalue problem:
        #      A*x = lambda*x :
        #       A - symmetric
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = None [not used]
        #       Minv_matvec = None [not used]
        #
        #  mode = 2:
        #    Solve the general eigenvalue problem:
        #      A*x = lambda*M*x
        #       A - symmetric
        #       M - symmetric positive definite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = left multiplication by M
        #       Minv_matvec = left multiplication by M^-1
        #
        #  mode = 3:
        #    Solve the general eigenvalue problem in shift-invert mode:
        #      A*x = lambda*M*x
        #       A - symmetric
        #       M - symmetric positive semi-definite
        #    Arguments should be
        #       matvec      = None [not used]
        #       M_matvec    = left multiplication by M
        #                     or None, if M is the identity
        #       Minv_matvec = left multiplication by [A-sigma*M]^-1
        #
        #  mode = 4:
        #    Solve the general eigenvalue problem in Buckling mode:
        #      A*x = lambda*AG*x
        #       A  - symmetric positive semi-definite
        #       AG - symmetric indefinite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = None [not used]
        #       Minv_matvec = left multiplication by [A-sigma*AG]^-1
        #
        #  mode = 5:
        #    Solve the general eigenvalue problem in Cayley-transformed mode:
        #      A*x = lambda*M*x
        #       A - symmetric
        #       M - symmetric positive semi-definite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = left multiplication by M
        #                     or None, if M is the identity
        #       Minv_matvec = left multiplication by [A-sigma*M]^-1
        if mode == 1:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=1")
            if M_matvec is not None:
                raise ValueError("M_matvec cannot be specified for mode=1")
            if Minv_matvec is not None:
                raise ValueError("Minv_matvec cannot be specified for mode=1")

            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=2")
            if M_matvec is None:
                raise ValueError("M_matvec must be specified for mode=2")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=2")

            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode == 3:
            if matvec is not None:
                raise ValueError("matvec must not be specified for mode=3")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=3")

            if M_matvec is None:
                self.OP = Minv_matvec
                self.OPa = Minv_matvec
                self.B = lambda x: x
                self.bmat = 'I'
            else:
                self.OP = lambda x: Minv_matvec(M_matvec(x))
                self.OPa = Minv_matvec
                self.B = M_matvec
                self.bmat = 'G'
        elif mode == 4:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=4")
            if M_matvec is not None:
                raise ValueError("M_matvec must not be specified for mode=4")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=4")
            self.OPa = Minv_matvec
            self.OP = lambda x: self.OPa(matvec(x))
            self.B = matvec
            self.bmat = 'G'
        elif mode == 5:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=5")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=5")

            self.OPa = Minv_matvec
            self.A_matvec = matvec

            if M_matvec is None:
                self.OP = lambda x: Minv_matvec(matvec(x) + sigma * x)
                self.B = lambda x: x
                self.bmat = 'I'
            else:
                self.OP = lambda x: Minv_matvec(matvec(x)
                                                + sigma * M_matvec(x))
                self.B = M_matvec
                self.bmat = 'G'
        else:
            raise ValueError("mode=%i not implemented" % mode)

        if which not in _SEUPD_WHICH:
            raise ValueError("which must be one of %s"
                             % ' '.join(_SEUPD_WHICH))
        if k >= n:
            raise ValueError("k must be less than ndim(A), k=%d" % k)

        _ArpackParams.__init__(self, n, k, tp, mode, sigma,
                               ncv, v0, maxiter, which, tol)

        if self.ncv > n or self.ncv <= k:
            raise ValueError("ncv must be k<ncv<=n, ncv=%s" % self.ncv)

        # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(self.ncv * (self.ncv + 8), self.tp)

        ltr = _type_conv[self.tp]
        if ltr not in ["s", "d"]:
            raise ValueError("Input matrix is not real-valued.")

        self._arpack_solver = _arpack.__dict__[ltr + 'saupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'seupd']

        self.iterate_infodict = _SAUPD_ERRORS[ltr]
        self.extract_infodict = _SEUPD_ERRORS[ltr]

        self.ipntr = np.zeros(11, arpack_int)

    def iterate(self):
        self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info = \
            self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                self.tol, self.resid, self.v, self.iparam,
                                self.ipntr, self.workd, self.workl, self.info)

        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            # initialization
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            # compute y = Op*x
            if self.mode == 1:
                self.workd[yslice] = self.OP(self.workd[xslice])
            elif self.mode == 2:
                self.workd[xslice] = self.OPb(self.workd[xslice])
                self.workd[yslice] = self.OPa(self.workd[xslice])
            elif self.mode == 5:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                Ax = self.A_matvec(self.workd[xslice])
                self.workd[yslice] = self.OPa(Ax + (self.sigma *
                                                    self.workd[Bxslice]))
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
        else:
            self.converged = True

            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        rvec = return_eigenvectors
        ierr = 0
        howmny = 'A'  # return all eigenvectors
        sselect = np.zeros(self.ncv, 'int')  # unused
        d, z, ierr = self._arpack_extract(rvec, howmny, sselect, self.sigma,
                                          self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v,
                                          self.iparam[0:7], self.ipntr,
                                          self.workd[0:2 * self.n],
                                          self.workl, ierr)
        if ierr != 0:
            raise ArpackError(ierr, infodict=self.extract_infodict)
        k_ok = self.iparam[4]
        d = d[:k_ok]
        z = z[:, :k_ok]

        if return_eigenvectors:
            return d, z
        else:
            return d


class _UnsymmetricArpackParams(_ArpackParams):
    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None,
                 Minv_matvec=None, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        # The following modes are supported:
        #  mode = 1:
        #    Solve the standard eigenvalue problem:
        #      A*x = lambda*x
        #       A - square matrix
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = None [not used]
        #       Minv_matvec = None [not used]
        #
        #  mode = 2:
        #    Solve the generalized eigenvalue problem:
        #      A*x = lambda*M*x
        #       A - square matrix
        #       M - symmetric, positive semi-definite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = left multiplication by M
        #       Minv_matvec = left multiplication by M^-1
        #
        #  mode = 3,4:
        #    Solve the general eigenvalue problem in shift-invert mode:
        #      A*x = lambda*M*x
        #       A - square matrix
        #       M - symmetric, positive semi-definite
        #    Arguments should be
        #       matvec      = None [not used]
        #       M_matvec    = left multiplication by M
        #                     or None, if M is the identity
        #       Minv_matvec = left multiplication by [A-sigma*M]^-1
        #    if A is real and mode==3, use the real part of Minv_matvec
        #    if A is real and mode==4, use the imag part of Minv_matvec
        #    if A is complex and mode==3,
        #       use real and imag parts of Minv_matvec
        if mode == 1:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=1")
            if M_matvec is not None:
                raise ValueError("M_matvec cannot be specified for mode=1")
            if Minv_matvec is not None:
                raise ValueError("Minv_matvec cannot be specified for mode=1")

            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=2")
            if M_matvec is None:
                raise ValueError("M_matvec must be specified for mode=2")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=2")

            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode in (3, 4):
            if matvec is None:
                raise ValueError("matvec must be specified "
                                 "for mode in (3,4)")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified "
                                 "for mode in (3,4)")

            self.matvec = matvec
            if tp in 'DF':  # complex type
                if mode == 3:
                    self.OPa = Minv_matvec
                else:
                    raise ValueError("mode=4 invalid for complex A")
            else:  # real type
                if mode == 3:
                    self.OPa = lambda x: np.real(Minv_matvec(x))
                else:
                    self.OPa = lambda x: np.imag(Minv_matvec(x))
            if M_matvec is None:
                self.B = lambda x: x
                self.bmat = 'I'
                self.OP = self.OPa
            else:
                self.B = M_matvec
                self.bmat = 'G'
                self.OP = lambda x: self.OPa(M_matvec(x))
        else:
            raise ValueError("mode=%i not implemented" % mode)

        if which not in _NEUPD_WHICH:
            raise ValueError("Parameter which must be one of %s"
                             % ' '.join(_NEUPD_WHICH))
        if k >= n - 1:
            raise ValueError("k must be less than ndim(A)-1, k=%d" % k)

        _ArpackParams.__init__(self, n, k, tp, mode, sigma,
                               ncv, v0, maxiter, which, tol)

        if self.ncv > n or self.ncv <= k + 1:
            raise ValueError("ncv must be k+1<ncv<=n, ncv=%s" % self.ncv)

        # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(3 * self.ncv * (self.ncv + 2), self.tp)

        ltr = _type_conv[self.tp]
        self._arpack_solver = _arpack.__dict__[ltr + 'naupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'neupd']

        self.iterate_infodict = _NAUPD_ERRORS[ltr]
        self.extract_infodict = _NEUPD_ERRORS[ltr]

        self.ipntr = np.zeros(14, arpack_int)

        if self.tp in 'FD':
            # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
            self.rwork = _aligned_zeros(self.ncv, self.tp.lower())
        else:
            self.rwork = None

    def iterate(self):
        if self.tp in 'fd':
            self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info =\
                self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                    self.tol, self.resid, self.v, self.iparam,
                                    self.ipntr, self.workd, self.workl,
                                    self.info)
        else:
            self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info =\
                self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                    self.tol, self.resid, self.v, self.iparam,
                                    self.ipntr, self.workd, self.workl,
                                    self.rwork, self.info)

        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            # initialization
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            # compute y = Op*x
            if self.mode in (1, 2):
                self.workd[yslice] = self.OP(self.workd[xslice])
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
        else:
            self.converged = True

            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        k, n = self.k, self.n

        ierr = 0
        howmny = 'A'  # return all eigenvectors
        sselect = np.zeros(self.ncv, 'int')  # unused
        sigmar = np.real(self.sigma)
        sigmai = np.imag(self.sigma)
        workev = np.zeros(3 * self.ncv, self.tp)

        if self.tp in 'fd':
            dr = np.zeros(k + 1, self.tp)
            di = np.zeros(k + 1, self.tp)
            zr = np.zeros((n, k + 1), self.tp)
            dr, di, zr, ierr = \
                self._arpack_extract(return_eigenvectors,
                       howmny, sselect, sigmar, sigmai, workev,
                       self.bmat, self.which, k, self.tol, self.resid,
                       self.v, self.iparam, self.ipntr,
                       self.workd, self.workl, self.info)
            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)
            nreturned = self.iparam[4]  # number of good eigenvalues returned

            # Build complex eigenvalues from real and imaginary parts
            d = dr + 1.0j * di

            # Arrange the eigenvectors: complex eigenvectors are stored as
            # real,imaginary in consecutive columns
            z = zr.astype(self.tp.upper())

            # The ARPACK nonsymmetric real and double interface (s,d)naupd
            # return eigenvalues and eigenvectors in real (float,double)
            # arrays.

            # Efficiency: this should check that return_eigenvectors == True
            #  before going through this construction.
            if sigmai == 0:
                i = 0
                while i <= k:
                    # check if complex
                    if abs(d[i].imag) != 0:
                        # this is a complex conjugate pair with eigenvalues
                        # in consecutive columns
                        if i < k:
                            z[:, i] = zr[:, i] + 1.0j * zr[:, i + 1]
                            z[:, i + 1] = z[:, i].conjugate()
                            i += 1
                        else:
                            #last eigenvalue is complex: the imaginary part of
                            # the eigenvector has not been returned
                            #this can only happen if nreturned > k, so we'll
                            # throw out this case.
                            nreturned -= 1
                    i += 1

            else:
                # real matrix, mode 3 or 4, imag(sigma) is nonzero:
                # see remark 3 in <s,d>neupd.f
                # Build complex eigenvalues from real and imaginary parts
                i = 0
                while i <= k:
                    if abs(d[i].imag) == 0:
                        d[i] = np.dot(zr[:, i], self.matvec(zr[:, i]))
                    else:
                        if i < k:
                            z[:, i] = zr[:, i] + 1.0j * zr[:, i + 1]
                            z[:, i + 1] = z[:, i].conjugate()
                            d[i] = ((np.dot(zr[:, i],
                                            self.matvec(zr[:, i]))
                                     + np.dot(zr[:, i + 1],
                                              self.matvec(zr[:, i + 1])))
                                    + 1j * (np.dot(zr[:, i],
                                                   self.matvec(zr[:, i + 1]))
                                            - np.dot(zr[:, i + 1],
                                                     self.matvec(zr[:, i]))))
                            d[i + 1] = d[i].conj()
                            i += 1
                        else:
                            #last eigenvalue is complex: the imaginary part of
                            # the eigenvector has not been returned
                            #this can only happen if nreturned > k, so we'll
                            # throw out this case.
                            nreturned -= 1
                    i += 1

            # Now we have k+1 possible eigenvalues and eigenvectors
            # Return the ones specified by the keyword "which"

            if nreturned <= k:
                # we got less or equal as many eigenvalues we wanted
                d = d[:nreturned]
                z = z[:, :nreturned]
            else:
                # we got one extra eigenvalue (likely a cc pair, but which?)
                if self.mode in (1, 2):
                    rd = d
                elif self.mode in (3, 4):
                    rd = 1 / (d - self.sigma)

                if self.which in ['LR', 'SR']:
                    ind = np.argsort(rd.real)
                elif self.which in ['LI', 'SI']:
                    # for LI,SI ARPACK returns largest,smallest
                    # abs(imaginary) (complex pairs come together)
                    ind = np.argsort(abs(rd.imag))
                else:
                    ind = np.argsort(abs(rd))

                if self.which in ['LR', 'LM', 'LI']:
                    ind = ind[-k:][::-1]
                elif self.which in ['SR', 'SM', 'SI']:
                    ind = ind[:k]

                d = d[ind]
                z = z[:, ind]
        else:
            # complex is so much simpler...
            d, z, ierr =\
                    self._arpack_extract(return_eigenvectors,
                           howmny, sselect, self.sigma, workev,
                           self.bmat, self.which, k, self.tol, self.resid,
                           self.v, self.iparam, self.ipntr,
                           self.workd, self.workl, self.rwork, ierr)

            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)

            k_ok = self.iparam[4]
            d = d[:k_ok]
            z = z[:, :k_ok]

        if return_eigenvectors:
            return d, z
        else:
            return d


def _aslinearoperator_with_dtype(m):
    m = aslinearoperator(m)
    if not hasattr(m, 'dtype'):
        x = np.zeros(m.shape[1])
        m.dtype = (m * x).dtype
    return m


class SpLuInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x=b
       using a sparse LU-decomposition of M
    """
    def __init__(self, M):
        self.M_lu = splu(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        # careful here: splu.solve will throw away imaginary
        # part of x if M is real
        x = np.asarray(x)
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return (self.M_lu.solve(np.real(x).astype(self.dtype))
                    + 1j * self.M_lu.solve(np.imag(x).astype(self.dtype)))
        else:
            return self.M_lu.solve(x.astype(self.dtype))


class LuInv(LinearOperator):
    """
    LuInv:
       helper class to repeatedly solve M*x=b
       using an LU-decomposition of M
    """
    def __init__(self, M):
        self.M_lu = lu_factor(M)
        self.shape = M.shape
        self.dtype = M.dtype

    def _matvec(self, x):
        return lu_solve(self.M_lu, x)


def gmres_loose(A, b, tol):
    """
    gmres with looser termination condition.
    """
    b = np.asarray(b)
    min_tol = 1000 * np.sqrt(b.size) * np.finfo(b.dtype).eps
    return gmres(A, b, tol=max(tol, min_tol), atol=0)


class IterInv(LinearOperator):
    """
    IterInv:
       helper class to repeatedly solve M*x=b
       using an iterative method.
    """
    def __init__(self, M, ifunc=gmres_loose, tol=0):
        self.M = M
        if hasattr(M, 'dtype'):
            self.dtype = M.dtype
        else:
            x = np.zeros(M.shape[1])
            self.dtype = (M * x).dtype
        self.shape = M.shape

        if tol <= 0:
            # when tol=0, ARPACK uses machine tolerance as calculated
            # by LAPACK's _LAMCH function.  We should match this
            tol = 2 * np.finfo(self.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        b, info = self.ifunc(self.M, x, tol=self.tol)
        if info != 0:
            raise ValueError("Error in inverting M: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        return b


class IterOpInv(LinearOperator):
    """
    IterOpInv:
       helper class to repeatedly solve [A-sigma*M]*x = b
       using an iterative method
    """
    def __init__(self, A, M, sigma, ifunc=gmres_loose, tol=0):
        self.A = A
        self.M = M
        self.sigma = sigma

        def mult_func(x):
            return A.matvec(x) - sigma * M.matvec(x)

        def mult_func_M_None(x):
            return A.matvec(x) - sigma * x

        x = np.zeros(A.shape[1])
        if M is None:
            dtype = mult_func_M_None(x).dtype
            self.OP = LinearOperator(self.A.shape,
                                     mult_func_M_None,
                                     dtype=dtype)
        else:
            dtype = mult_func(x).dtype
            self.OP = LinearOperator(self.A.shape,
                                     mult_func,
                                     dtype=dtype)
        self.shape = A.shape

        if tol <= 0:
            # when tol=0, ARPACK uses machine tolerance as calculated
            # by LAPACK's _LAMCH function.  We should match this
            tol = 2 * np.finfo(self.OP.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        b, info = self.ifunc(self.OP, x, tol=self.tol)
        if info != 0:
            raise ValueError("Error in inverting [A-sigma*M]: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        return b

    @property
    def dtype(self):
        return self.OP.dtype


def _fast_spmatrix_to_csc(A, hermitian=False):
    """Convert sparse matrix to CSC (by transposing, if possible)"""
    if (isspmatrix_csr(A) and hermitian
            and not np.issubdtype(A.dtype, np.complexfloating)):
        return A.T
    elif is_pydata_spmatrix(A):
        # No need to convert
        return A
    else:
        return A.tocsc()


def get_inv_matvec(M, hermitian=False, tol=0):
    if isdense(M):
        return LuInv(M).matvec
    elif isspmatrix(M) or is_pydata_spmatrix(M):
        M = _fast_spmatrix_to_csc(M, hermitian=hermitian)
        return SpLuInv(M).matvec
    else:
        return IterInv(M, tol=tol).matvec


def get_OPinv_matvec(A, M, sigma, hermitian=False, tol=0):
    if sigma == 0:
        return get_inv_matvec(A, hermitian=hermitian, tol=tol)

    if M is None:
        #M is the identity matrix
        if isdense(A):
            if (np.issubdtype(A.dtype, np.complexfloating)
                    or np.imag(sigma) == 0):
                A = np.copy(A)
            else:
                A = A + 0j
            A.flat[::A.shape[1] + 1] -= sigma
            return LuInv(A).matvec
        elif isspmatrix(A) or is_pydata_spmatrix(A):
            A = A - sigma * eye(A.shape[0])
            A = _fast_spmatrix_to_csc(A, hermitian=hermitian)
            return SpLuInv(A).matvec
        else:
            return IterOpInv(_aslinearoperator_with_dtype(A),
                             M, sigma, tol=tol).matvec
    else:
        if ((not isdense(A) and not isspmatrix(A) and not is_pydata_spmatrix(A)) or
                (not isdense(M) and not isspmatrix(M) and not is_pydata_spmatrix(A))):
            return IterOpInv(_aslinearoperator_with_dtype(A),
                             _aslinearoperator_with_dtype(M),
                             sigma, tol=tol).matvec
        elif isdense(A) or isdense(M):
            return LuInv(A - sigma * M).matvec
        else:
            OP = A - sigma * M
            OP = _fast_spmatrix_to_csc(OP, hermitian=hermitian)
            return SpLuInv(OP).matvec


# ARPACK is not threadsafe or reentrant (SAVE variables), so we need a
# lock and a re-entering check.
_ARPACK_LOCK = ReentrancyLock("Nested calls to eigs/eighs not allowed: "
                              "ARPACK is not re-entrant")


def eigs(A, k=6, M=None, sigma=None, which='LM', v0=None,
         ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
         Minv=None, OPinv=None, OPpart=None):
    """
    Find k eigenvalues and eigenvectors of the square matrix A.

    Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem
    for w[i] eigenvalues with corresponding eigenvectors x[i].

    If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
    generalized eigenvalue problem for w[i] eigenvalues
    with corresponding eigenvectors x[i]

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        An array, sparse matrix, or LinearOperator representing
        the operation ``A * x``, where A is a real or complex square matrix.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N-1. It is not possible to compute all
        eigenvectors of a matrix.
    M : ndarray, sparse matrix or LinearOperator, optional
        An array, sparse matrix, or LinearOperator representing
        the operation M*x for the generalized eigenvalue problem

            A * x = w * M * x.

        M must represent a real symmetric matrix if A is real, and must
        represent a complex Hermitian matrix if A is complex. For best
        results, the data type of M should be the same as that of A.
        Additionally:

            If `sigma` is None, M is positive definite

            If sigma is specified, M is positive semi-definite

        If sigma is None, eigs requires an operator to compute the solution
        of the linear equation ``M * x = b``.  This is done internally via a
        (sparse) LU decomposition for an explicit matrix M, or via an
        iterative solver for a general linear operator.  Alternatively,
        the user can supply the matrix or operator Minv, which gives
        ``x = Minv * b = M^-1 * b``.
    sigma : real or complex, optional
        Find eigenvalues near sigma using shift-invert mode.  This requires
        an operator to compute the solution of the linear system
        ``[A - sigma * M] * x = b``, where M is the identity matrix if
        unspecified. This is computed internally via a (sparse) LU
        decomposition for explicit matrices A & M, or via an iterative
        solver if either A or M is a general linear operator.
        Alternatively, the user can supply the matrix or operator OPinv,
        which gives ``x = OPinv * b = [A - sigma * M]^-1 * b``.
        For a real matrix A, shift-invert can either be done in imaginary
        mode or real mode, specified by the parameter OPpart ('r' or 'i').
        Note that when sigma is specified, the keyword 'which' (below)
        refers to the shifted eigenvalues ``w'[i]`` where:

            If A is real and OPpart == 'r' (default),
              ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.

            If A is real and OPpart == 'i',
              ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.

            If A is complex, ``w'[i] = 1/(w[i]-sigma)``.

    v0 : ndarray, optional
        Starting vector for iteration.
        Default: random
    ncv : int, optional
        The number of Lanczos vectors generated
        `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.
        Default: ``min(n, max(2*k + 1, 20))``
    which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : largest magnitude

            'SM' : smallest magnitude

            'LR' : largest real part

            'SR' : smallest real part

            'LI' : largest imaginary part

            'SI' : smallest imaginary part

        When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
        (see discussion in 'sigma', above).  ARPACK is generally better
        at finding large values than small values.  If small eigenvalues are
        desired, consider using shift-invert mode for better performance.
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed
        Default: ``n*10``
    tol : float, optional
        Relative accuracy for eigenvalues (stopping criterion)
        The default value of 0 implies machine precision.
    return_eigenvectors : bool, optional
        Return eigenvectors (True) in addition to eigenvalues
    Minv : ndarray, sparse matrix or LinearOperator, optional
        See notes in M, above.
    OPinv : ndarray, sparse matrix or LinearOperator, optional
        See notes in sigma, above.
    OPpart : {'r' or 'i'}, optional
        See notes in sigma, above

    Returns
    -------
    w : ndarray
        Array of k eigenvalues.
    v : ndarray
        An array of `k` eigenvectors.
        ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.
        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.

    See Also
    --------
    eigsh : eigenvalues and eigenvectors for symmetric matrix A
    svds : singular value decomposition for a matrix A

    Notes
    -----
    This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,
    ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to
    find the eigenvalues and eigenvectors [2]_.

    References
    ----------
    .. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    Find 6 eigenvectors of the identity matrix:

    >>> from scipy.sparse.linalg import eigs
    >>> id = np.eye(13)
    >>> vals, vecs = eigs(id, k=6)
    >>> vals
    array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
    >>> vecs.shape
    (13, 6)

    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix (shape=%s)' % (A.shape,))
    if M is not None:
        if M.shape != A.shape:
            raise ValueError('wrong M dimensions %s, should be %s'
                             % (M.shape, A.shape))
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. '
                          'This may adversely affect ARPACK convergence')

    n = A.shape[0]

    if k <= 0:
        raise ValueError("k=%d must be greater than 0." % k)

    if k >= n - 1:
        warnings.warn("k >= N - 1 for N * N square matrix. "
                      "Attempting to use scipy.linalg.eig instead.",
                      RuntimeWarning)

        if issparse(A):
            raise TypeError("Cannot use scipy.linalg.eig for sparse A with "
                            "k >= N - 1. Use scipy.linalg.eig(A.toarray()) or"
                            " reduce k.")
        if isinstance(A, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eig for LinearOperator "
                            "A with k >= N - 1.")
        if isinstance(M, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eig for LinearOperator "
                            "M with k >= N - 1.")

        return eig(A, b=M, right=return_eigenvectors)

    if sigma is None:
        matvec = _aslinearoperator_with_dtype(A).matvec

        if OPinv is not None:
            raise ValueError("OPinv should not be specified "
                             "with sigma = None.")
        if OPpart is not None:
            raise ValueError("OPpart should not be specified with "
                             "sigma = None or complex A")

        if M is None:
            #standard eigenvalue problem
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError("Minv should not be "
                                 "specified with M = None.")
        else:
            #general eigenvalue problem
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        #sigma is not None: shift-invert mode
        if np.issubdtype(A.dtype, np.complexfloating):
            if OPpart is not None:
                raise ValueError("OPpart should not be specified "
                                 "with sigma=None or complex A")
            mode = 3
        elif OPpart is None or OPpart.lower() == 'r':
            mode = 3
        elif OPpart.lower() == 'i':
            if np.imag(sigma) == 0:
                raise ValueError("OPpart cannot be 'i' if sigma is real")
            mode = 4
        else:
            raise ValueError("OPpart must be one of ('r','i')")

        matvec = _aslinearoperator_with_dtype(A).matvec
        if Minv is not None:
            raise ValueError("Minv should not be specified when sigma is")
        if OPinv is None:
            Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                           hermitian=False, tol=tol)
        else:
            OPinv = _aslinearoperator_with_dtype(OPinv)
            Minv_matvec = OPinv.matvec
        if M is None:
            M_matvec = None
        else:
            M_matvec = _aslinearoperator_with_dtype(M).matvec

    params = _UnsymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
                                      M_matvec, Minv_matvec, sigma,
                                      ncv, v0, maxiter, which, tol)

    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)


def eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None,
          ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
          Minv=None, OPinv=None, mode='normal'):
    """
    Find k eigenvalues and eigenvectors of the real symmetric square matrix
    or complex Hermitian matrix A.

    Solves ``A * x[i] = w[i] * x[i]``, the standard eigenvalue problem for
    w[i] eigenvalues with corresponding eigenvectors x[i].

    If M is specified, solves ``A * x[i] = w[i] * M * x[i]``, the
    generalized eigenvalue problem for w[i] eigenvalues
    with corresponding eigenvectors x[i].

    Note that there is no specialized routine for the case when A is a complex
    Hermitian matrix. In this case, ``eigsh()`` will call ``eigs()`` and return the
    real parts of the eigenvalues thus obtained.

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        A square operator representing the operation ``A * x``, where ``A`` is
        real symmetric or complex Hermitian. For buckling mode (see below)
        ``A`` must additionally be positive-definite.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N. It is not possible to compute all
        eigenvectors of a matrix.

    Returns
    -------
    w : array
        Array of k eigenvalues.
    v : array
        An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
        the eigenvector corresponding to the eigenvalue ``w[i]``.

    Other Parameters
    ----------------
    M : An N x N matrix, array, sparse matrix, or linear operator representing
        the operation ``M @ x`` for the generalized eigenvalue problem

            A @ x = w * M @ x.

        M must represent a real symmetric matrix if A is real, and must
        represent a complex Hermitian matrix if A is complex. For best
        results, the data type of M should be the same as that of A.
        Additionally:

            If sigma is None, M is symmetric positive definite.

            If sigma is specified, M is symmetric positive semi-definite.

            In buckling mode, M is symmetric indefinite.

        If sigma is None, eigsh requires an operator to compute the solution
        of the linear equation ``M @ x = b``. This is done internally via a
        (sparse) LU decomposition for an explicit matrix M, or via an
        iterative solver for a general linear operator.  Alternatively,
        the user can supply the matrix or operator Minv, which gives
        ``x = Minv @ b = M^-1 @ b``.
    sigma : real
        Find eigenvalues near sigma using shift-invert mode.  This requires
        an operator to compute the solution of the linear system
        ``[A - sigma * M] x = b``, where M is the identity matrix if
        unspecified.  This is computed internally via a (sparse) LU
        decomposition for explicit matrices A & M, or via an iterative
        solver if either A or M is a general linear operator.
        Alternatively, the user can supply the matrix or operator OPinv,
        which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
        Note that when sigma is specified, the keyword 'which' refers to
        the shifted eigenvalues ``w'[i]`` where:

            if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

            if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

            if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

        (see further discussion in 'mode' below)
    v0 : ndarray, optional
        Starting vector for iteration.
        Default: random
    ncv : int, optional
        The number of Lanczos vectors generated ncv must be greater than k and
        smaller than n; it is recommended that ``ncv > 2*k``.
        Default: ``min(n, max(2*k + 1, 20))``
    which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
        If A is a complex Hermitian matrix, 'BE' is invalid.
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : Largest (in magnitude) eigenvalues.

            'SM' : Smallest (in magnitude) eigenvalues.

            'LA' : Largest (algebraic) eigenvalues.

            'SA' : Smallest (algebraic) eigenvalues.

            'BE' : Half (k/2) from each end of the spectrum.

        When k is odd, return one more (k/2+1) from the high end.
        When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
        (see discussion in 'sigma', above).  ARPACK is generally better
        at finding large values than small values.  If small eigenvalues are
        desired, consider using shift-invert mode for better performance.
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        Default: ``n*10``
    tol : float
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.
    Minv : N x N matrix, array, sparse matrix, or LinearOperator
        See notes in M, above.
    OPinv : N x N matrix, array, sparse matrix, or LinearOperator
        See notes in sigma, above.
    return_eigenvectors : bool
        Return eigenvectors (True) in addition to eigenvalues.
        This value determines the order in which eigenvalues are sorted.
        The sort order is also dependent on the `which` variable.

            For which = 'LM' or 'SA':
                If `return_eigenvectors` is True, eigenvalues are sorted by
                algebraic value.

                If `return_eigenvectors` is False, eigenvalues are sorted by
                absolute value.

            For which = 'BE' or 'LA':
                eigenvalues are always sorted by algebraic value.

            For which = 'SM':
                If `return_eigenvectors` is True, eigenvalues are sorted by
                algebraic value.

                If `return_eigenvectors` is False, eigenvalues are sorted by
                decreasing absolute value.

    mode : string ['normal' | 'buckling' | 'cayley']
        Specify strategy to use for shift-invert mode.  This argument applies
        only for real-valued A and sigma != None.  For shift-invert mode,
        ARPACK internally solves the eigenvalue problem
        ``OP * x'[i] = w'[i] * B * x'[i]``
        and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
        into the desired eigenvectors and eigenvalues of the problem
        ``A * x[i] = w[i] * M * x[i]``.
        The modes are as follows:

            'normal' :
                OP = [A - sigma * M]^-1 @ M,
                B = M,
                w'[i] = 1 / (w[i] - sigma)

            'buckling' :
                OP = [A - sigma * M]^-1 @ A,
                B = A,
                w'[i] = w[i] / (w[i] - sigma)

            'cayley' :
                OP = [A - sigma * M]^-1 @ [A + sigma * M],
                B = M,
                w'[i] = (w[i] + sigma) / (w[i] - sigma)

        The choice of mode will affect which eigenvalues are selected by
        the keyword 'which', and can also impact the stability of
        convergence (see [2] for a discussion).

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.

        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.

    See Also
    --------
    eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
    svds : singular value decomposition for a matrix A

    Notes
    -----
    This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
    functions which use the Implicitly Restarted Lanczos Method to
    find the eigenvalues and eigenvectors [2]_.

    References
    ----------
    .. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    >>> from scipy.sparse.linalg import eigsh
    >>> identity = np.eye(13)
    >>> eigenvalues, eigenvectors = eigsh(identity, k=6)
    >>> eigenvalues
    array([1., 1., 1., 1., 1., 1.])
    >>> eigenvectors.shape
    (13, 6)

    """
    # complex Hermitian matrices should be solved with eigs
    if np.issubdtype(A.dtype, np.complexfloating):
        if mode != 'normal':
            raise ValueError("mode=%s cannot be used with "
                             "complex matrix A" % mode)
        if which == 'BE':
            raise ValueError("which='BE' cannot be used with complex matrix A")
        elif which == 'LA':
            which = 'LR'
        elif which == 'SA':
            which = 'SR'
        ret = eigs(A, k, M=M, sigma=sigma, which=which, v0=v0,
                   ncv=ncv, maxiter=maxiter, tol=tol,
                   return_eigenvectors=return_eigenvectors, Minv=Minv,
                   OPinv=OPinv)

        if return_eigenvectors:
            return ret[0].real, ret[1]
        else:
            return ret.real

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix (shape=%s)' % (A.shape,))
    if M is not None:
        if M.shape != A.shape:
            raise ValueError('wrong M dimensions %s, should be %s'
                             % (M.shape, A.shape))
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. '
                          'This may adversely affect ARPACK convergence')

    n = A.shape[0]

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k >= n:
        warnings.warn("k >= N for N * N square matrix. "
                      "Attempting to use scipy.linalg.eigh instead.",
                      RuntimeWarning)

        if issparse(A):
            raise TypeError("Cannot use scipy.linalg.eigh for sparse A with "
                            "k >= N. Use scipy.linalg.eigh(A.toarray()) or"
                            " reduce k.")
        if isinstance(A, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eigh for LinearOperator "
                            "A with k >= N.")
        if isinstance(M, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eigh for LinearOperator "
                            "M with k >= N.")

        return eigh(A, b=M, eigvals_only=not return_eigenvectors)

    if sigma is None:
        A = _aslinearoperator_with_dtype(A)
        matvec = A.matvec

        if OPinv is not None:
            raise ValueError("OPinv should not be specified "
                             "with sigma = None.")
        if M is None:
            #standard eigenvalue problem
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError("Minv should not be "
                                 "specified with M = None.")
        else:
            #general eigenvalue problem
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        # sigma is not None: shift-invert mode
        if Minv is not None:
            raise ValueError("Minv should not be specified when sigma is")

        # normal mode
        if mode == 'normal':
            mode = 3
            matvec = None
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                OPinv = _aslinearoperator_with_dtype(OPinv)
                Minv_matvec = OPinv.matvec
            if M is None:
                M_matvec = None
            else:
                M = _aslinearoperator_with_dtype(M)
                M_matvec = M.matvec

        # buckling mode
        elif mode == 'buckling':
            mode = 4
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            matvec = _aslinearoperator_with_dtype(A).matvec
            M_matvec = None

        # cayley-transform mode
        elif mode == 'cayley':
            mode = 5
            matvec = _aslinearoperator_with_dtype(A).matvec
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            if M is None:
                M_matvec = None
            else:
                M_matvec = _aslinearoperator_with_dtype(M).matvec

        # unrecognized mode
        else:
            raise ValueError("unrecognized mode '%s'" % mode)

    params = _SymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
                                    M_matvec, Minv_matvec, sigma,
                                    ncv, v0, maxiter, which, tol)

    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)


def _augmented_orthonormal_cols(x, k):
    # extract the shape of the x array
    n, m = x.shape
    # create the expanded array and copy x into it
    y = np.empty((n, m+k), dtype=x.dtype)
    y[:, :m] = x
    # do some modified gram schmidt to add k random orthonormal vectors
    for i in range(k):
        # sample a random initial vector
        v = np.random.randn(n)
        if np.iscomplexobj(x):
            v = v + 1j*np.random.randn(n)
        # subtract projections onto the existing unit length vectors
        for j in range(m+i):
            u = y[:, j]
            v -= (np.dot(v, u.conj()) / np.dot(u, u.conj())) * u
        # normalize v
        v /= np.sqrt(np.dot(v, v.conj()))
        # add v into the output array
        y[:, m+i] = v
    # return the expanded array
    return y


def _augmented_orthonormal_rows(x, k):
    return _augmented_orthonormal_cols(x.T, k).T


def _herm(x):
    return x.T.conj()


def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True,
         solver='arpack'):
    """Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.

    Parameters
    ----------
    A : {sparse matrix, LinearOperator}
        Array to compute the SVD on, of shape (M, N)
    k : int, optional
        Number of singular values and vectors to compute.
        Must be 1 <= k < min(A.shape).
    ncv : int, optional
        The number of Lanczos vectors generated
        ncv must be greater than k+1 and smaller than n;
        it is recommended that ncv > 2*k
        Default: ``min(n, max(2*k + 1, 20))``
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : str, ['LM' | 'SM'], optional
        Which `k` singular values to find:

            - 'LM' : largest singular values
            - 'SM' : smallest singular values

        .. versionadded:: 0.12.0
    v0 : ndarray, optional
        Starting vector for iteration, of length min(A.shape). Should be an
        (approximate) left singular vector if N > M and a right singular
        vector otherwise.
        Default: random

        .. versionadded:: 0.12.0
    maxiter : int, optional
        Maximum number of iterations.

        .. versionadded:: 0.12.0
    return_singular_vectors : bool or str, optional
        - True: return singular vectors (True) in addition to singular values.

        .. versionadded:: 0.12.0

        - "u": only return the u matrix, without computing vh (if N > M).
        - "vh": only return the vh matrix, without computing u (if N <= M).

        .. versionadded:: 0.16.0
    solver : str, optional
            Eigenvalue solver to use. Should be 'arpack' or 'lobpcg'.
            Default: 'arpack'

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
        If `return_singular_vectors` is "vh", this variable is not computed,
        and None is returned instead.
    s : ndarray, shape=(k,)
        The singular values.
    vt : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.
        If `return_singular_vectors` is "u", this variable is not computed,
        and None is returned instead.


    Notes
    -----
    This is a naive implementation using ARPACK or LOBPCG as an eigensolver
    on A.H * A or A * A.H, depending on which one is more efficient.

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import svds, eigs
    >>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
    >>> u, s, vt = svds(A, k=2)
    >>> s
    array([ 2.75193379,  5.6059665 ])
    >>> np.sqrt(eigs(A.dot(A.T), k=2)[0]).real
    array([ 5.6059665 ,  2.75193379])
    """
    if which == 'LM':
        largest = True
    elif which == 'SM':
        largest = False
    else:
        raise ValueError("which must be either 'LM' or 'SM'.")

    if not (isinstance(A, LinearOperator) or isspmatrix(A) or is_pydata_spmatrix(A)):
        A = np.asarray(A)

    n, m = A.shape

    if k <= 0 or k >= min(n, m):
        raise ValueError("k must be between 1 and min(A.shape), k=%d" % k)

    if isinstance(A, LinearOperator):
        if n > m:
            X_dot = A.matvec
            X_matmat = A.matmat
            XH_dot = A.rmatvec
            XH_mat = A.rmatmat
        else:
            X_dot = A.rmatvec
            X_matmat = A.rmatmat
            XH_dot = A.matvec
            XH_mat = A.matmat

            dtype = getattr(A, 'dtype', None)
            if dtype is None:
                dtype = A.dot(np.zeros([m, 1])).dtype

    else:
        if n > m:
            X_dot = X_matmat = A.dot
            XH_dot = XH_mat = _herm(A).dot
        else:
            XH_dot = XH_mat = A.dot
            X_dot = X_matmat = _herm(A).dot

    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        return XH_mat(X_matmat(x))

    XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                          matmat=matmat_XH_X,
                          shape=(min(A.shape), min(A.shape)))

    # Get a low rank approximation of the implicitly defined gramian matrix.
    # This is not a stable way to approach the problem.
    if solver == 'lobpcg':

        if k == 1 and v0 is not None:
            X = np.reshape(v0, (-1, 1))
        else:
            X = np.random.RandomState(52).randn(min(A.shape), k)

        eigvals, eigvec = lobpcg(XH_X, X, tol=tol ** 2, maxiter=maxiter,
                                 largest=largest)

    elif solver == 'arpack' or solver is None:
        eigvals, eigvec = eigsh(XH_X, k=k, tol=tol ** 2, maxiter=maxiter,
                                ncv=ncv, which=which, v0=v0)

    else:
        raise ValueError("solver must be either 'arpack', or 'lobpcg'.")

    # Gramian matrices have real non-negative eigenvalues.
    eigvals = np.maximum(eigvals.real, 0)

    # Use the sophisticated detection of small eigenvalues from pinvh.
    t = eigvec.dtype.char.lower()
    factor = {'f': 1E3, 'd': 1E6}
    cond = factor[t] * np.finfo(t).eps
    cutoff = cond * np.max(eigvals)

    # Get a mask indicating which eigenpairs are not degenerately tiny,
    # and create the re-ordered array of thresholded singular values.
    above_cutoff = (eigvals > cutoff)
    nlarge = above_cutoff.sum()
    nsmall = k - nlarge
    slarge = np.sqrt(eigvals[above_cutoff])
    s = np.zeros_like(eigvals)
    s[:nlarge] = slarge
    if not return_singular_vectors:
        return np.sort(s)

    if n > m:
        vlarge = eigvec[:, above_cutoff]
        ularge = X_matmat(vlarge) / slarge if return_singular_vectors != 'vh' else None
        vhlarge = _herm(vlarge)
    else:
        ularge = eigvec[:, above_cutoff]
        vhlarge = _herm(X_matmat(ularge) / slarge) if return_singular_vectors != 'u' else None

    u = _augmented_orthonormal_cols(ularge, nsmall) if ularge is not None else None
    vh = _augmented_orthonormal_rows(vhlarge, nsmall) if vhlarge is not None else None

    indexes_sorted = np.argsort(s)
    s = s[indexes_sorted]
    if u is not None:
        u = u[:, indexes_sorted]
    if vh is not None:
        vh = vh[indexes_sorted]

    return u, s, vh
